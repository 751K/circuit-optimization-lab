"""PDK / polarity registry: distinguishability + back-compat guarantees.

The transistor registry is PDK- and polarity-aware so multiple processes can
each provide PMOS/NMOS without colliding, while the calibrated AT4000TG PMOS
path stays byte-identical.  Generic sources (resistors, capacitors, ideal V/I
sources, controlled sources) are process-independent topology primitives and
are intentionally absent from this registry.
"""
import subprocess
import sys
import warnings

import pytest

import circuitopt.device_model as dm
from circuitopt.device_model import (
    TransistorModel,
    create_device,
    create_transistor,
    get_default_model_type,
    get_default_pdk,
    get_pdk,
    list_pdks,
    register_model,
    register_pdk,
    transistor_type,
)
from circuitopt.pmos_tft_model import PMOS_TFT


def test_at4000tg_is_default_pdk_with_pmos():
    assert get_default_pdk() == "at4000tg"
    assert "at4000tg" in list_pdks()
    pdk = get_pdk()  # default
    assert pdk.name == "at4000tg"
    assert "pmos" in pdk.devices
    assert pdk.model_type("pmos") == "at4000tg.pmos"


def test_alias_structured_key_and_default_resolve_to_same_class():
    """Legacy literal, structured key, registry default, and the convenience
    creator all map to the one PMOS_TFT class — back-compat is preserved."""
    assert get_default_model_type() == "at4000tg.pmos"
    a = create_device("pmos_tft", W=1000, L=20)        # legacy alias
    b = create_device("at4000tg.pmos", W=1000, L=20)   # structured key
    c = create_device(get_default_model_type(), W=1000, L=20)  # what solvers call
    d = create_transistor("pmos", W=1000, L=20)        # convenience creator
    assert type(a) is type(b) is type(c) is type(d) is PMOS_TFT
    # byte-identical DC at a representative bias — the calibrated default path
    assert a.get_Idc(40, 0, 20) == c.get_Idc(40, 0, 20)


def test_pmos_carries_polarity_and_pdk_identity():
    dev = create_device("pmos_tft", W=1000, L=20)
    assert dev.POLARITY == "pmos"
    assert dev.PDK == "at4000tg"


def test_second_pdk_and_nmos_polarity_stay_distinct():
    """A throwaway 2nd process (with both polarities) registers under distinct
    keys and does NOT disturb the default or the at4000tg.pmos calibrated path."""
    saved_models = dict(dm._model_registry)
    saved_pdks = dict(dm._pdk_registry)
    saved_default = dm._default_pdk
    try:
        class _StubFET(PMOS_TFT):
            """Placeholder reusing PMOS physics; identity differs by class."""
            POLARITY = "nmos"
            PDK = "stubproc"

        register_pdk("stubproc", {"pmos": PMOS_TFT, "nmos": _StubFET})

        # distinct structured keys for each polarity
        assert transistor_type("pmos", pdk="stubproc") == "stubproc.pmos"
        assert transistor_type("nmos", pdk="stubproc") == "stubproc.nmos"
        # registering without default= must NOT steal the default
        assert get_default_pdk() == "at4000tg"
        assert get_default_model_type() == "at4000tg.pmos"
        # the nmos key resolves to the stub, distinct from at4000tg.pmos
        assert type(create_device("stubproc.nmos", W=1000, L=20)) is _StubFET
        assert type(create_transistor("nmos", pdk="stubproc")) is _StubFET
        assert type(create_device("at4000tg.pmos", W=1000, L=20)) is PMOS_TFT
        assert {"at4000tg", "stubproc"} <= set(list_pdks())
    finally:
        dm._model_registry.clear()
        dm._model_registry.update(saved_models)
        dm._pdk_registry.clear()
        dm._pdk_registry.update(saved_pdks)
        dm._default_pdk = saved_default


def test_unknown_pdk_and_polarity_raise_cleanly():
    with pytest.raises(ValueError):
        get_pdk("nonexistent")
    with pytest.raises(ValueError):
        transistor_type("nmos")  # default PDK has no nmos device yet
    with pytest.raises(ValueError):
        create_device("at4000tg.nmos", W=1000, L=20)  # unregistered key


def test_generic_sources_are_not_in_the_transistor_registry():
    """R/C/ideal V/I/controlled sources are topology primitives, never models.
    The registry must contain only TransistorModel subclasses."""
    for key, cls in dm._model_registry.items():
        assert not any(tok in key for tok in
                       ("vsource", "resistor", "capacitor", "vccs",
                        "vcvs", "cccs", "ccvs", "isource"))
        assert isinstance(cls, type) and issubclass(cls, TransistorModel)


# ── Silent-override robustness (R5) ───────────────────────────────────────
# register_model must make an *accidental* name clash visible — two different
# classes claiming the same registry key/alias, last-import-wins — while
# leaving intentional substitution and same-class re-registration silent.

@pytest.fixture
def restore_registry():
    """Save/restore _model_registry so an override test can't leak state."""
    saved = dict(dm._model_registry)
    try:
        yield
    finally:
        dm._model_registry.clear()
        dm._model_registry.update(saved)


class _StubFET(PMOS_TFT):
    """Distinct-identity stub (differs from PMOS_TFT by __qualname__)."""
    POLARITY = "pmos"
    PDK = "stub"


def test_genuine_name_clash_warns_and_overrides(restore_registry):
    """A *different* class taking over an occupied key warns (RuntimeWarning)
    and still overrides — accidental collisions become visible, existing
    replace-semantics preserved."""
    with pytest.warns(RuntimeWarning, match="already registered"):
        register_model("pmos_tft", _StubFET)  # steals at4000tg's back-compat alias
    # override still executed — semantics unchanged
    assert dm.get_model_class("pmos_tft") is _StubFET


def test_same_class_reregistration_is_silent(restore_registry):
    """Re-registering the *identical* class (repeat import) must not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        register_model("pmos_tft", PMOS_TFT)  # same class already there
    assert dm.get_model_class("pmos_tft") is PMOS_TFT


def test_reloaded_class_same_qualname_is_silent(restore_registry):
    """importlib.reload rebinds the class to a *new* object under the same
    __module__.__qualname__.  The identity check would over-report, so a
    qualname match must suppress the warning."""
    # emulate a reload: a fresh class object carrying PMOS_TFT's identity
    reloaded = type("PMOS_TFT", (PMOS_TFT,), {})
    reloaded.__qualname__ = PMOS_TFT.__qualname__
    reloaded.__module__ = PMOS_TFT.__module__
    assert reloaded is not PMOS_TFT  # genuinely a new object
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        register_model("pmos_tft", reloaded)
    assert dm.get_model_class("pmos_tft") is reloaded


def test_import_circuitopt_emits_no_override_warnings():
    """Full PDK registration in normal import order must be silent — the
    anti-false-positive guard: `python -W error::RuntimeWarning -c 'import circuitopt'`
    exits clean."""
    proc = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", "import circuitopt"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"import circuitopt raised a RuntimeWarning:\n{proc.stderr}"
    )
