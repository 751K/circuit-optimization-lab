"""PDK / polarity registry: distinguishability + back-compat guarantees.

The transistor registry is PDK- and polarity-aware so multiple processes can
each provide PMOS/NMOS without colliding, while the calibrated AT4000TG PMOS
path stays byte-identical.  Generic sources (resistors, capacitors, ideal V/I
sources, controlled sources) are process-independent topology primitives and
are intentionally absent from this registry.
"""
import pytest

import core.device_model as dm
from core.device_model import (
    TransistorModel,
    create_device,
    create_transistor,
    get_default_model_type,
    get_default_pdk,
    get_pdk,
    list_pdks,
    register_pdk,
    transistor_type,
)
from core.pmos_tft_model import PMOS_TFT


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
