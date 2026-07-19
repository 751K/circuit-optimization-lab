"""R7: the OTFT root-selection recovery oracle lives in the compiled core.

`otft_reference_mode` switches the scalar model to ``OtftModel(reference=True)``
— the compiled reproduction of the retired Python ``_impl`` equations. These
tests pin (a) that the oracle is *not* a no-op (production and reference are
bit-distinct where the R6 root-selection incidents lived), (b) the thread-local
context semantics, and (c) that no Python reference-execution module remains.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from circuitopt.pmos_tft_model import PMOS_TFT, otft_reference_mode

try:
    import circuitopt_core
except ImportError:  # pragma: no cover - optional compiled wheel
    circuitopt_core = None

requires_rust_otft = pytest.mark.skipif(
    circuitopt_core is None or not hasattr(circuitopt_core, "OtftModel"),
    reason="circuitopt_core OtftModel is not installed",
)


@requires_rust_otft
def test_reference_oracle_is_bit_distinct_from_production():
    """The recovery lever is real: at a contact-dominated bias the reference
    oracle's `Vt` square (libm pow, ex-`_impl`) differs from production's
    ``powi`` by ~1 ULP in the contact current, while the channel current and
    the capacitances (which carry no `Vt` term) stay bit-identical."""
    t = PMOS_TFT(W=1000, L=20)
    prod = t._get_rust_model()
    with otft_reference_mode():
        ref = t._get_rust_model()
    assert prod is not ref

    # Contact-dominated stress point (from the R7 differential probe).
    point = (5.783524141765099, 4.348855453424909, 43.97557402023428,
             6.765993476290982, 4.117048005829847)
    pi = np.asarray(prod.eval_currents(*point))
    ri = np.asarray(ref.eval_currents(*point))
    assert pi[0] != ri[0]          # I_s_s1 carries the Vt-square divergence
    assert pi[3] == ri[3]          # Ich has no Vt term -> bit-identical
    np.testing.assert_array_equal(
        prod.capacitance_charges(*point), ref.capacitance_charges(*point))


@requires_rust_otft
def test_otft_reference_mode_context_is_scoped_and_reentrant():
    t = PMOS_TFT(W=1000, L=20)
    prod = t._get_rust_model()
    assert t._get_rust_model() is prod
    with otft_reference_mode():
        ref = t._get_rust_model()
        assert ref is not prod
        with otft_reference_mode():
            assert t._get_rust_model() is ref
        assert t._get_rust_model() is ref
    assert t._get_rust_model() is prod
    # Both instances stay cached on the device.
    assert t._rust_model is prod
    assert t._rust_model_ref is ref


def test_python_reference_execution_module_is_gone():
    """The permanent form of the R7 no-op probe: the package tree ships no
    `numba_kernels` module (the `_impl` oracle), and the device model exposes
    no Python `_impl` fallbacks, so no Python reference execution can be
    silently selected.

    (Checked on the package directory, not via import: in a multi-checkout
    editable install a deleted submodule can still resolve from another
    checkout through the editable finder.)"""
    import circuitopt
    import circuitopt.pmos_tft_model as model_module
    from pathlib import Path

    package_dir = Path(circuitopt.__path__[0])
    assert not (package_dir / "numba_kernels.py").exists()
    for retired in ("_eval_currents_impl", "_newton_internal_impl",
                    "_capacitances_impl", "_capacitance_charges_impl",
                    "terminal_derivatives_numba"):
        assert not hasattr(model_module, retired)
    # And the loaded package is the one whose tree was checked.
    assert Path(model_module.__file__).parent == package_dir
    assert importlib.util.find_spec("circuitopt") is not None
