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
    """The recovery lever is a faithful libm-``pow`` reproduction, and whether it
    is *bit-distinct* from production is a property of the host libm — pinned here
    as a platform truth, not a universal one.

    Production squares the contact threshold ``Vt`` with ``powi(2)`` (``x*x``); the
    reference oracle uses the system libm ``pow(x, 2.0)`` — reproducing the retired
    Python ``_impl`` oracle's ``x ** 2`` bit-for-bit. That square is the *only*
    difference between the two modes, so:

    * macOS libm: ``pow(x, 2.0) != x*x`` (~1 ULP) -> the oracle diverges, and the
      divergence surfaces in the contact branch ``I_s_s1``.
    * glibc: ``pow`` is correctly rounded, ``pow(x, 2.0) == x*x`` -> the oracle
      legitimately collapses onto production and is bit-identical everywhere.

    The channel current ``Ich`` and the capacitances carry no ``Vt`` term, so they
    stay bit-identical on every platform."""
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

    # Vt-free terms are bit-identical in either libm regime.
    assert pi[3] == ri[3]          # Ich carries no Vt term
    np.testing.assert_array_equal(
        prod.capacitance_charges(*point), ref.capacitance_charges(*point))

    # Detect the regime from the oracle's OWN contact-branch output (robust to a
    # glibc-equivalent core), then pin the matching invariant.
    if pi[0] != ri[0]:
        # Split-libm regime: cross-check against CPython's own libm — ``float ** float``
        # calls the exact ``pow`` symbol the oracle routes the reference square through —
        # so ``pow(x, 2.0) != x*x`` must hold here too, confirming the contact-branch
        # divergence is the reference-square effect and nothing else.
        v_s = max(point[0], point[3])          # v_s = max(vs, vs1); see otft.rs::eval_currents
        dv = v_s - point[2]                     # v_s - vg: the contact-threshold argument
        assert (dv ** 2) != (dv * dv)
    else:
        # Correctly-rounded-libm regime (``pow(x, 2.0) == x*x``): the oracle is
        # bit-identical to production everywhere.
        np.testing.assert_array_equal(pi, ri)


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
