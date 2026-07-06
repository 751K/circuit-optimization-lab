"""Regression: the factor-once Woodbury pnoise adjoint == the per-frequency direct splu.

``_time_domain_pnoise_adjoint`` factors the block-bidiagonal BE operator ``F(γ0)``
once and corrects each noise frequency with a rank-``ns`` Woodbury update instead
of refactoring the ``N·ns`` sparse ``F(γ)`` per frequency (measured 6.6× faster on
the chopper).  The chopper calibration test only guards the *end-to-end* IRN at a
~3 % tolerance — far too loose to catch a subtle bug in the rank-``ns`` update (a
wrong sign, a mis-indexed corner block, a dropped ``1/γ`` term).

This test pins the new numerical core directly: on a synthetic, well-conditioned
periodic linearization, the Woodbury path must reproduce the reference
per-frequency splu (``_force_direct=True`` — the same ``_zeta_direct`` path the
runtime fallback uses) to machine precision.  It also asserts the fast path
actually *engages* (zero per-frequency fallbacks) so the parity check is not a
vacuous direct-vs-direct comparison.  The shared block-bidiagonal assembly / FFT
store is exercised by both paths and separately validated end-to-end against
Cadence; here we isolate the factor-once shortcut.
"""
import numpy as np
import pytest

from circuitopt import diagnostics
from circuitopt.pnoise_solver import _time_domain_pnoise_adjoint


def _synthetic_linearization(ns=3, N=64, fundamental=225.0, seed=0):
    """Real, periodic, diagonally-dominant G(t)/C(t) → well-conditioned F(γ).

    Returns ``(Gf, Cf)`` as the Fourier coefficients the adjoint re-expands via
    ``ifft(·*N)``.  ``C(t)/h`` is scaled comparable to ``G(t)`` so the periodic
    corner block ``-BT[0]/γ`` — the *only* term the Woodbury update corrects — is
    non-negligible, i.e. the rank-``ns`` correction does real work every frequency.
    """
    rng = np.random.default_rng(seed)
    h = (1.0 / fundamental) / N
    t = np.arange(N) / N
    base_g = np.diag(np.arange(1, ns + 1) * 1e-3)          # strong diagonal conductance
    base_c = np.diag(np.full(ns, 1e-3 * h))                # C/h ≈ 1e-3, comparable to G
    offg = 1e-4 * rng.standard_normal((ns, ns))
    offc = 2e-4 * h * rng.standard_normal((ns, ns))
    Gt = np.empty((N, ns, ns))
    Ct = np.empty((N, ns, ns))
    for m in range(N):
        s, c = np.sin(2 * np.pi * t[m]), np.cos(2 * np.pi * t[m])
        Gt[m] = base_g * (1.0 + 0.3 * s) + offg * s        # periodic time variation
        Ct[m] = base_c * (1.0 + 0.2 * c) + offc * c
    return np.fft.fft(Gt, axis=0) / N, np.fft.fft(Ct, axis=0) / N


def test_woodbury_pnoise_adjoint_matches_direct():
    pytest.importorskip("scipy")
    ns, N, K, fundamental = 3, 64, 4, 225.0
    Gf, Cf = _synthetic_linearization(ns=ns, N=N, fundamental=fundamental, seed=1)
    nb = 2 * K + 1
    rng = np.random.default_rng(2)
    e = rng.standard_normal(nb * ns) + 1j * rng.standard_normal(nb * ns)
    freqs = np.array([1.0, 5.0, 20.0, 50.0, 100.0])

    diagnostics.reset()
    adjs = _time_domain_pnoise_adjoint(Gf, Cf, e, freqs, K, ns, fundamental)
    snap = diagnostics.snapshot()
    ref = _time_domain_pnoise_adjoint(Gf, Cf, e, freqs, K, ns, fundamental,
                                      _force_direct=True)

    assert adjs is not None and ref is not None            # scipy present ⇒ not None
    assert adjs.shape == (len(freqs), nb * ns)
    assert np.isfinite(adjs).all()

    # The factor-once path must actually engage on every frequency; otherwise the
    # parity assertion below degenerates into direct-vs-direct and proves nothing.
    assert snap.get("pnoise.td_woodbury_setup_fail", 0) == 0
    assert snap.get("pnoise.td_woodbury_freq_fallback", 0) == 0

    # ... and reproduce the reference per-frequency splu to machine precision.
    scale = np.abs(ref).max()
    assert scale > 0.0
    assert np.abs(adjs - ref).max() <= 1e-9 * scale


def test_woodbury_force_direct_leaves_no_fallback_note():
    """``_force_direct`` is a clean reference path — it must not log a fallback."""
    pytest.importorskip("scipy")
    ns, N, K, fundamental = 2, 48, 3, 225.0
    Gf, Cf = _synthetic_linearization(ns=ns, N=N, fundamental=fundamental, seed=3)
    e = np.ones((2 * K + 1) * ns, dtype=complex)
    diagnostics.reset()
    out = _time_domain_pnoise_adjoint(Gf, Cf, e, np.array([2.0, 40.0]), K, ns,
                                      fundamental, _force_direct=True)
    assert out is not None and np.isfinite(out).all()
    snap = diagnostics.snapshot()
    assert snap.get("pnoise.td_woodbury_freq_fallback", 0) == 0
    assert snap.get("pnoise.td_woodbury_setup_fail", 0) == 0
