"""OTFT scalar-model consistency tests.

History: these validated the Python/numba ``_impl`` kernels against the OO model
formulas. The ``_impl`` kernels were removed in v2.0.0 (R7) — they live on as
the compiled reference oracle (``circuitopt_core.OtftModel(..., reference=True)``,
selected in production by ``pmos_tft_model.otft_reference_mode``) — so the same
checks now pin the production and reference compiled modes against each other
and against finite differences of the public model API.
"""
import numpy as np
import pytest

import circuitopt.ac_solver as ac_solver
from circuitopt.pmos_tft_model import PMOS_TFT

try:
    import circuitopt_core
except ImportError:  # pragma: no cover - optional compiled wheel
    circuitopt_core = None

requires_rust_otft = pytest.mark.skipif(
    circuitopt_core is None or not hasattr(circuitopt_core, "OtftModel"),
    reason="circuitopt_core OtftModel is not installed",
)


def _reference_model(t):
    p = t.get_otft_params()
    values = [
        p.Vfb, p.Vss, p.Lc, p.lambda_, p.contact_scale, p.channel_exponent,
        p.current_scale, p.inv_Rleak, p.two_over_pi, p.cap_cgs1, p.cap_cgd1,
        p.cap_half_wl_ci, p.cap_cgs3_base, p.cap_cgd3_base, p.k1,
        p.gate_leak_g,
    ]
    return circuitopt_core.OtftModel(values, reference=True)


@requires_rust_otft
def test_eval_currents_production_matches_reference_oracle():
    """Production (powi Vt) and the reference oracle (libm-pow Vt) agree to
    ~1e-14 at production operating points — the same tolerance the retired
    rust-vs-``_impl`` comparison used."""
    t = PMOS_TFT(W=61365, L=61)
    ref_model = _reference_model(t)
    points = [
        (36.32147406780545, 29.07917946549335, 30.65),
        (29.07917946549335, 0.0, 5.5217968040937),
        (38.08434178857114, 5.5217968040937, 29.07917946549335),
        (40.0, 36.32147406780545, 9.84),
    ]
    for Vs, Vd, Vg in points:
        Vs1, Vd1 = t.get_op(Vs, Vd, Vg)
        got = np.array(t._eval_currents(Vs, Vd, Vg, Vs1, Vd1), float)
        ref = np.array(ref_model.eval_currents(Vs, Vd, Vg, Vs1, Vd1), float)
        np.testing.assert_allclose(got, ref, rtol=1e-14, atol=1e-24)


@requires_rust_otft
def test_capacitances_production_matches_reference_oracle():
    """Production reads (Cgss, Cgdd) off ``capacitance_charges``; the reference
    oracle uses the standalone capacitance equation. They agree to 1e-14."""
    t = PMOS_TFT(W=61365, L=61)
    ref_model = _reference_model(t)
    Vs, Vd, Vg = 36.32147406780545, 29.07917946549335, 30.65
    Vs1, Vd1 = t.get_op(Vs, Vd, Vg)
    got = ref_model.capacitances_pair(Vs, Vd, Vg, Vs1, Vd1)
    ref = t._capacitances_from_op(Vs, Vd, Vg, Vs1, Vd1)
    np.testing.assert_allclose(got, ref, rtol=1e-14, atol=1e-24)


@requires_rust_otft
def test_capacitance_charges_production_matches_reference_oracle():
    t = PMOS_TFT(W=61365, L=61)
    ref_model = _reference_model(t)
    Vs, Vd, Vg = 36.32147406780545, 29.07917946549335, 30.65
    Vs1, Vd1 = t.get_op(Vs, Vd, Vg)
    got = np.array(ref_model.capacitance_charges(Vs, Vd, Vg, Vs1, Vd1), float)
    ref = np.array(t._capacitance_charges_from_op(Vs, Vd, Vg, Vs1, Vd1), float)
    np.testing.assert_allclose(got, ref, rtol=1e-14, atol=1e-24)
    np.testing.assert_allclose(got[2:], t._capacitances_from_op(Vs, Vd, Vg, Vs1, Vd1),
                               rtol=1e-14, atol=1e-24)


def test_capacitance_branch_charge_local_derivative_matches_capacitance():
    t = PMOS_TFT(W=5000, L=30)
    Vs, Vd, Vg = 31.7, 30.8, 40.0
    h = 1e-6
    Vs1, Vd1 = t.get_op(Vs, Vd, Vg)
    qgs, qgd, Cgs, Cgd = t._capacitance_charges_from_op(Vs, Vd, Vg, Vs1, Vd1)

    Vs1p, Vd1p = t.get_op(Vs - h, Vd, Vg)
    qgs_p = t._capacitance_charges_from_op(Vs - h, Vd, Vg, Vs1p, Vd1p)[0]
    Vs1m, Vd1m = t.get_op(Vs + h, Vd, Vg)
    qgs_m = t._capacitance_charges_from_op(Vs + h, Vd, Vg, Vs1m, Vd1m)[0]
    np.testing.assert_allclose((qgs_p - qgs_m) / (2 * h), Cgs,
                               rtol=2e-4, atol=1e-16)

    Vs1p, Vd1p = t.get_op(Vs, Vd - h, Vg)
    qgd_p = t._capacitance_charges_from_op(Vs, Vd - h, Vg, Vs1p, Vd1p)[1]
    Vs1m, Vd1m = t.get_op(Vs, Vd + h, Vg)
    qgd_m = t._capacitance_charges_from_op(Vs, Vd + h, Vg, Vs1m, Vd1m)[1]
    np.testing.assert_allclose((qgd_p - qgd_m) / (2 * h), Cgd,
                               rtol=2e-4, atol=1e-16)
    assert np.isfinite(qgs)
    assert np.isfinite(qgd)


def test_capacitance_components_and_channel_charge_are_pdk_scaled():
    t = PMOS_TFT(W=20000, L=80)
    Vs, Vd, Vg = 30.65, 29.0, 0.0
    comps = t.get_capacitance_components(Vs, Vd, Vg)
    Cgss, Cgdd = t.get_capacitances(Vs, Vd, Vg)

    np.testing.assert_allclose(comps["Cgss"], Cgss, rtol=1e-14, atol=1e-24)
    np.testing.assert_allclose(comps["Cgdd"], Cgdd, rtol=1e-14, atol=1e-24)
    assert comps["Cgs2"] > 0.0
    assert comps["Cgd2"] > 0.0
    assert t.estimate_channel_charge(Vs, Vd, Vg) > 0.0


# (v2.0.0) test_numba_kernel_matches_python_impl_when_enabled,
# test_additional_numba_kernels_match_python_impl_when_enabled and
# test_pnoise_numba_kernels_match_reference_when_enabled were removed in R6
# (numba JIT deleted; comparisons became tautologies). In R7 the interpreted
# ``_impl`` kernels themselves were removed; the compiled reference oracle
# (OtftModel(reference=True)) carries their equations and is pinned above.


def test_get_ss_params_analytic_path_matches_finite_difference():
    """The analytic gm/gds fast path agrees with central finite differences of
    the public ``get_Idc``. (Formerly asserted via the retired numba-handle
    monkeypatch; the FD referee is now computed directly.)"""
    point = (40.0, 31.38, 0.0)
    fast_dev = PMOS_TFT(W=5000, L=30)
    fast = ac_solver.get_ss_params(5000, 30, *point, dev_inst=fast_dev)

    h = 1e-3
    fd_dev = PMOS_TFT(W=5000, L=30)
    Vs, Vd, Vg = point
    gm_fd = (fd_dev.get_Idc(Vs, Vd, Vg + h) - fd_dev.get_Idc(Vs, Vd, Vg - h)) / (2 * h)
    gds_fd = (fd_dev.get_Idc(Vs, Vd + h, Vg) - fd_dev.get_Idc(Vs, Vd - h, Vg)) / (2 * h)
    Cgs_fd, Cgd_fd = fd_dev.get_capacitances(Vs, Vd, Vg)

    np.testing.assert_allclose(fast["gm"], gm_fd, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(fast["gds"], gds_fd, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(fast["Cgs"], Cgs_fd, rtol=1e-14, atol=1e-24)
    np.testing.assert_allclose(fast["Cgd"], Cgd_fd, rtol=1e-14, atol=1e-24)


@requires_rust_otft
def test_production_terminal_derivatives_match_reference_oracle():
    """Production analytic-Jacobian terminal derivatives vs the reference
    oracle's finite-difference-from-base derivatives (hh=1e-3, hx=1e-6) —
    the same pairing the retired ``_impl`` A/B established."""
    t = PMOS_TFT(W=1000, L=20)
    prod_model = t._get_rust_model()
    ref_model = _reference_model(t)
    points = [
        (40.0, 0.0, 20.0),
        (36.32147406780545, 29.07917946549335, 30.65),
        (29.07917946549335, 0.0, 5.5217968040937),
        (38.08434178857114, 5.5217968040937, 29.07917946549335),
        (32.0, 31.7, 40.0),
    ]
    for Vs, Vd, Vg in points:
        Vs1, Vd1 = t.get_op(Vs, Vd, Vg)
        got = prod_model.terminal_derivatives(
            Vs, Vd, Vg, Vs1, Vd1, True, True, True, 1e-3)
        ref = ref_model.terminal_derivatives(
            Vs, Vd, Vg, Vs1, Vd1, True, True, True, 1e-3)
        assert got[0] == ref[0]
        np.testing.assert_allclose(got[1:], ref[1:], rtol=1e-7, atol=1e-17)


@requires_rust_otft
def test_signed_terminal_derivatives_allow_zero_crossing_current():
    """``use_abs=False`` keeps derivatives at a zero-current bias, and the
    public small-signal path stays finite there via its small-current fallback.

    (The retired ``_impl`` variant additionally injected a synthetic
    ``idc0 = 0.0`` to unit-test the absolute-current guard branch; the compiled
    binding derives idc0 internally, so the production-reachable zero-crossing
    behavior — ``get_ss_params``'s |Idc| < 1e-10 finite-difference fallback —
    is asserted instead.)"""
    t = PMOS_TFT(W=5000, L=30)
    prod_model = t._get_rust_model()
    Vs = Vd = 31.38
    Vg = 40.0
    Vs1, Vd1 = t.get_op(Vs, Vd, Vg)

    signed = prod_model.terminal_derivatives(
        Vs, Vd, Vg, Vs1, Vd1, True, True, False, 1e-3)
    assert signed[0]
    assert np.all(np.isfinite(signed[1:]))

    # Vs == Vd -> |Idc| is far below the 1e-10 guard; the public path must
    # take the finite-difference fallback and still return finite params.
    assert abs(t.get_Idc(Vs, Vd, Vg)) < 1e-10
    ss = t.get_ss_params(Vs, Vd, Vg)
    assert np.all(np.isfinite([ss["gm"], ss["gds"], ss["Cgs"], ss["Cgd"]]))
