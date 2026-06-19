import numpy as np

import core.ac_solver as ac_solver
from core.numba_kernels import (
    _capacitance_charges_impl,
    _capacitances_impl,
    _eval_currents_impl,
    _newton_internal_impl,
    _pnoise_fold_psd_impl,
    _pnoise_hb_blocks_impl,
    _residual_pair_jac_internal_impl,
    _terminal_derivatives_from_jac_impl,
    _terminal_derivatives_impl,
    capacitance_charges_numba,
    capacitances_numba,
    eval_currents_numba,
    newton_internal_numba,
    pnoise_fold_psd_numba,
    pnoise_hb_blocks_numba,
    terminal_derivatives_numba,
)
from core.pmos_tft_model import PMOS_TFT


def _kernel_args(t, Vs, Vd, Vg, Vs1, Vd1):
    return (
        Vs, Vd, Vg, Vs1, Vd1, t.Vfb, t.Vss, t.Lc, t.lambda_,
        t._contact_scale, t._channel_exponent, t._current_scale, t._inv_Rleak,
    )


def test_eval_currents_kernel_matches_model():
    t = PMOS_TFT(W=61365, L=61)
    points = [
        (36.32147406780545, 29.07917946549335, 30.65),
        (29.07917946549335, 0.0, 5.5217968040937),
        (38.08434178857114, 5.5217968040937, 29.07917946549335),
        (40.0, 36.32147406780545, 9.84),
    ]
    for Vs, Vd, Vg in points:
        Vs1, Vd1 = t.get_op(Vs, Vd, Vg)
        got = np.array(t._eval_currents(Vs, Vd, Vg, Vs1, Vd1), float)
        ref = np.array(_eval_currents_impl(*_kernel_args(t, Vs, Vd, Vg, Vs1, Vd1)), float)
        np.testing.assert_allclose(got, ref, rtol=1e-14, atol=1e-24)


def test_numba_kernel_matches_python_impl_when_enabled():
    if eval_currents_numba is None:
        return
    t = PMOS_TFT(W=1000, L=20)
    Vs, Vd, Vg = 40.0, 0.0, 20.0
    Vs1, Vd1 = t.get_op(Vs, Vd, Vg)
    args = _kernel_args(t, Vs, Vd, Vg, Vs1, Vd1)
    got = np.array(eval_currents_numba(*args), float)
    ref = np.array(_eval_currents_impl(*args), float)
    np.testing.assert_allclose(got, ref, rtol=1e-14, atol=1e-24)


def test_capacitance_kernel_matches_model_formula():
    t = PMOS_TFT(W=61365, L=61)
    Vs, Vd, Vg = 36.32147406780545, 29.07917946549335, 30.65
    Vs1, Vd1 = t.get_op(Vs, Vd, Vg)
    got = _capacitances_impl(
        Vs, Vd, Vg, Vs1, Vd1, t.Vfb, t._two_over_pi, t._cap_cgs1,
        t._cap_cgd1, t._cap_half_wl_ci, t._cap_cgs3_base,
        t._cap_cgd3_base, t.k1)
    ref = t._capacitances_from_op(Vs, Vd, Vg, Vs1, Vd1)
    np.testing.assert_allclose(got, ref, rtol=1e-14, atol=1e-24)


def test_capacitance_charge_kernel_matches_model_formula():
    t = PMOS_TFT(W=61365, L=61)
    Vs, Vd, Vg = 36.32147406780545, 29.07917946549335, 30.65
    Vs1, Vd1 = t.get_op(Vs, Vd, Vg)
    args = (
        Vs, Vd, Vg, Vs1, Vd1, t.Vfb, t._two_over_pi, t._cap_cgs1,
        t._cap_cgd1, t._cap_half_wl_ci, t._cap_cgs3_base,
        t._cap_cgd3_base, t.k1,
    )
    got = _capacitance_charges_impl(*args)
    ref = t._capacitance_charges_from_op(Vs, Vd, Vg, Vs1, Vd1)
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


def test_additional_numba_kernels_match_python_impl_when_enabled():
    if newton_internal_numba is None:
        return
    t = PMOS_TFT(W=1000, L=20)
    Vs, Vd, Vg = 40.0, 0.0, 20.0
    x0 = (Vs - 0.01 * (Vs - Vd), Vd + 0.01 * (Vs - Vd))
    args = (
        Vs, Vd, Vg, x0[0], x0[1], 1e-12, 40, t.Vfb, t.Vss, t.Lc,
        t.lambda_, t._contact_scale, t._channel_exponent, t._current_scale,
        t._inv_Rleak,
    )
    got = newton_internal_numba(*args)
    ref = _newton_internal_impl(*args)
    assert got[0] == ref[0]
    np.testing.assert_allclose(got[1:], ref[1:], rtol=1e-14, atol=1e-12)

    Vs1, Vd1 = got[1], got[2]
    cap_args = (
        Vs, Vd, Vg, Vs1, Vd1, t.Vfb, t._two_over_pi, t._cap_cgs1,
        t._cap_cgd1, t._cap_half_wl_ci, t._cap_cgs3_base,
        t._cap_cgd3_base, t.k1,
    )
    np.testing.assert_allclose(
        capacitances_numba(*cap_args), _capacitances_impl(*cap_args),
        rtol=1e-14, atol=1e-24)
    np.testing.assert_allclose(
        capacitance_charges_numba(*cap_args), _capacitance_charges_impl(*cap_args),
        rtol=1e-14, atol=1e-24)

    td_args = (
        Vs, Vd, Vg, Vs1, Vd1, True, True, True, 1e-3, 1e-6, t.Vfb, t.Vss,
        t.Lc, t.lambda_, t._contact_scale, t._channel_exponent,
        t._current_scale, t._inv_Rleak,
    )
    got_td = terminal_derivatives_numba(*td_args)
    ref_td = _terminal_derivatives_impl(*td_args)
    assert got_td[0] == ref_td[0]
    np.testing.assert_allclose(got_td[1:], ref_td[1:], rtol=1e-14, atol=1e-18)


def test_pnoise_numba_kernels_match_reference_when_enabled():
    if pnoise_hb_blocks_numba is None:
        return

    rng = np.random.default_rng(4)
    N = 8
    n = 2
    K = 2
    Gf = rng.normal(size=(N, n, n)) + 1j * rng.normal(size=(N, n, n))
    Cf = rng.normal(size=(N, n, n)) + 1j * rng.normal(size=(N, n, n))

    got_y, got_c = pnoise_hb_blocks_numba(Gf, Cf, K, 225.0)
    ref_y, ref_c = _pnoise_hb_blocks_impl.py_func(Gf, Cf, K, 225.0)
    np.testing.assert_allclose(got_y, ref_y, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(got_c, ref_c, rtol=1e-14, atol=1e-14)

    nfreq = 3
    nb = 2 * K + 1
    adjs = rng.normal(size=(nfreq, nb * n)) + 1j * rng.normal(size=(nfreq, nb * n))
    freqs = np.array([0.1, 10.0, 100.0])
    p_indices = np.array([[0, 2, 4, 6, 8], [-1, -1, -1, -1, -1]], dtype=np.int64)
    q_indices = np.array([[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]], dtype=np.int64)
    sth = rng.normal(size=(2, nb, nb)) + 1j * rng.normal(size=(2, nb, nb))
    sfl = rng.normal(size=(2, nb, nb)) + 1j * rng.normal(size=(2, nb, nb))

    got_out, got_dev = pnoise_fold_psd_numba(
        adjs, freqs, K, 225.0, p_indices, q_indices, sth, sfl)
    ref_out, ref_dev = _pnoise_fold_psd_impl.py_func(
        adjs, freqs, K, 225.0, p_indices, q_indices, sth, sfl)
    np.testing.assert_allclose(got_out, ref_out, rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(got_dev, ref_dev, rtol=1e-14, atol=1e-12)


def test_get_ss_params_numba_fast_path_matches_finite_difference(monkeypatch):
    point = (40.0, 31.38, 0.0)
    fast_dev = PMOS_TFT(W=5000, L=30)
    fast = ac_solver.get_ss_params(5000, 30, *point, dev_inst=fast_dev)

    monkeypatch.setattr(ac_solver, "terminal_derivatives_numba", None)
    fd_dev = PMOS_TFT(W=5000, L=30)
    ref = ac_solver.get_ss_params(5000, 30, *point, dev_inst=fd_dev)

    np.testing.assert_allclose(fast["gm"], ref["gm"], rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(fast["gds"], ref["gds"], rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(fast["Cgs"], ref["Cgs"], rtol=1e-14, atol=1e-24)
    np.testing.assert_allclose(fast["Cgd"], ref["Cgd"], rtol=1e-14, atol=1e-24)


def test_transient_analytic_terminal_derivatives_match_finite_difference():
    t = PMOS_TFT(W=1000, L=20)
    points = [
        (40.0, 0.0, 20.0),
        (36.32147406780545, 29.07917946549335, 30.65),
        (29.07917946549335, 0.0, 5.5217968040937),
        (38.08434178857114, 5.5217968040937, 29.07917946549335),
        (32.0, 31.7, 40.0),
    ]
    for Vs, Vd, Vg in points:
        Vs1, Vd1 = t.get_op(Vs, Vd, Vg)
        F0a, F0b, j00, j01, j10, j11 = _residual_pair_jac_internal_impl(
            Vs, Vd, Vg, Vs1, Vd1, t.Vfb, t.Vss, t.Lc, t.lambda_,
            t._contact_scale, t._channel_exponent, t._current_scale,
            t._inv_Rleak)
        Idc0 = F0b - (Vs1 - Vd1) / 0.1
        got = _terminal_derivatives_from_jac_impl(
            Vs, Vd, Vg, Vs1, Vd1, F0a, F0b, Idc0, j00, j01, j10, j11,
            True, True, True, 1e-3, t.Vfb, t.Vss, t.Lc, t.lambda_,
            t._contact_scale, t._channel_exponent, t._current_scale,
            t._inv_Rleak)
        ref = _terminal_derivatives_impl(
            Vs, Vd, Vg, Vs1, Vd1, True, True, True, 1e-3, 1e-6,
            t.Vfb, t.Vss, t.Lc, t.lambda_, t._contact_scale,
            t._channel_exponent, t._current_scale, t._inv_Rleak)
        assert got[0] == ref[0]
        np.testing.assert_allclose(got[1:], ref[1:], rtol=1e-7, atol=1e-17)


def test_signed_terminal_derivatives_allow_zero_crossing_current():
    t = PMOS_TFT(W=5000, L=30)
    Vs = Vd = 31.38
    Vg = 40.0
    Vs1, Vd1 = t.get_op(Vs, Vd, Vg)
    F0a, F0b, j00, j01, j10, j11 = _residual_pair_jac_internal_impl(
        Vs, Vd, Vg, Vs1, Vd1, t.Vfb, t.Vss, t.Lc, t.lambda_,
        t._contact_scale, t._channel_exponent, t._current_scale,
        t._inv_Rleak)

    signed = _terminal_derivatives_from_jac_impl(
        Vs, Vd, Vg, Vs1, Vd1, F0a, F0b, 0.0, j00, j01, j10, j11,
        True, True, False, 1e-3, t.Vfb, t.Vss, t.Lc, t.lambda_,
        t._contact_scale, t._channel_exponent, t._current_scale,
        t._inv_Rleak)
    abs_current = _terminal_derivatives_from_jac_impl(
        Vs, Vd, Vg, Vs1, Vd1, F0a, F0b, 0.0, j00, j01, j10, j11,
        True, True, True, 1e-3, t.Vfb, t.Vss, t.Lc, t.lambda_,
        t._contact_scale, t._channel_exponent, t._current_scale,
        t._inv_Rleak)

    assert signed[0]
    assert np.all(np.isfinite(signed[1:]))
    assert not abs_current[0]
