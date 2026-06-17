import numpy as np

from core.numba_kernels import (
    _capacitances_impl,
    _eval_currents_impl,
    _newton_internal_impl,
    _terminal_derivatives_impl,
    capacitances_numba,
    eval_currents_numba,
    newton_internal_numba,
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

    td_args = (
        Vs, Vd, Vg, Vs1, Vd1, True, True, 1e-3, 1e-6, t.Vfb, t.Vss,
        t.Lc, t.lambda_, t._contact_scale, t._channel_exponent,
        t._current_scale, t._inv_Rleak,
    )
    got_td = terminal_derivatives_numba(*td_args)
    ref_td = _terminal_derivatives_impl(*td_args)
    assert got_td[0] == ref_td[0]
    np.testing.assert_allclose(got_td[1:], ref_td[1:], rtol=1e-14, atol=1e-18)
