import numpy as np

from core.numba_kernels import _eval_currents_impl, eval_currents_numba
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
