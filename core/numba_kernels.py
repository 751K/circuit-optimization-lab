"""Optional numba-accelerated scalar kernels.

This module must remain importable without numba installed. Callers opt in via
the CIRCUIT_USE_NUMBA=1 environment variable, so normal short runs do not pay
numba's first-call compilation cost.
"""
import math
import os


USE_NUMBA = os.environ.get("CIRCUIT_USE_NUMBA", "").lower() in {"1", "true", "yes", "on"}

try:
    from numba import njit
except Exception:  # pragma: no cover - depends on optional dependency
    njit = None


NUMBA_AVAILABLE = USE_NUMBA and njit is not None


def _softplus_py(x):
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def _eval_currents_impl(Vs, Vd, Vg, Vs1, Vd1, Vfb, Vss, Lc, lambda_,
                        contact_scale, exponent, current_scale, inv_Rleak):
    v_s = Vs if Vs > Vs1 else Vs1
    v_s1 = Vs1 if Vs > Vs1 else Vs
    v_d = Vd if Vd1 > Vd else Vd1
    v_d1 = Vd1 if Vd1 > Vd else Vd

    Vt = -(0.0045 * (v_s - Vg) ** 2 + 0.7125 * (v_s - Vg) + 0.9625)
    Vods1 = Vss * _softplus_py((v_s - Vg + Vt) / Vss)
    Vodd1 = Vss * _softplus_py((v_s1 - Vg + Vt) / Vss)

    Ecsat = 17.0 / (abs(v_s - Vg) + 0.1)
    lambdac = 1.0 / (Lc * Ecsat)
    cmod = 1.0 + lambdac * (v_s - v_s1)
    Icont = contact_scale * (Vods1 ** exponent - Vodd1 ** exponent) * cmod
    I_s_s1 = Icont if Vs > Vs1 else -Icont

    arg_d1 = (v_d1 - Vg + Vfb) / Vss
    arg_d = (v_d - Vg + Vfb) / Vss
    Vods = Vss * _softplus_py(arg_d1)
    Vodd = Vss * _softplus_py(arg_d)
    chmod = 1.0 + lambda_ * (v_d1 - v_d)
    Ich = current_scale * (Vods ** exponent - Vodd ** exponent) * chmod

    I_d1_d_ch = Ich if Vs1 > Vd else -Ich
    I_d1_d_leak = (Vd1 - Vd + 0.1) * inv_Rleak
    I_d1_d = I_d1_d_ch + I_d1_d_leak
    I_s1_d1 = (Vs1 - Vd1) / 0.1
    return I_s_s1, I_s1_d1, I_d1_d, Ich, I_d1_d_leak


if NUMBA_AVAILABLE:
    _softplus_py = njit(cache=True)(_softplus_py)
    eval_currents_numba = njit(cache=True)(_eval_currents_impl)
else:
    eval_currents_numba = None
