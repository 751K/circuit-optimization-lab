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


def _residual_pair_impl(Vs, Vd, Vg, Vs1, Vd1, Vfb, Vss, Lc, lambda_,
                        contact_scale, exponent, current_scale, inv_Rleak):
    I_s_s1, I_s1_d1, I_d1_d, _, _ = _eval_currents_impl(
        Vs, Vd, Vg, Vs1, Vd1, Vfb, Vss, Lc, lambda_, contact_scale, exponent,
        current_scale, inv_Rleak)
    return I_s_s1 - I_s1_d1, I_s1_d1 - I_d1_d


def _newton_internal_impl(Vs, Vd, Vg, x0s, x0d, tol, maxit, Vfb, Vss, Lc, lambda_,
                          contact_scale, exponent, current_scale, inv_Rleak):
    Vs1 = x0s
    Vd1 = x0d
    hj = 1e-6
    for _ in range(maxit):
        r0a, r0b = _residual_pair_impl(
            Vs, Vd, Vg, Vs1, Vd1, Vfb, Vss, Lc, lambda_, contact_scale,
            exponent, current_scale, inv_Rleak)
        if abs(r0a) + abs(r0b) < tol:
            return True, Vs1, Vd1
        r1a, r1b = _residual_pair_impl(
            Vs, Vd, Vg, Vs1 + hj, Vd1, Vfb, Vss, Lc, lambda_, contact_scale,
            exponent, current_scale, inv_Rleak)
        r2a, r2b = _residual_pair_impl(
            Vs, Vd, Vg, Vs1, Vd1 + hj, Vfb, Vss, Lc, lambda_, contact_scale,
            exponent, current_scale, inv_Rleak)
        j00 = (r1a - r0a) / hj
        j01 = (r2a - r0a) / hj
        j10 = (r1b - r0b) / hj
        j11 = (r2b - r0b) / hj
        det = j00 * j11 - j01 * j10
        if det == 0.0 or not math.isfinite(det):
            return False, Vs1, Vd1
        d0 = -(j11 * r0a - j01 * r0b) / det
        d1 = -(-j10 * r0a + j00 * r0b) / det
        mx = abs(d0) if abs(d0) > abs(d1) else abs(d1)
        if mx > 2.0:
            scale = 2.0 / mx
            d0 *= scale
            d1 *= scale
            mx = 2.0
        Vs1 += d0
        Vd1 += d1
        if mx < 1e-13:
            if abs(r0a) + abs(r0b) < 1e-9:
                return True, Vs1, Vd1
            return False, Vs1, Vd1
    return False, Vs1, Vd1


def _capacitances_impl(Vs, Vd, Vg, Vs1, Vd1, Vfb, two_over_pi, cap_cgs1,
                       cap_cgd1, cap_half_wl_ci, cap_cgs3_base,
                       cap_cgd3_base, k1):
    v_s = Vs if Vs > Vs1 else Vs1
    v_d = Vd if Vd1 > Vd else Vd1

    arg_gs = v_s - Vg + Vfb
    Cgs2 = 1.43 * cap_half_wl_ci * (two_over_pi * math.atan(arg_gs * 0.6) + 1.0)
    Cgd2 = 0.33 * cap_half_wl_ci * (two_over_pi * math.atan(arg_gs * 2.01) + 1.0)

    arg_gd = -Vg + Vfb + v_d
    Cgs3 = 0.34 * cap_cgs3_base * (two_over_pi * math.atan(arg_gd * 0.21) + 1.0)
    Cgd3 = 0.52 * cap_cgd3_base * (two_over_pi * math.atan(arg_gd * 0.42) + 1.0)

    Cgss = k1 * (cap_cgs1 + Cgs2 + Cgs3) * 1e4 * 1e-12
    Cgdd = k1 * (cap_cgd1 + Cgd2 + Cgd3) * 1e4 * 1e-12
    return Cgss, Cgdd


def _eval_at_impl(Vs, Vd, Vg, Vs1, Vd1, Vfb, Vss, Lc, lambda_,
                  contact_scale, exponent, current_scale, inv_Rleak):
    I_s_s1, I_s1_d1, I_d1_d, _, _ = _eval_currents_impl(
        Vs, Vd, Vg, Vs1, Vd1, Vfb, Vss, Lc, lambda_, contact_scale, exponent,
        current_scale, inv_Rleak)
    return I_s_s1 - I_s1_d1, I_s1_d1 - I_d1_d, -I_d1_d


def _terminal_deriv_one_impl(vs_p, vd_p, vg_p, vs_m, vd_m, vg_m, Vs1, Vd1,
                             Idc0, j00, j01, j10, j11, ix0, ix1, det, sign,
                             HH, Vfb, Vss, Lc, lambda_, contact_scale, exponent,
                             current_scale, inv_Rleak):
    Fpa, Fpb, Ip = _eval_at_impl(
        vs_p, vd_p, vg_p, Vs1, Vd1, Vfb, Vss, Lc, lambda_, contact_scale,
        exponent, current_scale, inv_Rleak)
    Fma, Fmb, Im = _eval_at_impl(
        vs_m, vd_m, vg_m, Vs1, Vd1, Vfb, Vss, Lc, lambda_, contact_scale,
        exponent, current_scale, inv_Rleak)
    fu0 = (Fpa - Fma) / (2.0 * HH)
    fu1 = (Fpb - Fmb) / (2.0 * HH)
    Iu = (Ip - Im) / (2.0 * HH)
    y0 = (j11 * fu0 - j01 * fu1) / det
    y1 = (-j10 * fu0 + j00 * fu1) / det
    return sign * (Iu - ix0 * y0 - ix1 * y1)


def _terminal_derivatives_impl(Vs, Vd, Vg, Vs1, Vd1, need_gm, need_gds, HH, hx,
                               Vfb, Vss, Lc, lambda_, contact_scale, exponent,
                               current_scale, inv_Rleak):
    if not need_gm and not need_gds:
        return True, 0.0, 0.0
    F0a, F0b, Idc0 = _eval_at_impl(
        Vs, Vd, Vg, Vs1, Vd1, Vfb, Vss, Lc, lambda_, contact_scale, exponent,
        current_scale, inv_Rleak)
    if abs(Idc0) < 1e-30:
        return False, 0.0, 0.0
    Fpa, Fpb, Ip = _eval_at_impl(
        Vs, Vd, Vg, Vs1 + hx, Vd1, Vfb, Vss, Lc, lambda_, contact_scale,
        exponent, current_scale, inv_Rleak)
    j00 = (Fpa - F0a) / hx
    j10 = (Fpb - F0b) / hx
    ix0 = (Ip - Idc0) / hx
    Fpa, Fpb, Ip = _eval_at_impl(
        Vs, Vd, Vg, Vs1, Vd1 + hx, Vfb, Vss, Lc, lambda_, contact_scale,
        exponent, current_scale, inv_Rleak)
    j01 = (Fpa - F0a) / hx
    j11 = (Fpb - F0b) / hx
    ix1 = (Ip - Idc0) / hx
    det = j00 * j11 - j01 * j10
    if det == 0.0 or not math.isfinite(det):
        return False, 0.0, 0.0
    sign = 1.0 if Idc0 > 0.0 else -1.0
    gm = 0.0
    gds = 0.0
    if need_gm:
        gm = _terminal_deriv_one_impl(
            Vs, Vd, Vg + HH, Vs, Vd, Vg - HH, Vs1, Vd1, Idc0,
            j00, j01, j10, j11, ix0, ix1, det, sign, HH, Vfb, Vss, Lc,
            lambda_, contact_scale, exponent, current_scale, inv_Rleak)
    if need_gds:
        gds = _terminal_deriv_one_impl(
            Vs, Vd + HH, Vg, Vs, Vd - HH, Vg, Vs1, Vd1, Idc0,
            j00, j01, j10, j11, ix0, ix1, det, sign, HH, Vfb, Vss, Lc,
            lambda_, contact_scale, exponent, current_scale, inv_Rleak)
    return True, gm, gds


if NUMBA_AVAILABLE:
    # Keep cache disabled: this project is commonly run both as a package
    # (`core.numba_kernels`) and, historically, as flat modules. Numba's on-disk
    # cache stores the module path and stale flat-module cache entries can be
    # loaded after the package migration, breaking optional acceleration.
    _softplus_py = njit(cache=False)(_softplus_py)
    _eval_currents_impl = njit(cache=False)(_eval_currents_impl)
    _residual_pair_impl = njit(cache=False)(_residual_pair_impl)
    _eval_at_impl = njit(cache=False)(_eval_at_impl)
    _terminal_deriv_one_impl = njit(cache=False)(_terminal_deriv_one_impl)
    eval_currents_numba = _eval_currents_impl
    newton_internal_numba = njit(cache=False)(_newton_internal_impl)
    capacitances_numba = njit(cache=False)(_capacitances_impl)
    terminal_derivatives_numba = njit(cache=False)(_terminal_derivatives_impl)
else:
    eval_currents_numba = None
    newton_internal_numba = None
    capacitances_numba = None
    terminal_derivatives_numba = None
