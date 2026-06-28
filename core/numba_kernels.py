"""Optional numba-accelerated scalar kernels.

This module must remain importable without numba installed. When numba is
available the kernels are enabled by default; set CIRCUIT_USE_NUMBA=0/false/off
to force the pure-Python path for debugging. Numba's on-disk cache is enabled
by default; set CIRCUIT_NUMBA_CACHE=0/false/off to disable it.
"""
import math
import os

import numpy as np

try:
    from .adaptive_config import (
        ADAPTIVE_ACCEPT_WRMS,
        ADAPTIVE_DONE_ABS,
        ADAPTIVE_DONE_REL,
        ADAPTIVE_ERR_FLOOR,
        ADAPTIVE_GROWTH_MAX,
        ADAPTIVE_GROWTH_MIN,
        ADAPTIVE_INITIAL_MIN_DENOM,
        ADAPTIVE_INPUT_SLOPE_BREAK_FRACTION,
        ADAPTIVE_LTE_DIVISOR,
        ADAPTIVE_MIN_H_ABS,
        ADAPTIVE_MIN_H_REL,
        ADAPTIVE_SAFETY,
        ADAPTIVE_SCALE_FLOOR,
        ADAPTIVE_STEP_ORDER,
    )
except ImportError:  # pragma: no cover - legacy direct module import
    from adaptive_config import (
        ADAPTIVE_ACCEPT_WRMS,
        ADAPTIVE_DONE_ABS,
        ADAPTIVE_DONE_REL,
        ADAPTIVE_ERR_FLOOR,
        ADAPTIVE_GROWTH_MAX,
        ADAPTIVE_GROWTH_MIN,
        ADAPTIVE_INITIAL_MIN_DENOM,
        ADAPTIVE_INPUT_SLOPE_BREAK_FRACTION,
        ADAPTIVE_LTE_DIVISOR,
        ADAPTIVE_MIN_H_ABS,
        ADAPTIVE_MIN_H_REL,
        ADAPTIVE_SAFETY,
        ADAPTIVE_SCALE_FLOOR,
        ADAPTIVE_STEP_ORDER,
    )


_FALSE_ENV_VALUES = {"0", "false", "no", "off"}
_USE_NUMBA_FLAG = os.environ.get("CIRCUIT_USE_NUMBA")
USE_NUMBA = (_USE_NUMBA_FLAG is None or
             _USE_NUMBA_FLAG.lower() in {"1", "true", "yes", "on"})
if _USE_NUMBA_FLAG is not None and _USE_NUMBA_FLAG.lower() in _FALSE_ENV_VALUES:
    USE_NUMBA = False

_NUMBA_CACHE_FLAG = os.environ.get("CIRCUIT_NUMBA_CACHE")
NUMBA_CACHE = (_NUMBA_CACHE_FLAG is None or
               _NUMBA_CACHE_FLAG.lower() not in _FALSE_ENV_VALUES)
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


def _sigmoid_impl(x):
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _residual_pair_fd_jac_impl(Vs, Vd, Vg, Vs1, Vd1, hj, Vfb, Vss, Lc,
                               lambda_, contact_scale, exponent,
                               current_scale, inv_Rleak):
    r0a, r0b = _residual_pair_impl(
        Vs, Vd, Vg, Vs1, Vd1, Vfb, Vss, Lc, lambda_, contact_scale,
        exponent, current_scale, inv_Rleak)
    r1a, r1b = _residual_pair_impl(
        Vs, Vd, Vg, Vs1 + hj, Vd1, Vfb, Vss, Lc, lambda_, contact_scale,
        exponent, current_scale, inv_Rleak)
    r2a, r2b = _residual_pair_impl(
        Vs, Vd, Vg, Vs1, Vd1 + hj, Vfb, Vss, Lc, lambda_, contact_scale,
        exponent, current_scale, inv_Rleak)
    return (r0a, r0b, (r1a - r0a) / hj, (r2a - r0a) / hj,
            (r1b - r0b) / hj, (r2b - r0b) / hj)


def _residual_pair_jac_internal_impl(Vs, Vd, Vg, Vs1, Vd1, Vfb, Vss, Lc,
                                     lambda_, contact_scale, exponent,
                                     current_scale, inv_Rleak):
    # The compact model has branch min/max sign changes. Around those kinks the
    # old finite-difference Jacobian is the safer local linearization.
    if (abs(Vs - Vs1) < 1e-10 or abs(Vd1 - Vd) < 1e-10 or
            abs(Vs1 - Vd) < 1e-10):
        return _residual_pair_fd_jac_impl(
            Vs, Vd, Vg, Vs1, Vd1, 1e-6, Vfb, Vss, Lc, lambda_,
            contact_scale, exponent, current_scale, inv_Rleak)

    # Contact branch I_s_s1 and derivative with respect to Vs1.
    if Vs > Vs1:
        v_s = Vs
        v_s1 = Vs1
        dv = v_s - Vg
        Vt = -(0.0045 * dv ** 2 + 0.7125 * dv + 0.9625)
        arg_a = (v_s - Vg + Vt) / Vss
        arg_b = (v_s1 - Vg + Vt) / Vss
        A = Vss * _softplus_py(arg_a)
        B = Vss * _softplus_py(arg_b)
        Ap = A ** exponent
        Bp = B ** exponent
        Aem1 = Ap / A if A != 0.0 else 0.0
        Bem1 = Bp / B if B != 0.0 else 0.0
        Ecsat = 17.0 / (abs(dv) + 0.1)
        lambdac = 1.0 / (Lc * Ecsat)
        cmod = 1.0 + lambdac * (v_s - v_s1)
        Icont = contact_scale * (Ap - Bp) * cmod
        dB = _sigmoid_impl(arg_b)
        dIcont = contact_scale * (
            -exponent * Bem1 * dB * cmod -
            (Ap - Bp) * lambdac)
        I_s_s1 = Icont
        dIss_dVs1 = dIcont
    else:
        v_s = Vs1
        v_s1 = Vs
        dv = v_s - Vg
        Vt = -(0.0045 * dv ** 2 + 0.7125 * dv + 0.9625)
        dVt = -(0.009 * dv + 0.7125)
        arg_a = (v_s - Vg + Vt) / Vss
        arg_b = (v_s1 - Vg + Vt) / Vss
        A = Vss * _softplus_py(arg_a)
        B = Vss * _softplus_py(arg_b)
        Ap = A ** exponent
        Bp = B ** exponent
        Aem1 = Ap / A if A != 0.0 else 0.0
        Bem1 = Bp / B if B != 0.0 else 0.0
        Ecsat = 17.0 / (abs(dv) + 0.1)
        lambdac = 1.0 / (Lc * Ecsat)
        cmod = 1.0 + lambdac * (v_s - v_s1)
        Icont = contact_scale * (Ap - Bp) * cmod
        sign_dv = 1.0 if dv > 0.0 else -1.0
        dlambdac = sign_dv / (17.0 * Lc)
        dA = _sigmoid_impl(arg_a) * (1.0 + dVt)
        dB = _sigmoid_impl(arg_b) * dVt
        dcmod = dlambdac * (v_s - v_s1) + lambdac
        dIcont = contact_scale * (
            (exponent * Aem1 * dA - exponent * Bem1 * dB) * cmod +
            (Ap - Bp) * dcmod)
        I_s_s1 = -Icont
        dIss_dVs1 = -dIcont

    # Channel branch I_d1_d and derivative with respect to Vd1.
    if Vd1 > Vd:
        v_d = Vd
        v_d1 = Vd1
        arg_a = (v_d1 - Vg + Vfb) / Vss
        arg_b = (v_d - Vg + Vfb) / Vss
        A = Vss * _softplus_py(arg_a)
        B = Vss * _softplus_py(arg_b)
        Ap = A ** exponent
        Bp = B ** exponent
        Aem1 = Ap / A if A != 0.0 else 0.0
        chmod = 1.0 + lambda_ * (v_d1 - v_d)
        Ich = current_scale * (Ap - Bp) * chmod
        dA = _sigmoid_impl(arg_a)
        dIch = current_scale * (
            exponent * Aem1 * dA * chmod +
            (Ap - Bp) * lambda_)
    else:
        v_d = Vd1
        v_d1 = Vd
        arg_a = (v_d1 - Vg + Vfb) / Vss
        arg_b = (v_d - Vg + Vfb) / Vss
        A = Vss * _softplus_py(arg_a)
        B = Vss * _softplus_py(arg_b)
        Ap = A ** exponent
        Bp = B ** exponent
        Bem1 = Bp / B if B != 0.0 else 0.0
        chmod = 1.0 + lambda_ * (v_d1 - v_d)
        Ich = current_scale * (Ap - Bp) * chmod
        dB = _sigmoid_impl(arg_b)
        dIch = current_scale * (
            -exponent * Bem1 * dB * chmod -
            (Ap - Bp) * lambda_)
    sign_ch = 1.0 if Vs1 > Vd else -1.0
    I_d1_d = sign_ch * Ich + (Vd1 - Vd + 0.1) * inv_Rleak
    dId_dVd1 = sign_ch * dIch + inv_Rleak

    I_s1_d1 = (Vs1 - Vd1) / 0.1
    r0a = I_s_s1 - I_s1_d1
    r0b = I_s1_d1 - I_d1_d
    j00 = dIss_dVs1 - 10.0
    j01 = 10.0
    j10 = 10.0
    j11 = -10.0 - dId_dVd1
    return r0a, r0b, j00, j01, j10, j11


def _newton_internal_impl(Vs, Vd, Vg, x0s, x0d, tol, maxit, Vfb, Vss, Lc, lambda_,
                          contact_scale, exponent, current_scale, inv_Rleak):
    Vs1 = x0s
    Vd1 = x0d
    hj = 1e-6
    for _ in range(maxit):
        r0a, r0b, j00, j01, j10, j11 = _residual_pair_fd_jac_impl(
            Vs, Vd, Vg, Vs1, Vd1, hj, Vfb, Vss, Lc, lambda_,
            contact_scale, exponent, current_scale, inv_Rleak)
        if abs(r0a) + abs(r0b) < tol:
            return True, Vs1, Vd1
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


def _newton_internal_fast_impl(Vs, Vd, Vg, x0s, x0d, tol, maxit, Vfb, Vss, Lc,
                               lambda_, contact_scale, exponent, current_scale,
                               inv_Rleak):
    Vs1 = x0s
    Vd1 = x0d
    fd_fallbacks = 0
    for it in range(maxit):
        if (abs(Vs - Vs1) < 1e-10 or abs(Vd1 - Vd) < 1e-10 or
                abs(Vs1 - Vd) < 1e-10):
            fd_fallbacks += 1
        r0a, r0b, j00, j01, j10, j11 = _residual_pair_jac_internal_impl(
            Vs, Vd, Vg, Vs1, Vd1, Vfb, Vss, Lc, lambda_, contact_scale,
            exponent, current_scale, inv_Rleak)
        if abs(r0a) + abs(r0b) < tol:
            return True, Vs1, Vd1, it + 1, fd_fallbacks
        det = j00 * j11 - j01 * j10
        if det == 0.0 or not math.isfinite(det):
            return False, Vs1, Vd1, it + 1, fd_fallbacks
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
                return True, Vs1, Vd1, it + 1, fd_fallbacks
            return False, Vs1, Vd1, it + 1, fd_fallbacks
    return False, Vs1, Vd1, maxit, fd_fallbacks


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


def _atan_cap_integral_impl(y, scale, two_over_pi):
    ay = scale * y
    return y + two_over_pi * (y * math.atan(ay) -
                              0.5 * math.log1p(ay * ay) / scale)


def _capacitance_charges_impl(Vs, Vd, Vg, Vs1, Vd1, Vfb, two_over_pi,
                              cap_cgs1, cap_cgd1, cap_half_wl_ci,
                              cap_cgs3_base, cap_cgd3_base, k1):
    v_s = Vs if Vs > Vs1 else Vs1
    v_d = Vd if Vd1 > Vd else Vd1

    y_s = v_s - Vg + Vfb
    y_d = v_d - Vg + Vfb
    x_gs = Vg - Vs
    x_gd = Vg - Vd

    cgs2_coeff = 1.43 * cap_half_wl_ci
    cgd2_coeff = 0.33 * cap_half_wl_ci
    cgs3_coeff = 0.34 * cap_cgs3_base
    cgd3_coeff = 0.52 * cap_cgd3_base

    f_s_060 = two_over_pi * math.atan(y_s * 0.6) + 1.0
    f_s_201 = two_over_pi * math.atan(y_s * 2.01) + 1.0
    f_d_021 = two_over_pi * math.atan(y_d * 0.21) + 1.0
    f_d_042 = two_over_pi * math.atan(y_d * 0.42) + 1.0

    cgs_cross = cgs3_coeff * f_d_021
    cgd_cross = cgd2_coeff * f_s_201
    qscale = k1 * 1e4 * 1e-12

    qgs = qscale * (
        cap_cgs1 * x_gs
        - cgs2_coeff * _atan_cap_integral_impl(y_s, 0.6, two_over_pi)
        + cgs_cross * x_gs
    )
    qgd = qscale * (
        cap_cgd1 * x_gd
        + cgd_cross * x_gd
        - cgd3_coeff * _atan_cap_integral_impl(y_d, 0.42, two_over_pi)
    )
    Cgss = qscale * (cap_cgs1 + cgs2_coeff * f_s_060 + cgs_cross)
    Cgdd = qscale * (cap_cgd1 + cgd_cross + cgd3_coeff * f_d_042)
    return qgs, qgd, Cgss, Cgdd


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


def _terminal_derivatives_from_base_impl(
        Vs, Vd, Vg, Vs1, Vd1, F0a, F0b, Idc0, need_gm, need_gds, use_abs,
        HH, hx, Vfb, Vss, Lc, lambda_, contact_scale, exponent,
        current_scale, inv_Rleak):
    if not need_gm and not need_gds:
        return True, 0.0, 0.0
    if use_abs and abs(Idc0) < 1e-30:
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
    current_sign = sign if use_abs else -1.0
    gm = 0.0
    gds = 0.0
    if need_gm:
        gm = _terminal_deriv_one_impl(
            Vs, Vd, Vg + HH, Vs, Vd, Vg - HH, Vs1, Vd1, Idc0,
            j00, j01, j10, j11, ix0, ix1, det, current_sign, HH, Vfb, Vss, Lc,
            lambda_, contact_scale, exponent, current_scale, inv_Rleak)
    if need_gds:
        gds = _terminal_deriv_one_impl(
            Vs, Vd + HH, Vg, Vs, Vd - HH, Vg, Vs1, Vd1, Idc0,
            j00, j01, j10, j11, ix0, ix1, det, current_sign, HH, Vfb, Vss, Lc,
            lambda_, contact_scale, exponent, current_scale, inv_Rleak)
    return True, gm, gds


def _terminal_derivatives_from_jac_fdterm_impl(
        Vs, Vd, Vg, Vs1, Vd1, F0a, F0b, Idc0,
        j00, j01, j10, j11, need_gm, need_gds, use_abs,
        HH, Vfb, Vss, Lc, lambda_, contact_scale, exponent,
        current_scale, inv_Rleak):
    if not need_gm and not need_gds:
        return True, 0.0, 0.0
    if use_abs and abs(Idc0) < 1e-30:
        return False, 0.0, 0.0
    det = j00 * j11 - j01 * j10
    if det == 0.0 or not math.isfinite(det):
        return False, 0.0, 0.0
    ix0 = j10 - 10.0
    ix1 = j11 + 10.0
    sign = 1.0 if Idc0 > 0.0 else -1.0
    current_sign = sign if use_abs else -1.0
    gm = 0.0
    gds = 0.0
    if need_gm:
        gm = _terminal_deriv_one_impl(
            Vs, Vd, Vg + HH, Vs, Vd, Vg - HH, Vs1, Vd1, Idc0,
            j00, j01, j10, j11, ix0, ix1, det, current_sign, HH, Vfb, Vss, Lc,
            lambda_, contact_scale, exponent, current_scale, inv_Rleak)
    if need_gds:
        gds = _terminal_deriv_one_impl(
            Vs, Vd + HH, Vg, Vs, Vd - HH, Vg, Vs1, Vd1, Idc0,
            j00, j01, j10, j11, ix0, ix1, det, current_sign, HH, Vfb, Vss, Lc,
            lambda_, contact_scale, exponent, current_scale, inv_Rleak)
    return True, gm, gds


def _contact_diss_dvg_impl(Vs, Vg, Vs1, Vfb, Vss, Lc, contact_scale, exponent):
    if Vs > Vs1:
        v_s = Vs
        v_s1 = Vs1
        dv = v_s - Vg
        Vt = -(0.0045 * dv ** 2 + 0.7125 * dv + 0.9625)
        dVt = 0.009 * dv + 0.7125
        arg_a = (v_s - Vg + Vt) / Vss
        arg_b = (v_s1 - Vg + Vt) / Vss
        A = Vss * _softplus_py(arg_a)
        B = Vss * _softplus_py(arg_b)
        Ap = A ** exponent
        Bp = B ** exponent
        Aem1 = Ap / A if A != 0.0 else 0.0
        Bem1 = Bp / B if B != 0.0 else 0.0
        Ecsat = 17.0 / (abs(dv) + 0.1)
        lambdac = 1.0 / (Lc * Ecsat)
        cmod = 1.0 + lambdac * (v_s - v_s1)
        sign_dv = 1.0 if dv > 0.0 else -1.0
        dlambdac = -sign_dv / (17.0 * Lc)
        darg_num = -1.0 + dVt
        dA = _sigmoid_impl(arg_a) * darg_num
        dB = _sigmoid_impl(arg_b) * darg_num
        dcmod = dlambdac * (v_s - v_s1)
        return contact_scale * (
            (exponent * Aem1 * dA - exponent * Bem1 * dB) * cmod +
            (Ap - Bp) * dcmod)

    v_s = Vs1
    v_s1 = Vs
    dv = v_s - Vg
    Vt = -(0.0045 * dv ** 2 + 0.7125 * dv + 0.9625)
    dVt = 0.009 * dv + 0.7125
    arg_a = (v_s - Vg + Vt) / Vss
    arg_b = (v_s1 - Vg + Vt) / Vss
    A = Vss * _softplus_py(arg_a)
    B = Vss * _softplus_py(arg_b)
    Ap = A ** exponent
    Bp = B ** exponent
    Aem1 = Ap / A if A != 0.0 else 0.0
    Bem1 = Bp / B if B != 0.0 else 0.0
    Ecsat = 17.0 / (abs(dv) + 0.1)
    lambdac = 1.0 / (Lc * Ecsat)
    cmod = 1.0 + lambdac * (v_s - v_s1)
    sign_dv = 1.0 if dv > 0.0 else -1.0
    dlambdac = -sign_dv / (17.0 * Lc)
    darg_num = -1.0 + dVt
    dA = _sigmoid_impl(arg_a) * darg_num
    dB = _sigmoid_impl(arg_b) * darg_num
    dcmod = dlambdac * (v_s - v_s1)
    dIcont = contact_scale * (
        (exponent * Aem1 * dA - exponent * Bem1 * dB) * cmod +
        (Ap - Bp) * dcmod)
    return -dIcont


def _channel_partials_impl(Vs1, Vd, Vg, Vd1, Vfb, Vss, lambda_, exponent,
                           current_scale, inv_Rleak):
    if Vd1 > Vd:
        v_d = Vd
        v_d1 = Vd1
        arg_a = (v_d1 - Vg + Vfb) / Vss
        arg_b = (v_d - Vg + Vfb) / Vss
        A = Vss * _softplus_py(arg_a)
        B = Vss * _softplus_py(arg_b)
        Ap = A ** exponent
        Bp = B ** exponent
        Aem1 = Ap / A if A != 0.0 else 0.0
        Bem1 = Bp / B if B != 0.0 else 0.0
        chmod = 1.0 + lambda_ * (v_d1 - v_d)
        dIch_dVg = current_scale * exponent * (
            -Aem1 * _sigmoid_impl(arg_a) +
            Bem1 * _sigmoid_impl(arg_b)) * chmod
        dIch_dVd = current_scale * (
            -exponent * Bem1 * _sigmoid_impl(arg_b) * chmod -
            (Ap - Bp) * lambda_)
    else:
        v_d = Vd1
        v_d1 = Vd
        arg_a = (v_d1 - Vg + Vfb) / Vss
        arg_b = (v_d - Vg + Vfb) / Vss
        A = Vss * _softplus_py(arg_a)
        B = Vss * _softplus_py(arg_b)
        Ap = A ** exponent
        Bp = B ** exponent
        Aem1 = Ap / A if A != 0.0 else 0.0
        Bem1 = Bp / B if B != 0.0 else 0.0
        chmod = 1.0 + lambda_ * (v_d1 - v_d)
        dIch_dVg = current_scale * exponent * (
            -Aem1 * _sigmoid_impl(arg_a) +
            Bem1 * _sigmoid_impl(arg_b)) * chmod
        dIch_dVd = current_scale * (
            exponent * Aem1 * _sigmoid_impl(arg_a) * chmod +
            (Ap - Bp) * lambda_)

    sign_ch = 1.0 if Vs1 > Vd else -1.0
    dId_dVg = sign_ch * dIch_dVg
    dId_dVd = sign_ch * dIch_dVd - inv_Rleak
    return dId_dVg, dId_dVd


def _terminal_deriv_from_partials_impl(fu0, fu1, iu, j00, j01, j10, j11,
                                       ix0, ix1, det, current_sign):
    y0 = (j11 * fu0 - j01 * fu1) / det
    y1 = (-j10 * fu0 + j00 * fu1) / det
    return current_sign * (iu - ix0 * y0 - ix1 * y1)


def _terminal_derivatives_from_jac_impl(
        Vs, Vd, Vg, Vs1, Vd1, F0a, F0b, Idc0,
        j00, j01, j10, j11, need_gm, need_gds, use_abs,
        HH, Vfb, Vss, Lc, lambda_, contact_scale, exponent,
        current_scale, inv_Rleak):
    if not need_gm and not need_gds:
        return True, 0.0, 0.0
    if (abs(Vs - Vs1) < 1e-10 or abs(Vd1 - Vd) < 1e-10 or
            abs(Vs1 - Vd) < 1e-10):
        return _terminal_derivatives_from_jac_fdterm_impl(
            Vs, Vd, Vg, Vs1, Vd1, F0a, F0b, Idc0, j00, j01, j10, j11,
            need_gm, need_gds, use_abs, HH, Vfb, Vss, Lc, lambda_,
            contact_scale, exponent, current_scale, inv_Rleak)
    if use_abs and abs(Idc0) < 1e-30:
        return False, 0.0, 0.0
    det = j00 * j11 - j01 * j10
    if det == 0.0 or not math.isfinite(det):
        return False, 0.0, 0.0

    ix0 = j10 - 10.0
    ix1 = j11 + 10.0
    sign = 1.0 if Idc0 > 0.0 else -1.0
    current_sign = sign if use_abs else -1.0
    dId_dVg, dId_dVd = _channel_partials_impl(
        Vs1, Vd, Vg, Vd1, Vfb, Vss, lambda_, exponent, current_scale,
        inv_Rleak)

    gm = 0.0
    if need_gm:
        dIss_dVg = _contact_diss_dvg_impl(
            Vs, Vg, Vs1, Vfb, Vss, Lc, contact_scale, exponent)
        gm = _terminal_deriv_from_partials_impl(
            dIss_dVg, -dId_dVg, -dId_dVg,
            j00, j01, j10, j11, ix0, ix1, det, current_sign)

    gds = 0.0
    if need_gds:
        gds = _terminal_deriv_from_partials_impl(
            0.0, -dId_dVd, -dId_dVd,
            j00, j01, j10, j11, ix0, ix1, det, current_sign)
    return True, gm, gds


def _terminal_derivatives_impl(Vs, Vd, Vg, Vs1, Vd1, need_gm, need_gds, use_abs,
                               HH, hx,
                               Vfb, Vss, Lc, lambda_, contact_scale, exponent,
                               current_scale, inv_Rleak):
    F0a, F0b, Idc0 = _eval_at_impl(
        Vs, Vd, Vg, Vs1, Vd1, Vfb, Vss, Lc, lambda_, contact_scale, exponent,
        current_scale, inv_Rleak)
    return _terminal_derivatives_from_base_impl(
        Vs, Vd, Vg, Vs1, Vd1, F0a, F0b, Idc0, need_gm, need_gds, use_abs,
        HH, hx, Vfb, Vss, Lc, lambda_, contact_scale, exponent,
        current_scale, inv_Rleak)


def _term_value_impl(kind, ref, value, V, input_values):
    if kind == 0:      # solved
        return V[ref]
    if kind == 1:      # transient input
        return input_values[ref]
    return value       # rail / constant


def _fill_prev_terms_impl(
        Vp, input_prev,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1,
        op_cache_valid, op_cache_vs1, op_cache_vd1,
        cap_a_kind, cap_a_ref, cap_a_val,
        cap_b_kind, cap_b_ref, cap_b_val,
        cap_mode,
        prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd, cap_prev_dv):
    for pos in range(prev_vs.shape[0]):
        pVs = _term_value_impl(dev_s_kind[pos], dev_s_ref[pos],
                               dev_s_val[pos], Vp, input_prev)
        pVd = _term_value_impl(dev_d_kind[pos], dev_d_ref[pos],
                               dev_d_val[pos], Vp, input_prev)
        pVg = _term_value_impl(dev_g_kind[pos], dev_g_ref[pos],
                               dev_g_val[pos], Vp, input_prev)
        prev_vs[pos] = pVs
        prev_vd[pos] = pVd
        prev_vg[pos] = pVg
        ok, pVs1, pVd1, _, _, _ = _solve_internal_with_guesses_impl(
            pVs, pVd, pVg, op_cache_valid[pos], op_cache_vs1[pos],
            op_cache_vd1[pos], 1e-12, 40, p_Vfb[pos], p_Vss[pos], p_Lc[pos],
            p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
            p_current_scale[pos], p_inv_Rleak[pos])
        if not ok:
            return False
        op_cache_valid[pos] = True
        op_cache_vs1[pos] = pVs1
        op_cache_vd1[pos] = pVd1
        qgs, qgd, Cgs, Cgd = _capacitance_charges_impl(
            pVs, pVd, pVg, pVs1, pVd1, p_Vfb[pos], p_two_over_pi[pos],
            p_cap_cgs1[pos], p_cap_cgd1[pos], p_cap_half_wl_ci[pos],
            p_cap_cgs3_base[pos], p_cap_cgd3_base[pos], p_k1[pos])
        if cap_mode == 1:
            qgs = Cgs
            qgd = Cgd
        prev_cgs[pos] = qgs
        prev_cgd[pos] = qgd
    for pos in range(cap_prev_dv.shape[0]):
        pva = _term_value_impl(cap_a_kind[pos], cap_a_ref[pos],
                               cap_a_val[pos], Vp, input_prev)
        pvb = _term_value_impl(cap_b_kind[pos], cap_b_ref[pos],
                               cap_b_val[pos], Vp, input_prev)
        cap_prev_dv[pos] = pva - pvb
    return True


def _solve_internal_with_guesses_impl(Vs, Vd, Vg, cache_valid, cache_vs1,
                                      cache_vd1, tol, maxit, Vfb, Vss, Lc,
                                      lambda_, contact_scale, exponent,
                                      current_scale, inv_Rleak):
    attempts = 0
    inner_iters = 0
    fd_fallbacks = 0
    if cache_valid:
        attempts += 1
        ok, xs, xd, iters, nfd = _newton_internal_fast_impl(
            Vs, Vd, Vg, cache_vs1, cache_vd1, tol, maxit,
            Vfb, Vss, Lc, lambda_, contact_scale, exponent,
            current_scale, inv_Rleak)
        inner_iters += iters
        fd_fallbacks += nfd
        if ok:
            return True, xs, xd, attempts, inner_iters, fd_fallbacks

    # Same deterministic guesses used before the Python fsolve fallback. The
    # Numba path exits if these fail, letting transient_solver fall back to the
    # original robust Python path.
    span = Vs - Vd
    guesses = (
        (Vs - 0.01 * span, Vd + 0.01 * span),
        (Vs, Vd),
        (0.5 * (Vs + Vd), 0.5 * (Vs + Vd)),
        (Vs, Vs),
        (Vd, Vd),
    )
    for xs0, xd0 in guesses:
        attempts += 1
        ok, xs, xd, iters, nfd = _newton_internal_fast_impl(
            Vs, Vd, Vg, xs0, xd0, tol, maxit,
            Vfb, Vss, Lc, lambda_, contact_scale, exponent,
            current_scale, inv_Rleak)
        inner_iters += iters
        fd_fallbacks += nfd
        if ok:
            return True, xs, xd, attempts, inner_iters, fd_fallbacks
    return False, cache_vs1, cache_vd1, attempts, inner_iters, fd_fallbacks


def _solve_dense_neg_rhs_inplace_impl(A, b):
    """Solve A*x = -b in place.

    A is overwritten by its LU-like elimination state and b is overwritten by x.
    The transient Newton path does not need either object after the solve, so
    this avoids two array copies and one explicit rhs allocation per iteration.
    """
    n = A.shape[0]
    for i in range(n):
        b[i] = -b[i]

    for k in range(n):
        piv = k
        piv_abs = abs(A[k, k])
        for r in range(k + 1, n):
            val = abs(A[r, k])
            if val > piv_abs:
                piv = r
                piv_abs = val
        if piv_abs == 0.0 or not math.isfinite(piv_abs):
            return False, b
        if piv != k:
            for c in range(k, n):
                tmp = A[k, c]
                A[k, c] = A[piv, c]
                A[piv, c] = tmp
            tmp = b[k]
            b[k] = b[piv]
            b[piv] = tmp
        diag = A[k, k]
        for r in range(k + 1, n):
            factor = A[r, k] / diag
            if factor != 0.0:
                A[r, k] = 0.0
                for c in range(k + 1, n):
                    A[r, c] -= factor * A[k, c]
                b[r] -= factor * b[k]

    for i in range(n - 1, -1, -1):
        acc = b[i]
        for c in range(i + 1, n):
            acc -= A[i, c] * b[c]
        diag = A[i, i]
        if diag == 0.0 or not math.isfinite(diag):
            return False, b
        b[i] = acc / diag
        if not math.isfinite(b[i]):
            return False, b
    return True, b


def _stamp_transient_system_impl(
        V, Vp, input_now, input_prev, h, n, gmin, HH,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        dev_di, dev_gi, dev_si, dev_use_abs,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
        op_cache_valid, op_cache_vs1, op_cache_vd1,
        res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref,
        res_b_val, res_ai, res_bi, res_g,
        cap_a_kind, cap_a_ref, cap_a_val, cap_b_kind, cap_b_ref,
        cap_b_val, cap_ai, cap_bi, cap_value,
        isrc_pi, isrc_qi, isrc_value,
        dyn_pi, dyn_qi, dyn_input_idx,
        cap_mode,
        prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd, cap_prev_dv,
        R, J, profile_enabled, profile_stats,
        bdf_a0, bdf_a1, bdf_a2, prev2_cgs, prev2_cgd, cap_prev2_dv):
    # cap history weights: backward-Euler = (1, -1, 0); variable-step BDF2/gear2
    # passes (a0, a1, a2) with prev2_* the n-2 charges/dv. Only the charge-mode
    # (cap_mode 0) device caps and the linear/load caps use the BDF2 form.
    for i in range(n):
        R[i] = 0.0
        for j in range(n):
            J[i, j] = 0.0
    inv_h = 1.0 / h

    for pos in range(dev_di.shape[0]):
        Vs = _term_value_impl(dev_s_kind[pos], dev_s_ref[pos], dev_s_val[pos],
                              V, input_now)
        Vd = _term_value_impl(dev_d_kind[pos], dev_d_ref[pos], dev_d_val[pos],
                              V, input_now)
        Vg = _term_value_impl(dev_g_kind[pos], dev_g_ref[pos], dev_g_val[pos],
                              V, input_now)
        pVs = prev_vs[pos]
        pVd = prev_vd[pos]
        pVg = prev_vg[pos]

        ok, Vs1, Vd1, op_attempts, op_iters, op_fd = _solve_internal_with_guesses_impl(
            Vs, Vd, Vg, op_cache_valid[pos], op_cache_vs1[pos],
            op_cache_vd1[pos], 1e-12, 40, p_Vfb[pos], p_Vss[pos], p_Lc[pos],
            p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
            p_current_scale[pos], p_inv_Rleak[pos])
        if profile_enabled:
            profile_stats[1] += 1.0
            profile_stats[2] += op_attempts
            profile_stats[3] += op_iters
            profile_stats[4] += op_fd
        if not ok:
            return False
        op_cache_valid[pos] = True
        op_cache_vs1[pos] = Vs1
        op_cache_vd1[pos] = Vd1

        F0a, F0b, j00, j01, j10, j11 = _residual_pair_jac_internal_impl(
            Vs, Vd, Vg, Vs1, Vd1, p_Vfb[pos], p_Vss[pos], p_Lc[pos],
            p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
            p_current_scale[pos], p_inv_Rleak[pos])
        qgs, qgd, Cgs, Cgd = _capacitance_charges_impl(
            Vs, Vd, Vg, Vs1, Vd1, p_Vfb[pos], p_two_over_pi[pos],
            p_cap_cgs1[pos], p_cap_cgd1[pos], p_cap_half_wl_ci[pos],
            p_cap_cgs3_base[pos], p_cap_cgd3_base[pos], p_k1[pos])

        di = dev_di[pos]
        gi = dev_gi[pos]
        si = dev_si[pos]
        Idc0 = F0b - (Vs1 - Vd1) / 0.1
        I = abs(Idc0) if dev_use_abs[pos] else -Idc0
        if di >= 0:
            R[di] += I
        if si >= 0:
            R[si] -= I

        leak_g = p_gate_leak_g[pos]
        if leak_g != 0.0:
            i_sg = (Vs - Vg) * leak_g
            if si >= 0:
                R[si] -= i_sg
            if gi >= 0:
                R[gi] += i_sg
            i_dg = (Vd - Vg) * leak_g
            if di >= 0:
                R[di] -= i_dg
            if gi >= 0:
                R[gi] += i_dg

        if Cgs != 0.0:
            if cap_mode == 1:
                i_ab = 0.5 * (Cgs + prev_cgs[pos]) * ((Vg - Vs) - (pVg - pVs)) * inv_h
            else:  # charge (id 0) + any unrecognized mode -> L-stable Q-stamp
                i_ab = (bdf_a0 * qgs + bdf_a1 * prev_cgs[pos] +
                        bdf_a2 * prev2_cgs[pos]) * inv_h
            if gi >= 0:
                R[gi] -= i_ab
            if si >= 0:
                R[si] += i_ab
        if Cgd != 0.0:
            if cap_mode == 1:
                i_ab = 0.5 * (Cgd + prev_cgd[pos]) * ((Vg - Vd) - (pVg - pVd)) * inv_h
            else:  # charge (id 0) + any unrecognized mode -> L-stable Q-stamp
                i_ab = (bdf_a0 * qgd + bdf_a1 * prev_cgd[pos] +
                        bdf_a2 * prev2_cgd[pos]) * inv_h
            if gi >= 0:
                R[gi] -= i_ab
            if di >= 0:
                R[di] += i_ab

        need_gm = gi >= 0 or si >= 0
        need_gds = di >= 0 or si >= 0
        if (profile_enabled and (need_gm or need_gds) and
                (abs(Vs - Vs1) < 1e-10 or abs(Vd1 - Vd) < 1e-10 or
                 abs(Vs1 - Vd) < 1e-10)):
            profile_stats[5] += 1.0
        okd, gm, gds = _terminal_derivatives_from_jac_impl(
            Vs, Vd, Vg, Vs1, Vd1, F0a, F0b, Idc0, j00, j01, j10, j11,
            need_gm, need_gds, dev_use_abs[pos], HH, p_Vfb[pos],
            p_Vss[pos], p_Lc[pos], p_lambda[pos], p_contact_scale[pos],
            p_exponent[pos], p_current_scale[pos], p_inv_Rleak[pos])
        if not okd:
            return False
        dI_dVs = -(gm + gds)
        if di >= 0:
            J[di, di] += gds
            if gi >= 0:
                J[di, gi] += gm
            if si >= 0:
                J[di, si] += dI_dVs
        if si >= 0:
            if di >= 0:
                J[si, di] -= gds
            if gi >= 0:
                J[si, gi] -= gm
            J[si, si] -= dI_dVs

        if leak_g != 0.0:
            if si >= 0:
                J[si, si] -= leak_g
                if gi >= 0:
                    J[si, gi] += leak_g
            if di >= 0:
                J[di, di] -= leak_g
                if gi >= 0:
                    J[di, gi] += leak_g
            if gi >= 0:
                count = 0
                if si >= 0:
                    J[gi, si] += leak_g
                    count += 1
                if di >= 0:
                    J[gi, di] += leak_g
                    count += 1
                J[gi, gi] -= leak_g * count
        if Cgs != 0.0:
            gc = bdf_a0 * Cgs * inv_h
            if gi >= 0:
                J[gi, gi] -= gc
                if si >= 0:
                    J[gi, si] += gc
            if si >= 0:
                J[si, si] -= gc
                if gi >= 0:
                    J[si, gi] += gc
        if Cgd != 0.0:
            gc = bdf_a0 * Cgd * inv_h
            if gi >= 0:
                J[gi, gi] -= gc
                if di >= 0:
                    J[gi, di] += gc
            if di >= 0:
                J[di, di] -= gc
                if gi >= 0:
                    J[di, gi] += gc

    for k in range(n):
        R[k] -= V[k] * gmin
        J[k, k] -= gmin

    for pos in range(res_g.shape[0]):
        gval = res_g[pos]
        if gval != 0.0:
            va = _term_value_impl(res_a_kind[pos], res_a_ref[pos],
                                  res_a_val[pos], V, input_now)
            vb = _term_value_impl(res_b_kind[pos], res_b_ref[pos],
                                  res_b_val[pos], V, input_now)
            i_ab = (va - vb) * gval
            ai = res_ai[pos]
            bi = res_bi[pos]
            if ai >= 0:
                R[ai] -= i_ab
                J[ai, ai] -= gval
                if bi >= 0:
                    J[ai, bi] += gval
            if bi >= 0:
                R[bi] += i_ab
                J[bi, bi] -= gval
                if ai >= 0:
                    J[bi, ai] += gval

    for pos in range(isrc_value.shape[0]):
        pi = isrc_pi[pos]
        qi = isrc_qi[pos]
        val = isrc_value[pos]
        if pi >= 0:
            R[pi] -= val
        if qi >= 0:
            R[qi] += val

    for pos in range(dyn_input_idx.shape[0]):
        pi = dyn_pi[pos]
        qi = dyn_qi[pos]
        val = input_now[dyn_input_idx[pos]]
        if pi >= 0:
            R[pi] -= val
        if qi >= 0:
            R[qi] += val

    for pos in range(cap_value.shape[0]):
        cap = cap_value[pos]
        if cap != 0.0:
            va = _term_value_impl(cap_a_kind[pos], cap_a_ref[pos],
                                  cap_a_val[pos], V, input_now)
            vb = _term_value_impl(cap_b_kind[pos], cap_b_ref[pos],
                                  cap_b_val[pos], V, input_now)
            i_ab = cap * inv_h * (bdf_a0 * (va - vb) + bdf_a1 * cap_prev_dv[pos] +
                                  bdf_a2 * cap_prev2_dv[pos])
            gc = bdf_a0 * cap * inv_h
            ai = cap_ai[pos]
            bi = cap_bi[pos]
            if ai >= 0:
                R[ai] -= i_ab
                J[ai, ai] -= gc
                if bi >= 0:
                    J[ai, bi] += gc
            if bi >= 0:
                R[bi] += i_ab
                J[bi, bi] -= gc
                if ai >= 0:
                    J[bi, ai] += gc

    return True


def _transient_newton_impl(
        seed, Vp, input_now, input_prev, h, n, maxit, step_limit, vtol,
        gmin, fallback_accept, fallback_tol, HH,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        dev_di, dev_gi, dev_si, dev_use_abs,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
        op_cache_valid, op_cache_vs1, op_cache_vd1,
        res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref,
        res_b_val, res_ai, res_bi, res_g,
        cap_a_kind, cap_a_ref, cap_a_val, cap_b_kind, cap_b_ref,
        cap_b_val, cap_ai, cap_bi, cap_value,
        isrc_pi, isrc_qi, isrc_value,
        dyn_pi, dyn_qi, dyn_input_idx,
        cap_mode,
        clip_lo, clip_hi):
    V = seed.copy()
    R = np.empty(n)
    J = np.empty((n, n))
    profile_stats = np.zeros(24)
    prev_vs = np.empty(dev_di.shape[0])
    prev_vd = np.empty(dev_di.shape[0])
    prev_vg = np.empty(dev_di.shape[0])
    prev_cgs = np.empty(dev_di.shape[0])
    prev_cgd = np.empty(dev_di.shape[0])
    cap_prev_dv = np.empty(cap_value.shape[0])
    ok_prev = _fill_prev_terms_impl(
        Vp, input_prev,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1,
        op_cache_valid, op_cache_vs1, op_cache_vd1,
        cap_a_kind, cap_a_ref, cap_a_val,
        cap_b_kind, cap_b_ref, cap_b_val,
        cap_mode,
        prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd, cap_prev_dv)
    if not ok_prev:
        return V, 0, False, False
    prev = math.inf
    for it in range(maxit):
        ok = _stamp_transient_system_impl(
            V, Vp, input_now, input_prev, h, n, gmin, HH,
            dev_d_kind, dev_d_ref, dev_d_val,
            dev_g_kind, dev_g_ref, dev_g_val,
            dev_s_kind, dev_s_ref, dev_s_val,
            dev_di, dev_gi, dev_si, dev_use_abs,
            p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
            p_current_scale, p_inv_Rleak,
            p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
            p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
            op_cache_valid, op_cache_vs1, op_cache_vd1,
            res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref,
            res_b_val, res_ai, res_bi, res_g,
            cap_a_kind, cap_a_ref, cap_a_val, cap_b_kind, cap_b_ref,
            cap_b_val, cap_ai, cap_bi, cap_value,
            isrc_pi, isrc_qi, isrc_value,
            dyn_pi, dyn_qi, dyn_input_idx,
            cap_mode,
            prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd, cap_prev_dv,
            R, J, False, profile_stats,
            1.0, -1.0, 0.0, prev_cgs, prev_cgd, cap_prev_dv)
        if not ok:
            return V, it + 1, False, False

        if fallback_accept:
            rmax = 0.0
            for i in range(n):
                val = abs(R[i])
                if val > rmax:
                    rmax = val
            if rmax < fallback_tol:
                return V, it + 1, True, True

        solved, dV = _solve_dense_neg_rhs_inplace_impl(J, R)
        if not solved:
            return V, it + 1, False, True
        mx = 0.0
        for i in range(n):
            val = abs(dV[i])
            if val > mx:
                mx = val
        if mx > step_limit:
            scale = step_limit / mx
            for i in range(n):
                dV[i] *= scale
            mx = step_limit
        for i in range(n):
            V[i] += dV[i]
            if clip_lo <= clip_hi:
                if V[i] < clip_lo:
                    V[i] = clip_lo
                elif V[i] > clip_hi:
                    V[i] = clip_hi
        if mx < vtol:
            if fallback_accept:
                prev = mx
                continue
            return V, it + 1, True, True
        if it >= 4 and mx >= prev and mx < 1e-5:
            if fallback_accept:
                prev = mx
                continue
            return V, it + 1, True, True
        prev = mx
    return V, maxit, False, True


def _transient_newton_reuse_impl(
        seed, Vp, input_now, input_prev, h, n, maxit, step_limit, vtol,
        gmin, fallback_accept, fallback_tol, HH,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        dev_di, dev_gi, dev_si, dev_use_abs,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
        op_cache_valid, op_cache_vs1, op_cache_vd1,
        res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref,
        res_b_val, res_ai, res_bi, res_g,
        cap_a_kind, cap_a_ref, cap_a_val, cap_b_kind, cap_b_ref,
        cap_b_val, cap_ai, cap_bi, cap_value,
        isrc_pi, isrc_qi, isrc_value,
        dyn_pi, dyn_qi, dyn_input_idx,
        cap_mode,
        clip_lo, clip_hi,
        V, R, J, prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd, cap_prev_dv,
        profile_enabled, profile_stats,
        bdf_a0, bdf_a1, bdf_a2, prev2_cgs, prev2_cgd, cap_prev2_dv):
    for i in range(n):
        V[i] = seed[i]
    ok_prev = _fill_prev_terms_impl(
        Vp, input_prev,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1,
        op_cache_valid, op_cache_vs1, op_cache_vd1,
        cap_a_kind, cap_a_ref, cap_a_val,
        cap_b_kind, cap_b_ref, cap_b_val,
        cap_mode,
        prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd, cap_prev_dv)
    if not ok_prev:
        if profile_enabled:
            profile_stats[20] += 1.0
        return 0, False, False
    prev = math.inf
    last_rmax = math.inf
    last_mx = math.inf
    for it in range(maxit):
        ok = _stamp_transient_system_impl(
            V, Vp, input_now, input_prev, h, n, gmin, HH,
            dev_d_kind, dev_d_ref, dev_d_val,
            dev_g_kind, dev_g_ref, dev_g_val,
            dev_s_kind, dev_s_ref, dev_s_val,
            dev_di, dev_gi, dev_si, dev_use_abs,
            p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
            p_current_scale, p_inv_Rleak,
            p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
            p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
            op_cache_valid, op_cache_vs1, op_cache_vd1,
            res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref,
            res_b_val, res_ai, res_bi, res_g,
            cap_a_kind, cap_a_ref, cap_a_val, cap_b_kind, cap_b_ref,
            cap_b_val, cap_ai, cap_bi, cap_value,
            isrc_pi, isrc_qi, isrc_value,
            dyn_pi, dyn_qi, dyn_input_idx,
            cap_mode,
            prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd, cap_prev_dv,
            R, J, profile_enabled, profile_stats,
            bdf_a0, bdf_a1, bdf_a2, prev2_cgs, prev2_cgd, cap_prev2_dv)
        if not ok:
            if profile_enabled:
                profile_stats[20] += 1.0
                profile_stats[16] = last_rmax
                if last_rmax > profile_stats[17]:
                    profile_stats[17] = last_rmax
                profile_stats[18] = last_mx
                if last_mx > profile_stats[19]:
                    profile_stats[19] = last_mx
            return it + 1, False, False

        if profile_enabled or fallback_accept:
            last_rmax = 0.0
            for i in range(n):
                val = abs(R[i])
                if val > last_rmax:
                    last_rmax = val
        if fallback_accept:
            rmax = last_rmax
            if rmax < fallback_tol:
                return it + 1, True, True

        solved, dV = _solve_dense_neg_rhs_inplace_impl(J, R)
        if not solved:
            if profile_enabled:
                profile_stats[21] += 1.0
                profile_stats[16] = last_rmax
                if last_rmax > profile_stats[17]:
                    profile_stats[17] = last_rmax
                profile_stats[18] = last_mx
                if last_mx > profile_stats[19]:
                    profile_stats[19] = last_mx
            return it + 1, False, True
        mx = 0.0
        for i in range(n):
            val = abs(dV[i])
            if val > mx:
                mx = val
        last_mx = mx
        if mx > step_limit:
            scale = step_limit / mx
            for i in range(n):
                dV[i] *= scale
            mx = step_limit
        for i in range(n):
            V[i] += dV[i]
            if clip_lo <= clip_hi:
                if V[i] < clip_lo:
                    V[i] = clip_lo
                elif V[i] > clip_hi:
                    V[i] = clip_hi
        if mx < vtol:
            if fallback_accept:
                relaxed_tol = fallback_tol
                if relaxed_tol < 1e-6:
                    relaxed_tol = 1e-6
                if last_rmax < relaxed_tol:
                    if profile_enabled:
                        profile_stats[23] += 1.0
                    return it + 1, True, True
                prev = mx
                continue
            return it + 1, True, True
        if it >= 4 and mx >= prev and mx < 1e-5:
            if fallback_accept:
                relaxed_tol = fallback_tol
                if relaxed_tol < 1e-6:
                    relaxed_tol = 1e-6
                if last_rmax < relaxed_tol:
                    if profile_enabled:
                        profile_stats[23] += 1.0
                    return it + 1, True, True
                prev = mx
                continue
            return it + 1, True, True
        prev = mx

    if profile_enabled:
        profile_stats[22] += 1.0
        profile_stats[16] = last_rmax
        if last_rmax > profile_stats[17]:
            profile_stats[17] = last_rmax
        profile_stats[18] = last_mx
        if last_mx > profile_stats[19]:
            profile_stats[19] = last_mx
    return maxit, False, True


def _transient_solve_grid_impl(
        V0, tgrid, input_values, edge_mask, profile_enabled,
        max_step, flat_max_step, max_retry_subdivisions,
        n, maxit, step_limit, vtol,
        gmin, fallback_accept, fallback_tol, HH,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        dev_di, dev_gi, dev_si, dev_use_abs,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
        op_cache_valid, op_cache_vs1, op_cache_vd1,
        res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref,
        res_b_val, res_ai, res_bi, res_g,
        cap_a_kind, cap_a_ref, cap_a_val, cap_b_kind, cap_b_ref,
        cap_b_val, cap_ai, cap_bi, cap_value,
        isrc_pi, isrc_qi, isrc_value,
        dyn_pi, dyn_qi, dyn_input_idx,
        cap_mode,
        clip_lo, clip_hi):
    N = tgrid.shape[0]
    ninputs = input_values.shape[0]
    Vhist = np.zeros((N, n))
    for i in range(n):
        Vhist[0, i] = V0[i]

    input_start = np.empty(ninputs)
    input_end = np.empty(ninputs)
    in0 = np.empty(ninputs)
    in1 = np.empty(ninputs)
    piece_in0 = np.empty(ninputs)
    piece_in1 = np.empty(ninputs)
    Vp = V0.copy()
    Vwork = np.empty(n)
    R = np.empty(n)
    J = np.empty((n, n))
    prev_vs = np.empty(dev_di.shape[0])
    prev_vd = np.empty(dev_di.shape[0])
    prev_vg = np.empty(dev_di.shape[0])
    prev_cgs = np.empty(dev_di.shape[0])
    prev_cgd = np.empty(dev_di.shape[0])
    cap_prev_dv = np.empty(cap_value.shape[0])
    profile_stats = np.zeros(24)
    failed_interval_indices = np.full(N, -1, dtype=np.int64)
    failed_interval_count = 0
    nsubsteps = 0

    for k in range(1, N):
        nsubsteps_before_interval = nsubsteps
        h = tgrid[k] - tgrid[k - 1]
        if h <= 0.0:
            return False, Vhist, nsubsteps, k, profile_stats, failed_interval_indices
        for ii in range(ninputs):
            input_start[ii] = input_values[ii, k - 1]
            input_end[ii] = input_values[ii, k]
            in0[ii] = input_start[ii]
        if max_step > 0.0:
            interval_edge = False
            if edge_mask.shape[0] == N:
                interval_edge = bool(edge_mask[k] or edge_mask[k - 1])
            local_max_step = max_step
            if flat_max_step > 0.0 and not interval_edge:
                local_max_step = flat_max_step
            pieces = int(math.ceil(h / local_max_step))
            if pieces < 1:
                pieces = 1
        else:
            pieces = 1
        hpiece = h / pieces
        interval_edge = False
        if edge_mask.shape[0] == N:
            interval_edge = bool(edge_mask[k] or edge_mask[k - 1])
        interval_failed = False
        for j in range(pieces):
            frac = (j + 1.0) / pieces
            for ii in range(ninputs):
                in1[ii] = input_start[ii] + (input_end[ii] - input_start[ii]) * frac
                piece_in0[ii] = in0[ii]
                piece_in1[ii] = in1[ii]
            iters, ok, usable = _transient_newton_reuse_impl(
                Vp, Vp, in1, in0, hpiece, n, maxit, step_limit, vtol,
                gmin, fallback_accept, fallback_tol, HH,
                dev_d_kind, dev_d_ref, dev_d_val,
                dev_g_kind, dev_g_ref, dev_g_val,
                dev_s_kind, dev_s_ref, dev_s_val,
                dev_di, dev_gi, dev_si, dev_use_abs,
                p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
                p_current_scale, p_inv_Rleak,
                p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
                p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
                op_cache_valid, op_cache_vs1, op_cache_vd1,
                res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref,
                res_b_val, res_ai, res_bi, res_g,
                cap_a_kind, cap_a_ref, cap_a_val, cap_b_kind, cap_b_ref,
                cap_b_val, cap_ai, cap_bi, cap_value,
                isrc_pi, isrc_qi, isrc_value,
                dyn_pi, dyn_qi, dyn_input_idx,
                cap_mode,
                clip_lo, clip_hi,
                Vwork, R, J, prev_vs, prev_vd, prev_vg,
                prev_cgs, prev_cgd, cap_prev_dv,
                profile_enabled, profile_stats,
                1.0, -1.0, 0.0, prev_cgs, prev_cgd, cap_prev_dv)
            if not ok:
                retry_count = 1
                for _retry_pow in range(max_retry_subdivisions):
                    retry_count *= 2
                if retry_count <= 1:
                    profile_stats[10] += 1.0
                    if fallback_accept:
                        if profile_enabled:
                            profile_stats[0] += iters
                        return False, Vhist, nsubsteps_before_interval, k, profile_stats, failed_interval_indices
                    if profile_enabled:
                        profile_stats[0] += iters
                    interval_failed = True
                    break
                if profile_enabled:
                    profile_stats[0] += iters
                retry_ok = True
                for rr in range(retry_count):
                    retry_frac = (rr + 1.0) / retry_count
                    for ii in range(ninputs):
                        in1[ii] = (
                            piece_in0[ii] +
                            (piece_in1[ii] - piece_in0[ii]) * retry_frac
                        )
                    iters_r, ok_r, usable_r = _transient_newton_reuse_impl(
                        Vp, Vp, in1, in0, hpiece / retry_count,
                        n, maxit, step_limit, vtol, gmin, fallback_accept,
                        fallback_tol, HH,
                        dev_d_kind, dev_d_ref, dev_d_val,
                        dev_g_kind, dev_g_ref, dev_g_val,
                        dev_s_kind, dev_s_ref, dev_s_val,
                        dev_di, dev_gi, dev_si, dev_use_abs,
                        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale,
                        p_exponent, p_current_scale, p_inv_Rleak,
                        p_two_over_pi, p_cap_cgs1, p_cap_cgd1,
                        p_cap_half_wl_ci, p_cap_cgs3_base, p_cap_cgd3_base,
                        p_k1, p_gate_leak_g,
                        op_cache_valid, op_cache_vs1, op_cache_vd1,
                        res_a_kind, res_a_ref, res_a_val,
                        res_b_kind, res_b_ref, res_b_val, res_ai, res_bi,
                        res_g,
                        cap_a_kind, cap_a_ref, cap_a_val,
                        cap_b_kind, cap_b_ref, cap_b_val, cap_ai, cap_bi,
                        cap_value,
                        isrc_pi, isrc_qi, isrc_value,
                        dyn_pi, dyn_qi, dyn_input_idx,
                        cap_mode,
                        clip_lo, clip_hi,
                        Vwork, R, J, prev_vs, prev_vd, prev_vg,
                        prev_cgs, prev_cgd, cap_prev_dv,
                        profile_enabled, profile_stats,
                        1.0, -1.0, 0.0, prev_cgs, prev_cgd, cap_prev_dv)
                    if profile_enabled:
                        profile_stats[0] += iters_r
                    if not ok_r:
                        retry_ok = False
                        profile_stats[10] += 1.0
                        break
                    nsubsteps += 1
                    if profile_enabled:
                        if interval_edge:
                            profile_stats[6] += 1.0
                            profile_stats[8] += iters_r
                        else:
                            profile_stats[7] += 1.0
                            profile_stats[9] += iters_r
                    for i in range(n):
                        Vp[i] = Vwork[i]
                    for ii in range(ninputs):
                        in0[ii] = in1[ii]
                if not retry_ok:
                    if fallback_accept:
                        return False, Vhist, nsubsteps_before_interval, k, profile_stats, failed_interval_indices
                    interval_failed = True
                    break
            else:
                nsubsteps += 1
                if profile_enabled:
                    profile_stats[0] += iters
                    if interval_edge:
                        profile_stats[6] += 1.0
                        profile_stats[8] += iters
                    else:
                        profile_stats[7] += 1.0
                        profile_stats[9] += iters
                for i in range(n):
                    Vp[i] = Vwork[i]
                for ii in range(ninputs):
                    in0[ii] = in1[ii]
        if interval_failed:
            # Match the Python fallback's non-throwing transient behavior for
            # non-robust mode: keep the last accepted state, count the failed
            # interval, and continue the trajectory.
            profile_stats[13] += 1.0
            if interval_edge:
                profile_stats[14] += 1.0
            else:
                profile_stats[15] += 1.0
            if profile_enabled and failed_interval_count < N:
                failed_interval_indices[failed_interval_count] = k
                failed_interval_count += 1
        for i in range(n):
            Vhist[k, i] = Vp[i]
    if profile_enabled:
        profile_stats[11] = N - 1
        profile_stats[12] = nsubsteps
    return True, Vhist, nsubsteps, -1, profile_stats, failed_interval_indices


def _interp_inputs_at_time_impl(tgrid, input_values, tt, out):
    ninputs = input_values.shape[0]
    N = tgrid.shape[0]
    if ninputs == 0:
        return
    if tt <= tgrid[0]:
        for ii in range(ninputs):
            out[ii] = input_values[ii, 0]
        return
    if tt >= tgrid[N - 1]:
        for ii in range(ninputs):
            out[ii] = input_values[ii, N - 1]
        return
    k = 0
    for j in range(N - 1):
        if tgrid[j] <= tt <= tgrid[j + 1]:
            k = j
            break
    h = tgrid[k + 1] - tgrid[k]
    frac = 0.0 if h <= 0.0 else (tt - tgrid[k]) / h
    for ii in range(ninputs):
        out[ii] = input_values[ii, k] + frac * (input_values[ii, k + 1] - input_values[ii, k])


def _adaptive_error_impl(Vhalf, Vfull, n_nodes, reltol, vabstol, iabstol):
    n = Vhalf.shape[0]
    acc = 0.0
    for i in range(n):
        abstol = vabstol if i < n_nodes else iabstol
        scale = reltol * max(abs(Vhalf[i]), abs(Vfull[i])) + abstol
        if scale < ADAPTIVE_SCALE_FLOOR:
            scale = ADAPTIVE_SCALE_FLOOR
        e = ((Vhalf[i] - Vfull[i]) / ADAPTIVE_LTE_DIVISOR) / scale
        acc += e * e
    return math.sqrt(acc / n)


def _adaptive_next_h_impl(h, err):
    if err <= 0.0:
        fac = ADAPTIVE_GROWTH_MAX
    elif not math.isfinite(err):
        fac = ADAPTIVE_GROWTH_MIN
    else:
        fac = ADAPTIVE_SAFETY * err ** (-1.0 / ADAPTIVE_STEP_ORDER)
        if fac < ADAPTIVE_GROWTH_MIN:
            fac = ADAPTIVE_GROWTH_MIN
        elif fac > ADAPTIVE_GROWTH_MAX:
            fac = ADAPTIVE_GROWTH_MAX
    return h * fac


def _gear2_substep_newton_reuse_impl(
        seed, Vp, Vp2, input_now, input_prev, input_prev2, h_n, h_prev,
        n, maxit, step_limit, vtol,
        gmin, fallback_accept, fallback_tol, HH,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        dev_di, dev_gi, dev_si, dev_use_abs,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
        op_cache_valid, op_cache_vs1, op_cache_vd1,
        res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref,
        res_b_val, res_ai, res_bi, res_g,
        cap_a_kind, cap_a_ref, cap_a_val, cap_b_kind, cap_b_ref,
        cap_b_val, cap_ai, cap_bi, cap_value,
        isrc_pi, isrc_qi, isrc_value,
        dyn_pi, dyn_qi, dyn_input_idx,
        cap_mode, clip_lo, clip_hi,
        Vwork, R, J, prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd,
        cap_prev_dv, p2_vs, p2_vd, p2_vg, prev2_cgs, prev2_cgd,
        cap_prev2_dv, op2_valid, op2_vs1, op2_vd1,
        profile_enabled, profile_stats):
    if h_prev <= 0.0 or h_n / h_prev > 2.0:
        a0 = 1.0
        a1 = -1.0
        a2 = 0.0
    else:
        rho = h_n / h_prev
        a0 = (1.0 + 2.0 * rho) / (1.0 + rho)
        a1 = -(1.0 + rho)
        a2 = (rho * rho) / (1.0 + rho)
    ok2 = _fill_prev_terms_impl(
        Vp2, input_prev2,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1,
        op2_valid, op2_vs1, op2_vd1,
        cap_a_kind, cap_a_ref, cap_a_val,
        cap_b_kind, cap_b_ref, cap_b_val,
        cap_mode,
        p2_vs, p2_vd, p2_vg, prev2_cgs, prev2_cgd, cap_prev2_dv)
    if not ok2:
        if profile_enabled:
            profile_stats[20] += 1.0
        return 0, False, False
    return _transient_newton_reuse_impl(
        seed, Vp, input_now, input_prev, h_n, n, maxit, step_limit, vtol,
        gmin, fallback_accept, fallback_tol, HH,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        dev_di, dev_gi, dev_si, dev_use_abs,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
        op_cache_valid, op_cache_vs1, op_cache_vd1,
        res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref,
        res_b_val, res_ai, res_bi, res_g,
        cap_a_kind, cap_a_ref, cap_a_val, cap_b_kind, cap_b_ref,
        cap_b_val, cap_ai, cap_bi, cap_value,
        isrc_pi, isrc_qi, isrc_value,
        dyn_pi, dyn_qi, dyn_input_idx,
        cap_mode, clip_lo, clip_hi,
        Vwork, R, J, prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd,
        cap_prev_dv, profile_enabled, profile_stats,
        a0, a1, a2, prev2_cgs, prev2_cgd, cap_prev2_dv)


def _transient_solve_adaptive_gear2_impl(
        V0, tgrid_src, input_values_src, profile_enabled,
        max_step, adaptive_reltol, adaptive_vabstol, adaptive_iabstol,
        adaptive_max_steps, adaptive_h0,
        n, maxit, step_limit, vtol,
        gmin, fallback_accept, fallback_tol, HH,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        dev_di, dev_gi, dev_si, dev_use_abs,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
        op_cache_valid, op_cache_vs1, op_cache_vd1,
        res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref,
        res_b_val, res_ai, res_bi, res_g,
        cap_a_kind, cap_a_ref, cap_a_val, cap_b_kind, cap_b_ref,
        cap_b_val, cap_ai, cap_bi, cap_value,
        isrc_pi, isrc_qi, isrc_value,
        dyn_pi, dyn_qi, dyn_input_idx,
        cap_mode, clip_lo, clip_hi):
    Nsrc = tgrid_src.shape[0]
    ninputs = input_values_src.shape[0]
    max_steps = int(adaptive_max_steps)
    if max_steps < 1:
        max_steps = 1
    thist = np.empty(max_steps + 1)
    Vhist = np.empty((max_steps + 1, n))
    input_hist = np.empty((max_steps + 1, ninputs))
    profile_stats = np.zeros(24)
    for i in range(n):
        Vhist[0, i] = V0[i]
    thist[0] = tgrid_src[0]

    ndev = dev_di.shape[0]
    ncap = cap_value.shape[0]
    Vp = V0.copy()
    Vp2 = V0.copy()
    Vfull = np.empty(n)
    Vmid = np.empty(n)
    Vhalf2 = np.empty(n)
    Vwork = np.empty(n)
    R = np.empty(n)
    J = np.empty((n, n))
    prev_vs = np.empty(ndev); prev_vd = np.empty(ndev); prev_vg = np.empty(ndev)
    prev_cgs = np.empty(ndev); prev_cgd = np.empty(ndev)
    cap_prev_dv = np.empty(ncap)
    p2_vs = np.empty(ndev); p2_vd = np.empty(ndev); p2_vg = np.empty(ndev)
    prev2_cgs = np.empty(ndev); prev2_cgd = np.empty(ndev)
    cap_prev2_dv = np.empty(ncap)
    op2_valid = np.zeros(ndev, dtype=np.bool_)
    op2_vs1 = np.empty(ndev); op2_vd1 = np.empty(ndev)
    input_prev = np.empty(ninputs)
    input_prev2 = np.empty(ninputs)
    input_now = np.empty(ninputs)
    input_mid = np.empty(ninputs)
    _interp_inputs_at_time_impl(tgrid_src, input_values_src, thist[0], input_prev)
    for ii in range(ninputs):
        input_hist[0, ii] = input_prev[ii]
        input_prev2[ii] = input_prev[ii]

    t0 = tgrid_src[0]
    tend = tgrid_src[Nsrc - 1]
    span = tend - t0
    if span <= 0.0:
        return False, thist, Vhist, input_hist, 1, 0, 0, profile_stats
    max_step_eff = span if max_step <= 0.0 else min(max_step, span)
    if adaptive_h0 > 0.0:
        h = adaptive_h0
    else:
        denom = Nsrc - 1
        if denom < ADAPTIVE_INITIAL_MIN_DENOM:
            denom = ADAPTIVE_INITIAL_MIN_DENOM
        h = span / denom
        mindt = span
        for k in range(1, Nsrc):
            dt = tgrid_src[k] - tgrid_src[k - 1]
            if dt <= 0.0:
                return False, thist, Vhist, input_hist, 1, 0, 0, profile_stats
            if dt < mindt:
                mindt = dt
        if h > mindt:
            h = mindt
    if h <= 0.0 or not math.isfinite(h):
        h = span / 100.0
    if h > max_step_eff:
        h = max_step_eff
    min_h = max(ADAPTIVE_MIN_H_ABS, span * ADAPTIVE_MIN_H_REL)
    done_tol = max(ADAPTIVE_DONE_ABS, span * ADAPTIVE_DONE_REL)
    critical_times = np.empty(Nsrc)
    ncritical = 0
    if ninputs > 0 and Nsrc >= 3:
        global_slope = 1.0
        valid_slopes = True
        for k in range(1, Nsrc):
            dt = tgrid_src[k] - tgrid_src[k - 1]
            if dt <= 0.0:
                valid_slopes = False
                break
            for ii in range(ninputs):
                slope = (input_values_src[ii, k] - input_values_src[ii, k - 1]) / dt
                mag = abs(slope)
                if mag > global_slope:
                    global_slope = mag
        if valid_slopes:
            for k in range(1, Nsrc - 1):
                dt0 = tgrid_src[k] - tgrid_src[k - 1]
                dt1 = tgrid_src[k + 1] - tgrid_src[k]
                jump = 0.0
                for ii in range(ninputs):
                    s0 = (input_values_src[ii, k] - input_values_src[ii, k - 1]) / dt0
                    s1 = (input_values_src[ii, k + 1] - input_values_src[ii, k]) / dt1
                    mag = abs(s1 - s0)
                    if mag > jump:
                        jump = mag
                if jump > ADAPTIVE_INPUT_SLOPE_BREAK_FRACTION * global_slope:
                    critical_times[ncritical] = tgrid_src[k]
                    ncritical += 1
    h_prev = -1.0
    accepted = 0
    nsubsteps = 0
    nreject = 0
    tt = t0

    while accepted < max_steps and tt < tend - done_tol:
        if h_prev > 0.0 and h > ADAPTIVE_GROWTH_MAX * h_prev:
            h = ADAPTIVE_GROWTH_MAX * h_prev
        if h > max_step_eff:
            h = max_step_eff
        if tt + h > tend:
            h = tend - tt
        if ncritical > 0:
            for kc in range(ncritical):
                tc = critical_times[kc]
                if tc > tt + min_h:
                    if tc < tt + h:
                        h = tc - tt
                    break
        if h <= min_h:
            return False, thist, Vhist, input_hist, accepted + 1, nsubsteps, nreject, profile_stats
        _interp_inputs_at_time_impl(tgrid_src, input_values_src, tt + h, input_now)
        _interp_inputs_at_time_impl(tgrid_src, input_values_src, tt + 0.5 * h, input_mid)

        it1, ok_full, _ = _gear2_substep_newton_reuse_impl(
            Vp, Vp, Vp2, input_now, input_prev, input_prev2, h, h_prev,
            n, maxit, step_limit, vtol, gmin, fallback_accept, fallback_tol, HH,
            dev_d_kind, dev_d_ref, dev_d_val, dev_g_kind, dev_g_ref, dev_g_val,
            dev_s_kind, dev_s_ref, dev_s_val, dev_di, dev_gi, dev_si, dev_use_abs,
            p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
            p_current_scale, p_inv_Rleak, p_two_over_pi, p_cap_cgs1, p_cap_cgd1,
            p_cap_half_wl_ci, p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
            op_cache_valid, op_cache_vs1, op_cache_vd1,
            res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref, res_b_val,
            res_ai, res_bi, res_g, cap_a_kind, cap_a_ref, cap_a_val,
            cap_b_kind, cap_b_ref, cap_b_val, cap_ai, cap_bi, cap_value,
            isrc_pi, isrc_qi, isrc_value, dyn_pi, dyn_qi, dyn_input_idx,
            cap_mode, clip_lo, clip_hi, Vwork, R, J, prev_vs, prev_vd, prev_vg,
            prev_cgs, prev_cgd, cap_prev_dv, p2_vs, p2_vd, p2_vg,
            prev2_cgs, prev2_cgd, cap_prev2_dv, op2_valid, op2_vs1, op2_vd1,
            profile_enabled, profile_stats)
        for i in range(n):
            Vfull[i] = Vwork[i]

        it2, ok_mid, _ = _gear2_substep_newton_reuse_impl(
            Vp, Vp, Vp2, input_mid, input_prev, input_prev2, 0.5 * h, h_prev,
            n, maxit, step_limit, vtol, gmin, fallback_accept, fallback_tol, HH,
            dev_d_kind, dev_d_ref, dev_d_val, dev_g_kind, dev_g_ref, dev_g_val,
            dev_s_kind, dev_s_ref, dev_s_val, dev_di, dev_gi, dev_si, dev_use_abs,
            p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
            p_current_scale, p_inv_Rleak, p_two_over_pi, p_cap_cgs1, p_cap_cgd1,
            p_cap_half_wl_ci, p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
            op_cache_valid, op_cache_vs1, op_cache_vd1,
            res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref, res_b_val,
            res_ai, res_bi, res_g, cap_a_kind, cap_a_ref, cap_a_val,
            cap_b_kind, cap_b_ref, cap_b_val, cap_ai, cap_bi, cap_value,
            isrc_pi, isrc_qi, isrc_value, dyn_pi, dyn_qi, dyn_input_idx,
            cap_mode, clip_lo, clip_hi, Vwork, R, J, prev_vs, prev_vd, prev_vg,
            prev_cgs, prev_cgd, cap_prev_dv, p2_vs, p2_vd, p2_vg,
            prev2_cgs, prev2_cgd, cap_prev2_dv, op2_valid, op2_vs1, op2_vd1,
            profile_enabled, profile_stats)
        for i in range(n):
            Vmid[i] = Vwork[i]

        ok_half2 = False
        it3 = 0
        if ok_mid:
            it3, ok_half2, _ = _gear2_substep_newton_reuse_impl(
                Vmid, Vmid, Vp, input_now, input_mid, input_prev, 0.5 * h, 0.5 * h,
                n, maxit, step_limit, vtol, gmin, fallback_accept, fallback_tol, HH,
                dev_d_kind, dev_d_ref, dev_d_val, dev_g_kind, dev_g_ref, dev_g_val,
                dev_s_kind, dev_s_ref, dev_s_val, dev_di, dev_gi, dev_si, dev_use_abs,
                p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
                p_current_scale, p_inv_Rleak, p_two_over_pi, p_cap_cgs1, p_cap_cgd1,
                p_cap_half_wl_ci, p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
                op_cache_valid, op_cache_vs1, op_cache_vd1,
                res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref, res_b_val,
                res_ai, res_bi, res_g, cap_a_kind, cap_a_ref, cap_a_val,
                cap_b_kind, cap_b_ref, cap_b_val, cap_ai, cap_bi, cap_value,
                isrc_pi, isrc_qi, isrc_value, dyn_pi, dyn_qi, dyn_input_idx,
                cap_mode, clip_lo, clip_hi, Vwork, R, J, prev_vs, prev_vd, prev_vg,
                prev_cgs, prev_cgd, cap_prev_dv, p2_vs, p2_vd, p2_vg,
                prev2_cgs, prev2_cgd, cap_prev2_dv, op2_valid, op2_vs1, op2_vd1,
                profile_enabled, profile_stats)
            for i in range(n):
                Vhalf2[i] = Vwork[i]

        nsubsteps += 3
        if profile_enabled:
            profile_stats[0] += it1 + it2 + it3
        if ok_full and ok_mid and ok_half2:
            err = _adaptive_error_impl(Vhalf2, Vfull, n, adaptive_reltol,
                                       adaptive_vabstol, adaptive_iabstol)
        else:
            err = math.inf
        if err <= ADAPTIVE_ACCEPT_WRMS:
            tt += h
            accepted += 1
            thist[accepted] = tt
            for i in range(n):
                Vhist[accepted, i] = Vhalf2[i]
                Vp2[i] = Vp[i]
                Vp[i] = Vhalf2[i]
            for ii in range(ninputs):
                input_hist[accepted, ii] = input_now[ii]
                input_prev2[ii] = input_prev[ii]
                input_prev[ii] = input_now[ii]
            hit_critical = False
            if ncritical > 0:
                crit_tol = min_h
                if done_tol > crit_tol:
                    crit_tol = done_tol
                for kc in range(ncritical):
                    if abs(critical_times[kc] - tt) <= crit_tol:
                        hit_critical = True
                        break
            if hit_critical:
                for i in range(n):
                    Vp2[i] = Vp[i]
                for ii in range(ninputs):
                    input_prev2[ii] = input_prev[ii]
                h_prev = -1.0
            else:
                h_prev = h
            h = _adaptive_next_h_impl(h, max(err, ADAPTIVE_ERR_FLOOR))
        else:
            nreject += 1
            h = _adaptive_next_h_impl(h, err)
            if h < min_h:
                h = min_h
    ok = tt >= tend - done_tol
    return ok, thist, Vhist, input_hist, accepted + 1, nsubsteps, nreject, profile_stats


def _transient_solve_grid_gear2_impl(
        V0, tgrid, input_values, edge_mask, profile_enabled,
        max_step, flat_max_step, max_retry_subdivisions,
        n, maxit, step_limit, vtol,
        gmin, fallback_accept, fallback_tol, HH,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        dev_di, dev_gi, dev_si, dev_use_abs,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
        op_cache_valid, op_cache_vs1, op_cache_vd1,
        res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref,
        res_b_val, res_ai, res_bi, res_g,
        cap_a_kind, cap_a_ref, cap_a_val, cap_b_kind, cap_b_ref,
        cap_b_val, cap_ai, cap_bi, cap_value,
        isrc_pi, isrc_qi, isrc_value,
        dyn_pi, dyn_qi, dyn_input_idx,
        cap_mode, clip_lo, clip_hi):
    """Variable-step BDF2/gear2 grid solver with maxstep slicing and retry.

    Each accepted internal substep updates the BDF2 history tuple
    (x[n-1], x[n-2], h[n-1]), so subdivided raw transients keep the same history
    semantics as the Python gear2 solve_chunk path.
    """
    N = tgrid.shape[0]
    ninputs = input_values.shape[0]
    ndev = dev_di.shape[0]
    ncap = cap_value.shape[0]
    Vhist = np.zeros((N, n))
    for i in range(n):
        Vhist[0, i] = V0[i]
    Vp = V0.copy()
    Vp2 = V0.copy()
    Vwork = np.empty(n)
    R = np.empty(n)
    J = np.empty((n, n))
    prev_vs = np.empty(ndev); prev_vd = np.empty(ndev); prev_vg = np.empty(ndev)
    prev_cgs = np.empty(ndev); prev_cgd = np.empty(ndev)
    cap_prev_dv = np.empty(ncap)
    p2_vs = np.empty(ndev); p2_vd = np.empty(ndev); p2_vg = np.empty(ndev)
    prev2_cgs = np.empty(ndev); prev2_cgd = np.empty(ndev)
    cap_prev2_dv = np.empty(ncap)
    # separate internal-node cache for the prev2 fill so it does not overwrite the
    # main op_cache that reuse_impl seeds from (a far Vhist[k-2] seed there makes
    # the reuse internal solve land on the wrong multistable branch).
    op2_valid = np.zeros(ndev, dtype=np.bool_)
    op2_vs1 = np.empty(ndev); op2_vd1 = np.empty(ndev)
    input_start = np.empty(ninputs)
    input_end = np.empty(ninputs)
    input_cur = np.empty(ninputs)
    input_cur2 = np.empty(ninputs)
    input_next = np.empty(ninputs)
    piece_in0 = np.empty(ninputs)
    piece_in1 = np.empty(ninputs)
    profile_stats = np.zeros(24)
    failed = np.full(N, -1, dtype=np.int64)
    failed_interval_count = 0
    nsubsteps = 0
    for k in range(1, N):
        nsubsteps_before_interval = nsubsteps
        h_n = tgrid[k] - tgrid[k - 1]
        if h_n <= 0.0:
            return False, Vhist, nsubsteps, k, profile_stats, failed
        for ii in range(ninputs):
            input_start[ii] = input_values[ii, k - 1]
            input_end[ii] = input_values[ii, k]
            input_cur[ii] = input_start[ii]
        for i in range(n):
            Vp[i] = Vhist[k - 1, i]
        if k >= 2:
            h_prev_cur = tgrid[k - 1] - tgrid[k - 2]
            for ii in range(ninputs):
                input_cur2[ii] = input_values[ii, k - 2]
            for i in range(n):
                Vp2[i] = Vhist[k - 2, i]
        else:
            h_prev_cur = 0.0
            for ii in range(ninputs):
                input_cur2[ii] = input_start[ii]
            for i in range(n):
                Vp2[i] = Vp[i]
        interval_edge = False
        if edge_mask.shape[0] == N:
            interval_edge = bool(edge_mask[k] or edge_mask[k - 1])
        if max_step > 0.0:
            local_max_step = max_step
            if flat_max_step > 0.0 and not interval_edge:
                local_max_step = flat_max_step
            pieces = int(math.ceil(h_n / local_max_step))
            if pieces < 1:
                pieces = 1
        else:
            pieces = 1
        hpiece = h_n / pieces
        interval_failed = False
        for j in range(pieces):
            frac = (j + 1.0) / pieces
            for ii in range(ninputs):
                piece_in0[ii] = input_cur[ii]
                piece_in1[ii] = input_start[ii] + (input_end[ii] - input_start[ii]) * frac
            iters, ok, usable = _gear2_substep_newton_reuse_impl(
                Vp, Vp, Vp2, piece_in1, input_cur, input_cur2,
                hpiece, h_prev_cur,
                n, maxit, step_limit, vtol, gmin, fallback_accept,
                fallback_tol, HH,
                dev_d_kind, dev_d_ref, dev_d_val,
                dev_g_kind, dev_g_ref, dev_g_val,
                dev_s_kind, dev_s_ref, dev_s_val,
                dev_di, dev_gi, dev_si, dev_use_abs,
                p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale,
                p_exponent, p_current_scale, p_inv_Rleak,
                p_two_over_pi, p_cap_cgs1, p_cap_cgd1,
                p_cap_half_wl_ci, p_cap_cgs3_base, p_cap_cgd3_base,
                p_k1, p_gate_leak_g,
                op_cache_valid, op_cache_vs1, op_cache_vd1,
                res_a_kind, res_a_ref, res_a_val,
                res_b_kind, res_b_ref, res_b_val, res_ai, res_bi,
                res_g,
                cap_a_kind, cap_a_ref, cap_a_val,
                cap_b_kind, cap_b_ref, cap_b_val, cap_ai, cap_bi,
                cap_value,
                isrc_pi, isrc_qi, isrc_value,
                dyn_pi, dyn_qi, dyn_input_idx,
                cap_mode, clip_lo, clip_hi,
                Vwork, R, J, prev_vs, prev_vd, prev_vg,
                prev_cgs, prev_cgd, cap_prev_dv,
                p2_vs, p2_vd, p2_vg, prev2_cgs, prev2_cgd,
                cap_prev2_dv, op2_valid, op2_vs1, op2_vd1,
                profile_enabled, profile_stats)
            if profile_enabled:
                profile_stats[0] += iters
            if ok:
                nsubsteps += 1
                if profile_enabled:
                    if interval_edge:
                        profile_stats[6] += 1.0
                        profile_stats[8] += iters
                    else:
                        profile_stats[7] += 1.0
                        profile_stats[9] += iters
                for i in range(n):
                    Vp2[i] = Vp[i]
                    Vp[i] = Vwork[i]
                for ii in range(ninputs):
                    input_cur2[ii] = input_cur[ii]
                    input_cur[ii] = piece_in1[ii]
                h_prev_cur = hpiece
                continue

            profile_stats[10] += 1.0
            retry_count = 1
            for _retry_pow in range(max_retry_subdivisions):
                retry_count *= 2
            if retry_count <= 1:
                if fallback_accept:
                    return False, Vhist, nsubsteps_before_interval, k, profile_stats, failed
                interval_failed = True
                break
            retry_ok = True
            hretry = hpiece / retry_count
            for rr in range(retry_count):
                retry_frac = (rr + 1.0) / retry_count
                for ii in range(ninputs):
                    input_next[ii] = (
                        piece_in0[ii] +
                        (piece_in1[ii] - piece_in0[ii]) * retry_frac
                    )
                iters_r, ok_r, usable_r = _gear2_substep_newton_reuse_impl(
                    Vp, Vp, Vp2, input_next, input_cur, input_cur2,
                    hretry, h_prev_cur,
                    n, maxit, step_limit, vtol, gmin, fallback_accept,
                    fallback_tol, HH,
                    dev_d_kind, dev_d_ref, dev_d_val,
                    dev_g_kind, dev_g_ref, dev_g_val,
                    dev_s_kind, dev_s_ref, dev_s_val,
                    dev_di, dev_gi, dev_si, dev_use_abs,
                    p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale,
                    p_exponent, p_current_scale, p_inv_Rleak,
                    p_two_over_pi, p_cap_cgs1, p_cap_cgd1,
                    p_cap_half_wl_ci, p_cap_cgs3_base, p_cap_cgd3_base,
                    p_k1, p_gate_leak_g,
                    op_cache_valid, op_cache_vs1, op_cache_vd1,
                    res_a_kind, res_a_ref, res_a_val,
                    res_b_kind, res_b_ref, res_b_val, res_ai, res_bi,
                    res_g,
                    cap_a_kind, cap_a_ref, cap_a_val,
                    cap_b_kind, cap_b_ref, cap_b_val, cap_ai, cap_bi,
                    cap_value,
                    isrc_pi, isrc_qi, isrc_value,
                    dyn_pi, dyn_qi, dyn_input_idx,
                    cap_mode, clip_lo, clip_hi,
                    Vwork, R, J, prev_vs, prev_vd, prev_vg,
                    prev_cgs, prev_cgd, cap_prev_dv,
                    p2_vs, p2_vd, p2_vg, prev2_cgs, prev2_cgd,
                    cap_prev2_dv, op2_valid, op2_vs1, op2_vd1,
                    profile_enabled, profile_stats)
                if profile_enabled:
                    profile_stats[0] += iters_r
                if not ok_r:
                    retry_ok = False
                    profile_stats[10] += 1.0
                    break
                nsubsteps += 1
                if profile_enabled:
                    if interval_edge:
                        profile_stats[6] += 1.0
                        profile_stats[8] += iters_r
                    else:
                        profile_stats[7] += 1.0
                        profile_stats[9] += iters_r
                for i in range(n):
                    Vp2[i] = Vp[i]
                    Vp[i] = Vwork[i]
                for ii in range(ninputs):
                    input_cur2[ii] = input_cur[ii]
                    input_cur[ii] = input_next[ii]
                h_prev_cur = hretry
            if not retry_ok:
                if fallback_accept:
                    return False, Vhist, nsubsteps_before_interval, k, profile_stats, failed
                interval_failed = True
                break
        if interval_failed:
            profile_stats[13] += 1.0
            if interval_edge:
                profile_stats[14] += 1.0
            else:
                profile_stats[15] += 1.0
            if profile_enabled and failed_interval_count < N:
                failed[failed_interval_count] = k
                failed_interval_count += 1
        for i in range(n):
            Vhist[k, i] = Vp[i]
    if profile_enabled:
        profile_stats[11] = N - 1
        profile_stats[12] = nsubsteps
    return True, Vhist, nsubsteps, -1, profile_stats, failed


def _pnoise_hb_blocks_impl(Gf, Cf, K, fundamental, charge_caps):
    N = Gf.shape[0]
    n = Gf.shape[1]
    nb = 2 * K + 1
    size = nb * n
    Y_base = np.zeros((size, size), dtype=np.complex128)
    C_block = np.zeros((size, size), dtype=np.complex128)
    for kr_i in range(nb):
        kr = kr_i - K
        br = kr_i * n
        for kc_i in range(nb):
            kc = kc_i - K
            sideband = kr if charge_caps else kc
            sideband_omega = 2.0j * math.pi * sideband * fundamental
            bc = kc_i * n
            coeff_idx = (kr - kc) % N
            for r in range(n):
                rr = br + r
                for c in range(n):
                    cc = bc + c
                    c_coeff = Cf[coeff_idx, r, c]
                    Y_base[rr, cc] = Gf[coeff_idx, r, c] + sideband_omega * c_coeff
                    C_block[rr, cc] = c_coeff
    return Y_base, C_block


def _pac_term_value_impl(kind, ref, value, node_wave, input_wave, m):
    if kind == 0:      # solved node
        return node_wave[m, ref]
    if kind == 1:      # periodic large-signal input
        return input_wave[ref, m]
    return value       # rail / constant


def _pac_stamp_coeff_impl(M, Min, row_kind, row_ref, col_kind, col_ref, coeff):
    if row_kind != 0 or coeff == 0.0:
        return
    if col_kind == 0:
        M[row_ref, col_ref] += coeff
    elif col_kind == 1:
        Min[row_ref, col_ref] += coeff


def _pac_stamp_adm_impl(M, Min, p_kind, p_ref, q_kind, q_ref, y):
    if y == 0.0:
        return
    if p_kind == 0:
        M[p_ref, p_ref] += y
        _pac_stamp_coeff_impl(M, Min, p_kind, p_ref, q_kind, q_ref, -y)
    if q_kind == 0:
        M[q_ref, q_ref] += y
        _pac_stamp_coeff_impl(M, Min, q_kind, q_ref, p_kind, p_ref, -y)


def _pac_stamp_vccs_impl(M, Min, d_kind, d_ref, g_kind, g_ref,
                         s_kind, s_ref, gm):
    _pac_stamp_coeff_impl(M, Min, d_kind, d_ref, g_kind, g_ref, gm)
    _pac_stamp_coeff_impl(M, Min, d_kind, d_ref, s_kind, s_ref, -gm)
    _pac_stamp_coeff_impl(M, Min, s_kind, s_ref, g_kind, g_ref, -gm)
    _pac_stamp_coeff_impl(M, Min, s_kind, s_ref, s_kind, s_ref, gm)


def _pac_idc_solved_impl(Vs, Vd, Vg, cache_valid, cache_vs1, cache_vd1,
                         Vfb, Vss, Lc, lambda_, contact_scale, exponent,
                         current_scale, inv_Rleak):
    ok, Vs1, Vd1, _, _, _ = _solve_internal_with_guesses_impl(
        Vs, Vd, Vg, cache_valid, cache_vs1, cache_vd1, 1e-12, 40,
        Vfb, Vss, Lc, lambda_, contact_scale, exponent, current_scale,
        inv_Rleak)
    if not ok:
        return False, 0.0
    F0a, F0b, _, _, _, _ = _residual_pair_jac_internal_impl(
        Vs, Vd, Vg, Vs1, Vd1, Vfb, Vss, Lc, lambda_, contact_scale,
        exponent, current_scale, inv_Rleak)
    return True, F0b - (Vs1 - Vd1) / 0.1


def _pac_linearize_orbit_impl(
        node_wave, input_wave,
        dev_value_d_kind, dev_value_d_ref, dev_value_d_val,
        dev_value_g_kind, dev_value_g_ref, dev_value_g_val,
        dev_value_s_kind, dev_value_s_ref, dev_value_s_val,
        dev_stamp_d_kind, dev_stamp_d_ref,
        dev_stamp_g_kind, dev_stamp_g_ref,
        dev_stamp_s_kind, dev_stamp_s_ref,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1,
        res_a_kind, res_a_ref, res_b_kind, res_b_ref, res_g,
        cap_a_kind, cap_a_ref, cap_b_kind, cap_b_ref, cap_value,
        ndrive):
    N = node_wave.shape[0]
    n = node_wave.shape[1]
    Gt = np.zeros((N, n, n), dtype=np.float64)
    Ct = np.zeros((N, n, n), dtype=np.float64)
    Gin = np.zeros((N, n, ndrive), dtype=np.float64)
    Cin = np.zeros((N, n, ndrive), dtype=np.float64)
    op_cache_valid = np.zeros(p_Vfb.shape[0], dtype=np.bool_)
    op_cache_vs1 = np.zeros(p_Vfb.shape[0], dtype=np.float64)
    op_cache_vd1 = np.zeros(p_Vfb.shape[0], dtype=np.float64)

    for m in range(N):
        Gm = Gt[m]
        Cm = Ct[m]
        Gim = Gin[m]
        Cim = Cin[m]
        for k in range(n):
            Gm[k, k] += 1e-12

        for pos in range(res_g.shape[0]):
            _pac_stamp_adm_impl(
                Gm, Gim,
                res_a_kind[pos], res_a_ref[pos],
                res_b_kind[pos], res_b_ref[pos],
                res_g[pos])

        for pos in range(cap_value.shape[0]):
            _pac_stamp_adm_impl(
                Cm, Cim,
                cap_a_kind[pos], cap_a_ref[pos],
                cap_b_kind[pos], cap_b_ref[pos],
                cap_value[pos])

        for pos in range(p_Vfb.shape[0]):
            Vs = _pac_term_value_impl(
                dev_value_s_kind[pos], dev_value_s_ref[pos],
                dev_value_s_val[pos], node_wave, input_wave, m)
            Vd = _pac_term_value_impl(
                dev_value_d_kind[pos], dev_value_d_ref[pos],
                dev_value_d_val[pos], node_wave, input_wave, m)
            Vg = _pac_term_value_impl(
                dev_value_g_kind[pos], dev_value_g_ref[pos],
                dev_value_g_val[pos], node_wave, input_wave, m)

            ok, Vs1, Vd1, _, _, _ = _solve_internal_with_guesses_impl(
                Vs, Vd, Vg, op_cache_valid[pos], op_cache_vs1[pos],
                op_cache_vd1[pos], 1e-12, 40, p_Vfb[pos], p_Vss[pos],
                p_Lc[pos], p_lambda[pos], p_contact_scale[pos],
                p_exponent[pos], p_current_scale[pos], p_inv_Rleak[pos])
            if not ok:
                return False, Gt, Ct, Gin, Cin
            op_cache_valid[pos] = True
            op_cache_vs1[pos] = Vs1
            op_cache_vd1[pos] = Vd1

            F0a, F0b, j00, j01, j10, j11 = _residual_pair_jac_internal_impl(
                Vs, Vd, Vg, Vs1, Vd1, p_Vfb[pos], p_Vss[pos], p_Lc[pos],
                p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
                p_current_scale[pos], p_inv_Rleak[pos])
            Idc0 = F0b - (Vs1 - Vd1) / 0.1
            use_fd = abs(Idc0) < 1e-10
            gm = 0.0
            gds = 1e-12
            if not use_fd:
                okd, gm_neg, gds_neg = _terminal_derivatives_from_jac_impl(
                    Vs, Vd, Vg, Vs1, Vd1, F0a, F0b, Idc0, j00, j01, j10, j11,
                    True, True, False, 1e-3, p_Vfb[pos], p_Vss[pos], p_Lc[pos],
                    p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
                    p_current_scale[pos], p_inv_Rleak[pos])
                if okd and math.isfinite(gm_neg) and math.isfinite(gds_neg):
                    gm = -gm_neg
                    gds = -gds_neg
                else:
                    use_fd = True
            if use_fd:
                h = 1e-3
                okp, idp = _pac_idc_solved_impl(
                    Vs, Vd, Vg + h, True, Vs1, Vd1,
                    p_Vfb[pos], p_Vss[pos], p_Lc[pos], p_lambda[pos],
                    p_contact_scale[pos], p_exponent[pos],
                    p_current_scale[pos], p_inv_Rleak[pos])
                okm, idm = _pac_idc_solved_impl(
                    Vs, Vd, Vg - h, True, Vs1, Vd1,
                    p_Vfb[pos], p_Vss[pos], p_Lc[pos], p_lambda[pos],
                    p_contact_scale[pos], p_exponent[pos],
                    p_current_scale[pos], p_inv_Rleak[pos])
                okdp, iddp = _pac_idc_solved_impl(
                    Vs, Vd + h, Vg, True, Vs1, Vd1,
                    p_Vfb[pos], p_Vss[pos], p_Lc[pos], p_lambda[pos],
                    p_contact_scale[pos], p_exponent[pos],
                    p_current_scale[pos], p_inv_Rleak[pos])
                okdm, iddm = _pac_idc_solved_impl(
                    Vs, Vd - h, Vg, True, Vs1, Vd1,
                    p_Vfb[pos], p_Vss[pos], p_Lc[pos], p_lambda[pos],
                    p_contact_scale[pos], p_exponent[pos],
                    p_current_scale[pos], p_inv_Rleak[pos])
                if not (okp and okm and okdp and okdm):
                    return False, Gt, Ct, Gin, Cin
                gm = (idp - idm) / (2.0 * h)
                gds = (iddp - iddm) / (2.0 * h)
                if not math.isfinite(gm) or not math.isfinite(gds):
                    return False, Gt, Ct, Gin, Cin
                if gm < 0.0:
                    gm = 0.0
                if gds < 1e-12:
                    gds = 1e-12
            Cgs, Cgd = _capacitances_impl(
                Vs, Vd, Vg, Vs1, Vd1, p_Vfb[pos], p_two_over_pi[pos],
                p_cap_cgs1[pos], p_cap_cgd1[pos], p_cap_half_wl_ci[pos],
                p_cap_cgs3_base[pos], p_cap_cgd3_base[pos], p_k1[pos])

            dk = dev_stamp_d_kind[pos]
            dr = dev_stamp_d_ref[pos]
            gk = dev_stamp_g_kind[pos]
            gr = dev_stamp_g_ref[pos]
            sk = dev_stamp_s_kind[pos]
            sr = dev_stamp_s_ref[pos]
            _pac_stamp_adm_impl(Gm, Gim, dk, dr, sk, sr, gds)
            _pac_stamp_adm_impl(Cm, Cim, gk, gr, sk, sr, Cgs)
            _pac_stamp_adm_impl(Cm, Cim, gk, gr, dk, dr, Cgd)
            _pac_stamp_vccs_impl(Gm, Gim, dk, dr, gk, gr, sk, sr, gm)

    return True, Gt, Ct, Gin, Cin


def _pac_term_deriv_impl(kind, ref, node_dot, input_dot, m):
    # d/dt of a device terminal's large-signal waveform (matches term_derivative):
    # solved node -> node_dot, input/clock -> input_dot, rail -> 0.
    if kind == 0:
        return node_dot[m, ref]
    elif kind == 1:
        return input_dot[ref, m]
    return 0.0


def _pac_linearize_orbit_gate1_impl(
        node_wave, input_wave, node_dot, input_dot,
        dev_value_d_kind, dev_value_d_ref, dev_value_d_val,
        dev_value_g_kind, dev_value_g_ref, dev_value_g_val,
        dev_value_s_kind, dev_value_s_ref, dev_value_s_val,
        dev_stamp_d_kind, dev_stamp_d_ref,
        dev_stamp_g_kind, dev_stamp_g_ref,
        dev_stamp_s_kind, dev_stamp_s_ref,
        gate1_ref, p_R_cap, p_R_cap2,
        p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_Rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1,
        res_a_kind, res_a_ref, res_b_kind, res_b_ref, res_g,
        cap_a_kind, cap_a_ref, cap_b_kind, cap_b_ref, cap_value,
        ndrive, n_state, fd_step):
    # Verilog-A (non-conservative) cap linearization WITH the PMOS_TFT gate1 internal
    # node retained -- the numba twin of _assemble_pac_linearization_python (charge_caps
    # False, internal_gate_states True). Caps flow s/d <-> gate1 (not gate), with R_cap /
    # R_cap2 resistive branches, plus the edge-only multi-variable cross-coupling
    # [dC/dx * dx]*dV0/dt that the conservative charge fold drops.
    N = node_wave.shape[0]
    n = node_wave.shape[1]
    ndev = p_Vfb.shape[0]
    Gt = np.zeros((N, n_state, n_state), dtype=np.float64)
    Ct = np.zeros((N, n_state, n_state), dtype=np.float64)
    Gin = np.zeros((N, n_state, ndrive), dtype=np.float64)
    Cin = np.zeros((N, n_state, ndrive), dtype=np.float64)
    op_valid = np.zeros(ndev, dtype=np.bool_)
    op_vs1 = np.zeros(ndev, dtype=np.float64)
    op_vd1 = np.zeros(ndev, dtype=np.float64)
    h = fd_step

    for m in range(N):
        Gm = Gt[m]; Cm = Ct[m]; Gim = Gin[m]; Cim = Cin[m]
        for k in range(n):
            Gm[k, k] += 1e-12
        for pos in range(res_g.shape[0]):
            _pac_stamp_adm_impl(Gm, Gim, res_a_kind[pos], res_a_ref[pos],
                                res_b_kind[pos], res_b_ref[pos], res_g[pos])
        for pos in range(cap_value.shape[0]):
            _pac_stamp_adm_impl(Cm, Cim, cap_a_kind[pos], cap_a_ref[pos],
                                cap_b_kind[pos], cap_b_ref[pos], cap_value[pos])

        for pos in range(ndev):
            Vs = _pac_term_value_impl(dev_value_s_kind[pos], dev_value_s_ref[pos],
                                      dev_value_s_val[pos], node_wave, input_wave, m)
            Vd = _pac_term_value_impl(dev_value_d_kind[pos], dev_value_d_ref[pos],
                                      dev_value_d_val[pos], node_wave, input_wave, m)
            Vg = _pac_term_value_impl(dev_value_g_kind[pos], dev_value_g_ref[pos],
                                      dev_value_g_val[pos], node_wave, input_wave, m)
            ok, Vs1, Vd1, _, _, _ = _solve_internal_with_guesses_impl(
                Vs, Vd, Vg, op_valid[pos], op_vs1[pos], op_vd1[pos], 1e-12, 40,
                p_Vfb[pos], p_Vss[pos], p_Lc[pos], p_lambda[pos],
                p_contact_scale[pos], p_exponent[pos], p_current_scale[pos],
                p_inv_Rleak[pos])
            if not ok:
                return False, Gt, Ct, Gin, Cin
            op_valid[pos] = True; op_vs1[pos] = Vs1; op_vd1[pos] = Vd1

            F0a, F0b, j00, j01, j10, j11 = _residual_pair_jac_internal_impl(
                Vs, Vd, Vg, Vs1, Vd1, p_Vfb[pos], p_Vss[pos], p_Lc[pos],
                p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
                p_current_scale[pos], p_inv_Rleak[pos])
            Idc0 = F0b - (Vs1 - Vd1) / 0.1
            use_fd = abs(Idc0) < 1e-10
            gm = 0.0; gds = 1e-12
            if not use_fd:
                okd, gm_neg, gds_neg = _terminal_derivatives_from_jac_impl(
                    Vs, Vd, Vg, Vs1, Vd1, F0a, F0b, Idc0, j00, j01, j10, j11,
                    True, True, False, 1e-3, p_Vfb[pos], p_Vss[pos], p_Lc[pos],
                    p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
                    p_current_scale[pos], p_inv_Rleak[pos])
                if okd and math.isfinite(gm_neg) and math.isfinite(gds_neg):
                    gm = -gm_neg; gds = -gds_neg
                else:
                    use_fd = True
            if use_fd:
                hd = 1e-3
                okp, idp = _pac_idc_solved_impl(
                    Vs, Vd, Vg + hd, True, Vs1, Vd1, p_Vfb[pos], p_Vss[pos],
                    p_Lc[pos], p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
                    p_current_scale[pos], p_inv_Rleak[pos])
                okm2, idm = _pac_idc_solved_impl(
                    Vs, Vd, Vg - hd, True, Vs1, Vd1, p_Vfb[pos], p_Vss[pos],
                    p_Lc[pos], p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
                    p_current_scale[pos], p_inv_Rleak[pos])
                okdp, iddp = _pac_idc_solved_impl(
                    Vs, Vd + hd, Vg, True, Vs1, Vd1, p_Vfb[pos], p_Vss[pos],
                    p_Lc[pos], p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
                    p_current_scale[pos], p_inv_Rleak[pos])
                okdm, iddm = _pac_idc_solved_impl(
                    Vs, Vd - hd, Vg, True, Vs1, Vd1, p_Vfb[pos], p_Vss[pos],
                    p_Lc[pos], p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
                    p_current_scale[pos], p_inv_Rleak[pos])
                if not (okp and okm2 and okdp and okdm):
                    return False, Gt, Ct, Gin, Cin
                gm = (idp - idm) / (2.0 * hd)
                gds = (iddp - iddm) / (2.0 * hd)
                if not math.isfinite(gm) or not math.isfinite(gds):
                    return False, Gt, Ct, Gin, Cin
                if gm < 0.0:
                    gm = 0.0
                if gds < 1e-12:
                    gds = 1e-12
            Cgs, Cgd = _capacitances_impl(
                Vs, Vd, Vg, Vs1, Vd1, p_Vfb[pos], p_two_over_pi[pos],
                p_cap_cgs1[pos], p_cap_cgd1[pos], p_cap_half_wl_ci[pos],
                p_cap_cgs3_base[pos], p_cap_cgd3_base[pos], p_k1[pos])

            dk = dev_stamp_d_kind[pos]; dr = dev_stamp_d_ref[pos]
            gk = dev_stamp_g_kind[pos]; gr = dev_stamp_g_ref[pos]
            sk = dev_stamp_s_kind[pos]; sr = dev_stamp_s_ref[pos]
            g1r = gate1_ref[pos]                       # gate1 solved-node index (always >=0)
            inv_rc = 1.0 / p_R_cap[pos]
            inv_rc2 = 1.0 / p_R_cap2[pos]
            # channel gm/gds (to gate, no caps), then gate1 resistive + cap network.
            _pac_stamp_adm_impl(Gm, Gim, dk, dr, sk, sr, gds)
            _pac_stamp_vccs_impl(Gm, Gim, dk, dr, gk, gr, sk, sr, gm)
            _pac_stamp_adm_impl(Gm, Gim, 0, g1r, gk, gr, inv_rc)      # gate1 <-> gate
            _pac_stamp_adm_impl(Gm, Gim, sk, sr, 0, g1r, inv_rc2)     # s <-> gate1 leak
            _pac_stamp_adm_impl(Gm, Gim, dk, dr, 0, g1r, inv_rc2)     # d <-> gate1 leak
            _pac_stamp_adm_impl(Cm, Cim, sk, sr, 0, g1r, Cgs)         # Cgs to gate1
            _pac_stamp_adm_impl(Cm, Cim, dk, dr, 0, g1r, Cgd)         # Cgd to gate1

            # Edge-only cross-coupling: [dC/dx * dx]*dV0(s/d,gate1)/dt.
            dVs_dt = _pac_term_deriv_impl(dev_value_s_kind[pos], dev_value_s_ref[pos],
                                          node_dot, input_dot, m)
            dVd_dt = _pac_term_deriv_impl(dev_value_d_kind[pos], dev_value_d_ref[pos],
                                          node_dot, input_dot, m)
            dVg_dt = _pac_term_deriv_impl(dev_value_g_kind[pos], dev_value_g_ref[pos],
                                          node_dot, input_dot, m)
            # gate1 DC is LINEAR in (Vs,Vd,Vg) (KCL of R_cap/R_cap2), so its time
            # derivative is the same linear combo of the terminal derivatives -- exactly
            # the periodic derivative of dev._gate1_dc the Python path samples+differences.
            denom = inv_rc + 2.0 * inv_rc2
            dVg1_dt = (dVg_dt * inv_rc + (dVs_dt + dVd_dt) * inv_rc2) / denom
            vdot_sg1 = dVs_dt - dVg1_dt
            vdot_dg1 = dVd_dt - dVg1_dt
            if abs(vdot_sg1) < 1e-30 and abs(vdot_dg1) < 1e-30:
                continue
            # central-difference dC/d{Vs,Vd,Vg} (re-solving internals like get_capacitances)
            for axis in range(3):
                vsp = Vs; vdp = Vd; vgp = Vg
                vsm = Vs; vdm = Vd; vgm = Vg
                if axis == 0:
                    vsp = Vs + h; vsm = Vs - h
                elif axis == 1:
                    vdp = Vd + h; vdm = Vd - h
                else:
                    vgp = Vg + h; vgm = Vg - h
                okp2, vs1p, vd1p, _, _, _ = _solve_internal_with_guesses_impl(
                    vsp, vdp, vgp, True, Vs1, Vd1, 1e-12, 40, p_Vfb[pos], p_Vss[pos],
                    p_Lc[pos], p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
                    p_current_scale[pos], p_inv_Rleak[pos])
                okm3, vs1m, vd1m, _, _, _ = _solve_internal_with_guesses_impl(
                    vsm, vdm, vgm, True, Vs1, Vd1, 1e-12, 40, p_Vfb[pos], p_Vss[pos],
                    p_Lc[pos], p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
                    p_current_scale[pos], p_inv_Rleak[pos])
                if not (okp2 and okm3):
                    return False, Gt, Ct, Gin, Cin
                cgsp, cgdp = _capacitances_impl(
                    vsp, vdp, vgp, vs1p, vd1p, p_Vfb[pos], p_two_over_pi[pos],
                    p_cap_cgs1[pos], p_cap_cgd1[pos], p_cap_half_wl_ci[pos],
                    p_cap_cgs3_base[pos], p_cap_cgd3_base[pos], p_k1[pos])
                cgsm, cgdm = _capacitances_impl(
                    vsm, vdm, vgm, vs1m, vd1m, p_Vfb[pos], p_two_over_pi[pos],
                    p_cap_cgs1[pos], p_cap_cgd1[pos], p_cap_half_wl_ci[pos],
                    p_cap_cgs3_base[pos], p_cap_cgd3_base[pos], p_k1[pos])
                dCgs = (cgsp - cgsm) / (2.0 * h)
                dCgd = (cgdp - cgdm) / (2.0 * h)
                if axis == 0:
                    ck = sk; cr = sr
                elif axis == 1:
                    ck = dk; cr = dr
                else:
                    ck = gk; cr = gr
                # branch s->gate1 controlled by ctrl: +dCgs*vdot_sg1 ; d->gate1: +dCgd*vdot_dg1
                if vdot_sg1 != 0.0 and dCgs != 0.0:
                    cc = dCgs * vdot_sg1
                    _pac_stamp_coeff_impl(Gm, Gim, sk, sr, ck, cr, cc)
                    _pac_stamp_coeff_impl(Gm, Gim, 0, g1r, ck, cr, -cc)
                if vdot_dg1 != 0.0 and dCgd != 0.0:
                    cc = dCgd * vdot_dg1
                    _pac_stamp_coeff_impl(Gm, Gim, dk, dr, ck, cr, cc)
                    _pac_stamp_coeff_impl(Gm, Gim, 0, g1r, ck, cr, -cc)

    return True, Gt, Ct, Gin, Cin


def _pnoise_fold_psd_impl(adjs, freqs, K, fundamental,
                          p_indices, q_indices, sth_grids, mfl_grids):
    # sth_grids[si] is the Toeplitz power-harmonic matrix of the (white) thermal
    # source -> Z^H S_th Z.  mfl_grids[si] is the sqrt(PWR) modulation-amplitude
    # harmonic vector M_{-2K..2K} of the 1/f source; the cyclostationary flicker
    # output is sum_a |sum_r Z_r M_{r-a}|^2 / nu_a (M_{r-a}=mfl_grids[si, (r-a)+2K]).
    nfreq = freqs.shape[0]
    nsrc = p_indices.shape[0]
    nb = 2 * K + 1
    two_k = 2 * K
    out_psd = np.zeros(nfreq, dtype=np.float64)
    dev_psd = np.zeros((nsrc, nfreq), dtype=np.float64)
    inv_nu = np.empty(nb, dtype=np.float64)
    Z = np.empty(nb, dtype=np.complex128)
    for fi in range(nfreq):
        freq = freqs[fi]
        for a in range(nb):
            nu = abs(freq + (a - K) * fundamental)
            if nu < 1e-9:
                nu = 1e-9
            inv_nu[a] = 1.0 / nu

        adj = adjs[fi]
        for si in range(nsrc):
            for r in range(nb):
                pr = p_indices[si, r]
                qr = q_indices[si, r]
                zr = 0.0j
                if pr >= 0:
                    zr += adj[pr]
                if qr >= 0:
                    zr -= adj[qr]
                Z[r] = zr
            contrib = 0.0
            # thermal (white): Z^H S_th Z with the Toeplitz power-harmonic matrix.
            for r in range(nb):
                acc = 0.0j
                for c in range(nb):
                    acc += sth_grids[si, r, c] * Z[c].conjugate()
                contrib += (Z[r] * acc).real
            # flicker (1/f): sum_a |sum_r Z_r M_{r-a}|^2 / nu_a.
            for a in range(nb):
                base = two_k - a
                u = 0.0j
                for r in range(nb):
                    u += Z[r] * mfl_grids[si, base + r]
                contrib += (u.real * u.real + u.imag * u.imag) * inv_nu[a]
            if contrib < 0.0:
                contrib = 0.0
            dev_psd[si, fi] = contrib
            out_psd[fi] += contrib
    return out_psd, dev_psd


if NUMBA_AVAILABLE:
    # Enable disk cache so new Python processes can reuse compiled kernels.
    # Set CIRCUIT_NUMBA_CACHE=0 if stale cache debugging is needed.
    _softplus_py = njit(cache=NUMBA_CACHE)(_softplus_py)
    _eval_currents_impl = njit(cache=NUMBA_CACHE)(_eval_currents_impl)
    _residual_pair_impl = njit(cache=NUMBA_CACHE)(_residual_pair_impl)
    _sigmoid_impl = njit(cache=NUMBA_CACHE)(_sigmoid_impl)
    _residual_pair_fd_jac_impl = njit(cache=NUMBA_CACHE)(_residual_pair_fd_jac_impl)
    _residual_pair_jac_internal_impl = njit(cache=NUMBA_CACHE)(_residual_pair_jac_internal_impl)
    _newton_internal_impl = njit(cache=NUMBA_CACHE)(_newton_internal_impl)
    _newton_internal_fast_impl = njit(cache=NUMBA_CACHE)(_newton_internal_fast_impl)
    _capacitances_impl = njit(cache=NUMBA_CACHE)(_capacitances_impl)
    _atan_cap_integral_impl = njit(cache=NUMBA_CACHE)(_atan_cap_integral_impl)
    _capacitance_charges_impl = njit(cache=NUMBA_CACHE)(_capacitance_charges_impl)
    _eval_at_impl = njit(cache=NUMBA_CACHE)(_eval_at_impl)
    _terminal_deriv_one_impl = njit(cache=NUMBA_CACHE)(_terminal_deriv_one_impl)
    _terminal_derivatives_from_base_impl = njit(cache=NUMBA_CACHE)(_terminal_derivatives_from_base_impl)
    _terminal_derivatives_from_jac_fdterm_impl = njit(cache=NUMBA_CACHE)(_terminal_derivatives_from_jac_fdterm_impl)
    _contact_diss_dvg_impl = njit(cache=NUMBA_CACHE)(_contact_diss_dvg_impl)
    _channel_partials_impl = njit(cache=NUMBA_CACHE)(_channel_partials_impl)
    _terminal_deriv_from_partials_impl = njit(cache=NUMBA_CACHE)(_terminal_deriv_from_partials_impl)
    _terminal_derivatives_from_jac_impl = njit(cache=NUMBA_CACHE)(_terminal_derivatives_from_jac_impl)
    _terminal_derivatives_impl = njit(cache=NUMBA_CACHE)(_terminal_derivatives_impl)
    _term_value_impl = njit(cache=NUMBA_CACHE)(_term_value_impl)
    _solve_internal_with_guesses_impl = njit(cache=NUMBA_CACHE)(_solve_internal_with_guesses_impl)
    _fill_prev_terms_impl = njit(cache=NUMBA_CACHE)(_fill_prev_terms_impl)
    _solve_dense_neg_rhs_inplace_impl = njit(cache=NUMBA_CACHE)(_solve_dense_neg_rhs_inplace_impl)
    _stamp_transient_system_impl = njit(cache=NUMBA_CACHE)(_stamp_transient_system_impl)
    _transient_newton_impl = njit(cache=NUMBA_CACHE)(_transient_newton_impl)
    _transient_newton_reuse_impl = njit(cache=NUMBA_CACHE)(_transient_newton_reuse_impl)
    _transient_solve_grid_impl = njit(cache=NUMBA_CACHE)(_transient_solve_grid_impl)
    _gear2_substep_newton_reuse_impl = njit(cache=NUMBA_CACHE)(_gear2_substep_newton_reuse_impl)
    _interp_inputs_at_time_impl = njit(cache=NUMBA_CACHE)(_interp_inputs_at_time_impl)
    _adaptive_error_impl = njit(cache=NUMBA_CACHE)(_adaptive_error_impl)
    _adaptive_next_h_impl = njit(cache=NUMBA_CACHE)(_adaptive_next_h_impl)
    _transient_solve_adaptive_gear2_impl = njit(cache=NUMBA_CACHE)(_transient_solve_adaptive_gear2_impl)
    _transient_solve_grid_gear2_impl = njit(cache=NUMBA_CACHE)(_transient_solve_grid_gear2_impl)
    _pnoise_hb_blocks_impl = njit(cache=NUMBA_CACHE)(_pnoise_hb_blocks_impl)
    _pac_term_value_impl = njit(cache=NUMBA_CACHE)(_pac_term_value_impl)
    _pac_stamp_coeff_impl = njit(cache=NUMBA_CACHE)(_pac_stamp_coeff_impl)
    _pac_stamp_adm_impl = njit(cache=NUMBA_CACHE)(_pac_stamp_adm_impl)
    _pac_stamp_vccs_impl = njit(cache=NUMBA_CACHE)(_pac_stamp_vccs_impl)
    _pac_idc_solved_impl = njit(cache=NUMBA_CACHE)(_pac_idc_solved_impl)
    _pac_linearize_orbit_impl = njit(cache=NUMBA_CACHE)(_pac_linearize_orbit_impl)
    _pac_term_deriv_impl = njit(cache=NUMBA_CACHE)(_pac_term_deriv_impl)
    _pac_linearize_orbit_gate1_impl = njit(cache=NUMBA_CACHE)(_pac_linearize_orbit_gate1_impl)
    _pnoise_fold_psd_impl = njit(cache=NUMBA_CACHE)(_pnoise_fold_psd_impl)
    eval_currents_numba = _eval_currents_impl
    newton_internal_numba = _newton_internal_impl
    capacitances_numba = _capacitances_impl
    capacitance_charges_numba = _capacitance_charges_impl
    terminal_derivatives_numba = _terminal_derivatives_impl
    transient_newton_numba = _transient_newton_impl
    transient_solve_grid_numba = _transient_solve_grid_impl
    transient_solve_grid_gear2_numba = _transient_solve_grid_gear2_impl
    transient_solve_adaptive_gear2_numba = _transient_solve_adaptive_gear2_impl
    pnoise_hb_blocks_numba = _pnoise_hb_blocks_impl
    pac_hb_blocks_numba = _pnoise_hb_blocks_impl
    pac_linearize_orbit_numba = _pac_linearize_orbit_impl
    pac_linearize_orbit_gate1_numba = _pac_linearize_orbit_gate1_impl
    pnoise_fold_psd_numba = _pnoise_fold_psd_impl
else:
    eval_currents_numba = None
    newton_internal_numba = None
    capacitances_numba = None
    capacitance_charges_numba = None
    terminal_derivatives_numba = None
    transient_newton_numba = None
    transient_solve_grid_numba = None
    transient_solve_grid_gear2_numba = None
    transient_solve_adaptive_gear2_numba = None
    pnoise_hb_blocks_numba = None
    pac_hb_blocks_numba = None
    pac_linearize_orbit_numba = None
    pac_linearize_orbit_gate1_numba = None
    pnoise_fold_psd_numba = None
