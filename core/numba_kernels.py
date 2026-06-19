"""Optional numba-accelerated scalar kernels.

This module must remain importable without numba installed. When numba is
available the kernels are enabled by default; set CIRCUIT_USE_NUMBA=0/false/off
to force the pure-Python path for debugging. Numba's on-disk cache is enabled
by default; set CIRCUIT_NUMBA_CACHE=0/false/off to disable it.
"""
import math
import os

import numpy as np


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
        Cgs, Cgd = _capacitances_impl(
            pVs, pVd, pVg, pVs1, pVd1, p_Vfb[pos], p_two_over_pi[pos],
            p_cap_cgs1[pos], p_cap_cgd1[pos], p_cap_half_wl_ci[pos],
            p_cap_cgs3_base[pos], p_cap_cgd3_base[pos], p_k1[pos])
        prev_cgs[pos] = Cgs
        prev_cgd[pos] = Cgd
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
        prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd, cap_prev_dv,
        R, J, profile_enabled, profile_stats):
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
        Cgs, Cgd = _capacitances_impl(
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
            c_step = 0.5 * (Cgs + prev_cgs[pos])
            i_ab = c_step * inv_h * ((Vg - Vs) - (pVg - pVs))
            if gi >= 0:
                R[gi] -= i_ab
            if si >= 0:
                R[si] += i_ab
        if Cgd != 0.0:
            c_step = 0.5 * (Cgd + prev_cgd[pos])
            i_ab = c_step * inv_h * ((Vg - Vd) - (pVg - pVd))
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
            gc = Cgs * inv_h
            c_step = 0.5 * (Cgs + prev_cgs[pos])
            gc = c_step * inv_h
            if gi >= 0:
                J[gi, gi] -= gc
                if si >= 0:
                    J[gi, si] += gc
            if si >= 0:
                J[si, si] -= gc
                if gi >= 0:
                    J[si, gi] += gc
        if Cgd != 0.0:
            gc = Cgd * inv_h
            c_step = 0.5 * (Cgd + prev_cgd[pos])
            gc = c_step * inv_h
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
            i_ab = cap * inv_h * ((va - vb) - cap_prev_dv[pos])
            gc = cap * inv_h
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
        clip_lo, clip_hi):
    V = seed.copy()
    R = np.empty(n)
    J = np.empty((n, n))
    profile_stats = np.zeros(16)
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
            prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd, cap_prev_dv,
            R, J, False, profile_stats)
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
        clip_lo, clip_hi,
        V, R, J, prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd, cap_prev_dv,
        profile_enabled, profile_stats):
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
        prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd, cap_prev_dv)
    if not ok_prev:
        return 0, False, False
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
            prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd, cap_prev_dv,
            R, J, profile_enabled, profile_stats)
        if not ok:
            return it + 1, False, False

        if fallback_accept:
            rmax = 0.0
            for i in range(n):
                val = abs(R[i])
                if val > rmax:
                    rmax = val
            if rmax < fallback_tol:
                return it + 1, True, True

        solved, dV = _solve_dense_neg_rhs_inplace_impl(J, R)
        if not solved:
            return it + 1, False, True
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
            return it + 1, True, True
        if it >= 4 and mx >= prev and mx < 1e-5:
            if fallback_accept:
                prev = mx
                continue
            return it + 1, True, True
        prev = mx
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
    profile_stats = np.zeros(16)
    nsubsteps = 0

    for k in range(1, N):
        nsubsteps_before_interval = nsubsteps
        h = tgrid[k] - tgrid[k - 1]
        if h <= 0.0:
            return False, Vhist, nsubsteps, k, profile_stats
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
                clip_lo, clip_hi,
                Vwork, R, J, prev_vs, prev_vd, prev_vg,
                prev_cgs, prev_cgd, cap_prev_dv,
                profile_enabled, profile_stats)
            if not ok:
                retry_count = 1
                for _retry_pow in range(max_retry_subdivisions):
                    retry_count *= 2
                if retry_count <= 1:
                    profile_stats[10] += 1.0
                    if fallback_accept:
                        if profile_enabled:
                            profile_stats[0] += iters
                        return False, Vhist, nsubsteps_before_interval, k, profile_stats
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
                        clip_lo, clip_hi,
                        Vwork, R, J, prev_vs, prev_vd, prev_vg,
                        prev_cgs, prev_cgd, cap_prev_dv,
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
                        Vp[i] = Vwork[i]
                    for ii in range(ninputs):
                        in0[ii] = in1[ii]
                if not retry_ok:
                    if fallback_accept:
                        return False, Vhist, nsubsteps_before_interval, k, profile_stats
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
        for i in range(n):
            Vhist[k, i] = Vp[i]
    if profile_enabled:
        profile_stats[11] = N - 1
        profile_stats[12] = nsubsteps
    return True, Vhist, nsubsteps, -1, profile_stats


def _pnoise_hb_blocks_impl(Gf, Cf, K, fundamental):
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
            col_omega = 2.0j * math.pi * kc * fundamental
            bc = kc_i * n
            coeff_idx = (kr - kc) % N
            for r in range(n):
                rr = br + r
                for c in range(n):
                    cc = bc + c
                    c_coeff = Cf[coeff_idx, r, c]
                    Y_base[rr, cc] = Gf[coeff_idx, r, c] + col_omega * c_coeff
                    C_block[rr, cc] = c_coeff
    return Y_base, C_block


def _pnoise_fold_psd_impl(adjs, freqs, K, fundamental,
                          p_indices, q_indices, sth_grids, sfl_grids):
    nfreq = freqs.shape[0]
    nsrc = p_indices.shape[0]
    nb = 2 * K + 1
    out_psd = np.zeros(nfreq, dtype=np.float64)
    dev_psd = np.zeros((nsrc, nfreq), dtype=np.float64)
    inv_sqrt_nu = np.empty(nb, dtype=np.float64)
    for fi in range(nfreq):
        freq = freqs[fi]
        for r in range(nb):
            nu = abs(freq + (r - K) * fundamental)
            if nu < 1e-9:
                nu = 1e-9
            inv_sqrt_nu[r] = 1.0 / math.sqrt(nu)

        adj = adjs[fi]
        for si in range(nsrc):
            contrib = 0.0
            for r in range(nb):
                pr = p_indices[si, r]
                qr = q_indices[si, r]
                zr = 0.0j
                if pr >= 0:
                    zr += adj[pr]
                if qr >= 0:
                    zr -= adj[qr]
                for c in range(nb):
                    pc = p_indices[si, c]
                    qc = q_indices[si, c]
                    zc = 0.0j
                    if pc >= 0:
                        zc += adj[pc]
                    if qc >= 0:
                        zc -= adj[qc]
                    smat = (sth_grids[si, r, c] +
                            sfl_grids[si, r, c] * inv_sqrt_nu[r] * inv_sqrt_nu[c])
                    contrib += (zr * smat * zc.conjugate()).real
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
    _pnoise_hb_blocks_impl = njit(cache=NUMBA_CACHE)(_pnoise_hb_blocks_impl)
    _pnoise_fold_psd_impl = njit(cache=NUMBA_CACHE)(_pnoise_fold_psd_impl)
    eval_currents_numba = _eval_currents_impl
    newton_internal_numba = _newton_internal_impl
    capacitances_numba = _capacitances_impl
    capacitance_charges_numba = _capacitance_charges_impl
    terminal_derivatives_numba = _terminal_derivatives_impl
    transient_newton_numba = _transient_newton_impl
    transient_solve_grid_numba = _transient_solve_grid_impl
    pnoise_hb_blocks_numba = _pnoise_hb_blocks_impl
    pnoise_fold_psd_numba = _pnoise_fold_psd_impl
else:
    eval_currents_numba = None
    newton_internal_numba = None
    capacitances_numba = None
    capacitance_charges_numba = None
    terminal_derivatives_numba = None
    transient_newton_numba = None
    transient_solve_grid_numba = None
    pnoise_hb_blocks_numba = None
    pnoise_fold_psd_numba = None
