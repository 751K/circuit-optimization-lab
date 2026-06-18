"""Optional numba-accelerated scalar kernels.

This module must remain importable without numba installed. When numba is
available the kernels are enabled by default; set CIRCUIT_USE_NUMBA=0/false/off
to force the pure-Python path for debugging.
"""
import math
import os

import numpy as np


_USE_NUMBA_FLAG = os.environ.get("CIRCUIT_USE_NUMBA")
USE_NUMBA = (_USE_NUMBA_FLAG is None or
             _USE_NUMBA_FLAG.lower() in {"1", "true", "yes", "on"})
if _USE_NUMBA_FLAG is not None and _USE_NUMBA_FLAG.lower() in {"0", "false", "no", "off"}:
    USE_NUMBA = False

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


def _terminal_derivatives_impl(Vs, Vd, Vg, Vs1, Vd1, need_gm, need_gds, use_abs,
                               HH, hx,
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


def _term_value_impl(kind, ref, value, V, input_values):
    if kind == 0:      # solved
        return V[ref]
    if kind == 1:      # transient input
        return input_values[ref]
    return value       # rail / constant


def _solve_internal_with_guesses_impl(Vs, Vd, Vg, cache_valid, cache_vs1,
                                      cache_vd1, tol, maxit, Vfb, Vss, Lc,
                                      lambda_, contact_scale, exponent,
                                      current_scale, inv_Rleak):
    if cache_valid:
        ok, xs, xd = _newton_internal_impl(
            Vs, Vd, Vg, cache_vs1, cache_vd1, tol, maxit,
            Vfb, Vss, Lc, lambda_, contact_scale, exponent,
            current_scale, inv_Rleak)
        if ok:
            return True, xs, xd

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
        ok, xs, xd = _newton_internal_impl(
            Vs, Vd, Vg, xs0, xd0, tol, maxit,
            Vfb, Vss, Lc, lambda_, contact_scale, exponent,
            current_scale, inv_Rleak)
        if ok:
            return True, xs, xd
    return False, cache_vs1, cache_vd1


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
        dyn_pi, dyn_qi, dyn_input_idx):
    R = np.zeros(n)
    J = np.zeros((n, n))
    inv_h = 1.0 / h

    for pos in range(dev_di.shape[0]):
        Vs = _term_value_impl(dev_s_kind[pos], dev_s_ref[pos], dev_s_val[pos],
                              V, input_now)
        Vd = _term_value_impl(dev_d_kind[pos], dev_d_ref[pos], dev_d_val[pos],
                              V, input_now)
        Vg = _term_value_impl(dev_g_kind[pos], dev_g_ref[pos], dev_g_val[pos],
                              V, input_now)
        pVs = _term_value_impl(dev_s_kind[pos], dev_s_ref[pos], dev_s_val[pos],
                               Vp, input_prev)
        pVd = _term_value_impl(dev_d_kind[pos], dev_d_ref[pos], dev_d_val[pos],
                               Vp, input_prev)
        pVg = _term_value_impl(dev_g_kind[pos], dev_g_ref[pos], dev_g_val[pos],
                               Vp, input_prev)

        ok, Vs1, Vd1 = _solve_internal_with_guesses_impl(
            Vs, Vd, Vg, op_cache_valid[pos], op_cache_vs1[pos],
            op_cache_vd1[pos], 1e-12, 40, p_Vfb[pos], p_Vss[pos], p_Lc[pos],
            p_lambda[pos], p_contact_scale[pos], p_exponent[pos],
            p_current_scale[pos], p_inv_Rleak[pos])
        if not ok:
            return False, R, J
        op_cache_valid[pos] = True
        op_cache_vs1[pos] = Vs1
        op_cache_vd1[pos] = Vd1

        _, _, I_d1_d, _, _ = _eval_currents_impl(
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
        I = abs(-I_d1_d) if dev_use_abs[pos] else I_d1_d
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
            i_ab = Cgs * inv_h * ((Vg - Vs) - (pVg - pVs))
            if gi >= 0:
                R[gi] -= i_ab
            if si >= 0:
                R[si] += i_ab
        if Cgd != 0.0:
            i_ab = Cgd * inv_h * ((Vg - Vd) - (pVg - pVd))
            if gi >= 0:
                R[gi] -= i_ab
            if di >= 0:
                R[di] += i_ab

        need_gm = gi >= 0 or si >= 0
        need_gds = di >= 0 or si >= 0
        okd, gm, gds = _terminal_derivatives_impl(
            Vs, Vd, Vg, Vs1, Vd1, need_gm, need_gds, dev_use_abs[pos],
            HH, 1e-6, p_Vfb[pos], p_Vss[pos], p_Lc[pos], p_lambda[pos],
            p_contact_scale[pos], p_exponent[pos], p_current_scale[pos],
            p_inv_Rleak[pos])
        if not okd:
            return False, R, J
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
            pva = _term_value_impl(cap_a_kind[pos], cap_a_ref[pos],
                                   cap_a_val[pos], Vp, input_prev)
            pvb = _term_value_impl(cap_b_kind[pos], cap_b_ref[pos],
                                   cap_b_val[pos], Vp, input_prev)
            i_ab = cap * inv_h * ((va - vb) - (pva - pvb))
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

    return True, R, J


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
        dyn_pi, dyn_qi, dyn_input_idx):
    V = seed.copy()
    prev = math.inf
    for it in range(maxit):
        ok, R, J = _stamp_transient_system_impl(
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
            dyn_pi, dyn_qi, dyn_input_idx)
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
        if mx < vtol:
            return V, it + 1, True, True
        if it >= 4 and mx >= prev and mx < 1e-5:
            return V, it + 1, True, True
        prev = mx
    return V, maxit, False, True


if NUMBA_AVAILABLE:
    # Keep cache disabled: this project is commonly run both as a package
    # (`core.numba_kernels`) and, historically, as flat modules. Numba's on-disk
    # cache stores the module path and stale flat-module cache entries can be
    # loaded after the package migration, breaking optional acceleration.
    _softplus_py = njit(cache=False)(_softplus_py)
    _eval_currents_impl = njit(cache=False)(_eval_currents_impl)
    _residual_pair_impl = njit(cache=False)(_residual_pair_impl)
    _newton_internal_impl = njit(cache=False)(_newton_internal_impl)
    _capacitances_impl = njit(cache=False)(_capacitances_impl)
    _eval_at_impl = njit(cache=False)(_eval_at_impl)
    _terminal_deriv_one_impl = njit(cache=False)(_terminal_deriv_one_impl)
    _terminal_derivatives_impl = njit(cache=False)(_terminal_derivatives_impl)
    _term_value_impl = njit(cache=False)(_term_value_impl)
    _solve_internal_with_guesses_impl = njit(cache=False)(_solve_internal_with_guesses_impl)
    _solve_dense_neg_rhs_inplace_impl = njit(cache=False)(_solve_dense_neg_rhs_inplace_impl)
    _stamp_transient_system_impl = njit(cache=False)(_stamp_transient_system_impl)
    eval_currents_numba = _eval_currents_impl
    newton_internal_numba = _newton_internal_impl
    capacitances_numba = _capacitances_impl
    terminal_derivatives_numba = _terminal_derivatives_impl
    transient_newton_numba = njit(cache=False)(_transient_newton_impl)
else:
    eval_currents_numba = None
    newton_internal_numba = None
    capacitances_numba = None
    terminal_derivatives_numba = None
    transient_newton_numba = None
