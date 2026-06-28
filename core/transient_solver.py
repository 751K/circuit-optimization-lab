"""
Nonlinear transient solver for the AFE (backward-Euler + per-step Newton).

Integrates the full 6-node circuit DAE in time. Built on the same device model
(:class:`~device_model.TransistorModel`) and topology (AFE_TOPO) as the
DC/AC/Noise stack, so the steady state matches the DC solver and the
small-signal response matches the AC solver.

Method
------
- KCL at every solved node:  Σ device currents  +  Σ capacitor currents  = 0.
- Capacitor companion (backward Euler), cap branch between terminals a,b:
      i_ab ≈ C_step·[(Va-Vb)_n − (Va-Vb)_{n-1}] / h
  Linear capacitors use their fixed C. PMOS Cgss/Cgdd follow the AT_4000TG
  step companion selected by CIRCUIT_PMOS_TRANSIENT_CAP_MODE (or the per-call
  cap_mode_id): `charge` (default, L-stable Q-stamp), `average` (trapezoidal,
  the stable non-conservative form matching Cadence's feedthrough), `branch`,
  or `endpoint`/`veriloga` (the literal C(Vn)*dV companion, unstable in the
  shooting -- experiments only). AC/PAC/noise still use the local small-signal
  capacitances.
  The AT_4000TG model routes these caps through an internal gate1 node; this
  solver keeps the long-timescale R_cap2 leakage from source/drain to gate1 and
  collapses the 100 Ω gate-to-gate1 RC because its ns-scale time constant is far
  below the chopper edge times used here.
- Each step: solve the 6 node voltages with a damped Newton iteration using an
  analytic conductance Jacobian (gm/gds finite-diff of get_Idc + cap C/h + gmin),
  seeded from the previous step, with step limiting that keeps it on the physical
  branch. (Bare fsolve's poorly-scaled numeric Jacobian latched onto wrong roots
  of this positive-feedback circuit — gain didn't match the AC reference.)

Caps stamped: per device Cgs (gate-source) and Cgd (gate-drain), plus CL on the
two outputs. Inputs: M7 gate = vip(t), M8 gate = vin(t) (driven); other rails fixed.

CURRENT SIGN: default AFE device currents use abs(get_Idc), exactly like
ac_solve's KCL. Bidirectional pass switches can be listed in signed_devices so
their Verilog-A drain-terminal current keeps its physical sign when source/drain
voltages reverse.

This is the engine; chopper switch topologies can be driven through node_inputs.
Clock feedthrough from the Verilog-A Cgss/Cgdd terms is included when the switch
gate clocks are finite-edge waveforms. Additional explicit charge-injection
pulses can be supplied through current_inputs.
"""
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
try:
    from .topology import AFE_TOPO
    from .ac_solver import ac_solve, build_devices, _dev_corner, _dev_nf
    from .numba_kernels import (
        terminal_derivatives_numba,
        transient_newton_numba,
        transient_solve_grid_numba,
        transient_solve_grid_gear2_numba,
        transient_solve_adaptive_gear2_numba,
    )
    from .compiled_topology import CompiledTopology
except ImportError:  # pragma: no cover - legacy direct module import
    from topology import AFE_TOPO
    from ac_solver import ac_solve, build_devices, _dev_corner, _dev_nf
    from compiled_topology import CompiledTopology
    try:
        from numba_kernels import (
            terminal_derivatives_numba,
            transient_newton_numba,
            transient_solve_grid_numba,
            transient_solve_grid_gear2_numba,
            transient_solve_adaptive_gear2_numba,
        )
    except Exception:
        terminal_derivatives_numba = None
        transient_newton_numba = None
        transient_solve_grid_numba = None
        transient_solve_grid_gear2_numba = None
        transient_solve_adaptive_gear2_numba = None


_CAP_MODE = os.environ.get("CIRCUIT_PMOS_TRANSIENT_CAP_MODE", "charge").lower()
_USE_CHARGE_CAPS = _CAP_MODE in {"charge", "q", "qstamp", "q-stamp"}
_USE_AVERAGE_CAPS = _CAP_MODE in {"average", "avg", "trapezoid", "trap"}
_USE_BRANCH_CAPS = _CAP_MODE in {"branch", "self", "self-charge"}
# "endpoint" (a.k.a. legacy "veriloga"): the literal C(Vn)*dV endpoint companion.
# Non-conservative like Cadence's ddt(), but UNSTABLE in the shooting (non-monotone
# period map) -- kept for experiments only; the STABLE non-conservative operator is
# "average" (trapezoidal). Anything unrecognized falls back to charge (id 0, L-stable).
_USE_ENDPOINT_CAPS = _CAP_MODE in {"endpoint", "veriloga"}
_CAP_MODE_ID = (3 if _USE_BRANCH_CAPS else 1 if _USE_AVERAGE_CAPS
                else 2 if _USE_ENDPOINT_CAPS else 0)

# Use the numba gear2 grid for periodic/PSS and raw maxstep/retry transients.
# Set CIRCUIT_GEAR2_NUMBA=0 to fall back to the verified pure-Python gear2 loop.
_GEAR2_NUMBA_GRID = os.environ.get("CIRCUIT_GEAR2_NUMBA", "1").lower() in {"1", "true", "on"}


_CAP_MODE_IDS = {
    "charge": 0, "q": 0, "qstamp": 0, "q-stamp": 0,
    "average": 1, "avg": 1, "trapezoid": 1, "trap": 1,
    "endpoint": 2, "veriloga": 2,
    "branch": 3, "self": 3, "self-charge": 3,
}


@dataclass
class _PathOutcome:
    handled: bool = False
    Vhist: object = None
    tgrid: object = None
    input_values: object = None
    N: object = None
    nfail: object = None
    nretry: object = None
    nsubsteps: object = None
    gear2_done: bool = False
    gear2_numba_used: bool = False
    gear2_python_retry_used: bool = False
    adaptive_used: bool = False
    adaptive_numba_used: bool = False
    used_grid_numba: bool = False
    partial_grid_numba: bool = False
    python_start_idx: object = None
    profile_wall_s: object = None
    profile_stats: object = None
    numba_grid_error: object = None
    numba_grid_failed_index: object = None
    numba_grid_failed_substeps: object = None
    numba_grid_failed_profile: object = None
    numba_grid_failed_intervals: object = None


def _cap_mode_to_id(cap_mode):
    if cap_mode is None:
        return None
    key = str(cap_mode).lower()
    if key not in _CAP_MODE_IDS:
        raise ValueError(f"unknown cap_mode {cap_mode!r}; expected one of {sorted(_CAP_MODE_IDS)}")
    return _CAP_MODE_IDS[key]


def _term_arrays(terms):
    kind = np.empty(len(terms), dtype=np.int64)
    ref = np.empty(len(terms), dtype=np.int64)
    value = np.empty(len(terms), dtype=float)
    for pos, term in enumerate(terms):
        kind[pos] = int(term[0])
        if term[0] in (0, 1):
            ref[pos] = int(term[1])
            value[pos] = 0.0
        else:
            ref[pos] = 0
            value[pos] = float(term[1])
    return kind, ref, value


def _index_array(vals):
    return np.array([-1 if val is None else int(val) for val in vals],
                    dtype=np.int64)


def _k_device_states(ctx, V, input_vals):
    """Per-Newton operating data shared by residual and Jacobian."""
    dev_meta = ctx.dev_meta
    termv = ctx.termv
    out = [None] * len(dev_meta)
    for pos, (dev, signed, dterm, gterm, sterm, di, gi, si) in enumerate(dev_meta):
        Vs = termv(sterm, V, input_vals)
        Vd = termv(dterm, V, input_vals)
        Vg = termv(gterm, V, input_vals)
        try:
            Vs1, Vd1 = dev.get_op(Vs, Vd, Vg)
            _, _, I_d1_d, _, _ = dev._eval_currents(Vs, Vd, Vg, Vs1, Vd1)
            Idc = -I_d1_d
            Cgs, Cgd = dev._capacitances_from_op(Vs, Vd, Vg, Vs1, Vd1)
            I = I_d1_d if signed else abs(Idc)   # signed (always; see dev_meta)
        except Exception:
            I = Cgs = Cgd = 0.0
            Vs1 = Vd1 = 0.0
        out[pos] = (dev, signed, di, gi, si, Vs, Vd, Vg, Vs1, Vd1, I, Cgs, Cgd)
    return out


def _k_device_prev_cap_terms(ctx, V, input_vals):
    dev_meta = ctx.dev_meta
    termv = ctx.termv
    _cap_id = ctx.cap_id
    out = []
    for dev, _, dterm, gterm, sterm, *_ in dev_meta:
        Vs = termv(sterm, V, input_vals)
        Vd = termv(dterm, V, input_vals)
        Vg = termv(gterm, V, input_vals)
        try:
            Vs1, Vd1 = dev.get_op(Vs, Vd, Vg)
            if _cap_id == 3:
                qgs, qgd, _, _, Cgs, Cgd = dev._capacitance_branch_terms_from_op(
                    Vs, Vd, Vg, Vs1, Vd1)
            else:
                qgs, qgd, Cgs, Cgd = dev._capacitance_charges_from_op(
                    Vs, Vd, Vg, Vs1, Vd1)
            if _cap_id == 1:
                qgs, qgd = Cgs, Cgd
        except Exception:
            qgs = qgd = 0.0
        out.append((Vs, Vd, Vg, qgs, qgd))
    return out


def _k_load_cap_dv_values(ctx, V, input_vals):
    termv = ctx.termv
    return [termv(aterm, V, input_vals) - termv(bterm, V, input_vals)
            for aterm, bterm, _, _, _ in ctx.load_meta]


def _k_step_residual(ctx, V, input_now, states, prev_dev_terms, load_prev_dv, h,
                     cap_coeffs=(1.0, -1.0, 0.0), prev2_dev_terms=None,
                     load_prev2_dv=None):
    # cap_coeffs = (a0, a1, a2): backward-Euler is (1, -1, 0); variable-step
    # BDF2/gear2 passes (a0, a1, a2) with a2 weighting the n-2 history.  Only
    # the charge-mode (q-stamp) caps and the linear/load caps use BDF2 here.
    ca0, ca1, ca2 = cap_coeffs
    R = np.zeros(ctx.n_aug)
    inv_h = 1.0 / h
    termv = ctx.termv
    # device DC currents (into-node convention: +Id at drain, -Id at source)
    for _, _, di, _, si, _, _, _, _, _, I, _, _ in states:
        if di is not None:
            R[di] += I
        if si is not None:
            R[si] -= I
    for aterm, bterm, ai, bi, gval in ctx.res_meta:
        i_ab = (termv(aterm, V, input_now) - termv(bterm, V, input_now)) * gval
        if ai is not None:
            R[ai] -= i_ab
        if bi is not None:
            R[bi] += i_ab
    for pi, qi, Ival in ctx.isrc_meta:
        if pi is not None:
            R[pi] -= Ival
        if qi is not None:
            R[qi] += Ival
    for pi, qi, cpterm, cnterm, gm in ctx.vccs_meta:
        i_vccs = (termv(cpterm, V, input_now) - termv(cnterm, V, input_now)) * gm
        if pi is not None:
            R[pi] += i_vccs
        if qi is not None:
            R[qi] -= i_vccs
    for pi, qi, key_idx in ctx.dyn_isrc_meta:
        Ival = input_now[key_idx]
        if pi is not None:
            R[pi] -= Ival
        if qi is not None:
            R[qi] += Ival
    for aterm, bterm, pi, qi, bi, e_const, e_idx in ctx.vs_meta:
        ibr = V[bi]
        if pi is not None:
            R[pi] -= ibr
        if qi is not None:
            R[qi] += ibr
        E = e_const if e_idx < 0 else input_now[e_idx]
        R[bi] = termv(aterm, V, input_now) - termv(bterm, V, input_now) - E
    for aterm, bterm, cpterm, cnterm, pi, qi, cpi, cni, bi, mu in ctx.vcvs_meta:
        ibr = V[bi]
        if pi is not None:
            R[pi] -= ibr
        if qi is not None:
            R[qi] += ibr
        R[bi] = (termv(aterm, V, input_now) - termv(bterm, V, input_now)
                 - mu * (termv(cpterm, V, input_now) - termv(cnterm, V, input_now)))
    for pi, qi, ctrl_bi, beta in ctx.cccs_meta:
        I_out = beta * V[ctrl_bi]
        if pi is not None:
            R[pi] += I_out
        if qi is not None:
            R[qi] -= I_out
    for aterm, bterm, pi, qi, bi, ctrl_bi, gamma in ctx.ccvs_meta:
        ibr = V[bi]
        if pi is not None:
            R[pi] -= ibr
        if qi is not None:
            R[qi] += ibr
        R[bi] = (termv(aterm, V, input_now) - termv(bterm, V, input_now)
                 - gamma * V[ctrl_bi])
    for k in range(ctx.n):
        R[k] -= V[k] * ctx.gmin

    p2_terms = prev2_dev_terms if prev2_dev_terms is not None else prev_dev_terms
    for state, prev_terms, prev2_terms in zip(states, prev_dev_terms, p2_terms):
        _, _, di, gi, si, Vs, Vd, Vg, _, _, _, Cgs, Cgd = state
        pVs, pVd, pVg, pQgs, pQgd = prev_terms
        ppQgs, ppQgd = prev2_terms[3], prev2_terms[4]
        gate_leak_g = 1.0 / state[0].R_cap2
        if gate_leak_g != 0.0:
            i_sg = (Vs - Vg) * gate_leak_g
            if si is not None:
                R[si] -= i_sg
            if gi is not None:
                R[gi] += i_sg
            i_dg = (Vd - Vg) * gate_leak_g
            if di is not None:
                R[di] -= i_dg
            if gi is not None:
                R[gi] += i_dg
        if Cgs != 0.0:
            if ctx.cap_id == 1:
                i_ab = 0.5 * (Cgs + pQgs) * ((Vg - Vs) - (pVg - pVs)) * inv_h
            elif ctx.cap_id == 2:
                i_ab = Cgs * ((Vg - Vs) - (pVg - pVs)) * inv_h
            elif ctx.cap_id == 3:
                qgs_self, _, cgs_cross, _, _, _ = state[0]._capacitance_branch_terms_from_op(
                    Vs, Vd, Vg, state[8], state[9])
                i_ab = ((qgs_self - pQgs) +
                        cgs_cross * ((Vg - Vs) - (pVg - pVs))) * inv_h
            else:
                qgs = state[0]._capacitance_charges_from_op(
                    Vs, Vd, Vg, state[8], state[9])[0]
                i_ab = (ca0 * qgs + ca1 * pQgs + ca2 * ppQgs) * inv_h
            if gi is not None:
                R[gi] -= i_ab
            if si is not None:
                R[si] += i_ab
        if Cgd != 0.0:
            if ctx.cap_id == 1:
                i_ab = 0.5 * (Cgd + pQgd) * ((Vg - Vd) - (pVg - pVd)) * inv_h
            elif ctx.cap_id == 2:
                i_ab = Cgd * ((Vg - Vd) - (pVg - pVd)) * inv_h
            elif ctx.cap_id == 3:
                _, qgd_self, _, cgd_cross, _, _ = state[0]._capacitance_branch_terms_from_op(
                    Vs, Vd, Vg, state[8], state[9])
                i_ab = ((qgd_self - pQgd) +
                        cgd_cross * ((Vg - Vd) - (pVg - pVd))) * inv_h
            else:
                qgd = state[0]._capacitance_charges_from_op(
                    Vs, Vd, Vg, state[8], state[9])[1]
                i_ab = (ca0 * qgd + ca1 * pQgd + ca2 * ppQgd) * inv_h
            if gi is not None:
                R[gi] -= i_ab
            if di is not None:
                R[di] += i_ab
    load_p2 = load_prev2_dv if load_prev2_dv is not None else load_prev_dv
    for (aterm, bterm, ai, bi, cap), dv_pre, dv_pre2 in zip(
            ctx.load_meta, load_prev_dv, load_p2):
        if cap != 0.0:
            dv_now = termv(aterm, V, input_now) - termv(bterm, V, input_now)
            i_ab = cap * inv_h * (ca0 * dv_now + ca1 * dv_pre + ca2 * dv_pre2)
            if ai is not None:
                R[ai] -= i_ab
            if bi is not None:
                R[bi] += i_ab
    return R


def _k_terminal_derivatives(ctx, dev, signed, Vs, Vd, Vg, Vs1, Vd1,
                            need_gm, need_gds):
    """Terminal gm/gds via implicit differentiation of the solved internal nodes."""
    if not need_gm and not need_gds:
        return 0.0, 0.0
    if terminal_derivatives_numba is not None:
        try:
            ok, gm, gds = terminal_derivatives_numba(
                Vs, Vd, Vg, Vs1, Vd1, need_gm, need_gds, not signed, ctx.HH, 1e-6,
                dev.Vfb, dev.Vss, dev.Lc, dev.lambda_, dev._contact_scale,
                dev._channel_exponent, dev._current_scale, dev._inv_Rleak)
            if ok:
                return gm, gds
        except Exception:
            pass
    hx = 1e-6

    def eval_at(vs, vd, vg, xs1, xd1):
        I_s_s1, I_s1_d1, I_d1_d, _, _ = dev._eval_currents(vs, vd, vg, xs1, xd1)
        return I_s_s1 - I_s1_d1, I_s1_d1 - I_d1_d, -I_d1_d

    F0a, F0b, Idc0 = eval_at(Vs, Vd, Vg, Vs1, Vd1)
    if (not signed) and abs(Idc0) < 1e-30:
        raise FloatingPointError("abs(Idc) kink")
    Fpa, Fpb, Ip = eval_at(Vs, Vd, Vg, Vs1 + hx, Vd1)
    j00 = (Fpa - F0a) / hx
    j10 = (Fpb - F0b) / hx
    ix0 = (Ip - Idc0) / hx
    Fpa, Fpb, Ip = eval_at(Vs, Vd, Vg, Vs1, Vd1 + hx)
    j01 = (Fpa - F0a) / hx
    j11 = (Fpb - F0b) / hx
    ix1 = (Ip - Idc0) / hx
    det = j00 * j11 - j01 * j10
    if det == 0.0 or not np.isfinite(det):
        raise np.linalg.LinAlgError("singular internal Jacobian")

    sign = 1.0 if Idc0 > 0.0 else -1.0
    current_sign = -1.0 if signed else sign

    def deriv(vs_p, vd_p, vg_p, vs_m, vd_m, vg_m):
        Fpa, Fpb, Ip = eval_at(vs_p, vd_p, vg_p, Vs1, Vd1)
        Fma, Fmb, Im = eval_at(vs_m, vd_m, vg_m, Vs1, Vd1)
        fu0 = (Fpa - Fma) / (2 * ctx.HH)
        fu1 = (Fpb - Fmb) / (2 * ctx.HH)
        Iu = (Ip - Im) / (2 * ctx.HH)
        y0 = (j11 * fu0 - j01 * fu1) / det
        y1 = (-j10 * fu0 + j00 * fu1) / det
        return current_sign * (Iu - ix0 * y0 - ix1 * y1)

    gm = deriv(Vs, Vd, Vg + ctx.HH, Vs, Vd, Vg - ctx.HH) if need_gm else 0.0
    gds = deriv(Vs, Vd + ctx.HH, Vg, Vs, Vd - ctx.HH, Vg) if need_gds else 0.0
    return gm, gds


def _k_build_jac(ctx, V, states, prev_dev_terms, h, cap_a0=1.0):
    """Analytic conductance Jacobian dR/dV."""
    J = np.zeros((ctx.n_aug, ctx.n_aug))
    inv_h = 1.0 / h
    for dev, signed, di, gi, si, Vs, Vd, Vg, Vs1, Vd1, _, _, _ in states:
        get_idc = dev.get_Idc
        need_gm = gi is not None or si is not None
        need_gds = di is not None or si is not None
        try:
            gm, gds = _k_terminal_derivatives(
                ctx, dev, signed, Vs, Vd, Vg, Vs1, Vd1, need_gm, need_gds)
        except Exception:
            try:
                current = (lambda vs, vd, vg: -get_idc(vs, vd, vg)) if signed else (
                    lambda vs, vd, vg: abs(get_idc(vs, vd, vg)))
                gm = ((current(Vs, Vd, Vg + ctx.HH) -
                       current(Vs, Vd, Vg - ctx.HH)) / (2 * ctx.HH)
                      if need_gm else 0.0)
                gds = ((current(Vs, Vd + ctx.HH, Vg) -
                        current(Vs, Vd - ctx.HH, Vg)) / (2 * ctx.HH)
                       if need_gds else 0.0)
            except Exception:
                gm, gds = 0.0, 1e-12 if need_gds else 0.0
        dI_dVs = -(gm + gds)
        if di is not None:
            J[di, di] += gds
            if gi is not None:
                J[di, gi] += gm
            if si is not None:
                J[di, si] += dI_dVs
        if si is not None:
            if di is not None:
                J[si, di] -= gds
            if gi is not None:
                J[si, gi] -= gm
            J[si, si] -= dI_dVs
    for k in range(ctx.n):
        J[k, k] -= ctx.gmin
    for state, _prev_terms in zip(states, prev_dev_terms):
        dev, _, di, gi, si, _, _, _, _, _, _, Cgs, Cgd = state
        gate_leak_g = 1.0 / dev.R_cap2
        if gate_leak_g != 0.0:
            if si is not None:
                J[si, si] -= gate_leak_g
                if gi is not None:
                    J[si, gi] += gate_leak_g
            if di is not None:
                J[di, di] -= gate_leak_g
                if gi is not None:
                    J[di, gi] += gate_leak_g
            if gi is not None:
                if si is not None:
                    J[gi, si] += gate_leak_g
                if di is not None:
                    J[gi, di] += gate_leak_g
                J[gi, gi] -= gate_leak_g * (
                    (1 if si is not None else 0) +
                    (1 if di is not None else 0))
        if Cgs != 0.0:
            g = cap_a0 * Cgs * inv_h
            if gi is not None:
                J[gi, gi] -= g
                if si is not None:
                    J[gi, si] += g
            if si is not None:
                J[si, si] -= g
                if gi is not None:
                    J[si, gi] += g
        if Cgd != 0.0:
            g = cap_a0 * Cgd * inv_h
            if gi is not None:
                J[gi, gi] -= g
                if di is not None:
                    J[gi, di] += g
            if di is not None:
                J[di, di] -= g
                if gi is not None:
                    J[di, gi] += g
    for _, _, ai, bi, cap in ctx.load_meta:
        if cap != 0.0:
            g = cap_a0 * cap * inv_h
            if ai is not None:
                J[ai, ai] -= g
                if bi is not None:
                    J[ai, bi] += g
            if bi is not None:
                J[bi, bi] -= g
                if ai is not None:
                    J[bi, ai] += g
    for _, _, ai, bi, gval in ctx.res_meta:
        if gval != 0.0:
            if ai is not None:
                J[ai, ai] -= gval
                if bi is not None:
                    J[ai, bi] += gval
            if bi is not None:
                J[bi, bi] -= gval
                if ai is not None:
                    J[bi, ai] += gval
    for pi, qi, cpi, cni, gm in zip(ctx.vccs_pi, ctx.vccs_qi,
                                    ctx.vccs_cpi, ctx.vccs_cni, ctx.vccs_gm):
        if pi is not None and cpi is not None:
            J[pi, cpi] += gm
        if pi is not None and cni is not None:
            J[pi, cni] -= gm
        if qi is not None and cpi is not None:
            J[qi, cpi] -= gm
        if qi is not None and cni is not None:
            J[qi, cni] += gm
    for _aterm, _bterm, pi, qi, bi, _e_const, _e_idx in ctx.vs_meta:
        if pi is not None:
            J[pi, bi] -= 1.0
            J[bi, pi] += 1.0
        if qi is not None:
            J[qi, bi] += 1.0
            J[bi, qi] -= 1.0
    for _aterm, _bterm, _cpterm, _cnterm, pi, qi, cpi, cni, bi, mu in ctx.vcvs_meta:
        if pi is not None:
            J[pi, bi] -= 1.0
            J[bi, pi] += 1.0
        if qi is not None:
            J[qi, bi] += 1.0
            J[bi, qi] -= 1.0
        if cpi is not None:
            J[bi, cpi] -= mu
        if cni is not None:
            J[bi, cni] += mu
    for pi, qi, ctrl_bi, beta in ctx.cccs_meta:
        if pi is not None:
            J[pi, ctrl_bi] += beta
        if qi is not None:
            J[qi, ctrl_bi] -= beta
    for _aterm, _bterm, pi, qi, bi, ctrl_bi, gamma in ctx.ccvs_meta:
        if pi is not None:
            J[pi, bi] -= 1.0
            J[bi, pi] += 1.0
        if qi is not None:
            J[qi, bi] += 1.0
            J[bi, qi] -= 1.0
        J[bi, ctrl_bi] -= gamma
    return J


def _k_newton(ctx, seed, Vp, input_now, input_prev, h, maxit=None, vtol=1e-8):
    """Full-step Newton with the analytic Jacobian, converged on step size."""
    maxit = int(ctx.newton_maxit if maxit is None else maxit)
    step_limit = float(ctx.newton_step_limit)
    if ctx.use_numba_newton:
        ctx.numba_newton_attempts += 1
        try:
            Vn, iters, ok, usable = transient_newton_numba(
                np.asarray(seed, float), np.asarray(Vp, float),
                np.asarray(input_now, float), np.asarray(input_prev, float),
                float(h), int(ctx.n), maxit, step_limit, float(vtol),
                float(ctx.gmin),
                bool(ctx.fallback_full_jacobian or ctx.fallback_least_squares),
                float(ctx.fallback_tol), float(ctx.HH),
                ctx.dev_d_kind, ctx.dev_d_ref, ctx.dev_d_val,
                ctx.dev_g_kind, ctx.dev_g_ref, ctx.dev_g_val,
                ctx.dev_s_kind, ctx.dev_s_ref, ctx.dev_s_val,
                ctx.dev_di, ctx.dev_gi, ctx.dev_si, ctx.dev_use_abs,
                ctx.p_Vfb, ctx.p_Vss, ctx.p_Lc, ctx.p_lambda,
                ctx.p_contact_scale, ctx.p_exponent,
                ctx.p_current_scale, ctx.p_inv_Rleak,
                ctx.p_two_over_pi, ctx.p_cap_cgs1, ctx.p_cap_cgd1,
                ctx.p_cap_half_wl_ci, ctx.p_cap_cgs3_base,
                ctx.p_cap_cgd3_base, ctx.p_k1, ctx.p_gate_leak_g,
                ctx.op_cache_valid, ctx.op_cache_vs1, ctx.op_cache_vd1,
                ctx.res_a_kind, ctx.res_a_ref, ctx.res_a_val,
                ctx.res_b_kind, ctx.res_b_ref, ctx.res_b_val,
                ctx.res_ai, ctx.res_bi, ctx.res_g,
                ctx.cap_a_kind, ctx.cap_a_ref, ctx.cap_a_val,
                ctx.cap_b_kind, ctx.cap_b_ref, ctx.cap_b_val,
                ctx.cap_ai, ctx.cap_bi, ctx.cap_value,
                ctx.isrc_pi, ctx.isrc_qi, ctx.isrc_value,
                ctx.dyn_pi, ctx.dyn_qi, ctx.dyn_input_idx,
                int(ctx.cap_id),
                float(ctx.clip_lo), float(ctx.clip_hi),
            )
            if ok:
                ctx.numba_newton_success += 1
                return Vn, int(iters), True
            if usable:
                seed = Vn
        except Exception:
            pass
        ctx.numba_newton_fallback += 1

    V = np.array(seed, float)
    prev = np.inf
    prev_dev_terms = _k_device_prev_cap_terms(ctx, Vp, input_prev)
    load_prev_dv = _k_load_cap_dv_values(ctx, Vp, input_prev)
    for it in range(maxit):
        states = _k_device_states(ctx, V, input_now)
        R = _k_step_residual(ctx, V, input_now, states, prev_dev_terms,
                             load_prev_dv, h)
        if ((ctx.fallback_full_jacobian or ctx.fallback_least_squares) and
                np.linalg.norm(R, ord=np.inf) < float(ctx.fallback_tol)):
            return V, it + 1, True
        J = _k_build_jac(ctx, V, states, prev_dev_terms, h)
        try:
            dV = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            dV = np.linalg.lstsq(J, -R, rcond=None)[0]
        mx = np.max(np.abs(dV))
        if mx > step_limit:
            dV *= step_limit / mx
            mx = step_limit
        V = V + dV
        if ctx.clip_lo <= ctx.clip_hi:
            V = np.clip(V, ctx.clip_lo, ctx.clip_hi)
        if mx < vtol:
            if ctx.fallback_full_jacobian or ctx.fallback_least_squares:
                continue
            return V, it + 1, True
        if it >= 4 and mx >= prev and mx < 1e-5:
            if ctx.fallback_full_jacobian or ctx.fallback_least_squares:
                continue
            return V, it + 1, True
        prev = mx
    return V, maxit, False


def _k_newton_full_jac(ctx, seed, Vp, input_now, input_prev, h,
                       maxit=6, residual_tol=1e-8, vtol=1e-8):
    """Fallback Newton using a finite-difference Jacobian of the full residual."""
    V = np.array(seed, float)
    prev_dev_terms = _k_device_prev_cap_terms(ctx, Vp, input_prev)
    load_prev_dv = _k_load_cap_dv_values(ctx, Vp, input_prev)
    step_limit = float(ctx.newton_step_limit)

    def residual_at(z):
        states = _k_device_states(ctx, z, input_now)
        return _k_step_residual(ctx, z, input_now, states, prev_dev_terms,
                                load_prev_dv, h)

    for _ in range(int(maxit)):
        R = residual_at(V)
        norm = np.linalg.norm(R, ord=np.inf)
        if norm < residual_tol:
            return V, True
        J = np.zeros((ctx.n_aug, ctx.n_aug))
        for col in range(ctx.n_aug):
            eps = 1e-5 * max(1.0, abs(V[col]))
            zp = V.copy()
            zp[col] += eps
            J[:, col] = (residual_at(zp) - R) / eps
        try:
            dV = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            dV = np.linalg.lstsq(J, -R, rcond=None)[0]
        mx = np.max(np.abs(dV))
        if mx > step_limit:
            dV *= step_limit / mx
            mx = step_limit
        accepted = False
        for lam in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = V + lam * dV
            trial_norm = np.linalg.norm(residual_at(trial), ord=np.inf)
            if trial_norm < norm or trial_norm < residual_tol:
                V = trial
                accepted = True
                if trial_norm < residual_tol:
                    return V, True
                if lam * mx < vtol:
                    return V, False
                break
        if not accepted:
            return V, False
    return V, np.linalg.norm(residual_at(V), ord=np.inf) < residual_tol


def _k_gear2_step(ctx, seed, Vp, Vp2, input_now, input_prev, input_prev2,
                  h_n, h_prev, maxit, step_limit, vtol):
    """One variable-step BDF2 step with BE self-start."""
    if (Vp2 is None or h_prev is None or h_prev <= 0.0 or
            h_n / h_prev > 2.0):
        a0, a1, a2 = 1.0, -1.0, 0.0
        prev2_terms = None
        load_prev2 = None
    else:
        rho = h_n / h_prev
        a0 = (1.0 + 2.0 * rho) / (1.0 + rho)
        a1 = -(1.0 + rho)
        a2 = (rho * rho) / (1.0 + rho)
        prev2_terms = _k_device_prev_cap_terms(ctx, Vp2, input_prev2)
        load_prev2 = _k_load_cap_dv_values(ctx, Vp2, input_prev2)
    prev_terms = _k_device_prev_cap_terms(ctx, Vp, input_prev)
    load_prev = _k_load_cap_dv_values(ctx, Vp, input_prev)
    coeffs = (a0, a1, a2)
    V = np.array(seed, float)
    prevmx = np.inf
    for it in range(int(maxit)):
        states = _k_device_states(ctx, V, input_now)
        R = _k_step_residual(ctx, V, input_now, states, prev_terms, load_prev, h_n,
                             cap_coeffs=coeffs, prev2_dev_terms=prev2_terms,
                             load_prev2_dv=load_prev2)
        J = _k_build_jac(ctx, V, states, prev_terms, h_n, cap_a0=a0)
        try:
            dV = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            dV = np.linalg.lstsq(J, -R, rcond=None)[0]
        mx = float(np.max(np.abs(dV)))
        if mx > step_limit:
            dV *= step_limit / mx
            mx = step_limit
        V = V + dV
        if ctx.clip_lo <= ctx.clip_hi:
            V = np.clip(V, ctx.clip_lo, ctx.clip_hi)
        if mx < vtol:
            return V, it + 1, True
        if it >= 4 and mx >= prevmx and mx < 1e-5:
            return V, it + 1, True
        prevmx = mx
    return V, int(maxit), False


def _k_input_at_between(input_a, input_b, frac):
    return input_a + (input_b - input_a) * frac


def _k_gear2_terms(ctx, Vp, Vp2, input_prev, input_prev2, h_n, h_prev):
    if (Vp2 is None or h_prev is None or h_prev <= 0.0 or
            h_n / h_prev > 2.0):
        coeffs = (1.0, -1.0, 0.0)
        prev2_terms = None
        load_prev2 = None
    else:
        rho = h_n / h_prev
        coeffs = (
            (1.0 + 2.0 * rho) / (1.0 + rho),
            -(1.0 + rho),
            (rho * rho) / (1.0 + rho),
        )
        prev2_terms = _k_device_prev_cap_terms(ctx, Vp2, input_prev2)
        load_prev2 = _k_load_cap_dv_values(ctx, Vp2, input_prev2)
    prev_terms = _k_device_prev_cap_terms(ctx, Vp, input_prev)
    load_prev = _k_load_cap_dv_values(ctx, Vp, input_prev)
    return coeffs, prev_terms, load_prev, prev2_terms, load_prev2


def _k_gear2_full_jac(ctx, seed, Vp, Vp2, input_now, input_prev, input_prev2,
                      h_n, h_prev, maxit=6, residual_tol=1e-8, vtol=1e-8):
    V = np.array(seed, float)
    coeffs, prev_terms, load_prev, prev2_terms, load_prev2 = _k_gear2_terms(
        ctx, Vp, Vp2, input_prev, input_prev2, h_n, h_prev)
    step_limit = float(ctx.newton_step_limit)

    def residual_at(z):
        states = _k_device_states(ctx, z, input_now)
        return _k_step_residual(ctx, z, input_now, states, prev_terms, load_prev, h_n,
                                cap_coeffs=coeffs, prev2_dev_terms=prev2_terms,
                                load_prev2_dv=load_prev2)

    for _ in range(int(maxit)):
        R = residual_at(V)
        norm = np.linalg.norm(R, ord=np.inf)
        if norm < residual_tol:
            return V, True
        J = np.zeros((ctx.n_aug, ctx.n_aug))
        for col in range(ctx.n_aug):
            eps = 1e-5 * max(1.0, abs(V[col]))
            zp = V.copy()
            zp[col] += eps
            J[:, col] = (residual_at(zp) - R) / eps
        try:
            dV = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            dV = np.linalg.lstsq(J, -R, rcond=None)[0]
        mx = np.max(np.abs(dV))
        if mx > step_limit:
            dV *= step_limit / mx
            mx = step_limit
        accepted = False
        for lam in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = V + lam * dV
            trial_norm = np.linalg.norm(residual_at(trial), ord=np.inf)
            if trial_norm < norm or trial_norm < residual_tol:
                V = trial
                accepted = True
                if trial_norm < residual_tol:
                    return V, True
                if lam * mx < vtol:
                    return V, False
                break
        if not accepted:
            return V, False
    return V, np.linalg.norm(residual_at(V), ord=np.inf) < residual_tol


def _k_gear2_try_step(ctx, Vp, Vp2, input_prev, input_prev2, input_now,
                      h_n, h_prev, use_fallback=True):
    raised = False
    try:
        V, _, ok = _k_gear2_step(ctx, Vp, Vp, Vp2, input_now, input_prev,
                                 input_prev2, h_n, h_prev, ctx.newton_maxit,
                                 ctx.newton_step_limit, float(ctx.newton_vtol))
    except Exception:
        V, ok, raised = Vp, False, True
    if ok:
        return V, True, raised
    if not use_fallback:
        return V, False, raised
    if ctx.fallback_full_jacobian:
        for seed in (V, Vp):
            try:
                Vn, okn = _k_gear2_full_jac(
                    ctx, seed, Vp, Vp2, input_now, input_prev, input_prev2,
                    h_n, h_prev, residual_tol=float(ctx.fallback_tol),
                    vtol=float(ctx.newton_vtol))
                if okn:
                    return Vn, True, raised
            except Exception:
                raised = True
    if not ctx.fallback_least_squares:
        return V, False, raised
    try:
        from scipy.optimize import least_squares
        rails = [v for v in ctx.plan.rails.values() if isinstance(v, (int, float))]
        if not rails:
            return V, False, raised
        lo = min(rails) - 0.5
        hi = max(rails) + 0.5
        if ctx.n_aug > ctx.n:
            lo = np.concatenate([np.full(ctx.n, lo), np.full(ctx.n_aug - ctx.n, -np.inf)])
            hi = np.concatenate([np.full(ctx.n, hi), np.full(ctx.n_aug - ctx.n, np.inf)])
        coeffs, prev_terms, load_prev, prev2_terms, load_prev2 = _k_gear2_terms(
            ctx, Vp, Vp2, input_prev, input_prev2, h_n, h_prev)

        def residual(z):
            states = _k_device_states(ctx, z, input_now)
            return _k_step_residual(ctx, z, input_now, states, prev_terms, load_prev, h_n,
                                    cap_coeffs=coeffs, prev2_dev_terms=prev2_terms,
                                    load_prev2_dv=load_prev2)

        seed = np.clip(V, lo, hi)
        sol = least_squares(residual, seed, bounds=(lo, hi), x_scale="jac",
                            xtol=1e-11, ftol=1e-11, gtol=1e-11,
                            max_nfev=80)
        norm = np.linalg.norm(residual(sol.x), ord=np.inf)
        if norm < ctx.fallback_tol:
            return sol.x, True, raised
        return Vp, False, raised
    except Exception:
        return Vp, False, True


def _k_solve_gear2_chunk(ctx, Vp, Vp2, input_prev, input_prev2,
                         h_n, h_prev, input_now, depth=0):
    use_fallback = depth >= ctx.max_retry_subdivisions
    V, ok, _ = _k_gear2_try_step(ctx, Vp, Vp2, input_prev, input_prev2,
                                 input_now, h_n, h_prev,
                                 use_fallback=use_fallback)
    if ok:
        return V, Vp, input_prev, h_n, True, 1, 0
    if depth < ctx.max_retry_subdivisions and h_n > 0.0:
        mid_input = _k_input_at_between(input_prev, input_now, 0.5)
        left_v, left_prev2, left_input_prev2, left_hprev, left_ok, left_steps, left_retry = (
            _k_solve_gear2_chunk(ctx, Vp, Vp2, input_prev, input_prev2,
                                 0.5 * h_n, h_prev, mid_input, depth + 1)
        )
        if not left_ok:
            return Vp, Vp2, input_prev2, h_prev, False, left_steps, (
                left_retry + (1 if depth == 0 else 0))
        right_v, right_prev2, right_input_prev2, right_hprev, right_ok, right_steps, right_retry = (
            _k_solve_gear2_chunk(ctx, left_v, left_prev2, mid_input,
                                 left_input_prev2, 0.5 * h_n, left_hprev,
                                 input_now, depth + 1)
        )
        if right_ok:
            return (right_v, right_prev2, right_input_prev2, right_hprev,
                    True, left_steps + right_steps, left_retry + right_retry + 1)
        return Vp, Vp2, input_prev2, h_prev, False, (
            left_steps + right_steps), left_retry + right_retry + (1 if depth == 0 else 0)
    return Vp, Vp2, input_prev2, h_prev, False, 1, 1 if depth == 0 else 0


def _k_try_step(ctx, Vp, input_prev, input_now, h, use_fallback=True):
    V = Vp
    ok = False
    raised = False
    try:
        V, _, ok = _k_newton(ctx, Vp, Vp, input_now, input_prev, h,
                             vtol=float(ctx.newton_vtol))
    except Exception:
        raised = True
    if ok:
        return V, True, raised
    if not use_fallback:
        return Vp, False, raised
    if ctx.fallback_full_jacobian:
        for seed in (V, Vp):
            try:
                Vn, okn = _k_newton_full_jac(
                    ctx, seed, Vp, input_now, input_prev, h,
                    residual_tol=float(ctx.fallback_tol),
                    vtol=float(ctx.newton_vtol))
                if okn:
                    return Vn, True, raised
            except Exception:
                raised = True
    if not ctx.fallback_least_squares:
        return V, bool(ok), raised
    try:
        from scipy.optimize import least_squares
        rails = [v for v in ctx.plan.rails.values() if isinstance(v, (int, float))]
        if not rails:
            return V, False, raised
        lo = min(rails) - 0.5
        hi = max(rails) + 0.5
        if ctx.n_aug > ctx.n:
            lo = np.concatenate([np.full(ctx.n, lo), np.full(ctx.n_aug - ctx.n, -np.inf)])
            hi = np.concatenate([np.full(ctx.n, hi), np.full(ctx.n_aug - ctx.n, np.inf)])
        prev_dev_terms = _k_device_prev_cap_terms(ctx, Vp, input_prev)
        load_prev_dv = _k_load_cap_dv_values(ctx, Vp, input_prev)

        def residual(z):
            states = _k_device_states(ctx, z, input_now)
            return _k_step_residual(ctx, z, input_now, states, prev_dev_terms,
                                    load_prev_dv, h)

        seed = np.clip(V, lo, hi)
        sol = least_squares(residual, seed, bounds=(lo, hi), x_scale="jac",
                            xtol=1e-11, ftol=1e-11, gtol=1e-11,
                            max_nfev=80)
        norm = np.linalg.norm(residual(sol.x), ord=np.inf)
        if norm < ctx.fallback_tol:
            return sol.x, True, raised
        return Vp, False, raised
    except Exception:
        return Vp, False, True


def _k_solve_chunk(ctx, Vp, input_prev, input_now, h, depth=0):
    use_fallback = depth >= ctx.max_retry_subdivisions
    V, ok, _raised = _k_try_step(ctx, Vp, input_prev, input_now, h,
                                 use_fallback=use_fallback)
    if ok:
        return V, True, 1, 0
    if depth < ctx.max_retry_subdivisions and h > 0.0:
        mid_input = _k_input_at_between(input_prev, input_now, 0.5)
        left_v, left_ok, left_steps, left_retry = _k_solve_chunk(
            ctx, Vp, input_prev, mid_input, 0.5 * h, depth + 1)
        if not left_ok:
            return Vp, False, left_steps, left_retry + (1 if depth == 0 else 0)
        right_v, right_ok, right_steps, right_retry = _k_solve_chunk(
            ctx, left_v, mid_input, input_now, 0.5 * h, depth + 1)
        if left_ok and right_ok:
            return right_v, True, left_steps + right_steps, left_retry + right_retry + 1
    return Vp, False, 1, 1 if depth == 0 else 0


def _solve_fixed_gear2_numba(ctx, V0, tgrid, input_values, edge_mask_arr,
                             profile, gear2_retry_requested):
    out = _PathOutcome()
    if not (not ctx.adaptive and ctx.integration_method == "gear2" and
            _GEAR2_NUMBA_GRID and transient_solve_grid_gear2_numba is not None and
            ctx.n_aug == ctx.n):
        return out
    try:
        max_step_arg = -1.0 if ctx.max_step is None else float(ctx.max_step)
        flat_max_step_arg = -1.0 if ctx.flat_max_step is None else float(ctx.flat_max_step)
        t_profile0 = time.perf_counter()
        # Numba gear2 owns both periodic single-step grids and raw transient
        # maxstep/retry subdivision. If it rejects a robust step, Python retry
        # remains the correctness fallback in the driver.
        g2_orig_idx = None
        g2 = transient_solve_grid_gear2_numba(
            np.asarray(V0, float), np.asarray(tgrid, float),
            np.asarray(input_values, float), edge_mask_arr, profile,
            max_step_arg, flat_max_step_arg, int(ctx.max_retry_subdivisions),
            int(ctx.n), int(ctx.newton_maxit), float(ctx.newton_step_limit),
            float(ctx.newton_vtol), float(ctx.gmin),
            bool(ctx.fallback_full_jacobian or ctx.fallback_least_squares),
            float(ctx.fallback_tol), float(ctx.HH),
            ctx.dev_d_kind, ctx.dev_d_ref, ctx.dev_d_val,
            ctx.dev_g_kind, ctx.dev_g_ref, ctx.dev_g_val,
            ctx.dev_s_kind, ctx.dev_s_ref, ctx.dev_s_val,
            ctx.dev_di, ctx.dev_gi, ctx.dev_si, ctx.dev_use_abs,
            ctx.p_Vfb, ctx.p_Vss, ctx.p_Lc, ctx.p_lambda, ctx.p_contact_scale,
            ctx.p_exponent, ctx.p_current_scale, ctx.p_inv_Rleak,
            ctx.p_two_over_pi, ctx.p_cap_cgs1, ctx.p_cap_cgd1,
            ctx.p_cap_half_wl_ci, ctx.p_cap_cgs3_base, ctx.p_cap_cgd3_base,
            ctx.p_k1, ctx.p_gate_leak_g,
            ctx.op_cache_valid, ctx.op_cache_vs1, ctx.op_cache_vd1,
            ctx.res_a_kind, ctx.res_a_ref, ctx.res_a_val,
            ctx.res_b_kind, ctx.res_b_ref, ctx.res_b_val,
            ctx.res_ai, ctx.res_bi, ctx.res_g,
            ctx.cap_a_kind, ctx.cap_a_ref, ctx.cap_a_val,
            ctx.cap_b_kind, ctx.cap_b_ref, ctx.cap_b_val,
            ctx.cap_ai, ctx.cap_bi, ctx.cap_value,
            ctx.isrc_pi, ctx.isrc_qi, ctx.isrc_value,
            ctx.dyn_pi, ctx.dyn_qi, ctx.dyn_input_idx,
            int(ctx.cap_id), float(ctx.clip_lo), float(ctx.clip_hi),
        )
        ok_g2, Vfast, fast_substeps, fail_index, raw_profile, _rfi = g2
        out.profile_wall_s = time.perf_counter() - t_profile0
        if ok_g2:
            out.Vhist = (np.ascontiguousarray(Vfast[g2_orig_idx])
                         if g2_orig_idx is not None else Vfast)
            out.nsubsteps = int(fast_substeps)
            raw_profile_arr = np.asarray(raw_profile, float)
            out.nretry = int(raw_profile_arr[10])
            out.nfail = int(raw_profile_arr[13])
            out.gear2_done = True
            out.gear2_numba_used = True
            out.profile_stats = raw_profile_arr
            out.handled = True
        elif gear2_retry_requested:
            out.numba_grid_failed_index = int(fail_index)
            out.numba_grid_failed_substeps = int(fast_substeps)
            out.numba_grid_failed_profile = np.asarray(raw_profile, float)
            out.numba_grid_failed_intervals = (
                None if _rfi is None else np.asarray(_rfi, int))
    except Exception as exc:
        out.numba_grid_error = f"gear2: {type(exc).__name__}: {exc}"
    return out


def _solve_adaptive_gear2_numba(ctx, V0, tgrid, input_values, profile):
    out = _PathOutcome()
    if not (ctx.adaptive and ctx.integration_method == "gear2" and
            _GEAR2_NUMBA_GRID and transient_solve_adaptive_gear2_numba is not None and
            ctx.n_aug == ctx.n):
        return out
    try:
        max_step_arg = -1.0 if ctx.max_step is None else float(ctx.max_step)
        h0_arg = -1.0 if ctx.adaptive_h0 is None else float(ctx.adaptive_h0)
        t_profile0 = time.perf_counter()
        ad = transient_solve_adaptive_gear2_numba(
            np.asarray(V0, float), np.asarray(tgrid, float),
            np.asarray(input_values, float), profile,
            max_step_arg, float(ctx.adaptive_reltol), float(ctx.adaptive_vabstol),
            float(ctx.adaptive_iabstol), int(ctx.adaptive_max_steps), h0_arg,
            int(ctx.n), int(ctx.newton_maxit), float(ctx.newton_step_limit),
            float(ctx.newton_vtol), float(ctx.gmin),
            bool(ctx.fallback_full_jacobian or ctx.fallback_least_squares),
            float(ctx.fallback_tol), float(ctx.HH),
            ctx.dev_d_kind, ctx.dev_d_ref, ctx.dev_d_val,
            ctx.dev_g_kind, ctx.dev_g_ref, ctx.dev_g_val,
            ctx.dev_s_kind, ctx.dev_s_ref, ctx.dev_s_val,
            ctx.dev_di, ctx.dev_gi, ctx.dev_si, ctx.dev_use_abs,
            ctx.p_Vfb, ctx.p_Vss, ctx.p_Lc, ctx.p_lambda, ctx.p_contact_scale,
            ctx.p_exponent, ctx.p_current_scale, ctx.p_inv_Rleak,
            ctx.p_two_over_pi, ctx.p_cap_cgs1, ctx.p_cap_cgd1,
            ctx.p_cap_half_wl_ci, ctx.p_cap_cgs3_base, ctx.p_cap_cgd3_base,
            ctx.p_k1, ctx.p_gate_leak_g,
            ctx.op_cache_valid, ctx.op_cache_vs1, ctx.op_cache_vd1,
            ctx.res_a_kind, ctx.res_a_ref, ctx.res_a_val,
            ctx.res_b_kind, ctx.res_b_ref, ctx.res_b_val,
            ctx.res_ai, ctx.res_bi, ctx.res_g,
            ctx.cap_a_kind, ctx.cap_a_ref, ctx.cap_a_val,
            ctx.cap_b_kind, ctx.cap_b_ref, ctx.cap_b_val,
            ctx.cap_ai, ctx.cap_bi, ctx.cap_value,
            ctx.isrc_pi, ctx.isrc_qi, ctx.isrc_value,
            ctx.dyn_pi, ctx.dyn_qi, ctx.dyn_input_idx,
            int(ctx.cap_id), float(ctx.clip_lo), float(ctx.clip_hi),
        )
        ok_ad, tfast, Vfast, input_fast, naccept, fast_substeps, fast_rejects, raw_profile = ad
        if ok_ad:
            naccept = int(naccept)
            out.tgrid = np.ascontiguousarray(tfast[:naccept])
            out.Vhist = np.ascontiguousarray(Vfast[:naccept])
            out.input_values = np.ascontiguousarray(input_fast[:naccept].T)
            out.N = len(out.tgrid)
            out.nsubsteps = int(fast_substeps)
            out.nretry = int(fast_rejects)
            out.profile_stats = np.asarray(raw_profile, float)
            out.profile_wall_s = time.perf_counter() - t_profile0
            out.gear2_done = True
            out.adaptive_used = True
            out.adaptive_numba_used = True
            out.gear2_numba_used = True
            out.handled = True
    except Exception as exc:
        out.numba_grid_error = f"adaptive_gear2: {type(exc).__name__}: {exc}"
    return out


def _solve_be_numba(ctx, V0, tgrid, input_values, edge_mask_arr, profile):
    out = _PathOutcome()
    if not ((not ctx.adaptive) and transient_solve_grid_numba is not None and
            ctx.n_aug == ctx.n):
        return out
    try:
        max_step_arg = -1.0 if ctx.max_step is None else float(ctx.max_step)
        flat_max_step_arg = -1.0 if ctx.flat_max_step is None else float(ctx.flat_max_step)
        t_profile0 = time.perf_counter()
        grid_result = transient_solve_grid_numba(
            np.asarray(V0, float), np.asarray(tgrid, float),
            np.asarray(input_values, float), edge_mask_arr, profile,
            max_step_arg, flat_max_step_arg,
            int(ctx.max_retry_subdivisions),
            int(ctx.n), int(ctx.newton_maxit), float(ctx.newton_step_limit),
            float(ctx.newton_vtol), float(ctx.gmin),
            bool(ctx.fallback_full_jacobian or ctx.fallback_least_squares),
            float(ctx.fallback_tol), float(ctx.HH),
            ctx.dev_d_kind, ctx.dev_d_ref, ctx.dev_d_val,
            ctx.dev_g_kind, ctx.dev_g_ref, ctx.dev_g_val,
            ctx.dev_s_kind, ctx.dev_s_ref, ctx.dev_s_val,
            ctx.dev_di, ctx.dev_gi, ctx.dev_si, ctx.dev_use_abs,
            ctx.p_Vfb, ctx.p_Vss, ctx.p_Lc, ctx.p_lambda, ctx.p_contact_scale,
            ctx.p_exponent, ctx.p_current_scale, ctx.p_inv_Rleak,
            ctx.p_two_over_pi, ctx.p_cap_cgs1, ctx.p_cap_cgd1,
            ctx.p_cap_half_wl_ci, ctx.p_cap_cgs3_base, ctx.p_cap_cgd3_base,
            ctx.p_k1, ctx.p_gate_leak_g,
            ctx.op_cache_valid, ctx.op_cache_vs1, ctx.op_cache_vd1,
            ctx.res_a_kind, ctx.res_a_ref, ctx.res_a_val,
            ctx.res_b_kind, ctx.res_b_ref, ctx.res_b_val,
            ctx.res_ai, ctx.res_bi, ctx.res_g,
            ctx.cap_a_kind, ctx.cap_a_ref, ctx.cap_a_val,
            ctx.cap_b_kind, ctx.cap_b_ref, ctx.cap_b_val,
            ctx.cap_ai, ctx.cap_bi, ctx.cap_value,
            ctx.isrc_pi, ctx.isrc_qi, ctx.isrc_value,
            ctx.dyn_pi, ctx.dyn_qi, ctx.dyn_input_idx,
            int(ctx.cap_id), float(ctx.clip_lo), float(ctx.clip_hi),
        )
        if len(grid_result) == 5:
            ok_grid, Vfast, fast_substeps, fail_index, raw_profile = grid_result
            raw_failed_intervals = None
        else:
            (ok_grid, Vfast, fast_substeps, fail_index, raw_profile,
             raw_failed_intervals) = grid_result
        out.profile_wall_s = time.perf_counter() - t_profile0
        out.numba_grid_failed_intervals = (
            None if raw_failed_intervals is None else np.asarray(raw_failed_intervals, int))
        if ok_grid:
            out.Vhist = Vfast
            out.nsubsteps = int(fast_substeps)
            raw_profile_arr = np.asarray(raw_profile, float)
            out.nfail = int(raw_profile_arr[13])
            out.nretry = out.nfail
            ctx.numba_newton_attempts += out.nsubsteps
            ctx.numba_newton_success += out.nsubsteps
            out.used_grid_numba = True
            out.profile_stats = raw_profile_arr
            out.handled = True
        else:
            out.numba_grid_failed_index = int(fail_index)
            out.numba_grid_failed_substeps = int(fast_substeps)
            out.numba_grid_failed_profile = np.asarray(raw_profile, float)
            if out.numba_grid_failed_index is not None and out.numba_grid_failed_index > 1:
                out.Vhist = Vfast
                out.nsubsteps = int(fast_substeps)
                ctx.numba_newton_attempts += out.nsubsteps
                ctx.numba_newton_success += out.nsubsteps
                out.partial_grid_numba = True
                out.python_start_idx = int(out.numba_grid_failed_index)
    except Exception as exc:
        out.numba_grid_error = f"{type(exc).__name__}: {exc}"
    return out


def _solve_fixed_gear2_python(ctx, Vhist, tgrid, input_values, edge_mask_arr,
                              gear2_retry_requested):
    out = _PathOutcome()
    if ctx.adaptive or ctx.integration_method != "gear2":
        return out
    N = len(tgrid)
    nfail = 0
    nretry = 0
    nsubsteps = 0
    for k in range(1, N):
        h_n = tgrid[k] - tgrid[k - 1]
        if h_n <= 0.0:
            raise ValueError("tgrid must be strictly increasing")
        Vp = Vhist[k - 1]
        input_now = input_values[:, k]
        input_prev = input_values[:, k - 1]
        if k >= 2:
            Vp2 = Vhist[k - 2]
            input_prev2 = input_values[:, k - 2]
            h_prev = tgrid[k - 1] - tgrid[k - 2]
        else:
            Vp2 = input_prev2 = h_prev = None
        interval_edge = (len(edge_mask_arr) == N and
                         bool(edge_mask_arr[k] or edge_mask_arr[k - 1]))
        local_max_step = ctx.max_step
        if ctx.flat_max_step is not None and ctx.flat_max_step > 0.0 and not interval_edge:
            local_max_step = ctx.flat_max_step
        pieces = 1 if local_max_step is None else max(1, int(np.ceil(h_n / local_max_step)))
        Vcur = Vp
        Vcur2 = Vp2
        input_cur = input_prev
        input_cur2 = input_prev2
        h_prev_cur = h_prev
        interval_ok = True
        interval_retries = 0
        for j in range(pieces):
            f1 = (j + 1) / pieces
            input_next = _k_input_at_between(input_prev, input_now, f1)
            (Vcur, Vcur2, input_cur2, h_prev_cur, ok, steps, retries) = (
                _k_solve_gear2_chunk(ctx, Vcur, Vcur2, input_cur, input_cur2,
                                     h_n / pieces, h_prev_cur, input_next)
            )
            input_cur = input_next
            nsubsteps += steps
            interval_retries += retries
            if not ok:
                interval_ok = False
                break
        if not interval_ok:
            nfail += 1
        Vhist[k] = Vcur
        nretry += interval_retries
    out.handled = True
    out.Vhist = Vhist
    out.nfail = nfail
    out.nretry = nretry
    out.nsubsteps = nsubsteps
    out.gear2_done = True
    out.gear2_python_retry_used = bool(gear2_retry_requested)
    return out


def _solve_adaptive_gear2_python(ctx, V0, tgrid, input_values,
                                 nfail0, nretry0, nsubsteps0):
    out = _PathOutcome()
    if not (ctx.adaptive and ctx.integration_method == "gear2"):
        return out
    t_src = np.asarray(tgrid, float)
    if len(t_src) < 2 or not np.all(np.diff(t_src) > 0.0):
        raise ValueError("adaptive transient requires a strictly increasing tgrid")
    t0 = float(t_src[0])
    t_end = float(t_src[-1])
    span = t_end - t0
    if span <= 0.0:
        raise ValueError("adaptive transient requires tgrid[-1] > tgrid[0]")
    max_step_eff = span if ctx.max_step is None or ctx.max_step <= 0.0 else min(float(ctx.max_step), span)
    if ctx.adaptive_h0 is None:
        h = min(max_step_eff, max(span / max(16, len(t_src) - 1), 1e-18))
        if len(t_src) > 1:
            h = min(h, float(np.min(np.diff(t_src))))
    else:
        h = float(ctx.adaptive_h0)
    if h <= 0.0 or not np.isfinite(h):
        h = min(max_step_eff, span / 100.0)
    h = min(h, max_step_eff)
    min_h = max(1e-18, span * 1e-15)
    reltol = float(ctx.adaptive_reltol)
    vabstol = float(ctx.adaptive_vabstol)
    iabstol = float(ctx.adaptive_iabstol)
    max_adapt_steps = max(1, int(ctx.adaptive_max_steps))
    critical_times = []
    if input_values.shape[0] and len(t_src) >= 3:
        dt = np.diff(t_src)
        slopes = np.diff(input_values, axis=1) / dt[None, :]
        global_slope = max(1.0, float(np.max(np.abs(slopes))))
        for kk in range(1, len(t_src) - 1):
            jump = np.max(np.abs(slopes[:, kk] - slopes[:, kk - 1]))
            if jump > 0.1 * global_slope:
                critical_times.append(float(t_src[kk]))
    critical_times = np.asarray(critical_times, float)

    def input_at_time(tt):
        if input_values.shape[0] == 0:
            return np.empty(0, float)
        vals = np.empty(input_values.shape[0], float)
        for ii in range(input_values.shape[0]):
            vals[ii] = np.interp(tt, t_src, input_values[ii])
        return vals

    def wrms_lte(v_ref, v_full, lte):
        scale = reltol * np.maximum(np.abs(v_ref), np.abs(v_full))
        if ctx.n > 0:
            scale[:ctx.n] += vabstol
        if ctx.n_aug > ctx.n:
            scale[ctx.n:] += iabstol
        scale = np.maximum(scale, 1e-30)
        return float(np.sqrt(np.mean((lte / scale) ** 2)))

    def propose_h(step, err):
        if not np.isfinite(err) or err <= 0.0:
            fac = 2.0
        else:
            fac = 0.9 * err ** (-1.0 / 3.0)
            fac = min(2.0, max(0.2, fac))
        return step * fac

    t_list = [t0]
    v_list = [np.asarray(V0, float).copy()]
    in0 = input_at_time(t0)
    input_list = [in0.copy()]
    Vp = np.asarray(V0, float).copy()
    Vp2 = None
    input_prev = in0
    input_prev2 = None
    h_prev = None
    tt = t0
    nfail = int(nfail0)
    nretry = int(nretry0)
    nsubsteps = int(nsubsteps0)
    rejected_too_small = False
    for _adapt_iter in range(max_adapt_steps):
        if tt >= t_end - max(1e-18, 1e-13 * span):
            tt = t_end
            break
        if h_prev is not None:
            h = min(h, 2.0 * h_prev)
        h = min(h, max_step_eff, t_end - tt)
        if critical_times.size:
            upcoming = critical_times[critical_times > tt + min_h]
            if upcoming.size:
                next_critical = float(upcoming[0])
                if next_critical < tt + h:
                    h = next_critical - tt
        if h <= min_h:
            rejected_too_small = True
            nfail += 1
            break
        input_now = input_at_time(tt + h)
        input_mid = input_at_time(tt + 0.5 * h)
        Vfull, ok_full, _ = _k_gear2_try_step(
            ctx, Vp, Vp2, input_prev, input_prev2, input_now, h, h_prev,
            use_fallback=True)
        Vmid, ok_mid, _ = _k_gear2_try_step(
            ctx, Vp, Vp2, input_prev, input_prev2, input_mid, 0.5 * h, h_prev,
            use_fallback=True)
        ok_half2 = False
        Vhalf2 = Vp
        if ok_mid:
            Vhalf2, ok_half2, _ = _k_gear2_try_step(
                ctx, Vmid, Vp, input_mid, input_prev, input_now, 0.5 * h, 0.5 * h,
                use_fallback=True)
        nsubsteps += 3
        if ok_full and ok_mid and ok_half2:
            lte = (Vhalf2 - Vfull) / 3.0
            err = wrms_lte(Vhalf2, Vfull, lte)
        else:
            err = np.inf
        if err <= 1.0:
            tt = min(t_end, tt + h)
            t_list.append(tt)
            v_list.append(Vhalf2.copy())
            input_list.append(input_now.copy())
            Vp = Vhalf2
            input_prev = input_now
            hit_critical = False
            if critical_times.size:
                hit_critical = bool(np.any(np.isclose(
                    critical_times, tt, rtol=0.0,
                    atol=max(min_h, 1e-13 * span))))
            if hit_critical:
                Vp2 = None
                input_prev2 = None
                h_prev = None
            else:
                Vp2 = v_list[-2].copy()
                input_prev2 = input_list[-2].copy()
                h_prev = h
            h = propose_h(h, max(err, 1e-12))
        else:
            nretry += 1
            h = max(min_h, propose_h(h, err))
    else:
        nfail += 1

    out.tgrid = np.asarray(t_list, float)
    out.Vhist = np.vstack(v_list)
    if input_values.shape[0] == 0:
        out.input_values = np.empty((0, len(out.tgrid)), float)
    else:
        out.input_values = np.vstack(input_list).T
    out.N = len(out.tgrid)
    out.nfail = nfail
    out.nretry = nretry + (1 if rejected_too_small else 0)
    out.nsubsteps = nsubsteps
    out.gear2_done = True
    out.adaptive_used = True
    out.handled = True
    return out


def _solve_be_python(ctx, Vhist, tgrid, input_values, edge_mask_arr,
                     python_start_idx, nfail0, nretry0, nsubsteps0):
    out = _PathOutcome()
    N = len(tgrid)
    nfail = int(nfail0)
    nretry = int(nretry0)
    nsubsteps = int(nsubsteps0)
    for k in range(python_start_idx, N):
        h = tgrid[k] - tgrid[k - 1]
        if h <= 0.0:
            raise ValueError("tgrid must be strictly increasing")
        Vp = Vhist[k - 1]
        input_start = input_values[:, k - 1]
        input_end = input_values[:, k]
        interval_edge = (len(edge_mask_arr) == N and
                         bool(edge_mask_arr[k] or edge_mask_arr[k - 1]))
        local_max_step = ctx.max_step
        if ctx.flat_max_step is not None and ctx.flat_max_step > 0.0 and not interval_edge:
            local_max_step = ctx.flat_max_step
        pieces = 1 if local_max_step is None else max(1, int(np.ceil(h / local_max_step)))
        interval_ok = True
        interval_retries = 0
        in0 = input_start
        for j in range(pieces):
            f1 = (j + 1) / pieces
            in1 = _k_input_at_between(input_start, input_end, f1)
            Vp, ok, steps, retries = _k_solve_chunk(ctx, Vp, in0, in1, h / pieces)
            in0 = in1
            nsubsteps += steps
            interval_retries += retries
            if not ok:
                interval_ok = False
        Vhist[k] = Vp
        nretry += interval_retries
        if not interval_ok:
            nfail += 1
    out.handled = True
    out.Vhist = Vhist
    out.nfail = nfail
    out.nretry = nretry
    out.nsubsteps = nsubsteps
    return out


def _assemble_result(ctx, tgrid, Vhist, input_values, input_keys,
                     nfail, nretry, nsubsteps,
                     adaptive_used, adaptive_numba_used,
                     used_grid_numba, gear2_numba_used,
                     gear2_python_retry_used,
                     profile, profile_stats, profile_wall_s,
                     partial_grid_numba, numba_grid_error,
                     numba_grid_failed_index, numba_grid_failed_substeps,
                     numba_grid_failed_profile, numba_grid_failed_intervals):
    N = len(tgrid)
    idx = ctx.idx
    plan = ctx.plan
    nodes = {nm: Vhist[:, idx[nm]] for nm in plan.solved}
    out = np.zeros(N)
    for node, weight in plan.output_weights.items():
        out += weight * nodes[node]
    result = {"t": tgrid, "output": out, "vout": out, "nfail": nfail,
              "nretry": nretry, "nsubsteps": nsubsteps, "nodes": nodes,
              "transient_cap_mode": _CAP_MODE,
              "transient_cap_mode_id": int(ctx.cap_id)}
    if adaptive_used:
        result["adaptive"] = True
        result["adaptive_reltol"] = float(ctx.adaptive_reltol)
        result["adaptive_vabstol"] = float(ctx.adaptive_vabstol)
        result["adaptive_iabstol"] = float(ctx.adaptive_iabstol)
        result["adaptive_accepted_steps"] = int(max(0, len(tgrid) - 1))
        result["adaptive_rejected_steps"] = int(nretry)
        result["inputs"] = {
            key: input_values[pos].copy()
            for pos, key in enumerate(input_keys)
        }
    result["numba_grid_solver"] = bool(used_grid_numba or gear2_numba_used)
    result["numba_adaptive_solver"] = bool(adaptive_numba_used)
    if ctx.integration_method == "gear2":
        result["gear2_python_retry_solver"] = bool(gear2_python_retry_used)
    if ctx.numba_newton_attempts:
        result["numba_newton_attempts"] = ctx.numba_newton_attempts
        result["numba_newton_success"] = ctx.numba_newton_success
        result["numba_newton_fallback"] = ctx.numba_newton_fallback
    if profile:
        if profile_stats is None:
            profile_stats = np.zeros(24, dtype=float)
            profile_stats[0] = 0.0
            profile_stats[6] = 0.0
            profile_stats[7] = float(nsubsteps)
            profile_stats[10] = float(nfail)
            profile_stats[11] = float(N - 1)
            profile_stats[12] = float(nsubsteps)
        total_iters = float(profile_stats[0])
        edge_iters = float(profile_stats[8])
        flat_iters = float(profile_stats[9])
        iter_work = edge_iters + flat_iters
        edge_time_est = profile_wall_s * edge_iters / iter_work if iter_work else 0.0
        flat_time_est = profile_wall_s * flat_iters / iter_work if iter_work else 0.0
        result["transient_profile"] = {
            "enabled": True,
            "numba_grid_solver": bool(used_grid_numba or gear2_numba_used),
            "numba_grid_partial": bool(partial_grid_numba),
            "numba_grid_error": numba_grid_error,
            "numba_grid_failed_index": numba_grid_failed_index,
            "numba_grid_failed_substeps": int(numba_grid_failed_substeps),
            "numba_grid_failed_newton_iters": (
                int(numba_grid_failed_profile[0])
                if numba_grid_failed_profile is not None else 0),
            "numba_grid_failed_substep_failures": (
                int(numba_grid_failed_profile[10])
                if numba_grid_failed_profile is not None else 0),
            "numba_grid_failed_interval_failures": (
                int(numba_grid_failed_profile[13])
                if numba_grid_failed_profile is not None else 0),
            "numba_grid_failed_last_residual_inf": (
                float(numba_grid_failed_profile[16])
                if numba_grid_failed_profile is not None and len(numba_grid_failed_profile) > 16
                else 0.0),
            "numba_grid_failed_max_residual_inf": (
                float(numba_grid_failed_profile[17])
                if numba_grid_failed_profile is not None and len(numba_grid_failed_profile) > 17
                else 0.0),
            "numba_grid_failed_last_step_inf": (
                float(numba_grid_failed_profile[18])
                if numba_grid_failed_profile is not None and len(numba_grid_failed_profile) > 18
                else 0.0),
            "numba_grid_failed_max_step_inf": (
                float(numba_grid_failed_profile[19])
                if numba_grid_failed_profile is not None and len(numba_grid_failed_profile) > 19
                else 0.0),
            "numba_grid_failed_stamp_or_prev_count": (
                int(numba_grid_failed_profile[20])
                if numba_grid_failed_profile is not None and len(numba_grid_failed_profile) > 20
                else 0),
            "numba_grid_failed_linear_solve_count": (
                int(numba_grid_failed_profile[21])
                if numba_grid_failed_profile is not None and len(numba_grid_failed_profile) > 21
                else 0),
            "numba_grid_failed_maxit_count": (
                int(numba_grid_failed_profile[22])
                if numba_grid_failed_profile is not None and len(numba_grid_failed_profile) > 22
                else 0),
            "wall_time_s": float(profile_wall_s),
            "nsubsteps": int(nsubsteps),
            "intervals": int(profile_stats[11]),
            "newton_iters_total": int(profile_stats[0]),
            "newton_iters_avg": float(total_iters / nsubsteps) if nsubsteps else 0.0,
            "pmos_op_solves": int(profile_stats[1]),
            "pmos_internal_newton_attempts": int(profile_stats[2]),
            "pmos_internal_newton_iters": int(profile_stats[3]),
            "pmos_internal_newton_iters_avg": (
                float(profile_stats[3] / profile_stats[2]) if profile_stats[2] else 0.0),
            "internal_fd_jac_fallbacks": int(profile_stats[4]),
            "terminal_fd_jac_fallbacks": int(profile_stats[5]),
            "edge_substeps": int(profile_stats[6]),
            "flat_substeps": int(profile_stats[7]),
            "edge_newton_iters": int(profile_stats[8]),
            "flat_newton_iters": int(profile_stats[9]),
            "failed_substeps": int(profile_stats[10]),
            "failed_intervals": int(profile_stats[13]),
            "failed_edge_intervals": int(profile_stats[14]),
            "failed_flat_intervals": int(profile_stats[15]),
            "failed_interval_indices": (
                [int(v) for v in numba_grid_failed_intervals
                 if int(v) >= 0]
                if numba_grid_failed_intervals is not None else []
            ),
            "failed_last_residual_inf": (
                float(profile_stats[16]) if len(profile_stats) > 16 else 0.0),
            "failed_max_residual_inf": (
                float(profile_stats[17]) if len(profile_stats) > 17 else 0.0),
            "failed_last_step_inf": (
                float(profile_stats[18]) if len(profile_stats) > 18 else 0.0),
            "failed_max_step_inf": (
                float(profile_stats[19]) if len(profile_stats) > 19 else 0.0),
            "failed_stamp_or_prev_count": (
                int(profile_stats[20]) if len(profile_stats) > 20 else 0),
            "failed_linear_solve_count": (
                int(profile_stats[21]) if len(profile_stats) > 21 else 0),
            "failed_maxit_count": (
                int(profile_stats[22]) if len(profile_stats) > 22 else 0),
            "stalled_residual_accepts": (
                int(profile_stats[23]) if len(profile_stats) > 23 else 0),
            "edge_time_s_est": float(edge_time_est),
            "flat_time_s_est": float(flat_time_est),
            "time_estimate_basis": "newton_iteration_weighted",
        }
    if "VOP" in idx:
        result["vop"] = nodes["VOP"]
    if "VON" in idx:
        result["von"] = nodes["VON"]
    return result


def transient(sizes, bias, tgrid, vip=None, vin=None, nf=None, V0=None,
              topo=AFE_TOPO, inputs=None, node_inputs=None, current_inputs=None,
              corner=None,
              max_step=None, flat_max_step=None,
              max_retry_subdivisions=0, newton_maxit=30,
              newton_step_limit=5.0, newton_vtol=1e-8,
              fallback_full_jacobian=False,
              fallback_least_squares=False, fallback_tol=1e-9,
              signed_devices=None, profile=False, edge_mask=None,
              rail_margin=None, integration_method="be",
              gear2_be_fallback=True, cap_mode=None, cap_mode_id=None,
              adaptive=False, adaptive_reltol=1e-4, adaptive_vabstol=1e-6,
              adaptive_iabstol=1e-12, adaptive_max_steps=200000,
              adaptive_h0=None):
    """Backward-Euler (default) or gear2/BDF2 transient.

      integration_method : "be" (backward-Euler, 1st order; the default for the
               raw transient because its numba grid keeps substep subdivision +
               retry, which hard standalone transients rely on) or "gear2"
               (variable-step BDF2, 2nd order, numba-accelerated, with maxstep
               subdivision/retry support and step-ratio limiting). The PSS/chopper periodic path
               defaults to gear2 (it closes the chopper PAC switch-edge error to
               <1% and its grid is well-conditioned); raw transient callers can
               opt in on uniform/well-conditioned grids.
      tgrid : (N,) time points [s]
      vip,vin : legacy AFE M7/M8 gate waveforms [V]
      inputs : generic mapping {input_key: waveform}; device gates are mapped by
               topo.transient_inputs, e.g. {"M1": "in"}.
      node_inputs : mapping {node_name: input_key} to drive a (rail) NODE with a
               waveform — used for a testbench where the stimulus enters at source
               nodes and propagates through a front-end network, e.g.
               {"VINP": "vip", "VINN": "vin"}.
      current_inputs : time-varying ideal current sources. Each entry can be
               {"p": nplus, "q": nminus, "input": key} or (p, q, key).
               The waveform current flows p -> q, matching topology.isources.
      max_step : optional maximum internal step. Intervals larger than this are
               split linearly between adjacent input samples.
      flat_max_step : optional maximum internal step for intervals not marked by
               edge_mask. If omitted, max_step is used everywhere.
      max_retry_subdivisions : if Newton fails on a step, recursively bisect that
               step up to this depth before recording a failure.
      fallback_least_squares : if true, a failed Newton step is retried with a
               rail-bounded least-squares solve before substepping/failing.
      fallback_full_jacobian : if true, a failed Newton step is retried with an
               expensive finite-difference Jacobian of the full residual at the
               smallest retry subdivision.
      signed_devices : optional device names whose terminal current keeps the
               Verilog-A drain-terminal sign. The default AFE devices use the
               legacy abs(Idc) convention calibrated by AC/noise; bidirectional
               pass switches should be signed.
      profile : if true, include transient_profile counters in the result.
      edge_mask : optional boolean mask over tgrid points; intervals touching a
               true point are counted as edge work in transient_profile.
      rail_margin : optional voltage margin around numeric rails for topologies
               that need physical branch selection. If omitted, topologies with
               require_dc_in_box use a 2 V margin; other topologies are unbounded.
      V0    : optional initial solved-node vector.
    Returns dict: t, output, vout, nfail, and per-node arrays. AFE legacy vop/von
    fields are included when those nodes exist.
    """
    integration_method = str(integration_method).lower()
    if adaptive and integration_method != "gear2":
        raise ValueError("adaptive transient requires integration_method='gear2'")
    if cap_mode is not None and cap_mode_id is not None:
        requested = _cap_mode_to_id(cap_mode)
        if int(cap_mode_id) != int(requested):
            raise ValueError("cap_mode and cap_mode_id disagree")
    tft = build_devices(sizes, nf=nf, corner=corner, topo=topo)
    tgrid = np.asarray(tgrid, float)
    N = len(tgrid)
    # Per-call cap operator override (default = the module/env-selected mode).
    # The chopper PSS uses the trapezoidal "average" mode (id 1) -- a STABLE,
    # non-conservative C(V)*dV/dt discretization that matches Cadence's commutation
    # feedthrough (charge Q-stamp over-swings it ~26%); charge stays the default
    # everywhere else (it is L-stable on stiff tau>>T circuits where average rings).
    cap_mode_from_name = _cap_mode_to_id(cap_mode)
    _cap_id = int(_CAP_MODE_ID if cap_mode_from_name is None and cap_mode_id is None
                  else cap_mode_from_name if cap_mode_from_name is not None
                  else cap_mode_id)
    if inputs is None:
        inputs = {}
        if vip is not None:
            inputs["vip"] = vip
        if vin is not None:
            inputs["vin"] = vin
    inputs = {key: np.asarray(val, float) for key, val in inputs.items()}
    for key, val in inputs.items():
        if len(val) != N:
            raise ValueError(f"Input waveform {key!r} length {len(val)} != len(tgrid) {N}")
    input_keys = tuple(inputs)
    input_values = (np.vstack([inputs[key] for key in input_keys])
                    if input_keys else np.empty((0, N), float))
    node_inputs = dict(node_inputs or {})
    for node, key in node_inputs.items():
        if key not in inputs:
            raise ValueError(f"node_inputs[{node!r}] references missing waveform {key!r}")

    plan = CompiledTopology(topo, bias, input_keys=input_keys,
                            node_inputs=node_inputs, transient_inputs=True)
    idx, n = plan.idx, plan.n
    n_aug = plan.n_aug                 # n nodes + m ideal-voltage-source branch currents
    termv = plan.term_value
    # Every device uses its signed Verilog-A drain current. abs(Idc) was only
    # correct for never-reversing devices (forward PMOS: I_d1_d>0 so signed==abs)
    # but turned a *reverse*-biased pass-gate switch into an anti-restoring pump
    # (the SC-LPF runaway). signed==abs in forward, so the AFE amp/chopper are
    # unchanged; the chopper already listed its commutators in signed_devices.
    # `signed_devices` is retained (no-op now) for back-compat with callers.
    signed_devices = set(signed_devices or ())
    dev_meta = [(tft[item.name], True,
                 item.d, item.g, item.s, item.di, item.gi, item.si)
                for item in plan.devices]
    load_meta = [(item.a, item.b, item.ai, item.bi, item.value)
                 for item in plan.capacitors]
    res_meta = [(item.a, item.b, item.ai, item.bi, item.g)
                for item in plan.resistors]
    isrc_meta = [(item.pi, item.qi, item.value) for item in plan.isources]
    vccs_meta = [(item.pi, item.qi, item.cp, item.cn, item.gm)
                 for item in plan.vccs]
    # Ideal voltage sources (true MNA, Python path): (a_term, b_term, pi, qi, bi,
    # e_const, e_input_idx). Branch current is the unknown at V[bi]; constraint row
    # bi pins V_p - V_q = E. Vsource circuits force the pure-Python step path below.
    vs_meta = [(item.p, item.q, item.pi, item.qi, item.bi, item.e_const, item.e_input_idx)
               for item in plan.vsources]
    vcvs_meta = [(item.p, item.q, item.cp, item.cn, item.pi, item.qi,
                  item.cpi, item.cni, item.bi, item.mu)
                 for item in plan.vcvs]
    cccs_meta = [(item.pi, item.qi, item.ctrl_bi, item.beta)
                 for item in plan.cccs]
    ccvs_meta = [(item.p, item.q, item.pi, item.qi, item.bi,
                  item.ctrl_bi, item.gamma)
                 for item in plan.ccvs]
    dyn_isrc_meta = []
    for pos, item in enumerate(current_inputs or ()):
        if isinstance(item, dict):
            p_node = item["p"]
            q_node = item["q"]
            key = item["input"]
        else:
            p_node, q_node, key = item
        if key not in plan.input_index:
            raise ValueError(f"current_inputs[{pos}] references missing waveform {key!r}")
        pterm = plan.compile_term(p_node)
        qterm = plan.compile_term(q_node)
        dyn_isrc_meta.append((
            plan.solved_index(pterm),
            plan.solved_index(qterm),
            plan.input_index[key],
        ))

    dev_d_kind, dev_d_ref, dev_d_val = _term_arrays([item[2] for item in dev_meta])
    dev_g_kind, dev_g_ref, dev_g_val = _term_arrays([item[3] for item in dev_meta])
    dev_s_kind, dev_s_ref, dev_s_val = _term_arrays([item[4] for item in dev_meta])
    dev_di = _index_array(item[5] for item in dev_meta)
    dev_gi = _index_array(item[6] for item in dev_meta)
    dev_si = _index_array(item[7] for item in dev_meta)
    dev_use_abs = np.array([not item[1] for item in dev_meta], dtype=np.bool_)
    dev_objs = [item[0] for item in dev_meta]
    _np_params = [d.get_numba_params() for d in dev_objs]
    p_Vfb = np.array([p.Vfb for p in _np_params], dtype=float)
    p_Vss = np.array([p.Vss for p in _np_params], dtype=float)
    p_Lc = np.array([p.Lc for p in _np_params], dtype=float)
    p_lambda = np.array([p.lambda_ for p in _np_params], dtype=float)
    p_contact_scale = np.array([p.contact_scale for p in _np_params], dtype=float)
    p_exponent = np.array([p.channel_exponent for p in _np_params], dtype=float)
    p_current_scale = np.array([p.current_scale for p in _np_params], dtype=float)
    p_inv_Rleak = np.array([p.inv_Rleak for p in _np_params], dtype=float)
    p_two_over_pi = np.array([p.two_over_pi for p in _np_params], dtype=float)
    p_cap_cgs1 = np.array([p.cap_cgs1 for p in _np_params], dtype=float)
    p_cap_cgd1 = np.array([p.cap_cgd1 for p in _np_params], dtype=float)
    p_cap_half_wl_ci = np.array([p.cap_half_wl_ci for p in _np_params], dtype=float)
    p_cap_cgs3_base = np.array([p.cap_cgs3_base for p in _np_params], dtype=float)
    p_cap_cgd3_base = np.array([p.cap_cgd3_base for p in _np_params], dtype=float)
    p_k1 = np.array([p.k1 for p in _np_params], dtype=float)
    p_gate_leak_g = np.array([p.gate_leak_g for p in _np_params], dtype=float)
    op_cache_valid = np.zeros(len(dev_meta), dtype=np.bool_)
    op_cache_vs1 = np.zeros(len(dev_meta), dtype=float)
    op_cache_vd1 = np.zeros(len(dev_meta), dtype=float)

    res_a_kind, res_a_ref, res_a_val = _term_arrays([item[0] for item in res_meta])
    res_b_kind, res_b_ref, res_b_val = _term_arrays([item[1] for item in res_meta])
    res_ai = _index_array(item[2] for item in res_meta)
    res_bi = _index_array(item[3] for item in res_meta)
    res_g = np.array([item[4] for item in res_meta], dtype=float)

    cap_a_kind, cap_a_ref, cap_a_val = _term_arrays([item[0] for item in load_meta])
    cap_b_kind, cap_b_ref, cap_b_val = _term_arrays([item[1] for item in load_meta])
    cap_ai = _index_array(item[2] for item in load_meta)
    cap_bi = _index_array(item[3] for item in load_meta)
    cap_value = np.array([item[4] for item in load_meta], dtype=float)

    isrc_pi = _index_array(item[0] for item in isrc_meta)
    isrc_qi = _index_array(item[1] for item in isrc_meta)
    isrc_value = np.array([item[2] for item in isrc_meta], dtype=float)

    vccs_pi  = _index_array(item[0] for item in vccs_meta)
    vccs_qi  = _index_array(item[1] for item in vccs_meta)
    # For control nodes, extract solved index from the terminal tuple; rails → -1
    vccs_cpi = _index_array(
        item[2][1] if item[2][0] == 0 else None for item in vccs_meta)
    vccs_cni = _index_array(
        item[3][1] if item[3][0] == 0 else None for item in vccs_meta)
    vccs_gm  = np.array([item[4] for item in vccs_meta], dtype=float)

    dyn_pi = _index_array(item[0] for item in dyn_isrc_meta)
    dyn_qi = _index_array(item[1] for item in dyn_isrc_meta)
    dyn_input_idx = np.array([item[2] for item in dyn_isrc_meta], dtype=np.int64)
    if rail_margin is None and getattr(topo, "require_dc_in_box", False):
        rail_margin = 2.0
    clip_lo = np.inf
    clip_hi = -np.inf
    if rail_margin is not None:
        rails = [v for v in plan.rails.values() if isinstance(v, (int, float))]
        if rails:
            clip_lo = min(rails) - float(rail_margin)
            clip_hi = max(rails) + float(rail_margin)
    # Ideal voltage sources add branch-current unknowns (n_aug > n); the numba kernels
    # are fixed at n nodes, so those circuits run on the pure-Python n_aug path instead.
    use_numba_newton = transient_newton_numba is not None and n_aug == n
    numba_newton_attempts = 0
    numba_newton_success = 0
    numba_newton_fallback = 0

    # ── initial condition: DC op at static bias ──
    if V0 is None:
        ac = ac_solve(sizes, bias, np.array([1.0]), nf=nf, topo=topo,
                      corner=corner)
        dc = ac["dc_op"]
        V0 = np.array([dc[name] for name in topo.solved])
    V0 = np.asarray(V0, float)
    if V0.shape[0] < n_aug:                  # pad ideal-source branch currents (seed 0)
        V0 = np.concatenate([V0, np.zeros(n_aug - V0.shape[0])])
    if len(vs_meta) and input_values.shape[1] > 0:
        input0 = input_values[:, 0]
        Vseed = V0.copy()
        for aterm, bterm, pi, qi, _bi, e_const, e_idx in vs_meta:
            E = e_const if e_idx < 0 else input0[e_idx]
            if pi is not None and qi is not None:
                Vseed[pi] = Vseed[qi] + E
            elif pi is not None:
                Vseed[pi] = termv(bterm, Vseed, input0) + E
            elif qi is not None:
                Vseed[qi] = termv(aterm, Vseed, input0) - E
        V0 = Vseed
    Vhist = np.zeros((N, n_aug)); Vhist[0] = V0
    gmin = 1e-12

    HH = 1e-3   # finite-diff step for gm/gds (matches get_ss_params, Cadence-calibrated)

    nfail = 0
    nretry = 0
    nsubsteps = 0
    max_retry_subdivisions = int(max_retry_subdivisions or 0)
    max_step = None if max_step is None else float(max_step)
    flat_max_step = None if flat_max_step is None else float(flat_max_step)
    used_grid_numba = False
    partial_grid_numba = False
    python_start_idx = 1
    profile = bool(profile)
    profile_wall_s = 0.0
    profile_stats = None
    numba_grid_error = None
    numba_grid_failed_index = None
    numba_grid_failed_substeps = 0
    numba_grid_failed_profile = None
    numba_grid_failed_intervals = None
    if edge_mask is None:
        edge_mask_arr = np.empty(0, dtype=np.bool_)
    else:
        edge_mask_arr = np.asarray(edge_mask, dtype=np.bool_)
        if len(edge_mask_arr) != N:
            raise ValueError("edge_mask length must match tgrid")

    gear2_done = False
    gear2_numba_used = False
    gear2_python_retry_used = False
    gear2_retry_requested = (
        integration_method == "gear2" and gear2_be_fallback and
        (max_retry_subdivisions > 0 or
         (max_step is not None and max_step > 0.0) or
         (flat_max_step is not None and flat_max_step > 0.0))
    )

    # Marshalled solve context shared by module-level transient kernels.  It holds
    # the compiled topology arrays, Python metadata, scalar solver settings, and
    # the mutable PMOS op-point cache used by the numba Newton kernels.
    ctx = SimpleNamespace(
        plan=plan, idx=idx, n=n, n_aug=n_aug, termv=termv, tft=tft,
        dev_d_kind=dev_d_kind, dev_d_ref=dev_d_ref, dev_d_val=dev_d_val,
        dev_g_kind=dev_g_kind, dev_g_ref=dev_g_ref, dev_g_val=dev_g_val,
        dev_s_kind=dev_s_kind, dev_s_ref=dev_s_ref, dev_s_val=dev_s_val,
        dev_di=dev_di, dev_gi=dev_gi, dev_si=dev_si, dev_use_abs=dev_use_abs,
        dev_objs=dev_objs,
        p_Vfb=p_Vfb, p_Vss=p_Vss, p_Lc=p_Lc, p_lambda=p_lambda,
        p_contact_scale=p_contact_scale, p_exponent=p_exponent,
        p_current_scale=p_current_scale, p_inv_Rleak=p_inv_Rleak,
        p_two_over_pi=p_two_over_pi, p_cap_cgs1=p_cap_cgs1, p_cap_cgd1=p_cap_cgd1,
        p_cap_half_wl_ci=p_cap_half_wl_ci, p_cap_cgs3_base=p_cap_cgs3_base,
        p_cap_cgd3_base=p_cap_cgd3_base, p_k1=p_k1, p_gate_leak_g=p_gate_leak_g,
        op_cache_valid=op_cache_valid, op_cache_vs1=op_cache_vs1,
        op_cache_vd1=op_cache_vd1,
        res_a_kind=res_a_kind, res_a_ref=res_a_ref, res_a_val=res_a_val,
        res_b_kind=res_b_kind, res_b_ref=res_b_ref, res_b_val=res_b_val,
        res_ai=res_ai, res_bi=res_bi, res_g=res_g,
        cap_a_kind=cap_a_kind, cap_a_ref=cap_a_ref, cap_a_val=cap_a_val,
        cap_b_kind=cap_b_kind, cap_b_ref=cap_b_ref, cap_b_val=cap_b_val,
        cap_ai=cap_ai, cap_bi=cap_bi, cap_value=cap_value,
        isrc_pi=isrc_pi, isrc_qi=isrc_qi, isrc_value=isrc_value,
        vccs_pi=vccs_pi, vccs_qi=vccs_qi, vccs_cpi=vccs_cpi, vccs_cni=vccs_cni,
        vccs_gm=vccs_gm,
        dyn_pi=dyn_pi, dyn_qi=dyn_qi, dyn_input_idx=dyn_input_idx,
        dev_meta=dev_meta, load_meta=load_meta, res_meta=res_meta,
        isrc_meta=isrc_meta, vccs_meta=vccs_meta, vs_meta=vs_meta,
        vcvs_meta=vcvs_meta, cccs_meta=cccs_meta, ccvs_meta=ccvs_meta,
        dyn_isrc_meta=dyn_isrc_meta,
        gmin=gmin, HH=HH, clip_lo=clip_lo, clip_hi=clip_hi, cap_id=_cap_id,
        use_numba_newton=use_numba_newton, rail_margin=rail_margin,
        numba_newton_attempts=0, numba_newton_success=0,
        numba_newton_fallback=0,
        newton_maxit=newton_maxit, newton_step_limit=newton_step_limit,
        newton_vtol=newton_vtol, fallback_full_jacobian=fallback_full_jacobian,
        fallback_least_squares=fallback_least_squares, fallback_tol=fallback_tol,
        max_step=max_step, flat_max_step=flat_max_step,
        max_retry_subdivisions=max_retry_subdivisions, edge_mask_arr=edge_mask_arr,
        signed_devices=signed_devices, gear2_be_fallback=gear2_be_fallback,
        integration_method=integration_method,
        adaptive=adaptive, adaptive_reltol=adaptive_reltol,
        adaptive_vabstol=adaptive_vabstol, adaptive_iabstol=adaptive_iabstol,
        adaptive_max_steps=adaptive_max_steps, adaptive_h0=adaptive_h0,
    )

    fixed_g2 = _solve_fixed_gear2_numba(
        ctx, V0, tgrid, input_values, edge_mask_arr, profile,
        gear2_retry_requested)
    if fixed_g2.numba_grid_error is not None:
        numba_grid_error = fixed_g2.numba_grid_error
    if fixed_g2.profile_wall_s is not None:
        profile_wall_s = fixed_g2.profile_wall_s
    if fixed_g2.numba_grid_failed_index is not None:
        numba_grid_failed_index = fixed_g2.numba_grid_failed_index
        numba_grid_failed_substeps = int(fixed_g2.numba_grid_failed_substeps)
        numba_grid_failed_profile = fixed_g2.numba_grid_failed_profile
        numba_grid_failed_intervals = fixed_g2.numba_grid_failed_intervals
    if fixed_g2.handled:
        Vhist = fixed_g2.Vhist
        nsubsteps = int(fixed_g2.nsubsteps)
        nretry = int(fixed_g2.nretry)
        nfail = int(fixed_g2.nfail)
        gear2_done = True
        gear2_numba_used = True
        profile_stats = fixed_g2.profile_stats

    adaptive_used = False
    adaptive_numba_used = False
    if adaptive and integration_method == "gear2" and not gear2_done:
        adaptive_nb = _solve_adaptive_gear2_numba(ctx, V0, tgrid, input_values, profile)
        if adaptive_nb.numba_grid_error is not None:
            numba_grid_error = adaptive_nb.numba_grid_error
        if adaptive_nb.handled:
            tgrid = adaptive_nb.tgrid
            Vhist = adaptive_nb.Vhist
            input_values = adaptive_nb.input_values
            N = int(adaptive_nb.N)
            nsubsteps = int(adaptive_nb.nsubsteps)
            nretry = int(adaptive_nb.nretry)
            profile_stats = adaptive_nb.profile_stats
            profile_wall_s = adaptive_nb.profile_wall_s
            gear2_done = True
            adaptive_used = True
            adaptive_numba_used = True
            gear2_numba_used = True

    if adaptive and integration_method == "gear2" and not gear2_done:
        adaptive_py = _solve_adaptive_gear2_python(
            ctx, V0, tgrid, input_values, nfail, nretry, nsubsteps)
        if adaptive_py.handled:
            tgrid = adaptive_py.tgrid
            Vhist = adaptive_py.Vhist
            input_values = adaptive_py.input_values
            N = int(adaptive_py.N)
            nfail = int(adaptive_py.nfail)
            nretry = int(adaptive_py.nretry)
            nsubsteps = int(adaptive_py.nsubsteps)
            gear2_done = True
            adaptive_used = True

    if (not adaptive) and integration_method == "gear2" and not gear2_done:
        g2_py = _solve_fixed_gear2_python(
            ctx, Vhist, tgrid, input_values, edge_mask_arr,
            gear2_retry_requested)
        if g2_py.handled:
            Vhist = g2_py.Vhist
            nfail = int(g2_py.nfail)
            nretry = int(g2_py.nretry)
            nsubsteps = int(g2_py.nsubsteps)
            gear2_done = True
            gear2_python_retry_used = bool(g2_py.gear2_python_retry_used)

    # Graceful fallback: gear2's single-step Newton stalls on stiff transients
    # (e.g. the chopper switch edges), where it can fail a large fraction of
    # steps and drift.  When too many steps fail, the gear2 result is unreliable,
    # so re-run with the robust backward-Euler path (recursive bisection + LS).
    # The PSS/periodic path opts out (gear2_be_fallback=False): shooting manages
    # its own convergence and must not mix a BE orbit into the gear2 iteration.
    if ((not adaptive) and integration_method == "gear2" and gear2_done and gear2_be_fallback and
            nfail > max(8, int(0.10 * (N - 1)))):
        be_result = transient(
            sizes, bias, tgrid, vip=vip, vin=vin, nf=nf, V0=V0, topo=topo,
            inputs=inputs, node_inputs=node_inputs, current_inputs=current_inputs,
            corner=corner, max_step=max_step, flat_max_step=flat_max_step,
            max_retry_subdivisions=max_retry_subdivisions,
            newton_maxit=newton_maxit, newton_step_limit=newton_step_limit,
            newton_vtol=newton_vtol,
            fallback_full_jacobian=fallback_full_jacobian,
            fallback_least_squares=fallback_least_squares, fallback_tol=fallback_tol,
            signed_devices=signed_devices, profile=profile, edge_mask=edge_mask,
            rail_margin=rail_margin, integration_method="be",
            gear2_be_fallback=False)
        be_result["gear2_be_fallback_used"] = True
        be_result["gear2_nfail_before_fallback"] = int(nfail)
        return be_result

    if (not adaptive) and not gear2_done:
        be_nb = _solve_be_numba(ctx, V0, tgrid, input_values, edge_mask_arr, profile)
        if be_nb.numba_grid_error is not None:
            numba_grid_error = be_nb.numba_grid_error
            used_grid_numba = False
        if be_nb.profile_wall_s is not None:
            profile_wall_s = be_nb.profile_wall_s
        if be_nb.numba_grid_failed_intervals is not None:
            numba_grid_failed_intervals = be_nb.numba_grid_failed_intervals
        if be_nb.handled:
            Vhist = be_nb.Vhist
            nsubsteps = int(be_nb.nsubsteps)
            nfail = int(be_nb.nfail)
            nretry = int(be_nb.nretry)
            used_grid_numba = True
            profile_stats = be_nb.profile_stats
        elif be_nb.numba_grid_failed_index is not None:
            numba_grid_failed_index = be_nb.numba_grid_failed_index
            numba_grid_failed_substeps = int(be_nb.numba_grid_failed_substeps)
            numba_grid_failed_profile = be_nb.numba_grid_failed_profile
            if be_nb.partial_grid_numba:
                Vhist = be_nb.Vhist
                nsubsteps = int(be_nb.nsubsteps)
                partial_grid_numba = True
                python_start_idx = int(be_nb.python_start_idx)

    if not gear2_done and not used_grid_numba:
        be_py = _solve_be_python(
            ctx, Vhist, tgrid, input_values, edge_mask_arr,
            python_start_idx, nfail, nretry, nsubsteps)
        Vhist = be_py.Vhist
        nfail = int(be_py.nfail)
        nretry = int(be_py.nretry)
        nsubsteps = int(be_py.nsubsteps)

    return _assemble_result(
        ctx, tgrid, Vhist, input_values, input_keys,
        nfail, nretry, nsubsteps,
        adaptive_used, adaptive_numba_used,
        used_grid_numba, gear2_numba_used,
        gear2_python_retry_used,
        profile, profile_stats, profile_wall_s,
        partial_grid_numba, numba_grid_error,
        numba_grid_failed_index, numba_grid_failed_substeps,
        numba_grid_failed_profile, numba_grid_failed_intervals)



# ── self-consistency check vs the validated DC / AC solvers ──
if __name__ == "__main__":
    sizes = {"M6": (2264, 78), "M7": (61365, 61), "M8": (61365, 61),
             "M9": (3175, 468), "M10": (3175, 468), "M11": (465, 66),
             "M12": (894, 85), "M13": (894, 85), "M14": (5224, 46), "M15": (5224, 46)}
    bias = {"VDD": 40.0, "VCM": 30.65, "VB": 9.84, "VC": 16.0}

    # AC reference gain/BW (the transient must reproduce both)
    ac = ac_solve(sizes, bias, np.logspace(0, 4, 80))
    gain = ac["gains"].max(); bw = ac["bw_Hz"]; tau = 1 / (2 * np.pi * bw)
    print(f"AC ref: gain={gain:.4f} ({20*np.log10(gain):.2f} dB), BW={bw:.0f} Hz, tau={tau*1e3:.3f} ms")

    T = 0.004; N = 200; t = np.linspace(0, T, N)        # h≈20 µs (physically fine)
    vcm = np.full(N, bias["VCM"])

    # (1) steady state: hold vip=vin=VCM -> must sit at the DC op (no CM run-away)
    r0 = transient(sizes, bias, t, vcm, vcm)
    print(f"(1) quiescent drift = {r0['vout'][-1]*1e6:+.4f} µV  nfail={r0['nfail']}/{N-1}  (期望 0)")

    # (2) differential step vip-vin=1 mV at 0.5 ms -> settles to the small-signal gain
    dstep = 0.5e-3; ts = 0.5e-3
    vp = vcm + np.where(t >= ts, +dstep, 0.0)
    vn = vcm - np.where(t >= ts, +dstep, 0.0)
    r = transient(sizes, bias, t, vp, vn); vo = r["vout"]; settled = vo[-1]
    post = np.where(t >= ts)[0]; hit = post[np.abs(vo[post]) >= abs(settled) * (1 - np.exp(-1))]
    tau_tr = (t[hit[0]] - ts) if len(hit) else float("nan")
    print(f"(2) step 1mV: settled={settled*1e3:.4f} mV  gain={settled/(2*dstep):+.4f} (AC {gain:.4f})  "
          f"nfail={r['nfail']}/{N-1}")
    print(f"    tau(63%)={tau_tr*1e3:.3f} ms vs AC tau={tau*1e3:.3f} ms (multi-pole, ~order match)")
