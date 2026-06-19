"""
Nonlinear transient solver for the AFE (backward-Euler + per-step Newton).

Integrates the full 6-node circuit DAE in time. Built on the same device model
(PMOS_TFT.get_Idc / get_capacitances) and topology (AFE_TOPO) as the DC/AC/Noise
stack, so the steady state matches the DC solver and the small-signal response
matches the AC solver.

Method
------
- KCL at every solved node:  Σ device currents  +  Σ capacitor currents  = 0.
- Capacitor companion (backward Euler), cap C between terminals a,b:
      i_ab = C(V_n)·d(Va-Vb)/dt ≈ (C(V_n)/h)·[(Va-Vb)_n − (Va-Vb)_{n-1}]
  C is evaluated IMPLICITLY at the current step (Cgss/Cgdd from get_capacitances,
  re-evaluated each Newton iteration) — this is exactly the model's `Cgss(V)·ddt(V)`
  form, so it tracks the bias-dependent cap on fast edges (chopper), not just slow
  signals. The AT_4000TG model routes these caps through an internal gate1 node;
  this solver keeps the long-timescale R_cap2 leakage from source/drain to gate1
  and collapses the 100 Ω gate-to-gate1 RC because its ns-scale time constant is
  far below the chopper edge times used here.
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
import time

import numpy as np
try:
    from .pmos_tft_model import PMOS_TFT
    from .topology import AFE_TOPO
    from .ac_solver import ac_solve, _dev_nf
    from .numba_kernels import (
        terminal_derivatives_numba,
        transient_newton_numba,
        transient_solve_grid_numba,
    )
    from .compiled_topology import CompiledTopology
except ImportError:  # pragma: no cover - legacy direct module import
    from pmos_tft_model import PMOS_TFT
    from topology import AFE_TOPO
    from ac_solver import ac_solve, _dev_nf
    from compiled_topology import CompiledTopology
    try:
        from numba_kernels import (
            terminal_derivatives_numba,
            transient_newton_numba,
            transient_solve_grid_numba,
        )
    except Exception:
        terminal_derivatives_numba = None
        transient_newton_numba = None
        transient_solve_grid_numba = None


def transient(sizes, bias, tgrid, vip=None, vin=None, nf=None, V0=None,
              topo=AFE_TOPO, inputs=None, node_inputs=None, current_inputs=None,
              max_step=None, flat_max_step=None,
              max_retry_subdivisions=0, newton_maxit=30,
              newton_step_limit=5.0, newton_vtol=1e-8,
              fallback_full_jacobian=False,
              fallback_least_squares=False, fallback_tol=1e-9,
              signed_devices=None, profile=False, edge_mask=None,
              rail_margin=None):
    """Backward-Euler transient.
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
    devs = topo.devices
    tft = {name: PMOS_TFT(W=sizes[name][0], L=sizes[name][1], NF=_dev_nf(nf, name))
           for name, *_ in devs}
    tgrid = np.asarray(tgrid, float)
    N = len(tgrid)
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
    termv = plan.term_value
    signed_devices = set(signed_devices or ())
    dev_meta = [(tft[item.name], item.name in signed_devices,
                 item.d, item.g, item.s, item.di, item.gi, item.si)
                for item in plan.devices]
    load_meta = [(item.a, item.b, item.ai, item.bi, item.value)
                 for item in plan.capacitors]
    res_meta = [(item.a, item.b, item.ai, item.bi, item.g)
                for item in plan.resistors]
    isrc_meta = [(item.pi, item.qi, item.value) for item in plan.isources]
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

    dev_d_kind, dev_d_ref, dev_d_val = _term_arrays([item[2] for item in dev_meta])
    dev_g_kind, dev_g_ref, dev_g_val = _term_arrays([item[3] for item in dev_meta])
    dev_s_kind, dev_s_ref, dev_s_val = _term_arrays([item[4] for item in dev_meta])
    dev_di = _index_array(item[5] for item in dev_meta)
    dev_gi = _index_array(item[6] for item in dev_meta)
    dev_si = _index_array(item[7] for item in dev_meta)
    dev_use_abs = np.array([not item[1] for item in dev_meta], dtype=np.bool_)
    dev_objs = [item[0] for item in dev_meta]
    p_Vfb = np.array([dev.Vfb for dev in dev_objs], dtype=float)
    p_Vss = np.array([dev.Vss for dev in dev_objs], dtype=float)
    p_Lc = np.array([dev.Lc for dev in dev_objs], dtype=float)
    p_lambda = np.array([dev.lambda_ for dev in dev_objs], dtype=float)
    p_contact_scale = np.array([dev._contact_scale for dev in dev_objs], dtype=float)
    p_exponent = np.array([dev._channel_exponent for dev in dev_objs], dtype=float)
    p_current_scale = np.array([dev._current_scale for dev in dev_objs], dtype=float)
    p_inv_Rleak = np.array([dev._inv_Rleak for dev in dev_objs], dtype=float)
    p_two_over_pi = np.array([dev._two_over_pi for dev in dev_objs], dtype=float)
    p_cap_cgs1 = np.array([dev._cap_cgs1 for dev in dev_objs], dtype=float)
    p_cap_cgd1 = np.array([dev._cap_cgd1 for dev in dev_objs], dtype=float)
    p_cap_half_wl_ci = np.array([dev._cap_half_wl_ci for dev in dev_objs], dtype=float)
    p_cap_cgs3_base = np.array([dev._cap_cgs3_base for dev in dev_objs], dtype=float)
    p_cap_cgd3_base = np.array([dev._cap_cgd3_base for dev in dev_objs], dtype=float)
    p_k1 = np.array([dev.k1 for dev in dev_objs], dtype=float)
    p_gate_leak_g = np.array([1.0 / dev.R_cap2 for dev in dev_objs], dtype=float)
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
    use_numba_newton = transient_newton_numba is not None
    numba_newton_attempts = 0
    numba_newton_success = 0
    numba_newton_fallback = 0

    def device_states(V, input_vals):
        """Per-Newton operating data shared by residual and Jacobian."""
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
                I = I_d1_d if signed else abs(Idc)       # signed for bidirectional switches
            except Exception:
                I = Cgs = Cgd = 0.0
                Vs1 = Vd1 = 0.0
            out[pos] = (dev, signed, di, gi, si, Vs, Vd, Vg, Vs1, Vd1, I, Cgs, Cgd)
        return out

    def device_terminal_values(V, input_vals):
        return [(termv(sterm, V, input_vals),
                 termv(dterm, V, input_vals),
                 termv(gterm, V, input_vals))
                for _, _, dterm, gterm, sterm, *_ in dev_meta]

    def load_cap_dv_values(V, input_vals):
        return [termv(aterm, V, input_vals) - termv(bterm, V, input_vals)
                for aterm, bterm, _, _, _ in load_meta]

    # ── initial condition: DC op at static bias ──
    if V0 is None:
        ac = ac_solve(sizes, bias, np.array([1.0]), nf=nf, topo=topo)
        dc = ac["dc_op"]
        V0 = np.array([dc[name] for name in topo.solved])
    Vhist = np.zeros((N, n)); Vhist[0] = V0
    gmin = 1e-12

    def step_residual(V, input_now, states, prev_dev_terms, load_prev_dv, h):
        R = np.zeros(n)
        inv_h = 1.0 / h
        # device DC currents (into-node convention: +Id at drain, -Id at source)
        for _, _, di, _, si, _, _, _, _, _, I, _, _ in states:
            if di is not None:
                R[di] += I                               # KCL sign (drain current INTO
            if si is not None:
                R[si] -= I                               # node); raw get_Idc is negative
        for aterm, bterm, ai, bi, gval in res_meta:      # resistor branch current a -> b
            i_ab = (termv(aterm, V, input_now) - termv(bterm, V, input_now)) * gval
            if ai is not None:
                R[ai] -= i_ab
            if bi is not None:
                R[bi] += i_ab
        for pi, qi, Ival in isrc_meta:                   # ideal DC current source p -> q
            if pi is not None:
                R[pi] -= Ival
            if qi is not None:
                R[qi] += Ival
        for pi, qi, key_idx in dyn_isrc_meta:            # time-varying source p -> q
            Ival = input_now[key_idx]
            if pi is not None:
                R[pi] -= Ival
            if qi is not None:
                R[qi] += Ival
        for k in range(n):
            R[k] -= V[k] * gmin
        # capacitor companion: C(V_now)·[(Va-Vb)_now - (Va-Vb)_prev]/h  (C implicit; Cmap
        # is evaluated at the current iterate by the caller, ΔV_prev uses the prev step)
        for state, prev_terms in zip(states, prev_dev_terms):
            _, _, di, gi, si, Vs, Vd, Vg, _, _, _, Cgs, Cgd = state
            pVs, pVd, pVg = prev_terms
            gate_leak_g = 1.0 / state[0].R_cap2
            if gate_leak_g != 0.0:
                i_sg = (Vs - Vg) * gate_leak_g          # source -> gate1≈gate
                if si is not None:
                    R[si] -= i_sg
                if gi is not None:
                    R[gi] += i_sg
                i_dg = (Vd - Vg) * gate_leak_g          # drain -> gate1≈gate
                if di is not None:
                    R[di] -= i_dg
                if gi is not None:
                    R[gi] += i_dg
            if Cgs != 0.0:
                i_ab = Cgs * inv_h * ((Vg - Vs) - (pVg - pVs))  # gate-source
                if gi is not None:
                    R[gi] -= i_ab
                if si is not None:
                    R[si] += i_ab
            if Cgd != 0.0:
                i_ab = Cgd * inv_h * ((Vg - Vd) - (pVg - pVd))  # gate-drain
                if gi is not None:
                    R[gi] -= i_ab
                if di is not None:
                    R[di] += i_ab
        for (aterm, bterm, ai, bi, cap), dv_pre in zip(load_meta, load_prev_dv):
            if cap != 0.0:
                dv_now = termv(aterm, V, input_now) - termv(bterm, V, input_now)
                i_ab = cap * inv_h * (dv_now - dv_pre)
                if ai is not None:
                    R[ai] -= i_ab
                if bi is not None:
                    R[bi] += i_ab
        return R

    HH = 1e-3   # finite-diff step for gm/gds (matches get_ss_params, Cadence-calibrated)

    def terminal_derivatives(dev, signed, Vs, Vd, Vg, Vs1, Vd1, need_gm, need_gds):
        """Terminal gm/gds via implicit differentiation of the solved internal nodes.

        This is algebraically the same derivative that finite-differencing get_Idc
        measures, but avoids solving the 2-node internal OP four extra times per
        device per Newton iteration.
        """
        if not need_gm and not need_gds:
            return 0.0, 0.0
        if terminal_derivatives_numba is not None:
            try:
                ok, gm, gds = terminal_derivatives_numba(
                    Vs, Vd, Vg, Vs1, Vd1, need_gm, need_gds, not signed, HH, 1e-6,
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
        if abs(Idc0) < 1e-30:
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
        current_sign = sign if not signed else -1.0

        def deriv(vs_p, vd_p, vg_p, vs_m, vd_m, vg_m):
            Fpa, Fpb, Ip = eval_at(vs_p, vd_p, vg_p, Vs1, Vd1)
            Fma, Fmb, Im = eval_at(vs_m, vd_m, vg_m, Vs1, Vd1)
            fu0 = (Fpa - Fma) / (2 * HH)
            fu1 = (Fpb - Fmb) / (2 * HH)
            Iu = (Ip - Im) / (2 * HH)
            y0 = (j11 * fu0 - j01 * fu1) / det
            y1 = (-j10 * fu0 + j00 * fu1) / det
            return current_sign * (Iu - ix0 * y0 - ix1 * y1)

        gm = deriv(Vs, Vd, Vg + HH, Vs, Vd, Vg - HH) if need_gm else 0.0
        gds = deriv(Vs, Vd + HH, Vg, Vs, Vd - HH, Vg) if need_gds else 0.0
        return gm, gds

    def build_jac(V, states, h):
        """Analytic conductance Jacobian dR/dV (n×n).

        Device part = small-signal conductance stamp: gm=dI/dVg, gds=dI/dVd
        (finite-diff of get_Idc, same as get_ss_params), dI/dVs=-(gm+gds).
        Only SOLVED terminals get a column (driven inputs / rails are fixed).
        Plus capacitor companion (C/h) and gmin on the diagonal. A well-scaled
        Jacobian is exactly what bare fsolve lacked — it lets Newton track the
        correct (physical) branch of the positive-feedback circuit."""
        J = np.zeros((n, n))
        inv_h = 1.0 / h
        for dev, signed, di, gi, si, Vs, Vd, Vg, Vs1, Vd1, _, _, _ in states:
            get_idc = dev.get_Idc
            need_gm = gi is not None or si is not None
            need_gds = di is not None or si is not None
            try:                                         # abs() to match the residual's
                gm, gds = terminal_derivatives(dev, signed, Vs, Vd, Vg, Vs1, Vd1,
                                               need_gm, need_gds)
            except Exception:
                try:
                    current = (lambda vs, vd, vg: -get_idc(vs, vd, vg)) if signed else (
                        lambda vs, vd, vg: abs(get_idc(vs, vd, vg)))
                    gm = ((current(Vs, Vd, Vg + HH) - current(Vs, Vd, Vg - HH)) / (2 * HH)
                          if need_gm else 0.0)
                    gds = ((current(Vs, Vd + HH, Vg) - current(Vs, Vd - HH, Vg)) / (2 * HH)
                           if need_gds else 0.0)
                except Exception:
                    gm, gds = 0.0, 1e-12 if need_gds else 0.0
            dI_dVs = -(gm + gds)
            if di is not None:                          # row d: R[d] += I
                J[di, di] += gds
                if gi is not None:
                    J[di, gi] += gm
                if si is not None:
                    J[di, si] += dI_dVs
            if si is not None:                          # row s: R[s] -= I
                if di is not None:
                    J[si, di] -= gds
                if gi is not None:
                    J[si, gi] -= gm
                J[si, si] -= dI_dVs
        for k in range(n):
            J[k, k] -= gmin
        for dev, _, di, gi, si, _, _, _, _, _, _, Cgs, Cgd in states:
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
            if Cgs != 0.0:                              # gate-source
                g = Cgs * inv_h
                if gi is not None:
                    J[gi, gi] -= g
                    if si is not None:
                        J[gi, si] += g
                if si is not None:
                    J[si, si] -= g
                    if gi is not None:
                        J[si, gi] += g
            if Cgd != 0.0:                              # gate-drain
                g = Cgd * inv_h
                if gi is not None:
                    J[gi, gi] -= g
                    if di is not None:
                        J[gi, di] += g
                if di is not None:
                    J[di, di] -= g
                    if gi is not None:
                        J[di, gi] += g
        for _, _, ai, bi, cap in load_meta:
            if cap != 0.0:
                g = cap * inv_h
                if ai is not None:
                    J[ai, ai] -= g
                    if bi is not None:
                        J[ai, bi] += g
                if bi is not None:
                    J[bi, bi] -= g
                    if ai is not None:
                        J[bi, ai] += g
        for _, _, ai, bi, gval in res_meta:             # resistor conductance 1/R
            if gval != 0.0:
                if ai is not None:
                    J[ai, ai] -= gval
                    if bi is not None:
                        J[ai, bi] += gval
                if bi is not None:
                    J[bi, bi] -= gval
                    if ai is not None:
                        J[bi, ai] += gval
        # ideal DC current sources are constant -> no Jacobian contribution.
        return J

    def newton(seed, Vp, input_now, input_prev, h, maxit=None, vtol=1e-8):
        """Full-step Newton with the analytic Jacobian, converged on STEP SIZE |ΔV|.

        No residual-decrease line search: on this stiff cap-dominated system (gC=C/h
        dominates gm/gds at small h) the full Newton step briefly *raises* the residual
        while the capacitive feed-through settles, so a residual-monotone line search
        stalls (λ→0), returns a half-solved V, and the huge gC amplifies that error into
        an oscillation across timesteps. Full steps converge quadratically from the
        previous-step seed (verified: |R| 6e-9→1e-13→1e-16 in 2 iters at h=10µs).
        |ΔV|≤5 V/iter caps branch jumps; also accept once |ΔV| stalls at its floor.

        Caps are evaluated IMPLICITLY (at the current iterate V), matching the Verilog-A
        `Cgss(V)·ddt(V)`. Freezing them at the previous step is fine for slow signals but
        lags on fast edges (chopper) where Cgss/Cgdd swing with bias — implicit keeps the
        engine faithful there."""
        maxit = int(newton_maxit if maxit is None else maxit)
        step_limit = float(newton_step_limit)
        if use_numba_newton:
            nonlocal numba_newton_attempts, numba_newton_success, numba_newton_fallback
            numba_newton_attempts += 1
            try:
                Vn, iters, ok, usable = transient_newton_numba(
                    np.asarray(seed, float), np.asarray(Vp, float),
                    np.asarray(input_now, float), np.asarray(input_prev, float),
                    float(h), int(n), maxit, step_limit, float(vtol),
                    float(gmin),
                    bool(fallback_full_jacobian or fallback_least_squares),
                    float(fallback_tol), float(HH),
                    dev_d_kind, dev_d_ref, dev_d_val,
                    dev_g_kind, dev_g_ref, dev_g_val,
                    dev_s_kind, dev_s_ref, dev_s_val,
                    dev_di, dev_gi, dev_si, dev_use_abs,
                    p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
                    p_current_scale, p_inv_Rleak,
                    p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
                    p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
                    op_cache_valid, op_cache_vs1, op_cache_vd1,
                    res_a_kind, res_a_ref, res_a_val,
                    res_b_kind, res_b_ref, res_b_val, res_ai, res_bi, res_g,
                    cap_a_kind, cap_a_ref, cap_a_val,
                    cap_b_kind, cap_b_ref, cap_b_val, cap_ai, cap_bi, cap_value,
                    isrc_pi, isrc_qi, isrc_value,
                    dyn_pi, dyn_qi, dyn_input_idx,
                    float(clip_lo), float(clip_hi),
                )
                if ok:
                    numba_newton_success += 1
                    return Vn, int(iters), True
                if usable:
                    seed = Vn
            except Exception:
                pass
            numba_newton_fallback += 1

        V = np.array(seed, float)
        prev = np.inf
        prev_dev_terms = device_terminal_values(Vp, input_prev)
        load_prev_dv = load_cap_dv_values(Vp, input_prev)
        for it in range(maxit):
            states = device_states(V, input_now)         # implicit caps at current iterate
            R = step_residual(V, input_now, states, prev_dev_terms, load_prev_dv, h)
            if ((fallback_full_jacobian or fallback_least_squares) and
                    np.linalg.norm(R, ord=np.inf) < float(fallback_tol)):
                return V, it + 1, True
            J = build_jac(V, states, h)
            try:
                dV = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                dV = np.linalg.lstsq(J, -R, rcond=None)[0]
            mx = np.max(np.abs(dV))
            if mx > step_limit:
                dV *= step_limit / mx; mx = step_limit    # branch-safety step cap
            V = V + dV
            if clip_lo <= clip_hi:
                V = np.clip(V, clip_lo, clip_hi)
            if mx < vtol:
                if fallback_full_jacobian or fallback_least_squares:
                    # Stiff chopper edges can produce a tiny Newton update at a
                    # still-large KCL residual.  Let the next iteration verify
                    # the residual, otherwise fall through to the robust fallback.
                    continue
                return V, it + 1, True
            if it >= 4 and mx >= prev and mx < 1e-5:     # stalled at the numeric floor
                if fallback_full_jacobian or fallback_least_squares:
                    continue
                return V, it + 1, True
            prev = mx
        return V, maxit, False

    def newton_full_jac(seed, Vp, input_now, input_prev, h, maxit=6,
                        residual_tol=1e-8, vtol=1e-8):
        """Fallback Newton using a finite-difference Jacobian of the full residual.

        The analytic Jacobian intentionally omits dC/dV terms for speed. That is
        fine for ordinary AFE steps, but chopper clock edges can move the PMOS
        capacitances enough that those terms decide convergence. This path is
        expensive, so it is only used after the fast Newton has failed.
        """
        V = np.array(seed, float)
        prev_dev_terms = device_terminal_values(Vp, input_prev)
        load_prev_dv = load_cap_dv_values(Vp, input_prev)
        step_limit = float(newton_step_limit)

        def residual_at(z):
            states = device_states(z, input_now)
            return step_residual(z, input_now, states, prev_dev_terms,
                                 load_prev_dv, h)

        for _ in range(int(maxit)):
            R = residual_at(V)
            norm = np.linalg.norm(R, ord=np.inf)
            if norm < residual_tol:
                return V, True
            J = np.zeros((n, n))
            for col in range(n):
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

    nfail = 0
    nretry = 0
    nsubsteps = 0
    max_retry_subdivisions = int(max_retry_subdivisions or 0)
    max_step = None if max_step is None else float(max_step)
    flat_max_step = None if flat_max_step is None else float(flat_max_step)
    used_grid_numba = False
    profile = bool(profile)
    profile_wall_s = 0.0
    profile_stats = None
    if edge_mask is None:
        edge_mask_arr = np.empty(0, dtype=np.bool_)
    else:
        edge_mask_arr = np.asarray(edge_mask, dtype=np.bool_)
        if len(edge_mask_arr) != N:
            raise ValueError("edge_mask length must match tgrid")

    if transient_solve_grid_numba is not None:
        try:
            max_step_arg = -1.0 if max_step is None else float(max_step)
            flat_max_step_arg = -1.0 if flat_max_step is None else float(flat_max_step)
            t_profile0 = time.perf_counter()
            ok_grid, Vfast, fast_substeps, _, raw_profile = transient_solve_grid_numba(
                np.asarray(V0, float), np.asarray(tgrid, float),
                np.asarray(input_values, float), edge_mask_arr, profile,
                max_step_arg, flat_max_step_arg,
                int(n), int(newton_maxit), float(newton_step_limit),
                float(newton_vtol), float(gmin),
                bool(fallback_full_jacobian or fallback_least_squares),
                float(fallback_tol), float(HH),
                dev_d_kind, dev_d_ref, dev_d_val,
                dev_g_kind, dev_g_ref, dev_g_val,
                dev_s_kind, dev_s_ref, dev_s_val,
                dev_di, dev_gi, dev_si, dev_use_abs,
                p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
                p_current_scale, p_inv_Rleak,
                p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
                p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
                op_cache_valid, op_cache_vs1, op_cache_vd1,
                res_a_kind, res_a_ref, res_a_val,
                res_b_kind, res_b_ref, res_b_val, res_ai, res_bi, res_g,
                cap_a_kind, cap_a_ref, cap_a_val,
                cap_b_kind, cap_b_ref, cap_b_val, cap_ai, cap_bi, cap_value,
                isrc_pi, isrc_qi, isrc_value,
                dyn_pi, dyn_qi, dyn_input_idx,
                float(clip_lo), float(clip_hi),
            )
            profile_wall_s = time.perf_counter() - t_profile0
            if ok_grid:
                Vhist = Vfast
                nsubsteps = int(fast_substeps)
                numba_newton_attempts += nsubsteps
                numba_newton_success += nsubsteps
                used_grid_numba = True
                profile_stats = np.asarray(raw_profile, float)
        except Exception:
            used_grid_numba = False

    def input_at_between(input_a, input_b, frac):
        return input_a + (input_b - input_a) * frac

    def try_step(Vp, input_prev, input_now, h, use_fallback=True):
        V = Vp
        ok = False
        raised = False
        try:
            V, _, ok = newton(Vp, Vp, input_now, input_prev, h,
                              vtol=float(newton_vtol))
        except Exception:
            raised = True
        if ok:
            return V, True, raised
        if not use_fallback:
            return Vp, False, raised
        if fallback_full_jacobian:
            for seed in (V, Vp):
                try:
                    Vn, okn = newton_full_jac(
                        seed, Vp, input_now, input_prev, h,
                        residual_tol=float(fallback_tol),
                        vtol=float(newton_vtol))
                    if okn:
                        return Vn, True, raised
                except Exception:
                    raised = True
        if not fallback_least_squares:
            return V, bool(ok), raised
        try:
            from scipy.optimize import least_squares
            rails = [v for v in plan.rails.values() if isinstance(v, (int, float))]
            if not rails:
                return V, False, raised
            lo = min(rails) - 0.5
            hi = max(rails) + 0.5
            prev_dev_terms = device_terminal_values(Vp, input_prev)
            load_prev_dv = load_cap_dv_values(Vp, input_prev)

            def residual(z):
                states = device_states(z, input_now)
                return step_residual(z, input_now, states, prev_dev_terms,
                                     load_prev_dv, h)

            seed = np.clip(V, lo, hi)
            sol = least_squares(residual, seed, bounds=(lo, hi), x_scale="jac",
                                xtol=1e-11, ftol=1e-11, gtol=1e-11,
                                max_nfev=80)
            norm = np.linalg.norm(residual(sol.x), ord=np.inf)
            if norm < fallback_tol:
                return sol.x, True, raised
            return Vp, False, raised
        except Exception:
            return Vp, False, True

    def solve_chunk(Vp, input_prev, input_now, h, depth=0):
        use_fallback = depth >= max_retry_subdivisions
        V, ok, raised = try_step(Vp, input_prev, input_now, h,
                                 use_fallback=use_fallback)
        if ok:
            return V, True, 1, 0
        if depth < max_retry_subdivisions and h > 0.0:
            mid_input = input_at_between(input_prev, input_now, 0.5)
            left_v, left_ok, left_steps, left_retry = solve_chunk(
                Vp, input_prev, mid_input, 0.5 * h, depth + 1)
            if not left_ok:
                return Vp, False, left_steps, left_retry + (1 if depth == 0 else 0)
            right_v, right_ok, right_steps, right_retry = solve_chunk(
                left_v, mid_input, input_now, 0.5 * h, depth + 1)
            if left_ok and right_ok:
                return right_v, True, left_steps + right_steps, left_retry + right_retry + 1
        return Vp, False, 1, 1 if depth == 0 else 0

    if not used_grid_numba:
        for k in range(1, N):
            h = tgrid[k] - tgrid[k - 1]
            if h <= 0.0:
                raise ValueError("tgrid must be strictly increasing")
            Vp = Vhist[k - 1]
            input_start = input_values[:, k - 1]
            input_end = input_values[:, k]
            interval_edge = (len(edge_mask_arr) == N and
                             bool(edge_mask_arr[k] or edge_mask_arr[k - 1]))
            local_max_step = max_step
            if flat_max_step is not None and flat_max_step > 0.0 and not interval_edge:
                local_max_step = flat_max_step
            pieces = 1 if local_max_step is None else max(1, int(np.ceil(h / local_max_step)))
            interval_ok = True
            interval_retries = 0
            in0 = input_start
            for j in range(pieces):
                f1 = (j + 1) / pieces
                in1 = input_at_between(input_start, input_end, f1)
                Vp, ok, steps, retries = solve_chunk(Vp, in0, in1, h / pieces)
                in0 = in1
                nsubsteps += steps
                interval_retries += retries
                if not ok:
                    interval_ok = False
            Vhist[k] = Vp
            nretry += interval_retries
            if not interval_ok:
                nfail += 1

    nodes = {nm: Vhist[:, idx[nm]] for nm in plan.solved}
    out = np.zeros(N)
    for node, weight in plan.output_weights.items():
        out += weight * nodes[node]
    result = {"t": tgrid, "output": out, "vout": out, "nfail": nfail,
              "nretry": nretry, "nsubsteps": nsubsteps, "nodes": nodes}
    if numba_newton_attempts:
        result["numba_newton_attempts"] = numba_newton_attempts
        result["numba_newton_success"] = numba_newton_success
        result["numba_newton_fallback"] = numba_newton_fallback
        result["numba_grid_solver"] = used_grid_numba
    if profile:
        if profile_stats is None:
            profile_stats = np.zeros(16, dtype=float)
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
            "numba_grid_solver": bool(used_grid_numba),
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
            "edge_time_s_est": float(edge_time_est),
            "flat_time_s_est": float(flat_time_est),
            "time_estimate_basis": "newton_iteration_weighted",
        }
    if "VOP" in idx:
        result["vop"] = nodes["VOP"]
    if "VON" in idx:
        result["von"] = nodes["VON"]
    return result


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
