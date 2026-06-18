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
  signals. (The AT_4000TG model is itself capacitance-based / non-charge-conserving:
  its only dynamics are these two bias-dependent gate caps through an internal gate1
  node + 100 Ω; we omit that 100 Ω·C ≈ 13 ns RC as negligible vs the timescales here.)
- Each step: solve the 6 node voltages with a damped Newton iteration using an
  analytic conductance Jacobian (gm/gds finite-diff of get_Idc + cap C/h + gmin),
  seeded from the previous step, with step limiting that keeps it on the physical
  branch. (Bare fsolve's poorly-scaled numeric Jacobian latched onto wrong roots
  of this positive-feedback circuit — gain didn't match the AC reference.)

Caps stamped: per device Cgs (gate-source) and Cgd (gate-drain), plus CL on the
two outputs. Inputs: M7 gate = vip(t), M8 gate = vin(t) (driven); other rails fixed.

CURRENT SIGN: device currents use abs(get_Idc), exactly like ac_solve's KCL. raw
get_Idc is negative for these PMOS; using the raw (wrong-sign) value flips the
conductance Jacobian and turns the stable equilibrium into an *unstable* DAE — the
common mode then runs away. (Backward Euler with a large h artificially damps that
runaway, so coarse-step DC/gain checks pass and hide the bug; only physically fine
timesteps expose it. Eigen-check at the op: slowest pole −4.0e3 rad/s ≈ −3 dB BW.)

This is the engine; chopper switch topologies can be driven through node_inputs.
Clock feedthrough from the Verilog-A Cgss/Cgdd terms is included when the switch
gate clocks are finite-edge waveforms. Additional explicit charge-injection
pulses can be supplied through current_inputs.
"""
import numpy as np
try:
    from .pmos_tft_model import PMOS_TFT
    from .topology import AFE_TOPO
    from .ac_solver import ac_solve, _dev_nf
    from .numba_kernels import terminal_derivatives_numba
    from .compiled_topology import CompiledTopology
except ImportError:  # pragma: no cover - legacy direct module import
    from pmos_tft_model import PMOS_TFT
    from topology import AFE_TOPO
    from ac_solver import ac_solve, _dev_nf
    from compiled_topology import CompiledTopology
    try:
        from numba_kernels import terminal_derivatives_numba
    except Exception:
        terminal_derivatives_numba = None


def transient(sizes, bias, tgrid, vip=None, vin=None, nf=None, V0=None,
              topo=AFE_TOPO, inputs=None, node_inputs=None, current_inputs=None,
              max_step=None, max_retry_subdivisions=0, newton_maxit=30,
              newton_step_limit=5.0, newton_vtol=1e-8,
              fallback_least_squares=False, fallback_tol=1e-9):
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
      max_retry_subdivisions : if Newton fails on a step, recursively bisect that
               step up to this depth before recording a failure.
      fallback_least_squares : if true, a failed Newton step is retried with a
               rail-bounded least-squares solve before substepping/failing.
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
    dev_meta = [(tft[item.name], item.d, item.g, item.s, item.di, item.gi, item.si)
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

    def device_states(V, input_vals):
        """Per-Newton operating data shared by residual and Jacobian."""
        out = [None] * len(dev_meta)
        for pos, (dev, dterm, gterm, sterm, di, gi, si) in enumerate(dev_meta):
            Vs = termv(sterm, V, input_vals)
            Vd = termv(dterm, V, input_vals)
            Vg = termv(gterm, V, input_vals)
            try:
                Vs1, Vd1 = dev.get_op(Vs, Vd, Vg)
                _, _, I_d1_d, _, _ = dev._eval_currents(Vs, Vd, Vg, Vs1, Vd1)
                Idc = -I_d1_d
                Cgs, Cgd = dev._capacitances_from_op(Vs, Vd, Vg, Vs1, Vd1)
                I = abs(Idc)                             # abs() to match ac_solve's
            except Exception:
                I = Cgs = Cgd = 0.0
                Vs1 = Vd1 = 0.0
            out[pos] = (dev, di, gi, si, Vs, Vd, Vg, Vs1, Vd1, I, Cgs, Cgd)
        return out

    def device_terminal_values(V, input_vals):
        return [(termv(sterm, V, input_vals),
                 termv(dterm, V, input_vals),
                 termv(gterm, V, input_vals))
                for _, dterm, gterm, sterm, *_ in dev_meta]

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
        for _, di, _, si, _, _, _, _, _, I, _, _ in states:
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
            _, di, gi, si, Vs, Vd, Vg, _, _, _, Cgs, Cgd = state
            pVs, pVd, pVg = prev_terms
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

    def terminal_derivatives(dev, Vs, Vd, Vg, Vs1, Vd1, need_gm, need_gds):
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
                    Vs, Vd, Vg, Vs1, Vd1, need_gm, need_gds, HH, 1e-6,
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

        def deriv(vs_p, vd_p, vg_p, vs_m, vd_m, vg_m):
            Fpa, Fpb, Ip = eval_at(vs_p, vd_p, vg_p, Vs1, Vd1)
            Fma, Fmb, Im = eval_at(vs_m, vd_m, vg_m, Vs1, Vd1)
            fu0 = (Fpa - Fma) / (2 * HH)
            fu1 = (Fpb - Fmb) / (2 * HH)
            Iu = (Ip - Im) / (2 * HH)
            y0 = (j11 * fu0 - j01 * fu1) / det
            y1 = (-j10 * fu0 + j00 * fu1) / det
            return sign * (Iu - ix0 * y0 - ix1 * y1)

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
        for dev, di, gi, si, Vs, Vd, Vg, Vs1, Vd1, _, _, _ in states:
            get_idc = dev.get_Idc
            need_gm = gi is not None or si is not None
            need_gds = di is not None or si is not None
            try:                                         # abs() to match the residual's
                gm, gds = terminal_derivatives(dev, Vs, Vd, Vg, Vs1, Vd1, need_gm, need_gds)
            except Exception:
                try:
                    gm = ((abs(get_idc(Vs, Vd, Vg + HH)) - abs(get_idc(Vs, Vd, Vg - HH))) / (2 * HH)
                          if need_gm else 0.0)
                    gds = ((abs(get_idc(Vs, Vd + HH, Vg)) - abs(get_idc(Vs, Vd - HH, Vg))) / (2 * HH)
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
        for _, di, gi, si, _, _, _, _, _, _, Cgs, Cgd in states:
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
        V = np.array(seed, float)
        prev = np.inf
        prev_dev_terms = device_terminal_values(Vp, input_prev)
        load_prev_dv = load_cap_dv_values(Vp, input_prev)
        maxit = int(newton_maxit if maxit is None else maxit)
        step_limit = float(newton_step_limit)
        for it in range(maxit):
            states = device_states(V, input_now)         # implicit caps at current iterate
            R = step_residual(V, input_now, states, prev_dev_terms, load_prev_dv, h)
            J = build_jac(V, states, h)
            try:
                dV = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                dV = np.linalg.lstsq(J, -R, rcond=None)[0]
            mx = np.max(np.abs(dV))
            if mx > step_limit:
                dV *= step_limit / mx; mx = step_limit    # branch-safety step cap
            V = V + dV
            if mx < vtol:
                return V, it + 1, True
            if it >= 4 and mx >= prev and mx < 1e-5:     # stalled at the numeric floor
                return V, it + 1, True
            prev = mx
        return V, maxit, False

    nfail = 0
    nretry = 0
    nsubsteps = 0
    max_retry_subdivisions = int(max_retry_subdivisions or 0)
    max_step = None if max_step is None else float(max_step)

    def input_at_between(input_a, input_b, frac):
        return input_a + (input_b - input_a) * frac

    def try_step(Vp, input_prev, input_now, h):
        V = Vp
        ok = False
        raised = False
        try:
            V, _, ok = newton(Vp, Vp, input_now, input_prev, h,
                              vtol=float(newton_vtol))
        except Exception:
            raised = True
        if ok or not fallback_least_squares:
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
            return sol.x, False, raised
        except Exception:
            return V if not raised else Vp, False, True

    def solve_chunk(Vp, input_prev, input_now, h, depth=0):
        V, ok, raised = try_step(Vp, input_prev, input_now, h)
        if ok:
            return V, True, 1, 0
        if depth < max_retry_subdivisions and h > 0.0:
            mid_input = input_at_between(input_prev, input_now, 0.5)
            left_v, left_ok, left_steps, left_retry = solve_chunk(
                Vp, input_prev, mid_input, 0.5 * h, depth + 1)
            right_v, right_ok, right_steps, right_retry = solve_chunk(
                left_v, mid_input, input_now, 0.5 * h, depth + 1)
            if left_ok and right_ok:
                return right_v, True, left_steps + right_steps, left_retry + right_retry + 1
        return V if not raised else Vp, False, 1, 1 if depth == 0 else 0

    for k in range(1, N):
        h = tgrid[k] - tgrid[k - 1]
        if h <= 0.0:
            raise ValueError("tgrid must be strictly increasing")
        Vp = Vhist[k - 1]
        input_start = input_values[:, k - 1]
        input_end = input_values[:, k]
        pieces = 1 if max_step is None else max(1, int(np.ceil(h / max_step)))
        interval_ok = True
        interval_retries = 0
        for j in range(pieces):
            f0 = j / pieces
            f1 = (j + 1) / pieces
            in0 = input_at_between(input_start, input_end, f0)
            in1 = input_at_between(input_start, input_end, f1)
            Vp, ok, steps, retries = solve_chunk(Vp, in0, in1, h / pieces)
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
