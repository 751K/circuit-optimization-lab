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

This is the engine; chopper switches + charge injection are added on top (TODO).
"""
import numpy as np
try:
    from .pmos_tft_model import PMOS_TFT
    from .topology import AFE_TOPO
    from .ac_solver import ac_solve, _dev_nf
except ImportError:  # pragma: no cover - legacy direct module import
    from pmos_tft_model import PMOS_TFT
    from topology import AFE_TOPO
    from ac_solver import ac_solve, _dev_nf


def transient(sizes, bias, tgrid, vip=None, vin=None, nf=None, V0=None,
              topo=AFE_TOPO, inputs=None, node_inputs=None):
    """Backward-Euler transient.
      tgrid : (N,) time points [s]
      vip,vin : legacy AFE M7/M8 gate waveforms [V]
      inputs : generic mapping {input_key: waveform}; device gates are mapped by
               topo.transient_inputs, e.g. {"M1": "in"}.
      node_inputs : mapping {node_name: input_key} to drive a (rail) NODE with a
               waveform — used for a testbench where the stimulus enters at source
               nodes and propagates through a front-end network, e.g.
               {"VINP": "vip", "VINN": "vin"}.
      V0    : optional initial solved-node vector.
    Returns dict: t, output, vout, nfail, and per-node arrays. AFE legacy vop/von
    fields are included when those nodes exist.
    """
    idx, n = topo.idx, topo.n
    rails = topo.rail_values(bias)
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
    node_inputs = dict(node_inputs or {})
    for node, key in node_inputs.items():
        if key not in inputs:
            raise ValueError(f"node_inputs[{node!r}] references missing waveform {key!r}")

    def gtok(name, g):
        """Gate token: solved node / rail / driven transient input."""
        if name in topo.transient_inputs:
            key = topo.transient_inputs[name]
            if key not in inputs:
                raise ValueError(f"Missing transient input waveform {key!r} for device {name}")
            return ("input", key)
        return g

    def termv(term, V, input_vals):
        """voltage of a terminal token: solved node / driven input / driven node / rail."""
        if isinstance(term, tuple) and term[0] == "input":
            return input_vals[term[1]]
        if term in idx:
            return V[idx[term]]
        if term in node_inputs:
            return input_vals[node_inputs[term]]
        return rails[term]

    dev_meta = []
    for name, d, g, s in devs:
        gt = gtok(name, g)
        dev_meta.append((name, d, gt, s, idx.get(d), idx.get(gt), idx.get(s)))
    load_meta = [(a, b, idx.get(a), idx.get(b), cap) for a, b, cap in topo.cap_list()]
    res_meta = [(a, b, idx.get(a), idx.get(b), R) for _, a, b, R in topo.resistors]
    isrc_meta = [(p, q, idx.get(p), idx.get(q), I) for _, p, q, I in topo.isources]

    def device_states(V, input_vals):
        """Per-Newton operating data shared by residual and Jacobian."""
        out = []
        for name, d, gt, s, di, gi, si in dev_meta:
            Vs = termv(s, V, input_vals)
            Vd = termv(d, V, input_vals)
            Vg = termv(gt, V, input_vals)
            try:
                dev = tft[name]
                Vs1, Vd1 = dev.get_op(Vs, Vd, Vg)
                _, _, I_d1_d, _, _ = dev._eval_currents(Vs, Vd, Vg, Vs1, Vd1)
                Idc = -I_d1_d
                Cgs, Cgd = dev._capacitances_from_op(Vs, Vd, Vg, Vs1, Vd1)
                I = abs(Idc)                             # abs() to match ac_solve's
            except Exception:
                I = Cgs = Cgd = 0.0
                Vs1 = Vd1 = 0.0
            out.append((name, d, gt, s, di, gi, si, Vs, Vd, Vg, Vs1, Vd1, I, Cgs, Cgd))
        return out

    # ── initial condition: DC op at static bias ──
    if V0 is None:
        ac = ac_solve(sizes, bias, np.array([1.0]), nf=nf, topo=topo)
        dc = ac["dc_op"]
        V0 = np.array([dc[name] for name in topo.solved])
    Vhist = np.zeros((N, n)); Vhist[0] = V0
    gmin = 1e-12

    def step_residual(V, Vp, input_now, input_prev, states, h):
        R = np.zeros(n)
        # device DC currents (into-node convention: +Id at drain, -Id at source)
        for _, _, _, _, di, _, si, _, _, _, _, _, I, _, _ in states:
            if di is not None:
                R[di] += I                               # KCL sign (drain current INTO
            if si is not None:
                R[si] -= I                               # node); raw get_Idc is negative
        for a, b, ai, bi, Rval in res_meta:              # resistor branch current a -> b
            i_ab = (termv(a, V, input_now) - termv(b, V, input_now)) / Rval
            if ai is not None:
                R[ai] -= i_ab
            if bi is not None:
                R[bi] += i_ab
        for p, q, pi, qi, Ival in isrc_meta:             # ideal DC current source p -> q
            if pi is not None:
                R[pi] -= Ival
            if qi is not None:
                R[qi] += Ival
        for k in range(n):
            R[k] -= V[k] * gmin
        # capacitor companion: C(V_now)·[(Va-Vb)_now - (Va-Vb)_prev]/h  (C implicit; Cmap
        # is evaluated at the current iterate by the caller, ΔV_prev uses the prev step)
        def add_cap_res(a, b, ai, bi, C):
            if C == 0.0:
                return
            dv_now = termv(a, V, input_now) - termv(b, V, input_now)
            dv_pre = termv(a, Vp, input_prev) - termv(b, Vp, input_prev)
            i_ab = (C / h) * (dv_now - dv_pre)          # A -> B
            if ai is not None:
                R[ai] -= i_ab
            if bi is not None:
                R[bi] += i_ab
        for _, d, gt, s, di, gi, si, _, _, _, _, _, _, Cgs, Cgd in states:
            add_cap_res(gt, s, gi, si, Cgs)              # gate-source
            add_cap_res(gt, d, gi, di, Cgd)              # gate-drain
        for a, b, ai, bi, cap in load_meta:
            add_cap_res(a, b, ai, bi, cap)
        return R

    HH = 1e-3   # finite-diff step for gm/gds (matches get_ss_params, Cadence-calibrated)

    def terminal_derivatives(name, Vs, Vd, Vg, Vs1, Vd1, need_gm, need_gds):
        """Terminal gm/gds via implicit differentiation of the solved internal nodes.

        This is algebraically the same derivative that finite-differencing get_Idc
        measures, but avoids solving the 2-node internal OP four extra times per
        device per Newton iteration.
        """
        if not need_gm and not need_gds:
            return 0.0, 0.0
        dev = tft[name]
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
        for name, _, _, _, di, gi, si, Vs, Vd, Vg, Vs1, Vd1, _, _, _ in states:
            get_idc = tft[name].get_Idc
            need_gm = gi is not None or si is not None
            need_gds = di is not None or si is not None
            try:                                         # abs() to match the residual's
                gm, gds = terminal_derivatives(name, Vs, Vd, Vg, Vs1, Vd1, need_gm, need_gds)
            except Exception:
                try:
                    gm = ((abs(get_idc(Vs, Vd, Vg + HH)) - abs(get_idc(Vs, Vd, Vg - HH))) / (2 * HH)
                          if need_gm else 0.0)
                    gds = ((abs(get_idc(Vs, Vd + HH, Vg)) - abs(get_idc(Vs, Vd - HH, Vg))) / (2 * HH)
                           if need_gds else 0.0)
                except Exception:
                    gm, gds = 0.0, 1e-12 if need_gds else 0.0
            cols = []                                   # (column, dI/dVcol) for solved terms
            if di is not None: cols.append((di, gds))
            if gi is not None: cols.append((gi, gm))
            if si is not None: cols.append((si, -(gm + gds)))
            if di is not None:                          # row d: R[d] += I
                for c, val in cols: J[di, c] += val
            if si is not None:                          # row s: R[s] -= I
                for c, val in cols: J[si, c] -= val
        for k in range(n):
            J[k, k] -= gmin
        def add_cond_jac(ai, bi, g):
            """Stamp a plain conductance g between two solved terminals."""
            if g == 0.0:
                return
            if ai is not None:
                J[ai, ai] -= g
                if bi is not None: J[ai, bi] += g
            if bi is not None:
                J[bi, bi] -= g
                if ai is not None: J[bi, ai] += g

        def add_cap_jac(ai, bi, C):
            if C == 0.0:
                return
            add_cond_jac(ai, bi, C / h)            # capacitor companion conductance C/h
        for _, _, _, _, di, gi, si, _, _, _, _, _, _, Cgs, Cgd in states:
            add_cap_jac(gi, si, Cgs)                    # gate-source
            add_cap_jac(gi, di, Cgd)                    # gate-drain
        for _, _, ai, bi, cap in load_meta:
            add_cap_jac(ai, bi, cap)
        for _, _, ai, bi, Rval in res_meta:             # resistor conductance 1/R
            add_cond_jac(ai, bi, 1.0 / Rval)
        # ideal DC current sources are constant -> no Jacobian contribution.
        return J

    def newton(seed, Vp, input_now, input_prev, h, maxit=30, vtol=1e-8):
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
        for it in range(maxit):
            states = device_states(V, input_now)         # implicit caps at current iterate
            R = step_residual(V, Vp, input_now, input_prev, states, h)
            J = build_jac(V, states, h)
            try:
                dV = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                dV = np.linalg.lstsq(J, -R, rcond=None)[0]
            mx = np.max(np.abs(dV))
            if mx > 5.0:
                dV *= 5.0 / mx; mx = 5.0                 # branch-safety step cap
            V = V + dV
            if mx < vtol:
                return V, it + 1, True
            if it >= 4 and mx >= prev and mx < 1e-5:     # stalled at the numeric floor
                return V, it + 1, True
            prev = mx
        return V, maxit, False

    nfail = 0
    for k in range(1, N):
        h = tgrid[k] - tgrid[k - 1]
        Vp = Vhist[k - 1]
        input_now = {key: val[k] for key, val in inputs.items()}
        input_prev = {key: val[k - 1] for key, val in inputs.items()}
        try:
            V, nrm, ok = newton(Vp, Vp, input_now, input_prev, h)
            Vhist[k] = V
            if not ok:
                nfail += 1
        except Exception:
            Vhist[k] = Vp; nfail += 1

    nodes = {nm: Vhist[:, idx[nm]] for nm in topo.solved}
    out = np.zeros(N)
    for node, weight in topo.output_weights().items():
        out += weight * nodes[node]
    result = {"t": tgrid, "output": out, "vout": out, "nfail": nfail, "nodes": nodes}
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
