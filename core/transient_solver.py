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
from pmos_tft_model import PMOS_TFT
from topology import AFE_TOPO
from ac_solver import ac_solve, _dev_nf

CL = 5e-12


def transient(sizes, bias, tgrid, vip, vin, nf=None, V0=None):
    """Backward-Euler transient.
      tgrid : (N,) time points [s]
      vip,vin : (N,) M7/M8 gate (input) voltages [V]
      V0    : optional initial 6-vector of node voltages (else DC op @ vip=vin=VCM)
    Returns dict: t, vop, von, vout(=vop-von), and per-node arrays.
    """
    topo = AFE_TOPO
    idx, n = topo.idx, topo.n
    VDD, VB, VC, VCM = bias["VDD"], bias["VB"], bias["VC"], bias["VCM"]
    rails = {"VDD": VDD, "GND": 0.0, "VB": VB, "VC": VC, "VCM": VCM}
    devs = topo.devices
    tft = {name: PMOS_TFT(W=sizes[name][0], L=sizes[name][1], NF=_dev_nf(nf, name))
           for name, *_ in devs}
    tgrid = np.asarray(tgrid, float)
    vip = np.asarray(vip, float); vin = np.asarray(vin, float)
    N = len(tgrid)

    def gtok(name, g):
        """gate node TOKEN: input pair gates are the driven inputs VIP/VIN."""
        if name == "M7": return "VIP"
        if name == "M8": return "VIN"
        return g

    def termv(term, V, vp, vn):
        """voltage of a terminal token: solved node / driven input / rail."""
        if term in idx:
            return V[idx[term]]
        if term == "VIP": return vp
        if term == "VIN": return vn
        return rails[term]

    def caps_at(V, vp, vn):
        """List of (a_term, b_term, C) frozen at operating point (V, vp, vn)."""
        out = []
        for name, d, g, s in devs:
            gt = gtok(name, g)
            Vs = termv(s, V, vp, vn); Vd = termv(d, V, vp, vn); Vg = termv(gt, V, vp, vn)
            try:
                Cgs, Cgd = tft[name].get_capacitances(Vs, Vd, Vg)
            except Exception:
                Cgs = Cgd = 0.0
            out.append((gt, s, Cgs))      # gate-source (gate token honours VIP/VIN)
            out.append((gt, d, Cgd))      # gate-drain
        out.append(("VOP", "GND", CL))
        out.append(("VON", "GND", CL))
        return out

    # ── initial condition: DC op at vip=vin=VCM ──
    if V0 is None:
        ac = ac_solve(sizes, bias, np.array([1.0]), nf=nf)
        dc = ac["dc_op"]
        V0 = np.array([dc["VOP"], dc["VON"], dc["vfbp"], dc["vfbn"], dc["NET20"], dc["NET2"]])
    Vhist = np.zeros((N, n)); Vhist[0] = V0
    gmin = 1e-12

    def step_residual(V, Vp, vp, vn, vpp, vnp, Cmap, h):
        R = np.zeros(n)
        # device DC currents (into-node convention: +Id at drain, -Id at source)
        for name, d, g, s in devs:
            Vs = termv(s, V, vp, vn); Vd = termv(d, V, vp, vn)
            Vg = termv(gtok(name, g), V, vp, vn)
            try:
                I = abs(tft[name].get_Idc(Vs, Vd, Vg))   # abs() to match ac_solve's
            except Exception:                            # KCL sign (drain current INTO
                I = 0.0                                  # node); raw get_Idc is negative
            if d in idx: R[idx[d]] += I                  # for these PMOS, and the wrong
            if s in idx: R[idx[s]] -= I                  # sign makes the DAE unstable
        for k in range(n):
            R[k] -= V[k] * gmin
        # capacitor companion: C(V_now)·[(Va-Vb)_now - (Va-Vb)_prev]/h  (C implicit; Cmap
        # is evaluated at the current iterate by the caller, ΔV_prev uses the prev step)
        for (a, b, C) in Cmap:
            if C == 0.0:
                continue
            dv_now = termv(a, V, vp, vn) - termv(b, V, vp, vn)
            dv_pre = termv(a, Vp, vpp, vnp) - termv(b, Vp, vpp, vnp)
            i_ab = (C / h) * (dv_now - dv_pre)          # A -> B
            if a in idx: R[idx[a]] -= i_ab
            if b in idx: R[idx[b]] += i_ab
        return R

    HH = 1e-3   # finite-diff step for gm/gds (matches get_ss_params, Cadence-calibrated)

    def build_jac(V, vp, vn, Cmap, h):
        """Analytic conductance Jacobian dR/dV (n×n).

        Device part = small-signal conductance stamp: gm=dI/dVg, gds=dI/dVd
        (finite-diff of get_Idc, same as get_ss_params), dI/dVs=-(gm+gds).
        Only SOLVED terminals get a column (driven inputs / rails are fixed).
        Plus capacitor companion (C/h) and gmin on the diagonal. A well-scaled
        Jacobian is exactly what bare fsolve lacked — it lets Newton track the
        correct (physical) branch of the positive-feedback circuit."""
        J = np.zeros((n, n))
        for name, d, g, s in devs:
            gt = gtok(name, g)
            Vs = termv(s, V, vp, vn); Vd = termv(d, V, vp, vn); Vg = termv(gt, V, vp, vn)
            t = tft[name]
            try:                                         # abs() to match the residual's
                aId = lambda vs, vd, vg: abs(t.get_Idc(vs, vd, vg))   # I_into convention
                gm  = (aId(Vs, Vd, Vg + HH) - aId(Vs, Vd, Vg - HH)) / (2 * HH)
                gds = (aId(Vs, Vd + HH, Vg) - aId(Vs, Vd - HH, Vg)) / (2 * HH)
            except Exception:
                gm, gds = 0.0, 1e-12
            cols = []                                   # (column, dI/dVcol) for solved terms
            if d in idx:  cols.append((idx[d], gds))
            if gt in idx: cols.append((idx[gt], gm))
            if s in idx:  cols.append((idx[s], -(gm + gds)))
            if d in idx:                                # row d: R[d] += I
                for c, val in cols: J[idx[d], c] += val
            if s in idx:                                # row s: R[s] -= I
                for c, val in cols: J[idx[s], c] -= val
        for k in range(n):
            J[k, k] -= gmin
        for (a, b, C) in Cmap:                          # cap companion: i_ab=(C/h)(Va-Vb)
            if C == 0.0:
                continue
            gC = C / h
            ia, ib = idx.get(a), idx.get(b)
            if ia is not None:
                J[ia, ia] -= gC
                if ib is not None: J[ia, ib] += gC
            if ib is not None:
                J[ib, ib] -= gC
                if ia is not None: J[ib, ia] += gC
        return J

    def newton(seed, Vp, vp, vn, vpp, vnp, h, maxit=30, vtol=1e-8):
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
            Cmap = caps_at(V, vp, vn)                    # implicit: caps at current iterate
            R = step_residual(V, Vp, vp, vn, vpp, vnp, Cmap, h)
            J = build_jac(V, vp, vn, Cmap, h)
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
        try:
            V, nrm, ok = newton(Vp, Vp, vip[k], vin[k], vip[k - 1], vin[k - 1], h)
            Vhist[k] = V
            if not ok:
                nfail += 1
        except Exception:
            Vhist[k] = Vp; nfail += 1

    vop = Vhist[:, idx["VOP"]]; von = Vhist[:, idx["VON"]]
    return {"t": tgrid, "vop": vop, "von": von, "vout": vop - von, "nfail": nfail,
            "nodes": {nm: Vhist[:, idx[nm]] for nm in topo.solved}}


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
