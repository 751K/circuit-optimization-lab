"""
Small-signal AC solver using MNA (Modified Nodal Analysis).
Solves the full circuit at each frequency, computes gain and BW.
Includes ALL transistors + load capacitors.
"""
import numpy as np
try:
    from .pmos_tft_model import PMOS_TFT
    from .topology import AFE_TOPO
except ImportError:  # pragma: no cover - legacy direct module import
    from pmos_tft_model import PMOS_TFT
    from topology import AFE_TOPO


def _dev_corner(corner, name):
    """Resolve the model-shift dict for one device.

    corner may be:
      - None / {}                              -> nominal (no shift)
      - flat dict {'pvt0':.., 'pbeta0':..}     -> GLOBAL shift, same for all devices
                                                  (process corner)
      - per-device map {'M7':{...}, 'M8':{...}}-> PER-DEVICE shift (mismatch:
                                                  each device its own mvt0/mbeta0)
    """
    if not corner:
        return {}
    if any(isinstance(v, dict) for v in corner.values()):     # per-device map
        return corner.get(name, {})
    return corner                                             # global shift


def _dev_nf(nf, name):
    """Resolve NF (number of fingers) for one device. nf may be None (->1),
    an int (global), or a per-device map {'M7':120,...} (missing -> 1).
    NF doesn't change Idc/gm (current ∝ total W/L) but DOES change the gate
    capacitances (Cgs/Cgd via finger geometry), hence BW and the cap part of noise."""
    if not nf:
        return 1
    if isinstance(nf, dict):
        return int(nf.get(name, 1))
    return int(nf)


_AFE_SYMMETRIC_PAIRS = (("M7", "M8"), ("M9", "M10"), ("M12", "M13"), ("M14", "M15"))
_DC_FALLBACK_TOL = 1e-10


def _is_pairwise_symmetric_afe(sizes, nf, topo):
    """True only when the default AFE can be reduced to the 4-node symmetric DC solve."""
    if topo is not AFE_TOPO:
        return False
    for left, right in _AFE_SYMMETRIC_PAIRS:
        if sizes.get(left) != sizes.get(right):
            return False
        if _dev_nf(nf, left) != _dev_nf(nf, right):
            return False
    return True


def _dc_residual_ok(residuals, x, tol=1e-9):
    try:
        return np.linalg.norm(residuals(x), ord=np.inf) < tol
    except Exception:
        return False


def _bounded_least_squares_dc(residuals, guesses, topo, bias, tol=_DC_FALLBACK_TOL):
    """Last-resort bounded DC solve.

    fsolve is fast near a good root but can run to absurd voltages on bad AFE
    slider combinations. This fallback keeps nodes inside the rail box and only
    accepts a solution when KCL is still tight.
    """
    from scipy.optimize import least_squares
    rails = [v for v in topo.rail_values(bias).values() if isinstance(v, (int, float))]
    if not rails:
        return None
    lo = min(rails) - 0.5
    hi = max(rails) + 0.5
    best_x = None
    best_norm = np.inf
    for x0 in guesses[:6]:
        try:
            x0 = np.clip(np.asarray(x0, float), lo, hi)
            sol = least_squares(residuals, x0, bounds=(lo, hi), x_scale="jac",
                                xtol=1e-13, ftol=1e-13, gtol=1e-13,
                                max_nfev=1200)
            norm = np.linalg.norm(residuals(sol.x), ord=np.inf)
            if norm < best_norm:
                best_norm = norm
                best_x = sol.x
        except Exception:
            pass
    if best_x is not None and best_norm < tol:
        return best_x
    return None


def _symmetric_seed(sizes, bias, Id, gmin, seeds=None):
    """AFE-specific symmetric DC solve (matched halves: VON=VOP, VFBN=VFBP). 4 unknowns
    [net2, vop, vfb, net20]. Used as a POST-PROCESS guard: when the full 6-node solve
    latches to a symmetry-broken root (cross-coupled positive feedback) in a no-mismatch
    case, re-solving this symmetric system — seeded from the symmetrized average of the
    latched solution — recovers the physical symmetric branch that Spectre finds.
    Returns a {node:V} dict, or None."""
    from scipy.optimize import fsolve
    VDD, VCM, VB, VC = bias["VDD"], bias["VCM"], bias["VB"], bias["VC"]

    def f(u):
        net2, vop, vfb, net20 = u
        return [
            Id("M6", VDD, net2, VB) - 2 * Id("M7", net2, vop, VCM) - net2 * gmin,
            Id("M7", net2, vop, VCM) - Id("M9", vop, 0.0, vfb) - vop * gmin,
            Id("M13", net20, vfb, vop) - Id("M15", vfb, 0.0, 0.0) - vfb * gmin,
            Id("M11", VDD, net20, VC) - 2 * Id("M12", net20, vfb, vop) - net20 * gmin,
        ]
    trials = list(seeds or []) + [[VCM + 6, VCM - 1, 6.0, bias["VDD"] - 2],
                                  [VCM + 7, VCM - 4, max(VCM - 25, 4.0), VCM + 8],
                                  [VCM + 7, VCM - 4, VCM - 8, VCM + 15],
                                  [VCM + 9, VCM - 2, VCM - 10, VCM + 12],
                                  [VCM + 5, VCM - 6, VCM - 6, VCM + 18]]
    for u0 in trials:
        try:
            sol, _, ier, _ = fsolve(f, u0, full_output=True, xtol=1e-12, maxfev=4000)
            residual_norm = np.linalg.norm(f(sol), ord=np.inf)
            in_box = all(-0.5 <= v <= VDD + 0.5 for v in sol)
            if in_box and residual_norm < _DC_FALLBACK_TOL:
                n2, vop, vfb, n20 = sol
                return {"VOP": vop, "VON": vop, "VFBP": vfb, "VFBN": vfb,
                        "NET20": n20, "NET2": n2}
        except Exception:
            pass
    return None


def _symmetric_continuation(sizes, bias, Id, gmin):
    """AFE symmetric DC via SOURCE-RAMP continuation (power-up homotopy): scale all
    rails 0->1 and track the solution from the powered-down state. This follows the
    same physical branch Spectre's pseudo-transient/gmin-stepping converges to, so it
    selects the correct equilibrium even when the symmetric circuit is multistable
    (a 'normal-on' branch vs a degenerate near-off branch). 4 unknowns: net2,vop,vfb,net20.
    Returns a symmetric {node:V} dict (used as the PRIMARY DC seed), or None."""
    from scipy.optimize import fsolve
    VDD, VCM, VB, VC = bias["VDD"], bias["VCM"], bias["VB"], bias["VC"]

    def f(u, sc):
        net2, vop, vfb, net20 = u
        Vdd, Vcm, Vb, Vc = VDD * sc, VCM * sc, VB * sc, VC * sc
        return [
            Id("M6", Vdd, net2, Vb) - 2 * Id("M7", net2, vop, Vcm) - net2 * gmin,
            Id("M7", net2, vop, Vcm) - Id("M9", vop, 0.0, vfb) - vop * gmin,
            Id("M13", net20, vfb, vop) - Id("M15", vfb, 0.0, 0.0) - vfb * gmin,
            Id("M11", Vdd, net20, Vc) - 2 * Id("M12", net20, vfb, vop) - net20 * gmin,
        ]
    def track(seed_sets):
        u = np.array([VCM, VCM, VCM, VCM]) * 0.1      # near powered-down
        for sc in np.linspace(0.1, 1.0, 19):
            ok = False
            for seed in seed_sets(u, sc):
                try:
                    s, _, ier, _ = fsolve(lambda z: f(z, sc), seed,
                                          full_output=True, xtol=1e-12, maxfev=4000)
                    if _dc_residual_ok(lambda z: f(z, sc), s, tol=_DC_FALLBACK_TOL):
                        u = s; ok = True; break
                except Exception:
                    pass
            if not ok:
                return None
        n2, vop, vfb, n20 = u
        return {"VOP": vop, "VON": vop, "VFBP": vfb, "VFBN": vfb,
                "NET20": n20, "NET2": n2}

    original = track(lambda u, sc: (u, np.array([VCM+7, VCM-4, VCM-8, VCM+15]) * sc))
    if original is not None:
        return original

    def low_vfb_seeds(u, sc):
        return (
            u,
            np.array([VCM + 6, VCM - 1, 6.0, VDD - 2]) * sc,
            np.array([VCM + 7, VCM - 4, max(VCM - 25, 4.0), VCM + 8]) * sc,
        )
    return track(low_vfb_seeds)


def get_ss_params(W, L, Vs, Vd, Vg, corner=None, nf=1, dev_inst=None):
    """Small-signal parameters at a DC operating point.

    gm/gds are the *terminal* values, extracted by finite-differencing the full
    terminal current get_Idc (which solves the OTFT internal contact nodes).
    The channel gm from _eval_channel is degenerated by the contact resistance;
    Spectre's AC sees the terminal value. Using terminal gm matches Cadence to
    <0.05 dB / <0.1 Hz across the band; channel gm was 0.8 dB / 18 Hz off.

    corner: optional dict of model process shifts, e.g. {'pvt0':.., 'pbeta0':..}.
    dev_inst: optional pre-built PMOS_TFT instance to reuse (warm-start cache).
    """
    t = dev_inst if dev_inst is not None else PMOS_TFT(W=W, L=L, NF=nf, **(corner or {}))
    h = 1e-3
    try:
        Id = lambda vs, vd, vg: t.get_Idc(vs, vd, vg)
        gm  = (Id(Vs, Vd, Vg + h) - Id(Vs, Vd, Vg - h)) / (2 * h)
        gds = (Id(Vs, Vd + h, Vg) - Id(Vs, Vd - h, Vg)) / (2 * h)
        Cgss, Cgdd = t.get_capacitances(Vs, Vd, Vg)
        s1, d1 = t.get_op(Vs, Vd, Vg)
        Ich = t._eval_channel(Vs, Vd, Vg, s1, d1)["Ich"]
        return {"gm": gm, "gds": gds, "Cgs": Cgss, "Cgd": Cgdd, "Ich": Ich}
    except Exception:
        return {"gm": 0, "gds": 1e-12, "Cgs": 0, "Cgd": 0, "Ich": 0}


def ac_solve(sizes, bias, freqs, corner=None, x0_guess=None, topo=AFE_TOPO, nf=None):
    """
    Full small-signal AC analysis — topology supplied by `topo` (default AFE_TOPO).

    sizes: dict of {name: (W, L)}
    bias: dict with VDD, VCM, VB, VC
    freqs: array of frequencies (Hz)
    corner: process shifts — flat dict (global) or per-device map (mismatch); see _dev_corner.
    x0_guess: optional DC seed, either a {node: V} dict (e.g. a prior dc_op) or a vector.

    DC KCL, per-device bias mapping, and the AC terminal list are all DERIVED from
    `topo` (see topology.py) — no hand-written per-device wiring here.
    """
    from scipy.optimize import fsolve
    VCM = topo.default_guess_value(bias)
    gmin = 1e-12

    # ── pre-build device instances so the warm-start Newton cache survives
    #     across fsolve iterations instead of being reset on every Id() call.
    _dev_inst = {
        name: PMOS_TFT(W=sizes[name][0], L=sizes[name][1],
                       NF=_dev_nf(nf, name), **_dev_corner(corner, name))
        for name, *_ in topo.devices
    }

    def Id(name, Vs, Vd, Vg):
        try:
            return abs(_dev_inst[name].get_Idc(Vs, Vd, Vg))
        except Exception:
            return 1e-18

    # ── 1. DC solve (residuals built from the topology) ──
    residuals = lambda x: topo.dc_residuals(x, bias, Id, gmin)
    per_dev = bool(corner) and any(isinstance(v, dict) for v in corner.values())
    symmetric_fast = (x0_guess is None and not per_dev
                      and _is_pairwise_symmetric_afe(sizes, nf, topo))
    guesses = []
    nv = None
    have_symmetric_seed = False
    if x0_guess is not None:
        # A seed was supplied (e.g. an MC/corner sweep seeded from the nominal op that
        # was itself found by continuation): trust it — it already encodes the right
        # branch — and SKIP the (expensive) continuation. Keeps in-loop sweeps fast.
        guesses.append(topo.guess_vector(x0_guess, default=VCM)
                       if isinstance(x0_guess, dict) else list(x0_guess))
    else:
        if symmetric_fast:
            symv = _symmetric_seed(sizes, bias, Id, gmin)
            svec = topo.guess_vector(symv) if symv is not None else None
            if svec is not None and _dc_residual_ok(residuals, svec):
                guesses.append(svec)
                have_symmetric_seed = True
        if symmetric_fast and nv is None and not have_symmetric_seed:
            # Cold solve: run source-ramp continuation on the symmetric system to land on
            # the physical power-up branch Spectre picks (handles symmetry-broken AND
            # degenerate multistable points). Tried before the full 6-node fallback.
            cont = _symmetric_continuation(sizes, bias, Id, gmin)
            if cont is not None:
                cvec = topo.guess_vector(cont)
                guesses.append(cvec)
        guesses.extend(topo.dc_guess_vectors(bias))

    if nv is None:
        if not guesses:
            guesses.extend(topo.dc_guess_vectors(bias))
        for x0 in guesses:
            try:
                sol, _, ier, _ = fsolve(residuals, x0, full_output=True, xtol=1e-12, maxfev=3000)
                if _dc_residual_ok(residuals, sol, tol=_DC_FALLBACK_TOL) or (per_dev and ier == 1):
                    break
            except Exception:
                pass
        else:
            # ── FALLBACK (runs ONLY when every standard guess failed; never alters
            # already-converged points). Goal: pick the SAME physical branch Spectre
            # picks, even for multistable points. ──
            base_g = guesses[0] if guesses else [VCM] * topo.n

            def _solve(bias_d, gm, x0):
                try:
                    s, _, ier, _ = fsolve(lambda z: topo.dc_residuals(z, bias_d, Id, gm),
                                          x0, full_output=True, xtol=1e-12, maxfev=4000)
                    rfun = lambda z: topo.dc_residuals(z, bias_d, Id, gm)
                    return s if (_dc_residual_ok(rfun, s, tol=_DC_FALLBACK_TOL) or
                                 (per_dev and ier == 1)) else None
                except Exception:
                    return None

            sol = None
            # (a) SOURCE-RAMP continuation: scale all rails 0->1, tracking the power-up
            #     trajectory. This follows the physical (Spectre) branch through
            #     multistable regions instead of jumping to an alternate equilibrium.
            x = np.array(base_g) * 0.2
            ramp_ok = True
            for lam in np.linspace(0.2, 1.0, 17):
                bl = {k: (v * lam if isinstance(v, (int, float)) else v)
                      for k, v in bias.items()}
                s = _solve(bl, gmin, x)
                if s is None:
                    s = _solve(bl, gmin, np.array(base_g) * lam)   # re-seed at this step
                    if s is None:
                        ramp_ok = False; break
                x = s
            if ramp_ok:
                sol = x
            # (b) gmin-stepping backup if the ramp couldn't track all the way
            if sol is None:
                flat = topo.guess_vector({n: VCM for n in topo.solved})
                rails = [v for v in topo.rail_values(bias).values() if isinstance(v, (int, float))]
                lo = min(rails) + 0.1 if rails else -np.inf
                hi = max(rails) - 0.1 if rails else np.inf
                for x0 in guesses + [flat]:
                    xc = list(np.clip(x0, lo, hi))
                    good = True
                    for gm in (1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12):
                        s = _solve(bias, gm, xc)
                        if s is None:
                            good = False; break
                        xc = list(s)
                    if good:
                        sol = xc; break
                if sol is None and not per_dev:
                    sol = _bounded_least_squares_dc(residuals, guesses + [flat], topo, bias)
            if sol is None:
                return None  # DC didn't converge even with continuation

        nv = topo.node_vals(sol)                  # {node_name: voltage}, full asymmetric op

    # ── PHYSICALITY GUARD ── No internal node can sit above the supply or below ground
    # here. A solution with e.g. net20 > VDD means the tail M11 is reversed — a
    # non-physical alternate branch Spectre never picks. Re-solve seeded strictly inside
    # the rails and prefer an in-box solution. (Validated designs are already in-box, so
    # this never fires for them.)
    if (topo is AFE_TOPO) and not per_dev and not topo.in_voltage_box(nv, bias):
        VDD = bias["VDD"]
        box_guesses = ({"VOP": VDD*0.6, "VON": VDD*0.6, "VFBP": VDD*0.4, "VFBN": VDD*0.4,
                        "NET20": VDD-4, "NET2": min(VCM+7, VDD-2)},
                       {"VOP": VDD*0.5, "VON": VDD*0.5, "VFBP": VDD*0.3, "VFBN": VDD*0.3,
                        "NET20": VDD-3, "NET2": VDD-5})
        for g in box_guesses:
            try:
                s2, _, ier, _ = fsolve(residuals, topo.guess_vector(g),
                                       full_output=True, xtol=1e-12, maxfev=4000)
            except Exception:
                continue
            if (_dc_residual_ok(residuals, s2, tol=_DC_FALLBACK_TOL) and
                    topo.in_voltage_box(topo.node_vals(s2), bias)):
                nv = topo.node_vals(s2); break
        if not topo.in_voltage_box(nv, bias):
            box_vecs = [topo.guess_vector(g) for g in box_guesses]
            s3 = _bounded_least_squares_dc(residuals, box_vecs + guesses, topo, bias)
            if s3 is None:
                return None
            nv = topo.node_vals(s3)

    # ── SYMMETRY GUARD ── No per-device mismatch ⇒ physical op is symmetric (VOP=VON,
    # VFBP=VFBN). If fsolve latched to a symmetry-broken root, re-solve the symmetric
    # system seeded from the symmetrized average (right next to the physical root).
    # Only fires on no-mismatch + clearly-asymmetric, so symmetric (validated) points
    # are untouched.
    if ((topo is AFE_TOPO) and (not per_dev) and
            (abs(nv["VOP"] - nv["VON"]) > 1e-2 or
             abs(nv["VFBP"] - nv["VFBN"]) > 1e-2)):
        avg = [nv["NET2"], 0.5 * (nv["VOP"] + nv["VON"]),
               0.5 * (nv["VFBP"] + nv["VFBN"]), nv["NET20"]]
        symv = _symmetric_seed(sizes, bias, Id, gmin, seeds=[avg])
        if symv is not None:
            nv = symv                             # physical symmetric branch (Spectre-matching)

    bpts = topo.bias_points(nv, bias)             # per-device (Vs, Vd, Vg)

    # ── 2. Small-signal params at the true per-device DC op ──
    ss = {name: get_ss_params(sizes[name][0], sizes[name][1], *bpts[name],
                              corner=_dev_corner(corner, name), nf=_dev_nf(nf, name),
                              dev_inst=_dev_inst[name])
          for name, *_ in topo.devices}

    # ── 3. Build & solve the small-signal MNA (terminals from the topology) ──
    try:
        from .ac_mna import _stamp_mos, _stamp_adm
    except ImportError:  # pragma: no cover - legacy direct module import
        from ac_mna import _stamp_mos, _stamp_adm
    NN = topo.n
    drive = topo.input_drives
    # Normalize the gain by the differential input magnitude. The stimulus is either
    # a per-gate drive (input_drives) or, for a front-end testbench, AC sources at
    # NODES (ac_drives) that propagate through the passive network to the gates.
    ac_drives = topo.ac_drives
    norm_vals = list(ac_drives.values()) if ac_drives else list(drive.values())
    if not norm_vals:
        vin_norm = 1.0
    elif len(norm_vals) > 1 and max(norm_vals) > min(norm_vals):
        vin_norm = max(norm_vals) - min(norm_vals)
    else:
        vin_norm = max(abs(v) for v in norm_vals) or 1.0
    devs = topo.ac_devices(drive=drive)
    out_weights = topo.output_weights()

    gains = []
    for f in freqs:
        jw = 2j * np.pi * f
        Y = np.zeros((NN, NN), dtype=complex)
        RHS = np.zeros(NN, dtype=complex)
        for name, d, g, s in devs:
            p = ss[name]
            _stamp_mos(Y, RHS, d, g, s, p["gm"], p["gds"], p["Cgs"], p["Cgd"], jw)
        for a, b, cap in topo.cap_list():
            _stamp_adm(Y, RHS, topo.ac_term(a, ac_drives), topo.ac_term(b, ac_drives), jw * cap)
        for name, a, b, R in topo.resistors:
            _stamp_adm(Y, RHS, topo.ac_term(a, ac_drives), topo.ac_term(b, ac_drives), 1.0 / R)
        # ideal current sources are open-circuit in the small-signal AC system.
        V = np.linalg.solve(Y, RHS)
        out = sum(weight * V[topo.idx[node]] for node, weight in out_weights.items())
        gains.append(abs(out) / vin_norm)

    gains = np.array(gains)
    Av_dc = gains[0]
    Av_dc_dB = 20 * np.log10(max(Av_dc, 1e-9))
    peak = gains.max()
    a3 = peak / np.sqrt(2)
    ipk = int(np.argmax(gains))
    bw_Hz = freqs[-1]
    for i in range(ipk, len(gains)):
        if gains[i] < a3:
            bw_Hz = freqs[i]; break

    dc_op = topo.dc_op_with_aliases(nv)
    return {
        "Av_dc_dB": Av_dc_dB,
        "peak_dB": 20 * np.log10(max(peak, 1e-9)),
        "bw_Hz": bw_Hz,
        "gains": gains,
        "freqs": freqs,
        "dc_op": dc_op,
        "ss": ss,
    }


# ── Test with best design from sweep ──
if __name__ == "__main__":
    sizes = {
        "M6": (3000, 150), "M7": (25000, 150), "M8": (25000, 150),
        "M9": (12000, 500), "M10": (12000, 500),
        "M11": (300, 100), "M12": (500, 80), "M13": (500, 80),
        "M14": (2000, 500), "M15": (2000, 500),
    }
    bias = {"VDD": 40.0, "VCM": 32.0, "VB": 20.0, "VC": 26.0}

    freqs = np.logspace(-2, 4, 400)  # 0.01 Hz to 10 kHz

    print("Running AC analysis...")
    result = ac_solve(sizes, bias, freqs)

    if result:
        print(f"DC gain: {result['Av_dc_dB']:.1f} dB   peak: {result['peak_dB']:.1f} dB  (Cadence 19.96 dB)")
        print(f"-3dB BW: {result['bw_Hz']:.1f} Hz  (Cadence 52.3 Hz)")
        print(f"DC op: net2={result['dc_op']['net2']:.1f}V VOP={result['dc_op']['VOP']:.1f}V vfb={result['dc_op']['vfb']:.1f}V")
        print(f"")
        for name in ["M7","M9","M12","M14"]:
            s = result["ss"][name]
            print(f"{name}: gm={s['gm']*1e9:.0f}nS gds={s['gds']*1e9:.2f}nS Cgs={s['Cgs']*1e12:.1f}pF Cgd={s['Cgd']*1e12:.1f}pF")
    else:
        print("DC did not converge")
