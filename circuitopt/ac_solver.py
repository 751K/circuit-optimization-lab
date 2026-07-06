"""
Small-signal AC solver using MNA (Modified Nodal Analysis).
Solves the full circuit at each frequency, computes gain and BW.
Includes ALL transistors + load capacitors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from .device_factory import (dev_corner, dev_nf, is_per_device_corner,
                             build_devices, get_ss_params, resolve_binding)
from .dc_solver import (DC_FALLBACK_TOL, bounded_least_squares_dc,
                        dc_residual_ok, is_afe_topology,
                        is_pairwise_symmetric_afe, symmetric_continuation,
                        symmetric_seed)
from .topology import AFE_TOPO
from .compiled_topology import CompiledTopology
from . import diagnostics

if TYPE_CHECKING:
    from .device_factory import CircuitBinding


def bw_from_gain(freqs, gains):
    """-3 dB bandwidth from a gain-vs-frequency curve.

    Uses log-space interpolation between the two frequency points that bracket
    the -3 dB crossing for smooth results on logarithmic sweeps.
    """
    freqs = np.asarray(freqs, float)
    gains = np.asarray(gains, float)
    if len(freqs) == 0 or len(gains) == 0:
        return np.nan
    peak = float(np.max(gains))
    a3 = peak / np.sqrt(2.0)
    ipk = int(np.argmax(gains))
    bw = float(freqs[-1])
    for i in range(ipk + 1, len(gains)):
        if gains[i] <= a3:
            f0, f1 = float(freqs[i - 1]), float(freqs[i])
            g0, g1 = float(gains[i - 1]), float(gains[i])
            if g1 == g0:
                bw = f1
            elif f0 > 0.0 and f1 > 0.0:
                x0, x1 = np.log10(f0), np.log10(f1)
                x = x0 + (a3 - g0) * (x1 - x0) / (g1 - g0)
                bw = float(10.0 ** np.clip(x, min(x0, x1), max(x0, x1)))
            else:
                bw = float(f0 + (a3 - g0) * (f1 - f0) / (g1 - g0))
            break
    return bw


def ac_solve(sizes: Mapping[str, tuple[float, float]], bias: Mapping[str, float],
             freqs: np.ndarray, corner: str | Mapping[str, Any] | None = None,
             x0_guess: Any = None, topo: Any = None,
             nf: int | Mapping[str, int] | None = None,
             model_types: Mapping[str, str] | None = None,
             device_kwargs: Mapping[str, Mapping[str, Any]] | None = None, *,
             binding: CircuitBinding | None = None) -> dict | None:
    """
    Full small-signal AC analysis — topology supplied by `topo` (default AFE_TOPO).

    sizes: dict of {name: (W, L)}
    bias: dict with VDD, VCM, VB, VC
    freqs: array of frequencies (Hz)
    corner: process shifts — flat dict (global) or per-device map (mismatch); see dev_corner.
    x0_guess: optional DC seed, either a {node: V} dict (e.g. a prior dc_op) or a vector.
    binding: optional :class:`CircuitBinding` supplying defaults for
        topo/nf/corner/model_types/device_kwargs/x0_guess; explicit non-None kwargs
        override it (binding=None reproduces the legacy path exactly).

    DC KCL, per-device bias mapping, and the AC terminal list are all DERIVED from
    `topo` (see topology.py) — no hand-written per-device wiring here.
    """
    topo, nf, corner, model_types, device_kwargs, x0_guess = resolve_binding(
        binding, topo=topo, nf=nf, corner=corner, model_types=model_types,
        device_kwargs=device_kwargs, x0_guess=x0_guess)
    if topo is None:
        topo = AFE_TOPO
    from scipy.optimize import fsolve
    VCM = topo.default_guess_value(bias)
    gmin = 1e-12
    dc_tol = getattr(topo, "dc_tol", None) or DC_FALLBACK_TOL
    plan = CompiledTopology(topo, bias)
    branch_currents = {}                           # ideal voltage-source currents (p->q interior)

    # ── pre-build device instances so the warm-start Newton cache survives
    #     across fsolve iterations instead of being reset on every Id() call.
    _dev_inst = build_devices(sizes, nf=nf, corner=corner, topo=topo,
                              model_types=model_types, device_kwargs=device_kwargs)

    def Id(name, Vs, Vd, Vg):
        try:
            dev = _dev_inst[name]
            # kcl_sign is +1 for source-high (PMOS/OTFT) — byte-identical to abs() —
            # and -1 for source-low (NMOS), whose drain current leaves the drain.
            return getattr(dev, "kcl_sign", 1.0) * abs(dev.get_Idc(Vs, Vd, Vg))
        except Exception as exc:
            diagnostics.note_critical(
                "model.idc_eval_zeroed", exc,
                detail=f"{name} drain current -> 1e-18 (device eval failed)")
            return 1e-18

    # ── 1. DC solve (residuals built from the topology) ──
    residuals = lambda x: plan.dc_residuals(x, Id, gmin)
    per_dev = is_per_device_corner(corner)
    symmetric_fast = (x0_guess is None and not per_dev
                      and is_pairwise_symmetric_afe(sizes, nf, topo))
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
            symv = symmetric_seed(sizes, bias, Id, gmin)
            svec = topo.guess_vector(symv) if symv is not None else None
            if svec is not None and dc_residual_ok(residuals, svec):
                guesses.append(svec)
                have_symmetric_seed = True
        if symmetric_fast and nv is None and not have_symmetric_seed:
            # Cold solve: run source-ramp continuation on the symmetric system to land on
            # the physical power-up branch Spectre picks (handles symmetry-broken AND
            # degenerate multistable points). Tried before the full 6-node fallback.
            cont = symmetric_continuation(sizes, bias, Id, gmin)
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
                if dc_residual_ok(residuals, sol, tol=dc_tol) or (per_dev and ier == 1):
                    break
            except Exception as exc:
                diagnostics.note("dc.fsolve_guess_fail", exc)
        else:
            # ── FALLBACK (runs ONLY when every standard guess failed; never alters
            # already-converged points). Goal: pick the SAME physical branch Spectre
            # picks, even for multistable points. ──
            base_g = guesses[0] if guesses else [VCM] * topo.n_aug

            def _solve(bias_d, gm, x0):
                try:
                    step_plan = plan if bias_d is bias else CompiledTopology(topo, bias_d)
                    rfun = lambda z: step_plan.dc_residuals(z, Id, gm)
                    s, _, ier, _ = fsolve(rfun, x0, full_output=True, xtol=1e-12,
                                          maxfev=4000)
                    return s if (dc_residual_ok(rfun, s, tol=dc_tol) or
                                 (per_dev and ier == 1)) else None
                except Exception as exc:
                    diagnostics.note("dc.fallback_solve_fail", exc)
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
                    sol = bounded_least_squares_dc(residuals, guesses + [flat], topo, bias,
                                                    tol=dc_tol)
            if sol is None:
                return None  # DC didn't converge even with continuation

        nv = topo.node_vals(sol)                  # {node_name: voltage}, full asymmetric op
        if topo.n_branches:                       # voltage-source branch currents
            branch_currents = {}
            for k, (name, *_r) in enumerate(topo.vsources):
                branch_currents[name] = float(sol[topo.n + k])
            offset = len(topo.vsources)
            for k, (name, *_r) in enumerate(topo.vcvs):
                branch_currents[name] = float(sol[topo.n + offset + k])
            offset += len(topo.vcvs)
            for k, (name, *_r) in enumerate(topo.ccvs):
                branch_currents[name] = float(sol[topo.n + offset + k])

    if getattr(topo, "require_dc_in_box", False) and not topo.in_voltage_box(nv, bias):
        sbox = bounded_least_squares_dc(residuals, guesses, topo, bias, tol=dc_tol)
        if sbox is None:
            return None
        nv = topo.node_vals(sbox)
        if not topo.in_voltage_box(nv, bias):
            return None

    # ── PHYSICALITY GUARD ── No internal node can sit above the supply or below ground
    # here. A solution with e.g. net20 > VDD means the tail M11 is reversed — a
    # non-physical alternate branch Spectre never picks. Re-solve seeded strictly inside
    # the rails and prefer an in-box solution. (Validated designs are already in-box, so
    # this never fires for them.)
    if is_afe_topology(topo) and not per_dev and not topo.in_voltage_box(nv, bias):
        VDD = bias["VDD"]
        box_guesses = ({"VOP": VDD*0.6, "VON": VDD*0.6, "VFBP": VDD*0.4, "VFBN": VDD*0.4,
                        "NET20": VDD-4, "NET2": min(VCM+7, VDD-2)},
                       {"VOP": VDD*0.5, "VON": VDD*0.5, "VFBP": VDD*0.3, "VFBN": VDD*0.3,
                        "NET20": VDD-3, "NET2": VDD-5})
        for g in box_guesses:
            try:
                s2, _, ier, _ = fsolve(residuals, topo.guess_vector(g),
                                       full_output=True, xtol=1e-12, maxfev=4000)
            except Exception as exc:
                diagnostics.note("dc.box_guess_fail", exc)
                continue
            if (dc_residual_ok(residuals, s2, tol=DC_FALLBACK_TOL) and
                    topo.in_voltage_box(topo.node_vals(s2), bias)):
                nv = topo.node_vals(s2); break
        if not topo.in_voltage_box(nv, bias):
            box_vecs = [topo.guess_vector(g) for g in box_guesses]
            s3 = bounded_least_squares_dc(residuals, box_vecs + guesses, topo, bias)
            if s3 is None:
                return None
            nv = topo.node_vals(s3)

    # ── SYMMETRY GUARD ── No per-device mismatch ⇒ physical op is symmetric (VOP=VON,
    # VFBP=VFBN). If fsolve latched to a symmetry-broken root, re-solve the symmetric
    # system seeded from the symmetrized average (right next to the physical root).
    # Only fires on no-mismatch + clearly-asymmetric, so symmetric (validated) points
    # are untouched.
    if (is_afe_topology(topo) and (not per_dev) and
            (abs(nv["VOP"] - nv["VON"]) > 1e-2 or
             abs(nv["VFBP"] - nv["VFBN"]) > 1e-2)):
        avg = [nv["NET2"], 0.5 * (nv["VOP"] + nv["VON"]),
               0.5 * (nv["VFBP"] + nv["VFBN"]), nv["NET20"]]
        symv = symmetric_seed(sizes, bias, Id, gmin, seeds=[avg])
        if symv is not None:
            nv = symv                             # physical symmetric branch (Spectre-matching)

    bpts = plan.bias_points(nv)                   # per-device (Vs, Vd, Vg)

    # ── 2. Small-signal params at the true per-device DC op ──
    ss = {name: get_ss_params(sizes[name][0], sizes[name][1], *bpts[name],
                              corner=dev_corner(corner, name), nf=dev_nf(nf, name),
                              dev_inst=_dev_inst[name])
          for name, *_ in topo.devices}

    # ── 3. Build & solve the small-signal MNA (terminals from the topology) ──
    from .ac_mna import (stamp_adm, stamp_mos_lti, stamp_vccs, stamp_vsource,
                         stamp_vcvs, stamp_cccs, stamp_ccvs)
    NN = plan.n_aug
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
    devs = plan.ac_devices(drive=drive, node_drives=ac_drives)
    ac_caps = plan.ac_capacitors(ac_drives)
    ac_res = plan.ac_resistors(ac_drives)
    out_weights = plan.output_weights

    G = np.zeros((NN, NN), dtype=complex)
    C = np.zeros((NN, NN), dtype=complex)
    RHS_G = np.zeros(NN, dtype=complex)
    RHS_C = np.zeros(NN, dtype=complex)
    for name, d, g, s in devs:
        p = ss[name]
        stamp_mos_lti(G, C, RHS_G, RHS_C, d, g, s,
                       p["gm"], p["gds"], p["Cgs"], p["Cgd"])
    for a, b, cap in ac_caps:
        stamp_adm(C, RHS_C, a, b, cap)
    for _, a, b, _, gval in ac_res:
        stamp_adm(G, RHS_G, a, b, gval)
    for p, q, cp, cn, gm in plan.ac_vccs(ac_drives):
        stamp_vccs(G, RHS_G, p, cp, cn, gm)
    for p, q, bi, e_ac in plan.ac_vsources(ac_drives):  # voltage source: short (E_ac=0)
        stamp_vsource(G, RHS_G, p, q, bi, e_ac)
    for p, q, cp, cn, bi, mu in plan.ac_vcvs(ac_drives):   # VCVS: noiseless
        stamp_vcvs(G, RHS_G, p, q, cp, cn, bi, mu)
    for p, q, ctrl_bi, beta in plan.ac_cccs(ac_drives):    # CCCS: noiseless
        stamp_cccs(G, RHS_G, p, q, ctrl_bi, beta)
    for p, q, ctrl_bi, bi, gamma in plan.ac_ccvs(ac_drives): # CCVS: noiseless
        stamp_ccvs(G, RHS_G, p, q, ctrl_bi, bi, gamma)
    # ideal current sources are open-circuit in the small-signal AC system.
    jw = (2j * np.pi) * np.asarray(freqs, dtype=float)
    Y = G[None, :, :] + jw[:, None, None] * C[None, :, :]
    RHS = RHS_G[None, :] + jw[:, None] * RHS_C[None, :]
    V = np.linalg.solve(Y, RHS[..., None])[..., 0]
    out = np.zeros(len(freqs), dtype=complex)
    for node, weight in out_weights.items():
        out += weight * V[:, plan.idx[node]]
    response = out / vin_norm
    gains = np.abs(response)
    Av_dc = gains[0]
    Av_dc_dB = 20 * np.log10(max(Av_dc, 1e-9))
    peak = gains.max()
    # -3 dB BW via log-space interpolation between the bracketing grid points.
    # The raw "first grid point below peak/√2" pick is biased high and grid-
    # dependent (e.g. +6.8% on a 20 pt/decade sweep); bw_from_gain matches the
    # interpolated Cadence reference and the pac_solver/chopper BW paths.
    bw_Hz = bw_from_gain(freqs, gains)

    dc_op = topo.dc_op_with_aliases(nv)
    return {
        "Av_dc_dB": Av_dc_dB,
        "peak_dB": 20 * np.log10(max(peak, 1e-9)),
        "bw_Hz": bw_Hz,
        "gains": gains,
        "response": response,
        "freqs": freqs,
        "dc_op": dc_op,
        "branch_currents": branch_currents,
        "ss": ss,
        "corner": corner,
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
        print("")
        for name in ["M7","M9","M12","M14"]:
            s = result["ss"][name]
            print(f"{name}: gm={s['gm']*1e9:.0f}nS gds={s['gds']*1e9:.2f}nS Cgs={s['Cgs']*1e12:.1f}pF Cgd={s['Cgd']*1e12:.1f}pF")
    else:
        print("DC did not converge")
