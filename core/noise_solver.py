"""
Small-signal NOISE solver for the AFE, built on the same validated MNA used by
ac_solver/ac_mna.

Method (matches what Spectre does):
  1. DC solve -> operating point of every device (reuse ac_solver.ac_solve).
  2. Build the SAME 6-node small-signal Y matrix as the AC analysis. For noise
     the inputs vip/vin carry no signal, so the M7/M8 gates are AC ground; the
     Y matrix is therefore identical to the AC Y (vip/vin only ever sat in the
     AC RHS).
  3. Each transistor's channel produces a drain-source current-noise PSD
     S_id(f) = S_thermal + S_flicker(f)   [A^2/Hz]   (pmos_tft_model.get_noise_psd)
     Inject a unit current between that device's drain and source nodes, solve,
     and read the transimpedance to the differential output:
         Z_k(f) = (vop - von) / i_inject
     The device's output-noise contribution PSD is |Z_k|^2 * S_id_k.
  4. Sum over devices (uncorrelated) -> total differential output noise PSD.
  5. Refer to the amplifier input through the validated gain |H_amp(f)| to get
     the input-referred noise (IRN), the spec quantity.

Ground-truth check (Cadence Spectre, afe_gt/tb_noise.raw/noiseAnal.noise):
  total output 0.05-100 Hz = 2010 uVrms ; IRN = 209.6 uVrms ;
  M12=M13=47%, M14=M15=1.7%, M7=M8=1.1%, M9=M10=0.3%.
"""
import numpy as np
try:
    from .pmos_tft_model import PMOS_TFT
    from .ac_mna import _stamp_adm, _stamp_mos_lti
    from .ac_solver import ac_solve, _dev_corner, _dev_nf
    from .topology import AFE_TOPO
    from .compiled_topology import CompiledTopology
except ImportError:  # pragma: no cover - legacy direct module import
    from pmos_tft_model import PMOS_TFT
    from ac_mna import _stamp_adm, _stamp_mos_lti
    from ac_solver import ac_solve, _dev_corner, _dev_nf
    from topology import AFE_TOPO
    from compiled_topology import CompiledTopology


_KB = 1.380649e-23          # Boltzmann constant [J/K]
_TEMP = 300.15              # physical temperature for resistor thermal noise [K]


def device_psd(W, L, Vs, Vd, Vg, freqs, corner=None, nf=1):
    """Drain-current noise PSD A^2/Hz over freqs: S_th + S_fl_1Hz/f."""
    t = PMOS_TFT(W=W, L=L, NF=nf, **(corner or {}))
    try:
        S_th, S_fl_1 = t.get_noise_psd(Vs, Vd, Vg, frequency=1.0)
    except Exception:
        return np.zeros_like(freqs), 0.0, 0.0
    return S_th + S_fl_1 / freqs, S_th, S_fl_1


def noise_analysis(sizes, bias, freqs, corner=None, x0_guess=None, topo=AFE_TOPO, nf=None):
    # ── 1. DC + small-signal params + gain (reuse the validated AC solver) ──
    ac = ac_solve(sizes, bias, freqs, corner=corner, x0_guess=x0_guess, topo=topo, nf=nf)
    if ac is None:
        return None
    dc = ac["dc_op"]
    ss = ac["ss"]
    Hmag = ac["gains"]                      # |vop-von|/vin_diff at each freq
    plan = CompiledTopology(topo, bias)

    # per-device bias (Vs,Vd,Vg) + AC terminals — DERIVED from the topology.
    # Noise: inputs carry no signal -> M7/M8 gates are AC ground (drive={}),
    # so the Y matrix equals the AC Y. Only the RHS (injected noise) differs.
    node_vals = {nm: dc[nm] for nm in plan.solved}
    bpts = plan.bias_points(node_vals)
    devs = plan.ac_devices(drive={})
    ac_caps = plan.ac_capacitors()
    ac_res = plan.ac_resistors()
    inj = {name: (d, s) for name, d, g, s in devs}   # drain/source for noise injection
    NN = plan.n

    # per-device noise PSD
    psd = {}
    psd_split = {}
    for name in bpts:
        W, L = sizes[name]
        Vs, Vd, Vg = bpts[name]
        S, S_th, S_fl1 = device_psd(W, L, Vs, Vd, Vg, freqs,
                                    corner=_dev_corner(corner, name), nf=_dev_nf(nf, name))
        psd[name] = S
        psd_split[name] = (S_th, S_fl1)

    # ── 2/3. per-frequency: build Y, get transimpedance per device ──
    out_psd = np.zeros(len(freqs))                       # total output V^2/Hz
    dev_psd = {name: np.zeros(len(freqs)) for name in bpts}  # per-device output V^2/Hz
    for rname, *_ in topo.resistors:                     # resistors are noise sources too
        dev_psd[rname] = np.zeros(len(freqs))
    res_inj = [(rname, a, b, 4.0 * _KB * _TEMP / R)
               for rname, a, b, R, _ in ac_res]  # (name, term_a, term_b, S_th)
    sense = plan.output_sense(dtype=complex)

    G = np.zeros((NN, NN), dtype=complex)
    C = np.zeros((NN, NN), dtype=complex)
    RHS_G = np.zeros(NN, dtype=complex)
    RHS_C = np.zeros(NN, dtype=complex)
    for name, d, g, s in devs:
        p = ss[name]
        _stamp_mos_lti(G, C, RHS_G, RHS_C, d, g, s,
                       p["gm"], p["gds"], p["Cgs"], p["Cgd"])
    for a, b, cap in ac_caps:
        _stamp_adm(C, RHS_C, a, b, cap)
    for _, a, b, _, gval in ac_res:
        _stamp_adm(G, RHS_G, a, b, gval)

    jw = (2j * np.pi) * np.asarray(freqs, dtype=float)
    Y = G[None, :, :] + jw[:, None, None] * C[None, :, :]
    # transfer from injecting unit current at node j to (vop - von):
    #   t[j] = (e_vop - e_von)^T Y^-1[:,j]
    tvec = np.linalg.solve(
        np.swapaxes(Y, 1, 2),
        np.broadcast_to(sense, (len(freqs), NN))[..., None],
    )[..., 0]

    def transimpedance(term_d, term_s):
        Z = np.zeros(len(freqs), dtype=complex)
        if term_d[0] == "n":
            Z += tvec[:, term_d[1]]
        if term_s[0] == "n":
            Z -= tvec[:, term_s[1]]
        return Z

    for name in bpts:
        d, s = inj[name]
        contrib = (np.abs(transimpedance(d, s)) ** 2) * psd[name]
        dev_psd[name] = contrib
        out_psd += contrib
    for rname, ta, tb, S_th in res_inj:                  # resistor thermal noise 4kT/R
        contrib = (np.abs(transimpedance(ta, tb)) ** 2) * S_th
        dev_psd[rname] = contrib
        out_psd += contrib

    # ── 4/5. integrate + input-refer ──
    return {
        "freqs": freqs,
        "out_psd": out_psd,          # differential output noise PSD V^2/Hz
        "dev_psd": dev_psd,          # per-device output PSD
        "Hmag": Hmag,                # |amplifier gain|
        "response": ac.get("response"),
        "irn_psd": out_psd / Hmag ** 2,
        "psd_split": psd_split,
        "dc": dc,
    }


def bpts_order(sizes):
    afe_order = ["M6","M7","M8","M9","M10","M11","M12","M13","M14","M15"]
    ordered = [m for m in afe_order if m in sizes]
    return ordered + [m for m in sizes if m not in ordered]


def band_rms(freqs, psd, f_lo, f_hi):
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sqrt(np.trapezoid(psd[mask], freqs[mask])))


# ── run + compare to Cadence ground truth ────────────────────────────
if __name__ == "__main__":
    sizes = {
        "M6": (3000, 150), "M7": (25000, 150), "M8": (25000, 150),
        "M9": (12000, 500), "M10": (12000, 500),
        "M11": (300, 100), "M12": (500, 80), "M13": (500, 80),
        "M14": (2000, 500), "M15": (2000, 500),
    }
    bias = {"VDD": 40.0, "VCM": 32.0, "VB": 20.0, "VC": 26.0}
    freqs = np.logspace(-2, 4, 121)   # match Cadence grid (0.01..10k, dec=20)

    r = noise_analysis(sizes, bias, freqs)
    F_LO, F_HI = 0.05, 100.0

    Vout = band_rms(freqs, r["out_psd"], F_LO, F_HI)
    IRN = band_rms(freqs, r["irn_psd"], F_LO, F_HI)
    print(f"Python total OUTPUT noise {F_LO}-{F_HI} Hz = {Vout*1e6:.1f} uVrms  (Cadence 2010)")
    print(f"Python IRN               {F_LO}-{F_HI} Hz = {IRN*1e6:.1f} uVrms  (Cadence 209.6, spec<=44.5)")
    print(f"midband gain |H| = {r['Hmag'].max():.3f}")

    print("\nPer-device contribution (0.05-100 Hz):")
    var = {nm: band_rms(freqs, p, F_LO, F_HI)**2 for nm, p in r["dev_psd"].items()}
    tot = sum(var.values())
    cad = {"M12":47.0,"M13":47.0,"M14":1.7,"M15":1.7,"M7":1.1,"M8":1.1,"M9":0.3,"M10":0.3,"M6":0,"M11":0}
    for nm in sorted(var, key=lambda k:-var[k]):
        print(f"  {nm:<4} {np.sqrt(var[nm])*1e6:8.1f} uVrms  {var[nm]/tot*100:5.1f}%   (Cadence {cad.get(nm,0):.1f}%)")
