"""AFE testbench: dry-electrode model + AC-coupling high-pass in front of the
validated AFE core (Figure 1).

Signal chain, per differential side (source -> amplifier gate):

    Vin --[ R_EL || C_EL ]-- E --[ C_AC series ]-- IN --( amp gate )
                                       |
                                    [ R_AC ] -- VCM        ( + C_AC blocks DC,
                                                             R_AC biases IN to VCM )

  * Dry electrode model:  R_EL = 1 MΩ  ∥  C_EL = 10 nF   (pole ≈ 16 Hz)
  * AC coupling high-pass: C_AC = 100 nF, R_AC = 33 MΩ to VCM
                           f_hp = 1/(2π·R_AC·C_AC) ≈ 0.048 Hz  ("filter < 0.05 Hz")
  * Output load:           C_L = 5 pF on VOP/VON (the AFE's existing load)

The amplifier core is the existing 10-transistor AFE. Its input pair gates (M7/M8),
which used to sit on the VCM rail, become SOLVED nodes (INP/INN) fed by the front
end; the differential stimulus is applied as AC drives at the Vinp/Vinn source nodes
(topology.ac_drives) and propagates through the passive network.

DC note: with C_AC blocking DC, the gates are defined only by the weak R_AC to VCM,
and the AFE is multistable — the generic DC solve will not find the physical branch
on its own. `dc_seed()` therefore solves the bare AFE (which has the robust
symmetric continuation) and seeds the testbench solve from it. Pass the seed as
`x0_guess` to ac_solve / noise_analysis and as `V0` to transient.

Run directly for an AC + noise + transient summary:
    python examples/afe_testbench.py
"""
import numpy as np

from circuitopt.ac_solver import ac_solve
from circuitopt.noise_solver import band_rms, noise_analysis
from circuitopt.topology import AFE_TOPO, Topology
from circuitopt.transient_solver import transient


# "Final locked" AFE design (matches docs/core_overview calibration).
DEFAULT_SIZES = {
    "M6": (2264, 78), "M7": (61365, 61), "M8": (61365, 61),
    "M9": (3175, 468), "M10": (3175, 468), "M11": (465, 66),
    "M12": (894, 85), "M13": (894, 85), "M14": (5224, 46), "M15": (5224, 46),
}
DEFAULT_BIAS = {"VDD": 40.0, "VCM": 30.65, "VB": 9.84, "VC": 16.0,
                "VINP": 0.0, "VINN": 0.0}

# Front-end / output component values from Figure 1.
R_EL, C_EL = 1e6, 10e-9          # dry electrode (parallel RC)
R_AC, C_AC = 33e6, 100e-9        # AC coupling high-pass (~0.05 Hz)
C_LOAD = 5e-12                   # output load on VOP/VON

# Testbench-specific solved nodes appended after the 6 AFE core nodes.
_AFE_CORE = ("VOP", "VON", "VFBP", "VFBN", "NET20", "NET2")


def build_afe_testbench(sizes=None, bias=None, *, r_el=R_EL, c_el=C_EL,
                        r_ac=R_AC, c_ac=C_AC, c_load=C_LOAD, drive=0.5):
    """Build the testbench Topology. Returns (topo, sizes, bias)."""
    sizes = dict(sizes or DEFAULT_SIZES)
    bias = {**DEFAULT_BIAS, **(bias or {})}
    topo = Topology(
        solved=[*_AFE_CORE, "INP", "INN", "EP", "EN"],
        devices=[
            ("M6", "NET2", "VB", "VDD"),
            ("M7", "VOP", "INP", "NET2"),     # + input gate <- front end
            ("M8", "VON", "INN", "NET2"),     # - input gate <- front end
            ("M9", "GND", "VFBP", "VOP"),
            ("M10", "GND", "VFBN", "VON"),
            ("M11", "NET20", "VC", "VDD"),
            ("M12", "VFBN", "VOP", "NET20"),
            ("M13", "VFBP", "VON", "NET20"),
            ("M14", "GND", "GND", "VFBN"),
            ("M15", "GND", "GND", "VFBP"),
        ],
        rails={"VDD": "VDD", "GND": 0.0, "VB": "VB", "VC": "VC", "VCM": "VCM",
               "VINP": "VINP", "VINN": "VINN"},
        outputs=("VOP", "VON"),
        input_drives={},                       # stimulus enters at the source nodes, not gates
        ac_drives={"VINP": +drive, "VINN": -drive},
        resistors=[
            ("REL_P", "VINP", "EP", r_el), ("REL_N", "VINN", "EN", r_el),
            ("RAC_P", "INP", "VCM", r_ac), ("RAC_N", "INN", "VCM", r_ac),
        ],
        capacitors=[
            ("CEL_P", "VINP", "EP", c_el), ("CEL_N", "VINN", "EN", c_el),
            ("CAC_P", "EP", "INP", c_ac), ("CAC_N", "EN", "INN", c_ac),
        ],
        load_caps=[("VOP", "GND", c_load), ("VON", "GND", c_load)],
        transient_inputs={},                   # gates are driven via node_inputs at Vinp/Vinn
    )
    return topo, sizes, bias


def dc_seed(sizes, bias):
    """DC seed for the testbench, taken from the robust bare-AFE solve."""
    ac0 = ac_solve(sizes, bias, np.array([1.0]), topo=AFE_TOPO)
    if ac0 is None:
        raise RuntimeError("bare-AFE DC solve failed; cannot seed the testbench")
    d = ac0["dc_op"]
    seed = {n: d[n] for n in _AFE_CORE}
    seed.update({"INP": bias["VCM"], "INN": bias["VCM"], "EP": 0.0, "EN": 0.0})
    return seed


def _bw_edges(freqs, gains):
    """(-3 dB low, peak, -3 dB high) corner frequencies of the bandpass."""
    peak = gains.max()
    ipk = int(np.argmax(gains))
    thr = peak / np.sqrt(2)
    lo = next((freqs[i] for i in range(ipk, -1, -1) if gains[i] < thr), freqs[0])
    hi = next((freqs[i] for i in range(ipk, len(gains)) if gains[i] < thr), freqs[-1])
    return lo, freqs[ipk], hi, peak


def main():
    topo, sizes, bias = build_afe_testbench()
    seed = dc_seed(sizes, bias)

    # ── AC: the bandpass ──
    freqs = np.logspace(-3, 4, 211)
    ac = ac_solve(sizes, bias, freqs, topo=topo, x0_guess=seed)
    lo, fpk, hi, peak = _bw_edges(freqs, ac["gains"])
    print("=== AFE testbench ===")
    print(f"DC op: INP=INN={ac['dc_op']['INP']:.3f} V (VCM {bias['VCM']}), "
          f"VOP=VON={ac['dc_op']['VOP']:.3f} V")
    print(f"AC: peak {20*np.log10(peak):.2f} dB @ {fpk:.3g} Hz; "
          f"-3 dB band {lo:.3g} Hz .. {hi:.3g} Hz")

    # ── Noise: input-referred over the ECG band ──
    fn = np.logspace(-2, 3, 121)
    nz = noise_analysis(sizes, bias, fn, topo=topo, x0_guess=seed)
    irn = band_rms(fn, nz["irn_psd"], 0.05, 100.0) * 1e6
    print(f"Noise: input-referred 0.05-100 Hz = {irn:.1f} uVrms "
          f"(resistors {sorted(k for k in nz['dev_psd'] if k.startswith('R'))} included)")

    # ── Transient: in-band differential sine ──
    f0, amp = 10.0, 0.5e-3
    t = np.linspace(0, 6 / f0, 1200)
    vip = amp * np.sin(2 * np.pi * f0 * t)
    tr = transient(sizes, bias, t, topo=topo, V0=np.array([seed[n] for n in topo.solved]),
                   inputs={"vip": vip, "vin": -vip}, node_inputs={"VINP": "vip", "VINN": "vin"})
    half = tr["output"][len(t) // 2:]
    out_amp = (half.max() - half.min()) / 2             # output differential, zero-to-peak
    g_tr = out_amp / (2 * amp)                          # / input differential (vip-vin) zero-to-peak
    g_ac = ac_solve(sizes, bias, np.array([f0]), topo=topo, x0_guess=seed)["gains"][0]
    print(f"Transient @ {f0:.0f} Hz: gain {g_tr:.3f} (AC {g_ac:.3f}), nfail={tr['nfail']}/{len(t)-1}")


if __name__ == "__main__":
    main()
