"""
DC sweep: VIN → VOUT + small-signal gain for a PMOS inverter-based amplifier.

Two panels:
  Top:    VOUT vs VIN transfer curves for multiple W/L combos
  Bottom: Gain = dVOUT/dVIN vs VIN — the small-signal gain at each bias

Usage:
    python examples/sweep_vin_vout.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from core.circuit_loader import load_circuit_json
from core.compiled_topology import CompiledTopology
from core.device_model import create_device


def dc_solve_one(topo, bias, sizes, nf_map, gmin=1e-12):
    """Solve DC op for a single bias point. Returns {node: voltage} or None."""
    plan = CompiledTopology(topo, bias)
    dev_inst = {}
    for name, *_ in topo.devices:
        w, l = sizes[name]
        dev_inst[name] = create_device("pmos_tft", W=w, L=l, NF=nf_map.get(name, 1))

    def Id(name, Vs, Vd, Vg):
        try:
            return abs(dev_inst[name].get_Idc(Vs, Vd, Vg))
        except Exception:
            return 1e-18

    residuals = lambda x: plan.dc_residuals(x, Id, gmin)
    VCM = topo.default_guess_value(bias)
    guesses = topo.dc_guess_vectors(bias)

    for x0 in guesses:
        try:
            sol, _, ier, _ = fsolve(residuals, x0, full_output=True,
                                    xtol=1e-12, maxfev=3000)
            if np.linalg.norm(residuals(sol), ord=np.inf) < 1e-10:
                return topo.node_vals(sol)
        except Exception:
            pass

    # fallback: source-ramp continuation
    base = np.array(guesses[0] if guesses else [VCM] * topo.n)
    x = base * 0.1
    for lam in np.linspace(0.1, 1.0, 20):
        bl = {k: (v * lam if isinstance(v, (int, float)) else v) for k, v in bias.items()}
        rp = CompiledTopology(topo, bl)
        try:
            s, _, ier, _ = fsolve(lambda z: rp.dc_residuals(z, Id, gmin),
                                  x, full_output=True, xtol=1e-12, maxfev=4000)
            if np.linalg.norm(rp.dc_residuals(s, Id, gmin), ord=np.inf) < 1e-10:
                x = s
            else:
                return None
        except Exception:
            return None
    return topo.node_vals(x)


def main():
    spec = load_circuit_json("examples/single_stage.json")
    topo = spec.topology
    VDD = spec.bias["VDD"]

    default_nf = {
        name: spec.nf.get(name, 1) if isinstance(spec.nf, dict) else (spec.nf or 1)
        for name, *_ in topo.devices
    }

    combos = [
        ("ref W2k/1.5k",  {"MPU": (2000, 80), "MLD": (1500, 80)}),
        ("W3k/1k",         {"MPU": (3000, 80), "MLD": (1000, 80)}),
        ("W1.5k/2k",       {"MPU": (1500, 80), "MLD": (2000, 80)}),
        ("W4k/0.8k",       {"MPU": (4000, 80), "MLD": (800,  80)}),
        ("W2k/3k",         {"MPU": (2000, 80), "MLD": (3000, 80)}),
        ("W1k/4k",         {"MPU": (1000, 80), "MLD": (4000, 80)}),
    ]

    n_vin = 201  # higher res for smooth derivative
    vin_sweep = np.linspace(0, VDD, n_vin)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for label, sizes in combos:
        vout = np.full(n_vin, np.nan)
        for i, vin in enumerate(vin_sweep):
            nv = dc_solve_one(topo, {"VDD": VDD, "VIN": vin}, sizes, default_nf)
            if nv is not None and "OUT" in nv:
                vout[i] = nv["OUT"]

        mask = ~np.isnan(vout)
        vin_ok = vin_sweep[mask]
        vout_ok = vout[mask]

        # ── Transfer curve ──
        ax1.plot(vin_ok, vout_ok, label=label, linewidth=1.5)

        # ── Gain = dVOUT / dVIN (numerical derivative, smoothed) ──
        if len(vin_ok) > 10:
            # smooth VOUT slightly for cleaner derivative
            from scipy.ndimage import uniform_filter1d
            vout_sm = uniform_filter1d(vout_ok, size=7)
            gain = np.gradient(vout_sm, vin_ok)
            gain_db = 20 * np.log10(np.abs(gain) + 1e-20)
            # clip extreme values for readability
            gain_db = np.clip(gain_db, -60, 60)
            ax2.plot(vin_ok, gain_db, linewidth=1.2)

    # ── Top panel: transfer ──
    ax1.plot([0, VDD], [0, VDD], "k--", alpha=0.15, label="VOUT=VIN")
    ax1.set_ylabel("VOUT (V)")
    ax1.set_title("PMOS Inverter Amplifier — DC Transfer & Small-Signal Gain")
    ax1.legend(fontsize=7.5, ncol=2)
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(-0.5, VDD + 0.5)

    # ── Bottom panel: gain ──
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.15)
    ax2.set_xlabel("VIN (V)")
    ax2.set_ylabel("Gain (dB)")
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(0, VDD)

    fig.tight_layout()
    fig.savefig("results/vin_vout_sweep.png", dpi=150)
    plt.show()
    print("Saved → results/vin_vout_sweep.png")


if __name__ == "__main__":
    main()
