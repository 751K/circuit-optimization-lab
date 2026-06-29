"""Corner table + mismatch Monte-Carlo for an AFE design, with a histogram figure.

Thin driver over core.corners (the consolidated corner / mismatch / latch tools).

Usage:  python examples/mc_mismatch.py [n] [seed]
Writes: results/mc_mismatch.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.corners import corner_table, latch_screen, mismatch_mc

# robust re-size: weak cross-coupled feedback (M12/M13) removes the slow-corner
# latch-up; gain comes from the input pair + output ro instead of regeneration.
SIZES = {"M6": (30000, 73), "M7": (67000, 32), "M8": (67000, 32),
         "M9": (10500, 470), "M10": (10500, 470), "M11": (1060, 50),
         "M12": (320, 350), "M13": (320, 350), "M14": (6000, 70), "M15": (6000, 70)}
NF = {"M7": 224, "M8": 224}
BIAS = {"VDD": 40.0, "VCM": 33.8, "VB": 11.0, "VC": 17.5}

SPEC = dict(gain=24.0, bw=600.0, irn=50.0)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print("=== nominal corner table ===")
    for c, m in corner_table(SIZES, BIAS, nf=NF).items():
        print(f"  {c:8}: gain={m['gain_peak_dB']:.2f}dB  BW={m['bw_Hz']:.0f}Hz  IRN={m['irn_uV']:.1f}uV")
    print(f"latch screen (worst differential kick, slow): {latch_screen(SIZES, BIAS, nf=NF):.2f} V "
          f"(small => robust)\n")

    print(f"=== mismatch MC: 3 corners x {n} ===")
    data = {}
    for c in ("typical", "slow", "fast"):
        mc = mismatch_mc(SIZES, BIAS, nf=NF, base=c, n=n, seed=seed)
        data[c] = mc
        s = mc["summary"]
        print(f"--- {c} ---  latch-up {s['latched']}/{s['n']} = {s['latch_rate']*100:.1f}%")
        for k, lbl in (("gain_peak_dB", "gain dB"), ("bw_Hz", "BW Hz"), ("irn_uV", "IRN uV")):
            d = s[k]
            print(f"    {lbl:8} mean {d['mean']:8.2f}  std {d['std']:7.2f}  P5 {d['p5']:8.2f}  P95 {d['p95']:8.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    cols = {"typical": "#2563eb", "slow": "#dc2626", "fast": "#16a34a"}
    panels = [("gain_peak_dB", "gain (dB)", SPEC["gain"], ">"),
              ("bw_Hz", "-3 dB BW (Hz)", SPEC["bw"], ">"),
              ("irn_uV", "IRN 0.05–100 Hz (µVrms)", SPEC["irn"], "<")]
    for j, (key, label, spec, sense) in enumerate(panels):
        for c in ("typical", "slow", "fast"):
            arr, latched = data[c]["arrays"][key], data[c]["latched"]
            vals = arr[~latched]
            ax[j].hist(vals, bins=30, alpha=0.55, color=cols[c], label=f"{c} (μ={vals.mean():.1f})")
        ax[j].axvline(spec, color="k", ls="--", lw=1.3, label=f"spec {sense}{spec:g}")
        ax[j].set_xlabel(label); ax[j].set_ylabel("runs"); ax[j].legend(fontsize=8)
    fig.suptitle(f"AFE mismatch MC — non-latched runs; slow latch-up "
                 f"{data['slow']['summary']['latch_rate']*100:.0f}%", fontsize=11)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/mc_mismatch.png", dpi=150, bbox_inches="tight")
    print("saved results/mc_mismatch.png")


if __name__ == "__main__":
    main()
