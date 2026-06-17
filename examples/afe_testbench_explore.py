"""Run the design-space exploration on the AFE testbench and render a slide figure.

Optimizes the input pair (M7/M8) width, length and finger count (NF) plus the
output-stage width, subject to passband gain and bandwidth constraints, and maps
the area vs input-referred-noise trade-off. NF is a first-class variable: it
reshapes the gate-finger geometry, so it moves bandwidth and area without changing
the channel W·L (hence flicker noise). Figure:

  left  — area vs IRN scatter; feasible points colored by input-pair NF, the
          Pareto front highlighted, the baseline design marked.
  right — AC magnitude (the bandpass) of the baseline vs the min-area Pareto pick.

Usage:  python examples/afe_testbench_explore.py [n] [seed]
Writes: results/afe_testbench_explore.{csv,png}
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.ac_solver import ac_solve
from core.explore import apply_variables, explore, parse_explore, write_csv
from examples.afe_testbench import build_afe_testbench, dc_seed

EXPLORE_CFG = {
    "variables": {
        "in_pair_W":  {"min": 30000, "max": 90000, "int": True, "targets": ["M7.W", "M8.W"]},
        "in_pair_L":  {"min": 50, "max": 120, "int": True, "targets": ["M7.L", "M8.L"]},
        "in_pair_NF": {"min": 1, "max": 200, "int": True, "targets": ["M7.NF", "M8.NF"]},
        "out_W":      {"min": 2000, "max": 6000, "int": True, "targets": ["M9.W", "M10.W"]},
    },
    "constraints": {"gain_peak_dB": {"min": 20.0}, "bw_Hz": {"min": 100.0}},
    "objectives": {"area": "min", "irn_uV": "min"},
    "band": [0.05, 100.0],
    "freqs": {"start": -2, "stop": 3, "num": 81},
}


def _bandpass(topo, sizes, bias, nf=None):
    f = np.logspace(-3, 4, 160)
    ac = ac_solve(sizes, bias, f, topo=topo, nf=nf, x0_guess=dc_seed(sizes, bias))
    return f, 20 * np.log10(np.maximum(ac["gains"], 1e-12))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    topo, sizes, bias = build_afe_testbench()
    cfg = parse_explore(EXPLORE_CFG)

    def progress(done, total):
        print(f"\r  evaluating {done}/{total}", end="", flush=True)

    res = explore(topo, sizes, bias, None, cfg, n=n, seed=seed, method="lhs",
                  seed_fn=dc_seed, progress=progress)
    print("\n" + "candidates %d  converged %d  feasible %d  pareto %d" % (
        res["summary"]["n"], res["summary"]["converged"],
        res["summary"]["feasible"], res["summary"]["pareto"]))

    os.makedirs("results", exist_ok=True)
    write_csv(res, "results/afe_testbench_explore.csv")

    cands = [c for c in res["candidates"] if c["converged"]]
    feas = [c for c in cands if c["feasible"]]
    par = sorted([c for c in feas if c["pareto"]], key=lambda c: c["metrics"]["area"])

    def xy(c):  # area in 1e6 model-units, IRN in uV
        return c["metrics"]["area"] / 1e6, c["metrics"]["irn_uV"]

    # baseline (default design) evaluated in the testbench, and the min-area pick
    f_b, g_b = _bandpass(topo, sizes, bias)  # default nf=None
    pick = par[0]
    p_sizes, p_bias, p_nf = apply_variables(cfg.variables, pick["vars"], sizes, bias)
    f_p, g_p = _bandpass(topo, p_sizes, p_bias, p_nf)

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "figure.dpi": 150})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # ── left: area vs IRN trade-off ──
    if cands:
        infeas = [c for c in cands if not c["feasible"]]
        if infeas:
            ix, iy = zip(*(xy(c) for c in infeas))
            axL.scatter(ix, iy, s=22, c="#cbd5e1", marker="o", label="infeasible", zorder=1)
    if feas:
        fx, fy = zip(*(xy(c) for c in feas))
        nfc = [c["vars"]["in_pair_NF"] for c in feas]
        sc = axL.scatter(fx, fy, s=42, c=nfc, cmap="viridis", edgecolors="white",
                         linewidths=0.4, label="feasible", zorder=2)
        cb = fig.colorbar(sc, ax=axL, pad=0.02)
        cb.set_label("input-pair fingers (NF)")
    if par:
        px, py = zip(*(xy(c) for c in par))
        axL.plot(px, py, "-", color="#dc2626", lw=1.8, zorder=3)
        axL.scatter(px, py, s=70, facecolors="none", edgecolors="#dc2626",
                    linewidths=1.8, label="Pareto front", zorder=4)
    # baseline marker (default design evaluated in the testbench)
    from core.explore import evaluate
    base_area = base_irn = None
    bm = evaluate(topo, sizes, bias, None, cfg.freqs, cfg.band, x0_guess=dc_seed(sizes, bias))
    if bm:
        base_area, base_irn = bm["area"] / 1e6, bm["irn_uV"]
        axL.scatter([base_area], [base_irn], marker="*", s=260, color="#111827",
                    edgecolors="white", linewidths=0.6, label="baseline", zorder=5)
    axL.set_xlabel(r"core area  ($10^{6}$ model-units)")
    axL.set_ylabel("input-referred noise  0.05–100 Hz  (µVrms)")
    axL.set_title("Design-space exploration: area vs noise")
    axL.grid(True, alpha=0.25)
    axL.legend(loc="upper right", framealpha=0.9, fontsize=10)

    # ── right: bandpass before/after ──
    axR.semilogx(f_b, g_b, color="#6b7280", lw=2.0, label="baseline")
    axR.semilogx(f_p, g_p, color="#2563eb", lw=2.0, label="min-area Pareto pick")
    axR.axhline(g_p.max() - 3, color="#dc2626", ls="--", lw=1.0, alpha=0.8)
    axR.text(1e-3, g_p.max() - 3 + 0.4, "-3 dB", color="#dc2626", fontsize=9)
    axR.set_xlim(1e-3, 1e4)
    axR.set_xlabel("frequency (Hz)")
    axR.set_ylabel("|gain| (dB)")
    axR.set_title("AC response (electrode + AC-coupling + amp)")
    axR.grid(True, which="both", alpha=0.2)
    axR.legend(loc="lower center", fontsize=10)

    fig.tight_layout()
    out = "results/afe_testbench_explore.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out} and results/afe_testbench_explore.csv")

    if bm:
        print(f"baseline: area={base_area:.1f}e6  IRN={base_irn:.1f}uV")
    pm = pick["metrics"]
    print(f"min-area Pareto pick: area={pm['area']/1e6:.1f}e6  IRN={pm['irn_uV']:.1f}uV  "
          f"gain={pm['gain_peak_dB']:.1f}dB  BW={pm['bw_Hz']:.0f}Hz  "
          f"NF={pick['vars']['in_pair_NF']}  W/L(M7)={p_sizes['M7']}")


if __name__ == "__main__":
    main()
