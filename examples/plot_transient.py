"""Plot what the transient signals actually look like, for both testbenches.

Two figures (saved under ``results/`` by default):

* ``transient_afe.png``   — the AFE core amplifying an in-band differential sine:
  the applied stimulus vs. the amplified differential output (VOP−VON).
* ``transient_chopper.png`` — one steady-state chopper period (the converged PSS
  orbit): the amplifier-core differential (the up-modulated / chopped signal at
  f_chop) vs. the RC-filtered differential output (the demodulated baseband with
  its residual chopping ripple).

Run (also available as ``python -m core plot {afe,chopper,transient}``):

    python examples/plot_transient.py                       # both
    python examples/plot_transient.py --afe --f0 20 --amp 1e-3
    python examples/plot_transient.py --chopper --f-chop 200 --input-diff 2e-3
    python examples/plot_transient.py --out-dir /tmp/plots

Use the conda ``daily`` env (Numba) for speed — see docs/environment_performance.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                   # headless: write PNGs, no display
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


# ── AFE core: sine in → amplified sine out ────────────────────────────────────
def plot_afe(f0: float = 10.0, amp: float = 0.5e-3, periods: float = 6.0,
             npts: int = 1200, out_dir: Path | str = RESULTS) -> Path:
    from core.transient_solver import transient
    from examples.afe_testbench import build_afe_testbench, dc_seed

    topo, sizes, bias = build_afe_testbench()
    seed = dc_seed(sizes, bias)
    t = np.linspace(0.0, periods / f0, npts)
    vip = amp * np.sin(2 * np.pi * f0 * t)              # differential half-swing
    tr = transient(sizes, bias, t, topo=topo,
                   V0=np.array([seed[n] for n in topo.solved]),
                   inputs={"vip": vip, "vin": -vip},
                   node_inputs={"VINP": "vip", "VINN": "vin"})
    out_diff = tr["output"]                             # VOP − VON (weighted output)
    half = out_diff[len(t) // 2:]
    gain = ((half.max() - half.min()) / 2) / (2 * amp)  # out zero-pk / in zero-pk (differential)

    tm = t * 1e3                                        # ms
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(tm, vip * 1e3, label="vip", color="tab:blue")
    ax1.plot(tm, -vip * 1e3, label="vin", color="tab:cyan", ls="--")
    ax1.set_ylabel("input [mV]")
    ax1.set_title(f"AFE transient — differential sine {f0:.0f} Hz, ±{amp*1e3:.2g} mV "
                  f"(gain ≈ {gain:.2f})")
    ax1.legend(loc="upper right"); ax1.grid(alpha=0.3)

    ax2.plot(tm, out_diff * 1e3, color="tab:red", label="VOP − VON")
    ax2.set_ylabel("output [mV]"); ax2.set_xlabel("time [ms]")
    ax2.legend(loc="upper right"); ax2.grid(alpha=0.3)

    path = Path(out_dir); path.mkdir(parents=True, exist_ok=True)
    path = path / "transient_afe.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"AFE: gain≈{gain:.3f}, nfail={tr['nfail']}/{len(t)-1}  ->  {path}")
    return path


# ── Chopper: one steady-state period of the converged PSS orbit ───────────────
def plot_chopper(f_chop: float = 225.0, input_diff: float = 1e-3,
                 case: str = "chopper_design3_typical",
                 out_dir: Path | str = RESULTS) -> Path:
    from core.calibration import _sizes, load_reference, resolve_adaptive_config
    from core.chopper import pmos_chopper_pss

    md = load_reference(RESULTS.parent / "calibration" / case)["metadata"]
    c = md["circuit"]; s = md.get("solver", {})
    sizes = _sizes(md); bias = dict(c["bias"]); nf = c.get("nf", 1)
    pss = pmos_chopper_pss(
        sizes, bias, f_chop, switch_size=tuple(c.get("switch_size", (5000, 30))),
        switch_nf=int(c.get("switch_nf", 1)), nf=nf,
        edge_time=float(c.get("edge_time", 20e-6)), input_diff=input_diff,
        input_common_mode=float(c.get("input_common_mode", bias["VCM"])),
        charge_injection=bool(c.get("charge_injection", False)),
        output_filter=tuple(c.get("output_filter", (1e6, 680e-12))),
        tstab_periods=int(s.get("tstab_periods", 2)), n_points=int(s.get("n_points", 321)),
        max_shooting_iters=int(s.get("max_shooting_iters", 5)),
        integration_method=s.get("integration_method", "gear2"),
        analytic_jacobian=True, fallback_least_squares=False, corner=md.get("corner", "typical"),
        adaptive_config=resolve_adaptive_config(s))

    nodes = pss["nodes"]
    tm = pss["t"] * 1e3                                 # ms
    amp_diff = (nodes["CH_AMP_OP"] - nodes["CH_AMP_ON"]) * 1e3     # up-modulated (chopped)
    filt_diff = (nodes["CH_VOP_F"] - nodes["CH_VON_F"]) * 1e3      # demodulated + ripple

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(tm, amp_diff, color="tab:purple")
    ax1.set_ylabel("amp core\nVOP−VON [mV]")
    ax1.set_title(f"Chopper steady-state orbit — f_chop={f_chop:.0f} Hz, "
                  f"input_diff={input_diff*1e3:.2g} mV (one period = {pss['period']*1e3:.2f} ms)")
    ax1.grid(alpha=0.3)
    ax1.annotate("signal up-modulated to f_chop (square-wave chopping)",
                 xy=(0.02, 0.05), xycoords="axes fraction", fontsize=8, color="gray")

    ax2.plot(tm, filt_diff, color="tab:green")
    ax2.set_ylabel("filtered out\nVOP_F−VON_F [mV]"); ax2.set_xlabel("time [ms]")
    ax2.grid(alpha=0.3)
    ax2.annotate("RC-filtered baseband (residual chop ripple)",
                 xy=(0.02, 0.05), xycoords="axes fraction", fontsize=8, color="gray")

    path = Path(out_dir); path.mkdir(parents=True, exist_ok=True)
    path = path / "transient_chopper.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    dc_out = float(np.mean(filt_diff))
    print(f"Chopper: filtered VOP_F−VON_F ≈ {dc_out:.3f} mV "
          f"(gain ≈ {abs(dc_out)/(input_diff*1e3):.2f}), shooting_iters={pss['shooting_iters']}"
          f"  ->  {path}")
    return path


def build_arg_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Attach the transient-plot options to ``ap`` (or a fresh parser)."""
    ap = ap or argparse.ArgumentParser(description="Plot transient signals (AFE / chopper)")
    ap.add_argument("--afe", action="store_true", help="only the AFE sine transient")
    ap.add_argument("--chopper", action="store_true", help="only the chopper orbit")
    ap.add_argument("--f0", type=float, default=10.0, help="AFE sine frequency [Hz] (default: 10)")
    ap.add_argument("--amp", type=float, default=0.5e-3,
                    help="AFE differential half-amplitude [V] (default: 5e-4)")
    ap.add_argument("--periods", type=float, default=6.0,
                    help="AFE sine periods to simulate (default: 6)")
    ap.add_argument("--npts", type=int, default=1200, help="AFE time points (default: 1200)")
    ap.add_argument("--f-chop", type=float, default=225.0,
                    help="chopper frequency [Hz] (default: 225)")
    ap.add_argument("--input-diff", type=float, default=1e-3,
                    help="chopper DC differential input [V] (default: 1e-3)")
    ap.add_argument("--case", default="chopper_design3_typical",
                    help="chopper calibration case dir (default: chopper_design3_typical)")
    ap.add_argument("--out-dir", default=str(RESULTS), help="output directory (default: results/)")
    return ap


def run(args) -> list[Path]:
    both = not (args.afe or args.chopper)
    out = []
    if args.afe or both:
        out.append(plot_afe(f0=args.f0, amp=args.amp, periods=args.periods,
                            npts=args.npts, out_dir=args.out_dir))
    if args.chopper or both:
        out.append(plot_chopper(f_chop=args.f_chop, input_diff=args.input_diff,
                               case=args.case, out_dir=args.out_dir))
    return out


def main(argv=None):
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
