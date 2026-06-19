"""CLI entry point — run a circuit JSON through configured analyses or exploration.

Usage::

    python -m core examples/periodic_rc.json
    python -m core examples/periodic_rc.json -a ac,noise,pss
    python -m core examples/afe_explore.json --explore -n 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    from .analysis_dispatch import run_analysis_suite
    from .circuit_loader import load_circuit_json
    from .explore import explore, load_explore_json
    from .noise_solver import band_rms
except ImportError:  # pragma: no cover - legacy direct module import
    from analysis_dispatch import run_analysis_suite
    from circuit_loader import load_circuit_json
    from explore import explore, load_explore_json
    from noise_solver import band_rms

_ANALYSIS_NAMES = ["ac", "noise", "transient", "pss", "pac", "pnoise"]


def _format_analysis_summary(results):
    lines = []
    for name in _ANALYSIS_NAMES:
        if name not in results or results[name] is None:
            continue
        r = results[name]
        if name == "ac":
            lines.append(
                f"  AC:    gain={r.get('Av_dc_dB', np.nan):.2f} dB  "
                f"BW={r.get('bw_Hz', np.nan):.1f} Hz"
            )
        elif name == "noise":
            irn = r.get("irn_uV_band")
            out = r.get("out_uV_band")
            parts = []
            if irn is not None:
                parts.append(f"IRN={irn:.2f} µVrms")
            if out is not None:
                parts.append(f"out={out:.2f} µVrms")
            if parts:
                lines.append(f"  Noise: {'  '.join(parts)}")
            else:
                lines.append("  Noise: computed")
        elif name == "transient":
            n = len(r.get("nodes", []))
            nfail = r.get("nfail", 0)
            lines.append(f"  Tran:  {n} steps  nfail={nfail}")
        elif name == "pss":
            conv = "✓" if r.get("converged") else "✗"
            res = r.get("residual_norm", np.nan)
            runs = r.get("shooting_period_runs", "?")
            lines.append(f"  PSS:   converged={conv}  residual={res:.2e}  period_runs={runs}")
        elif name == "pac":
            gain = r.get("Av_dc_dB")
            bw = r.get("bw_Hz")
            parts = []
            if gain is not None and np.isfinite(gain):
                parts.append(f"gain={gain:.2f} dB")
            if bw is not None and np.isfinite(bw):
                parts.append(f"BW={bw:.1f} Hz")
            if parts:
                lines.append(f"  PAC:   {'  '.join(parts)}")
            else:
                lines.append("  PAC:   computed")
        elif name == "pnoise":
            irn = r.get("irn_uV_band")
            if irn is not None:
                lines.append(f"  PNoise: IRN={irn:.2f} µVrms")
            else:
                lines.append("  PNoise: computed")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run a circuit JSON through configured analyses or exploration.",
        epilog="Without --explore, runs analyses from the JSON 'analyses' block.",
    )
    ap.add_argument("circuit", help="Path to circuit JSON file")
    ap.add_argument(
        "-a", "--analysis",
        help="Comma-separated analyses to run (default: all configured). "
             f"Choices: {','.join(_ANALYSIS_NAMES)}",
        default=None,
    )
    ap.add_argument(
        "--explore", action="store_true",
        help="Run design-space exploration (requires 'explore' block in JSON)",
    )
    ap.add_argument(
        "--noise-band", nargs=2, type=float, default=(0.05, 100.0), metavar=("LO", "HI"),
        help="IRN integration band in Hz (default: 0.05 100.0)",
    )
    ap.add_argument(
        "-n", "--n", type=int, default=200,
        help="Number of candidates for exploration (default: 200)",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for exploration")
    ap.add_argument("--method", choices=("lhs", "random"), default="lhs",
                    help="Sampling method for exploration (default: lhs)")
    ap.add_argument(
        "-o", "--output", default=None,
        help="Write JSON results to file",
    )
    ap.add_argument("--no-numba", action="store_true", help="Disable Numba acceleration")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = ap.parse_args(argv)

    if args.no_numba:
        os.environ["CIRCUIT_USE_NUMBA"] = "0"

    if not os.path.exists(args.circuit):
        ap.error(f"file not found: {args.circuit}")

    if args.explore:
        topo, sizes, bias, nf, cfg = load_explore_json(args.circuit)

        def progress(done, total):
            if not args.quiet:
                print(f"\r  evaluating {done}/{total}", end="", flush=True)

        if not args.quiet:
            print(f"Exploring {args.circuit}  (n={args.n}, method={args.method})")
        results = explore(topo, sizes, bias, nf, cfg, n=args.n, seed=args.seed,
                          method=args.method, progress=progress)
        if not args.quiet:
            print()
        from .explore import _format_summary
        print(_format_summary(results))
        if args.output:
            from .explore import write_csv, write_jsonl
            os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
            write_csv(results, args.output + ".csv")
            write_jsonl(results, args.output + ".jsonl")
            print(f"wrote {args.output}.csv and {args.output}.jsonl")
        return results

    # ── analysis mode ──
    selected = None
    if args.analysis:
        selected = [s.strip().lower() for s in args.analysis.split(",")]
        unknown = set(selected) - set(_ANALYSIS_NAMES)
        if unknown:
            ap.error(f"unknown analysis: {', '.join(sorted(unknown))}")

    if not args.quiet:
        what = ",".join(selected) if selected else "all configured"
        print(f"Running {what} analyses for {args.circuit}")

    results = run_analysis_suite(args.circuit, selected=selected)

    # Apply band RMS for noise analyses if not already computed
    lo, hi = args.noise_band
    for key in ("noise", "pnoise"):
        r = results.get(key)
        if r is None:
            continue
        if key == "noise" and "irn_uV_band" not in r:
            freqs = np.asarray(r.get("freqs", []))
            if len(freqs):
                r["irn_uV_band"] = band_rms(freqs, r["irn_psd"], lo, hi) * 1e6
                r["out_uV_band"] = band_rms(freqs, r["out_psd"], lo, hi) * 1e6

    if not args.quiet:
        print(_format_analysis_summary(results))

    if args.output:
        serializable = {}
        for name, r in results.items():
            if r is None:
                continue
            if isinstance(r, dict):
                serializable[name] = {
                    k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in r.items()
                    if not callable(v) and not k.startswith("_")
                }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"wrote {args.output}")

    return results


if __name__ == "__main__":
    main()
