"""CLI entry point — run circuit analyses, exploration, corners, mismatch, or chopper.

Usage::

    # Analysis dispatch (default)
    python -m circuitopt examples/periodic_rc.json
    python -m circuitopt examples/periodic_rc.json -a ac,noise,pss

    # Exploration
    python -m circuitopt examples/afe_explore.json --explore -n 300

    # Corners
    python -m circuitopt corners examples/afe_explore.json
    python -m circuitopt corners examples/afe_explore.json --corner slow --freqs-num 61

    # Mismatch Monte Carlo
    python -m circuitopt mc examples/afe_explore.json -n 300 --seed 1
    python -m circuitopt mc examples/afe_explore.json --corner typical --quiet

    # Chopper analysis
    python -m circuitopt chopper examples/afe_explore.json --level ideal
    python -m circuitopt chopper examples/afe_explore.json --level pss --f-chop 225
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# ── Numba flag pre-scan backstop ──────────────────────────────────────────────
# The authoritative `--no-numba` pre-scan lives in circuitopt/__init__.py, which runs
# (and bakes numba_kernels' USE_NUMBA flag via its transitive solver imports)
# *before* this module executes under `python -m circuitopt`. This repeat here is a
# cheap backstop for any path that reaches __main__ without that having run; it
# must still precede the solver imports below. If both are somehow bypassed,
# _assert_numba_flag() turns the silent no-op into a loud SystemExit.
if "--no-numba" in sys.argv:
    os.environ["CIRCUIT_USE_NUMBA"] = "0"

from .analysis_dispatch import run_analysis_suite
from .chopper import (chopper_analysis, pmos_chopper_analysis,
                      pmos_chopper_lptv_analysis, pmos_chopper_pac,
                      pmos_chopper_pnoise, pmos_chopper_pss,
                      pmos_chopper_transient)
from .circuit_loader import load_circuit_json
from .corners import corner_table, mismatch_mc_from_dict
from .dataset import add_cli_args as dataset_add_cli_args
from .dataset import run_cli as dataset_run_cli
from .explore import add_cli_args as explore_add_cli_args
from .explore import run_cli as explore_run_cli
from .noise_solver import band_rms
# The service subpackage's CLI glue is fastapi-free (fastapi/uvicorn are imported
# lazily inside serve_run_cli), so importing it here never pulls the serve extra.
from .service import add_cli_args as serve_add_cli_args
from .service import run_cli as serve_run_cli

_ANALYSIS_NAMES = ["ac", "noise", "transient", "pss", "pac", "pnoise"]
_SUBCOMMANDS = ["run", "corners", "mc", "chopper", "adc", "explore", "plot", "dataset", "serve"]
_CHOPPER_LEVELS = ["ideal", "pmos", "lptv", "pss", "pac", "pnoise", "transient"]


# ── shared helpers ───────────────────────────────────────────────────────────

def _assert_numba_flag(args):
    """Fail loudly if the pure-Python engine was requested but Numba is still active.

    Both ``--no-numba`` and ``--engine python`` select the pure-Python engine. The
    real work is done by the argv pre-scan in ``circuitopt/__init__.py``
    (``_engine.apply_engine_env`` sets CIRCUIT_USE_NUMBA=0 before
    ``circuitopt.numba_kernels`` is imported and bakes its flags). This guard is a
    tripwire: if someone reorders the imports, imports a solver module before
    ``circuitopt``'s pre-scan runs, or otherwise defeats it, the request would
    silently no-op again. Checking the *baked* value here converts that silent
    failure into a loud one instead of a wrong-but-quiet run.
    """
    wants_python = (getattr(args, "no_numba", False)
                    or getattr(args, "engine", None) == "python")
    if not wants_python:
        return
    from . import numba_kernels
    if numba_kernels.USE_NUMBA:
        raise SystemExit(
            "the pure-Python engine was requested (--no-numba or --engine python) "
            "but Numba is already active (circuitopt.numba_kernels.USE_NUMBA is True). "
            "The CIRCUIT_USE_NUMBA flag is baked when numba_kernels is first imported; "
            "a solver module was imported before the argv pre-scan in "
            "circuitopt/__init__.py could set it."
        )


def _load_spec(path):
    """Load a CircuitSpec from a JSON path, or raise SystemExit."""
    if not os.path.exists(path):
        raise SystemExit(f"file not found: {path}")
    return load_circuit_json(path)


def _freqs_from_args(args):
    """Build a frequency grid from --freqs-* CLI flags."""
    return np.logspace(np.log10(args.freqs_start), np.log10(args.freqs_stop),
                       args.freqs_num)


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


def _add_freqs_args(parser):
    """Add --freqs-* arguments to a parser."""
    parser.add_argument("--freqs-start", type=float, default=0.01,
                        help="Start frequency in Hz (default: 0.01)")
    parser.add_argument("--freqs-stop", type=float, default=1e4,
                        help="Stop frequency in Hz (default: 10000)")
    parser.add_argument("--freqs-num", type=int, default=121,
                        help="Number of log-spaced frequency points (default: 121)")


def _add_noise_band_arg(parser):
    parser.add_argument("--noise-band", nargs=2, type=float, default=(0.05, 100.0),
                        metavar=("LO", "HI"),
                        help="IRN integration band in Hz (default: 0.05 100.0)")


def _add_output_arg(parser):
    parser.add_argument("-o", "--output", default=None,
                        help="Write results to file (JSON for analysis, CSV+JSONL for explore/mc)")


def _add_engine_arg(parser):
    """Add ``--engine`` to a subcommand that also offers ``--no-numba``.

    The flag only appears in ``--help`` and gets argparse validation here; its
    *effect* comes from the argv pre-scan (``circuitopt._engine.apply_engine_env``
    via ``circuitopt/__init__.py``), which runs before this parser is built.
    """
    parser.add_argument("--engine", choices=("rust", "numba", "python"), default=None,
                        help="Compute engine (default: numba). 'python' == --no-numba; "
                             "'rust' uses circuitopt_core when installed, else warns and "
                             "falls back to numba. See circuitopt._engine.")


# ── subcommand: run (analysis dispatch, default) ──────────────────────────────

def _add_run_parser(subparsers):
    p = subparsers.add_parser(
        "run",
        help="Run analyses configured in the JSON 'analyses' block (default)",
    )
    p.add_argument("circuit", help="Path to circuit JSON file")
    p.add_argument(
        "-a", "--analysis",
        help="Comma-separated analyses to run (default: all configured). "
             f"Choices: {','.join(_ANALYSIS_NAMES)}",
        default=None,
    )
    p.add_argument("--corner", default=None,
                   help="Process corner override: OTFT typical|slow|fast, or silicon "
                        "tt|ss|ff|sf|fs (SKY130) / nom|tt|ss|ff|sf|fs (FreePDK45; sf = "
                        "NMOS slow + PMOS fast, fs the reverse)")
    _add_noise_band_arg(p)
    _add_output_arg(p)
    p.add_argument("--no-numba", action="store_true", help="Disable Numba acceleration")
    _add_engine_arg(p)
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p


def _cmd_run(args):
    _assert_numba_flag(args)

    selected = None
    if args.analysis:
        selected = [s.strip().lower() for s in args.analysis.split(",")]
        unknown = set(selected) - set(_ANALYSIS_NAMES)
        if unknown:
            raise SystemExit(f"unknown analysis: {', '.join(sorted(unknown))}")

    if not args.quiet:
        what = ",".join(selected) if selected else "all configured"
        print(f"Running {what} analyses for {args.circuit}")

    results = run_analysis_suite(args.circuit, selected=selected, corner=args.corner)

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


# ── subcommand: explore ──────────────────────────────────────────────────────

def _add_explore_parser(subparsers):
    p = subparsers.add_parser("explore", help="Run design-space exploration")
    # Feature args (positional + sampling/corner/output/quiet) come from the single
    # source in circuitopt.explore so this subcommand can't drift from `python -m circuitopt.explore`.
    explore_add_cli_args(p)
    # Subcommand-level mechanism — not a feature arg, so it stays here.
    p.add_argument("--no-numba", action="store_true", help="Disable Numba acceleration")
    _add_engine_arg(p)
    return p


def _cmd_explore(args):
    _assert_numba_flag(args)
    return explore_run_cli(args)


# ── subcommand: dataset ──────────────────────────────────────────────────────

def _add_dataset_parser(subparsers):
    p = subparsers.add_parser(
        "dataset", help="Build a labeled surrogate dataset from an 'explore' config")
    # Feature args come from the single source in circuitopt.dataset so this subcommand
    # can't drift from `python -m circuitopt.dataset`.
    dataset_add_cli_args(p)
    # Subcommand-level mechanism — not a feature arg, so it stays here.
    p.add_argument("--no-numba", action="store_true", help="Disable Numba acceleration")
    _add_engine_arg(p)
    return p


def _cmd_dataset(args):
    _assert_numba_flag(args)
    return dataset_run_cli(args)


# ── subcommand: serve ────────────────────────────────────────────────────────

def _add_serve_parser(subparsers):
    p = subparsers.add_parser(
        "serve", help="Start the local FastAPI service over the solvers (needs the 'serve' extra)")
    # Feature args come from the single source in circuitopt.service.cli so this
    # subcommand can't drift from `python -m circuitopt.service`.
    serve_add_cli_args(p)
    return p


def _cmd_serve(args):
    return serve_run_cli(args)


# ── subcommand: corners ──────────────────────────────────────────────────────

def _add_corners_parser(subparsers):
    p = subparsers.add_parser("corners", help="Run process-corner sweep (typ/slow/fast)")
    p.add_argument("circuit", help="Path to circuit JSON file")
    _add_freqs_args(p)
    _add_noise_band_arg(p)
    _add_output_arg(p)
    p.add_argument("--no-numba", action="store_true", help="Disable Numba acceleration")
    _add_engine_arg(p)
    p.add_argument("--quiet", action="store_true", help="Suppress per-corner output")
    return p


def _cmd_corners(args):
    _assert_numba_flag(args)

    spec = _load_spec(args.circuit)
    freqs = _freqs_from_args(args)
    lo, hi = args.noise_band

    if not args.quiet:
        print(f"Corner sweep for {args.circuit}")
        print(f"  freqs: {args.freqs_start:.2g}–{args.freqs_stop:.2g} Hz ({args.freqs_num} pts)")
        print(f"  band:  {lo}–{hi} Hz")

    table = corner_table(spec.sizes, spec.bias, nf=spec.nf,
                         topo=spec.topology, freqs=freqs)
    for corner_name, metrics in table.items():
        if metrics is None:
            print(f"  {corner_name:>7s}:  (failed)")
            continue
        print(f"  {corner_name:>7s}:  "
              f"gain={metrics['gain_peak_dB']:.2f} dB  "
              f"BW={metrics['bw_Hz']:.0f} Hz  "
              f"IRN={metrics['irn_uV']:.2f} µVrms")

    if args.output:
        csv_lines = ["corner,gain_peak_dB,bw_Hz,irn_uV"]
        for corner_name, metrics in table.items():
            if metrics is None:
                continue
            csv_lines.append(
                f"{corner_name},{metrics['gain_peak_dB']:.4f},"
                f"{metrics['bw_Hz']:.1f},{metrics['irn_uV']:.3f}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write("\n".join(csv_lines) + "\n")
        print(f"wrote {args.output}")

    return table


# ── subcommand: mc (mismatch Monte Carlo) ────────────────────────────────────

def _add_mc_parser(subparsers):
    p = subparsers.add_parser("mc", help="Run per-device mismatch Monte Carlo")
    p.add_argument("circuit", help="Path to circuit JSON file")
    p.add_argument("-n", "--n", type=int, default=200,
                   help="Number of MC samples (default: 200)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--corner", choices=("typical", "slow", "fast"), default="typical",
                   help="Base process corner (default: typical)")
    _add_freqs_args(p)
    _add_noise_band_arg(p)
    _add_output_arg(p)
    p.add_argument("--no-numba", action="store_true", help="Disable Numba acceleration")
    _add_engine_arg(p)
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p


def _cmd_mc(args):
    _assert_numba_flag(args)

    if not os.path.exists(args.circuit):
        raise SystemExit(f"file not found: {args.circuit}")
    with open(args.circuit, "r", encoding="utf-8") as f:
        data = json.load(f)
    freqs = _freqs_from_args(args)
    lo, hi = args.noise_band

    if not args.quiet:
        print(f"Mismatch MC for {args.circuit}")
        print(f"  n={args.n}  seed={args.seed}  corner={args.corner}")
        print(f"  freqs: {args.freqs_start:.2g}–{args.freqs_stop:.2g} Hz ({args.freqs_num} pts)")
        print(f"  band:  {lo}–{hi} Hz")

    # mismatch_mc_from_dict is the shared CLI/service entry (parses the circuit +
    # calls mismatch_mc), so `circuit-opt mc` and POST /jobs/mc can't drift.
    mc = mismatch_mc_from_dict(data, n=args.n, seed=args.seed, corner=args.corner,
                               freqs=freqs, band=(lo, hi))

    summary = mc["summary"]
    latch_rate = float(mc["latched"].mean())

    print(f"  latch_rate: {latch_rate*100:.1f}%")
    if "irn_uV" in summary:
        irn = summary["irn_uV"]
        print(f"  IRN:        {irn['mean']:.2f} ± {irn['std']:.2f} µVrms  "
              f"(P5={irn['p5']:.2f}  P95={irn['p95']:.2f})")
    if "gain_peak_dB" in summary:
        g = summary["gain_peak_dB"]
        print(f"  gain:       {g['mean']:.2f} ± {g['std']:.2f} dB")
    if "bw_Hz" in summary:
        b = summary["bw_Hz"]
        print(f"  BW:         {b['mean']:.0f} ± {b['std']:.0f} Hz")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        out = {
            "latch_rate": latch_rate,
            "n_samples": args.n,
            "seed": args.seed,
            "corner": args.corner,
            "summary": {k: {sk: float(sv) for sk, sv in v.items()}
                        for k, v in summary.items()},
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"wrote {args.output}")

    return mc


# ── subcommand: chopper ──────────────────────────────────────────────────────

def _add_chopper_parser(subparsers):
    p = subparsers.add_parser("chopper", help="Run chopper analysis (ideal / PMOS / PSS / PAC / PNoise)")
    p.add_argument("circuit", help="Path to circuit JSON file")
    p.add_argument("--level", choices=_CHOPPER_LEVELS, default="ideal",
                   help="Chopper analysis level (default: ideal). "
                        "ideal=square-wave LPTV, pmos=static-phase, "
                        "lptv=PMOS sideband fold, pss/pac/pnoise=first-principles, "
                        "transient=hard-switched")
    p.add_argument("--f-chop", type=float, default=225.0,
                   help="Chopper frequency in Hz (default: 225)")
    p.add_argument("--switch-w", type=float, default=5000.0,
                   help="Switch width in µm (default: 5000)")
    p.add_argument("--switch-l", type=float, default=30.0,
                   help="Switch length in µm (default: 30)")
    p.add_argument("--edge-time", type=float, default=20e-6,
                   help="Clock rise/fall time in seconds (default: 20e-6)")
    p.add_argument("--max-harmonic", type=int, default=31,
                   help="Max harmonic for ideal/LPTV folding (default: 31)")
    p.add_argument("--max-sideband", type=int, default=10,
                   help="Max sideband for PNoise (default: 10)")
    p.add_argument("--tstab-periods", type=int, default=2,
                   help="Stabilization periods before PSS shooting (default: 2)")
    p.add_argument("--n-points", type=int, default=121,
                   help="Time points per period for PSS/transient (default: 121)")
    p.add_argument("--n-periods", type=float, default=8.0,
                   help="Simulation duration in periods for transient (default: 8)")
    _add_freqs_args(p)
    _add_noise_band_arg(p)
    _add_output_arg(p)
    p.add_argument("--no-numba", action="store_true", help="Disable Numba acceleration")
    _add_engine_arg(p)
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p


def _cmd_chopper(args):
    _assert_numba_flag(args)

    spec = _load_spec(args.circuit)
    freqs = _freqs_from_args(args)
    lo, hi = args.noise_band
    switch_size = (args.switch_w, args.switch_l)
    band = (lo, hi)

    if not args.quiet:
        print(f"Chopper analysis ({args.level}) for {args.circuit}")
        print(f"  f_chop={args.f_chop} Hz  switch={args.switch_w:.0f}/{args.switch_l:.0f}")

    level = args.level

    # ── ideal chopper ──
    if level == "ideal":
        result = chopper_analysis(
            spec.sizes, spec.bias, freqs,
            f_chop=args.f_chop,
            topo=spec.topology, nf=spec.nf,
            max_harmonic=args.max_harmonic,
            band=band,
        )
        print(f"  peak: {result['peak_dB']:.2f} dB  "
              f"IRN: {result['irn_uV_band']:.2f} µVrms")

    # ── PMOS static-phase chopper ──
    elif level == "pmos":
        result = pmos_chopper_analysis(
            spec.sizes, spec.bias, freqs,
            switch_size=switch_size,
            band=band,
            nf=spec.nf,
        )
        print(f"  peak: {result['peak_dB']:.2f} dB  "
              f"IRN: {result['irn_uV_band']:.2f} µVrms")

    # ── PMOS LPTV sideband fold ──
    elif level == "lptv":
        result = pmos_chopper_lptv_analysis(
            spec.sizes, spec.bias, freqs,
            args.f_chop,
            switch_size=switch_size,
            edge_time=args.edge_time,
            nf=spec.nf,
            max_harmonic=args.max_harmonic,
            band=band,
        )
        print(f"  peak: {result['peak_dB']:.2f} dB  "
              f"BW: {result['bw_Hz']:.1f} Hz  "
              f"IRN: {result['irn_uV_band']:.2f} µVrms")

    # ── PSS ──
    elif level == "pss":
        result = pmos_chopper_pss(
            spec.sizes, spec.bias,
            args.f_chop,
            switch_size=switch_size,
            edge_time=args.edge_time,
            tstab_periods=args.tstab_periods,
            n_points=args.n_points,
            nf=spec.nf,
        )
        conv = "✓" if result.get("converged") else "✗"
        res = result.get("residual_norm", np.nan)
        runs = result.get("shooting_period_runs", "?")
        print(f"  converged={conv}  residual={res:.2e}  period_runs={runs}")

    # ── PAC ──
    elif level == "pac":
        pss = pmos_chopper_pss(
            spec.sizes, spec.bias,
            args.f_chop,
            switch_size=switch_size,
            edge_time=args.edge_time,
            tstab_periods=args.tstab_periods,
            n_points=args.n_points,
            nf=spec.nf,
        )
        result = pmos_chopper_pac(
            spec.sizes, spec.bias, freqs,
            args.f_chop,
            pss_result=pss,
            nf=spec.nf,
        )
        gain = result.get("Av_dc_dB")
        bw = result.get("bw_Hz")
        if gain is not None and np.isfinite(gain):
            print(f"  gain: {gain:.2f} dB  BW: {bw:.1f} Hz")
        else:
            print("  PAC: computed")

    # ── PNoise ──
    elif level == "pnoise":
        pss = pmos_chopper_pss(
            spec.sizes, spec.bias,
            args.f_chop,
            switch_size=switch_size,
            edge_time=args.edge_time,
            tstab_periods=args.tstab_periods,
            n_points=args.n_points,
            nf=spec.nf,
        )
        pac = pmos_chopper_pac(
            spec.sizes, spec.bias, freqs,
            args.f_chop,
            pss_result=pss,
            nf=spec.nf,
        )
        result = pmos_chopper_pnoise(
            spec.sizes, spec.bias, freqs,
            args.f_chop,
            pss_result=pss,
            pac_result=pac,
            nf=spec.nf,
            max_sideband=args.max_sideband,
            band=band,
        )
        irn = result.get("irn_uV_band")
        if irn is not None:
            print(f"  IRN: {irn:.2f} µVrms")
        else:
            print("  PNoise: computed")

    # ── transient ──
    elif level == "transient":
        n_periods = args.n_periods
        t_end = n_periods / args.f_chop
        n_steps = int(n_periods * args.n_points)
        t = np.linspace(0, t_end, n_steps)
        result = pmos_chopper_transient(
            spec.sizes, spec.bias, t,
            args.f_chop,
            switch_size=switch_size,
            edge_time=args.edge_time,
            nf=spec.nf,
        )
        nfail = result.get("nfail", 0)
        print(f"  steps: {len(t)}  nfail={nfail}")

    if args.output and result:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        serializable = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in result.items()
            if not callable(v) and not k.startswith("_")
        }
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"wrote {args.output}")

    return result


# ── subcommand: plot ─────────────────────────────────────────────────────────

_PLOT_KINDS = ["all", "transient", "bode", "afe", "chopper", "ac", "pac"]


def _add_plot_parser(subparsers):
    p = subparsers.add_parser(
        "plot", help="Render signal plots (transient waveforms, AC/PAC Bode) to PNG")
    p.add_argument("kind", nargs="?", default="all", choices=_PLOT_KINDS,
                   help="what to plot (default: all). transient=afe+chopper waveforms, "
                        "bode=ac+pac; or a single one: afe/chopper/ac/pac")
    p.add_argument("--f0", type=float, default=10.0,
                   help="AFE transient sine frequency [Hz] (default: 10)")
    p.add_argument("--amp", type=float, default=0.5e-3,
                   help="AFE transient differential half-amplitude [V] (default: 5e-4)")
    p.add_argument("--f-chop", type=float, default=225.0,
                   help="chopper frequency [Hz] for chopper/pac plots (default: 225)")
    p.add_argument("--input-diff", type=float, default=1e-3,
                   help="chopper transient DC differential input [V] (default: 1e-3)")
    p.add_argument("--npts", type=int, default=None,
                   help="Bode frequency points (per-plot default when omitted)")
    p.add_argument("--out-dir", default="results", help="output directory (default: results)")
    p.add_argument("--no-numba", action="store_true", help="Disable Numba acceleration")
    _add_engine_arg(p)
    p.add_argument("--quiet", action="store_true", help="Suppress the summary line")
    return p


def _cmd_plot(args):
    _assert_numba_flag(args)
    try:
        from examples import plot_bode as pbd
        from examples import plot_transient as ptr
    except ImportError as exc:                          # matplotlib is an optional dep
        raise SystemExit(f"plotting needs matplotlib ({exc}); pip install matplotlib")

    kind = args.kind
    outs = []
    if kind in ("all", "transient", "afe"):
        outs.append(ptr.plot_afe(f0=args.f0, amp=args.amp, out_dir=args.out_dir))
    if kind in ("all", "transient", "chopper"):
        outs.append(ptr.plot_chopper(f_chop=args.f_chop, input_diff=args.input_diff,
                                     out_dir=args.out_dir))
    if kind in ("all", "bode", "ac"):
        kw = {"out_dir": args.out_dir}
        if args.npts:
            kw["npts"] = args.npts
        outs.append(pbd.plot_ac(**kw))
    if kind in ("all", "bode", "pac"):
        kw = {"f_chop": args.f_chop, "out_dir": args.out_dir}
        if args.npts:
            kw["npts"] = args.npts
        outs.append(pbd.plot_pac(**kw))
    if not args.quiet:
        print(f"wrote {len(outs)} figure(s) to {args.out_dir}/")
    return outs


# ── subcommand: ADC conversion ───────────────────────────────────────────────

def _add_adc_parser(subparsers):
    p = subparsers.add_parser(
        "adc", help="Run a closed-loop transistor-level ADC conversion or ramp sweep")
    p.add_argument("circuit", help="Path to a circuit JSON carrying an 'adc' block")
    p.add_argument("--vin", type=float, default=None,
                   help="single conversion input voltage (the default mode; runs at "
                        "0.5 V when no mode flag is given at all)")
    p.add_argument("--sweep", type=int, default=None, metavar="N",
                   help="run N uniformly spaced ramp samples instead of one conversion")
    p.add_argument("--sine", type=int, default=None, metavar="N",
                   help="run an N-sample coherent sine conversion and FFT metrics")
    p.add_argument("--mc", type=int, default=None, metavar="N",
                   help="run an N-trial per-instance mismatch Monte-Carlo (uses the "
                        "circuit's adc.mismatch config; --seed/--workers/--corner apply)")
    p.add_argument("--tone-bin", type=int, default=3,
                   help="coherent sine FFT bin (default: 3)")
    p.add_argument("--sample-rate", type=float, default=10e6,
                   help="reported ADC sample rate in Hz (default: 10e6)")
    p.add_argument("--amplitude", type=float, default=None,
                   help="sine peak amplitude (default: 0.45*vref)")
    p.add_argument("--offset", type=float, default=None,
                   help="sine DC offset (default: 0.5*vref)")
    p.add_argument("--corner", default=None, choices=["nom", "ss", "ff"],
                   help="FreePDK45 process corner override")
    p.add_argument("--explore", default=None, metavar="CONFIG",
                   help="run ADC design-space exploration from a standalone SAR-explore "
                        "config JSON (mutually exclusive with --vin/--sweep/--sine/--mc)")
    p.add_argument("-n", "--n", type=int, default=50,
                   help="explore: number of candidates (default: 50)")
    p.add_argument("--seed", type=int, default=0, help="explore: RNG seed")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel conversions (sweep/sine) or candidates (explore); "
                        "default 1 (serial). Bit decisions within a conversion stay serial.")
    p.add_argument("--csv", default=None, help="explore: write candidate rows to this CSV")
    p.add_argument("--jsonl", default=None, help="explore: write candidate rows to this JSONL")
    p.add_argument("--plot", nargs="?", const="results", default=None, metavar="DIR",
                   help="render the figure(s) matching the run mode into DIR "
                        "(default: results/); needs matplotlib")
    _add_output_arg(p)
    p.add_argument("--quiet", action="store_true", help="Suppress summary output")
    return p


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()
                if not callable(v) and not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _cmd_adc_explore(args):
    """ADC design-space exploration path of the ``adc`` subcommand (``--explore``)."""
    from .sar_explore import (format_sar_summary, load_sar_explore_json,
                              sar_explore, sar_write_csv, sar_write_jsonl)
    if not os.path.exists(args.explore):
        raise SystemExit(f"file not found: {args.explore}")
    if not os.path.exists(args.circuit):
        raise SystemExit(f"file not found: {args.circuit}")
    spec, cfg = load_sar_explore_json(args.explore, circuit_path=args.circuit)

    def progress(done, total):
        if not args.quiet:
            print(f"\r  evaluating {done}/{total}", end="", flush=True)

    if not args.quiet:
        print(f"ADC explore {args.explore}  (circuit={args.circuit}, n={args.n}, "
              f"workers={args.workers})")
    results = sar_explore(spec, cfg, n=args.n, seed=args.seed, corner=args.corner,
                          workers=args.workers, progress=progress)
    if not args.quiet:
        print()
    print(format_sar_summary(results))
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        sar_write_csv(results, args.csv)
        if not args.quiet:
            print(f"wrote {args.csv}")
    if args.jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.jsonl)) or ".", exist_ok=True)
        sar_write_jsonl(results, args.jsonl)
        if not args.quiet:
            print(f"wrote {args.jsonl}")
    return results


_ADC_PLOT_FUNCS = {"vin": "plot_sar_conversion", "sweep": "plot_sar_static",
                   "sine": "plot_sar_spectrum", "mc": "plot_sar_mc"}


def _adc_plot(args, mode, result, spec):
    """Render the ADC figure matching ``mode`` into ``args.plot`` (a no-op when unset).

    matplotlib is an optional dep, so a missing install degrades with the same clean
    SystemExit message style as ``_cmd_plot``.
    """
    if getattr(args, "plot", None) is None:
        return
    try:
        from examples import plot_adc as pad
    except ImportError as exc:                          # matplotlib is an optional dep
        raise SystemExit(f"plotting needs matplotlib ({exc}); pip install matplotlib")
    func = getattr(pad, _ADC_PLOT_FUNCS[mode])
    path = func(result, spec.adc, out_dir=args.plot) if mode == "vin" \
        else func(result, out_dir=args.plot)
    if not args.quiet:
        print(f"wrote {path}")
    return path


def _cmd_adc(args):
    from .sar import run_sar_conversion, run_sar_signal, run_sar_sweep
    # ── run-mode mutual exclusion ──
    # Exactly one of the five run modes may be requested; --vin's default is None so
    # an explicit `--vin 0.5` counts as choosing the single-conversion mode (bare
    # `adc circuit.json` still falls back to a 0.5 V conversion below).
    given = [flag for flag, value in (("--vin", args.vin), ("--sweep", args.sweep),
                                      ("--sine", args.sine), ("--mc", args.mc),
                                      ("--explore", args.explore))
             if value is not None]
    if len(given) > 1:
        raise SystemExit(f"choose only one run mode: {' and '.join(given)} "
                         "are mutually exclusive")
    # ── ADC design-space exploration ──
    if args.explore is not None:
        return _cmd_adc_explore(args)
    spec = _load_spec(args.circuit)
    if spec.adc is None:
        raise SystemExit("circuit JSON has no 'adc' workflow block")
    # ── mismatch Monte-Carlo ──
    if args.mc is not None:
        from .sar_mc import sar_mismatch_mc
        if args.mc < 1:
            raise SystemExit("--mc requires at least one trial")
        result = sar_mismatch_mc(spec, n=args.mc, seed=args.seed, corner=args.corner,
                                 workers=args.workers)
        summary = result["summary"]
        if not args.quiet:
            print(
                f"SAR mismatch MC: n={summary['n']}  yield={summary['yield'] * 100:.1f}%  "
                f"monotonic={summary['monotonic_rate'] * 100:.0f}%  "
                f"max|DNL| worst={summary['max_abs_dnl']['worst']:.3f} LSB  "
                f"max|INL| worst={summary['max_abs_inl']['worst']:.3f} LSB")
        _adc_plot(args, "mc", result, spec)
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
            with open(args.output, "w") as fh:
                json.dump(_jsonable(result), fh, indent=2, default=str)
            if not args.quiet:
                print(f"wrote {args.output}")
        return result
    if args.sine is not None:
        if args.sine < 8:
            raise SystemExit("--sine requires at least 8 samples")
        vref = float(spec.adc["vref"])
        offset = 0.5 * vref if args.offset is None else args.offset
        amplitude = 0.45 * vref if args.amplitude is None else args.amplitude
        phase = 2.0 * np.pi * args.tone_bin * np.arange(args.sine) / args.sine
        vin = offset + amplitude * np.sin(phase)
        if np.min(vin) < 0.0 or np.max(vin) > vref:
            raise SystemExit("sine input leaves the ADC range [0, vref]")
        result = run_sar_signal(
            spec, vin, args.sample_rate, corner=args.corner,
            fundamental_bin=args.tone_bin, workers=args.workers)
        metrics = result["metrics"]
        if not args.quiet:
            print(
                f"SAR sine: {args.sine} samples  SNDR={metrics['sndr_db']:.2f} dB  "
                f"SFDR={metrics['sfdr_db']:.2f} dB  ENOB={metrics['enob']:.2f}  "
                f"power={result['average_power_w'] * 1e6:.2f} uW")
    elif args.sweep is None:
        vin = 0.5 if args.vin is None else args.vin     # bare `adc circuit.json` default
        result = run_sar_conversion(spec, vin, corner=args.corner)
        if not args.quiet:
            bits = "".join(str(int(v)) for v in result["bits"])
            print(f"SAR: Vin={result['vin']:.6g} V  code={result['code']}  bits={bits}")
    else:
        if args.sweep < (1 << int(spec.adc["n_bits"])):
            raise SystemExit("--sweep must contain at least 2**n_bits samples")
        vref = float(spec.adc["vref"])
        vin = (np.arange(args.sweep) + 0.5) * vref / args.sweep
        result = run_sar_sweep(spec, vin, corner=args.corner, workers=args.workers)
        metrics = result["metrics"]
        if not args.quiet:
            print(
                f"SAR sweep: {args.sweep} conversions  "
                f"max|DNL|={metrics['max_abs_dnl']:.3f} LSB  "
                f"max|INL|={metrics['max_abs_inl']:.3f} LSB  "
                f"missing={len(metrics['missing_codes'])}")
    mode = "sine" if args.sine is not None else ("vin" if args.sweep is None else "sweep")
    _adc_plot(args, mode, result, spec)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(_jsonable(result), fh, indent=2, default=str)
        if not args.quiet:
            print(f"wrote {args.output}")
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def _is_subcommand(arg):
    """Check if an argument string is a known subcommand name."""
    return arg in _SUBCOMMANDS


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # ── determine whether a subcommand is present ──
    # Walk argv looking for the first positional (non-flag) argument.
    # If it's a subcommand, route to the subparser chain.
    # Otherwise, fall back to the legacy "run" path for backward compatibility.
    subcmd = None
    for a in argv:
        if not a.startswith("-") and _is_subcommand(a):
            subcmd = a
            break

    # ── build the full parser ──
    ap = argparse.ArgumentParser(
        description="Local circuit solvers CLI — analyses, exploration, corners, mismatch, chopper.",
    )
    sub = ap.add_subparsers(dest="command", help="Subcommand")

    _add_run_parser(sub)
    _add_explore_parser(sub)
    _add_corners_parser(sub)
    _add_mc_parser(sub)
    _add_chopper_parser(sub)
    _add_adc_parser(sub)
    _add_plot_parser(sub)
    _add_dataset_parser(sub)
    _add_serve_parser(sub)

    # If --help/-h is the only argument, show the full subcommand listing
    if set(argv) <= {"--help", "-h"}:
        ap.print_help()
        return None

    if subcmd is not None:
        # Explicit subcommand — parse normally
        args = ap.parse_args(argv)
    else:
        # Backward-compatible path: no subcommand given.
        # Check for --explore flag and map accordingly.
        if "--explore" in argv or any(a.startswith("--explore") for a in argv):
            # Remove --explore flag and treat as "explore" subcommand
            clean = [a for a in argv if a != "--explore"]
            # Prepend the subcommand name so argparse routes correctly
            clean.insert(0, "explore")
            args = ap.parse_args(clean)
        else:
            # Default: treat as "run" subcommand
            args = ap.parse_args(["run"] + argv)

    # ── dispatch ──
    # NOTE: the handlers return their result payloads (dicts/…) for programmatic
    # callers, but ``main`` must return an *exit status* — the setuptools console
    # script wraps this in ``sys.exit(main())``, and ``sys.exit(<truthy dict>)``
    # would print the dict to stderr and exit 1. So swallow the payload here and
    # return None (→ exit 0) on success; errors still raise SystemExit as before.
    cmd = args.command
    handlers = {
        "run": _cmd_run,
        "explore": _cmd_explore,
        "corners": _cmd_corners,
        "mc": _cmd_mc,
        "chopper": _cmd_chopper,
        "adc": _cmd_adc,
        "plot": _cmd_plot,
        "dataset": _cmd_dataset,
        "serve": _cmd_serve,
    }
    handler = handlers.get(cmd)
    if handler is None:
        ap.print_help()
        return None
    handler(args)
    return None


if __name__ == "__main__":
    main()
