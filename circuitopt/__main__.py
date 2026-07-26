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
from collections.abc import Mapping, Sequence

import numpy as np

# Engine resolution (and rejection of the retired numba switches) happens in
# circuitopt/__init__.py via circuitopt._engine.apply_engine_env, which runs
# before this module under `python -m circuitopt`. Nothing engine-related needs
# to happen here anymore — rust is the only engine (v2.0.0).

from .analysis_dispatch import run_analysis_suite
from .chopper import (chopper_analysis, pmos_chopper_analysis,
                      pmos_chopper_lptv_analysis, pmos_chopper_pac,
                      pmos_chopper_pnoise, pmos_chopper_pss,
                      pmos_chopper_transient)
from .circuit_loader import load_circuit_json
from .corners import corner_table, mismatch_mc_from_dict, silicon_corner_names
from .device_factory import is_silicon_model_types
from .dataset import add_cli_args as dataset_add_cli_args
from .dataset import run_cli as dataset_run_cli
from .explore import add_cli_args as explore_add_cli_args
from .explore import run_cli as explore_run_cli
from .noise_solver import band_rms
from .run_contract import evaluate_signoff
# The service subpackage's CLI glue is fastapi-free (fastapi/uvicorn are imported
# lazily inside serve_run_cli), so importing it here never pulls the serve extra.
from .service import add_cli_args as serve_add_cli_args
from .service import run_cli as serve_run_cli
from .mcp import add_cli_args as mcp_add_cli_args
from .mcp import run_cli as mcp_run_cli

_ANALYSIS_NAMES = ["ac", "noise", "transient", "pss", "pac", "pnoise"]
_SUBCOMMANDS = [
    "run", "signoff", "corners", "mc", "chopper", "adc",
    "explore", "plot", "dataset", "serve", "mcp",
]
_CHOPPER_LEVELS = ["ideal", "pmos", "lptv", "pss", "pac", "pnoise", "transient"]


# ── shared helpers ───────────────────────────────────────────────────────────

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
    """Add ``--engine`` to a subcommand that also carries the retired ``--no-numba``.

    The flag only appears in ``--help`` and is validated by the argv pre-scan
    (``circuitopt._engine.apply_engine_env`` via ``circuitopt/__init__.py``),
    which runs before this parser is built. As of v2.0.0 ``rust`` is the only
    accepted value; ``--engine python``/``numba`` (and ``--no-numba`` /
    ``CIRCUIT_USE_NUMBA``) are hard errors that point at the CHANGELOG.
    """
    parser.add_argument("--engine", choices=("rust",), default=None,
                        help="Compute engine (only 'rust'; the 'python' and 'numba' "
                             "engines were removed in v2.0.0). See circuitopt._engine.")


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
    p.add_argument("--no-numba", action="store_true",
                   help="Removed in v2.0.0 (errors): the numba engine no longer exists")
    _add_engine_arg(p)
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p


def _cmd_run(args):

    selected = None
    if args.analysis:
        selected = [s.strip().lower() for s in args.analysis.split(",")]
        unknown = set(selected) - set(_ANALYSIS_NAMES)
        if unknown:
            raise SystemExit(f"unknown analysis: {', '.join(sorted(unknown))}")

    if not args.quiet:
        what = ",".join(selected) if selected else "all configured"
        print(f"Running {what} analyses for {args.circuit}")

    spec = _load_spec(args.circuit)
    results = run_analysis_suite(spec, selected=selected, corner=args.corner)

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

    signoff = evaluate_signoff(spec, results)
    payload = {
        "status": "valid",
        "results": results,
        "signoff": signoff,
    }
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(_jsonable(payload), f, indent=2, default=str)
        print(f"wrote {args.output}")

    return payload


# ── subcommand: signoff campaign ─────────────────────────────────────────────

def _add_signoff_parser(subparsers):
    p = subparsers.add_parser(
        "signoff",
        help="Run a multi-testbench process/voltage/temperature signoff campaign",
    )
    p.add_argument("campaign", help="Path to signoff campaign JSON")
    p.add_argument("--workers", type=int, default=1,
                   help="PVT points evaluated concurrently (default: 1)")
    _add_output_arg(p)
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p


def _cmd_signoff(args):
    from .signoff_campaign import run_signoff_campaign

    if not os.path.exists(args.campaign):
        raise SystemExit(f"file not found: {args.campaign}")

    def progress(done, total):
        if not args.quiet:
            print(f"\r  evaluating PVT point {done}/{total}", end="", flush=True)

    if not args.quiet:
        print(f"Signoff campaign {args.campaign}  (workers={args.workers})")
    result = run_signoff_campaign(
        args.campaign,
        workers=args.workers,
        progress=progress,
    )
    if not args.quiet:
        print()
        points = result["summary"]["points"]
        print(
            f"  status={result['status']}  total={points['total']}  "
            f"pass={points['pass']}  fail={points['fail']}  "
            f"invalid={points['invalid']}"
        )
        worst = result["worst_case"]
        if worst is not None:
            print(
                f"  worst: case={worst['case']} "
                f"pvt={worst['corner']}/{worst['temperature_c']:g}C/"
                f"{worst['supply_v']:g}V "
                f"measurement={worst['measurement'] or 'invalid'}"
            )
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(result), handle, indent=2, default=str)
        if not args.quiet:
            print(f"wrote {args.output}")
    return result


# ── subcommand: explore ──────────────────────────────────────────────────────

def _add_explore_parser(subparsers):
    p = subparsers.add_parser("explore", help="Run design-space exploration")
    # Feature args (positional + sampling/corner/output/quiet) come from the single
    # source in circuitopt.explore so this subcommand can't drift from `python -m circuitopt.explore`.
    explore_add_cli_args(p)
    # Subcommand-level mechanism — not a feature arg, so it stays here.
    p.add_argument("--no-numba", action="store_true",
                   help="Removed in v2.0.0 (errors): the numba engine no longer exists")
    _add_engine_arg(p)
    return p


def _cmd_explore(args):
    return explore_run_cli(args)


# ── subcommand: dataset ──────────────────────────────────────────────────────

def _add_dataset_parser(subparsers):
    p = subparsers.add_parser(
        "dataset", help="Build a labeled surrogate dataset from an 'explore' config")
    # Feature args come from the single source in circuitopt.dataset so this subcommand
    # can't drift from `python -m circuitopt.dataset`.
    dataset_add_cli_args(p)
    # Subcommand-level mechanism — not a feature arg, so it stays here.
    p.add_argument("--no-numba", action="store_true",
                   help="Removed in v2.0.0 (errors): the numba engine no longer exists")
    _add_engine_arg(p)
    return p


def _cmd_dataset(args):
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


# ── subcommand: mcp ──────────────────────────────────────────────────────────

def _add_mcp_parser(subparsers):
    p = subparsers.add_parser(
        "mcp",
        help="Start the local Model Context Protocol server (needs the 'mcp' extra)",
    )
    mcp_add_cli_args(p)
    return p


def _cmd_mcp(args):
    return mcp_run_cli(args)


# ── subcommand: corners ──────────────────────────────────────────────────────

def _csv_floats(text):
    """Parse a comma-separated float list (a CLI PVT axis); empty -> None.

    Use the ``--flag=-40,27,125`` form for negative values so argparse does not read
    the leading ``-`` as another option."""
    items = [s.strip() for s in str(text).split(",") if s.strip()]
    if not items:
        return None
    return tuple(float(s) for s in items)


def _add_corners_parser(subparsers):
    p = subparsers.add_parser("corners", help="Run process-corner sweep (typ/slow/fast)")
    p.add_argument("circuit", help="Path to circuit JSON file")
    _add_freqs_args(p)
    _add_noise_band_arg(p)
    _add_output_arg(p)
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel corner workers (default: 1)")
    p.add_argument("--temps", type=_csv_floats, default=None,
                   help="Silicon only: comma-separated temperature axis in °C, e.g. "
                        "--temps=-40,27,125 (default: single 27 °C, output unchanged)")
    p.add_argument("--vdd-scale", type=_csv_floats, default=None, dest="vdd_scale",
                   help="Silicon only: comma-separated supply-scale factors applied to "
                        "the whole bias, e.g. --vdd-scale=0.9,1.0,1.1 (default: 1.0)")
    p.add_argument("--no-numba", action="store_true",
                   help="Removed in v2.0.0 (errors): the numba engine no longer exists")
    _add_engine_arg(p)
    p.add_argument("--quiet", action="store_true", help="Suppress per-corner output")
    return p


def _corner_row_line(corner_name, metrics, indent):
    """One formatted corner row (or a (failed) marker) at the given indent."""
    pad = " " * indent
    if metrics is None:
        return f"{pad}{corner_name:>7s}:  (failed)"
    return (f"{pad}{corner_name:>7s}:  "
            f"gain={metrics['gain_peak_dB']:.2f} dB  "
            f"BW={metrics['bw_Hz']:.0f} Hz  "
            f"IRN={metrics['irn_uV']:.2f} µVrms")


def _corner_grid_rows(table, temps, vdd_scale):
    """Yield ``(corner, temp_c|None, vdd_scale|None, metrics)`` over a nested table."""
    for corner_name, node in table.items():
        if temps is not None and vdd_scale is not None:
            for temp_c, vnode in node.items():
                for scale, m in vnode.items():
                    yield corner_name, temp_c, scale, m
        elif temps is not None:
            for temp_c, m in node.items():
                yield corner_name, temp_c, None, m
        else:                                   # vdd_scale only
            for scale, m in node.items():
                yield corner_name, None, scale, m


def _print_corner_grid(table, temps, vdd_scale):
    """Print a PVT corner grid grouped by ``(temp, vdd)`` slice."""
    slices = {}
    for corner_name, temp_c, scale, m in _corner_grid_rows(table, temps, vdd_scale):
        slices.setdefault((temp_c, scale), []).append((corner_name, m))
    for (temp_c, scale), rows in slices.items():
        parts = ([f"T={temp_c:g} °C"] if temp_c is not None else []) + \
                ([f"Vdd×{scale:g}"] if scale is not None else [])
        print(f"  [{'  '.join(parts)}]")
        for corner_name, m in rows:
            print(_corner_row_line(corner_name, m, indent=6))


def _write_corner_csv(output, table, temps, vdd_scale):
    """Write the corner table as CSV. The (corner, gain, bw, irn) columns are the
    frozen default; ``temp_c`` / ``vdd_scale`` columns are added only for the axes
    that are active, so a default (no-axis) sweep is byte-for-byte unchanged."""
    axis_cols = (["temp_c"] if temps is not None else []) + \
                (["vdd_scale"] if vdd_scale is not None else [])
    lines = [",".join(["corner", *axis_cols, "gain_peak_dB", "bw_Hz", "irn_uV"])]
    if temps is None and vdd_scale is None:
        rows = ((c, None, None, m) for c, m in table.items())
    else:
        rows = _corner_grid_rows(table, temps, vdd_scale)
    for corner_name, temp_c, scale, m in rows:
        if m is None:
            continue
        axis_vals = ([f"{temp_c:g}"] if temps is not None else []) + \
                    ([f"{scale:g}"] if vdd_scale is not None else [])
        lines.append(",".join([corner_name, *axis_vals,
                               f"{m['gain_peak_dB']:.4f}", f"{m['bw_Hz']:.1f}",
                               f"{m['irn_uV']:.3f}"]))
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with open(output, "w") as f:
        f.write("\n".join(lines) + "\n")


def _cmd_corners(args):

    spec = _load_spec(args.circuit)
    freqs = _freqs_from_args(args)
    lo, hi = args.noise_band
    temps, vdd_scale = args.temps, args.vdd_scale

    # Carry the per-device model binding so a silicon circuit keeps its BSIM4 cards
    # (and routes through the compiled campaign) instead of silently reverting to the
    # default OTFT PDK; silicon then sweeps card corners, OTFT keeps typical/slow/fast.
    binding = spec.binding()
    silicon = is_silicon_model_types(binding.model_types)
    corners = (silicon_corner_names(binding.model_types) if silicon
               else ("typical", "slow", "fast"))

    # The PVT axes are silicon-only (corner_table enforces the same guard); fail early
    # and cleanly here so an OTFT circuit gets a message, not a partial table + trace.
    if (temps is not None or vdd_scale is not None) and not silicon:
        raise SystemExit(
            "corners: --temps/--vdd-scale require an all-silicon circuit; this "
            "OTFT/default-PDK circuit has no temperature or supply-scale axis")

    if not args.quiet:
        print(f"Corner sweep for {args.circuit}")
        print(f"  freqs: {args.freqs_start:.2g}–{args.freqs_stop:.2g} Hz ({args.freqs_num} pts)")
        print(f"  band:  {lo}–{hi} Hz")
        print(f"  corners: {', '.join(corners)}")
        print(f"  workers: {args.workers}")
        if temps is not None:
            print(f"  temps: {', '.join(f'{t:g}' for t in temps)} °C")
        if vdd_scale is not None:
            print(f"  vdd_scale: {', '.join(f'{v:g}' for v in vdd_scale)}")

    table = corner_table(spec.sizes, spec.bias, nf=spec.nf, topo=spec.topology,
                         corners=corners, freqs=freqs, workers=args.workers,
                         binding=binding, temps=temps, vdd_scale=vdd_scale)

    if temps is None and vdd_scale is None:
        # Frozen default output: the flat per-corner rows, byte-for-byte unchanged.
        for corner_name, metrics in table.items():
            print(_corner_row_line(corner_name, metrics, indent=2))
    else:
        _print_corner_grid(table, temps, vdd_scale)

    if args.output:
        _write_corner_csv(args.output, table, temps, vdd_scale)
        print(f"wrote {args.output}")

    return table


# ── subcommand: mc (mismatch Monte Carlo) ────────────────────────────────────

def _add_mc_parser(subparsers):
    p = subparsers.add_parser("mc", help="Run per-device mismatch Monte Carlo")
    p.add_argument("circuit", help="Path to circuit JSON file")
    p.add_argument("-n", "--n", type=int, default=200,
                   help="Number of MC samples (default: 200)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel MC workers (default: 1)")
    p.add_argument("--corner", choices=("typical", "slow", "fast"), default="typical",
                   help="Base process corner (default: typical)")
    _add_freqs_args(p)
    _add_noise_band_arg(p)
    _add_output_arg(p)
    p.add_argument("--no-numba", action="store_true",
                   help="Removed in v2.0.0 (errors): the numba engine no longer exists")
    _add_engine_arg(p)
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p


def _cmd_mc(args):

    if not os.path.exists(args.circuit):
        raise SystemExit(f"file not found: {args.circuit}")
    with open(args.circuit, "r", encoding="utf-8") as f:
        data = json.load(f)
    freqs = _freqs_from_args(args)
    lo, hi = args.noise_band

    if not args.quiet:
        print(f"Mismatch MC for {args.circuit}")
        print(f"  n={args.n}  seed={args.seed}  corner={args.corner}  workers={args.workers}")
        print(f"  freqs: {args.freqs_start:.2g}–{args.freqs_stop:.2g} Hz ({args.freqs_num} pts)")
        print(f"  band:  {lo}–{hi} Hz")

    # mismatch_mc_from_dict is the shared CLI/service entry (parses the circuit +
    # calls mismatch_mc), so `circuit-opt mc` and POST /jobs/mc can't drift.
    mc = mismatch_mc_from_dict(data, n=args.n, seed=args.seed, corner=args.corner,
                               freqs=freqs, band=(lo, hi), workers=args.workers)

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
    p.add_argument("--no-numba", action="store_true",
                   help="Removed in v2.0.0 (errors): the numba engine no longer exists")
    _add_engine_arg(p)
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p


def _cmd_chopper(args):

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
    p.add_argument("--no-numba", action="store_true",
                   help="Removed in v2.0.0 (errors): the numba engine no longer exists")
    _add_engine_arg(p)
    p.add_argument("--quiet", action="store_true", help="Suppress the summary line")
    return p


def _cmd_plot(args):
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
    p.add_argument("--transitions", nargs="?", const="carries", default=None,
                   metavar="CODES",
                   help="locate physical code-transition voltages by lockstep "
                        "bisection and report DNL/INL at them. Default target "
                        "set 'carries' = both DNL bins around every binary "
                        "major carry plus the offset transition; or a comma-"
                        "separated list of transition codes. Resolves each "
                        "transition to --tol-lsb (finer than a full ramp's "
                        "+/-0.5 LSB) in O(log 1/tol) conversions instead of "
                        "2**n_bits — the 12-bit final-verification mode")
    p.add_argument("--tol-lsb", type=float, default=0.05,
                   help="transitions: bisection tolerance in LSB (default 0.05)")
    p.add_argument("--bracket-lsb", type=float, default=2.0,
                   help="transitions: initial search bracket around each ideal "
                        "transition in LSB (default 2.0; a bracket that misses "
                        "widens to the full range automatically)")
    p.add_argument("--sweep-points", type=int, default=None, metavar="M",
                   help="mc only: subsample each trial's code-center sweep to M "
                        "points (overrides adc.mismatch.sweep_points; yield then "
                        "gates on code errors instead of transition DNL/INL — "
                        "the screening mode for 12-bit-class resolutions)")
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
                   help="parallel conversions (sweep/sine, and inside each explore "
                        "candidate's sweep) or MC trials; default 1 (serial). "
                        "Bit decisions within a conversion stay serial.")
    p.add_argument("--csv", default=None, help="explore: write candidate rows to this CSV")
    p.add_argument("--jsonl", default=None, help="explore: write candidate rows to this JSONL")
    p.add_argument("--plot", nargs="?", const="results", default=None, metavar="DIR",
                   help="render the figure(s) matching the run mode into DIR "
                        "(default: results/); needs matplotlib")
    p.add_argument("--waveforms", action="store_true",
                   help="record full per-conversion waveforms and keep them in "
                        "--output; without it -o stores codes, decision traces "
                        "and power/metrics only (kilobytes instead of tens of "
                        "megabytes for a 64-code sweep)")
    _add_output_arg(p)
    p.add_argument("--quiet", action="store_true", help="Suppress summary output")
    return p


_DROP = object()


def _jsonable(value):
    """A JSON-ready copy of a solver payload.

    Callables and private keys were always excluded. Opaque objects are too:
    a PSS result carries its ``Topology`` for downstream PAC/PNoise, and the
    serializer's ``default=str`` would otherwise write a
    ``<... object at 0x...>`` memory address into the file — meaningless to a
    reader and different on every run, which also makes the payload
    irreproducible. Complex values are kept; their text form is stable.
    """
    if callable(value):
        return _DROP
    if isinstance(value, Mapping):
        ready = {}
        for key, item in value.items():
            converted_key = _jsonable(key)
            if converted_key is _DROP or isinstance(
                converted_key, (Mapping, list, set, frozenset)
            ):
                continue
            text_key = str(converted_key)
            if text_key.startswith("_"):
                continue
            converted = _jsonable(item)
            if converted is not _DROP:
                ready[text_key] = converted
        return ready
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return _DROP if converted is value else _jsonable(converted)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        ready = []
        for item in value:
            converted = _jsonable(item)
            if converted is not _DROP:
                ready.append(converted)
        return ready
    if isinstance(value, (set, frozenset)):
        ready = []
        for item in value:
            converted = _jsonable(item)
            if converted is not _DROP:
                ready.append(converted)
        return sorted(ready, key=lambda item: (type(item).__name__, repr(item)))
    if value is None or isinstance(value, (str, int, float, bool, complex)):
        return value
    return _DROP


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

# The three waveform-bearing fields of a finalized conversion. Everything else
# (codes, bits, decision traces, power scalars) is kilobyte-sized.
_ADC_WAVEFORM_KEYS = ("t", "input_waveforms", "transient")


def _slim_adc_result(result):
    """``result`` without per-conversion waveform payloads.

    A 64-code sweep's waveforms serialize to an 82 MB JSON and take longer to
    write than the codes take to solve, so ``-o`` keeps the codes, bits,
    decision traces and power/metric scalars unless ``--waveforms`` asks for
    the full payload. Works on a sweep/sine result (slims each entry of
    ``conversions``) and on a bare single-conversion result alike.
    """
    def slim(conversion):
        return {key: value for key, value in conversion.items()
                if key not in _ADC_WAVEFORM_KEYS}

    if "conversions" in result:
        return {**result,
                "conversions": [slim(item) for item in result["conversions"]]}
    return slim(result)


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
                                      ("--transitions", args.transitions),
                                      ("--explore", args.explore))
             if value is not None]
    if len(given) > 1:
        raise SystemExit(f"choose only one run mode: {' and '.join(given)} "
                         "are mutually exclusive")
    # ── --waveforms contract ──
    # Its only observable effect is the --output payload (plus, for --sweep,
    # actually recording trajectories), so reject the silent no-op forms
    # instead of accepting them: without -o nothing would change, and the
    # mc/explore rows never carried waveforms in the first place.
    if args.waveforms:
        if (args.mc is not None or args.explore is not None
                or args.transitions is not None):
            raise SystemExit(
                "--waveforms applies to --vin/--sweep/--sine conversions only")
        if args.output is None:
            raise SystemExit("--waveforms only changes --output; pass -o too")
    # ── ADC design-space exploration ──
    if args.explore is not None:
        return _cmd_adc_explore(args)
    spec = _load_spec(args.circuit)
    if spec.adc is None:
        raise SystemExit("circuit JSON has no 'adc' workflow block")
    if args.sweep_points is not None and args.mc is None:
        raise SystemExit("--sweep-points applies to --mc only")
    # ── mismatch Monte-Carlo ──
    if args.mc is not None:
        from .sar_mc import sar_mismatch_mc
        if args.mc < 1:
            raise SystemExit("--mc requires at least one trial")
        override = ({"sweep_points": args.sweep_points}
                    if args.sweep_points is not None else None)
        result = sar_mismatch_mc(spec, n=args.mc, seed=args.seed, corner=args.corner,
                                 workers=args.workers, config=override)
        summary = result["summary"]
        if not args.quiet:
            if summary["subsampled"]:
                print(
                    f"SAR mismatch MC (subsampled {summary['sweep_points']} pts): "
                    f"n={summary['n']}  yield={summary['yield'] * 100:.1f}%  "
                    f"monotonic={summary['monotonic_rate'] * 100:.0f}%  "
                    f"max|code err| worst="
                    f"{summary['max_abs_code_err']['worst']:.0f} LSB")
            else:
                print(
                    f"SAR mismatch MC: n={summary['n']}  "
                    f"yield={summary['yield'] * 100:.1f}%  "
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
    # ── transition bisection (final-verification DNL/INL) ──
    if args.transitions is not None:
        from .sar import run_sar_transitions
        if args.plot is not None:
            raise SystemExit("--transitions has no figure; drop --plot")
        if args.transitions == "carries":
            codes = None
        else:
            try:
                codes = [int(x) for x in args.transitions.split(",") if x.strip()]
            except ValueError:
                raise SystemExit(
                    "--transitions takes 'carries' or a comma-separated "
                    "code list") from None
            if not codes:
                raise SystemExit("--transitions code list is empty")
        try:
            result = run_sar_transitions(
                spec, codes, corner=args.corner, tol_lsb=args.tol_lsb,
                bracket_lsb=args.bracket_lsb, workers=args.workers)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        if not args.quiet:
            worst = ""
            if len(result["dnl"]):
                at = result["dnl_codes"][int(np.argmax(np.abs(result["dnl"])))]
                worst = f" @ code {at}"
            print(
                f"SAR transitions: {len(result['targets'])} located to "
                f"{result['tol_lsb']:g} LSB in {result['conversions']} "
                f"conversions ({result['rounds']} rounds)  "
                f"max|DNL|={result['max_abs_dnl']:.3f} LSB{worst}  "
                f"max|INL|={result['max_abs_inl']:.3f} LSB"
                + (f"  unmeasured={result['unmeasured']}"
                   if result["unmeasured"] else ""))
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                        exist_ok=True)
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
        if args.sweep < 2:
            raise SystemExit("--sweep needs at least 2 samples")
        levels = 1 << int(spec.adc["n_bits"])
        vref = float(spec.adc["vref"])
        vin = (np.arange(args.sweep) + 0.5) * vref / args.sweep
        result = run_sar_sweep(
            spec,
            vin,
            corner=args.corner,
            workers=args.workers,
            # Only --waveforms needs trajectories recorded: the sweep figure
            # (plot_sar_static) and the -o default both read codes/metrics
            # alone, and recording+serializing 64 conversions costs more than
            # solving them (~1.0 s record + ~1.7 s for the 82 MB write).
            include_transients=args.waveforms,
        )
        metrics = result["metrics"]
        if metrics["subsampled"]:
            if not args.quiet:
                print(
                    f"SAR sweep (subsampled {args.sweep} of {levels}): "
                    f"max|code err|={metrics['max_abs_code_err']:.0f} LSB  "
                    f"monotonic={'yes' if metrics['monotonic'] else 'NO'}")
        elif not args.quiet:
            print(
                f"SAR sweep: {args.sweep} conversions  "
                f"max|DNL|={metrics['max_abs_dnl']:.3f} LSB  "
                f"max|INL|={metrics['max_abs_inl']:.3f} LSB  "
                f"missing={len(metrics['missing_codes'])}")
    mode = "sine" if args.sine is not None else ("vin" if args.sweep is None else "sweep")
    _adc_plot(args, mode, result, spec)
    if args.output:
        payload = result if args.waveforms else _slim_adc_result(result)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(_jsonable(payload), fh, indent=2, default=str)
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
    _add_signoff_parser(sub)
    _add_explore_parser(sub)
    _add_corners_parser(sub)
    _add_mc_parser(sub)
    _add_chopper_parser(sub)
    _add_adc_parser(sub)
    _add_plot_parser(sub)
    _add_dataset_parser(sub)
    _add_serve_parser(sub)
    _add_mcp_parser(sub)

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
        "signoff": _cmd_signoff,
        "explore": _cmd_explore,
        "corners": _cmd_corners,
        "mc": _cmd_mc,
        "chopper": _cmd_chopper,
        "adc": _cmd_adc,
        "plot": _cmd_plot,
        "dataset": _cmd_dataset,
        "serve": _cmd_serve,
        "mcp": _cmd_mcp,
    }
    handler = handlers.get(cmd)
    if handler is None:
        ap.print_help()
        return None
    handler(args)
    return None


if __name__ == "__main__":
    main()
