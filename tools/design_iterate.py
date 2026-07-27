#!/usr/bin/env python3
"""Drive a PVT signoff campaign from a parameterised design generator.

``circuit-opt signoff`` evaluates ONE design point: to try a different device
size or compensation value you must edit the generator, regenerate the decks,
and re-run.  Analog sizing needs the opposite loop -- dozens of variants per
hour, none of them worth committing.  This tool supplies that loop for any
design whose netlists come from a generator module exposing

* module-level constants (``CC = 900e-15``, ``SZ = {...}``, ``MULT = {...}``), and
* ``all_testbenches() -> {filename: deck_dict}``

by overriding constants in memory, writing the decks to a temporary directory,
and running the campaign against them.  The repository is never touched.

Three views, matching the three questions that actually come up while sizing:

``run``    Which specs fail, and AT WHICH CORNERS?  The stock campaign reports
           one global ``worst_case``; that is the wrong summary for a design
           decision, because the variant with the better worst margin routinely
           has FEWER passing points.  ``run`` prints per-spec pass counts plus
           the corner list behind every failing constraint, which is what says
           "common mode fails cold, settling fails hot".

``map``    How is one measurement DISTRIBUTED over the grid?  Prints range and
           spread, the mean grouped along each PVT axis (a temperature-dominated
           spread means a device offset; a supply-dominated one means a
           reference mismatch), and the trim that minimises the worst deviation
           when the measurement is a signed error that a single knob can shift.

``trace``  What does the trajectory look like at one point?  Dumps selected node
           voltages at chosen instants for one case.

Examples::

    tools/design_iterate.py run --generator tsmc28_mdac_ota_gen \\
        --manifest examples/tsmc28hpcp_mdac_ota_signoff.json CC=850e-15 RZ=350

    tools/design_iterate.py map --generator tsmc28_mdac_ota_gen \\
        --manifest examples/tsmc28hpcp_mdac_ota_signoff.json \\
        --case residue_plus_fs16 --measurement checkpoint_error --signed \\
        SZ:M11=139.286/0.30

    tools/design_iterate.py trace --generator tsmc28_mdac_ota_gen \\
        --manifest examples/tsmc28hpcp_mdac_ota_signoff.json \\
        --case residue_plus_fs16 --point ff/27/0.85 --nodes OUTP,OUTN,CTRL2
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "examples", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ── parameter overrides ─────────────────────────────────────────────────────────
def apply_overrides(generator, pairs):
    """Apply ``NAME=VALUE`` / ``SZ:DEV=W/L`` / ``MULT:DEV=M`` to a generator module.

    Every form asserts the target exists: a typo must fail loudly rather than
    silently evaluate the unmodified design (a whole afternoon was once spent
    reading results from a variant whose override never landed)."""
    applied = []
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"override {pair!r} is not NAME=VALUE")
        if name.startswith("SZ:"):
            device = name[3:]
            table = getattr(generator, "SZ", None)
            if not isinstance(table, dict) or device not in table:
                raise SystemExit(f"generator has no SZ entry {device!r}")
            width, _, length = value.partition("/")
            table[device] = (float(width), float(length))
            applied.append(f"SZ[{device}]=({width},{length})")
        elif name.startswith("MULT:"):
            device = name[5:]
            table = getattr(generator, "MULT", None)
            if not isinstance(table, dict):
                raise SystemExit("generator has no MULT table")
            table[device] = int(value)
            applied.append(f"MULT[{device}]={value}")
        else:
            if not hasattr(generator, name):
                raise SystemExit(f"generator has no constant {name!r}")
            setattr(generator, name, float(value))
            applied.append(f"{name}={value}")
    return applied


def staged_manifest(generator, manifest_path):
    """Write the generator's decks + the manifest into a fresh temp directory."""
    staging = Path(tempfile.mkdtemp(prefix="design-iterate-"))
    for filename, deck in generator.all_testbenches().items():
        (staging / filename).write_text(json.dumps(deck), encoding="utf-8")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    staged = staging / "signoff.json"
    staged.write_text(json.dumps(manifest), encoding="utf-8")
    return staging, staged


def point_label(pvt):
    return f"{pvt['corner']}/{int(pvt['temperature_c'])}/{pvt['supply_v']}"


def measurement(case_result, name):
    signoff = case_result.get("signoff") or {}
    return signoff.get("measurements", {}).get(name, {}).get("value")


# ── run: scoreboard ─────────────────────────────────────────────────────────────
def cmd_run(args, generator, manifest_path, applied):
    from circuitopt.signoff_campaign import run_signoff_campaign

    staging, staged = staged_manifest(generator, manifest_path)
    try:
        started = time.time()
        result = run_signoff_campaign(str(staged), workers=args.workers)
        elapsed = time.time() - started
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    points = result["summary"]["points"]
    print(f"[{args.tag}] {' '.join(applied) or 'baseline'}  ({elapsed:.0f}s)")
    print(f"  points: {points['pass']}/{points['total']} pass"
          + (f", {points['invalid']} invalid" if points['invalid'] else ""))

    cases = result["summary"]["cases"]
    width = max(len(name) for name in cases)
    for name, counts in cases.items():
        flag = "" if counts["pass"] == points["total"] else "   <-"
        print(f"    {name:<{width}}  {counts['pass']:>3}/{points['total']}{flag}")

    # Per-constraint failure corners: the view that drives sizing decisions.
    failures = {}
    for point in result["points"]:
        label = point_label(point["pvt"])
        for case_name, case in point["cases"].items():
            if case["passed"]:
                continue
            signoff = case.get("signoff") or {}
            for constraint, detail in (signoff.get("constraints") or {}).items():
                if isinstance(detail, dict) and detail.get("passed") is False:
                    failures.setdefault(f"{case_name}:{constraint}", []).append(label)
            if case["status"] == "invalid":
                failures.setdefault(f"{case_name}:INVALID", []).append(label)
    if failures:
        print("  failing constraints (corner list):")
        for key in sorted(failures, key=lambda k: (-len(failures[k]), k)):
            corners = failures[key]
            shown = " ".join(corners[:args.max_corners])
            more = f" ... (+{len(corners) - args.max_corners})" if len(
                corners) > args.max_corners else ""
            print(f"    {key} x{len(corners)}: {shown}{more}")

    worst = result.get("worst_case") or {}
    if worst:
        margin = worst.get("normalized_margin")
        print(f"  global worst: {worst.get('case')}/{worst.get('measurement')} "
              f"margin={margin if margin is None else f'{margin:.3f}'} "
              f"@{worst.get('corner')}/{worst.get('temperature_c')}/{worst.get('supply_v')}")
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  wrote {args.output}")
    return 0 if result["passed"] else 1


# ── map: distribution + root-cause grouping + trim ──────────────────────────────
def cmd_map(args, generator, manifest_path, applied):
    from circuitopt.signoff_campaign import run_signoff_campaign

    staging, staged = staged_manifest(generator, manifest_path)
    try:
        result = run_signoff_campaign(str(staged), workers=args.workers)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    scale, unit = args.scale, args.unit
    samples = []
    for point in result["points"]:
        case = point["cases"].get(args.case)
        if case is None:
            raise SystemExit(f"case {args.case!r} is not in the manifest")
        value = measurement(case, args.measurement)
        if isinstance(value, (int, float)):
            samples.append((point["pvt"], float(value) * scale))
    if not samples:
        raise SystemExit(
            f"no finite {args.measurement!r} values in case {args.case!r}")

    values = [v for _, v in samples]
    lo, hi = min(values), max(values)
    print(f"[{args.tag}] {' '.join(applied) or 'baseline'}")
    print(f"  {args.case}/{args.measurement}: n={len(values)}  "
          f"min={lo:+.4g}{unit}  max={hi:+.4g}{unit}  span={hi - lo:.4g}{unit}  "
          f"mean={sum(values) / len(values):+.4g}{unit}")
    worst = max(values, key=abs)
    worst_at = point_label(next(p for p, v in samples if v == worst))
    verdict = ""
    if args.limit is not None:
        verdict = ("  PASS" if abs(worst) < args.limit else "  FAIL")
        verdict += f" (limit {args.limit:g}{unit})"
    print(f"  worst |value| = {abs(worst):.4g}{unit} @{worst_at}{verdict}")

    # Grouping along each PVT axis is the root-cause discriminator: a spread
    # that lives on the temperature axis is a device-level offset, one that
    # lives on the supply axis is a reference/level-shift mismatch.
    for axis, key in (("corner", "corner"), ("temp", "temperature_c"),
                      ("supply", "supply_v")):
        groups = {}
        for pvt, value in samples:
            groups.setdefault(pvt[key], []).append(value)
        ordered = sorted(groups.items(), key=lambda kv: str(kv[0]))
        means = {k: sum(v) / len(v) for k, v in ordered}
        spread = max(means.values()) - min(means.values())
        body = "  ".join(f"{k}:{m:+.4g}" for k, m in means.items())
        print(f"  by {axis:<7} {body}   (spread {spread:.4g}{unit})")

    if args.signed:
        # A signed error that one knob shifts uniformly: the trim that minimises
        # the worst |error| puts the extremes equidistant from zero.
        shift = -(lo + hi) / 2.0
        residual = (hi - lo) / 2.0
        verdict = ""
        if args.limit is not None:
            verdict = "  PASS" if residual < args.limit else "  FAIL"
            verdict += f" (limit {args.limit:g}{unit})"
        print(f"  optimal common trim: {shift:+.4g}{unit} "
              f"-> worst |value| {residual:.4g}{unit}{verdict}")
    if args.per_point:
        print("  per point:")
        for pvt, value in sorted(samples, key=lambda s: -abs(s[1])):
            print(f"    {point_label(pvt):>16}  {value:+.4g}{unit}")
    return 0


# ── trace: single-point trajectory anatomy ──────────────────────────────────────
def cmd_trace(args, generator, manifest_path, applied):
    import numpy as np
    from circuitopt.analysis_dispatch import run_analysis_suite
    from circuitopt.circuit_loader import circuit_from_dict
    from circuitopt.compact_models.bsim4 import isolated_native_device_cache
    from circuitopt.signoff_campaign import (
        _load_case_bases, load_campaign_json, prepare_case_dict,
    )

    staging, staged = staged_manifest(generator, manifest_path)
    try:
        config, staged_path = load_campaign_json(staged)
        cases = _load_case_bases(config, staged_path)
        try:
            case = next(c for c in cases if c["name"] == args.case)
        except StopIteration:
            raise SystemExit(f"case {args.case!r} is not in the manifest") from None
        corner, temperature, supply = parse_point(args.point)
        deck = prepare_case_dict(
            case["base"], case["overrides"], corner=corner,
            temperature_c=temperature, supply_v=supply,
            nominal_supply_v=config["pvt"]["nominal_supply_v"],
            supply_bias_key=config["pvt"]["supply_bias_key"])
        spec = circuit_from_dict(deck)
        with isolated_native_device_cache():
            results = run_analysis_suite(spec)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    transient = results.get("transient")
    if transient is None:
        raise SystemExit(f"case {args.case!r} has no transient analysis")
    t = np.asarray(transient["t"], float)
    nodes = transient["nodes"]
    requested = [n.strip() for n in args.nodes.split(",")] if args.nodes else []
    missing = [n for n in requested if n not in nodes]
    if missing:
        raise SystemExit(f"unsolved node(s): {', '.join(missing)}")
    columns = requested or sorted(nodes)
    instants = ([float(x) for x in args.instants.split(",")] if args.instants
                else list(np.linspace(t[0], t[-1], 9)))

    print(f"[{args.tag}] {' '.join(applied) or 'baseline'}")
    print(f"  {args.case} @ {args.point}")
    header = ["t[ns]"] + columns
    print("   " + "  ".join(f"{h:>10s}" for h in header))
    for instant in instants:
        index = int(np.argmin(np.abs(t - instant)))
        row = [f"{t[index] * 1e9:10.3f}"]
        row += [f"{float(np.asarray(nodes[c])[index]):10.5f}" for c in columns]
        print("   " + "  ".join(row))
    return 0


def parse_point(text):
    parts = text.split("/")
    if len(parts) != 3:
        raise SystemExit(f"--point must be corner/temp/supply, got {text!r}")
    return parts[0], float(parts[1]), float(parts[2])


# ── CLI ─────────────────────────────────────────────────────────────────────────
def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub):
        sub.add_argument("--generator", required=True,
                         help="importable generator module (examples/ is on the path)")
        sub.add_argument("--manifest", required=True, help="signoff campaign JSON")
        sub.add_argument("--workers", type=int, default=8)
        sub.add_argument("--tag", default="trial", help="label for the printout")
        sub.add_argument("overrides", nargs="*",
                         help="NAME=VALUE | SZ:DEV=W/L | MULT:DEV=M")
        return sub

    run = common(subparsers.add_parser(
        "run", help="scoreboard: per-spec pass counts + failing-corner lists"))
    run.add_argument("--max-corners", type=int, default=6,
                     help="corners listed per failing constraint (default: 6)")
    run.add_argument("-o", "--output", help="also write the full campaign JSON")

    mapper = common(subparsers.add_parser(
        "map", help="distribution of one measurement over the PVT grid"))
    mapper.add_argument("--case", required=True)
    mapper.add_argument("--measurement", required=True)
    mapper.add_argument("--scale", type=float, default=1.0,
                        help="multiply values (e.g. 1e3 for mV)")
    mapper.add_argument("--unit", default="", help="unit suffix for the printout")
    mapper.add_argument("--signed", action="store_true",
                        help="report the optimal common trim (signed errors)")
    mapper.add_argument("--limit", type=float,
                        help="spec limit, in the scaled unit, for a PASS/FAIL verdict")
    mapper.add_argument("--per-point", action="store_true")

    trace = common(subparsers.add_parser(
        "trace", help="node trajectories for one case at one PVT point"))
    trace.add_argument("--case", required=True)
    trace.add_argument("--point", required=True, help="corner/temp/supply")
    trace.add_argument("--nodes", help="comma-separated solved nodes")
    trace.add_argument("--instants", help="comma-separated times in seconds")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    generator = importlib.import_module(args.generator)
    if not hasattr(generator, "all_testbenches"):
        raise SystemExit(
            f"{args.generator} exposes no all_testbenches(); it cannot drive a campaign")
    applied = apply_overrides(generator, args.overrides)
    handler = {"run": cmd_run, "map": cmd_map, "trace": cmd_trace}[args.command]
    return handler(args, generator, args.manifest, applied)


if __name__ == "__main__":
    raise SystemExit(main())
