"""Design-space exploration / optimization layer.

This is the "optimization" the project is named for: given a circuit, a set of
design variables (sizes / bias) with ranges, feasibility constraints, and one or
more objectives, sample candidates, evaluate each through the validated solvers,
filter by constraints, and Pareto-select the trade-off front. It deliberately
stays a simple, reliable physics-surrogate search (no ML): the calibrated solvers
are fast enough to screen hundreds of candidates locally before sending the
recommended few to Cadence.

Configuration lives in an ``explore`` block inside a full circuit JSON (see
``examples/single_stage.json`` and ``examples/afe_explore.json``):

    "explore": {
      "variables": {
        "in_pair_W": {"min": 40000, "max": 90000, "targets": ["M7.W", "M8.W"]},
        "VCM":       {"min": 28.0,  "max": 33.0}
      },
      "constraints": {"gain_dB": {"min": 20}, "bw_Hz": {"min": 100},
                      "irn_uV": {"max": 44.5}},
      "objectives":  {"area": "min", "power_uW": "min"},
      "band":  [0.05, 100.0],
      "freqs": {"start": -2, "stop": 3, "num": 81}
    }

A variable targets ``"DEV.W"`` / ``"DEV.L"`` (a device size) or a bare bias key
(e.g. ``"VCM"``). ``targets`` lets one variable drive several keys at once, which
is how matched/symmetric pairs (M7=M8, M9=M10, ...) are kept identical — required
for the AFE's symmetric DC continuation to stay on the physical branch.

Metrics: ``gain_dB``, ``bw_Hz`` (from ac_solve), ``irn_uV`` (band-integrated
input-referred noise), ``power_uW`` (top-rail supply current x rail voltage) and
``area`` (sum of per-device ``g_area``).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os

import numpy as np

from .ac_solver import ac_solve
from .device_factory import CircuitBinding
from .noise_solver import band_rms, noise_analysis
from .circuit_loader import circuit_from_dict, models_from_config
from . import diagnostics


METRICS = ("gain_dB", "gain_peak_dB", "bw_Hz", "irn_uV", "power_uW", "area")
NOISE_METRICS = frozenset({"irn_uV"})


# ── configuration ────────────────────────────────────────────────────────
class Variable:
    """One design variable: a uniform range applied to one or more targets.

    target ``"DEV.W"`` / ``"DEV.L"`` -> a device size entry; a bare ``"VCM"`` ->
    a bias key. ``round`` (decimals) snaps samples to a grid; ``int`` rounds to a
    whole number (handy for W/L)."""

    def __init__(self, name, lo, hi, targets=None, round_to=None, is_int=False):
        if hi < lo:
            raise ValueError(f"variable {name!r}: max < min")
        self.name = name
        self.lo = float(lo)
        self.hi = float(hi)
        self.targets = list(targets) if targets else [name]
        self.round_to = round_to
        self.is_int = is_int

    def coerce(self, value):
        if self.round_to is not None:
            value = round(value, self.round_to)
        if self.is_int:
            value = float(int(round(value)))
        return value


class ExploreConfig:
    def __init__(self, variables, constraints, objectives, band, freqs):
        self.variables = variables          # list[Variable]
        self.constraints = constraints      # metric -> {"min"?, "max"?}
        self.objectives = objectives        # metric -> "min" | "max"
        self.band = band                    # (f_lo, f_hi) for IRN
        self.freqs = freqs                  # np.ndarray of analysis frequencies


def parse_explore(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("'explore' must be an object")
    raw_vars = cfg.get("variables")
    if not isinstance(raw_vars, dict) or not raw_vars:
        raise ValueError("'explore.variables' must be a non-empty object")
    variables = []
    for name, spec in raw_vars.items():
        if not isinstance(spec, dict) or "min" not in spec or "max" not in spec:
            raise ValueError(f"variable {name!r} must have numeric 'min' and 'max'")
        variables.append(Variable(
            name=str(name), lo=spec["min"], hi=spec["max"],
            targets=spec.get("targets"),
            round_to=spec.get("round"),
            is_int=bool(spec.get("int", False)),
        ))

    constraints = {}
    for metric, bound in (cfg.get("constraints") or {}).items():
        if metric not in METRICS:
            raise ValueError(f"constraint on unknown metric {metric!r}; known: {METRICS}")
        if not isinstance(bound, dict) or not ({"min", "max"} & bound.keys()):
            raise ValueError(f"constraint {metric!r} needs 'min' and/or 'max'")
        constraints[metric] = {k: float(v) for k, v in bound.items() if k in ("min", "max")}

    objectives = {}
    for metric, sense in (cfg.get("objectives") or {}).items():
        if metric not in METRICS:
            raise ValueError(f"objective on unknown metric {metric!r}; known: {METRICS}")
        if sense not in ("min", "max"):
            raise ValueError(f"objective {metric!r} sense must be 'min' or 'max'")
        objectives[metric] = sense
    if not objectives:
        raise ValueError("'explore.objectives' must name at least one metric")

    band = tuple(cfg.get("band", (0.05, 100.0)))
    fspec = cfg.get("freqs", {})
    start = fspec.get("start", -2)
    stop = fspec.get("stop", 4)
    num = int(fspec.get("num", 121))
    freqs = np.logspace(start, stop, num)
    return ExploreConfig(variables, constraints, objectives, band, freqs)


def explore_setup_from_dict(data):
    """Return (topo, base_sizes, base_bias, nf, ExploreConfig) from a parsed dict.

    Dict sibling of :func:`load_explore_json` — the single place that turns a
    circuit-JSON object carrying an ``explore`` block into the exploration inputs,
    so the CLI (which reads a file) and the service (which receives a body dict)
    share one parse path. Legacy ``builtin_topology`` configs are rejected."""
    cfg = parse_explore(data.get("explore"))
    if "builtin_topology" in data:
        raise ValueError("builtin_topology configs are deprecated; use a full circuit JSON")
    spec = circuit_from_dict(data)
    topo, sizes, bias, nf = spec.topology, dict(spec.sizes), dict(spec.bias), spec.nf
    return topo, sizes, bias, nf, cfg


def load_explore_json(path):
    """Return (topo, base_sizes, base_bias, nf, ExploreConfig) from a JSON file.

    The file must be a full circuit JSON carrying an ``explore`` block. Legacy
    ``builtin_topology`` configs are intentionally no longer accepted; keeping
    the topology in JSON makes circuit changes explicit and solver-independent."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return explore_setup_from_dict(data)


def explore_from_dict(data, n=200, seed=0, method="lhs", corner=None,
                      progress=None, should_stop=None):
    """Run a full exploration from a parsed circuit-JSON *dict*. Returns the
    results dict (same shape as :func:`explore`).

    The single shared entry point for `circuit-opt explore` (via :func:`run_cli`)
    and the service's ``POST /api/v1/jobs/explore`` — both parse the ``explore``
    block and bind any silicon models here, so the two surfaces can never drift.
    ``progress``/``should_stop`` are threaded straight to :func:`explore` for the
    background-job progress/cancel hooks."""
    topo, sizes, bias, nf, cfg = explore_setup_from_dict(data)
    # Bind any non-default per-device models (silicon sky130/freepdk45); without
    # this a silicon config silently falls back to the default OTFT PDK.
    model_types, device_kwargs = models_from_config(data)
    return explore(topo, sizes, bias, nf, cfg, n=n, seed=seed, method=method,
                   progress=progress, corner=corner, model_types=model_types,
                   device_kwargs=device_kwargs, should_stop=should_stop)


# ── sampling ──────────────────────────────────────────────────────────────
def _lhs_unit(rng, n, d):
    """Latin-hypercube samples in [0,1): one stratified draw per dimension."""
    edges = np.linspace(0.0, 1.0, n + 1)
    lo, span = edges[:n], edges[1:] - edges[:n]
    pts = lo[:, None] + rng.uniform(size=(n, d)) * span[:, None]
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def sample(variables, n, seed=0, method="lhs"):
    """Return a list of {var_name: value} dicts."""
    rng = np.random.default_rng(seed)
    d = len(variables)
    if method == "lhs":
        unit = _lhs_unit(rng, n, d)
    elif method == "random":
        unit = rng.uniform(size=(n, d))
    else:
        raise ValueError(f"unknown sampling method {method!r} (use 'lhs' or 'random')")
    out = []
    for row in unit:
        vals = {}
        for v, u in zip(variables, row):
            vals[v.name] = v.coerce(v.lo + u * (v.hi - v.lo))
        out.append(vals)
    return out


def apply_variables(variables, var_values, base_sizes, base_bias, base_nf=None):
    """Build (sizes, bias, nf) for one candidate by writing each variable to its
    targets. Targets are ``DEV.W`` / ``DEV.L`` (sizes), ``DEV.NF`` (number of
    fingers, integer), or a bare key (bias)."""
    sizes = {k: list(v) for k, v in base_sizes.items()}
    bias = dict(base_bias)
    nf_over = {}
    by_name = {v.name: v for v in variables}
    for name, value in var_values.items():
        for target in by_name[name].targets:
            if "." in target:
                dev, attr = target.split(".", 1)
                if dev not in sizes:
                    raise ValueError(f"variable target {target!r}: unknown device {dev!r}")
                if attr == "W":
                    sizes[dev][0] = value
                elif attr == "L":
                    sizes[dev][1] = value
                elif attr == "NF":
                    nf_over[dev] = max(1, int(round(value)))
                else:
                    raise ValueError(f"variable target {target!r}: attr must be W, L, or NF")
            else:
                bias[target] = value
    sizes = {k: tuple(v) for k, v in sizes.items()}
    if nf_over:
        nf = dict(base_nf) if isinstance(base_nf, dict) else (
            {name: int(base_nf) for name in sizes} if base_nf else {})
        nf.update(nf_over)
    else:
        nf = base_nf
    return sizes, bias, nf


# ── evaluation ──────────────────────────────────────────────────────────────
def _supply_power_uW(topo, bias, ss):
    """Top-rail supply current x rail voltage. Assumes the highest-voltage rail is
    the supply and all current enters through devices sourced from it (true for
    PMOS-top topologies like the AFE and the single-stage example)."""
    rail_v = topo.rail_values(bias)
    numeric = {k: v for k, v in rail_v.items() if isinstance(v, (int, float))}
    if not numeric:
        return 0.0
    top_rail = max(numeric, key=numeric.get)
    top_v = numeric[top_rail]
    i_supply = sum(abs(ss[name].get("Ich", 0.0))
                   for name, d, g, s in topo.devices if s == top_rail)
    return top_v * i_supply * 1e6


def _area(binding, sizes):
    """Sum of per-device ``g_area`` under *binding*. Each device is built with its own
    model type so a mixed-process circuit sums each PDK's real area metric (OTFT
    ``g_area`` is a padded layout box, not W·L); an all-default binding is byte-identical
    to the old path. ``g_area`` is a geometry metric (independent of the PVT corner)."""
    return float(sum(dev.g_area for dev in binding.build(sizes).values()))


def _needs_noise(constraints=None, objectives=None, require_noise=None):
    if require_noise is not None:
        return bool(require_noise)
    if constraints is None and objectives is None:
        return True
    needed = set((constraints or {}).keys()) | set((objectives or {}).keys())
    return bool(needed & NOISE_METRICS)


def _non_noise_constraints(constraints):
    return {k: v for k, v in (constraints or {}).items() if k not in NOISE_METRICS}


def _has_finite_metrics(metrics, names):
    return all(np.isfinite(metrics.get(name, float("nan"))) for name in names)


def evaluate(topo, sizes, bias, nf, freqs, band, x0_guess=None, corner=None,
             constraints=None, objectives=None, require_noise=None,
             model_types=None, device_kwargs=None, *, binding=None):
    """Run the solvers for one candidate -> metrics dict, or None if DC fails.

    x0_guess seeds the DC solve — required for topologies whose generic DC solve
    is not robust on its own (e.g. the AC-coupled AFE testbench, seeded from the
    bare-AFE operating point).
    corner applies a process shift (flat dict, e.g. the slow corner) or per-device
    mismatch map; passed straight through to the solvers.
    model_types / device_kwargs bind non-default per-device models (e.g. silicon
    SKY130 nmos/pmos); ``None`` keeps the default-PDK path byte-for-byte unchanged.

    Two calling conventions, one behavior. ``binding`` (a :class:`CircuitBinding`
    supplying topo / model_types / device_kwargs) is the **preferred** path and the
    one all internal callers now use — it carries the per-device model map so no
    candidate silently reverts to the default PDK. The bare ``model_types`` /
    ``device_kwargs`` kwargs are the **legacy** path, kept only for external scripts;
    with ``binding=None`` a binding is constructed from them, so the result is
    equivalent. When both are given, an explicit non-``None`` kwarg (and the explicit
    ``topo`` / ``nf``) **overrides** the binding's corresponding field, so a legacy
    caller passing the cluster directly is byte-identical to the old kwargs path.
    ``corner`` and ``x0_guess`` stay per-candidate and are threaded to the solvers
    explicitly.

    When constraints/objectives are supplied, noise is evaluated lazily: AC-derived
    metrics are checked first, and `irn_uV` is computed only if it is required and
    the candidate has not already failed non-noise constraints. Direct callers that
    omit constraints/objectives keep the old behavior and compute noise by default."""
    # One binding drives ac_solve / noise_analysis / _area so a candidate never loses
    # its per-device models. Explicit kwargs override the passed binding's fields; with
    # binding=None the cluster is built from the explicit args (legacy path).
    if binding is None:
        binding = CircuitBinding(topo=topo, model_types=model_types,
                                 device_kwargs=device_kwargs, nf=nf)
    else:
        binding = dataclasses.replace(
            binding, topo=topo, nf=nf,
            model_types=model_types if model_types is not None else binding.model_types,
            device_kwargs=device_kwargs if device_kwargs is not None else binding.device_kwargs)
    ac = ac_solve(sizes, bias, freqs, binding=binding, x0_guess=x0_guess, corner=corner)
    if ac is None:
        return None
    irn_uV = float("nan")
    try:
        power_uW = float(_supply_power_uW(binding.topo, bias, ac["ss"]))
    except Exception as exc:
        diagnostics.note("explore.power_eval_fail", exc)
        power_uW = float("nan")
    metrics = {
        "gain_dB": float(ac["Av_dc_dB"]),          # gain at the lowest analysis freq
        "gain_peak_dB": float(ac["peak_dB"]),      # passband peak (the spec gain for a bandpass)
        "bw_Hz": float(ac["bw_Hz"]),
        "irn_uV": float(irn_uV),
        "power_uW": power_uW,
        "area": _area(binding, sizes),
        "_noise_evaluated": False,
    }
    if (_needs_noise(constraints, objectives, require_noise) and
            is_feasible(metrics, _non_noise_constraints(constraints))):
        try:
            noise = noise_analysis(sizes, bias, freqs, binding=binding,
                                   x0_guess=ac["dc_op"], corner=corner)
            if noise is not None:
                metrics["irn_uV"] = float(band_rms(freqs, noise["irn_psd"],
                                                   band[0], band[1]) * 1e6)
            metrics["_noise_evaluated"] = True
        except Exception as exc:
            diagnostics.note("explore.irn_eval_fail", exc)
            metrics["irn_uV"] = float("nan")
    return metrics


def is_feasible(metrics, constraints):
    for metric, bound in constraints.items():
        value = metrics.get(metric)
        if value is None or not np.isfinite(value):
            return False
        if "min" in bound and value < bound["min"]:
            return False
        if "max" in bound and value > bound["max"]:
            return False
    return True


def pareto_front(rows, objectives):
    """Indices (into rows) of the non-dominated set over the given objectives.
    Each row is a metrics dict. 'min' objectives kept as-is, 'max' negated so the
    test is uniformly 'smaller is better'."""
    keys = list(objectives)
    if not rows:
        return []
    pts = []
    for r in rows:
        pts.append([r[k] if objectives[k] == "min" else -r[k] for k in keys])
    pts = np.asarray(pts, dtype=float)
    n = len(rows)
    front = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if j == i:
                continue
            # j dominates i: no worse in all, strictly better in at least one
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


# ── driver ──────────────────────────────────────────────────────────────────
def explore(topo, base_sizes, base_bias, nf, cfg, n=200, seed=0, method="lhs",
            progress=None, seed_fn=None, corner=None, model_types=None,
            device_kwargs=None, should_stop=None):
    """Sample, evaluate, constrain, and Pareto-select. Returns a results dict.

    seed_fn(sizes, bias) -> x0_guess optionally provides a per-candidate DC seed
    (e.g. the bare-AFE operating point for the testbench).
    corner applies a process shift (e.g. the slow corner) to every evaluation; for a
    SKY130 corner name it is routed onto the silicon devices instead.
    model_types / device_kwargs bind non-default per-device models (silicon).

    ``progress(done, total)`` — the existing 2-arg callback, fired after each
    candidate finishes (``done`` 1-based, ``total`` == ``n``). ``should_stop()`` —
    optional zero-arg predicate checked *before* each candidate; returning ``True``
    finishes early over the candidates already evaluated and adds
    ``"stopped_early": True`` to the results (and its ``summary``). Cancellation is
    cooperative — a candidate already in flight runs to completion first. Both
    default ``None``; with both ``None`` the result is unchanged from the pre-hook
    behaviour (same seed → same candidates)."""
    # One binding carries topo + the per-device model map to every candidate, so a
    # silicon config never silently falls back to the default OTFT PDK. ``at_corner``
    # bakes a silicon corner onto the device kwargs and clears the solver corner; an
    # OTFT corner stays on ``binding.corner`` and is threaded to the solvers as the
    # per-candidate ``corner=``. Per-candidate nf is applied inside ``evaluate``.
    binding = CircuitBinding(topo=topo, model_types=model_types,
                             device_kwargs=device_kwargs, nf=nf).at_corner(corner)
    corner = binding.corner
    samples = sample(cfg.variables, n, seed=seed, method=method)
    candidates = []
    stopped_early = False
    for i, var_values in enumerate(samples):
        if should_stop is not None and should_stop():
            stopped_early = True
            break
        sizes, bias, cand_nf = apply_variables(cfg.variables, var_values,
                                               base_sizes, base_bias, base_nf=nf)
        x0 = seed_fn(sizes, bias) if seed_fn is not None else None
        metrics = evaluate(topo, sizes, bias, cand_nf, cfg.freqs, cfg.band,
                           binding=binding, x0_guess=x0, corner=corner,
                           constraints=cfg.constraints, objectives=cfg.objectives)
        complete = bool(metrics is not None and _has_finite_metrics(metrics, cfg.objectives))
        candidates.append({
            "idx": i,
            "vars": var_values,
            "metrics": metrics,
            "converged": metrics is not None,
            "feasible": bool(metrics is not None and complete and
                             is_feasible(metrics, cfg.constraints)),
            "pareto": False,
            "noise_evaluated": bool(metrics and metrics.get("_noise_evaluated", False)),
        })
        if progress is not None:
            progress(i + 1, n)

    feasible = [c for c in candidates if c["feasible"]]
    front_local = pareto_front([c["metrics"] for c in feasible], cfg.objectives)
    for k in front_local:
        feasible[k]["pareto"] = True

    summary = {
        "n": n,
        "converged": sum(c["converged"] for c in candidates),
        "feasible": len(feasible),
        "pareto": len(front_local),
        "noise_evaluated": sum(c["noise_evaluated"] for c in candidates),
        "best": {},
    }
    for metric, sense in cfg.objectives.items():
        pool = [c for c in feasible]
        if pool:
            pick = (min if sense == "min" else max)(pool, key=lambda c: c["metrics"][metric])
            summary["best"][metric] = {"idx": pick["idx"], "value": pick["metrics"][metric],
                                       "vars": pick["vars"]}
    results = {"candidates": candidates, "summary": summary,
               "variables": [v.name for v in cfg.variables],
               "objectives": cfg.objectives}
    if stopped_early:
        # Early cancellation: expose how many candidates actually ran (``n`` above
        # is the requested count) plus the flag, both at the top level and in the
        # summary, mirroring mismatch_mc's ``stopped_early`` contract.
        summary["evaluated"] = len(candidates)
        summary["stopped_early"] = True
        results["stopped_early"] = True
    return results


# ── output ────────────────────────────────────────────────────────────────
def _flat_rows(results, metrics=None):
    """Flatten a results dict to (rows, fieldnames).

    ``metrics`` names the per-candidate metric columns; it defaults to this module's
    AC :data:`METRICS` so every existing caller is byte-for-byte unchanged. The SAR
    ADC explorer (:mod:`circuitopt.sar_explore`), whose candidates carry a different
    metric set, passes its own tuple here so it can reuse ``write_csv``/``write_jsonl``
    verbatim instead of duplicating them. Metric keys absent from a candidate render
    as blank/``None`` rather than raising, so a mixed metric set is tolerated."""
    metric_names = METRICS if metrics is None else tuple(metrics)
    var_names = results["variables"]
    rows = []
    for c in results["candidates"]:
        row = {"idx": c["idx"]}
        for name in var_names:
            row[f"var_{name}"] = c["vars"].get(name)
        for m in metric_names:
            row[m] = c["metrics"].get(m) if c["metrics"] else None
        row["converged"] = int(c["converged"])
        row["feasible"] = int(c["feasible"])
        row["pareto"] = int(c["pareto"])
        rows.append(row)
    return rows, ["idx"] + [f"var_{n}" for n in var_names] + list(metric_names) + \
        ["converged", "feasible", "pareto"]


def write_csv(results, path, metrics=None):
    rows, fields = _flat_rows(results, metrics)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(results, path, metrics=None):
    rows, _ = _flat_rows(results, metrics)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────
def _format_summary(results):
    s = results["summary"]
    lines = [f"candidates: {s['n']}   converged: {s['converged']}   "
             f"feasible: {s['feasible']}   pareto: {s['pareto']}   "
             f"noise: {s['noise_evaluated']}"]
    if s["best"]:
        lines.append("best feasible per objective:")
        for metric, info in s["best"].items():
            sense = results["objectives"][metric]
            lines.append(f"  {metric} ({sense}) = {info['value']:.4g}  [candidate #{info['idx']}]")
    return "\n".join(lines)


def add_cli_args(parser):
    """Register the explore feature's own arguments on ``parser``.

    Single source of truth for both ``python -m circuitopt.explore`` and the
    ``python -m circuitopt explore`` subcommand — keeps the two CLI surfaces from
    drifting. Feature-only: subcommand-level mechanisms (e.g. ``--no-numba``)
    stay with their host parser, not here.

    The output prefix accepts both ``--out`` (module-CLI spelling) and
    ``-o/--output`` (subcommand spelling); they share the ``output`` dest so
    either name works from either surface."""
    parser.add_argument("config", help="JSON file carrying an 'explore' block")
    parser.add_argument("-n", "--n", type=int, default=200,
                        help="Number of candidates (default: 200)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--method", choices=("lhs", "random"), default="lhs",
                        help="Sampling method (default: lhs)")
    parser.add_argument("--corner", default=None,
                        help="Process corner: OTFT typical|slow|fast, or silicon "
                             "tt|ss|ff|sf|fs (SKY130) / nom|ss|ff (FreePDK45)")
    parser.add_argument("-o", "--out", "--output", dest="output", default=None,
                        help="Output path prefix (writes <prefix>.csv/.jsonl)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-candidate progress")
    return parser


def run_cli(args):
    """Execute the explore feature from parsed ``args``. Returns the results dict."""
    with open(args.config, "r", encoding="utf-8") as f:
        data = json.load(f)

    def progress(done, total):
        if not args.quiet:
            print(f"\r  evaluating {done}/{total}", end="", flush=True)

    if not args.quiet:
        print(f"Exploring {args.config}  (n={args.n}, method={args.method})")
    # explore_from_dict is the shared CLI/service entry: it parses the explore
    # block + binds silicon models once, so both surfaces stay in lock-step.
    results = explore_from_dict(data, n=args.n, seed=args.seed, method=args.method,
                                corner=args.corner, progress=progress)
    if not args.quiet:
        print()
    print(_format_summary(results))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        write_csv(results, args.output + ".csv")
        write_jsonl(results, args.output + ".jsonl")
        print(f"wrote {args.output}.csv and {args.output}.jsonl")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Circuit design-space exploration / optimization.")
    add_cli_args(parser)
    args = parser.parse_args(argv)
    return run_cli(args)


if __name__ == "__main__":
    main()
