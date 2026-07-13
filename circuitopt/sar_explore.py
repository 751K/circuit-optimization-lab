"""Design-space exploration for the transistor-level SAR ADC workflow.

This is the ADC-metric sibling of :mod:`circuitopt.explore`. That module screens an
analog block on *AC* metrics (gain/bw/irn/power/area) from ``ac_solve``; those
numbers say nothing about an ADC's linearity, so a SAR needs its own evaluation
loop over the closed-loop ngspice conversion oracle. Everything that is not
ADC-specific is reused verbatim from :mod:`circuitopt.explore` — the :class:`Variable`
range/targets mechanism, Latin-hypercube :func:`sample`, :func:`pareto_front`,
:func:`is_feasible` and the ``write_csv``/``write_jsonl`` writers — so the two
explorers share one sampling / feasibility / Pareto / output surface and can never
drift on those axes.

Config JSON (a *standalone* file, unlike explore's embedded block) mirrors the
explore config shape plus a ``"circuit"`` pointer::

    {
      "circuit": "freepdk45_sar3.json",          // resolved relative to this file
      "sweep_points": 8,                          // code-center samples (subsample for speed)
      "variables": {
        "in_pair_W": {"min": 1.0, "max": 4.0, "round": 2, "targets": ["W:M1", "W:M2"]},
        "unit_cap":  {"min": 8e-15, "max": 1.2e-14, "targets": ["C:C0P", "C:C0N"]}
      },
      "constraints": {"missing_codes": {"max": 0}, "monotonic": {"min": 1}},
      "objectives":  {"max_abs_dnl": "min", "power_uw": "min"},
      "dynamic": {"n_samples": 64, "cycles": 5}   // optional coherent-sine SNDR/ENOB
    }

Target encoding (see :func:`apply_sar_variables`), consistent colon-prefixed forms:

  * ``"W:<dev>"`` / ``"L:<dev>"`` / ``"NF:<dev>"`` — a device size / finger count.
    The native explore dotted form (``"<dev>.W"``) is also accepted.
  * ``"C:<name>"`` — a CDAC capacitor value [F].
  * a bare key — a bias entry (as in explore).

One variable may drive several targets (its ``targets`` is a list), so a matched
cap group (``C0P``+``C0N``) or a differential transistor pair (``M1``+``M2``) moves
as one knob. Capacitor edits are applied to a per-candidate *copy* of the topology
(the shallow-copy pattern of :func:`circuitopt.sar_mc.perturb_capacitors`) so the
loaded spec is never mutated.

Metrics per candidate: ``max_abs_dnl``, ``max_abs_inl``, ``missing_codes`` (count),
``monotonic`` (0/1), ``power_uw``, ``conv_time_ns``, ``energy_per_conv_pj`` and —
only when a ``"dynamic"`` block is present — ``enob``, ``sndr_db``, ``sfdr_db``.
"""
from __future__ import annotations

import dataclasses
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from . import diagnostics
from .circuit_loader import CircuitSpec, load_circuit_json
from .adc import dynamic_metrics, static_ramp_metrics
from .explore import (Variable, apply_variables, is_feasible, pareto_front,
                      sample, write_csv, write_jsonl)
from .sar import _sar_config, run_sar_conversion
from .sar_mc import _copy_with_capacitors


# ADC-metric column/constraint/objective vocabulary. Kept as a fixed tuple so the
# CSV/JSONL columns are stable and a config typo is rejected instead of silently
# constraining nothing. The dynamic trio is always present in a metrics dict (NaN
# when no ``dynamic`` block ran), which keeps the column set constant.
METRICS = ("max_abs_dnl", "max_abs_inl", "missing_codes", "monotonic",
           "power_uw", "conv_time_ns", "energy_per_conv_pj",
           "enob", "sndr_db", "sfdr_db")


# ── configuration ────────────────────────────────────────────────────────────
class SarExploreConfig:
    def __init__(self, variables, constraints, objectives, *, sweep_points=None,
                 dynamic=None):
        self.variables = variables          # list[Variable]
        self.constraints = constraints      # metric -> {"min"?, "max"?}
        self.objectives = objectives        # metric -> "min" | "max"
        self.sweep_points = sweep_points    # int | None (subsample the code-center sweep)
        self.dynamic = dynamic              # {"n_samples", "cycles", ...} | None


def parse_sar_explore(cfg) -> SarExploreConfig:
    """Validate the standalone SAR-explore config dict into a :class:`SarExploreConfig`.

    Mirrors :func:`circuitopt.explore.parse_explore` (same variable/constraint/objective
    shape) but scores against the ADC :data:`METRICS` vocabulary and additionally
    accepts ``sweep_points`` / ``dynamic``.
    """
    if not isinstance(cfg, dict):
        raise ValueError("SAR explore config must be an object")
    raw_vars = cfg.get("variables")
    if not isinstance(raw_vars, dict) or not raw_vars:
        raise ValueError("'variables' must be a non-empty object")
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
        raise ValueError("'objectives' must name at least one metric")

    sweep_points = cfg.get("sweep_points")
    if sweep_points is not None:
        sweep_points = int(sweep_points)
        if sweep_points < 2:
            raise ValueError("'sweep_points' must be at least 2")

    dynamic = cfg.get("dynamic")
    if dynamic is not None:
        if not isinstance(dynamic, dict):
            raise ValueError("'dynamic' must be an object")
        n_samples = int(dynamic.get("n_samples", 64))
        cycles = int(dynamic.get("cycles", 5))
        if n_samples < 8 or cycles < 1 or 2 * cycles >= n_samples:
            raise ValueError("dynamic needs n_samples>=8 and 1<=cycles<n_samples/2")
        dynamic = {"n_samples": n_samples, "cycles": cycles,
                   "amplitude": dynamic.get("amplitude"),
                   "offset": dynamic.get("offset")}
    return SarExploreConfig(variables, constraints, objectives,
                            sweep_points=sweep_points, dynamic=dynamic)


def sar_explore_from_dict(data, *, base_dir=".", circuit_path=None):
    """Return ``(spec, SarExploreConfig)`` from a parsed standalone config dict.

    ``circuit_path`` (the CLI positional) is authoritative when given; otherwise the
    config's ``"circuit"`` key is resolved relative to ``base_dir``. If both are
    present and resolve to different files, that is a configuration error.
    """
    cfg = parse_sar_explore(data)
    cfg_circuit = data.get("circuit")
    resolved_cfg = (os.path.normpath(os.path.join(base_dir, cfg_circuit))
                    if cfg_circuit else None)
    if circuit_path is not None:
        if resolved_cfg is not None and \
                os.path.abspath(resolved_cfg) != os.path.abspath(circuit_path):
            raise ValueError(
                f"config 'circuit' ({resolved_cfg!r}) differs from the supplied "
                f"circuit ({circuit_path!r})")
        spec = load_circuit_json(circuit_path)
    elif resolved_cfg is not None:
        spec = load_circuit_json(resolved_cfg)
    else:
        raise ValueError("no circuit given: pass circuit_path or a 'circuit' key")
    if spec.adc is None:
        raise ValueError("the referenced circuit has no 'adc' workflow block")
    return spec, cfg


def load_sar_explore_json(path, *, circuit_path=None):
    """Return ``(spec, SarExploreConfig)`` from a standalone SAR-explore JSON file.

    Sibling of :func:`sar_explore_from_dict`; the config's ``"circuit"`` is resolved
    relative to the config file's own directory.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sar_explore_from_dict(data, base_dir=os.path.dirname(os.path.abspath(path)),
                                 circuit_path=circuit_path)


# ── candidate construction ────────────────────────────────────────────────────
def _normalize_target(target: str):
    """Map a SAR-explore target string to ``(kind, key)``.

    ``"C:<name>"`` -> ``("cap", name)``; ``"W:M1"``/``"L:M1"``/``"NF:M1"`` ->
    ``("size", "M1.W")`` (the native :func:`circuitopt.explore.apply_variables` dotted
    form); a dotted ``"M1.W"`` or a bare bias key passes straight through as
    ``("size", target)`` for ``apply_variables`` to interpret.
    """
    if target.startswith("C:"):
        return "cap", target[2:]
    for prefix in ("W:", "L:", "NF:"):
        if target.startswith(prefix):
            return "size", f"{target[len(prefix):]}.{prefix[:-1]}"
    return "size", target


def apply_sar_variables(variables, var_values, spec: CircuitSpec) -> CircuitSpec:
    """Build one candidate ``CircuitSpec`` by writing each variable to its targets.

    Size/bias/NF targets are delegated to :func:`circuitopt.explore.apply_variables`
    (so those semantics can never diverge from the analog explorer); ``C:`` targets
    rebind capacitor values on a shallow copy of the topology via
    :func:`circuitopt.sar_mc._copy_with_capacitors`, never touching the loaded spec.
    A ``C:`` target naming an unknown capacitor is rejected.
    """
    by_name = {v.name: v for v in variables}
    cap_updates: dict[str, float] = {}
    size_targets: dict[str, list[str]] = defaultdict(list)
    size_values: dict[str, float] = {}
    for name, value in var_values.items():
        for target in by_name[name].targets:
            kind, key = _normalize_target(target)
            if kind == "cap":
                cap_updates[key] = float(value)
            else:
                size_targets[name].append(key)
                size_values[name] = value

    if size_targets:
        temp_vars = [Variable(n, by_name[n].lo, by_name[n].hi, targets=tg)
                     for n, tg in size_targets.items()]
        sizes, bias, nf = apply_variables(temp_vars, size_values, spec.sizes,
                                          spec.bias, base_nf=spec.nf)
        # CircuitSpec is a frozen dataclass; replace gives a new instance carrying
        # the candidate's sizes/bias/nf without mutating the loaded spec.
        spec = dataclasses.replace(spec, sizes=sizes, bias=bias, nf=nf)

    if cap_updates:
        known = {name for name, *_ in spec.topology.capacitors}
        unknown = set(cap_updates) - known
        if unknown:
            raise ValueError(f"C: target(s) name unknown capacitor(s): {sorted(unknown)}")
        new_caps = [(name, a, b, cap_updates.get(name, value))
                    for name, a, b, value in spec.topology.capacitors]
        spec = _copy_with_capacitors(spec, new_caps)
    return spec


# ── evaluation ────────────────────────────────────────────────────────────────
def _dynamic_sar_metrics(spec, scfg, dyn, corner):
    """Coherent-sine SNDR/SFDR/ENOB for one candidate (only when configured)."""
    vref = scfg["vref"]
    n_samples = dyn["n_samples"]
    cycles = dyn["cycles"]
    offset = 0.5 * vref if dyn.get("offset") is None else float(dyn["offset"])
    amplitude = 0.45 * vref if dyn.get("amplitude") is None else float(dyn["amplitude"])
    phase = 2.0 * np.pi * cycles * np.arange(n_samples) / n_samples
    vin = np.clip(offset + amplitude * np.sin(phase), 0.0, vref)
    codes = np.array(
        [run_sar_conversion(spec, float(v), config=scfg, corner=corner)["code"]
         for v in vin], dtype=np.int64)
    m = dynamic_metrics(codes, 1.0, fundamental_bin=cycles)
    return {"enob": float(m["enob"]), "sndr_db": float(m["sndr_db"]),
            "sfdr_db": float(m["sfdr_db"])}


def evaluate_sar(spec: CircuitSpec, cfg: SarExploreConfig, *, corner=None) -> dict | None:
    """Run one candidate's code-center static sweep -> ADC metrics dict (or ``None``).

    The sweep is the SAR code-center ramp ``(arange(2**n)+0.5)/2**n * vref`` —
    the input that lands one sample squarely in each code bin — optionally
    subsampled to ``cfg.sweep_points`` conversions for speed. Linearity is guarded
    exactly like :func:`circuitopt.sar_mc._trial_metrics`: a non-monotonic code set
    (possible under aggressive sizing) is scored as a failure (DNL/INL ``inf``,
    ``monotonic=0``) rather than crashing :func:`static_ramp_metrics`.

    Power is the mean per-conversion ``total_power_w`` over the sweep; conversion
    time is the SAR timing grid span; energy is their product. The dynamic trio is
    ``NaN`` unless ``cfg.dynamic`` is set. Returns ``None`` if a conversion raises
    (the candidate is scored non-converged upstream)."""
    scfg = _sar_config(spec)
    n_bits = scfg["n_bits"]
    vref = scfg["vref"]
    levels = 1 << n_bits
    vin_full = (np.arange(levels) + 0.5) / levels * vref
    sp = cfg.sweep_points
    if sp is not None and sp < levels:
        idx = np.unique(np.linspace(0, levels - 1, sp).round().astype(int))
        vin = vin_full[idx]
    else:
        vin = vin_full

    try:
        convs = [run_sar_conversion(spec, float(v), config=scfg, corner=corner)
                 for v in vin]
    except Exception as exc:                       # pragma: no cover - candidate skip
        diagnostics.note("sar_explore.conversion_fail", exc)
        return None

    codes = np.array([c["code"] for c in convs], dtype=np.int64)
    power_w = float(np.mean([c["total_power_w"] for c in convs]))
    present = np.unique(codes)
    missing = int(levels - present.size)
    monotonic = bool(np.all(np.diff(codes) >= 0))
    # Conversion time = the SAR time-grid span (sample phase + n_bits+1 bit periods).
    conv_time_s = scfg["sample_end"] + (n_bits + 1) * scfg["bit_period"]

    metrics = {
        "missing_codes": float(missing),
        "monotonic": 1.0 if monotonic else 0.0,
        "power_uw": power_w * 1e6,
        "conv_time_ns": conv_time_s * 1e9,
        "energy_per_conv_pj": power_w * conv_time_s * 1e12,
        "enob": float("nan"),
        "sndr_db": float("nan"),
        "sfdr_db": float("nan"),
    }
    if monotonic:
        m = static_ramp_metrics(vin, codes, n_bits, vmin=0.0, vmax=vref)
        dnl = float(m["max_abs_dnl"])
        inl = float(m["max_abs_inl"])
        # A subsampled sweep can leave every transition undefined -> NaN; treat that
        # as a failed (worst-case) linearity so the objective/constraint stays finite.
        metrics["max_abs_dnl"] = dnl if np.isfinite(dnl) else float("inf")
        metrics["max_abs_inl"] = inl if np.isfinite(inl) else float("inf")
    else:
        metrics["max_abs_dnl"] = float("inf")
        metrics["max_abs_inl"] = float("inf")

    if cfg.dynamic is not None:
        try:
            metrics.update(_dynamic_sar_metrics(spec, scfg, cfg.dynamic, corner))
        except Exception as exc:                   # pragma: no cover - dynamic skip
            diagnostics.note("sar_explore.dynamic_fail", exc)
    return metrics


def _has_finite_objectives(metrics, objectives):
    return metrics is not None and all(
        np.isfinite(metrics.get(name, float("nan"))) for name in objectives)


# ── driver ────────────────────────────────────────────────────────────────────
def sar_explore(spec: CircuitSpec, cfg: SarExploreConfig, *, n=50, seed=0,
                method="lhs", corner=None, workers=1, progress=None) -> dict:
    """Sample -> evaluate each SAR candidate -> constrain -> Pareto-select.

    Returns a results dict with the same shape as :func:`circuitopt.explore.explore`
    (``candidates`` rows with ``idx``/``vars``/``metrics``/``converged``/``feasible``/
    ``pareto`` plus a ``summary``), so the reused ``write_csv``/``write_jsonl`` writers
    accept it directly (pass :data:`METRICS`).

    ``workers`` parallelises *across candidates* on a thread pool (each candidate's
    code-center sweep still runs serially inside :func:`evaluate_sar` — the pools are
    never nested). ``progress(done, total)`` fires from the main thread as each
    candidate finishes, with a monotonic completed count.
    """
    if workers is None or workers < 1:
        raise ValueError("workers must be a positive integer")
    samples = sample(cfg.variables, n, seed=seed, method=method)

    def _eval(i):
        cand_spec = apply_sar_variables(cfg.variables, samples[i], spec)
        metrics = evaluate_sar(cand_spec, cfg, corner=corner)
        complete = _has_finite_objectives(metrics, cfg.objectives)
        return i, {
            "idx": i,
            "vars": samples[i],
            "metrics": metrics,
            "converged": metrics is not None,
            "feasible": bool(metrics is not None and complete and
                             is_feasible(metrics, cfg.constraints)),
            "pareto": False,
        }

    candidates: list[dict | None] = [None] * n
    if workers == 1:
        for i in range(n):
            _, cand = _eval(i)
            candidates[i] = cand
            if progress is not None:
                progress(i + 1, n)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_eval, i) for i in range(n)]
            for done, fut in enumerate(as_completed(futures), start=1):
                i, cand = fut.result()
                candidates[i] = cand           # final order stays by candidate index
                if progress is not None:
                    progress(done, n)
    candidates = [c for c in candidates if c is not None]

    feasible = [c for c in candidates if c["feasible"]]
    front_local = pareto_front([c["metrics"] for c in feasible], cfg.objectives)
    for k in front_local:
        feasible[k]["pareto"] = True

    summary = {
        "n": n,
        "converged": sum(c["converged"] for c in candidates),
        "feasible": len(feasible),
        "pareto": len(front_local),
        "best": {},
    }
    for metric, sense in cfg.objectives.items():
        if feasible:
            pick = (min if sense == "min" else max)(
                feasible, key=lambda c: c["metrics"][metric])
            summary["best"][metric] = {"idx": pick["idx"],
                                       "value": pick["metrics"][metric],
                                       "vars": pick["vars"]}
    return {"candidates": candidates, "summary": summary,
            "variables": [v.name for v in cfg.variables],
            "objectives": cfg.objectives}


def sar_write_csv(results, path):
    """CSV writer for SAR-explore results (reuses explore's writer with SAR columns)."""
    write_csv(results, path, metrics=METRICS)


def sar_write_jsonl(results, path):
    """JSONL writer for SAR-explore results (reuses explore's writer with SAR columns)."""
    write_jsonl(results, path, metrics=METRICS)


# ── CLI helpers ───────────────────────────────────────────────────────────────
def format_sar_summary(results) -> str:
    s = results["summary"]
    lines = [f"candidates: {s['n']}   converged: {s['converged']}   "
             f"feasible: {s['feasible']}   pareto: {s['pareto']}"]
    if s["best"]:
        lines.append("best feasible per objective:")
        for metric, info in s["best"].items():
            sense = results["objectives"][metric]
            lines.append(f"  {metric} ({sense}) = {info['value']:.4g}  "
                         f"[candidate #{info['idx']}]")
    return "\n".join(lines)
