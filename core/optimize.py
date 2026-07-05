"""Surrogate-accelerated design optimization — the build → train → OPTIMIZE → VERIFY loop.

Screen a large candidate pool with a trained surrogate (µs/candidate), keep the
constrained Pareto front, then verify the top-K with the calibrated solver. The
surrogate *proposes* (bulk screening, orders of magnitude faster); the Cadence-
calibrated solver *disposes* (final word on the shortlist). This is the payoff of
the surrogate roadmap: fast refinement without giving up sign-off accuracy on
the designs you actually keep.

Needs the optional scikit-learn dependency (via :mod:`core.surrogate`).

Usage::

    python -m core.optimize examples/afe_explore.json results/models/afe_typical.pkl \\
        -n 100000 --top-k 20
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np

from . import surrogate as sg
from .circuit_loader import models_from_config
from .dataset import candidate_circuit, load_dataset_config, split_variables
from .device_factory import CircuitBinding
from .explore import (evaluate, is_feasible,
                      pareto_front, sample)


def _design_matrix(samples, var_names):
    """Stack sampled ``{var: value}`` dicts into an ``(n, d)`` matrix in ``var_names`` order."""
    return np.array([[s[name] for name in var_names] for s in samples], dtype=float)


def _spread_pick(front_idx, metric_rows, objectives, k):
    """Pick ``k`` Pareto points spread along the first objective (even sampling of the front)."""
    if len(front_idx) <= k:
        return list(front_idx)
    key = list(objectives)[0]
    ordered = sorted(front_idx, key=lambda i: metric_rows[i][key])
    step = len(ordered) / k
    return [ordered[min(len(ordered) - 1, int(j * step))] for j in range(k)]


def optimize(config_path, surrogate_path, *, n_screen=100000, top_k=20, seed=0,
             method="lhs", corner=None, freqs=None, verify=True):
    """Screen ``n_screen`` designs with the surrogate, take the constrained Pareto
    front, and (``verify``) re-evaluate the top-K on the solver. Returns a report dict.

    Constraints / objectives come from the config's ``explore`` block. ``freqs``
    should match the surrogate's training grid (default: the wide 0.01 Hz–10 kHz grid)."""
    config_dict, topo, base_sizes, base_bias, nf, cfg = load_dataset_config(config_path)
    model_types, device_kwargs = models_from_config(config_dict)
    # One binding carries topo + the per-device model map into the verify pass, so the
    # solver re-run never loses the silicon models. ``at_corner`` routes a SKY130 corner
    # onto the silicon devices and clears the solver corner; an OTFT corner stays on
    # ``binding.corner`` and is threaded to ``evaluate`` as the per-candidate ``corner=``.
    binding = CircuitBinding(topo=topo, model_types=model_types,
                             device_kwargs=device_kwargs, nf=nf).at_corner(corner)
    corner = binding.corner
    if freqs is None:
        freqs = np.logspace(-2, 4, 101)          # match the surrogate's training grid
    cfg.freqs = np.asarray(freqs, float)
    model = sg.load(surrogate_path)

    # ── screen: sample a big pool and predict every candidate with the surrogate ──
    samples = sample(cfg.variables, n_screen, seed=seed, method=method)
    X = _design_matrix(samples, model.var_names)
    t0 = time.time()
    Y = model.predict(X)
    t_screen = time.time() - t0
    metric_rows = [{lab: float(Y[i, j]) for j, lab in enumerate(model.label_names)}
                   for i in range(len(Y))]

    # ── constrain + Pareto-select on the surrogate's predictions ──
    feasible = [i for i, m in enumerate(metric_rows) if is_feasible(m, cfg.constraints)]
    front_local = pareto_front([metric_rows[i] for i in feasible], cfg.objectives)
    front = [feasible[k] for k in front_local]
    picks = _spread_pick(front, metric_rows, cfg.objectives, top_k)

    report = {
        "n_screen": n_screen, "screen_seconds": t_screen,
        "surrogate_feasible": len(feasible), "surrogate_pareto": len(front),
        "objectives": dict(cfg.objectives), "constraints": dict(cfg.constraints),
        "verified": [],
    }
    if not verify:
        report["picks"] = [samples[i] for i in picks]
        return report

    # ── verify: run the calibrated solver on the shortlist ──
    # Every variable kind (size/bias + structural cap/resistor/clock) is applied via
    # the shared candidate_circuit builder, so a swept passive value is honored in the
    # verify pass (not silently held at its base value).
    size_vars, struct_vars, _ = split_variables(cfg.variables)
    t0 = time.time()
    for i in picks:
        cand_topo, sizes, bias, cand_nf = candidate_circuit(
            config_dict, topo, base_sizes, base_bias, nf, size_vars, struct_vars, samples[i])
        cand_binding = dataclasses.replace(binding, topo=cand_topo, nf=cand_nf)
        true = evaluate(cand_topo, sizes, bias, cand_nf, cfg.freqs, cfg.band,
                        binding=cand_binding, corner=corner, require_noise=True)
        pred = metric_rows[i]
        entry = {"design": samples[i], "surrogate": pred, "solver": None,
                 "solver_feasible": None}
        if true is not None:
            solver_metrics = {lab: true.get(lab) for lab in model.label_names}
            entry["solver"] = solver_metrics
            entry["solver_feasible"] = bool(is_feasible(true, cfg.constraints))
        report["verified"].append(entry)
    report["verify_seconds"] = time.time() - t0
    report["solver_ms_per_design"] = (report["verify_seconds"] / max(1, len(picks))) * 1e3
    return report


def _summ_errors(report):
    """Median surrogate-vs-solver relative error (%) per label over verified picks."""
    labels = list(report["verified"][0]["surrogate"]) if report["verified"] else []
    out = {}
    for lab in labels:
        rels = []
        for e in report["verified"]:
            if e["solver"] and e["solver"].get(lab) is not None:
                t = float(e["solver"][lab])
                rels.append(abs(float(e["surrogate"][lab]) - t) / max(abs(t), 1e-30))
        if rels:
            out[lab] = float(np.median(rels) * 100.0)
    return out


def _format_report(report):
    r = report
    scr_rate = r["n_screen"] / max(1e-9, r["screen_seconds"])
    lines = [
        f"screened {r['n_screen']} designs with the surrogate in {r['screen_seconds']:.2f}s "
        f"({scr_rate:,.0f}/s)",
        f"  feasible: {r['surrogate_feasible']}   Pareto front: {r['surrogate_pareto']}   "
        f"objectives: {r['objectives']}",
    ]
    if r["verified"]:
        n_ok = sum(e["solver_feasible"] is True for e in r["verified"])
        errs = _summ_errors(r)
        lines.append(
            f"verified top-{len(r['verified'])} on the solver in {r['verify_seconds']:.2f}s "
            f"({r['solver_ms_per_design']:.1f} ms/design)")
        lines.append(f"  solver-confirmed feasible: {n_ok}/{len(r['verified'])}")
        lines.append("  surrogate-vs-solver median error on the shortlist: "
                     + "  ".join(f"{lab}={e:.2f}%" for lab, e in errs.items()))
        full_solve = r["solver_ms_per_design"] * 1e-3 * r["n_screen"]
        lines.append(f"  speedup: screening all {r['n_screen']} on the solver would take "
                     f"~{full_solve:.0f}s; the surrogate did it in {r['screen_seconds']:.2f}s "
                     f"→ ~{full_solve / max(1e-9, r['screen_seconds']):,.0f}×")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Surrogate-accelerated design optimization (screen → Pareto → verify).")
    p.add_argument("config", help="circuit JSON with an 'explore' block")
    p.add_argument("surrogate", help="trained surrogate (joblib .pkl)")
    p.add_argument("-n", "--n-screen", type=int, default=100000, help="candidate pool size")
    p.add_argument("--top-k", type=int, default=20, help="Pareto points to verify on the solver")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corner", default=None,
                   help="verify-pass process corner: OTFT typical|slow|fast or SKY130 "
                        "tt|ss|ff|sf|fs (screen uses the surrogate's training corner)")
    p.add_argument("--no-verify", action="store_true", help="skip the solver verification pass")
    args = p.parse_args(argv)
    try:
        report = optimize(args.config, args.surrogate, n_screen=args.n_screen,
                          top_k=args.top_k, seed=args.seed, corner=args.corner,
                          verify=not args.no_verify)
    except ImportError as exc:
        raise SystemExit(str(exc))
    print(_format_report(report))
    return report


if __name__ == "__main__":
    main()
