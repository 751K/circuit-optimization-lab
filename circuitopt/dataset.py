"""Surrogate dataset builder — turn the validated solvers into a labeled training set.

The solver stack is calibrated against Cadence, and :mod:`circuitopt.explore` already
samples a design space and evaluates each candidate through it. A surrogate model
needs the *same* samples, but as a dataset rather than a Pareto front. This module
is the "Teacher simulator + Dataset builder" role of the ML-surrogate roadmap:
it reuses explore's sampling and validated per-candidate evaluation and adds
only the three concerns explore omits.

* **No filtering.** Constraints / Pareto selection are a *search* concern. A
  training set keeps *every* sample, including DC-failed ones — those are
  classification / constraint-boundary labels, not garbage to drop.
* **Full labels.** explore evaluates noise lazily (only when a constraint needs
  it); here noise is always evaluated so every convergent design carries the
  complete label set (``gain_dB``, ``bw_Hz``, ``irn_uV``, ``power_uW``, ``area`` …).
* **Provenance.** A manifest records the schema version, solver git commit (and
  whether the tree was dirty), a topology hash, the PDK, the corner, the sampling
  seed/method, and the variable ranges. A consumer can then reject out-of-domain
  designs instead of silently extrapolating, and a dataset is reproducible from
  (config + seed + solver commit).

Outputs (``<out>`` prefix): ``<out>.jsonl`` (one design/label row per line,
human-debuggable), ``<out>.manifest.json`` (provenance), and optionally
``<out>.npz`` (dense ``X`` / ``Y`` matrices for training, NaN where a label is
missing). This is deliberately a thin, dependency-free layer; Parquet / ML training
live downstream.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

from . import diagnostics
from .circuit_loader import circuit_from_dict, models_from_config
from .device_factory import (CORNERS, SKY130_CORNERS, CircuitBinding,
                             apply_silicon_corner)
from .device_model import get_default_model_type
from .explore import (METRICS, apply_variables,
                      evaluate, parse_explore, sample)
from .transient_solver import transient

SCHEMA_VERSION = "1.2"

# Label groups: each is an opt-in bundle of columns. The default is the AC/noise
# group (unchanged from schema 1.0). The transient group adds stimulus-agnostic
# waveform features (peak-to-peak / mean / RMS / max-slew / final value — no
# step-response assumptions), computed from the config's *validated* periodic
# transient, so it never fabricates a stimulus.
AC_NOISE_LABELS = tuple(METRICS)          # gain_dB gain_peak_dB bw_Hz irn_uV power_uW area
TRANSIENT_LABELS = ("out_pp", "out_mean", "out_rms", "slew_rate", "final_value")
# pss group: periodic steady-state quality + orbit output, for periodic circuits
# (chopper / SC). ``pss_converged`` (1/0) is the trust flag; the orbit features are
# taken over one converged period. These are genuinely PSS-derived — phase margin
# (an AC loop-gain metric) and settling (a step-response metric) are *not* PSS and
# stay out of this group.
PSS_LABELS = ("pss_converged", "pss_residual", "pss_iters", "pss_out_pp", "pss_out_mean")
# pac / pnoise groups: the LPTV small-signal transfer and folded periodic noise
# around the PSS orbit — the chopper's actual figures of merit (conversion gain,
# band-integrated input-referred noise). Both require the config's validated
# ``analyses.pac`` / ``analyses.pnoise`` blocks (drive, freqs, time_domain, band):
# a hard-switched circuit measured with the wrong PAC settings converges to a
# silently wrong gain, so these are never fabricated.
PAC_LABELS = ("pac_gain", "pac_gain_dB", "pac_bw_Hz")
PNOISE_LABELS = ("pnoise_out_uV", "pnoise_irn_uV")
LABEL_GROUPS = {"ac_noise": AC_NOISE_LABELS, "transient": TRANSIENT_LABELS,
                "pss": PSS_LABELS, "pac": PAC_LABELS, "pnoise": PNOISE_LABELS}
_PERIODIC_LABELS = (frozenset(TRANSIENT_LABELS) | frozenset(PSS_LABELS) |
                    frozenset(PAC_LABELS) | frozenset(PNOISE_LABELS))
DEFAULT_GROUPS = ("ac_noise",)
LABELS = AC_NOISE_LABELS                   # back-compat alias (default group's labels)


def _labels_for(groups):
    """Ordered label columns for the selected groups (group order preserved)."""
    labels = []
    for g in groups:
        if g not in LABEL_GROUPS:
            raise ValueError(f"unknown label group {g!r}; known: {sorted(LABEL_GROUPS)}")
        labels.extend(LABEL_GROUPS[g])
    return tuple(labels)


# ── provenance ──────────────────────────────────────────────────────────────
def _git_provenance(cwd=None):
    """Return ``{"commit": <sha or None>, "dirty": <bool or None>}``.

    Records which solver produced the labels — the trust anchor a surrogate needs
    to know it was trained against a specific, reproducible simulator revision.
    Degrades to ``None`` outside a git checkout (never raises)."""
    cwd = cwd or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() or None
        dirty = None
        if commit is not None:
            status = subprocess.run(
                ["git", "-C", cwd, "status", "--porcelain"],
                capture_output=True, text=True, timeout=5).stdout
            dirty = bool(status.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _topology_hash(config_dict):
    """Stable fingerprint of the circuit structure + base params (excludes the
    ``explore`` block, whose ranges are recorded separately in the manifest).

    Two datasets with the same ``topology_hash`` describe the same circuit; a
    surrogate is only valid within one topology, so this pins that identity."""
    structural = {k: v for k, v in config_dict.items() if k != "explore"}
    blob = json.dumps(structural, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")
    return "sha1:" + hashlib.sha1(blob).hexdigest()


def _resolve_corner(name):
    """(shift-dict-or-None, canonical-name) from a corner name.

    ``None`` / ``"typical"`` → nominal (no shift, matching explore's default);
    ``"slow"`` / ``"fast"`` → the :data:`circuitopt.corners.CORNERS` process shift."""
    if name is None or name == "typical":
        return None, "typical"
    if name not in CORNERS:
        raise ValueError(f"unknown corner {name!r}; known: {sorted(CORNERS)}")
    return CORNERS[name], name


# ── rows ────────────────────────────────────────────────────────────────────
def _finite_or_none(value):
    """JSON-safe scalar: non-finite floats (NaN/±inf) → ``None`` (valid JSON)."""
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _row(idx, var_values, metrics, extra=None, labels=LABELS):
    """One dataset record: design inputs, label outputs, and status flags.

    A DC-failed candidate (``metrics is None``) is kept with null labels and
    ``dc_converged=False`` — failures are labels, never dropped. ``extra`` holds the
    periodic-group features (transient / pss; ``None`` if those groups are off or
    the analysis failed); ``labels`` selects which columns to emit."""
    dc_converged = metrics is not None
    noise_evaluated = bool(metrics and metrics.get("_noise_evaluated", False))
    values = {}
    for name in labels:
        if name in _PERIODIC_LABELS:
            values[name] = _finite_or_none(extra.get(name)) if extra else None
        else:
            values[name] = _finite_or_none(metrics.get(name)) if metrics else None
    metrics_finite = dc_converged and all(values[n] is not None for n in labels)
    return {
        "idx": int(idx),
        "design": {k: v for k, v in var_values.items()},
        "metrics": values,
        "status": {
            "dc_converged": dc_converged,
            "noise_evaluated": noise_evaluated,
            "metrics_finite": metrics_finite,
        },
    }


def _transient_features(tr):
    """Stimulus-agnostic waveform features from a ``transient()`` result.

    Deliberately assumes nothing about the excitation (no step/settling
    semantics): peak-to-peak, mean, RMS, max slew rate, and final value are
    well-defined for *any* output waveform. Non-finite output → all-NaN."""
    t = np.asarray(tr["t"], float)
    out = np.asarray(tr["output"], float)
    if t.size < 2 or not np.isfinite(out).all():
        return {k: float("nan") for k in TRANSIENT_LABELS}
    dt = np.diff(t)
    slew = float(np.max(np.abs(np.diff(out) / dt))) if np.all(dt > 0) else float("nan")
    return {
        "out_pp": float(out.max() - out.min()),
        "out_mean": float(out.mean()),
        "out_rms": float(np.sqrt(np.mean(out * out))),
        "slew_rate": slew,
        "final_value": float(out[-1]),
    }


def _pss_features(res):
    """Periodic steady-state quality + orbit-output features from a ``pss_solve`` result.

    ``pss_converged`` (1.0/0.0) is the trust flag; ``pss_residual`` / ``pss_iters``
    describe the shooting solve; ``pss_out_pp`` / ``pss_out_mean`` summarize the
    converged orbit's weighted output (peak-to-peak ripple and DC level)."""
    out = np.asarray(res.get("output", []), float)
    if out.size and np.isfinite(out).all():
        pp, mean = float(out.max() - out.min()), float(out.mean())
    else:
        pp = mean = float("nan")
    return {
        "pss_converged": 1.0 if res.get("converged") else 0.0,
        "pss_residual": float(res.get("residual_norm", float("nan"))),
        "pss_iters": float(res.get("shooting_iters", float("nan"))),
        "pss_out_pp": pp,
        "pss_out_mean": mean,
    }


def _pac_features(res):
    """LPTV small-signal transfer features from a ``pac_solve`` result.

    ``pac_gain`` is |H| at the *lowest* analysis frequency — for a chopper, the
    demodulated baseband conversion gain; ``pac_gain_dB`` is its dB value and
    ``pac_bw_Hz`` the −3 dB corner within the PAC grid (NaN when the response
    never drops 3 dB inside the grid — an informative "band not clipped" label)."""
    gains = np.asarray(res.get("gains", ()), float)
    g0 = float(gains[0]) if gains.size else float("nan")
    return {
        "pac_gain": g0,
        "pac_gain_dB": float(res.get("Av_dc_dB", float("nan"))),
        "pac_bw_Hz": float(res.get("bw_Hz", float("nan"))),
    }


def _pnoise_features(res):
    """Band-integrated periodic noise from a ``pnoise_solve`` result.

    ``pnoise_out_uV`` integrates the folded output PSD over the config's analysis
    band; ``pnoise_irn_uV`` refers it to the input through the PAC sideband-0
    gain — the chopper AFE's headline metric."""
    return {
        "pnoise_out_uV": float(res.get("out_uV_band", float("nan"))),
        "pnoise_irn_uV": float(res.get("irn_uV_band", float("nan"))),
    }


def _candidate_spec(base_spec, binding, sizes, bias):
    """The base spec re-pointed at one candidate's sizes/bias/models.

    Models/topo/nf ride on ``binding`` (which carries the dataset-level silicon corner
    baked by ``at_corner``); this only re-points sizes/bias so the periodic context and
    the suite dispatcher see the candidate's design. ``spec.binding()`` inside the
    dispatcher then reconstructs the same per-device map from ``binding``'s fields."""
    return dataclasses.replace(
        base_spec, sizes=sizes, bias=bias, nf=binding.nf,
        model_types=binding.model_types, device_kwargs=binding.device_kwargs)


def _run_transient_features(base_spec, binding, periodic, sizes, bias, corner_shift):
    """Run the config's *validated* periodic transient for one candidate → features.

    The stimulus waveforms/grid come straight from :func:`build_periodic_context`
    (the same path ``run -a transient`` uses), so the transient is never fabricated
    — only the device sizes/bias vary. ``binding`` carries topo / nf / the per-device
    model map (with any silicon corner already baked in)."""
    from .analysis_dispatch import build_periodic_context   # lazy: avoids import cost
    spec = _candidate_spec(base_spec, binding, sizes, bias)
    ctx = build_periodic_context(spec, periodic)           # waveforms may read bias
    corner_kw = {} if corner_shift is None else {"corner": corner_shift}
    tr = transient(sizes, bias, ctx["tgrid"], binding=binding,
                   inputs=ctx["inputs"], node_inputs=ctx["node_inputs"],
                   current_inputs=ctx["current_inputs"],
                   signed_devices=ctx["signed_devices"], **corner_kw)
    return _transient_features(tr)


# Direct periodic label groups: run one validated periodic analysis straight on the
# candidate. Needs the config's ``periodic`` block.
_PERIODIC_RUNNERS = {"transient": _run_transient_features}

# Suite label groups: the PSS→PAC→PNoise chain, routed through
# ``run_analysis_suite`` so the config's ``analyses`` solver settings (integration
# method, shooting tolerances, ``time_domain``, drive, band) apply exactly as in
# ``run -a pss,pac,pnoise``; the PSS orbit is computed once and shared, and PNoise
# reuses the PAC gains for input referral.
_SUITE_GROUPS = ("pss", "pac", "pnoise")
_SUITE_FEATURES = {"pss": _pss_features, "pac": _pac_features,
                   "pnoise": _pnoise_features}


def _run_suite_features(base_spec, binding, groups, sizes, bias, corner_shift):
    """Run the validated PSS/PAC/PNoise chain once for one candidate → features.

    The dataset's per-candidate corner (``corner_shift``) overwrites any
    per-analysis corner from the config so every label in a row describes the same
    process point as the AC/noise labels. The PSS config is always threaded in
    (even when the ``pss`` group itself is off) because PAC/PNoise hang off it.
    ``binding`` carries topo / nf / the per-device model map; the candidate spec below
    re-points sizes/bias/models so the dispatcher's ``spec.binding()`` reconstructs it."""
    from .analysis_dispatch import run_analysis_suite       # lazy: avoids import cost
    spec = _candidate_spec(base_spec, binding, sizes, bias)
    base_analyses = spec.analyses or {}
    analyses = {}
    for name in _SUITE_GROUPS:
        if name in groups or name == "pss":
            cfg = dict(base_analyses.get(name) or {})
            cfg["corner"] = corner_shift
            analyses[name] = cfg
    results = run_analysis_suite(spec, analyses, selected=list(groups))
    feats = {}
    for g in groups:
        feats.update(_SUITE_FEATURES[g](results[g]))
    return feats


# ── structural / stimulus / corner design axes (extend the DEV.attr grammar) ──
# A design variable normally targets ``DEV.W/.L/.NF`` (sizes) or a bare bias key.
# Extra target kinds address elements outside sizes/bias:
#   * ``<CapName>.C`` (cap) / ``periodic.frequency`` (clock) — *structural*: live in
#     the topology / stimulus, so the candidate rebuilds its circuit from a patch.
#   * ``pvt0`` / ``pbeta0`` (corner) — the continuous global process shift, routed
#     into each candidate's ``evaluate(corner=...)``. Sampling these turns the
#     discrete corner into a continuous PVT axis (global process Monte-Carlo) so one
#     surrogate can interpolate to *any* process point, not just the named corners.
_CORNER_TARGETS = ("pvt0", "pbeta0")


def _target_kind(target):
    if target in _CORNER_TARGETS:
        return "corner"
    if target == "periodic.frequency":
        return "clock"
    if "." in target and target.rsplit(".", 1)[1] == "C":
        return "cap"
    if "." in target and target.rsplit(".", 1)[1] == "R":
        return "resistor"
    return "size_bias"                                     # DEV.W/.L/.NF or a bias key


def _var_is_structural(var):
    return any(_target_kind(t) in ("cap", "clock", "resistor") for t in var.targets)


def _var_is_corner(var):
    return any(_target_kind(t) == "corner" for t in var.targets)


def _corner_shift(corner_vars, var_values, base_shift):
    """Per-candidate process shift ``{pvt0, pbeta0}`` from sampled corner variables,
    starting from ``base_shift`` (the dataset-level ``--corner``, or 0)."""
    shift = {"pvt0": 0.0, "pbeta0": 0.0}
    if isinstance(base_shift, dict):
        shift.update(base_shift)
    for v in corner_vars:
        for target in v.targets:
            if _target_kind(target) == "corner":
                shift[target] = float(var_values[v.name])
    return shift


def _set_named_cap(config, name, value):
    for c in config.get("capacitors", []) or []:
        if isinstance(c, dict) and c.get("name") == name:
            c["C"] = float(value)
            return
        if isinstance(c, list) and len(c) == 3 and c[0] == name:   # [a, b, C] has no name
            break
    raise ValueError(f"capacitor variable {name!r}.C: no named capacitor {name!r} "
                     "in the config's 'capacitors' list")


def _set_named_resistor(config, name, value):
    for r in config.get("resistors", []) or []:
        if isinstance(r, dict) and r.get("name") == name:
            r["R"] = float(value)
            return
        if isinstance(r, list) and len(r) == 4 and r[0] == name:   # [name, a, b, R]
            r[3] = float(value)
            return
    raise ValueError(f"resistor variable {name!r}.R: no named resistor {name!r} "
                     "in the config's 'resistors' list")


def _patch_structural(config_dict, struct_vars, var_values):
    """Deep-copy the config and apply this candidate's cap/resistor-value / clock vars."""
    patched = copy.deepcopy(config_dict)
    for var in struct_vars:
        value = var_values[var.name]
        for target in var.targets:
            kind = _target_kind(target)
            if kind == "clock":
                per = patched.get("periodic")
                if not isinstance(per, dict):
                    raise ValueError("periodic.frequency variable needs a 'periodic' block")
                per["frequency"] = float(value)
            elif kind == "cap":
                _set_named_cap(patched, target.rsplit(".", 1)[0], value)
            elif kind == "resistor":
                _set_named_resistor(patched, target.rsplit(".", 1)[0], value)
    return patched


def split_variables(variables):
    """Partition explore variables into ``(size_bias, structural, corner)`` groups.

    Structural vars (``<Cap>.C`` / ``<Res>.R`` / ``periodic.frequency``) rebuild the
    circuit from a patched config; corner vars drive the PVT shift; the rest are the
    fast fixed-topology size/bias path."""
    corner_vars = [v for v in variables if _var_is_corner(v)]
    struct_vars = [v for v in variables if _var_is_structural(v)]
    size_vars = [v for v in variables
                 if not (_var_is_corner(v) or _var_is_structural(v))]
    return size_vars, struct_vars, corner_vars


def candidate_circuit(config_dict, topo, base_sizes, base_bias, nf,
                      size_vars, struct_vars, var_values):
    """``(topo, sizes, bias, nf)`` for one candidate design.

    Size/bias vars write into a copy of ``base_sizes``/``base_bias``; structural vars
    (cap/resistor/clock) rebuild the topology from a patched config. Shared by the
    dataset builder and the optimizer so both apply *every* variable kind identically."""
    size_names = {v.name for v in size_vars}
    size_vals = {k: v for k, v in var_values.items() if k in size_names}
    sizes, bias, cand_nf = apply_variables(size_vars, size_vals, base_sizes,
                                           base_bias, base_nf=nf)
    if struct_vars:
        patched = _patch_structural(config_dict, struct_vars, var_values)
        return circuit_from_dict(patched).topology, sizes, bias, cand_nf
    return topo, sizes, bias, cand_nf


def build_dataset(topo, base_sizes, base_bias, nf, cfg, *, n=200, seed=0,
                  method="lhs", corner=None, label_groups=DEFAULT_GROUPS,
                  seed_fn=None, progress=None, config_dict=None, config_path=None,
                  model_types=None, device_kwargs=None, workers=1):
    """Sample the design space and evaluate every candidate → dataset dict.

    Returns ``{"manifest": {...}, "rows": [row, ...]}`` where each ``row`` is a
    :func:`_row`. Unlike :func:`circuitopt.explore.explore` this applies **no** constraint
    or Pareto filtering and always evaluates noise, so the result is a complete,
    failure-retaining teacher dataset. ``seed_fn(sizes, bias) -> x0`` optionally
    provides a per-candidate DC seed (as for the AC-coupled AFE testbench).
    ``model_types`` / ``device_kwargs`` bind non-default per-device models (silicon);
    a SKY130 ``corner`` is routed onto those devices rather than an OTFT PVT shift."""
    if workers is None or workers < 1:
        raise ValueError("workers must be a positive integer")

    # A SKY130 corner is baked into each silicon device's card (a device kwarg); an
    # OTFT corner name/shift-map stays on the solver's continuous PVT path. One binding
    # then carries topo + the (silicon-baked) per-device model map to every candidate
    # runner, so no label group can drop the models and silently revert to OTFT
    # (bug #95). ``at_corner`` performs the same silicon-corner baking as the manifest
    # computation below; ``base_binding.corner`` is unused here — the dataset threads
    # its per-candidate PVT ``eff_shift`` to the solvers explicitly.
    device_kwargs, otft_corner = apply_silicon_corner(model_types, device_kwargs, corner)
    base_binding = CircuitBinding(topo=topo, model_types=model_types,
                                  device_kwargs=device_kwargs, nf=nf)
    si_corner = corner if (isinstance(corner, str) and corner in SKY130_CORNERS) else None
    shift, corner_name = _resolve_corner(otft_corner)
    if si_corner:
        corner_name = si_corner
    groups = tuple(label_groups)
    labels = _labels_for(groups)
    direct_groups = [g for g in groups if g in _PERIODIC_RUNNERS]       # transient
    suite_groups = [g for g in groups if g in _SUITE_GROUPS]            # pss/pac/pnoise
    periodic_groups = direct_groups + suite_groups
    # Structural axes (``<Cap>.C`` / ``periodic.frequency``) change the topology or
    # stimulus, so those candidates get the circuit rebuilt from a patched config;
    # pure size/bias axes keep the fast fixed-topology path.
    size_vars, struct_vars, corner_vars = split_variables(cfg.variables)
    size_names = {v.name for v in size_vars}
    if struct_vars and not config_dict:
        raise ValueError("capacitor/clock design variables need the source config "
                         "(config_dict) to rebuild the circuit per candidate")
    if periodic_groups and not (config_dict and config_dict.get("periodic")):
        raise ValueError(f"label group(s) {periodic_groups} require a 'periodic' "
                         "stimulus block in the config")
    # PAC/PNoise labels are only meaningful under the config's validated analysis
    # settings (drive, freqs, time_domain, band) — fail fast rather than label 5000
    # candidates with a fabricated setup.
    for g in ("pac", "pnoise"):
        if g in groups:
            acfg = (config_dict.get("analyses") or {}).get(g)
            if not isinstance(acfg, dict):
                raise ValueError(f"label group {g!r} requires an 'analyses.{g}' block "
                                 "(freqs / input_drive / solver settings) in the config")
            if ("input_drive" not in acfg and
                    len(config_dict.get("periodic", {}).get("inputs", {})) != 1):
                raise ValueError(f"'analyses.{g}' needs an explicit input_drive when "
                                 "the periodic block has multiple inputs")
    if corner_vars:                       # sampled corner ⇒ dataset spans PVT, not one point
        corner_name = "sampled"
    # Fixed-topology fast path builds the spec once; structural axes rebuild per candidate.
    base_spec = (circuit_from_dict(config_dict)
                 if (periodic_groups and not struct_vars) else None)

    samples = sample(cfg.variables, n, seed=seed, method=method)
    def evaluate_candidate(i, var_values):
        size_vals = {k: v for k, v in var_values.items() if k in size_names}
        sizes, bias, cand_nf = apply_variables(size_vars, size_vals,
                                               base_sizes, base_bias, base_nf=nf)
        eff_shift = _corner_shift(corner_vars, var_values, shift) if corner_vars else shift
        if struct_vars:
            patched = _patch_structural(config_dict, struct_vars, var_values)
            spec = circuit_from_dict(patched)
            cand_topo, periodic = spec.topology, patched.get("periodic")
        else:
            spec, cand_topo = base_spec, topo
            periodic = config_dict.get("periodic") if config_dict else None
        # This candidate's binding: the dataset binding re-pointed at its topo/nf.
        # model_types / device_kwargs (with any silicon corner baked) ride along, so
        # every label group runs against the same per-device models — no group can
        # drop them (bug #95).
        cand_binding = dataclasses.replace(base_binding, topo=cand_topo, nf=cand_nf)
        x0 = seed_fn(sizes, bias) if seed_fn is not None else None
        metrics = evaluate(cand_topo, sizes, bias, cand_nf, cfg.freqs, cfg.band,
                           binding=cand_binding, x0_guess=x0, corner=eff_shift,
                           require_noise=True)
        extra = {}
        if periodic_groups and metrics is not None and spec is not None:  # DC converged
            for g in direct_groups:                        # transient
                try:
                    extra.update(_PERIODIC_RUNNERS[g](spec, cand_binding, periodic,
                                                      sizes, bias, eff_shift))
                except Exception as exc:
                    diagnostics.note(f"dataset.{g}_eval_fail", exc)
            if suite_groups:                               # pss/pac/pnoise, one chain
                try:
                    extra.update(_run_suite_features(spec, cand_binding, suite_groups,
                                                     sizes, bias, eff_shift))
                except Exception as exc:
                    diagnostics.note(f"dataset.{'+'.join(suite_groups)}_eval_fail", exc)
        return _row(i, var_values, metrics, extra or None, labels)

    rows = [None] * n
    if workers == 1:
        for i, var_values in enumerate(samples):
            rows[i] = evaluate_candidate(i, var_values)
            if progress is not None:
                progress(i + 1, n)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_candidate, i, var_values): i
                       for i, var_values in enumerate(samples)}
            for completed, future in enumerate(as_completed(futures), start=1):
                rows[futures[future]] = future.result()
                if progress is not None:
                    progress(completed, n)
    rows = [row for row in rows if row is not None]

    counts = {
        "total": len(rows),
        "dc_converged": sum(r["status"]["dc_converged"] for r in rows),
        "metrics_finite": sum(r["status"]["metrics_finite"] for r in rows),
        "noise_evaluated": sum(r["status"]["noise_evaluated"] for r in rows),
    }
    freqs = np.asarray(cfg.freqs, float)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "solver": _git_provenance(),
        "config_path": config_path,
        "topology_hash": _topology_hash(config_dict) if config_dict else None,
        "pdk": get_default_model_type(),
        "models": dict(model_types or {}),     # per-device non-default bindings (silicon)
        "corner": corner_name,
        "sampling": {"n": int(n), "seed": int(seed), "method": method},
        "variables": {v.name: {"min": v.lo, "max": v.hi, "targets": list(v.targets),
                               "kind": ("corner" if _var_is_corner(v) else
                                        "structural" if _var_is_structural(v) else "size_bias")}
                      for v in cfg.variables},
        "band": [float(cfg.band[0]), float(cfg.band[1])],
        "freqs": {"n_points": int(freqs.size),
                  "f_min": float(freqs.min()), "f_max": float(freqs.max())},
        "label_groups": list(groups),
        "labels": list(labels),
        "counts": counts,
    }
    return {"manifest": manifest, "rows": rows}


# ── loading ─────────────────────────────────────────────────────────────────
def load_dataset_config(path):
    """Return ``(config_dict, topo, sizes, bias, nf, ExploreConfig)`` from a JSON.

    Like :func:`circuitopt.explore.load_explore_json` but also returns the raw config
    dict so the manifest can hash the topology."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "builtin_topology" in data:
        raise ValueError("builtin_topology configs are deprecated; use a full circuit JSON")
    cfg = parse_explore(data.get("explore"))
    spec = circuit_from_dict(data)
    return (data, spec.topology, dict(spec.sizes), dict(spec.bias), spec.nf, cfg)


# ── writers ─────────────────────────────────────────────────────────────────
def write_jsonl(dataset, path):
    """Write one design/label row per line (canonical, human-debuggable)."""
    with open(path, "w", encoding="utf-8") as f:
        for row in dataset["rows"]:
            f.write(json.dumps(row) + "\n")


def write_manifest(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset["manifest"], f, indent=2, sort_keys=True)
        f.write("\n")


def to_arrays(dataset):
    """Dense ``(X, Y, var_names, label_names, dc_converged, metrics_finite)``.

    ``X`` is ``(n, n_vars)`` design inputs (always populated — a design is defined
    even when its DC solve failed); ``Y`` is ``(n, n_labels)`` with ``NaN`` where a
    label is missing. Feed ``X``/``Y`` straight into a regressor after masking rows
    on ``metrics_finite``."""
    var_names = list(dataset["manifest"]["variables"].keys())
    label_names = list(dataset["manifest"]["labels"])
    rows = dataset["rows"]
    X = np.array([[float(r["design"][v]) for v in var_names] for r in rows],
                 dtype=float).reshape(len(rows), len(var_names))
    Y = np.array([[np.nan if r["metrics"][m] is None else float(r["metrics"][m])
                   for m in label_names] for r in rows],
                 dtype=float).reshape(len(rows), len(label_names))
    dc = np.array([r["status"]["dc_converged"] for r in rows], dtype=bool)
    fin = np.array([r["status"]["metrics_finite"] for r in rows], dtype=bool)
    return X, Y, var_names, label_names, dc, fin


def write_npz(dataset, path):
    X, Y, var_names, label_names, dc, fin = to_arrays(dataset)
    np.savez(path, X=X, Y=Y,
             var_names=np.array(var_names, dtype=object),
             label_names=np.array(label_names, dtype=object),
             dc_converged=dc, metrics_finite=fin,
             manifest=json.dumps(dataset["manifest"]))


def _flat_records(dataset):
    """One flat dict per row for columnar output: ``design_<var>`` inputs, bare
    label columns (the ML targets), and the three ``status`` booleans."""
    var_names = list(dataset["manifest"]["variables"])
    label_names = list(dataset["manifest"]["labels"])
    recs = []
    for r in dataset["rows"]:
        rec = {"idx": r["idx"]}
        for v in var_names:
            rec[f"design_{v}"] = r["design"][v]
        for m in label_names:
            rec[m] = r["metrics"][m]
        rec.update(r["status"])
        recs.append(rec)
    return recs


def write_parquet(dataset, path):
    """Write the flat table to Parquet. Requires the optional ``pyarrow`` dep."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:                      # optional dependency
        raise ImportError("Parquet output needs pyarrow; pip install pyarrow") from exc
    pq.write_table(pa.Table.from_pylist(_flat_records(dataset)), path)


def write_dataset(dataset, out_prefix, *, npz=True, parquet=False):
    """Write ``<out_prefix>.jsonl`` + ``.manifest.json`` (+ ``.npz`` / ``.parquet``).

    Returns a ``{kind: path}`` dict. ``parquet=True`` needs the optional ``pyarrow``
    dependency (raises a clear ``ImportError`` otherwise)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or ".", exist_ok=True)
    paths = {"jsonl": out_prefix + ".jsonl", "manifest": out_prefix + ".manifest.json"}
    write_jsonl(dataset, paths["jsonl"])
    write_manifest(dataset, paths["manifest"])
    if npz:
        paths["npz"] = out_prefix + ".npz"
        write_npz(dataset, paths["npz"])
    if parquet:
        paths["parquet"] = out_prefix + ".parquet"
        write_parquet(dataset, paths["parquet"])
    return paths


# ── CLI ─────────────────────────────────────────────────────────────────────
def run_from_config(config_path, *, n=200, seed=0, method="lhs", corner=None,
                    label_groups=DEFAULT_GROUPS, freqs=None, out=None, npz=True,
                    parquet=False, progress=None, workers=1):
    """Load a config, build the dataset, and (if ``out``) write it. Returns the dataset.

    ``freqs`` (an array) overrides the config's AC/noise analysis grid — e.g. to
    push the top decade up so ``bw_Hz`` isn't clipped at the grid ceiling."""
    config_dict, topo, sizes, bias, nf, cfg = load_dataset_config(config_path)
    model_types, device_kwargs = models_from_config(config_dict)
    if freqs is not None:
        cfg.freqs = np.asarray(freqs, float)
    dataset = build_dataset(topo, sizes, bias, nf, cfg, n=n, seed=seed, method=method,
                            corner=corner, label_groups=label_groups, progress=progress,
                            config_dict=config_dict, config_path=config_path,
                            model_types=model_types, device_kwargs=device_kwargs,
                            workers=workers)
    if out:
        dataset["_paths"] = write_dataset(dataset, out, npz=npz, parquet=parquet)
    return dataset


def _format_summary(dataset):
    m = dataset["manifest"]
    c = m["counts"]
    sv = m["solver"]
    commit = (sv.get("commit") or "?")[:9] + ("+dirty" if sv.get("dirty") else "")
    return (f"dataset schema {m['schema_version']}  corner={m['corner']}  "
            f"groups={'+'.join(m['label_groups'])}  solver={commit}\n"
            f"  samples: {c['total']}   dc_converged: {c['dc_converged']}   "
            f"labeled: {c['metrics_finite']}   noise: {c['noise_evaluated']}\n"
            f"  topology_hash: {m['topology_hash']}")


def add_cli_args(parser):
    """Register the dataset feature's own arguments on ``parser``.

    Single source of truth for both ``python -m circuitopt.dataset`` and the
    ``python -m circuitopt dataset`` subcommand — keeps the two CLI surfaces from
    drifting. Feature-only: subcommand-level mechanisms (e.g. ``--no-numba``)
    stay with their host parser, not here."""
    parser.add_argument("config", help="JSON file carrying an 'explore' block")
    parser.add_argument("-n", "--n", type=int, default=200,
                        help="Number of samples (default: 200)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel candidate workers (default: 1)")
    parser.add_argument("--method", choices=("lhs", "random"), default="lhs",
                        help="Sampling method (default: lhs)")
    parser.add_argument("--corner", default="typical",
                        help="Process corner: OTFT typical|slow|fast, or silicon "
                             "tt|ss|ff|sf|fs (SKY130) / nom|ss|ff (FreePDK45) "
                             "(default: typical)")
    parser.add_argument("--labels", default="ac_noise",
                        help="Comma list of label groups: ac_noise, transient, pss, "
                             "pac, pnoise (periodic groups need a 'periodic' block; "
                             "pac/pnoise also need their 'analyses' blocks; default: "
                             "ac_noise)")
    parser.add_argument("--freqs-start", type=float, default=-2.0,
                        help="AC grid start decade (log10 Hz) when --freqs-stop is "
                             "given")
    parser.add_argument("--freqs-stop", type=float, default=None,
                        help="Override AC grid top decade (log10 Hz), e.g. 4 = 10 kHz "
                             "(avoids bw_Hz clipping at the ceiling)")
    parser.add_argument("--freqs-num", type=int, default=101,
                        help="AC grid points when --freqs-stop is given")
    parser.add_argument("--out", default=None,
                        help="Output path prefix (writes "
                             "<prefix>.jsonl/.manifest.json/.npz)")
    parser.add_argument("--no-npz", action="store_true",
                        help="Skip the dense .npz output")
    parser.add_argument("--parquet", action="store_true",
                        help="Also write a .parquet table (needs the optional "
                             "pyarrow dep)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-sample progress")
    return parser


def run_cli(args):
    """Execute the dataset feature from parsed ``args``. Returns the dataset dict."""
    def progress(done, total):
        if not args.quiet:
            print(f"\r  evaluating {done}/{total}", end="", flush=True)

    if not args.quiet:
        print(f"Building dataset from {args.config}  "
              f"(n={args.n}, method={args.method}, corner={args.corner}, "
              f"labels={args.labels}, workers={args.workers})")
    groups = tuple(g.strip() for g in args.labels.split(",") if g.strip())
    freqs = (np.logspace(args.freqs_start, args.freqs_stop, args.freqs_num)
             if args.freqs_stop is not None else None)
    try:
        dataset = run_from_config(args.config, n=args.n, seed=args.seed,
                                  method=args.method, corner=args.corner,
                                  label_groups=groups, freqs=freqs, out=args.out,
                                  npz=not args.no_npz, parquet=args.parquet,
                                  progress=progress, workers=args.workers)
    except ImportError as exc:                  # e.g. --parquet without pyarrow
        raise SystemExit(str(exc))
    except ValueError as exc:                   # e.g. transient labels w/o periodic
        raise SystemExit(str(exc))
    if not args.quiet:
        print()
    print(_format_summary(dataset))
    if args.out:
        print("wrote " + ", ".join(dataset["_paths"].values()))
    return dataset


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Build a labeled surrogate dataset from an explore config.")
    add_cli_args(p)
    args = p.parse_args(argv)
    return run_cli(args)


if __name__ == "__main__":
    main()
