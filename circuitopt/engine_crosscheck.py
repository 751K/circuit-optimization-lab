"""Run one signoff case through both engines and diff the trajectories.

The internal BSIM4 engine and the ngspice model-card path solve the same
netlist by different code.  Where they agree, both are probably right; where
they diverge in SHAPE -- not by microvolts but by a different answer -- one of
them has a defect, and nothing else in the toolchain will say so.

This is not hypothetical.  The native transient's DC start compiled the
topology without an input matrix, so every voltage source whose value was a
waveform key collapsed to 0 V EMF and the first step slammed the full source
swing through the coupling capacitors.  On the MDAC hold-phase bench that put
+0.42 V of common-mode drift on a floating virtual ground and failed every
residue case at every PVT point, in two different amplifier topologies, while
ngspice held the same trajectory flat.  The golden corpus stayed bit-exact
throughout, because no golden case exercised that path -- which is exactly how
a defect of that size stays hidden.  A standing cross-check catches the next
one on the first run.

The comparison is deliberately shape-first: peak absolute deviation per node
over the shared time grid, ranked worst-first, alongside the deviation once
settled.  Two solvers with different step controllers never agree to the last
digit -- during a 450 mV/ns slew a few picoseconds of step-placement
difference alone reads as tens of millivolts -- so the pair matters: a peak
that coincides with a fast transition is placement, while a divergence that
is still there at the end of the window is a disagreement about the circuit.

Nodes held by the stimulus are excluded by default (they reproduce their own
waveform in both engines, and a 20 ps clock edge on two step grids otherwise
dominates the ranking), as are the handful of samples that straddle an input
discontinuity, where interpolating one grid onto the other manufactures an
error the size of the step itself.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .analysis_dispatch import build_periodic_context, run_analysis_suite
from .circuit_loader import circuit_from_dict
from .compact_models.bsim4 import isolated_native_device_cache
from .device_model import list_pdks
from .signoff_campaign import (
    CampaignConfigurationError,
    _load_case_bases,
    load_campaign_json,
    prepare_case_dict,
)


class CrosscheckError(RuntimeError):
    """The two engines could not be compared on this case."""


# The ``<pdk>_ngspice`` model-card classes register on import of their own
# module, which nothing else pulls in.  A PDK gaining an oracle path adds one
# line here; the generic ``circuitopt.<pdk>_model`` fallback covers the case
# where the module is named after the pdk itself.
_ORACLE_MODULES = {
    "tsmc28hpcp": "circuitopt.tsmc28_model",
    "freepdk45": "circuitopt.freepdk45_model",
}


def ensure_oracle_registered(pdk: str) -> None:
    """Import whatever module registers ``<pdk>_ngspice``, if it is not already.

    Public because this lazy import is the *only* way the oracle classes reach
    the registry, and every caller that hand-rolled it (a bare
    ``import circuitopt.tsmc28_model  # noqa``) is one refactor away from
    silently severing its oracle path -- which has happened once already.
    Silent when the pdk has no oracle module; callers check
    :func:`list_pdks` afterwards and report their own error.
    """
    import importlib

    if f"{pdk}_ngspice" in set(list_pdks()):
        return
    for module in (_ORACLE_MODULES.get(pdk), f"circuitopt.{pdk}_model"):
        if module is None:
            continue
        try:
            importlib.import_module(module)
        except ImportError:
            continue
        if f"{pdk}_ngspice" in set(list_pdks()):
            return


def oracle_deck(deck: Mapping[str, Any]) -> dict[str, Any]:
    """Re-point a deck's models at the ``<pdk>_ngspice`` model-card classes.

    Portable netlists bind the engine-neutral pdk name; the ngspice path needs
    the adapter-backed registry entry.  Rewriting here keeps the committed JSON
    engine-neutral -- the same split that silently severed the TSMC28 oracle
    campaign once already."""
    out = deepcopy(dict(deck))
    rewritten = set()
    for model in (out.get("models") or {}).values():
        pdk = model.get("pdk")
        if pdk is None or str(pdk).endswith("_ngspice"):
            continue
        ensure_oracle_registered(str(pdk))
        available = set(list_pdks())
        alias = f"{pdk}_ngspice"
        if alias not in available:
            raise CrosscheckError(
                f"no ngspice model-card registry entry {alias!r} for pdk {pdk!r}; "
                f"registered: {', '.join(sorted(available))}")
        model["pdk"] = alias
        rewritten.add(alias)
    if not rewritten:
        raise CrosscheckError("deck binds no rewritable pdk models")
    return out


def _case_deck(manifest, case_name, corner, temperature_c, supply_v):
    config, manifest_path = load_campaign_json(manifest)
    cases = _load_case_bases(config, manifest_path)
    try:
        case = next(c for c in cases if c["name"] == case_name)
    except StopIteration:
        available = ", ".join(c["name"] for c in cases)
        raise CampaignConfigurationError(
            f"case {case_name!r} is not in the manifest; available: {available}"
        ) from None
    pvt = config["pvt"]
    return prepare_case_dict(
        case["base"], case["overrides"], corner=corner,
        temperature_c=temperature_c, supply_v=supply_v,
        nominal_supply_v=pvt["nominal_supply_v"],
        supply_bias_key=pvt["supply_bias_key"])


def driven_nodes(spec, context) -> set[str]:
    """Nodes carrying the stimulus rather than the circuit's answer.

    A node held by an ideal source at a waveform value reproduces that waveform
    in both engines by construction, so comparing it tests the testbench, not
    the solver.  Worse, it dominates the ranking: a 20 ps clock edge sampled on
    two different step grids reads as a full-swing disagreement."""
    keys = set(context["inputs"])
    driven = set(context["node_inputs"])
    for source in getattr(spec.topology, "vsources", ()) or ():
        _name, p, q, value = source
        if isinstance(value, str) and value in keys:
            driven.update({p, q})
    return driven


def _edge_mask(context, tgrid, *, relative_step=0.05, outlier_ratio=10.0):
    """Samples adjacent to an input discontinuity, which cannot be compared.

    Both engines are right at a step edge; they merely place sub-steps
    differently, and interpolating one onto the other's grid across the
    discontinuity manufactures an error the size of the step itself.  Masking
    the samples that straddle each jump removes the artifact without hiding a
    real divergence, which by definition persists away from the edge.

    A jump has to be large BOTH in absolute terms and relative to the
    waveform's typical step.  Size alone is not enough: a smooth ramp sampled
    on a coarse grid moves a large fraction of its span every sample and would
    mask the entire window, discarding exactly the sweep a slew comparison
    needs."""
    mask = np.zeros(tgrid.size, dtype=bool)
    for waveform in context["inputs"].values():
        values = np.asarray(waveform, float)
        if values.size != tgrid.size or values.size < 2:
            continue
        span = float(np.max(values) - np.min(values))
        if span <= 0.0:
            continue
        steps = np.abs(np.diff(values))
        typical = float(np.median(steps))
        threshold = max(relative_step * span, outlier_ratio * typical)
        jumps = np.nonzero(steps > threshold)[0]
        for index in jumps:
            mask[max(0, index - 1):min(tgrid.size, index + 3)] = True
    return mask


def _transient_config(spec):
    analyses = spec.analyses or {}
    cfg = analyses.get("transient")
    if cfg is None:
        raise CrosscheckError(
            "case has no transient analysis to cross-check "
            f"(analyses: {', '.join(sorted(analyses)) or 'none'})")
    return dict(cfg)


def crosscheck_case(
    manifest: str | Path,
    case_name: str,
    *,
    corner: str,
    temperature_c: float,
    supply_v: float,
    nodes: list[str] | None = None,
    include_driven: bool = False,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Run ``case_name`` at one PVT point on both engines and diff the nodes."""
    from .ngspice_transient import transient_ngspice

    deck = _case_deck(manifest, case_name, corner, temperature_c, supply_v)
    spec = circuit_from_dict(deck)
    cfg = _transient_config(spec)

    with isolated_native_device_cache():
        native = run_analysis_suite(spec, selected=["transient"])["transient"]

    # Rebuild the same stimulus the dispatcher fed the native run, on the grid
    # the native run actually reported, so the two solvers see one excitation.
    tgrid = np.asarray(native["t"], float)
    periodic = spec.periodic or {}
    merged = dict(periodic)
    merged.update(cfg.get("periodic") or {})
    if not merged:
        raise CrosscheckError("case has no periodic stimulus to replay")
    context = build_periodic_context(spec, merged, tgrid=tgrid)

    oracle_spec = circuit_from_dict(oracle_deck(deck))
    binding = oracle_spec.binding()
    base_kwargs = binding.device_kwargs or {}
    kelvin = float(temperature_c) + 273.15
    device_kwargs = {
        name: dict(base_kwargs.get(name, {}), temperature=kelvin)
        for name, *_ in oracle_spec.topology.devices
    }
    seed = oracle_spec.topology.dc_guesses[0]
    v0 = np.array([seed.get(node, 0.0) for node in oracle_spec.topology.solved])
    oracle = transient_ngspice(
        oracle_spec.sizes, oracle_spec.bias, tgrid, topo=oracle_spec.topology,
        nf=oracle_spec.nf, model_types=binding.model_types,
        device_kwargs=device_kwargs, corner=corner, V0=v0,
        inputs=context["inputs"], node_inputs=context["node_inputs"],
        timeout=timeout)

    shared = sorted(set(native["nodes"]) & set(oracle["nodes"]))
    stimulus = driven_nodes(spec, context)
    if nodes:
        missing = sorted(set(nodes) - set(shared))
        if missing:
            raise CrosscheckError(
                f"node(s) not reported by both engines: {', '.join(missing)}")
        shared = list(nodes)
    elif not include_driven:
        shared = [node for node in shared if node not in stimulus]
    if not shared:
        raise CrosscheckError("no solved nodes left to compare")

    edges = _edge_mask(context, tgrid)
    usable = ~edges
    if not usable.any():
        raise CrosscheckError("every sample straddles an input discontinuity")
    oracle_t = np.asarray(oracle["t"], float)
    rows = []
    for node in shared:
        a = np.asarray(native["nodes"][node], float)
        b = np.interp(tgrid, oracle_t, np.asarray(oracle["nodes"][node], float))
        delta = np.where(usable, np.abs(a - b), 0.0)
        index = int(np.argmax(delta))
        rows.append({
            "node": node,
            "max_abs_delta_v": float(delta[index]),
            "at_time_s": float(tgrid[index]),
            "native_v": float(a[index]),
            "oracle_v": float(b[index]),
            "final_delta_v": float(abs(a[-1] - b[-1])),
        })
    rows.sort(key=lambda row: -row["max_abs_delta_v"])
    return {
        "case": case_name,
        "pvt": {"corner": corner, "temperature_c": float(temperature_c),
                "supply_v": float(supply_v)},
        "samples": int(tgrid.size),
        "compared_samples": int(usable.sum()),
        "skipped_edge_samples": int(edges.sum()),
        "excluded_driven_nodes": sorted(stimulus & set(native["nodes"]))
        if not include_driven and not nodes else [],
        "nodes": rows,
        "worst_abs_delta_v": rows[0]["max_abs_delta_v"] if rows else 0.0,
        "worst_node": rows[0]["node"] if rows else None,
        "worst_peak_time_s": rows[0]["at_time_s"] if rows else 0.0,
        "worst_final_delta_v": max((r["final_delta_v"] for r in rows), default=0.0),
    }


def format_crosscheck(report: Mapping[str, Any], *, tolerance_v: float,
                      peak_tolerance_v: float | None = None,
                      top: int | None = None) -> str:
    """Render the diff.  ``tolerance_v`` gates the SETTLED deviation.

    The verdict deliberately keys on the settled column: that is where a real
    disagreement about the circuit lives, and it is what the defect this tool
    exists for looked like (a common mode that drifted to +0.42 V and stayed).
    A peak is reported alongside and flagged against a looser bound, because a
    peak during a fast transition mostly measures step placement."""
    if peak_tolerance_v is None:
        peak_tolerance_v = 10.0 * tolerance_v
    pvt = report["pvt"]
    lines = [
        f"case {report['case']} @ {pvt['corner']}/{pvt['temperature_c']:g}/"
        f"{pvt['supply_v']:g}   {report.get('compared_samples', report['samples'])}"
        f"/{report['samples']} samples compared"
        + (f" ({report['skipped_edge_samples']} straddle an input edge)"
           if report.get("skipped_edge_samples") else ""),
        f"  {'node':<12}  {'max |delta|':>12}  {'at':>10}  "
        f"{'native':>10}  {'oracle':>10}  {'final |d|':>10}",
    ]
    rows = report["nodes"]
    for row in (rows if top is None else rows[:top]):
        flag = ("  <-" if (row["final_delta_v"] > tolerance_v
                           or row["max_abs_delta_v"] > peak_tolerance_v) else "")
        lines.append(
            f"  {row['node']:<12}  {row['max_abs_delta_v'] * 1e3:>9.3f} mV  "
            f"{row['at_time_s'] * 1e9:>7.3f} ns  {row['native_v']:>10.5f}  "
            f"{row['oracle_v']:>10.5f}  {row['final_delta_v'] * 1e3:>7.3f} mV{flag}")
    if report.get("excluded_driven_nodes"):
        lines.append("  stimulus nodes excluded: "
                     + ", ".join(report["excluded_driven_nodes"]))
    settled = report["worst_final_delta_v"]
    verdict = "AGREE" if settled <= tolerance_v else "DIVERGE"
    lines.append(
        f"  worst settled {settled * 1e3:.3f} mV vs {tolerance_v * 1e3:.3f} mV "
        f"-> {verdict}   (worst peak {report['worst_abs_delta_v'] * 1e3:.3f} mV"
        + (f" on {report['worst_node']}" if report.get("worst_node") else "")
        + f" @ {report['worst_peak_time_s'] * 1e9:.3f} ns)")
    return "\n".join(lines)
