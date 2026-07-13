"""Full-circuit model-card transient simulation through ngspice.

The fast :mod:`circuitopt.ngspice_device` adapter stores DC and small-signal
characterisation grids, not the four-terminal BSIM4 charge state required by a
large-signal transient.  This backend keeps ngspice as the FreePDK45 oracle: it
renders the complete :class:`~circuitopt.topology.Topology`, runs ``.tran`` with
the original model cards, and maps the resulting waveforms back to circuitopt's
standard transient result shape.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .device_factory import apply_silicon_corner
from .ngspice_char import _run_ngspice
from .ngspice_render import (
    _current_input, _element, _ident, _pwl_lines, build_node_map,
    nodeset_line, render_controlled, render_devices, render_passives,
    render_rail_sources, resolve_common_temperature, resolve_ngspice_preamble)


@dataclass(frozen=True)
class RenderedTransient:
    netlist: str
    node_names: tuple[str, ...]
    branch_names: tuple[str, ...]
    command_args: tuple[str, ...] = ()
    process: str = "freepdk45"


def render_ngspice_transient_netlist(
    sizes: Mapping[str, tuple[float, float]],
    bias: Mapping[str, float],
    tgrid: Sequence[float],
    *,
    topo,
    output_path: str,
    nf: int | Mapping[str, int] | None = None,
    V0=None,
    inputs: Mapping[str, Any] | None = None,
    node_inputs: Mapping[str, str] | None = None,
    current_inputs: Sequence[Any] | None = None,
    corner: str | Mapping[str, Any] | None = None,
    model_types: Mapping[str, str] | None = None,
    device_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    integration_method: str = "be",
    max_step: float | None = None,
    mismatch: Mapping[str, float] | None = None,
    extra_options: Mapping[str, Any] | None = None,
) -> RenderedTransient:
    """Render a complete model-card-backed ``.tran`` deck and its column map.

    ``mismatch`` maps a device name to a threshold-voltage offset in volts, emitted
    as the BSIM4 instance parameter ``delvto`` on that transistor's M-line. This is
    the injection hook for per-instance Vth mismatch Monte-Carlo (see
    :mod:`circuitopt.sar_mc`): ``delvto`` shifts the flat-band/Vth of that one
    instance without touching the shared model card, so paired devices stay
    independent. ``None`` (or an all-zero map) renders the byte-identical nominal
    deck — zero offsets are skipped rather than emitted as ``delvto=0`` so a
    zero-sigma trial reproduces the nominal netlist exactly.
    """
    tgrid = np.asarray(tgrid, float)
    if tgrid.ndim != 1 or len(tgrid) < 2 or tgrid[0] < 0.0 or np.any(np.diff(tgrid) <= 0.0):
        raise ValueError("tgrid must be one-dimensional, non-negative, and strictly increasing")
    inputs = {str(k): np.asarray(v, float) for k, v in (inputs or {}).items()}
    node_inputs = {str(k): str(v) for k, v in (node_inputs or {}).items()}
    current_inputs = tuple(current_inputs or ())
    model_types = dict(model_types or {})
    device_kwargs, solver_corner = apply_silicon_corner(
        model_types, device_kwargs, corner)
    if solver_corner not in (None, {}):
        raise ValueError(
            f"ngspice transient requires a supported silicon corner, got {corner!r}")
    device_kwargs = {k: dict(v) for k, v in (device_kwargs or {}).items()}
    mismatch = {str(k): float(v) for k, v in (mismatch or {}).items()}

    device_names = {name for name, *_ in topo.devices}
    unknown_mismatch = sorted(set(mismatch) - device_names)
    if unknown_mismatch:
        raise ValueError(
            f"mismatch references unknown devices: {', '.join(unknown_mismatch)}")

    # Per-polarity corner card resolution (nom/tt/ss/ff + mixed sf/fs) and the single
    # common circuit temperature — shared with the .ac/.noise/.op oracles.
    adapter, _corner, preamble = resolve_ngspice_preamble(
        model_types, device_kwargs, device_names)
    temp_c = resolve_common_temperature(device_kwargs, device_names)

    node_map, node = build_node_map(topo, bias, node_inputs)

    process = adapter.name if adapter is not None else "FreePDK45"
    lines = [f"* circuitopt {process} full-charge transient"]
    lines.extend(preamble)
    method = str(integration_method).lower()
    if method not in {"be", "gear2"}:
        raise ValueError(f"integration_method must be 'be' or 'gear2', got {method!r}")
    lines.append(f".options temp={temp_c:g} method=gear maxord={1 if method == 'be' else 2}")
    if extra_options:
        # e.g. {"reltol": 1e-5, "vntol": 1e-9} — tighter solver tolerances for
        # sub-0.1% settling measurements (ngspice's default reltol=1e-3 leaves a
        # ~100 uV numerical band on ~0.5 V nodes). None/{} renders byte-identically.
        opts = " ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                        for k, v in extra_options.items())
        lines.append(f".options {opts}")

    rail_lines, branch_vectors = render_rail_sources(topo, bias, node_inputs, node)
    lines.extend(rail_lines)

    def waveform(key: str):
        if key not in inputs:
            raise ValueError(f"transient source references missing input waveform {key!r}")
        values = inputs[key]
        if values.shape != tgrid.shape:
            raise ValueError(f"input waveform {key!r} length differs from tgrid")
        return values

    for driven_node, key in node_inputs.items():
        if driven_node not in node_map:
            raise ValueError(f"node_inputs references unknown node {driven_node!r}")
        source = _element("V", "node_" + driven_node)
        lines.extend(_pwl_lines(source, node_map[driven_node], "0", tgrid, waveform(key)))
        branch_vectors.append((f"node:{driven_node}", source))

    gate_nodes = {}
    for name, *_ in topo.devices:
        key = topo.transient_inputs.get(name)
        if key is None:
            continue
        gate_nodes[name] = "n_gate_" + _ident(name)
        source = _element("V", "gate_" + name)
        lines.extend(_pwl_lines(source, gate_nodes[name], "0", tgrid, waveform(str(key))))
        branch_vectors.append((f"gate:{name}", source))

    dev_lines, dev_branches = render_devices(
        topo, sizes, bias, node_inputs, node, nf=nf, model_types=model_types,
        device_kwargs=device_kwargs, mismatch=mismatch, gate_nodes=gate_nodes,
        adapter=adapter)
    lines.extend(dev_lines)
    branch_vectors.extend(dev_branches)

    lines.extend(render_passives(topo, node))

    ctrl_lines, ctrl_branches, _names = render_controlled(
        topo, node, tgrid=tgrid, waveform_fn=waveform)
    lines.extend(ctrl_lines)
    branch_vectors.extend(ctrl_branches)

    for pos, item in enumerate(current_inputs):
        p, q, key = _current_input(item)
        source = _element("I", f"wave_{pos}")
        lines.extend(_pwl_lines(source, node(p), node(q), tgrid, waveform(key)))

    nodeset = nodeset_line(topo, node_map, V0)
    if nodeset is not None:
        lines.append(nodeset)

    print_step = float(np.min(np.diff(tgrid)))
    tmax = print_step if max_step is None else float(max_step)
    if tmax <= 0.0:
        raise ValueError("max_step must be positive")
    vectors = [f"v({node_map[name]})" for name in topo.solved]
    vectors.extend(f"i({source})" for _, source in branch_vectors)
    lines.extend([
        ".control",
        "set wr_singlescale",
        "set wr_vecnames",
        f"tran {print_step:.17g} {tgrid[-1]:.17g} 0 {tmax:.17g}",
        f"wrdata {output_path} " + " ".join(vectors),
        ".endc",
        ".end",
    ])
    return RenderedTransient(
        netlist="\n".join(lines) + "\n",
        node_names=tuple(topo.solved),
        branch_names=tuple(name for name, _ in branch_vectors),
        command_args=adapter.command_args if adapter is not None else (),
        process=process,
    )


def render_freepdk45_transient_netlist(*args, **kwargs) -> RenderedTransient:
    """Compatibility name for the generic ngspice transient renderer."""
    return render_ngspice_transient_netlist(*args, **kwargs)


def transient_ngspice(
    sizes, bias, tgrid, *, topo, nf=None, V0=None, inputs=None,
    node_inputs=None, current_inputs=None, corner=None, model_types=None,
    device_kwargs=None, integration_method="be", max_step=None,
    mismatch=None, extra_options=None, timeout: float = 300.0,
):
    """Run a FreePDK45 full-charge transient and return circuitopt waveforms.

    ``mismatch`` is threaded straight to
    :func:`render_freepdk45_transient_netlist` as per-device ``delvto`` offsets.
    """
    requested_t = np.asarray(tgrid, float)
    with tempfile.TemporaryDirectory(prefix="circuitopt-fp45-tran-") as td:
        output_path = os.path.join(td, "waveforms.dat")
        deck_path = os.path.join(td, "deck.cir")
        rendered = render_ngspice_transient_netlist(
            sizes, bias, requested_t, topo=topo, output_path=output_path,
            nf=nf, V0=V0, inputs=inputs, node_inputs=node_inputs,
            current_inputs=current_inputs, corner=corner, model_types=model_types,
            device_kwargs=device_kwargs, integration_method=integration_method,
            max_step=max_step, mismatch=mismatch, extra_options=extra_options,
        )
        with open(deck_path, "w", encoding="ascii") as fh:
            fh.write(rendered.netlist)
        _run_ngspice(deck_path, output_path, timeout=timeout,
                     what=f"{rendered.process} full-circuit transient",
                     extra_args=rendered.command_args)
        raw = np.loadtxt(output_path, skiprows=1, ndmin=2)

    expected_cols = 1 + len(rendered.node_names) + len(rendered.branch_names)
    if raw.shape[1] != expected_cols:
        raise RuntimeError(
            f"ngspice transient returned {raw.shape[1]} columns, expected {expected_cols}")
    order = np.argsort(raw[:, 0], kind="stable")
    raw = raw[order]
    # Keep the final sample at duplicate breakpoints.
    _, reverse_pos = np.unique(raw[::-1, 0], return_index=True)
    keep = np.sort(len(raw) - 1 - reverse_pos)
    raw = raw[keep]
    sim_t = raw[:, 0]
    if requested_t[0] < sim_t[0] - 1e-18 or requested_t[-1] > sim_t[-1] + 1e-15:
        raise RuntimeError(
            f"ngspice transient range [{sim_t[0]}, {sim_t[-1]}] does not cover tgrid")

    pos = 1
    nodes = {}
    for name in rendered.node_names:
        nodes[name] = np.interp(requested_t, sim_t, raw[:, pos])
        pos += 1
    branch_currents = {}
    for name in rendered.branch_names:
        branch_currents[name] = np.interp(requested_t, sim_t, raw[:, pos])
        pos += 1
    if topo.outputs:
        output = sum(nodes[name] * weight for name, weight in topo.output_weights().items())
    else:
        output = nodes[topo.solved[0]]
    result = {
        "t": requested_t,
        "nodes": nodes,
        "output": output,
        "vout": output,
        "branch_currents": branch_currents,
        "nfail": 0,
        "backend": "ngspice",
        "ngspice_transient": True,
        "process": rendered.process,
    }
    for alias, node_name in topo.aliases.items():
        if node_name in nodes:
            result[alias] = nodes[node_name]
    if "VOP" in nodes:
        result["vop"] = nodes["VOP"]
    if "VON" in nodes:
        result["von"] = nodes["VON"]
    return result
