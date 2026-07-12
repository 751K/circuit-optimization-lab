"""Full-circuit FreePDK45 transient simulation through ngspice.

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
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .device_factory import apply_silicon_corner, dev_nf
from .ngspice_char import _run_ngspice
from .toolchain import pdk_root


_IDENT = re.compile(r"[^A-Za-z0-9_.$]")


def _ident(value: Any) -> str:
    text = _IDENT.sub("_", str(value))
    return text or "unnamed"


def _element(prefix: str, name: str) -> str:
    value = _ident(name)
    return value if value[:1].lower() == prefix.lower() else prefix + value


def _rail_value(topo, bias, name: str) -> float:
    ref = topo.rails[name]
    if isinstance(ref, str):
        if ref not in bias:
            raise ValueError(f"rail {name!r} references missing bias value {ref!r}")
        return float(bias[ref])
    return float(ref)


def _pwl_lines(name: str, p: str, q: str, tgrid, values) -> list[str]:
    t = np.asarray(tgrid, float)
    v = np.asarray(values, float)
    if v.shape != t.shape:
        raise ValueError(f"waveform {name!r} shape {v.shape} != tgrid shape {t.shape}")
    if t[0] > 0.0:
        t = np.insert(t, 0, 0.0)
        v = np.insert(v, 0, v[0])
    pairs = [f"{tx:.17g} {vx:.17g}" for tx, vx in zip(t, v)]
    lines = [f"{name} {p} {q} PWL("]
    for pos in range(0, len(pairs), 6):
        suffix = ")" if pos + 6 >= len(pairs) else ""
        lines.append("+ " + " ".join(pairs[pos:pos + 6]) + suffix)
    return lines


def _current_input(item):
    if isinstance(item, Mapping):
        return str(item["p"]), str(item["q"]), str(item["input"])
    p, q, key = item
    return str(p), str(q), str(key)


@dataclass(frozen=True)
class RenderedTransient:
    netlist: str
    node_names: tuple[str, ...]
    branch_names: tuple[str, ...]


def render_freepdk45_transient_netlist(
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
) -> RenderedTransient:
    """Render a complete FreePDK45 ``.tran`` deck and its output-column map."""
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
        raise ValueError(f"FreePDK45 transient requires nom/ss/ff corner, got {corner!r}")
    device_kwargs = {k: dict(v) for k, v in (device_kwargs or {}).items()}

    device_names = {name for name, *_ in topo.devices}
    missing = sorted(device_names - set(model_types))
    if missing:
        raise NotImplementedError(
            "ngspice FreePDK45 transient requires every transistor to be explicitly "
            f"bound to freepdk45; missing model bindings: {', '.join(missing)}")
    bad = {name: mt for name, mt in model_types.items()
           if name in device_names and not str(mt).startswith("freepdk45.")}
    if bad:
        raise NotImplementedError(
            "mixed FreePDK45/other-model ngspice transient is not supported: "
            + ", ".join(f"{k}={v}" for k, v in sorted(bad.items())))

    corners = {str(device_kwargs.get(name, {}).get("corner", "nom"))
               for name in device_names}
    if not corners <= {"nom", "ss", "ff"} or len(corners) != 1:
        raise ValueError(f"one common FreePDK45 corner is required, got {sorted(corners)}")
    process_corner = next(iter(corners))
    temperatures = {float(device_kwargs.get(name, {}).get("temperature", 300.15))
                    for name in device_names}
    if len(temperatures) != 1:
        raise ValueError("ngspice uses one circuit temperature; per-device temperatures differ")
    temp_c = next(iter(temperatures)) - 273.15

    card_dir = os.path.join(pdk_root(), "freepdk45", f"models_{process_corner}")
    cards = {
        "nmos": os.path.join(card_dir, "NMOS_VTG.inc"),
        "pmos": os.path.join(card_dir, "PMOS_VTG.inc"),
    }
    used_polarities = {str(model_types[name]).rsplit(".", 1)[-1] for name in device_names}
    if not used_polarities <= {"nmos", "pmos"}:
        raise ValueError(f"unknown FreePDK45 transistor types: {sorted(used_polarities)}")
    for polarity in used_polarities:
        if not os.path.isfile(cards[polarity]):
            raise RuntimeError(f"FreePDK45 model card not found: {cards[polarity]}; set PDK_ROOT")

    # SPICE identifiers are case-insensitive. Keep one stable mapping and reject
    # collisions instead of silently shorting two topology nodes together.
    all_nodes = list(topo.solved) + list(topo.rails)
    node_map = {name: "n_" + _ident(name) for name in all_nodes}
    lowered = [value.lower() for value in node_map.values()]
    if len(lowered) != len(set(lowered)):
        raise ValueError("topology contains node names that collide in ngspice")

    def node(name: str) -> str:
        if name in node_inputs or name in topo.idx:
            return node_map[name]
        return "0" if _rail_value(topo, bias, name) == 0.0 else node_map[name]

    lines = ["* circuitopt FreePDK45 full-charge transient"]
    for polarity in sorted(used_polarities):
        lines.append(f'.include "{cards[polarity]}"')
    method = str(integration_method).lower()
    if method not in {"be", "gear2"}:
        raise ValueError(f"integration_method must be 'be' or 'gear2', got {method!r}")
    lines.append(f".options temp={temp_c:g} method=gear maxord={1 if method == 'be' else 2}")

    branch_vectors: list[tuple[str, str]] = []
    for rail in topo.rails:
        if rail in node_inputs:
            continue
        value = _rail_value(topo, bias, rail)
        if value == 0.0:
            continue
        source = _element("V", "rail_" + rail)
        lines.append(f"{source} {node(rail)} 0 {value:.17g}")
        branch_vectors.append((f"rail:{rail}", source))

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

    for name, d, g, s in topo.devices:
        model_type = str(model_types[name])
        polarity = model_type.rsplit(".", 1)[-1]
        model = "NMOS_VTG" if polarity == "nmos" else "PMOS_VTG"
        kwargs = device_kwargs.get(name, {})
        vb = float(kwargs.get("vb", 0.0))
        if vb == 0.0:
            bulk = "0"
        else:
            matching_rail = next((rail for rail in topo.rails
                                  if rail not in node_inputs
                                  and _rail_value(topo, bias, rail) == vb), None)
            if matching_rail is not None:
                bulk = node(matching_rail)
            else:
                bulk = "n_bulk_" + _ident(name)
                source = _element("V", "bulk_" + name)
                lines.append(f"{source} {bulk} 0 {vb:.17g}")
                branch_vectors.append((f"bulk:{name}", source))
        W, L = sizes[name]
        gate = gate_nodes.get(name, node(g))
        lines.append(
            f"{_element('M', name)} {node(d)} {gate} {node(s)} {bulk} {model} "
            f"w={float(W):.17g}u l={float(L):.17g}u nf={dev_nf(nf, name)}")

    for name, a, b, value in topo.resistors:
        lines.append(f"{_element('R', name)} {node(a)} {node(b)} {float(value):.17g}")
    cap_seen = set()
    for pos, (a, b, value) in enumerate(topo.load_caps):
        cname = _element("C", f"load_{pos}")
        lines.append(f"{cname} {node(a)} {node(b)} {float(value):.17g}")
        cap_seen.add(cname.lower())
    for name, a, b, value in topo.capacitors:
        cname = _element("C", name)
        if cname.lower() in cap_seen:
            raise ValueError(f"duplicate capacitor name after ngspice mapping: {name!r}")
        lines.append(f"{cname} {node(a)} {node(b)} {float(value):.17g}")
        cap_seen.add(cname.lower())
    for name, p, q, value in topo.isources:
        lines.append(f"{_element('I', name)} {node(p)} {node(q)} {float(value):.17g}")

    controlled_names = {
        name: _element("V", name) for name, *_ in topo.vsources
    }
    controlled_names.update({
        name: _element("E", name) for name, *_ in topo.vcvs
    })
    controlled_names.update({
        name: _element("H", name) for name, *_ in topo.ccvs
    })
    for name, p, q, value in topo.vsources:
        source = controlled_names[name]
        if isinstance(value, str):
            lines.extend(_pwl_lines(source, node(p), node(q), tgrid, waveform(value)))
        else:
            lines.append(f"{source} {node(p)} {node(q)} {float(value):.17g}")
        branch_vectors.append((name, source))
    for name, p, q, cp, cn, mu in topo.vcvs:
        source = controlled_names[name]
        lines.append(f"{source} {node(p)} {node(q)} {node(cp)} {node(cn)} {float(mu):.17g}")
        branch_vectors.append((name, source))
    for name, p, q, ctrl_name, gamma in topo.ccvs:
        source = controlled_names[name]
        ctrl = controlled_names.get(ctrl_name)
        if ctrl is None:
            raise ValueError(f"CCVS {name!r} references unavailable source {ctrl_name!r}")
        lines.append(f"{source} {node(p)} {node(q)} {ctrl} {float(gamma):.17g}")
        branch_vectors.append((name, source))
    for name, p, q, cp, cn, gm in topo.vccs:
        lines.append(
            f"{_element('G', name)} {node(p)} {node(q)} {node(cp)} {node(cn)} {float(gm):.17g}")
    for name, p, q, ctrl_name, beta in topo.cccs:
        ctrl = controlled_names.get(ctrl_name)
        if ctrl is None:
            raise ValueError(f"CCCS {name!r} references unavailable source {ctrl_name!r}")
        lines.append(f"{_element('F', name)} {node(p)} {node(q)} {ctrl} {float(beta):.17g}")

    for pos, item in enumerate(current_inputs):
        p, q, key = _current_input(item)
        source = _element("I", f"wave_{pos}")
        lines.extend(_pwl_lines(source, node(p), node(q), tgrid, waveform(key)))

    if V0 is not None:
        values = np.asarray(V0, float)
        if values.ndim != 1 or len(values) < topo.n:
            raise ValueError("V0 must contain at least one value per solved node")
        lines.append(".nodeset " + " ".join(
            f"v({node_map[name]})={values[pos]:.17g}"
            for pos, name in enumerate(topo.solved)))

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
    )


def transient_ngspice(
    sizes, bias, tgrid, *, topo, nf=None, V0=None, inputs=None,
    node_inputs=None, current_inputs=None, corner=None, model_types=None,
    device_kwargs=None, integration_method="be", max_step=None,
    timeout: float = 300.0,
):
    """Run a FreePDK45 full-charge transient and return circuitopt waveforms."""
    requested_t = np.asarray(tgrid, float)
    with tempfile.TemporaryDirectory(prefix="circuitopt-fp45-tran-") as td:
        output_path = os.path.join(td, "waveforms.dat")
        deck_path = os.path.join(td, "deck.cir")
        rendered = render_freepdk45_transient_netlist(
            sizes, bias, requested_t, topo=topo, output_path=output_path,
            nf=nf, V0=V0, inputs=inputs, node_inputs=node_inputs,
            current_inputs=current_inputs, corner=corner, model_types=model_types,
            device_kwargs=device_kwargs, integration_method=integration_method,
            max_step=max_step,
        )
        with open(deck_path, "w", encoding="ascii") as fh:
            fh.write(rendered.netlist)
        _run_ngspice(deck_path, output_path, timeout=timeout,
                     what="FreePDK45 full-circuit transient")
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
    }
    for alias, node_name in topo.aliases.items():
        if node_name in nodes:
            result[alias] = nodes[node_name]
    if "VOP" in nodes:
        result["vop"] = nodes["VOP"]
    if "VON" in nodes:
        result["von"] = nodes["VON"]
    return result
