"""Shared model-card ngspice deck-rendering primitives.

The full-circuit ngspice oracles — ``.tran`` (:mod:`circuitopt.ngspice_transient`)
and ``.ac`` / ``.noise`` / ``.op`` (:mod:`circuitopt.ngspice_ac`) — all render the
SAME circuit network into a SPICE deck: model-card ``.include`` lines, transistor
M-lines (with ``w``/``l``/``nf``/``delvto`` and bulk handling), R/C/I passives, and
the E/G/F/H controlled sources. This module owns that shared rendering so the two
backends cannot drift, and so per-polarity corner routing (nom/tt/ss/ff + the mixed
sf/fs) lives in exactly one place.

The line-emitting helpers are written to reproduce the transient renderer's output
byte-for-byte for the nom/ss/ff decks that predate the sf/fs work (locked by
``tests/test_pvt_machinery.py`` against golden fixtures), so ``.include`` order,
identifier mangling, ``%.17g`` formatting and the ``wrdata`` branch order are all
preserved.
"""
from __future__ import annotations

import os
import re
from typing import Any, Mapping

import numpy as np

from .device_factory import dev_mult, dev_nf
from .freepdk45_model import FREEPDK45_CORNERS, corner_card_dir, normalize_corner
from .ngspice_process import adapter_for_model_types
from .toolchain import pdk_root

_IDENT = re.compile(r"[^A-Za-z0-9_.$]")


def _ident(value: Any) -> str:
    text = _IDENT.sub("_", str(value))
    return text or "unnamed"


def _element(prefix: str, name: str) -> str:
    """SPICE element name: ensure it starts with the right type letter (M/R/C/...)."""
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


# ── corner / card resolution (single source of truth; handles sf/fs mixing) ────
def resolve_freepdk45_cards(model_types, device_kwargs, device_names):
    """Resolve the per-polarity model-card paths for a full-circuit ngspice deck.

    Every transistor must be bound to ``freepdk45.{nmos,pmos}`` or the explicit
    ``freepdk45_ngspice.{nmos,pmos}`` oracle aliases (mixed-PDK decks are
    rejected). All devices share one silicon corner name (one of
    :data:`~circuitopt.freepdk45_model.FREEPDK45_CORNERS`, matched case-insensitively
    via :func:`~circuitopt.freepdk45_model.normalize_corner`; an unknown name raises
    :class:`ValueError` — never a silent nominal deck), which is resolved to a
    ``models_<dir>`` DIRECTORY PER POLARITY: nom/tt/ss/ff give the same directory for
    both classes, while sf routes nmos->ss / pmos->ff and fs routes nmos->ff /
    pmos->ss. Each used polarity's card must exist on disk.

    Returns ``(process_corner, cards, used_polarities)`` where ``cards`` maps
    ``"nmos"``/``"pmos"`` to an absolute ``.inc`` path (only for polarities actually
    present). With no transistors returns ``(None, {}, set())`` so a purely passive /
    controlled-source testbench renders without any ``.include``.
    """
    model_types = dict(model_types or {})
    device_kwargs = {k: dict(v) for k, v in (device_kwargs or {}).items()}
    device_names = set(device_names)
    if not device_names:
        return None, {}, set()

    missing = sorted(device_names - set(model_types))
    if missing:
        raise NotImplementedError(
            "ngspice FreePDK45 full-circuit analysis requires every transistor to be "
            f"explicitly bound to freepdk45; missing model bindings: {', '.join(missing)}")
    prefixes = ("freepdk45.", "freepdk45_ngspice.")
    bad = {
        name: mt
        for name, mt in model_types.items()
        if name in device_names and not str(mt).startswith(prefixes)
    }
    if bad:
        raise NotImplementedError(
            "mixed FreePDK45/other-model ngspice analysis is not supported: "
            + ", ".join(f"{k}={v}" for k, v in sorted(bad.items())))

    # normalize_corner: case-insensitive, None/"" -> nom, and an UNKNOWN name raises
    # ValueError naming the valid set — same strictness as the grid path, so a corner
    # typo can never silently render a nominal deck.
    corners = {normalize_corner(device_kwargs.get(name, {}).get("corner", "nom"))
               for name in device_names}
    if len(corners) != 1:
        raise ValueError(
            f"one common FreePDK45 corner from {sorted(FREEPDK45_CORNERS)} is required, "
            f"got {sorted(corners)}")
    process_corner = next(iter(corners))

    used_polarities = {str(model_types[name]).rsplit(".", 1)[-1] for name in device_names}
    if not used_polarities <= {"nmos", "pmos"}:
        raise ValueError(f"unknown FreePDK45 transistor types: {sorted(used_polarities)}")

    fp45 = os.path.join(pdk_root(), "freepdk45")
    cards = {}
    for polarity in used_polarities:
        dev = "NMOS_VTG" if polarity == "nmos" else "PMOS_VTG"
        path = os.path.join(fp45, f"models_{corner_card_dir(polarity, process_corner)}",
                            f"{dev}.inc")
        if not os.path.isfile(path):
            raise RuntimeError(f"FreePDK45 model card not found: {path}; set PDK_ROOT")
        cards[polarity] = path
    return process_corner, cards, used_polarities


def resolve_common_temperature(device_kwargs, device_names,
                               override: float | None = None) -> float:
    """°C for ``.options temp=``. ``override`` (Kelvin) wins; else the single common
    per-device ``temperature`` (Kelvin) from device_kwargs; else 300.15 K (27 °C)."""
    if override is not None:
        return float(override) - 273.15
    temps = {float(device_kwargs.get(name, {}).get("temperature", 300.15))
             for name in device_names} or {300.15}
    if len(temps) != 1:
        raise ValueError("ngspice uses one circuit temperature; per-device temperatures differ")
    return next(iter(temps)) - 273.15


def include_lines(cards: Mapping[str, str]) -> list[str]:
    """``.include`` lines, one per used polarity, in stable (sorted) order."""
    return [f'.include "{cards[polarity]}"' for polarity in sorted(cards)]


def resolve_ngspice_preamble(model_types, device_kwargs, device_names):
    """Resolve the process adapter and model-card lines for a complete deck.

    FreePDK45 retains its historical flat-card renderer. Adapter-backed processes
    provide their own library-section preamble and simulator command arguments.
    Returns ``(adapter, corner, lines)``.
    """
    adapter = adapter_for_model_types(model_types, device_names)
    if adapter is not None:
        corner, lines = adapter.deck_preamble(model_types, device_kwargs, device_names)
        return adapter, corner, list(lines)
    corner, cards, _polarities = resolve_freepdk45_cards(
        model_types, device_kwargs, device_names)
    return None, corner, include_lines(cards)


# ── node mapping ───────────────────────────────────────────────────────────────
def build_node_map(topo, bias, node_inputs):
    """``({name: ngspice_node}, node_fn)`` for a topology.

    SPICE identifiers are case-insensitive, so collisions after mangling are rejected
    rather than silently shorting two nodes. ``node_fn(name)`` returns ``"0"`` for a
    ground rail (0 V, not overridden by a driven input) and the mapped ``n_...`` name
    otherwise — identical to the transient renderer's mapping."""
    node_inputs = dict(node_inputs or {})
    all_nodes = list(topo.solved) + list(topo.rails)
    node_map = {name: "n_" + _ident(name) for name in all_nodes}
    lowered = [value.lower() for value in node_map.values()]
    if len(lowered) != len(set(lowered)):
        raise ValueError("topology contains node names that collide in ngspice")

    def node(name: str) -> str:
        if name in node_inputs or name in topo.idx:
            return node_map[name]
        return "0" if _rail_value(topo, bias, name) == 0.0 else node_map[name]

    return node_map, node


# ── rail sources ───────────────────────────────────────────────────────────────
def render_rail_sources(topo, bias, node_inputs, node, *, ac=None):
    """Rail voltage sources ``V... n_rail 0 <value>``, skipping driven/0 V rails.

    ``ac`` optionally maps a rail name to ``(magnitude, phase_deg)`` to append an
    ``ac <mag> <phase>`` small-signal stimulus to that rail's source (AC path). With
    ``ac=None`` the output is byte-identical to the transient renderer. Returns
    ``(lines, branch_vectors)`` where each branch vector is ``("rail:<name>", src)``."""
    node_inputs = dict(node_inputs or {})
    ac = dict(ac or {})
    lines: list[str] = []
    branch_vectors: list[tuple[str, str]] = []
    for rail in topo.rails:
        if rail in node_inputs:
            continue
        value = _rail_value(topo, bias, rail)
        if value == 0.0:
            continue
        source = _element("V", "rail_" + rail)
        line = f"{source} {node(rail)} 0 {value:.17g}"
        if rail in ac:
            mag, phase = ac[rail]
            line += f" ac {float(mag):g} {float(phase):g}"
        lines.append(line)
        branch_vectors.append((f"rail:{rail}", source))
    return lines, branch_vectors


# ── transistors ────────────────────────────────────────────────────────────────
def render_devices(topo, sizes, bias, node_inputs, node, *, nf=None, model_types,
                   device_kwargs=None, mismatch=None, gate_nodes=None, adapter=None,
                   mult=None):
    """Transistor M/X lines plus any per-device bulk sources.

    ``gate_nodes`` maps a device to a driven gate node (transient PWL gate); absent,
    the gate uses ``node(g)``. ``mismatch`` maps a device to a ``delvto`` Vth offset
    (skipped when 0, so a zero-sigma deck is byte-identical). ``mult`` maps a device
    to an integer ``m=`` multiplicity (None/1 -> omitted, so a single-instance deck is
    byte-identical); ``m=N`` folds N identical parallel instances into one line.
    Returns ``(lines, branch_vectors)`` with each bulk source recorded as
    ``("bulk:<name>", src)``. Byte-identical to the transient renderer's device block."""
    node_inputs = dict(node_inputs or {})
    device_kwargs = device_kwargs or {}
    mismatch = {str(k): float(v) for k, v in (mismatch or {}).items()}
    gate_nodes = dict(gate_nodes or {})
    if mult is None:
        # Structural multiplicity travels on the topology (loader: device "M" field)
        # so no oracle signature needs to thread it; the kwarg stays as an override.
        mult = getattr(topo, "device_mult", None) or None
    lines: list[str] = []
    branch_vectors: list[tuple[str, str]] = []
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
        dvt = mismatch.get(name, 0.0)
        m = dev_mult(mult, name)
        if adapter is None:
            line = (f"{_element('M', name)} {node(d)} {gate} {node(s)} {bulk} {model} "
                    f"w={float(W):.17g}u l={float(L):.17g}u nf={dev_nf(nf, name)}")
            if m > 1:
                line += f" m={m}"
            if dvt != 0.0:
                line += f" delvto={dvt:.17g}"
        else:
            line = adapter.render_instance(
                name=_element("X", name), d=node(d), g=gate, s=node(s), b=bulk,
                model_type=model_type, width_um=float(W), length_um=float(L),
                nf=dev_nf(nf, name), mismatch=dvt, mult=m)
        lines.append(line)
    return lines, branch_vectors


# ── passives ───────────────────────────────────────────────────────────────────
def render_passives(topo, node):
    """R / load-cap / capacitor / current-source lines (byte-identical to transient)."""
    lines: list[str] = []
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
    return lines


# ── controlled sources (E/G/F/H) + ideal vsources ──────────────────────────────
def render_controlled(topo, node, *, tgrid=None, waveform_fn=None, ac=None):
    """Ideal vsources and E/G/F/H controlled sources.

    Transient path: pass ``tgrid`` + ``waveform_fn`` so a vsource whose value is a
    waveform key renders as a PWL source. AC path: pass ``ac`` mapping a vsource name
    to ``(magnitude, phase_deg)`` to stamp an ``ac`` stimulus on it. With both unset a
    constant vsource is emitted exactly as the transient renderer does. Returns
    ``(lines, branch_vectors)``; branch order is vsources, then VCVS, then CCVS (matching
    :class:`~circuitopt.topology.Topology.vsource_index`)."""
    ac = dict(ac or {})
    lines: list[str] = []
    branch_vectors: list[tuple[str, str]] = []
    controlled_names = {name: _element("V", name) for name, *_ in topo.vsources}
    controlled_names.update({name: _element("E", name) for name, *_ in topo.vcvs})
    controlled_names.update({name: _element("H", name) for name, *_ in topo.ccvs})

    for name, p, q, value in topo.vsources:
        source = controlled_names[name]
        if isinstance(value, str):
            if tgrid is None or waveform_fn is None:
                raise ValueError(
                    f"vsource {name!r} has a waveform value but this analysis has no time grid")
            lines.extend(_pwl_lines(source, node(p), node(q), tgrid, waveform_fn(value)))
        else:
            line = f"{source} {node(p)} {node(q)} {float(value):.17g}"
            if name in ac:
                mag, phase = ac[name]
                line += f" ac {float(mag):g} {float(phase):g}"
            lines.append(line)
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
    return lines, branch_vectors, controlled_names


def nodeset_line(topo, node_map, V0) -> str | None:
    """``.nodeset v(node)=val ...`` for the solved nodes, or ``None`` when V0 is absent."""
    if V0 is None:
        return None
    values = np.asarray(V0, float)
    if values.ndim != 1 or len(values) < topo.n:
        raise ValueError("V0 must contain at least one value per solved node")
    return ".nodeset " + " ".join(
        f"v({node_map[name]})={values[pos]:.17g}" for pos, name in enumerate(topo.solved))


def seed_vector(topo, x0_guess) -> "np.ndarray | None":
    """A ``.nodeset`` seed vector (length ``topo.n``) from a ``{node: V}`` dict or an
    array-like, in solved order. ``None`` -> no seed. Missing nodes default to 0."""
    if x0_guess is None:
        return None
    if isinstance(x0_guess, Mapping):
        return np.array([float(x0_guess.get(n, 0.0)) for n in topo.solved], float)
    vec = np.asarray(x0_guess, float).ravel()
    if len(vec) < topo.n:
        raise ValueError("x0_guess vector shorter than the number of solved nodes")
    return vec[:topo.n]
