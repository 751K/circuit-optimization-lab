"""Physical operating-point measurements derived from solved source currents."""
from __future__ import annotations

import numpy as np

from .run_contract import ModelEvaluationError, SimulationInvalid


def _rail_for_bulk(topo, bias, device, source_node):
    rail_values = topo.rail_values(bias)
    voltage = float(getattr(device, "vb", 0.0))
    matches = [
        name for name, value in rail_values.items()
        if np.isclose(float(value), voltage, rtol=0.0, atol=1e-12)
    ]
    explicit = (getattr(device, "binding", {}) or {}).get("bulk_rail")
    if explicit:
        if explicit not in rail_values:
            raise SimulationInvalid(
                "invalid_bulk_supply",
                f"explicit bulk rail {explicit!r} does not exist",
                analysis="dc_power",
            )
        if explicit not in matches:
            raise SimulationInvalid(
                "invalid_bulk_supply",
                f"bulk rail {explicit!r} voltage does not match vb={voltage:g} V",
                analysis="dc_power",
            )
        return explicit
    if source_node in matches:
        return source_node
    polarity = str(getattr(device, "POLARITY", "")).lower()
    preferred = (
        ("GND", "VSS", "VSSA", "VSSD")
        if polarity == "nmos"
        else ("VDD", "VCC", "VDDA", "VDDD")
    )
    named = [name for name in preferred if name in matches]
    if len(named) == 1:
        return named[0]
    return matches[0] if len(matches) == 1 else None


def source_power(topo, bias, node_vals, devices, branch_currents) -> dict:
    """DC power delivered by every explicit rail, from terminal/branch currents."""
    rail_values = {
        name: float(value) for name, value in topo.rail_values(bias).items()
    }
    delivered = {name: 0.0 for name in rail_values}

    def add(node, current_into_element):
        if node in delivered:
            delivered[node] += float(current_into_element)

    for name, drain, gate, source in topo.devices:
        dev = devices[name]
        vs = topo.node_v(source, node_vals, bias)
        vd = topo.node_v(drain, node_vals, bias)
        vg = topo.node_v(gate, node_vals, bias)
        if getattr(dev, "HAS_TERMINAL_LINEARIZATION", False):
            try:
                currents = np.asarray(
                    dev.get_terminal_currents(vs, vd, vg), float)
            except Exception as exc:
                raise ModelEvaluationError(
                    name, "DC terminal-current power evaluation", exc) from exc
            if currents.shape != (4,) or not np.all(np.isfinite(currents)):
                raise ValueError(f"{name}: invalid four-terminal DC currents")
            add(drain, currents[0])
            add(gate, currents[1])
            add(source, currents[2])
            bulk_rail = _rail_for_bulk(topo, bias, dev, source)
            if bulk_rail is None and currents[3] != 0.0:
                raise SimulationInvalid(
                    "unresolved_bulk_supply",
                    f"{name}: non-zero bulk current cannot be assigned to one "
                    "explicit rail; bind the bulk voltage to a unique rail",
                    analysis="dc_power",
                )
            if bulk_rail is not None:
                add(bulk_rail, currents[3])
        else:
            try:
                current = (
                    getattr(dev, "kcl_sign", 1.0)
                    * abs(dev.get_Idc(vs, vd, vg))
                )
            except Exception as exc:
                raise ModelEvaluationError(
                    name, "DC power evaluation", exc) from exc
            add(drain, -current)
            add(source, current)

    for _name, a, b, resistance in topo.resistors:
        current = (
            topo.node_v(a, node_vals, bias) - topo.node_v(b, node_vals, bias)
        ) / resistance
        add(a, current)
        add(b, -current)
    for _name, p, q, current in topo.isources:
        add(p, current)
        add(q, -current)
    for _name, p, q, cp, cn, gm in topo.vccs:
        current = gm * (
            topo.node_v(cp, node_vals, bias) - topo.node_v(cn, node_vals, bias)
        )
        add(p, -current)
        add(q, current)
    for name, p, q, _value in topo.vsources:
        current = float(branch_currents[name])
        add(p, current)
        add(q, -current)
    for name, p, q, *_rest in (*topo.vcvs, *topo.ccvs):
        current = float(branch_currents[name])
        add(p, current)
        add(q, -current)
    for _name, p, q, ctrl_name, beta in topo.cccs:
        current = beta * float(branch_currents[ctrl_name])
        add(p, -current)
        add(q, current)

    per_source = {
        name: rail_values[name] * current for name, current in delivered.items()
    }
    if not all(np.isfinite(value) for value in per_source.values()):
        raise ValueError("non-finite source power")
    return {
        "per_source_w": per_source,
        "source_currents_a": delivered,
        "source_voltages_v": rail_values,
        "total_w": float(sum(per_source.values())),
    }


def operating_regions(topo, bias, node_vals, devices, *, margin=0.0) -> dict:
    """Per-MOS BSIM operating region with signed PMOS-safe saturation checks."""
    rows = {}
    for name, drain, gate, source in topo.devices:
        dev = devices[name]
        getter = getattr(dev, "get_operating_point", None)
        if not callable(getter):
            rows[name] = {"status": "unsupported", "saturated": None}
            continue
        vs = topo.node_v(source, node_vals, bias)
        vd = topo.node_v(drain, node_vals, bias)
        vg = topo.node_v(gate, node_vals, bias)
        try:
            op = dict(getter(vs, vd, vg))
        except Exception as exc:
            raise ModelEvaluationError(
                name, "saturation operating-point evaluation", exc) from exc
        vds = float(op["vds"])
        vdsat = float(op["vdsat"])
        if not all(np.isfinite(value) for value in (vds, vdsat)):
            raise ValueError(f"{name}: non-finite saturation operating point")
        rows[name] = {
            "status": "valid",
            "saturated": bool(abs(vds) >= abs(vdsat) + float(margin)),
            "vds_v": vds,
            "vdsat_v": vdsat,
            "headroom_v": abs(vds) - abs(vdsat),
        }
    return rows
