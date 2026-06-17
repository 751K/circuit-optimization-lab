"""JSON circuit description loader.

The loader converts a compact JSON netlist/config into the objects used by the
solvers: Topology, device sizes, bias values, and optional NF data. It is kept
dependency-free so circuit definitions can live outside Python code.
"""
from dataclasses import dataclass
import json

from topology import Topology


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    topology: Topology
    sizes: dict
    bias: dict
    nf: dict | int | None = None


def _as_number(value, field):
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _load_devices(raw_devices):
    devices = []
    sizes = {}
    nf = {}
    if not isinstance(raw_devices, list) or not raw_devices:
        raise ValueError("devices must be a non-empty list")

    for i, item in enumerate(raw_devices):
        where = f"devices[{i}]"
        if isinstance(item, dict):
            try:
                name = item["name"]
                drain = item["drain"]
                gate = item["gate"]
                source = item["source"]
            except KeyError as exc:
                raise ValueError(f"{where} missing {exc.args[0]!r}") from exc
            if "W" in item and "L" in item:
                sizes[name] = (_as_number(item["W"], f"{where}.W"),
                               _as_number(item["L"], f"{where}.L"))
            if "NF" in item:
                nf[name] = int(item["NF"])
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            name, drain, gate, source = item
        else:
            raise ValueError(f"{where} must be a device object or [name, drain, gate, source]")
        devices.append((str(name), str(drain), str(gate), str(source)))
    return devices, sizes, nf


def _load_sizes(raw_sizes, embedded_sizes):
    sizes = dict(embedded_sizes)
    if raw_sizes is None:
        pass
    elif isinstance(raw_sizes, dict):
        for name, value in raw_sizes.items():
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"sizes[{name!r}] must be [W, L]")
            sizes[str(name)] = (_as_number(value[0], f"sizes[{name!r}][0]"),
                                _as_number(value[1], f"sizes[{name!r}][1]"))
    else:
        raise ValueError("sizes must be an object mapping device name to [W, L]")
    return sizes


def _load_load_caps(raw_caps):
    out = []
    for i, item in enumerate(raw_caps or []):
        if isinstance(item, dict):
            try:
                a = item["a"]
                b = item["b"]
                c = item["C"]
            except KeyError as exc:
                raise ValueError(f"load_caps[{i}] missing {exc.args[0]!r}") from exc
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            a, b, c = item
        else:
            raise ValueError(f"load_caps[{i}] must be {{a,b,C}} or [a,b,C]")
        out.append((str(a), str(b), _as_number(c, f"load_caps[{i}].C")))
    return out


def _validate_nodes(topo):
    known = set(topo.solved) | set(topo.rails)
    for name, d, g, s in topo.devices:
        for node in (d, g, s):
            if node not in known:
                raise ValueError(f"Device {name} references unknown node {node!r}")
    for a, b, _ in topo.load_caps:
        for node in (a, b):
            if node not in known:
                raise ValueError(f"load_caps references unknown node {node!r}")
    for node in topo.outputs:
        if node not in topo.idx:
            raise ValueError(f"Output node {node!r} must be a solved node")
    names = {name for name, *_ in topo.devices}
    for name in topo.input_drives:
        if name not in names:
            raise ValueError(f"input_drives references unknown device {name!r}")
    for name in topo.transient_inputs:
        if name not in names:
            raise ValueError(f"transient_inputs references unknown device {name!r}")


def circuit_from_dict(data):
    """Build CircuitSpec from a parsed JSON object."""
    if not isinstance(data, dict):
        raise ValueError("Circuit JSON root must be an object")
    name = str(data.get("name", "unnamed"))
    try:
        solved = data["solved"]
        rails = data["rails"]
        raw_devices = data["devices"]
    except KeyError as exc:
        raise ValueError(f"Circuit JSON missing required field {exc.args[0]!r}") from exc
    if not isinstance(rails, dict):
        raise ValueError("rails must be an object")

    devices, embedded_sizes, embedded_nf = _load_devices(raw_devices)
    sizes = _load_sizes(data.get("sizes"), embedded_sizes)
    missing_sizes = [dev for dev, *_ in devices if dev not in sizes]
    if missing_sizes:
        raise ValueError(f"Missing W/L sizes for devices: {', '.join(missing_sizes)}")

    nf = data.get("nf")
    if nf is None:
        nf = embedded_nf or None
    elif isinstance(nf, dict):
        merged = dict(embedded_nf)
        merged.update({str(k): int(v) for k, v in nf.items()})
        nf = merged
    else:
        nf = int(nf)

    topo = Topology(
        solved=[str(x) for x in solved],
        devices=devices,
        rails={str(k): v for k, v in rails.items()},
        outputs=tuple(str(x) for x in data.get("outputs", ())),
        input_drives={str(k): float(v) for k, v in data.get("input_drives", {}).items()},
        load_caps=_load_load_caps(data.get("load_caps")),
        dc_guesses=[{str(k): float(v) for k, v in guess.items()}
                    for guess in data.get("dc_guesses", [])],
        aliases={str(k): str(v) for k, v in data.get("aliases", {}).items()},
        transient_inputs={str(k): str(v) for k, v in data.get("transient_inputs", {}).items()},
    )
    _validate_nodes(topo)
    bias = {str(k): float(v) for k, v in data.get("bias", {}).items()}
    return CircuitSpec(name=name, topology=topo, sizes=sizes, bias=bias, nf=nf)


def load_circuit_json(path):
    """Load a circuit JSON file and return CircuitSpec."""
    with open(path, "r", encoding="utf-8") as f:
        return circuit_from_dict(json.load(f))
