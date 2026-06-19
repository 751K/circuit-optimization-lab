"""JSON circuit description loader.

The loader converts a compact JSON netlist/config into the objects used by the
solvers: Topology, device sizes, bias values, and optional NF data. It is kept
dependency-free so circuit definitions can live outside Python code.
"""
from dataclasses import dataclass
import json

try:
    from .topology import Topology
except ImportError:  # pragma: no cover - legacy direct module import
    from topology import Topology


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    topology: Topology
    sizes: dict
    bias: dict
    nf: dict | int | None = None
    periodic: dict | None = None
    analyses: dict | None = None


def _as_number(value, field):
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _load_devices(raw_devices):
    devices = []
    sizes = {}
    nf = {}
    if not isinstance(raw_devices, list):
        raise ValueError("devices must be a list")

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


def _load_elements(raw_items, label, term_keys, value_key, positive=False):
    """Parse two-terminal elements into (name, term0, term1, value) tuples.

    Object form: {"name", <term_keys[0]>, <term_keys[1]>, <value_key>}.
    Tuple form:  [name, term0, term1, value]."""
    out = []
    for i, item in enumerate(raw_items or []):
        where = f"{label}[{i}]"
        if isinstance(item, dict):
            try:
                name = item["name"]
                t0 = item[term_keys[0]]
                t1 = item[term_keys[1]]
                val = item[value_key]
            except KeyError as exc:
                raise ValueError(f"{where} missing {exc.args[0]!r}") from exc
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            name, t0, t1, val = item
        else:
            raise ValueError(f"{where} must be an object or "
                             f"[name, {term_keys[0]}, {term_keys[1]}, {value_key}]")
        value = _as_number(val, f"{where}.{value_key}")
        if positive and value <= 0:
            raise ValueError(f"{where}.{value_key} must be positive")
        out.append((str(name), str(t0), str(t1), value))
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
    for label, elements in (("Resistor", topo.resistors),
                            ("Capacitor", topo.capacitors),
                            ("Current source", topo.isources)):
        for name, x, y, _ in elements:
            for node in (x, y):
                if node not in known:
                    raise ValueError(f"{label} {name} references unknown node {node!r}")
    for node in topo.outputs:
        if node not in topo.idx:
            raise ValueError(f"Output node {node!r} must be a solved node")
    names = {name for name, *_ in topo.devices}
    for name in topo.input_drives:
        if name not in names:
            raise ValueError(f"input_drives references unknown device {name!r}")
    for node in topo.ac_drives:
        if node not in known:
            raise ValueError(f"ac_drives references unknown node {node!r}")
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
        ac_drives={str(k): float(v) for k, v in data.get("ac_drives", {}).items()},
        load_caps=_load_load_caps(data.get("load_caps")),
        dc_guesses=[{str(k): float(v) for k, v in guess.items()}
                    for guess in data.get("dc_guesses", [])],
        aliases={str(k): str(v) for k, v in data.get("aliases", {}).items()},
        transient_inputs={str(k): str(v) for k, v in data.get("transient_inputs", {}).items()},
        resistors=_load_elements(data.get("resistors"), "resistors", ("a", "b"), "R",
                                 positive=True),
        capacitors=_load_elements(data.get("capacitors"), "capacitors", ("a", "b"), "C",
                                  positive=True),
        isources=_load_elements(data.get("current_sources"), "current_sources",
                                ("nplus", "nminus"), "I"),
    )
    _validate_nodes(topo)
    bias = {str(k): float(v) for k, v in data.get("bias", {}).items()}
    periodic = data.get("periodic")
    if periodic is not None and not isinstance(periodic, dict):
        raise ValueError("periodic must be an object")
    analyses = data.get("analyses")
    if analyses is not None and not isinstance(analyses, dict):
        raise ValueError("analyses must be an object")
    return CircuitSpec(
        name=name, topology=topo, sizes=sizes, bias=bias, nf=nf,
        periodic=dict(periodic) if periodic is not None else None,
        analyses=dict(analyses) if analyses is not None else None,
    )


def load_circuit_json(path):
    """Load a circuit JSON file and return CircuitSpec."""
    with open(path, "r", encoding="utf-8") as f:
        return circuit_from_dict(json.load(f))
