"""JSON circuit description loader.

The loader converts a compact JSON netlist/config into the objects used by the
solvers: Topology, device sizes, bias values, and optional NF data. It is kept
dependency-free so circuit definitions can live outside Python code.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

from .topology import Topology

if TYPE_CHECKING:
    from pathlib import Path

    from .device_factory import CircuitBinding


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    topology: Topology
    sizes: dict
    bias: dict
    nf: dict | int | None = None
    periodic: dict | None = None
    analyses: dict | None = None
    model_types: dict | None = None      # device name -> model-registry key (e.g. "sky130.nmos")
    device_kwargs: dict | None = None    # device name -> extra ctor kwargs (vb, corner, ...)
    adc: dict | None = None              # optional ADC conversion workflow configuration

    def binding(self) -> CircuitBinding:
        """Bundle this spec's structure + process binding + default DC seed.

        Returns a :class:`circuitopt.device_factory.CircuitBinding` capturing ``topo``,
        ``model_types``, ``device_kwargs``, ``nf`` and the default DC seed (the
        first dict ``dc_guess``, matching the inline seed each analysis uses), so a
        caller can pass ``binding=`` instead of threading the whole cluster.
        """
        from .device_factory import CircuitBinding
        dc_seed = next((g for g in self.topology.dc_guesses if isinstance(g, dict)), None)
        return CircuitBinding(
            topo=self.topology, model_types=self.model_types,
            device_kwargs=self.device_kwargs, nf=self.nf, dc_seed=dc_seed,
        )


def _as_number(value, field):
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _load_devices(raw_devices):
    devices = []
    sizes = {}
    nf = {}
    mult = {}
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
            if "M" in item:
                m = int(item["M"])
                if m < 1:
                    raise ValueError(f"{where}.M must be >= 1, got {m}")
                mult[name] = m
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            name, drain, gate, source = item
        else:
            raise ValueError(f"{where} must be a device object or [name, drain, gate, source]")
        devices.append((str(name), str(drain), str(gate), str(source)))
    return devices, sizes, nf, mult


# Per-device constructor kwargs accepted in a ``models`` entry (besides ``type``).
# Restricted to a known set so a typo raises instead of being silently dropped by the
# device constructor's ``**kwargs``. ``vb``/``extract_w``/``temperature`` are floats,
# ``corner`` a SKY130 corner name, ``NF`` an integer finger count.
_MODEL_KWARGS = ("vb", "corner", "extract_w", "temperature", "NF")


def _load_models(raw_models, devices):
    """Parse the optional ``models`` block: per-device PDK model type + ctor kwargs.

    ``{"M1": {"type": "sky130.nmos", "vb": 1.8}}`` becomes
    ``({"M1": "sky130.nmos"}, {"M1": {"vb": 1.8}})``. ``type`` names a model-registry
    key (see :func:`circuitopt.device_model.register_pdk`); the remaining keys are forwarded
    to the device constructor. Devices absent from the block fall back to the default
    PDK, so the block is purely additive (an OTFT config omits it entirely)."""
    model_types, device_kwargs = {}, {}
    if raw_models is None:
        return model_types, device_kwargs
    if not isinstance(raw_models, dict):
        raise ValueError("models must be an object mapping device name to {type, ...kwargs}")
    dev_names = {name for name, *_ in devices}
    for name, spec in raw_models.items():
        if name not in dev_names:
            raise ValueError(f"models[{name!r}]: unknown device {name!r}")
        if not isinstance(spec, dict):
            raise ValueError(f"models[{name!r}] must be an object with a 'type' and/or kwargs")
        kwargs = {}
        for key, value in spec.items():
            if key == "type":
                model_types[str(name)] = str(value)
            elif key in _MODEL_KWARGS:
                kwargs[key] = int(value) if key == "NF" else value
            else:
                raise ValueError(f"models[{name!r}]: unknown key {key!r}; known: 'type', "
                                 + ", ".join(repr(k) for k in _MODEL_KWARGS))
        if kwargs:
            device_kwargs[str(name)] = kwargs
    return model_types, device_kwargs


def models_from_config(
    data: dict,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """``(model_types, device_kwargs)`` from a parsed circuit dict's ``models`` block.

    A thin accessor so CLI drivers (dataset / optimize / explore) can pull the model
    mapping without re-running the full :func:`circuit_from_dict` parse."""
    if not isinstance(data, dict) or "devices" not in data:
        raise ValueError("circuit dict must carry a 'devices' list")
    devices, _, _, _ = _load_devices(data["devices"])
    return _load_models(data.get("models"), devices)


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


def _load_vccs(raw_items):
    """Parse VCCS elements into (name, p, q, ctrl_p, ctrl_n, gm) tuples.

    Object form: {"name": "G1", "p": "OUT", "q": "GND",
                   "ctrl_p": "IN", "ctrl_n": "GND", "gm": 1e-4}
    Tuple form:  ["G1", "OUT", "GND", "IN", "GND", 1e-4]
    """
    out = []
    for i, item in enumerate(raw_items or []):
        where = f"vccs[{i}]"
        if isinstance(item, dict):
            try:
                name = item["name"]
                p = item["p"]
                q = item["q"]
                cp = item["ctrl_p"]
                cn = item["ctrl_n"]
                gm = item["gm"]
            except KeyError as exc:
                raise ValueError(f"{where} missing {exc.args[0]!r}") from exc
        elif isinstance(item, (list, tuple)) and len(item) == 6:
            name, p, q, cp, cn, gm = item
        else:
            raise ValueError(f"{where} must be an object or "
                             f"[name, p, q, ctrl_p, ctrl_n, gm]")
        out.append((str(name), str(p), str(q), str(cp), str(cn),
                    _as_number(gm, f"{where}.gm")))
    return out


def _load_vsources(raw_items):
    """Parse ideal voltage sources into (name, p, q, value) tuples.

    `value` is a constant EMF (number) or a transient input-waveform key (string).
    Object form: {"name": "V1", "p": "IN", "q": "GND", "value": 1.0}
    Tuple form:  ["V1", "IN", "GND", 1.0]
    """
    out = []
    for i, item in enumerate(raw_items or []):
        where = f"vsources[{i}]"
        if isinstance(item, dict):
            try:
                name = item["name"]
                p = item["p"]
                q = item["q"]
                value = item["value"]
            except KeyError as exc:
                raise ValueError(f"{where} missing {exc.args[0]!r}") from exc
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            name, p, q, value = item
        else:
            raise ValueError(f"{where} must be an object or [name, p, q, value]")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"{where}.value must be a number (EMF) or a waveform-key string")
        v = float(value) if isinstance(value, (int, float)) else str(value)
        out.append((str(name), str(p), str(q), v))
    return out


def _load_vcvs(raw_items):
    """Parse VCVS elements into (name, p, q, cp, cn, mu) tuples.

    Object form: {"name": "E1", "p": "OUT", "q": "GND",
                   "cp": "INP", "cn": "INN", "mu": 10.0}
    Tuple form:  ["E1", "OUT", "GND", "INP", "INN", 10.0]
    """
    out = []
    for i, item in enumerate(raw_items or []):
        where = f"vcvs[{i}]"
        if isinstance(item, dict):
            try:
                name = item["name"]
                p = item["p"]; q = item["q"]
                cp = item["cp"]; cn = item["cn"]
                mu = item["mu"]
            except KeyError as exc:
                raise ValueError(f"{where} missing {exc.args[0]!r}") from exc
        elif isinstance(item, (list, tuple)) and len(item) == 6:
            name, p, q, cp, cn, mu = item
        else:
            raise ValueError(f"{where} must be an object or [name, p, q, cp, cn, mu]")
        out.append((str(name), str(p), str(q), str(cp), str(cn),
                    _as_number(mu, f"{where}.mu")))
    return out


def _load_cccs(raw_items):
    """Parse CCCS elements into (name, p, q, ctrl_name, beta) tuples.

    Object form: {"name": "F1", "p": "OUT", "q": "GND",
                   "ctrl_name": "V1", "beta": 2.0}
    Tuple form:  ["F1", "OUT", "GND", "V1", 2.0]
    """
    out = []
    for i, item in enumerate(raw_items or []):
        where = f"cccs[{i}]"
        if isinstance(item, dict):
            try:
                name = item["name"]
                p = item["p"]; q = item["q"]
                ctrl_name = item["ctrl_name"]
                beta = item["beta"]
            except KeyError as exc:
                raise ValueError(f"{where} missing {exc.args[0]!r}") from exc
        elif isinstance(item, (list, tuple)) and len(item) == 5:
            name, p, q, ctrl_name, beta = item
        else:
            raise ValueError(f"{where} must be an object or [name, p, q, ctrl_name, beta]")
        out.append((str(name), str(p), str(q), str(ctrl_name),
                    _as_number(beta, f"{where}.beta")))
    return out


def _load_ccvs(raw_items):
    """Parse CCVS elements into (name, p, q, ctrl_name, gamma) tuples.

    Object form: {"name": "H1", "p": "OUT", "q": "GND",
                   "ctrl_name": "V1", "gamma": 100.0}
    Tuple form:  ["H1", "OUT", "GND", "V1", 100.0]
    """
    out = []
    for i, item in enumerate(raw_items or []):
        where = f"ccvs[{i}]"
        if isinstance(item, dict):
            try:
                name = item["name"]
                p = item["p"]; q = item["q"]
                ctrl_name = item["ctrl_name"]
                gamma = item["gamma"]
            except KeyError as exc:
                raise ValueError(f"{where} missing {exc.args[0]!r}") from exc
        elif isinstance(item, (list, tuple)) and len(item) == 5:
            name, p, q, ctrl_name, gamma = item
        else:
            raise ValueError(f"{where} must be an object or [name, p, q, ctrl_name, gamma]")
        out.append((str(name), str(p), str(q), str(ctrl_name),
                    _as_number(gamma, f"{where}.gamma")))
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
    for name, p, q, cp, cn, _ in topo.vccs:
        for node in (p, q, cp, cn):
            if node not in known:
                raise ValueError(f"VCCS {name} references unknown node {node!r}")
    for name, p, q, _ in topo.vsources:
        for node in (p, q):
            if node not in known:
                raise ValueError(f"Voltage source {name} references unknown node {node!r}")
        if p == q:
            raise ValueError(f"Voltage source {name} has identical terminals {p!r}")
        if p not in topo.idx and q not in topo.idx:
            raise ValueError(f"Voltage source {name} must connect at least one solved node "
                             f"(both {p!r} and {q!r} are rails)")
    for name, p, q, cp, cn, _ in topo.vcvs:
        for node in (p, q, cp, cn):
            if node not in known:
                raise ValueError(f"VCVS {name} references unknown node {node!r}")
        if p == q:
            raise ValueError(f"VCVS {name} has identical output terminals {p!r}")
        if p not in topo.idx and q not in topo.idx:
            raise ValueError(f"VCVS {name} must connect at least one solved node "
                             f"(both {p!r} and {q!r} are rails)")
    for name, p, q, ctrl_name, _ in topo.cccs:
        for node in (p, q):
            if node not in known:
                raise ValueError(f"CCCS {name} references unknown node {node!r}")
        if ctrl_name not in topo.vsource_index:
            raise ValueError(f"CCCS {name} references unknown branch source {ctrl_name!r}")
    for name, p, q, ctrl_name, _ in topo.ccvs:
        for node in (p, q):
            if node not in known:
                raise ValueError(f"CCVS {name} references unknown node {node!r}")
        if p == q:
            raise ValueError(f"CCVS {name} has identical output terminals {p!r}")
        if p not in topo.idx and q not in topo.idx:
            raise ValueError(f"CCVS {name} must connect at least one solved node "
                             f"(both {p!r} and {q!r} are rails)")
        if ctrl_name not in topo.vsource_index:
            raise ValueError(f"CCVS {name} references unknown branch source {ctrl_name!r}")
    for node in topo.outputs:
        if node not in topo.idx:
            raise ValueError(f"Output node {node!r} must be a solved node")
    names = {name for name, *_ in topo.devices}
    for name in topo.input_drives:
        if name not in names:
            raise ValueError(f"input_drives references unknown device {name!r}")
    for node in topo.ac_drives:
        if node not in known and node not in topo.vsource_index:
            raise ValueError(f"ac_drives references unknown node or source {node!r}")
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

    devices, embedded_sizes, embedded_nf, embedded_mult = _load_devices(raw_devices)
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
        vccs=_load_vccs(data.get("vccs")),
        vsources=_load_vsources(data.get("vsources")),
        vcvs=_load_vcvs(data.get("vcvs")),
        cccs=_load_cccs(data.get("cccs")),
        ccvs=_load_ccvs(data.get("ccvs")),
        device_mult=embedded_mult,
    )
    _validate_nodes(topo)
    bias = {str(k): float(v) for k, v in data.get("bias", {}).items()}
    periodic = data.get("periodic")
    if periodic is not None and not isinstance(periodic, dict):
        raise ValueError("periodic must be an object")
    analyses = data.get("analyses")
    if analyses is not None and not isinstance(analyses, dict):
        raise ValueError("analyses must be an object")
    adc = data.get("adc")
    if adc is not None and not isinstance(adc, dict):
        raise ValueError("adc must be an object")
    model_types, device_kwargs = _load_models(data.get("models"), devices)
    return CircuitSpec(
        name=name, topology=topo, sizes=sizes, bias=bias, nf=nf,
        periodic=dict(periodic) if periodic is not None else None,
        analyses=dict(analyses) if analyses is not None else None,
        model_types=model_types or None, device_kwargs=device_kwargs or None,
        adc=dict(adc) if adc is not None else None,
    )


def load_circuit_json(path: str | Path) -> CircuitSpec:
    """Load a circuit JSON file and return CircuitSpec."""
    with open(path, "r", encoding="utf-8") as f:
        return circuit_from_dict(json.load(f))
