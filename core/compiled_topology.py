"""Precompiled topology metadata shared by circuit analyses.

`Topology` remains the declarative source of truth. `CompiledTopology` is the
runtime view: node names are resolved once into compact terminal tokens and
two-terminal elements are expanded into stamp-ready metadata. Solvers can then
share the same terminal/index convention instead of rebuilding it locally.
"""
from dataclasses import dataclass

import numpy as np


TERM_SOLVED = 0
TERM_INPUT = 1
TERM_RAIL = 2


@dataclass(frozen=True)
class DevicePlan:
    name: str
    d_node: str
    g_node: str
    s_node: str
    d: tuple
    g: tuple
    s: tuple
    di: int | None
    gi: int | None
    si: int | None


@dataclass(frozen=True)
class ResistorPlan:
    name: str
    a_node: str
    b_node: str
    a: tuple
    b: tuple
    ai: int | None
    bi: int | None
    value: float
    g: float


@dataclass(frozen=True)
class CapacitorPlan:
    name: str
    a_node: str
    b_node: str
    a: tuple
    b: tuple
    ai: int | None
    bi: int | None
    value: float


@dataclass(frozen=True)
class CurrentSourcePlan:
    name: str
    p_node: str
    q_node: str
    p: tuple
    q: tuple
    pi: int | None
    qi: int | None
    value: float


class CompiledTopology:
    """Bias/input-specific compiled view of a `Topology`.

    Parameters
    ----------
    topo
        Declarative topology object.
    bias
        Numeric bias dictionary used to resolve rails.
    input_keys
        Ordered transient input keys. Used only when compiling transient input
        terminals.
    node_inputs
        Mapping from node name to transient input key for time-domain driven
        nodes.
    transient_inputs
        If true, device gates listed in `topo.transient_inputs` are compiled as
        time-domain input terminals.
    """

    def __init__(self, topo, bias, input_keys=(), node_inputs=None,
                 transient_inputs=False):
        self.topo = topo
        self.solved = tuple(topo.solved)
        self.idx = topo.idx
        self.n = topo.n
        self.rails = topo.rail_values(bias)
        self.input_keys = tuple(input_keys)
        self.input_index = {key: i for i, key in enumerate(self.input_keys)}
        self.node_inputs = dict(node_inputs or {})
        self.use_transient_inputs = bool(transient_inputs)
        self.output_weights = topo.output_weights()

        for node, key in self.node_inputs.items():
            if key not in self.input_index:
                raise ValueError(f"node_inputs[{node!r}] references missing waveform {key!r}")

        self.devices = self._compile_devices()
        self.resistors = self._compile_resistors()
        self.capacitors = self._compile_capacitors()
        self.isources = self._compile_isources()

    # -- scalar terminal tokens -------------------------------------------------
    def _transient_gate_node(self, name, gate):
        if self.use_transient_inputs and name in self.topo.transient_inputs:
            key = self.topo.transient_inputs[name]
            if key not in self.input_index:
                raise ValueError(f"Missing transient input waveform {key!r} for device {name}")
            return ("input", key)
        return gate

    def compile_term(self, node):
        if isinstance(node, tuple) and node[0] == "input":
            return (TERM_INPUT, self.input_index[node[1]])
        if node in self.idx:
            return (TERM_SOLVED, self.idx[node])
        if node in self.node_inputs:
            return (TERM_INPUT, self.input_index[self.node_inputs[node]])
        if node in self.rails:
            return (TERM_RAIL, float(self.rails[node]))
        raise KeyError(f"Unknown topology node {node!r}")

    @staticmethod
    def solved_index(term):
        return term[1] if term[0] == TERM_SOLVED else None

    @staticmethod
    def term_value(term, vector, input_values=None):
        kind, ref = term
        if kind == TERM_SOLVED:
            return vector[ref]
        if kind == TERM_INPUT:
            if input_values is None:
                raise ValueError("input_values required for transient input terminal")
            return input_values[ref]
        return ref

    def _term_value_from_nodes(self, term, node_vals):
        kind, ref = term
        if kind == TERM_SOLVED:
            return node_vals[self.solved[ref]]
        if kind == TERM_INPUT:
            raise ValueError("Transient input terminal cannot be used in DC bias mapping")
        return ref

    # -- compiled scalar/DC metadata -------------------------------------------
    def _compile_devices(self):
        out = []
        for name, d, g, s in self.topo.devices:
            gt = self._transient_gate_node(name, g)
            dt = self.compile_term(d)
            gt = self.compile_term(gt)
            st = self.compile_term(s)
            out.append(DevicePlan(
                name=name, d_node=d, g_node=g, s_node=s, d=dt, g=gt, s=st,
                di=self.solved_index(dt), gi=self.solved_index(gt),
                si=self.solved_index(st)))
        return tuple(out)

    def _compile_resistors(self):
        out = []
        for name, a, b, value in self.topo.resistors:
            at = self.compile_term(a)
            bt = self.compile_term(b)
            out.append(ResistorPlan(
                name=name, a_node=a, b_node=b, a=at, b=bt,
                ai=self.solved_index(at), bi=self.solved_index(bt),
                value=float(value), g=1.0 / float(value)))
        return tuple(out)

    def _compile_capacitors(self):
        out = []
        for i, (a, b, value) in enumerate(self.topo.load_caps):
            at = self.compile_term(a)
            bt = self.compile_term(b)
            out.append(CapacitorPlan(
                name=f"load_cap_{i}", a_node=a, b_node=b, a=at, b=bt,
                ai=self.solved_index(at), bi=self.solved_index(bt),
                value=float(value)))
        for name, a, b, value in self.topo.capacitors:
            at = self.compile_term(a)
            bt = self.compile_term(b)
            out.append(CapacitorPlan(
                name=name, a_node=a, b_node=b, a=at, b=bt,
                ai=self.solved_index(at), bi=self.solved_index(bt),
                value=float(value)))
        return tuple(out)

    def _compile_isources(self):
        out = []
        for name, p, q, value in self.topo.isources:
            pt = self.compile_term(p)
            qt = self.compile_term(q)
            out.append(CurrentSourcePlan(
                name=name, p_node=p, q_node=q, p=pt, q=qt,
                pi=self.solved_index(pt), qi=self.solved_index(qt),
                value=float(value)))
        return tuple(out)

    def dc_residuals(self, x, Idfun, gmin):
        """KCL residual using compiled terminal tokens."""
        res = np.zeros(self.n)
        for dev in self.devices:
            i = Idfun(dev.name,
                      self.term_value(dev.s, x),
                      self.term_value(dev.d, x),
                      self.term_value(dev.g, x))
            if dev.di is not None:
                res[dev.di] += i
            if dev.si is not None:
                res[dev.si] -= i
        for item in self.resistors:
            i_ab = (self.term_value(item.a, x) - self.term_value(item.b, x)) * item.g
            if item.ai is not None:
                res[item.ai] -= i_ab
            if item.bi is not None:
                res[item.bi] += i_ab
        for item in self.isources:
            if item.pi is not None:
                res[item.pi] -= item.value
            if item.qi is not None:
                res[item.qi] += item.value
        for k in range(self.n):
            res[k] -= x[k] * gmin
        return res

    def bias_points(self, node_vals):
        """Per-device (Vs, Vd, Vg) from a solved operating-point dict."""
        return {
            dev.name: (
                self._term_value_from_nodes(dev.s, node_vals),
                self._term_value_from_nodes(dev.d, node_vals),
                self._term_value_from_nodes(dev.g, node_vals),
            )
            for dev in self.devices
        }

    # -- AC/noise terminal metadata --------------------------------------------
    def ac_term(self, node, drives=None):
        if node in self.idx:
            return ("n", self.idx[node])
        if drives and node in drives:
            return ("v", float(drives[node]))
        return ("v", 0.0)

    def ac_devices(self, drive=None, node_drives=None):
        drive = drive or {}
        node_drives = node_drives or {}

        def term(node, role, dev_name):
            if node in self.idx:
                return ("n", self.idx[node])
            if node in node_drives:
                return ("v", float(node_drives[node]))
            if role == "g":
                return ("v", float(drive.get(dev_name, 0.0)))
            return ("v", 0.0)

        return tuple(
            (name, term(d, "d", name), term(g, "g", name), term(s, "s", name))
            for name, d, g, s in self.topo.devices
        )

    def ac_capacitors(self, drives=None):
        return tuple(
            (self.ac_term(item.a_node, drives), self.ac_term(item.b_node, drives),
             item.value)
            for item in self.capacitors
        )

    def ac_resistors(self, drives=None):
        return tuple(
            (item.name, self.ac_term(item.a_node, drives),
             self.ac_term(item.b_node, drives), item.value, item.g)
            for item in self.resistors
        )

    def output_sense(self, dtype=complex):
        sense = np.zeros(self.n, dtype=dtype)
        for node, weight in self.output_weights.items():
            sense[self.idx[node]] = weight
        return sense
