"""Precompiled topology metadata shared by circuit analyses.

`Topology` remains the declarative source of truth. `CompiledTopology` is the
runtime view: node names are resolved once into compact terminal tokens and
circuit elements (resistors, capacitors, current sources, VCCS) are expanded
into stamp-ready metadata. Solvers can then share the same terminal/index
convention instead of rebuilding it locally.
"""
from dataclasses import dataclass

import numpy as np


TERM_SOLVED = 0
TERM_INPUT = 1
TERM_RAIL = 2


def term_arrays(terms):
    """Split terminal tokens into parallel (kind, ref, value) int/float arrays.

    A token is ``(kind, ref_or_value)``: for a solved/input terminal
    (``kind`` in {``TERM_SOLVED``, ``TERM_INPUT``}) the second field is an
    integer index (``ref``); for a rail it is a float bias (``value``).  Both
    the raw-transient marshal and the OSDI transient marshal build the same
    stamp-ready arrays, so this helper lives with the topology tokens they
    share.
    """
    kind = np.empty(len(terms), dtype=np.int64)
    ref = np.empty(len(terms), dtype=np.int64)
    value = np.empty(len(terms), dtype=float)
    for pos, term in enumerate(terms):
        kind[pos] = int(term[0])
        if term[0] in (0, 1):
            ref[pos] = int(term[1])
            value[pos] = 0.0
        else:
            ref[pos] = 0
            value[pos] = float(term[1])
    return kind, ref, value


def index_array(vals):
    """Pack optional integer indices into an int64 array (``None`` -> ``-1``)."""
    return np.array([-1 if val is None else int(val) for val in vals],
                    dtype=np.int64)


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


@dataclass(frozen=True)
class VccsPlan:
    name: str
    p_node: str
    q_node: str
    cp_node: str
    cn_node: str
    p: tuple
    q: tuple
    cp: tuple
    cn: tuple
    pi: int | None
    qi: int | None
    cpi: int | None
    cni: int | None
    gm: float


@dataclass(frozen=True)
class VsourcePlan:
    """Ideal voltage source (true MNA). Adds branch-current unknown `bi = n + k` and a
    constraint row V_p - V_q = E. `e_const` is the constant EMF; `e_input_idx >= 0`
    selects a per-timestep transient waveform (time-varying E), else -1."""
    name: str
    p_node: str
    q_node: str
    p: tuple
    q: tuple
    pi: int | None
    qi: int | None
    bi: int
    e_const: float
    e_input_idx: int


@dataclass(frozen=True)
class VcvsPlan:
    """VCVS (voltage-controlled voltage source): V_p - V_q = mu*(V_cp - V_cn).
    Adds a branch-current unknown ``bi``; control nodes cp/cn are compiled to terminal
    tokens."""
    name: str
    p_node: str
    q_node: str
    cp_node: str
    cn_node: str
    p: tuple
    q: tuple
    cp: tuple
    cn: tuple
    pi: int | None
    qi: int | None
    cpi: int | None
    cni: int | None
    bi: int
    mu: float


@dataclass(frozen=True)
class CccsPlan:
    """CCCS (current-controlled current source): I_out = beta * I_ctrl.
    Controls on branch current of a voltage source / VCVS / CCVS. No new branch current."""
    name: str
    p_node: str
    q_node: str
    ctrl_name: str
    p: tuple
    q: tuple
    pi: int | None
    qi: int | None
    ctrl_bi: int
    beta: float


@dataclass(frozen=True)
class CcvsPlan:
    """CCVS (current-controlled voltage source): V_p - V_q = gamma * I_ctrl.
    Adds a branch-current unknown ``bi``; controls on branch current of another source."""
    name: str
    p_node: str
    q_node: str
    ctrl_name: str
    p: tuple
    q: tuple
    pi: int | None
    qi: int | None
    ctrl_bi: int
    bi: int
    gamma: float


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

        self.n_branches = topo.n_branches
        self.n_aug = self.n + self.n_branches
        self.devices = self._compile_devices()
        self.resistors = self._compile_resistors()
        self.capacitors = self._compile_capacitors()
        self.isources = self._compile_isources()
        self.vccs = self._compile_vccs()
        self.vsources = self._compile_vsources()
        self.vcvs = self._compile_vcvs()
        self.cccs = self._compile_cccs()
        self.ccvs = self._compile_ccvs()

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

    def _compile_vccs(self):
        out = []
        for name, p, q, cp, cn, gm in self.topo.vccs:
            pt = self.compile_term(p)
            qt = self.compile_term(q)
            cpt = self.compile_term(cp)
            cnt = self.compile_term(cn)
            out.append(VccsPlan(
                name=name, p_node=p, q_node=q, cp_node=cp, cn_node=cn,
                p=pt, q=qt, cp=cpt, cn=cnt,
                pi=self.solved_index(pt), qi=self.solved_index(qt),
                cpi=self.solved_index(cpt), cni=self.solved_index(cnt),
                gm=float(gm)))
        return tuple(out)

    def _compile_vsources(self):
        out = []
        for k, (name, p, q, value) in enumerate(self.topo.vsources):
            pt = self.compile_term(p)
            qt = self.compile_term(q)
            if isinstance(value, (int, float)):
                e_const, e_input_idx = float(value), -1
            elif value in self.input_index:
                e_const, e_input_idx = 0.0, self.input_index[value]
            else:                                   # unknown string -> 0 EMF (no DC bias)
                e_const, e_input_idx = 0.0, -1
            out.append(VsourcePlan(
                name=name, p_node=p, q_node=q, p=pt, q=qt,
                pi=self.solved_index(pt), qi=self.solved_index(qt),
                bi=self.n + k, e_const=e_const, e_input_idx=e_input_idx))
        return tuple(out)

    def _compile_vcvs(self):
        """Compile VCVS elements into VcvsPlan tuples."""
        out = []
        offset = len(self.topo.vsources)
        for k, (name, p, q, cp, cn, mu) in enumerate(self.topo.vcvs):
            pt = self.compile_term(p)
            qt = self.compile_term(q)
            cpt = self.compile_term(cp)
            cnt = self.compile_term(cn)
            out.append(VcvsPlan(
                name=name, p_node=p, q_node=q, cp_node=cp, cn_node=cn,
                p=pt, q=qt, cp=cpt, cn=cnt,
                pi=self.solved_index(pt), qi=self.solved_index(qt),
                cpi=self.solved_index(cpt), cni=self.solved_index(cnt),
                bi=self.n + offset + k, mu=float(mu)))
        return tuple(out)

    def _compile_cccs(self):
        """Compile CCCS elements into CccsPlan tuples."""
        out = []
        for name, p, q, ctrl_name, beta in self.topo.cccs:
            pt = self.compile_term(p)
            qt = self.compile_term(q)
            ctrl_bi = self.topo.vsource_index.get(ctrl_name)
            if ctrl_bi is None:
                raise ValueError(f"CCCS {name!r} references unknown branch source {ctrl_name!r}")
            out.append(CccsPlan(
                name=name, p_node=p, q_node=q, ctrl_name=ctrl_name,
                p=pt, q=qt, pi=self.solved_index(pt), qi=self.solved_index(qt),
                ctrl_bi=ctrl_bi, beta=float(beta)))
        return tuple(out)

    def _compile_ccvs(self):
        """Compile CCVS elements into CcvsPlan tuples."""
        out = []
        offset = len(self.topo.vsources) + len(self.topo.vcvs)
        for k, (name, p, q, ctrl_name, gamma) in enumerate(self.topo.ccvs):
            pt = self.compile_term(p)
            qt = self.compile_term(q)
            ctrl_bi = self.topo.vsource_index.get(ctrl_name)
            if ctrl_bi is None:
                raise ValueError(f"CCVS {name!r} references unknown branch source {ctrl_name!r}")
            out.append(CcvsPlan(
                name=name, p_node=p, q_node=q, ctrl_name=ctrl_name,
                p=pt, q=qt, pi=self.solved_index(pt), qi=self.solved_index(qt),
                ctrl_bi=ctrl_bi, bi=self.n + offset + k, gamma=float(gamma)))
        return tuple(out)

    def dc_residuals(self, x, Idfun, gmin):
        """KCL residual using compiled terminal tokens (length n_aug; gmin on node rows)."""
        res = np.zeros(self.n_aug)
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
        for item in self.vccs:
            vc = (self.term_value(item.cp, x)
                  - self.term_value(item.cn, x))
            i = item.gm * vc
            if item.pi is not None:
                res[item.pi] += i
            if item.qi is not None:
                res[item.qi] -= i
        for item in self.vsources:                  # ideal voltage source (true MNA)
            ibr = x[item.bi]                         # branch current p->q (unknown)
            if item.pi is not None:
                res[item.pi] -= ibr                 # current leaves p
            if item.qi is not None:
                res[item.qi] += ibr                 # and enters q
            res[item.bi] = (self.term_value(item.p, x)
                            - self.term_value(item.q, x) - item.e_const)
        for item in self.vcvs:                       # VCVS: V_p - V_q = mu*(V_cp - V_cn)
            ibr = x[item.bi]
            if item.pi is not None:
                res[item.pi] -= ibr
            if item.qi is not None:
                res[item.qi] += ibr
            res[item.bi] = (self.term_value(item.p, x) - self.term_value(item.q, x)
                            - item.mu * (self.term_value(item.cp, x)
                                         - self.term_value(item.cn, x)))
        for item in self.cccs:                       # CCCS: I_out = beta * I_ctrl
            I_out = item.beta * x[item.ctrl_bi]
            if item.pi is not None:
                res[item.pi] += I_out
            if item.qi is not None:
                res[item.qi] -= I_out
        for item in self.ccvs:                       # CCVS: V_p - V_q = gamma * I_ctrl
            ibr = x[item.bi]
            if item.pi is not None:
                res[item.pi] -= ibr
            if item.qi is not None:
                res[item.qi] += ibr
            res[item.bi] = (self.term_value(item.p, x) - self.term_value(item.q, x)
                            - item.gamma * x[item.ctrl_bi])
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

    def ac_vccs(self, drives=None):
        return tuple(
            (self.ac_term(item.p_node, drives),
             self.ac_term(item.q_node, drives),
             self.ac_term(item.cp_node, drives),
             self.ac_term(item.cn_node, drives),
             item.gm)
            for item in self.vccs
        )

    def ac_vsources(self, drives=None):
        """Small-signal MNA tokens for ideal voltage sources: (p_term, q_term, bi, E_ac).
        A DC bias source is an AC short (E_ac = 0); a stimulus EMF (keyed by source name
        in `drives`) becomes the constraint RHS."""
        out = []
        for item in self.vsources:
            e_ac = complex(drives[item.name]) if (drives and item.name in drives) else 0.0
            out.append((self.ac_term(item.p_node, drives),
                        self.ac_term(item.q_node, drives), item.bi, e_ac))
        return tuple(out)

    def ac_vcvs(self, drives=None):
        """Small-signal tokens for VCVS: (p_term, q_term, cp_term, cn_term, bi, mu).
        Constraint row enforces V_p - V_q = mu*(V_cp - V_cn)."""
        out = []
        for item in self.vcvs:
            out.append((self.ac_term(item.p_node, drives),
                        self.ac_term(item.q_node, drives),
                        self.ac_term(item.cp_node, drives),
                        self.ac_term(item.cn_node, drives),
                        item.bi, item.mu))
        return tuple(out)

    def ac_cccs(self, drives=None):
        """Small-signal tokens for CCCS: (p_term, q_term, ctrl_bi, beta).
        I_out = beta * I_ctrl injected p→q."""
        out = []
        for item in self.cccs:
            out.append((self.ac_term(item.p_node, drives),
                        self.ac_term(item.q_node, drives),
                        item.ctrl_bi, item.beta))
        return tuple(out)

    def ac_ccvs(self, drives=None):
        """Small-signal tokens for CCVS: (p_term, q_term, ctrl_bi, bi, gamma).
        Constraint row enforces V_p - V_q = gamma * I_ctrl."""
        out = []
        for item in self.ccvs:
            out.append((self.ac_term(item.p_node, drives),
                        self.ac_term(item.q_node, drives),
                        item.ctrl_bi, item.bi, item.gamma))
        return tuple(out)

    def output_sense(self, dtype=complex):
        sense = np.zeros(self.n_aug, dtype=dtype)
        for node, weight in self.output_weights.items():
            sense[self.idx[node]] = weight
        return sense
