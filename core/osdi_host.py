"""OSDI 0.4 host — load an OpenVAF-compiled ``.osdi`` and evaluate one device.

Binds the OSDI 0.4 ABI (``openvaf/osdi/header/osdi_0_4.h`` in the OpenVAF-Reloaded
tree) via :mod:`ctypes` so the solver can call a compiled Verilog-A compact model
(e.g. BSIM4) in-process. This is the bridge behind a silicon PDK: OpenVAF compiles
the standard model ``.va`` → native ``.osdi``; this module loads it and exposes a
single-device evaluator; :mod:`core.osdi_device` adapts that onto
:class:`~core.device_model.TransistorModel`.

**This slice** loads a ``.osdi``, self-checks the struct binding against the
module's exported ``OSDI_DESCRIPTOR_SIZE``, and enumerates each descriptor's nodes,
parameters, and op-vars. Device evaluation (setup + eval + read gm/gds/caps/noise)
is built on top of the structures defined here.

Run standalone to introspect a model::

    python -m core.osdi_host /path/to/bsim4.osdi
"""
from __future__ import annotations

import ctypes as C
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import diagnostics
from .numba_kernels import _osdi_op_solve_impl, _osdi_schur_impl

# ── OSDI 0.4 flag constants (osdi_0_4.h) ─────────────────────────────────
OSDI_VERSION_MAJOR_CURR = 0
OSDI_VERSION_MINOR_CURR = 4

PARA_TY_MASK = 3
PARA_TY_REAL, PARA_TY_INT, PARA_TY_STR = 0, 1, 2
PARA_KIND_MASK = 3 << 30
PARA_KIND_MODEL = 0 << 30
PARA_KIND_INST = 1 << 30
PARA_KIND_OPVAR = 2 << 30

ACCESS_FLAG_READ = 0
ACCESS_FLAG_SET = 1
ACCESS_FLAG_INSTANCE = 4

# eval() `flags` selectors
CALC_RESIST_RESIDUAL = 1
CALC_REACT_RESIDUAL = 2
CALC_RESIST_JACOBIAN = 4
CALC_REACT_JACOBIAN = 8
CALC_NOISE = 16
CALC_OP = 32
ANALYSIS_NOISE = 1024
ANALYSIS_DC = 2048
ANALYSIS_AC = 4096
ANALYSIS_TRAN = 8192
ANALYSIS_STATIC = 32768

EVAL_RET_FLAG_LIM = 1
EVAL_RET_FLAG_FATAL = 2
EVAL_RET_FLAG_FINISH = 4
EVAL_RET_FLAG_STOP = 8

NOISE_TYPE_WHITE, NOISE_TYPE_FLICKER, NOISE_TYPE_TABLE = 0, 1, 2

_PARA_TY_NAME = {PARA_TY_REAL: "real", PARA_TY_INT: "int", PARA_TY_STR: "str"}
_PARA_KIND_NAME = {PARA_KIND_MODEL: "model", PARA_KIND_INST: "instance",
                   PARA_KIND_OPVAR: "opvar"}


# ── ABI structs (mirror osdi_0_4.h field-for-field, in order) ────────────
class OsdiNodePair(C.Structure):
    _fields_ = [("node_1", C.c_uint32), ("node_2", C.c_uint32)]


class OsdiJacobianEntry(C.Structure):
    _fields_ = [("nodes", OsdiNodePair), ("react_ptr_off", C.c_uint32),
                ("flags", C.c_uint32)]


class OsdiNode(C.Structure):
    _fields_ = [
        ("name", C.c_char_p), ("units", C.c_char_p), ("residual_units", C.c_char_p),
        ("resist_residual_off", C.c_uint32), ("react_residual_off", C.c_uint32),
        ("resist_limit_rhs_off", C.c_uint32), ("react_limit_rhs_off", C.c_uint32),
        ("is_flow", C.c_bool),
    ]


class OsdiParamOpvar(C.Structure):
    _fields_ = [
        ("name", C.POINTER(C.c_char_p)), ("num_alias", C.c_uint32),
        ("description", C.c_char_p), ("units", C.c_char_p),
        ("flags", C.c_uint32), ("len", C.c_uint32),
    ]


class OsdiNoiseSource(C.Structure):
    _fields_ = [("name", C.c_char_p), ("nodes", OsdiNodePair)]


class OsdiNatureRef(C.Structure):
    _fields_ = [("ref_type", C.c_uint32), ("index", C.c_uint32)]


class OsdiAbsDelayInfo(C.Structure):
    _fields_ = [("input_node_1", C.c_uint32), ("input_node_2", C.c_uint32),
                ("output_node", C.c_uint32), ("delay_offset", C.c_uint32),
                ("max_delay_offset", C.c_uint32)]


class OsdiDescriptor(C.Structure):
    # Function pointers are kept as c_void_p here; the eval slice wires the ones
    # it needs with explicit CFUNCTYPEs. Field order/count must match the header
    # exactly so offsets line up (validated against OSDI_DESCRIPTOR_SIZE).
    _fields_ = [
        ("name", C.c_char_p),
        ("num_nodes", C.c_uint32),
        ("num_terminals", C.c_uint32),
        ("nodes", C.POINTER(OsdiNode)),
        ("num_jacobian_entries", C.c_uint32),
        ("jacobian_entries", C.POINTER(OsdiJacobianEntry)),
        ("num_collapsible", C.c_uint32),
        ("collapsible", C.POINTER(OsdiNodePair)),
        ("collapsed_offset", C.c_uint32),
        ("noise_sources", C.POINTER(OsdiNoiseSource)),
        ("num_noise_src", C.c_uint32),
        ("num_params", C.c_uint32),
        ("num_instance_params", C.c_uint32),
        ("num_opvars", C.c_uint32),
        ("param_opvar", C.POINTER(OsdiParamOpvar)),
        ("node_mapping_offset", C.c_uint32),
        ("jacobian_ptr_resist_offset", C.c_uint32),
        ("num_states", C.c_uint32),
        ("state_idx_off", C.c_uint32),
        ("bound_step_offset", C.c_uint32),
        ("instance_size", C.c_uint32),
        ("model_size", C.c_uint32),
        ("access", C.c_void_p),
        ("setup_model", C.c_void_p),
        ("setup_instance", C.c_void_p),
        ("eval", C.c_void_p),
        ("load_noise", C.c_void_p),
        ("load_residual_resist", C.c_void_p),
        ("load_residual_react", C.c_void_p),
        ("load_limit_rhs_resist", C.c_void_p),
        ("load_limit_rhs_react", C.c_void_p),
        ("load_spice_rhs_dc", C.c_void_p),
        ("load_spice_rhs_tran", C.c_void_p),
        ("load_jacobian_resist", C.c_void_p),
        ("load_jacobian_react", C.c_void_p),
        ("load_jacobian_tran", C.c_void_p),
        ("given_flag_model", C.c_void_p),
        ("given_flag_instance", C.c_void_p),
        ("num_resistive_jacobian_entries", C.c_uint32),
        ("num_reactive_jacobian_entries", C.c_uint32),
        ("write_jacobian_array_resist", C.c_void_p),
        ("write_jacobian_array_react", C.c_void_p),
        ("num_inputs", C.c_uint32),
        ("inputs", C.POINTER(OsdiNodePair)),
        ("load_jacobian_with_offset_resist", C.c_void_p),
        ("load_jacobian_with_offset_react", C.c_void_p),
        ("unknown_nature", C.POINTER(OsdiNatureRef)),
        ("residual_nature", C.POINTER(OsdiNatureRef)),
        ("noise_source_type", C.POINTER(C.c_uint32)),
        ("load_noise_params", C.c_void_p),
        ("absdelay_count", C.c_uint32),
        ("absdelay_info", C.POINTER(OsdiAbsDelayInfo)),
    ]


# ── Pythonic views ───────────────────────────────────────────────────────
@dataclass
class Param:
    name: str
    kind: str          # "model" | "instance" | "opvar"
    dtype: str         # "real" | "int" | "str"
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    units: str = ""


@dataclass
class Node:
    name: str
    is_flow: bool
    units: str = ""


@dataclass
class ModelInfo:
    """Introspected, Python-friendly summary of one OSDI descriptor."""
    name: str
    nodes: List[Node]
    num_terminals: int
    params: List[Param]
    num_states: int
    num_noise_src: int
    instance_size: int
    model_size: int
    index: int                       # descriptor index within the library

    @property
    def terminals(self) -> List[str]:
        return [n.name for n in self.nodes[:self.num_terminals]]

    @property
    def internal_nodes(self) -> List[str]:
        return [n.name for n in self.nodes[self.num_terminals:]]

    def params_by_kind(self, kind: str) -> List[Param]:
        return [p for p in self.params if p.kind == kind]


def _s(p) -> str:
    """Decode a ctypes char_p (or NULL) to str."""
    return p.decode() if p else ""


def _decode_param(po: OsdiParamOpvar) -> Param:
    # name[] holds the primary name at [0] plus ``num_alias`` further aliases.
    n_names = min(po.num_alias + 1, 32)
    aliases = [po.name[i].decode() for i in range(n_names) if po.name[i]]
    kind = _PARA_KIND_NAME.get(po.flags & PARA_KIND_MASK, "?")
    dtype = _PARA_TY_NAME.get(po.flags & PARA_TY_MASK, "?")
    return Param(name=aliases[0], kind=kind, dtype=dtype, aliases=aliases,
                 description=_s(po.description), units=_s(po.units))


@dataclass
class OsdiLibrary:
    """A loaded ``.osdi`` shared object plus its parsed descriptors."""
    path: str
    lib: C.CDLL
    version: tuple
    models: List[ModelInfo]
    _descriptors: object = None      # ctypes array, kept alive with the lib

    def model(self, name: Optional[str] = None) -> ModelInfo:
        if name is None:
            return self.models[0]
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"{name!r} not in {[m.name for m in self.models]}")


# ── Host log callback ────────────────────────────────────────────────────
# OpenVAF models emit diagnostics through a *settable* global function pointer
# ``osdi_log`` that the host must point at a real callback after load (ngspice's
# INIT_CALLBACK). Left NULL, the first diagnostic (e.g. a default-parameter
# warning in setup_instance) calls a null pointer and segfaults.
_OSDI_LOG_T = C.CFUNCTYPE(None, C.c_void_p, C.c_char_p, C.c_uint32)


def _default_log(handle, msg, lvl):  # pragma: no cover - only fires on model diagnostics
    if (lvl & 7) >= 3:  # WARN / ERR / FATAL
        import sys
        print(f"[osdi] {msg.decode(errors='replace') if msg else ''}", file=sys.stderr)


_LOG_CB = _OSDI_LOG_T(_default_log)  # module lifetime; models hold its address


def _install_log_callback(lib) -> None:
    try:
        slot = C.c_void_p.in_dll(lib, "osdi_log")
    except ValueError:
        return  # model without the symbol
    slot.value = C.cast(_LOG_CB, C.c_void_p).value


def load_osdi(path: str) -> OsdiLibrary:
    """Load a ``.osdi`` and parse its descriptors (no device eval yet).

    Self-checks the ctypes struct layout against the library's exported
    ``OSDI_DESCRIPTOR_SIZE``; a mismatch means the ABI binding drifted from the
    header and is raised rather than risking a later segfault.
    """
    lib = C.CDLL(path)
    _install_log_callback(lib)

    major = C.c_uint32.in_dll(lib, "OSDI_VERSION_MAJOR").value
    minor = C.c_uint32.in_dll(lib, "OSDI_VERSION_MINOR").value
    if (major, minor) != (OSDI_VERSION_MAJOR_CURR, OSDI_VERSION_MINOR_CURR):
        raise RuntimeError(
            f"{path}: OSDI {major}.{minor}, host binds {OSDI_VERSION_MAJOR_CURR}."
            f"{OSDI_VERSION_MINOR_CURR}")

    desc_size = C.c_uint32.in_dll(lib, "OSDI_DESCRIPTOR_SIZE").value
    if desc_size != C.sizeof(OsdiDescriptor):
        raise RuntimeError(
            f"OsdiDescriptor layout mismatch: library says {desc_size} bytes, "
            f"ctypes struct is {C.sizeof(OsdiDescriptor)} — ABI binding is stale")

    num = C.c_uint32.in_dll(lib, "OSDI_NUM_DESCRIPTORS").value
    descriptors = (OsdiDescriptor * num).in_dll(lib, "OSDI_DESCRIPTORS")

    models: List[ModelInfo] = []
    for idx in range(num):
        d = descriptors[idx]
        nodes = [Node(name=_s(d.nodes[i].name), is_flow=bool(d.nodes[i].is_flow),
                      units=_s(d.nodes[i].units)) for i in range(d.num_nodes)]
        # param_opvar holds all params (model+instance) then opvars;
        # num_instance_params is a subset count of num_params, not extra entries.
        n_po = d.num_params + d.num_opvars
        params = [_decode_param(d.param_opvar[i]) for i in range(n_po)]
        models.append(ModelInfo(
            name=_s(d.name), nodes=nodes, num_terminals=d.num_terminals,
            params=params, num_states=d.num_states, num_noise_src=d.num_noise_src,
            instance_size=d.instance_size, model_size=d.model_size, index=idx))

    return OsdiLibrary(path=path, lib=lib, version=(major, minor),
                       models=models, _descriptors=descriptors)


# ── Eval-path ABI: sim structs + callable function typedefs ──────────────
class OsdiSimParas(C.Structure):
    _fields_ = [("names", C.POINTER(C.c_char_p)), ("vals", C.POINTER(C.c_double)),
                ("names_str", C.POINTER(C.c_char_p)), ("vals_str", C.POINTER(C.c_char_p))]


class OsdiSimInfo(C.Structure):
    # Matches osdi_0_4.rs (NOT the stale .h): trailing history_ctx + query_past_state.
    _fields_ = [("paras", OsdiSimParas), ("abstime", C.c_double),
                ("prev_solve", C.POINTER(C.c_double)), ("prev_state", C.POINTER(C.c_double)),
                ("next_state", C.POINTER(C.c_double)), ("flags", C.c_uint32),
                ("history_ctx", C.c_void_p), ("query_past_state", C.c_void_p)]


class OsdiInitError(C.Structure):
    _fields_ = [("code", C.c_uint32), ("parameter_id", C.c_uint32)]  # payload union = u32


class OsdiInitInfo(C.Structure):
    _fields_ = [("flags", C.c_uint32), ("num_errors", C.c_uint32),
                ("errors", C.POINTER(OsdiInitError))]


_ACCESS_T = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint32, C.c_uint32)
# void*-only twins of eval/load for the numba op-solve kernel: numba calls
# ctypes fn pointers natively when every arg is a plain word (addresses passed
# as runtime ints — nothing process-specific is baked into the cached code).
_EVAL_VP_T = C.CFUNCTYPE(C.c_uint32, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p)
_LOAD_RESID_VP_T = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_void_p)
_LOAD_JAC_VP_T = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)
_SETUP_MODEL_T = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.POINTER(OsdiSimParas),
                             C.POINTER(OsdiInitInfo))
_SETUP_INST_T = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_void_p, C.c_double, C.c_uint32,
                            C.POINTER(OsdiSimParas), C.POINTER(OsdiInitInfo))
_EVAL_T = C.CFUNCTYPE(C.c_uint32, C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(OsdiSimInfo))
_LOAD_RESID_T = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.POINTER(C.c_double))
_LOAD_JAC_RESIST_T = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)
_LOAD_JAC_REACT_T = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_double)
_LOAD_NOISE_T = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_double, C.POINTER(C.c_double))

_U32_MAX = 0xFFFFFFFF
# eval() flag set for a DC operating point (resistive residual + Jacobian).
_DC_OP_FLAGS = (CALC_RESIST_RESIDUAL | CALC_RESIST_JACOBIAN | ANALYSIS_DC | ANALYSIS_STATIC)


class Device:
    """One OSDI model+instance: set params, solve a DC operating point.

    Owns a model buffer and an instance buffer (16-byte-aligned, zeroed), runs
    the OSDI setup, honours node-collapsing, and solves the device's internal
    nodes at fixed terminal voltages (a small dense Newton), then Schur-complements
    the reduced Jacobian to terminal conductances → Id, gm, gds.
    """

    def __init__(self, osdi: OsdiLibrary, params: Dict[str, float], *,
                 model_name: Optional[str] = None, temperature: float = 300.15):
        self._osdi = osdi
        info = osdi.model(model_name)
        self.info = info
        d = osdi._descriptors[info.index]
        self._d = d
        self.n_nodes = int(d.num_nodes)
        self.n_term = int(d.num_terminals)
        self.terminals = info.terminals

        # callable function pointers
        self._access = _ACCESS_T(d.access)
        self._c_setup_model = _SETUP_MODEL_T(d.setup_model)
        self._c_setup_inst = _SETUP_INST_T(d.setup_instance)
        self._c_eval = _EVAL_T(d.eval)
        self._c_load_resid = _LOAD_RESID_T(d.load_residual_resist)
        self._c_load_jac = _LOAD_JAC_RESIST_T(d.load_jacobian_resist)
        self._c_load_jac_react = _LOAD_JAC_REACT_T(d.load_jacobian_react)
        self._c_load_resid_react = _LOAD_RESID_T(d.load_residual_react)
        self._c_load_noise = _LOAD_NOISE_T(d.load_noise)
        # void*-signature twins for the numba op-solve / transient kernels
        self._k_eval = _EVAL_VP_T(d.eval)
        self._k_load_resid = _LOAD_RESID_VP_T(d.load_residual_resist)
        self._k_load_jac = _LOAD_JAC_VP_T(d.load_jacobian_resist)
        self._k_load_resid_react = _LOAD_RESID_VP_T(d.load_residual_react)
        self._k_load_jac_react = _LOAD_JAC_REACT_T(d.load_jacobian_react)
        self.num_noise_src = int(d.num_noise_src)
        self._noise_type = ([int(d.noise_source_type[i]) for i in range(self.num_noise_src)]
                            if self.num_noise_src and d.noise_source_type else [])

        # param name -> id (model params only; opvars/instance handled separately)
        self._pid: Dict[str, int] = {}
        self._ptype: Dict[str, str] = {}
        for i, p in enumerate(info.params):
            if p.kind != "model":
                continue
            for nm in p.aliases:
                self._pid[nm] = i
                self._ptype[nm] = p.dtype

        # 16-byte-aligned zeroed buffers (macOS malloc → 16-align; numpy uses it)
        self._model_buf = np.zeros(max(int(d.model_size), 1), dtype=np.uint8)
        self._inst_buf = np.zeros(max(int(d.instance_size), 1), dtype=np.uint8)
        self._model = C.c_void_p(self._model_buf.ctypes.data)
        self._inst = C.c_void_p(self._inst_buf.ctypes.data)
        self._handle = C.create_string_buffer(b"pyosdi")
        self._handle_p = C.cast(self._handle, C.c_void_p)

        # simulator params queried via $simparam — mirror ngspice get_simparams
        # (osdiload.c). tnom is the nominal/extraction temp in °C; the operating
        # temperature (K) is passed separately to setup_instance.
        self._sp_name_b = [b"iniLim", b"gmin", b"gdev", b"tnom", b"simulatorVersion",
                           b"sourceScaleFactor", b"epsmin", b"reltol", b"vntol", b"abstol"]
        n = len(self._sp_name_b)
        self._sp_names = (C.c_char_p * (n + 1))(*self._sp_name_b, None)
        self._sp_vals = (C.c_double * n)(0.0, 1e-12, 0.0, 27.0, 46.0,
                                         1.0, 1e-28, 1e-3, 1e-6, 1e-12)
        self._sp_str = (C.c_char_p * 1)()
        self._sp = OsdiSimParas(
            names=C.cast(self._sp_names, C.POINTER(C.c_char_p)),
            vals=C.cast(self._sp_vals, C.POINTER(C.c_double)),
            names_str=C.cast(self._sp_str, C.POINTER(C.c_char_p)), vals_str=None)

        self._setup(params, temperature)
        self._build_collapse()
        self._wire_node_mapping()
        self._wire_jacobian()
        # scratch reused across evals
        self._v = np.zeros(self.n_nodes, dtype=np.float64)
        self._resid = np.zeros(self.n_nodes, dtype=np.float64)
        self._build_scatter()
        self._info_cache: Dict[int, OsdiSimInfo] = {}
        self._last_redv: Optional[np.ndarray] = None
        self._op_memo: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._lin_memo: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._solve_args: Optional[tuple] = None

    # ── setup ────────────────────────────────────────────────────────────
    def _set_model_param(self, name: str, val) -> None:
        pid = self._pid.get(name)
        if pid is None:
            raise KeyError(f"{self.info.name}: no model param {name!r}")
        addr = self._access(None, self._model, pid, ACCESS_FLAG_SET)
        if not addr:
            raise RuntimeError(f"access(SET) returned NULL for {name!r}")
        if self._ptype[name] == "int":
            C.c_int32.from_address(addr).value = int(val)
        else:
            C.c_double.from_address(addr).value = float(val)

    def _check_init(self, res: OsdiInitInfo, what: str) -> None:
        if res.num_errors:
            e0 = res.errors[0]
            raise RuntimeError(f"{what}: OSDI init error code={e0.code} "
                               f"param_id={e0.parameter_id}")

    def _setup(self, params: Dict[str, float], temperature: float) -> None:
        self._unknown = [k for k in params if k not in self._pid]
        for name, val in params.items():
            if name in self._pid:
                self._set_model_param(name, val)
        res = OsdiInitInfo()
        self._c_setup_model(self._handle_p, self._model, C.byref(self._sp), C.byref(res))
        self._check_init(res, "setup_model")
        res = OsdiInitInfo()
        self._c_setup_inst(self._handle_p, self._inst, self._model, float(temperature),
                           C.c_uint32(self.n_term), C.byref(self._sp), C.byref(res))
        self._check_init(res, "setup_instance")

    def _build_collapse(self) -> None:
        """Replicate melange collapse_nodes → node map (device node -> reduced id)."""
        d = self._d
        collapsed = (C.c_bool * int(d.num_collapsible)).from_address(
            self._inst.value + int(d.collapsed_offset)) if d.num_collapsible else []
        nmap = list(range(self.n_nodes))
        for i in range(int(d.num_collapsible)):
            if not collapsed[i]:
                continue
            pair = d.collapsible[i]
            mapped_from = nmap[pair.node_1]
            to = pair.node_2
            gnd = (to == _U32_MAX)
            mapped_to = _U32_MAX if gnd else nmap[to]
            if not gnd and mapped_to == _U32_MAX:
                gnd = True
            if mapped_from < self.n_term and (gnd or mapped_to < self.n_term):
                continue
            if not gnd and mapped_from < mapped_to:
                mapped_from, mapped_to = mapped_to, mapped_from
            for j in range(self.n_nodes):
                m = nmap[j]
                if m == mapped_from:
                    nmap[j] = mapped_to
                elif m != _U32_MAX and m > mapped_from:
                    nmap[j] = m - 1
        self._nmap = nmap
        self._n_red = 1 + max((m for m in nmap if m != _U32_MAX), default=self.n_term - 1)

    def _wire_node_mapping(self) -> None:
        """Identity node map: the model indexes prev_solve/residual for device
        node ``i`` through ``node_mapping[i]``; setting it to ``i`` means our
        vectors are indexed directly by device node (we reduce collapse in
        Python via ``_nmap``)."""
        nm = (C.c_uint32 * self.n_nodes).from_address(
            self._inst.value + int(self._d.node_mapping_offset))
        for i in range(self.n_nodes):
            nm[i] = i

    def _wire_jacobian(self) -> None:
        """Point the instance's resistive-Jacobian pointer slots at our storage."""
        d = self._d
        n = int(d.num_jacobian_entries)
        self._jac_vals = (C.c_double * n)()
        base = C.addressof(self._jac_vals)
        # instance holds an array of `double*` slots; point each at our storage.
        # Store raw addresses (byref temporaries don't survive being persisted).
        slots = (C.c_void_p * n).from_address(
            self._inst.value + int(d.jacobian_ptr_resist_offset))
        for k in range(n):
            slots[k] = base + k * C.sizeof(C.c_double)
        # reactive Jacobian (dQ/dV): pointer lives at each entry's react_ptr_off
        self._react_vals = (C.c_double * n)()
        rbase = C.addressof(self._react_vals)
        for k in range(n):
            off = int(d.jacobian_entries[k].react_ptr_off)
            if off != _U32_MAX:
                C.c_void_p.from_address(self._inst.value + off).value = \
                    rbase + k * C.sizeof(C.c_double)
        self._jac_entries = d.jacobian_entries
        self._n_jac = n
        # numpy views over the ctypes value stores (shared memory, zero-copy)
        self._jac_np = (np.ctypeslib.as_array(self._jac_vals) if n
                        else np.zeros(0, dtype=np.float64))
        self._react_np = (np.ctypeslib.as_array(self._react_vals) if n
                          else np.zeros(0, dtype=np.float64))

    def _build_scatter(self) -> None:
        """Precompute the node/Jacobian scatter index arrays used by every eval.

        The per-entry ctypes walks (``jacobian_entries[k].nodes.node_1`` …) are
        the dominant Python overhead when done per call; resolving them once into
        flat numpy index arrays turns each eval's scatter into a few ``bincount``s.
        """
        nred = self._n_red
        nmap = np.array(self._nmap, dtype=np.int64)
        valid = nmap != _U32_MAX
        self._node_sel = np.nonzero(valid)[0]                 # device nodes kept
        self._node_red_idx = nmap[valid].astype(np.intp)      # → reduced ids
        n = self._n_jac
        rows = np.empty(n, dtype=np.int64)
        cols = np.empty(n, dtype=np.int64)
        react = np.empty(n, dtype=np.int64)
        for k in range(n):
            e = self._jac_entries[k]
            # entry (node_1, node_2) is d(residual@node_1)/d(V@node_2):
            # node_1 = equation row, node_2 = variable column.
            rows[k] = self._nmap[e.nodes.node_1]
            cols[k] = self._nmap[e.nodes.node_2]
            react[k] = int(e.react_ptr_off)
        ok = (rows != _U32_MAX) & (cols != _U32_MAX)
        self._jac_sel = np.nonzero(ok)[0].astype(np.intp)
        self._jac_flat = (rows[ok] * nred + cols[ok]).astype(np.intp)
        rok = ok & (react != _U32_MAX)
        self._react_sel = np.nonzero(rok)[0].astype(np.intp)
        self._react_flat = (rows[rok] * nred + cols[rok]).astype(np.intp)
        ni = nred - self.n_term
        self._eye_int = np.eye(ni) if ni > 0 else None   # internal-block identity

    # ── eval ─────────────────────────────────────────────────────────────
    def _eval_reduced(self, redv: np.ndarray):
        """Eval at reduced node voltages → (R_reduced, J_reduced) dense.

        Readable single-eval reference; ``operating_point``'s hot loop runs the
        same per-eval body inside ``numba_kernels._osdi_op_solve_impl`` — keep
        the two in lockstep when touching either.
        """
        self._set_nodes(redv)
        info = self._sim_info(_DC_OP_FLAGS)
        ret = self._c_eval(self._handle_p, self._inst, self._model, C.byref(info))
        if ret & EVAL_RET_FLAG_FATAL:
            raise RuntimeError("OSDI eval returned $fatal")
        # load_* functions STAMP (+=) into their destinations, SPICE-style, so
        # clear both before each load or values accumulate across evals/calls.
        self._resid.fill(0.0)
        self._jac_np[:] = 0.0
        self._c_load_resid(self._inst, self._model,
                           self._resid.ctypes.data_as(C.POINTER(C.c_double)))
        self._c_load_jac(self._inst, self._model)
        nred = self._n_red
        R = np.bincount(self._node_red_idx, weights=self._resid[self._node_sel],
                        minlength=nred)
        J = np.bincount(self._jac_flat, weights=self._jac_np[self._jac_sel],
                        minlength=nred * nred).reshape(nred, nred)
        return R, J

    def _set_nodes(self, redv: np.ndarray) -> None:
        v = self._v
        v.fill(0.0)
        v[self._node_sel] = redv[self._node_red_idx]

    def _sim_info(self, flags: int) -> "OsdiSimInfo":
        # reused across evals: self._v never reallocates, so prev_solve is stable
        info = self._info_cache.get(flags)
        if info is None:
            info = OsdiSimInfo(
                paras=self._sp, abstime=0.0,
                prev_solve=self._v.ctypes.data_as(C.POINTER(C.c_double)),
                prev_state=None, next_state=None, flags=flags,
                history_ctx=None, query_past_state=None)
            self._info_cache[flags] = info
        return info

    def _op_solve_args(self) -> tuple:
        """Marshalling tuple for the numba op-solve kernel (built once).

        Everything is a runtime value — fn-pointer objects, buffer addresses,
        index arrays — so the jitted kernel stays process-portable (cacheable).
        """
        args = self._solve_args
        if args is None:
            info = self._sim_info(_DC_OP_FLAGS)
            args = (self._k_eval, self._k_load_resid, self._k_load_jac,
                    self._handle_p.value, self._inst.value, self._model.value,
                    C.addressof(info), self._resid.ctypes.data,
                    self._v, self._resid, self._jac_np,
                    self._node_sel, self._node_red_idx,
                    self._jac_sel, self._jac_flat)
            self._solve_args = args
        return args

    def _eval_react(self, redv: np.ndarray) -> np.ndarray:
        """Reduced reactive Jacobian (dQ/dV) at the given reduced voltages."""
        self._set_nodes(redv)
        info = self._sim_info(CALC_REACT_RESIDUAL | CALC_REACT_JACOBIAN | ANALYSIS_AC)
        if self._c_eval(self._handle_p, self._inst, self._model,
                        C.byref(info)) & EVAL_RET_FLAG_FATAL:
            raise RuntimeError("OSDI reactive eval returned $fatal")
        self._react_np[:] = 0.0
        self._c_load_jac_react(self._inst, self._model, 1.0)
        nred = self._n_red
        return np.bincount(self._react_flat, weights=self._react_np[self._react_sel],
                           minlength=nred * nred).reshape(nred, nred)

    def charges(self, redv: np.ndarray) -> np.ndarray:
        """Reduced-node reactive charges Q [C] at the given voltages (for transient
        dQ/dt). Uses the OSDI reactive residual."""
        self._set_nodes(redv)
        info = self._sim_info(CALC_REACT_RESIDUAL | ANALYSIS_TRAN)
        if self._c_eval(self._handle_p, self._inst, self._model,
                        C.byref(info)) & EVAL_RET_FLAG_FATAL:
            raise RuntimeError("OSDI charge eval returned $fatal")
        self._resid.fill(0.0)
        self._c_load_resid_react(self._inst, self._model,
                                 self._resid.ctypes.data_as(C.POINTER(C.c_double)))
        return np.bincount(self._node_red_idx, weights=self._resid[self._node_sel],
                           minlength=self._n_red)

    def noise_psd(self, freq: float) -> np.ndarray:
        """Per-noise-source output current PSD [A^2/Hz] at the last op point.

        Requires a prior :meth:`operating_point`; re-evaluates with CALC_NOISE at
        that bias so the model computes the (op-dependent) noise sources, then
        ``load_noise`` returns their frequency-dependent PSD.
        """
        if getattr(self, "_last_redv", None) is not None:
            self._set_nodes(self._last_redv)
            info = self._sim_info(CALC_RESIST_RESIDUAL | CALC_NOISE | CALC_OP
                                  | ANALYSIS_NOISE)
            self._c_eval(self._handle_p, self._inst, self._model, C.byref(info))
        out = np.zeros(max(self.num_noise_src, 1), dtype=np.float64)
        self._c_load_noise(self._inst, self._model, float(freq),
                           out.ctypes.data_as(C.POINTER(C.c_double)))
        return out[:self.num_noise_src]

    def noise_by_type(self, freq: float):
        """Output-current PSD [A^2/Hz] split into (thermal/white, flicker) at freq."""
        psd = self.noise_psd(freq)
        thermal = flicker = 0.0
        for i, t in enumerate(self._noise_type):
            if t == NOISE_TYPE_FLICKER:
                flicker += psd[i]
            else:                       # white + table folded into thermal
                thermal += psd[i]
        return float(thermal), float(flicker)

    _OP_MEMO_MAX = 128

    def operating_point(self, vd: float, vg: float, vs: float, vb: float,
                        *, tol: float = 1e-12, max_iter: int = 100,
                        gmin: float = 1e-12, with_caps: bool = True) -> Dict[str, float]:
        """Solve internal nodes at fixed terminal V; return Id, gm, gds, gmb.

        Terminal order is the descriptor's (bsim4va: d, g, s, b). Id is the
        converged residual (current into drain); gm/gds/gmb come from the Schur
        complement of the reduced resistive Jacobian onto the 4 terminals. A small
        ``gmin`` regularises the internal block (some BSIM4 internal nodes are
        DC-floating, so the raw internal Jacobian is singular — the SPICE fix).

        Repeated calls at the *exact* same bias return a memoised result (so e.g.
        a per-frequency noise sweep pays one Newton solve, and re-solves are
        bit-reproducible). The returned dict is shared with the memo — callers
        must treat it as read-only. New biases warm-start the internal-node Newton from the
        previous solution — during a circuit DC solve the same device is evaluated
        at nearby biases, cutting iterations — with a cold vs-fill restart if the
        warm start fails, and one extra "polish" step after hitting ``tol`` so the
        converged point is path-independent to ~1e-16.

        ``with_caps=False`` skips the reactive (dQ/dV) eval and omits the C* keys —
        the DC root-finder only needs Id, so its inner loop shouldn't pay for
        capacitances. A later ``with_caps=True`` call at the same bias upgrades the
        memoised entry with one reactive eval (no re-solve).
        """
        key = (float(vd), float(vg), float(vs), float(vb), tol, max_iter, gmin)
        memo = self._op_memo.get(key)
        if memo is not None:
            self._op_memo.move_to_end(key)
            op, redv_m, has_caps = memo
            self._last_redv = redv_m         # noise_psd re-eval stays consistent
            if with_caps and not has_caps:
                Cm = self._eval_react(redv_m)
                di, gi, si = 0, 1, 2
                op.update(Cgg=float(Cm[gi, gi]), Cgs=float(-Cm[gi, si]),
                          Cgd=float(-Cm[gi, di]), Cdd=float(Cm[di, di]))
                self._op_memo[key] = (op, redv_m, True)
            return op                    # shared with the memo — treat as read-only
        T = self.n_term
        nred = self._n_red
        ni = nred - T
        eye = self._eye_int
        term = [vd, vg, vs, vb][:T]
        starts = []
        if ni and self._last_redv is not None and self._last_redv.shape[0] == nred:
            starts.append(self._last_redv[T:].copy())     # warm: last solve's internals
        starts.append(np.full(ni, vs, dtype=np.float64))  # cold: legacy vs-fill
        R = J = redv = None
        solve_args = self._op_solve_args()
        for attempt, internals in enumerate(starts):
            redv = np.empty(nred, dtype=np.float64)
            redv[:T] = term
            redv[T:] = internals
            # a warm start either pays off within a few steps or won't at all —
            # cap it so a bad hint falls through to the cold restart cheaply
            iters = min(20, max_iter) if attempt < len(starts) - 1 else max_iter
            # whole internal Newton (polish / unphysical-bias bail / singular
            # rescue) runs inside the jitted kernel, OSDI eval called in-loop
            status, R, J = _osdi_op_solve_impl(*solve_args, redv, T, iters,
                                               float(tol), float(gmin))
            if status == 3:
                raise RuntimeError("OSDI eval returned $fatal")
            if status == 0 or attempt == len(starts) - 1:
                if status != 0:
                    # budget exhausted / bailed / diverged on the final start:
                    # keep legacy behavior (report via converged_resid), no raise
                    diagnostics.note("osdi.op_not_converged",
                                     detail=f"{self.info.name} bias={key[:4]}")
                break
            diagnostics.note("osdi.warm_start_fallback",
                             detail=f"{self.info.name}: warm-start Newton failed, "
                                    f"cold restart at bias={key[:4]}")
        # Schur complement onto terminals: G = Jtt - Jti (Jii+gmin)^-1 Jit
        if ni:
            ok, G = _osdi_schur_impl(J, T, gmin)
            if not ok:
                G = J[:T, :T] - J[:T, T:] @ np.linalg.solve(
                    J[T:, T:] + gmin * eye, J[T:, :T])
        else:
            G = J[:T, :T]
        self._last_redv = redv.copy()    # for noise_psd re-eval at this bias
        di, gi, si, bi = 0, 1, 2, 3
        op = {
            "Id": float(R[di]),
            "gm": float(G[di, gi]),           # dId/dVg = d(resid@drain)/dV(gate)
            "gds": float(G[di, di]),          # dId/dVd
            "gmb": float(G[di, bi]) if T > 3 else 0.0,
            "vth_bias": vg - vs,
            "converged_resid": float(np.max(np.abs(R[T:]))) if nred > T else 0.0,
            "n_internal": nred - T,
            "unknown_params": list(self._unknown),
        }
        if with_caps:
            # terminal capacitances from the reactive Jacobian dQ/dV at the op point
            Cm = self._eval_react(redv)
            op.update(Cgg=float(Cm[gi, gi]), Cgs=float(-Cm[gi, si]),
                      Cgd=float(-Cm[gi, di]), Cdd=float(Cm[di, di]))
        self._op_memo[key] = (op, self._last_redv, with_caps)
        while len(self._op_memo) > self._OP_MEMO_MAX:
            self._op_memo.popitem(last=False)
        return op                        # shared with the memo — treat as read-only

    _LIN_MEMO_MAX = 1024

    def terminal_linearization(self, vd: float, vg: float, vs: float, vb: float,
                               *, gmin: float = 1e-12):
        """Quasi-static terminal small-signal ``(G4, C4)`` at a bias (memoised).

        First-order elimination of the internal nodes from the full linearized
        device (rows/cols ordered d, g, s, b):

            G = Jtt − Jti·X,                 X = (Jii + gmin·I)⁻¹ Jit
            C = Ctt − Cti·X − Jti·(W − V·X), W = (…)⁻¹ Cit,  V = (…)⁻¹ Cii

        valid for analysis frequencies far below the internal-node poles
        (~THz for BSIM4's rds/charge nodes). This is what the periodic
        (PAC/PNoise) linearization along a PSS orbit stamps per time sample.
        The returned arrays are shared with the memo — treat as read-only.
        """
        key = (float(vd), float(vg), float(vs), float(vb))
        memo = self._lin_memo.get(key)
        if memo is not None:
            self._lin_memo.move_to_end(key)
            return memo
        self.operating_point(vd, vg, vs, vb, with_caps=False)
        redv = self._last_redv
        _, J = self._eval_reduced(redv)
        Cm = self._eval_react(redv)
        T = self.n_term
        ni = self._n_red - T
        if ni:
            Aii = J[T:, T:] + gmin * self._eye_int
            try:
                X = np.linalg.solve(Aii, J[T:, :T])
                W = np.linalg.solve(Aii, Cm[T:, :T])
                V = np.linalg.solve(Aii, Cm[T:, T:])
            except np.linalg.LinAlgError:
                X = np.linalg.lstsq(Aii, J[T:, :T], rcond=None)[0]
                W = np.linalg.lstsq(Aii, Cm[T:, :T], rcond=None)[0]
                V = np.linalg.lstsq(Aii, Cm[T:, T:], rcond=None)[0]
            G4 = J[:T, :T] - J[:T, T:] @ X
            C4 = Cm[:T, :T] - Cm[:T, T:] @ X - J[:T, T:] @ (W - V @ X)
        else:
            G4 = J[:T, :T].copy()
            C4 = Cm[:T, :T].copy()
        out = (G4, C4)
        self._lin_memo[key] = out
        while len(self._lin_memo) > self._LIN_MEMO_MAX:
            self._lin_memo.popitem(last=False)
        return out


def _main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m core.osdi_host <model.osdi>")
        return 2
    osdi = load_osdi(argv[1])
    print(f"{osdi.path}  (OSDI {osdi.version[0]}.{osdi.version[1]})")
    for m in osdi.models:
        mp = m.params_by_kind("model")
        ip = m.params_by_kind("instance")
        ov = m.params_by_kind("opvar")
        print(f"\nmodule '{m.name}'  "
              f"terminals={m.terminals}  internal={m.internal_nodes}")
        print(f"  states={m.num_states}  noise_src={m.num_noise_src}  "
              f"inst_size={m.instance_size}B  model_size={m.model_size}B")
        print(f"  params: {len(mp)} model, {len(ip)} instance, {len(ov)} opvars")
        print(f"  instance params: {[p.name for p in ip]}")
        print(f"  opvars: {[p.name for p in ov][:40]}")
        print(f"  first 40 model params: {[p.name for p in mp][:40]}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv))
