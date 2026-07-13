"""
Nonlinear transient solver for the AFE (backward-Euler + per-step Newton).

Integrates the full 6-node circuit DAE in time. Built on the same device model
(:class:`~device_model.TransistorModel`) and topology (AFE_TOPO) as the
DC/AC/Noise stack, so the steady state matches the DC solver and the
small-signal response matches the AC solver.

Method
------
- KCL at every solved node:  Σ device currents  +  Σ capacitor currents  = 0.
- Capacitor companion (backward Euler), cap branch between terminals a,b:
      i_ab ≈ C_step·[(Va-Vb)_n − (Va-Vb)_{n-1}] / h
  Linear capacitors use their fixed C. PMOS Cgss/Cgdd follow the AT_4000TG
  step companion selected by CIRCUIT_PMOS_TRANSIENT_CAP_MODE (or the per-call
  cap_mode_id): `charge` (default, L-stable Q-stamp) or `average`
  (trapezoidal, the stable non-conservative form matching Cadence's
  feedthrough). AC/PAC/noise still use the local small-signal capacitances.
  The AT_4000TG model routes these caps through an internal gate1 node; this
  solver keeps the long-timescale R_cap2 leakage from source/drain to gate1 and
  collapses the 100 Ω gate-to-gate1 RC because its ns-scale time constant is far
  below the chopper edge times used here.
- Each step: solve the 6 node voltages with a damped Newton iteration using an
  analytic conductance Jacobian (gm/gds finite-diff of get_Idc + cap C/h + gmin),
  seeded from the previous step, with step limiting that keeps it on the physical
  branch. (Bare fsolve's poorly-scaled numeric Jacobian latched onto wrong roots
  of this positive-feedback circuit — gain didn't match the AC reference.)

Caps stamped: per device Cgs (gate-source) and Cgd (gate-drain), plus CL on the
two outputs. Inputs: M7 gate = vip(t), M8 gate = vin(t) (driven); other rails fixed.

CURRENT SIGN: devices use the signed Verilog-A drain-terminal current. The old
``signed_devices`` knob is retained for API compatibility, but it is no longer a
per-device behavior switch.

This is the engine; chopper switch topologies can be driven through node_inputs.
Clock feedthrough from the Verilog-A Cgss/Cgdd terms is included when the switch
gate clocks are finite-edge waveforms. Additional explicit charge-injection
pulses can be supplied through current_inputs.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from .adaptive_config import (
    resolve_adaptive_config,
)
from .topology import AFE_TOPO
from .ac_solver import ac_solve
from .device_factory import build_devices, resolve_binding
from .transient_profile import (
    PROFILE_EDGE_NEWTON_ITERS,
    PROFILE_EDGE_SUBSTEPS,
    PROFILE_FAILED_EDGE_INTERVALS,
    PROFILE_FAILED_FLAT_INTERVALS,
    PROFILE_FAILED_INTERVALS,
    PROFILE_FAILED_LAST_RESIDUAL_INF,
    PROFILE_FAILED_LAST_STEP_INF,
    PROFILE_FAILED_LINEAR_SOLVE_COUNT,
    PROFILE_FAILED_MAX_RESIDUAL_INF,
    PROFILE_FAILED_MAX_STEP_INF,
    PROFILE_FAILED_MAXIT_COUNT,
    PROFILE_FAILED_STAMP_OR_PREV_COUNT,
    PROFILE_FAILED_SUBSTEPS,
    PROFILE_FLAT_NEWTON_ITERS,
    PROFILE_FLAT_SUBSTEPS,
    PROFILE_INTERNAL_FD_JAC_FALLBACKS,
    PROFILE_INTERVALS,
    PROFILE_LEN,
    PROFILE_NEWTON_ITERS,
    PROFILE_PMOS_INTERNAL_NEWTON_ATTEMPTS,
    PROFILE_PMOS_INTERNAL_NEWTON_ITERS,
    PROFILE_PMOS_OP_SOLVES,
    PROFILE_STALLED_RESIDUAL_ACCEPTS,
    PROFILE_SUBSTEPS,
    PROFILE_TERMINAL_FD_JAC_FALLBACKS,
)
from .compiled_topology import CompiledTopology, index_array, term_arrays
from . import diagnostics

if TYPE_CHECKING:
    from .device_factory import CircuitBinding

# Single source of the transient drivers: these `_impl` kernels are the jitted
# functions when Numba is installed and the raw pure-Python functions otherwise
# (never None), so the no-Numba transient path runs the *same* kernels interpreted
# rather than a hand-mirrored OO copy.
from .numba_kernels import (
    _transient_solve_adaptive_gear2_impl,
    _transient_solve_grid_gear2_impl,
    _transient_solve_grid_impl,
)


_CAP_MODE = os.environ.get("CIRCUIT_PMOS_TRANSIENT_CAP_MODE", "charge").lower()
_USE_AVERAGE_CAPS = _CAP_MODE in {"average", "avg", "trapezoid", "trap"}
_CAP_MODE_ID = 1 if _USE_AVERAGE_CAPS else 0


_CAP_MODE_IDS = {
    "charge": 0, "q": 0, "qstamp": 0, "q-stamp": 0,
    "average": 1, "avg": 1, "trapezoid": 1, "trap": 1,
}


@dataclass
class _PathOutcome:
    handled: bool = False
    Vhist: object = None
    tgrid: object = None
    input_values: object = None
    N: object = None
    nfail: object = None
    nretry: object = None
    nsubsteps: object = None
    gear2_done: bool = False
    gear2_numba_used: bool = False
    gear2_python_retry_used: bool = False
    adaptive_used: bool = False
    adaptive_numba_used: bool = False
    used_grid_numba: bool = False
    partial_grid_numba: bool = False
    python_start_idx: object = None
    profile_wall_s: object = None
    profile_stats: object = None
    numba_grid_error: object = None
    numba_grid_failed_index: object = None
    numba_grid_failed_substeps: object = None
    numba_grid_failed_profile: object = None
    numba_grid_failed_intervals: object = None


@dataclass
class _NewtonStats:
    """Mutable diagnostic counters for the numba per-step Newton (held by the
    otherwise-frozen :class:`_TransientCtx`; non-critical, reported in the result)."""
    attempts: int = 0
    success: int = 0
    fallback: int = 0


@dataclass(frozen=True, slots=True)
class _TopologyCtx:
    """Compiled topology and solved-node indexing."""
    plan: object
    idx: dict
    n: int
    n_aug: int
    termv: object
    tft: dict
    signed_devices: set


@dataclass(frozen=True, slots=True)
class _DeviceCtx:
    """PMOS device metadata plus dense arrays consumed by Python/Numba kernels."""
    dev_meta: list
    dev_objs: list
    dev_d_kind: np.ndarray; dev_d_ref: np.ndarray; dev_d_val: np.ndarray
    dev_g_kind: np.ndarray; dev_g_ref: np.ndarray; dev_g_val: np.ndarray
    dev_s_kind: np.ndarray; dev_s_ref: np.ndarray; dev_s_val: np.ndarray
    dev_di: np.ndarray; dev_gi: np.ndarray; dev_si: np.ndarray
    dev_use_abs: np.ndarray
    p_Vfb: np.ndarray; p_Vss: np.ndarray; p_Lc: np.ndarray; p_lambda: np.ndarray
    p_contact_scale: np.ndarray; p_exponent: np.ndarray; p_current_scale: np.ndarray
    p_inv_Rleak: np.ndarray; p_two_over_pi: np.ndarray; p_cap_cgs1: np.ndarray
    p_cap_cgd1: np.ndarray; p_cap_half_wl_ci: np.ndarray; p_cap_cgs3_base: np.ndarray
    p_cap_cgd3_base: np.ndarray; p_k1: np.ndarray; p_gate_leak_g: np.ndarray


@dataclass(frozen=True, slots=True)
class _PassiveCtx:
    """Linear passive elements, in both Python metadata and Numba array form."""
    load_meta: list
    res_meta: list
    res_a_kind: np.ndarray; res_a_ref: np.ndarray; res_a_val: np.ndarray
    res_b_kind: np.ndarray; res_b_ref: np.ndarray; res_b_val: np.ndarray
    res_ai: np.ndarray; res_bi: np.ndarray; res_g: np.ndarray
    cap_a_kind: np.ndarray; cap_a_ref: np.ndarray; cap_a_val: np.ndarray
    cap_b_kind: np.ndarray; cap_b_ref: np.ndarray; cap_b_val: np.ndarray
    cap_ai: np.ndarray; cap_bi: np.ndarray; cap_value: np.ndarray


@dataclass(frozen=True, slots=True)
class _SourceCtx:
    """Independent/dependent source metadata and Numba source arrays."""
    isrc_meta: list
    vccs_meta: list
    vs_meta: list
    vcvs_meta: list
    cccs_meta: list
    ccvs_meta: list
    dyn_isrc_meta: list
    isrc_pi: np.ndarray; isrc_qi: np.ndarray; isrc_value: np.ndarray
    vccs_pi: np.ndarray; vccs_qi: np.ndarray; vccs_cpi: np.ndarray
    vccs_cni: np.ndarray; vccs_gm: np.ndarray
    dyn_pi: np.ndarray; dyn_qi: np.ndarray; dyn_input_idx: np.ndarray
    # Branch-element arrays for the Numba augmented (n_aug > n) path; empty when
    # n_aug == n (amp/chopper) so the Numba node path is unaffected.
    vs_a_kind: np.ndarray; vs_a_ref: np.ndarray; vs_a_val: np.ndarray
    vs_b_kind: np.ndarray; vs_b_ref: np.ndarray; vs_b_val: np.ndarray
    vs_pi: np.ndarray; vs_qi: np.ndarray; vs_bi: np.ndarray
    vs_e_const: np.ndarray; vs_e_idx: np.ndarray
    vcvs_a_kind: np.ndarray; vcvs_a_ref: np.ndarray; vcvs_a_val: np.ndarray
    vcvs_b_kind: np.ndarray; vcvs_b_ref: np.ndarray; vcvs_b_val: np.ndarray
    vcvs_cp_kind: np.ndarray; vcvs_cp_ref: np.ndarray; vcvs_cp_val: np.ndarray
    vcvs_cn_kind: np.ndarray; vcvs_cn_ref: np.ndarray; vcvs_cn_val: np.ndarray
    vcvs_pi: np.ndarray; vcvs_qi: np.ndarray; vcvs_cpi: np.ndarray
    vcvs_cni: np.ndarray; vcvs_bi: np.ndarray; vcvs_mu: np.ndarray
    cccs_pi: np.ndarray; cccs_qi: np.ndarray
    cccs_ctrl_bi: np.ndarray; cccs_beta: np.ndarray
    ccvs_a_kind: np.ndarray; ccvs_a_ref: np.ndarray; ccvs_a_val: np.ndarray
    ccvs_b_kind: np.ndarray; ccvs_b_ref: np.ndarray; ccvs_b_val: np.ndarray
    ccvs_pi: np.ndarray; ccvs_qi: np.ndarray; ccvs_bi: np.ndarray
    ccvs_ctrl_bi: np.ndarray; ccvs_gamma: np.ndarray


@dataclass(frozen=True, slots=True)
class _SolverOptions:
    """Scalar solver controls shared by the execution paths."""
    gmin: float; HH: float; clip_lo: float; clip_hi: float; cap_id: int
    rail_margin: object
    newton_maxit: int; newton_step_limit: float; newton_vtol: float
    fallback_full_jacobian: bool; fallback_least_squares: bool; fallback_tol: float
    max_step: object; flat_max_step: object; max_retry_subdivisions: int
    edge_mask_arr: np.ndarray; gear2_be_fallback: bool
    integration_method: str
    adaptive: bool; adaptive_config: object


@dataclass(frozen=True, slots=True)
class _RuntimeCaches:
    """Per-run mutable caches/counters kept out of the immutable problem data."""
    op_cache_valid: np.ndarray
    op_cache_vs1: np.ndarray
    op_cache_vd1: np.ndarray
    stats: _NewtonStats


class _TransientCtx:
    """Immutable marshalled solve context shared by the transient kernels.

    Built once by :func:`_marshal_transient` from the grouped sub-contexts
    (:class:`_TopologyCtx`, :class:`_DeviceCtx`, ...).  Their fields are
    **flattened onto this object at construction**, so the hot kernels read
    ``ctx.dev_d_kind`` as a direct instance-dict lookup.  (An earlier
    ``__getattr__`` proxy resolved each flat name by linear-searching the six
    groups on *every* access — ~half of transient runtime, since the inner
    kernels touch dozens of fields per step.)  The grouped objects stay
    reachable as ``ctx.topology`` etc.

    Treated as immutable: nothing rebinds a field after construction; the
    op-cache arrays and the ``stats`` counter are mutated in place.
    """
    _GROUPS = ("topology", "devices", "passives", "sources", "solver", "runtime")

    def __init__(self, *, topology, devices, passives, sources, solver, runtime):
        d = self.__dict__
        for name, group in zip(
                self._GROUPS,
                (topology, devices, passives, sources, solver, runtime)):
            d[name] = group
            for f in fields(group):
                d[f.name] = getattr(group, f.name)


@dataclass(frozen=True, slots=True)
class _TransientMarshal:
    """Return bundle from transient input/topology marshalling."""
    ctx: _TransientCtx
    tgrid: np.ndarray
    input_keys: tuple
    input_values: np.ndarray
    inputs: dict
    node_inputs: dict
    V0: np.ndarray
    Vhist: np.ndarray
    edge_mask_arr: np.ndarray
    profile: bool
    gear2_retry_requested: bool


_NUMBA_SHARED_ARG_GROUPS = (
    ("solver", (
        "n", "maxit", "step_limit", "vtol", "gmin",
        "fallback_accept", "fallback_tol", "HH",
    )),
    ("device_terms", (
        "dev_d_kind", "dev_d_ref", "dev_d_val",
        "dev_g_kind", "dev_g_ref", "dev_g_val",
        "dev_s_kind", "dev_s_ref", "dev_s_val",
    )),
    ("device_nodes", ("dev_di", "dev_gi", "dev_si", "dev_use_abs")),
    ("model_params", (
        "p_Vfb", "p_Vss", "p_Lc", "p_lambda", "p_contact_scale",
        "p_exponent", "p_current_scale", "p_inv_Rleak",
        "p_two_over_pi", "p_cap_cgs1", "p_cap_cgd1",
        "p_cap_half_wl_ci", "p_cap_cgs3_base", "p_cap_cgd3_base",
        "p_k1", "p_gate_leak_g",
    )),
    ("op_cache", ("op_cache_valid", "op_cache_vs1", "op_cache_vd1")),
    ("passives", (
        "res_a_kind", "res_a_ref", "res_a_val",
        "res_b_kind", "res_b_ref", "res_b_val",
        "res_ai", "res_bi", "res_g",
        "cap_a_kind", "cap_a_ref", "cap_a_val",
        "cap_b_kind", "cap_b_ref", "cap_b_val",
        "cap_ai", "cap_bi", "cap_value",
    )),
    ("sources", (
        "isrc_pi", "isrc_qi", "isrc_value",
        "dyn_pi", "dyn_qi", "dyn_input_idx",
    )),
    ("cap_clip", ("cap_mode", "clip_lo", "clip_hi")),
    ("vsources", (
        "vs_a_kind", "vs_a_ref", "vs_a_val",
        "vs_b_kind", "vs_b_ref", "vs_b_val",
        "vs_pi", "vs_qi", "vs_bi", "vs_e_const", "vs_e_idx",
    )),
    ("vcvs", (
        "vcvs_a_kind", "vcvs_a_ref", "vcvs_a_val",
        "vcvs_b_kind", "vcvs_b_ref", "vcvs_b_val",
        "vcvs_cp_kind", "vcvs_cp_ref", "vcvs_cp_val",
        "vcvs_cn_kind", "vcvs_cn_ref", "vcvs_cn_val",
        "vcvs_pi", "vcvs_qi", "vcvs_cpi", "vcvs_cni", "vcvs_bi", "vcvs_mu",
    )),
    ("cccs", ("cccs_pi", "cccs_qi", "cccs_ctrl_bi", "cccs_beta")),
    ("ccvs", (
        "ccvs_a_kind", "ccvs_a_ref", "ccvs_a_val",
        "ccvs_b_kind", "ccvs_b_ref", "ccvs_b_val",
        "ccvs_pi", "ccvs_qi", "ccvs_bi", "ccvs_ctrl_bi", "ccvs_gamma",
    )),
)
_NUMBA_GRID_ARG_GROUPS = (
    ("run", ("V0", "tgrid", "input_values", "edge_mask", "profile_enabled")),
    ("step", ("max_step", "flat_max_step", "max_retry_subdivisions")),
) + _NUMBA_SHARED_ARG_GROUPS
_NUMBA_GRID_ARG_NAMES = tuple(
    group_name for group_name, _field_names in _NUMBA_GRID_ARG_GROUPS)
_NUMBA_ADAPTIVE_GEAR2_ARG_GROUPS = (
    ("run", ("V0", "tgrid_src", "input_values_src", "profile_enabled")),
    ("step", (
        "max_step", "adaptive_reltol", "adaptive_vabstol",
        "adaptive_iabstol", "adaptive_max_steps", "adaptive_h0",
    )),
) + _NUMBA_SHARED_ARG_GROUPS
_NUMBA_ADAPTIVE_GEAR2_ARG_NAMES = tuple(
    group_name for group_name, _field_names in _NUMBA_ADAPTIVE_GEAR2_ARG_GROUPS)


def _checked_numba_args(names, args):
    if len(args) != len(names):
        raise AssertionError(f"numba arg packer produced {len(args)} args for {len(names)} names")
    return args


def _checked_numba_arg_groups(groups, args):
    _checked_numba_args(tuple(name for name, _fields in groups), args)
    for (name, field_names), values in zip(groups, args):
        if len(values) != len(field_names):
            raise AssertionError(
                f"numba arg group {name!r} produced {len(values)} fields "
                f"for {len(field_names)} names")
    return args


def _numba_shared_kernel_arg_groups(ctx):
    """Shared grouped tail for transient Numba kernels.

    Keep all Python-side kernel calls going through this function so a
    device/source/stamp field cannot be added to only one execution path. The
    Python/Numba boundary sees semantic groups; individual hot-path kernels may
    still use positional scalars internally for nopython performance.
    """
    return (
        (
            int(ctx.n),
            int(ctx.newton_maxit),
            float(ctx.newton_step_limit),
            float(ctx.newton_vtol),
            float(ctx.gmin),
            bool(ctx.fallback_full_jacobian or ctx.fallback_least_squares),
            float(ctx.fallback_tol),
            float(ctx.HH),
        ),
        (
            ctx.dev_d_kind,
            ctx.dev_d_ref,
            ctx.dev_d_val,
            ctx.dev_g_kind,
            ctx.dev_g_ref,
            ctx.dev_g_val,
            ctx.dev_s_kind,
            ctx.dev_s_ref,
            ctx.dev_s_val,
        ),
        (
            ctx.dev_di,
            ctx.dev_gi,
            ctx.dev_si,
            ctx.dev_use_abs,
        ),
        (
            ctx.p_Vfb,
            ctx.p_Vss,
            ctx.p_Lc,
            ctx.p_lambda,
            ctx.p_contact_scale,
            ctx.p_exponent,
            ctx.p_current_scale,
            ctx.p_inv_Rleak,
            ctx.p_two_over_pi,
            ctx.p_cap_cgs1,
            ctx.p_cap_cgd1,
            ctx.p_cap_half_wl_ci,
            ctx.p_cap_cgs3_base,
            ctx.p_cap_cgd3_base,
            ctx.p_k1,
            ctx.p_gate_leak_g,
        ),
        (
            ctx.op_cache_valid,
            ctx.op_cache_vs1,
            ctx.op_cache_vd1,
        ),
        (
            ctx.res_a_kind,
            ctx.res_a_ref,
            ctx.res_a_val,
            ctx.res_b_kind,
            ctx.res_b_ref,
            ctx.res_b_val,
            ctx.res_ai,
            ctx.res_bi,
            ctx.res_g,
            ctx.cap_a_kind,
            ctx.cap_a_ref,
            ctx.cap_a_val,
            ctx.cap_b_kind,
            ctx.cap_b_ref,
            ctx.cap_b_val,
            ctx.cap_ai,
            ctx.cap_bi,
            ctx.cap_value,
        ),
        (
            ctx.isrc_pi,
            ctx.isrc_qi,
            ctx.isrc_value,
            ctx.dyn_pi,
            ctx.dyn_qi,
            ctx.dyn_input_idx,
        ),
        (
            int(ctx.cap_id),
            float(ctx.clip_lo),
            float(ctx.clip_hi),
        ),
        (
            ctx.vs_a_kind, ctx.vs_a_ref, ctx.vs_a_val,
            ctx.vs_b_kind, ctx.vs_b_ref, ctx.vs_b_val,
            ctx.vs_pi, ctx.vs_qi, ctx.vs_bi, ctx.vs_e_const, ctx.vs_e_idx,
        ),
        (
            ctx.vcvs_a_kind, ctx.vcvs_a_ref, ctx.vcvs_a_val,
            ctx.vcvs_b_kind, ctx.vcvs_b_ref, ctx.vcvs_b_val,
            ctx.vcvs_cp_kind, ctx.vcvs_cp_ref, ctx.vcvs_cp_val,
            ctx.vcvs_cn_kind, ctx.vcvs_cn_ref, ctx.vcvs_cn_val,
            ctx.vcvs_pi, ctx.vcvs_qi, ctx.vcvs_cpi, ctx.vcvs_cni, ctx.vcvs_bi, ctx.vcvs_mu,
        ),
        (ctx.cccs_pi, ctx.cccs_qi, ctx.cccs_ctrl_bi, ctx.cccs_beta),
        (
            ctx.ccvs_a_kind, ctx.ccvs_a_ref, ctx.ccvs_a_val,
            ctx.ccvs_b_kind, ctx.ccvs_b_ref, ctx.ccvs_b_val,
            ctx.ccvs_pi, ctx.ccvs_qi, ctx.ccvs_bi, ctx.ccvs_ctrl_bi, ctx.ccvs_gamma,
        ),
    )


def _numba_grid_kernel_args(ctx, V0, tgrid, input_values, edge_mask_arr, profile):
    max_step_arg = -1.0 if ctx.max_step is None else float(ctx.max_step)
    flat_max_step_arg = -1.0 if ctx.flat_max_step is None else float(ctx.flat_max_step)
    args = (
        (
            np.asarray(V0, float),
            np.asarray(tgrid, float),
            np.asarray(input_values, float),
            edge_mask_arr,
            bool(profile),
        ),
        (
            max_step_arg,
            flat_max_step_arg,
            int(ctx.max_retry_subdivisions),
        ),
    ) + _numba_shared_kernel_arg_groups(ctx)
    return _checked_numba_arg_groups(_NUMBA_GRID_ARG_GROUPS, args)


def _numba_adaptive_gear2_kernel_args(ctx, V0, tgrid, input_values, profile):
    max_step_arg = -1.0 if ctx.max_step is None else float(ctx.max_step)
    acfg = ctx.adaptive_config
    h0_arg = -1.0 if acfg.h0 is None else float(acfg.h0)
    args = (
        (
            np.asarray(V0, float),
            np.asarray(tgrid, float),
            np.asarray(input_values, float),
            bool(profile),
        ),
        (
            max_step_arg,
            float(acfg.reltol),
            float(acfg.vabstol),
            float(acfg.iabstol),
            int(acfg.max_steps),
            h0_arg,
        ),
    ) + _numba_shared_kernel_arg_groups(ctx)
    return _checked_numba_arg_groups(_NUMBA_ADAPTIVE_GEAR2_ARG_GROUPS, args)


def _cap_mode_to_id(cap_mode):
    if cap_mode is None:
        return None
    key = str(cap_mode).lower()
    if key not in _CAP_MODE_IDS:
        raise ValueError(f"unknown cap_mode {cap_mode!r}; expected one of {sorted(_CAP_MODE_IDS)}")
    return _CAP_MODE_IDS[key]


def _solve_fixed_gear2_numba(ctx, V0, tgrid, input_values, edge_mask_arr,
                             profile, gear2_retry_requested):
    out = _PathOutcome()
    if not (not ctx.adaptive and ctx.integration_method == "gear2" and
            ctx.n_aug >= ctx.n):
        return out
    try:
        t_profile0 = time.perf_counter()
        # Numba gear2 owns both periodic single-step grids and raw transient
        # maxstep/retry subdivision. If it rejects a robust step, Python retry
        # remains the correctness fallback in the driver.
        g2_orig_idx = None
        g2 = _transient_solve_grid_gear2_impl(
            *_numba_grid_kernel_args(ctx, V0, tgrid, input_values,
                                     edge_mask_arr, profile))
        ok_g2, Vfast, fast_substeps, fail_index, raw_profile, _rfi = g2
        out.profile_wall_s = time.perf_counter() - t_profile0
        if ok_g2:
            out.Vhist = (np.ascontiguousarray(Vfast[g2_orig_idx])
                         if g2_orig_idx is not None else Vfast)
            out.nsubsteps = int(fast_substeps)
            raw_profile_arr = np.asarray(raw_profile, float)
            out.nretry = int(raw_profile_arr[PROFILE_FAILED_SUBSTEPS])
            out.nfail = int(raw_profile_arr[PROFILE_FAILED_INTERVALS])
            out.gear2_done = True
            out.gear2_numba_used = True
            out.profile_stats = raw_profile_arr
            out.handled = True
        elif gear2_retry_requested:
            out.numba_grid_failed_index = int(fail_index)
            out.numba_grid_failed_substeps = int(fast_substeps)
            out.numba_grid_failed_profile = np.asarray(raw_profile, float)
            out.numba_grid_failed_intervals = (
                None if _rfi is None else np.asarray(_rfi, int))
    except Exception as exc:
        out.numba_grid_error = f"gear2: {type(exc).__name__}: {exc}"
        diagnostics.note("transient.numba_grid_gear2_error", exc)
    return out


def _solve_adaptive_gear2_numba(ctx, V0, tgrid, input_values, profile):
    out = _PathOutcome()
    if not (ctx.adaptive and ctx.integration_method == "gear2" and
            ctx.n_aug >= ctx.n):
        return out
    try:
        t_profile0 = time.perf_counter()
        ad = _transient_solve_adaptive_gear2_impl(
            *_numba_adaptive_gear2_kernel_args(ctx, V0, tgrid,
                                               input_values, profile))
        ok_ad, tfast, Vfast, input_fast, naccept, fast_substeps, fast_rejects, raw_profile = ad
        if ok_ad:
            naccept = int(naccept)
            out.tgrid = np.ascontiguousarray(tfast[:naccept])
            out.Vhist = np.ascontiguousarray(Vfast[:naccept])
            out.input_values = np.ascontiguousarray(input_fast[:naccept].T)
            out.N = len(out.tgrid)
            out.nsubsteps = int(fast_substeps)
            out.nretry = int(fast_rejects)
            out.profile_stats = np.asarray(raw_profile, float)
            out.profile_wall_s = time.perf_counter() - t_profile0
            out.gear2_done = True
            out.adaptive_used = True
            out.adaptive_numba_used = True
            out.gear2_numba_used = True
            out.handled = True
    except Exception as exc:
        out.numba_grid_error = f"adaptive_gear2: {type(exc).__name__}: {exc}"
        diagnostics.note("transient.numba_adaptive_error", exc)
    return out


def _solve_be_numba(ctx, V0, tgrid, input_values, edge_mask_arr, profile):
    out = _PathOutcome()
    if not ((not ctx.adaptive) and ctx.n_aug >= ctx.n):
        return out
    try:
        t_profile0 = time.perf_counter()
        grid_result = _transient_solve_grid_impl(
            *_numba_grid_kernel_args(ctx, V0, tgrid, input_values,
                                     edge_mask_arr, profile))
        if len(grid_result) == 5:
            ok_grid, Vfast, fast_substeps, fail_index, raw_profile = grid_result
            raw_failed_intervals = None
        else:
            (ok_grid, Vfast, fast_substeps, fail_index, raw_profile,
             raw_failed_intervals) = grid_result
        out.profile_wall_s = time.perf_counter() - t_profile0
        out.numba_grid_failed_intervals = (
            None if raw_failed_intervals is None else np.asarray(raw_failed_intervals, int))
        if ok_grid:
            out.Vhist = Vfast
            out.nsubsteps = int(fast_substeps)
            raw_profile_arr = np.asarray(raw_profile, float)
            out.nfail = int(raw_profile_arr[PROFILE_FAILED_INTERVALS])
            out.nretry = out.nfail
            ctx.stats.attempts += out.nsubsteps
            ctx.stats.success += out.nsubsteps
            out.used_grid_numba = True
            out.profile_stats = raw_profile_arr
            out.handled = True
        else:
            out.numba_grid_failed_index = int(fail_index)
            out.numba_grid_failed_substeps = int(fast_substeps)
            out.numba_grid_failed_profile = np.asarray(raw_profile, float)
            if out.numba_grid_failed_index is not None and out.numba_grid_failed_index > 1:
                out.Vhist = Vfast
                out.nsubsteps = int(fast_substeps)
                ctx.stats.attempts += out.nsubsteps
                ctx.stats.success += out.nsubsteps
                out.partial_grid_numba = True
                out.python_start_idx = int(out.numba_grid_failed_index)
    except Exception as exc:
        out.numba_grid_error = f"{type(exc).__name__}: {exc}"
        diagnostics.note("transient.numba_be_grid_error", exc)
    return out


def _assemble_result(ctx, tgrid, Vhist, input_values, input_keys,
                     nfail, nretry, nsubsteps,
                     adaptive_used, adaptive_numba_used,
                     used_grid_numba, gear2_numba_used,
                     gear2_python_retry_used,
                     profile, profile_stats, profile_wall_s,
                     partial_grid_numba, numba_grid_error,
                     numba_grid_failed_index, numba_grid_failed_substeps,
                     numba_grid_failed_profile, numba_grid_failed_intervals):
    def pval(stats, slot, default=0.0):
        return float(stats[slot]) if stats is not None and len(stats) > slot else float(default)

    def pint(stats, slot, default=0):
        return int(pval(stats, slot, default))

    N = len(tgrid)
    idx = ctx.idx
    plan = ctx.plan
    nodes = {nm: Vhist[:, idx[nm]] for nm in plan.solved}
    out = np.zeros(N)
    for node, weight in plan.output_weights.items():
        out += weight * nodes[node]
    result = {"t": tgrid, "output": out, "vout": out, "nfail": nfail,
              "nretry": nretry, "nsubsteps": nsubsteps, "nodes": nodes,
              "transient_cap_mode": _CAP_MODE,
              "transient_cap_mode_id": int(ctx.cap_id)}
    if adaptive_used:
        result["adaptive"] = True
        result["adaptive_reltol"] = float(ctx.adaptive_config.reltol)
        result["adaptive_vabstol"] = float(ctx.adaptive_config.vabstol)
        result["adaptive_iabstol"] = float(ctx.adaptive_config.iabstol)
        result["adaptive_accepted_steps"] = int(max(0, len(tgrid) - 1))
        result["adaptive_rejected_steps"] = int(nretry)
        result["inputs"] = {
            key: input_values[pos].copy()
            for pos, key in enumerate(input_keys)
        }
    result["numba_grid_solver"] = bool(used_grid_numba or gear2_numba_used)
    result["numba_adaptive_solver"] = bool(adaptive_numba_used)
    if ctx.integration_method == "gear2":
        result["gear2_python_retry_solver"] = bool(gear2_python_retry_used)
    if ctx.stats.attempts:
        result["numba_newton_attempts"] = ctx.stats.attempts
        result["numba_newton_success"] = ctx.stats.success
        result["numba_newton_fallback"] = ctx.stats.fallback
    if profile:
        if profile_stats is None:
            profile_stats = np.zeros(PROFILE_LEN, dtype=float)
            profile_stats[PROFILE_NEWTON_ITERS] = 0.0
            profile_stats[PROFILE_EDGE_SUBSTEPS] = 0.0
            profile_stats[PROFILE_FLAT_SUBSTEPS] = float(nsubsteps)
            profile_stats[PROFILE_FAILED_SUBSTEPS] = float(nfail)
            profile_stats[PROFILE_INTERVALS] = float(N - 1)
            profile_stats[PROFILE_SUBSTEPS] = float(nsubsteps)
        total_iters = float(profile_stats[PROFILE_NEWTON_ITERS])
        edge_iters = float(profile_stats[PROFILE_EDGE_NEWTON_ITERS])
        flat_iters = float(profile_stats[PROFILE_FLAT_NEWTON_ITERS])
        iter_work = edge_iters + flat_iters
        edge_time_est = profile_wall_s * edge_iters / iter_work if iter_work else 0.0
        flat_time_est = profile_wall_s * flat_iters / iter_work if iter_work else 0.0
        result["transient_profile"] = {
            "enabled": True,
            "numba_grid_solver": bool(used_grid_numba or gear2_numba_used),
            "numba_grid_partial": bool(partial_grid_numba),
            "numba_grid_error": numba_grid_error,
            "numba_grid_failed_index": numba_grid_failed_index,
            "numba_grid_failed_substeps": int(numba_grid_failed_substeps),
            "numba_grid_failed_newton_iters": (
                pint(numba_grid_failed_profile, PROFILE_NEWTON_ITERS)),
            "numba_grid_failed_substep_failures": (
                pint(numba_grid_failed_profile, PROFILE_FAILED_SUBSTEPS)),
            "numba_grid_failed_interval_failures": (
                pint(numba_grid_failed_profile, PROFILE_FAILED_INTERVALS)),
            "numba_grid_failed_last_residual_inf": (
                pval(numba_grid_failed_profile, PROFILE_FAILED_LAST_RESIDUAL_INF)),
            "numba_grid_failed_max_residual_inf": (
                pval(numba_grid_failed_profile, PROFILE_FAILED_MAX_RESIDUAL_INF)),
            "numba_grid_failed_last_step_inf": (
                pval(numba_grid_failed_profile, PROFILE_FAILED_LAST_STEP_INF)),
            "numba_grid_failed_max_step_inf": (
                pval(numba_grid_failed_profile, PROFILE_FAILED_MAX_STEP_INF)),
            "numba_grid_failed_stamp_or_prev_count": (
                pint(numba_grid_failed_profile, PROFILE_FAILED_STAMP_OR_PREV_COUNT)),
            "numba_grid_failed_linear_solve_count": (
                pint(numba_grid_failed_profile, PROFILE_FAILED_LINEAR_SOLVE_COUNT)),
            "numba_grid_failed_maxit_count": (
                pint(numba_grid_failed_profile, PROFILE_FAILED_MAXIT_COUNT)),
            "wall_time_s": float(profile_wall_s),
            "nsubsteps": int(nsubsteps),
            "intervals": int(profile_stats[PROFILE_INTERVALS]),
            "newton_iters_total": int(profile_stats[PROFILE_NEWTON_ITERS]),
            "newton_iters_avg": float(total_iters / nsubsteps) if nsubsteps else 0.0,
            "pmos_op_solves": int(profile_stats[PROFILE_PMOS_OP_SOLVES]),
            "pmos_internal_newton_attempts": int(profile_stats[PROFILE_PMOS_INTERNAL_NEWTON_ATTEMPTS]),
            "pmos_internal_newton_iters": int(profile_stats[PROFILE_PMOS_INTERNAL_NEWTON_ITERS]),
            "pmos_internal_newton_iters_avg": (
                float(profile_stats[PROFILE_PMOS_INTERNAL_NEWTON_ITERS] / profile_stats[PROFILE_PMOS_INTERNAL_NEWTON_ATTEMPTS]) if profile_stats[PROFILE_PMOS_INTERNAL_NEWTON_ATTEMPTS] else 0.0),
            "internal_fd_jac_fallbacks": int(profile_stats[PROFILE_INTERNAL_FD_JAC_FALLBACKS]),
            "terminal_fd_jac_fallbacks": int(profile_stats[PROFILE_TERMINAL_FD_JAC_FALLBACKS]),
            "edge_substeps": int(profile_stats[PROFILE_EDGE_SUBSTEPS]),
            "flat_substeps": int(profile_stats[PROFILE_FLAT_SUBSTEPS]),
            "edge_newton_iters": int(profile_stats[PROFILE_EDGE_NEWTON_ITERS]),
            "flat_newton_iters": int(profile_stats[PROFILE_FLAT_NEWTON_ITERS]),
            "failed_substeps": int(profile_stats[PROFILE_FAILED_SUBSTEPS]),
            "failed_intervals": int(profile_stats[PROFILE_FAILED_INTERVALS]),
            "failed_edge_intervals": int(profile_stats[PROFILE_FAILED_EDGE_INTERVALS]),
            "failed_flat_intervals": int(profile_stats[PROFILE_FAILED_FLAT_INTERVALS]),
            "failed_interval_indices": (
                [int(v) for v in numba_grid_failed_intervals
                 if int(v) >= 0]
                if numba_grid_failed_intervals is not None else []
            ),
            "failed_last_residual_inf": (
                pval(profile_stats, PROFILE_FAILED_LAST_RESIDUAL_INF)),
            "failed_max_residual_inf": (
                pval(profile_stats, PROFILE_FAILED_MAX_RESIDUAL_INF)),
            "failed_last_step_inf": (
                pval(profile_stats, PROFILE_FAILED_LAST_STEP_INF)),
            "failed_max_step_inf": (
                pval(profile_stats, PROFILE_FAILED_MAX_STEP_INF)),
            "failed_stamp_or_prev_count": (
                pint(profile_stats, PROFILE_FAILED_STAMP_OR_PREV_COUNT)),
            "failed_linear_solve_count": (
                pint(profile_stats, PROFILE_FAILED_LINEAR_SOLVE_COUNT)),
            "failed_maxit_count": (
                pint(profile_stats, PROFILE_FAILED_MAXIT_COUNT)),
            "stalled_residual_accepts": (
                pint(profile_stats, PROFILE_STALLED_RESIDUAL_ACCEPTS)),
            "edge_time_s_est": float(edge_time_est),
            "flat_time_s_est": float(flat_time_est),
            "time_estimate_basis": "newton_iteration_weighted",
        }
    if "VOP" in idx:
        result["vop"] = nodes["VOP"]
    if "VON" in idx:
        result["von"] = nodes["VON"]
    return result


def _marshal_transient(
        sizes, bias, tgrid, *, vip=None, vin=None, nf=None, V0=None,
        topo=AFE_TOPO, inputs=None, node_inputs=None, current_inputs=None,
        corner=None, model_types=None, device_kwargs=None,
        max_step=None, flat_max_step=None,
        max_retry_subdivisions=0, newton_maxit=30,
        newton_step_limit=5.0, newton_vtol=1e-8,
        fallback_full_jacobian=False, fallback_least_squares=False,
        fallback_tol=1e-9, signed_devices=None, profile=False,
        edge_mask=None, rail_margin=None, integration_method="be",
        gear2_be_fallback=True, cap_mode=None, cap_mode_id=None,
        adaptive=False, adaptive_reltol=1e-4, adaptive_vabstol=1e-6,
        adaptive_iabstol=1e-12, adaptive_max_steps=200000,
        adaptive_h0=None, adaptive_config=None):
    """Validate public inputs and compile the immutable transient solve context."""
    integration_method = str(integration_method).lower()
    adaptive_config = resolve_adaptive_config(
        adaptive_config,
        adaptive_reltol=adaptive_reltol,
        adaptive_vabstol=adaptive_vabstol,
        adaptive_iabstol=adaptive_iabstol,
        adaptive_max_steps=adaptive_max_steps,
        adaptive_h0=adaptive_h0,
    )
    if adaptive and integration_method != "gear2":
        raise ValueError("adaptive transient requires integration_method='gear2'")
    if cap_mode is not None and cap_mode_id is not None:
        requested = _cap_mode_to_id(cap_mode)
        if int(cap_mode_id) != int(requested):
            raise ValueError("cap_mode and cap_mode_id disagree")

    tft = build_devices(sizes, nf=nf, corner=corner, topo=topo,
                        model_types=model_types, device_kwargs=device_kwargs)
    tgrid = np.asarray(tgrid, float)
    N = len(tgrid)

    # Per-call cap operator override (default = the module/env-selected mode).
    # The chopper PSS uses the trapezoidal "average" mode (id 1) -- a STABLE,
    # non-conservative C(V)*dV/dt discretization that matches Cadence's commutation
    # feedthrough (charge Q-stamp over-swings it ~26%); charge stays the default
    # everywhere else (it is L-stable on stiff tau>>T circuits where average rings).
    cap_mode_from_name = _cap_mode_to_id(cap_mode)
    _cap_id = int(_CAP_MODE_ID if cap_mode_from_name is None and cap_mode_id is None
                  else cap_mode_from_name if cap_mode_from_name is not None
                  else cap_mode_id)
    if _cap_id not in (0, 1):
        raise ValueError("cap_mode_id must be 0 (charge) or 1 (average)")

    if inputs is None:
        inputs = {}
        if vip is not None:
            inputs["vip"] = vip
        if vin is not None:
            inputs["vin"] = vin
    inputs = {key: np.asarray(val, float) for key, val in inputs.items()}
    for key, val in inputs.items():
        if len(val) != N:
            raise ValueError(f"Input waveform {key!r} length {len(val)} != len(tgrid) {N}")
    input_keys = tuple(inputs)
    input_values = (np.vstack([inputs[key] for key in input_keys])
                    if input_keys else np.empty((0, N), float))
    node_inputs = dict(node_inputs or {})
    for node, key in node_inputs.items():
        if key not in inputs:
            raise ValueError(f"node_inputs[{node!r}] references missing waveform {key!r}")

    plan = CompiledTopology(topo, bias, input_keys=input_keys,
                            node_inputs=node_inputs, transient_inputs=True)
    idx, n = plan.idx, plan.n
    n_aug = plan.n_aug                 # n nodes + m ideal-voltage-source branch currents
    termv = plan.term_value

    # Every device uses its signed Verilog-A drain current. abs(Idc) was only
    # correct for never-reversing devices (forward PMOS: I_d1_d>0 so signed==abs)
    # but turned a *reverse*-biased pass-gate switch into an anti-restoring pump.
    # `signed_devices` is retained as a no-op compatibility parameter.
    signed_devices = set(signed_devices or ())
    dev_meta = [(tft[item.name], True,
                 item.d, item.g, item.s, item.di, item.gi, item.si)
                for item in plan.devices]
    load_meta = [(item.a, item.b, item.ai, item.bi, item.value)
                 for item in plan.capacitors]
    res_meta = [(item.a, item.b, item.ai, item.bi, item.g)
                for item in plan.resistors]
    isrc_meta = [(item.pi, item.qi, item.value) for item in plan.isources]
    vccs_meta = [(item.pi, item.qi, item.cp, item.cn, item.gm)
                 for item in plan.vccs]
    # Ideal voltage sources (true MNA, Python path): (a_term, b_term, pi, qi, bi,
    # e_const, e_input_idx). Branch current is the unknown at V[bi]; constraint row
    # bi pins V_p - V_q = E. Vsource circuits force the pure-Python n_aug path.
    vs_meta = [(item.p, item.q, item.pi, item.qi, item.bi, item.e_const, item.e_input_idx)
               for item in plan.vsources]
    vcvs_meta = [(item.p, item.q, item.cp, item.cn, item.pi, item.qi,
                  item.cpi, item.cni, item.bi, item.mu)
                 for item in plan.vcvs]
    cccs_meta = [(item.pi, item.qi, item.ctrl_bi, item.beta)
                 for item in plan.cccs]
    ccvs_meta = [(item.p, item.q, item.pi, item.qi, item.bi,
                  item.ctrl_bi, item.gamma)
                 for item in plan.ccvs]

    dyn_isrc_meta = []
    for pos, item in enumerate(current_inputs or ()):
        if isinstance(item, dict):
            p_node = item["p"]
            q_node = item["q"]
            key = item["input"]
        else:
            p_node, q_node, key = item
        if key not in plan.input_index:
            raise ValueError(f"current_inputs[{pos}] references missing waveform {key!r}")
        pterm = plan.compile_term(p_node)
        qterm = plan.compile_term(q_node)
        dyn_isrc_meta.append((
            plan.solved_index(pterm),
            plan.solved_index(qterm),
            plan.input_index[key],
        ))

    dev_d_kind, dev_d_ref, dev_d_val = term_arrays([item[2] for item in dev_meta])
    dev_g_kind, dev_g_ref, dev_g_val = term_arrays([item[3] for item in dev_meta])
    dev_s_kind, dev_s_ref, dev_s_val = term_arrays([item[4] for item in dev_meta])
    dev_di = index_array(item[5] for item in dev_meta)
    dev_gi = index_array(item[6] for item in dev_meta)
    dev_si = index_array(item[7] for item in dev_meta)
    dev_use_abs = np.array([not item[1] for item in dev_meta], dtype=np.bool_)
    dev_objs = [item[0] for item in dev_meta]
    _np_params = [d.get_numba_params() for d in dev_objs]
    p_Vfb = np.array([p.Vfb for p in _np_params], dtype=float)
    p_Vss = np.array([p.Vss for p in _np_params], dtype=float)
    p_Lc = np.array([p.Lc for p in _np_params], dtype=float)
    p_lambda = np.array([p.lambda_ for p in _np_params], dtype=float)
    p_contact_scale = np.array([p.contact_scale for p in _np_params], dtype=float)
    p_exponent = np.array([p.channel_exponent for p in _np_params], dtype=float)
    p_current_scale = np.array([p.current_scale for p in _np_params], dtype=float)
    p_inv_Rleak = np.array([p.inv_Rleak for p in _np_params], dtype=float)
    p_two_over_pi = np.array([p.two_over_pi for p in _np_params], dtype=float)
    p_cap_cgs1 = np.array([p.cap_cgs1 for p in _np_params], dtype=float)
    p_cap_cgd1 = np.array([p.cap_cgd1 for p in _np_params], dtype=float)
    p_cap_half_wl_ci = np.array([p.cap_half_wl_ci for p in _np_params], dtype=float)
    p_cap_cgs3_base = np.array([p.cap_cgs3_base for p in _np_params], dtype=float)
    p_cap_cgd3_base = np.array([p.cap_cgd3_base for p in _np_params], dtype=float)
    p_k1 = np.array([p.k1 for p in _np_params], dtype=float)
    p_gate_leak_g = np.array([p.gate_leak_g for p in _np_params], dtype=float)
    op_cache_valid = np.zeros(len(dev_meta), dtype=np.bool_)
    op_cache_vs1 = np.zeros(len(dev_meta), dtype=float)
    op_cache_vd1 = np.zeros(len(dev_meta), dtype=float)

    res_a_kind, res_a_ref, res_a_val = term_arrays([item[0] for item in res_meta])
    res_b_kind, res_b_ref, res_b_val = term_arrays([item[1] for item in res_meta])
    res_ai = index_array(item[2] for item in res_meta)
    res_bi = index_array(item[3] for item in res_meta)
    res_g = np.array([item[4] for item in res_meta], dtype=float)

    cap_a_kind, cap_a_ref, cap_a_val = term_arrays([item[0] for item in load_meta])
    cap_b_kind, cap_b_ref, cap_b_val = term_arrays([item[1] for item in load_meta])
    cap_ai = index_array(item[2] for item in load_meta)
    cap_bi = index_array(item[3] for item in load_meta)
    cap_value = np.array([item[4] for item in load_meta], dtype=float)

    isrc_pi = index_array(item[0] for item in isrc_meta)
    isrc_qi = index_array(item[1] for item in isrc_meta)
    isrc_value = np.array([item[2] for item in isrc_meta], dtype=float)

    vccs_pi = index_array(item[0] for item in vccs_meta)
    vccs_qi = index_array(item[1] for item in vccs_meta)
    # For control nodes, extract solved index from the terminal tuple; rails -> -1.
    vccs_cpi = index_array(
        item[2][1] if item[2][0] == 0 else None for item in vccs_meta)
    vccs_cni = index_array(
        item[3][1] if item[3][0] == 0 else None for item in vccs_meta)
    vccs_gm = np.array([item[4] for item in vccs_meta], dtype=float)

    dyn_pi = index_array(item[0] for item in dyn_isrc_meta)
    dyn_qi = index_array(item[1] for item in dyn_isrc_meta)
    dyn_input_idx = np.array([item[2] for item in dyn_isrc_meta], dtype=np.int64)

    # Branch-element Numba arrays (augmented n_aug > n path). Mirror the Python
    # _k_step_residual / _k_build_jac branch stamping; built unconditionally and
    # empty for n_aug == n circuits.
    vs_a_kind, vs_a_ref, vs_a_val = term_arrays([item[0] for item in vs_meta])
    vs_b_kind, vs_b_ref, vs_b_val = term_arrays([item[1] for item in vs_meta])
    vs_pi = index_array(item[2] for item in vs_meta)
    vs_qi = index_array(item[3] for item in vs_meta)
    vs_bi = index_array(item[4] for item in vs_meta)
    vs_e_const = np.array([item[5] for item in vs_meta], dtype=float)
    vs_e_idx = index_array(item[6] for item in vs_meta)

    vcvs_a_kind, vcvs_a_ref, vcvs_a_val = term_arrays([item[0] for item in vcvs_meta])
    vcvs_b_kind, vcvs_b_ref, vcvs_b_val = term_arrays([item[1] for item in vcvs_meta])
    vcvs_cp_kind, vcvs_cp_ref, vcvs_cp_val = term_arrays([item[2] for item in vcvs_meta])
    vcvs_cn_kind, vcvs_cn_ref, vcvs_cn_val = term_arrays([item[3] for item in vcvs_meta])
    vcvs_pi = index_array(item[4] for item in vcvs_meta)
    vcvs_qi = index_array(item[5] for item in vcvs_meta)
    vcvs_cpi = index_array(item[6] for item in vcvs_meta)
    vcvs_cni = index_array(item[7] for item in vcvs_meta)
    vcvs_bi = index_array(item[8] for item in vcvs_meta)
    vcvs_mu = np.array([item[9] for item in vcvs_meta], dtype=float)

    cccs_pi = index_array(item[0] for item in cccs_meta)
    cccs_qi = index_array(item[1] for item in cccs_meta)
    cccs_ctrl_bi = index_array(item[2] for item in cccs_meta)
    cccs_beta = np.array([item[3] for item in cccs_meta], dtype=float)

    ccvs_a_kind, ccvs_a_ref, ccvs_a_val = term_arrays([item[0] for item in ccvs_meta])
    ccvs_b_kind, ccvs_b_ref, ccvs_b_val = term_arrays([item[1] for item in ccvs_meta])
    ccvs_pi = index_array(item[2] for item in ccvs_meta)
    ccvs_qi = index_array(item[3] for item in ccvs_meta)
    ccvs_bi = index_array(item[4] for item in ccvs_meta)
    ccvs_ctrl_bi = index_array(item[5] for item in ccvs_meta)
    ccvs_gamma = np.array([item[6] for item in ccvs_meta], dtype=float)

    if rail_margin is None and getattr(topo, "require_dc_in_box", False):
        rail_margin = 2.0
    clip_lo = np.inf
    clip_hi = -np.inf
    if rail_margin is not None:
        rails = [v for v in plan.rails.values() if isinstance(v, (int, float))]
        if rails:
            clip_lo = min(rails) - float(rail_margin)
            clip_hi = max(rails) + float(rail_margin)

    if V0 is None:
        ac = ac_solve(sizes, bias, np.array([1.0]), nf=nf, topo=topo,
                      corner=corner, model_types=model_types,
                      device_kwargs=device_kwargs)
        dc = ac["dc_op"]
        V0 = np.array([dc[name] for name in topo.solved])
    V0 = np.asarray(V0, float)
    if V0.shape[0] < n_aug:                  # pad ideal-source branch currents
        V0 = np.concatenate([V0, np.zeros(n_aug - V0.shape[0])])
    if len(vs_meta) and input_values.shape[1] > 0:
        input0 = input_values[:, 0]
        Vseed = V0.copy()
        for aterm, bterm, pi, qi, _bi, e_const, e_idx in vs_meta:
            E = e_const if e_idx < 0 else input0[e_idx]
            if pi is not None and qi is not None:
                Vseed[pi] = Vseed[qi] + E
            elif pi is not None:
                Vseed[pi] = termv(bterm, Vseed, input0) + E
            elif qi is not None:
                Vseed[qi] = termv(aterm, Vseed, input0) - E
        V0 = Vseed
    Vhist = np.zeros((N, n_aug))
    Vhist[0] = V0

    if edge_mask is None:
        edge_mask_arr = np.empty(0, dtype=np.bool_)
    else:
        edge_mask_arr = np.asarray(edge_mask, dtype=np.bool_)
        if len(edge_mask_arr) != N:
            raise ValueError("edge_mask length must match tgrid")

    max_retry_subdivisions = int(max_retry_subdivisions or 0)
    max_step = None if max_step is None else float(max_step)
    flat_max_step = None if flat_max_step is None else float(flat_max_step)
    gmin = 1e-12
    HH = 1e-3   # finite-diff step for gm/gds (matches get_ss_params)
    gear2_retry_requested = (
        integration_method == "gear2" and gear2_be_fallback and
        (max_retry_subdivisions > 0 or
         (max_step is not None and max_step > 0.0) or
         (flat_max_step is not None and flat_max_step > 0.0))
    )

    topology_ctx = _TopologyCtx(
        plan=plan, idx=idx, n=n, n_aug=n_aug, termv=termv, tft=tft,
        signed_devices=signed_devices)
    device_ctx = _DeviceCtx(
        dev_meta=dev_meta, dev_objs=dev_objs,
        dev_d_kind=dev_d_kind, dev_d_ref=dev_d_ref, dev_d_val=dev_d_val,
        dev_g_kind=dev_g_kind, dev_g_ref=dev_g_ref, dev_g_val=dev_g_val,
        dev_s_kind=dev_s_kind, dev_s_ref=dev_s_ref, dev_s_val=dev_s_val,
        dev_di=dev_di, dev_gi=dev_gi, dev_si=dev_si, dev_use_abs=dev_use_abs,
        p_Vfb=p_Vfb, p_Vss=p_Vss, p_Lc=p_Lc, p_lambda=p_lambda,
        p_contact_scale=p_contact_scale, p_exponent=p_exponent,
        p_current_scale=p_current_scale, p_inv_Rleak=p_inv_Rleak,
        p_two_over_pi=p_two_over_pi, p_cap_cgs1=p_cap_cgs1, p_cap_cgd1=p_cap_cgd1,
        p_cap_half_wl_ci=p_cap_half_wl_ci, p_cap_cgs3_base=p_cap_cgs3_base,
        p_cap_cgd3_base=p_cap_cgd3_base, p_k1=p_k1, p_gate_leak_g=p_gate_leak_g)
    passive_ctx = _PassiveCtx(
        load_meta=load_meta, res_meta=res_meta,
        res_a_kind=res_a_kind, res_a_ref=res_a_ref, res_a_val=res_a_val,
        res_b_kind=res_b_kind, res_b_ref=res_b_ref, res_b_val=res_b_val,
        res_ai=res_ai, res_bi=res_bi, res_g=res_g,
        cap_a_kind=cap_a_kind, cap_a_ref=cap_a_ref, cap_a_val=cap_a_val,
        cap_b_kind=cap_b_kind, cap_b_ref=cap_b_ref, cap_b_val=cap_b_val,
        cap_ai=cap_ai, cap_bi=cap_bi, cap_value=cap_value)
    source_ctx = _SourceCtx(
        isrc_meta=isrc_meta, vccs_meta=vccs_meta, vs_meta=vs_meta,
        vcvs_meta=vcvs_meta, cccs_meta=cccs_meta, ccvs_meta=ccvs_meta,
        dyn_isrc_meta=dyn_isrc_meta,
        isrc_pi=isrc_pi, isrc_qi=isrc_qi, isrc_value=isrc_value,
        vccs_pi=vccs_pi, vccs_qi=vccs_qi, vccs_cpi=vccs_cpi,
        vccs_cni=vccs_cni, vccs_gm=vccs_gm,
        dyn_pi=dyn_pi, dyn_qi=dyn_qi, dyn_input_idx=dyn_input_idx,
        vs_a_kind=vs_a_kind, vs_a_ref=vs_a_ref, vs_a_val=vs_a_val,
        vs_b_kind=vs_b_kind, vs_b_ref=vs_b_ref, vs_b_val=vs_b_val,
        vs_pi=vs_pi, vs_qi=vs_qi, vs_bi=vs_bi,
        vs_e_const=vs_e_const, vs_e_idx=vs_e_idx,
        vcvs_a_kind=vcvs_a_kind, vcvs_a_ref=vcvs_a_ref, vcvs_a_val=vcvs_a_val,
        vcvs_b_kind=vcvs_b_kind, vcvs_b_ref=vcvs_b_ref, vcvs_b_val=vcvs_b_val,
        vcvs_cp_kind=vcvs_cp_kind, vcvs_cp_ref=vcvs_cp_ref, vcvs_cp_val=vcvs_cp_val,
        vcvs_cn_kind=vcvs_cn_kind, vcvs_cn_ref=vcvs_cn_ref, vcvs_cn_val=vcvs_cn_val,
        vcvs_pi=vcvs_pi, vcvs_qi=vcvs_qi, vcvs_cpi=vcvs_cpi,
        vcvs_cni=vcvs_cni, vcvs_bi=vcvs_bi, vcvs_mu=vcvs_mu,
        cccs_pi=cccs_pi, cccs_qi=cccs_qi,
        cccs_ctrl_bi=cccs_ctrl_bi, cccs_beta=cccs_beta,
        ccvs_a_kind=ccvs_a_kind, ccvs_a_ref=ccvs_a_ref, ccvs_a_val=ccvs_a_val,
        ccvs_b_kind=ccvs_b_kind, ccvs_b_ref=ccvs_b_ref, ccvs_b_val=ccvs_b_val,
        ccvs_pi=ccvs_pi, ccvs_qi=ccvs_qi, ccvs_bi=ccvs_bi,
        ccvs_ctrl_bi=ccvs_ctrl_bi, ccvs_gamma=ccvs_gamma)
    solver_opts = _SolverOptions(
        gmin=gmin, HH=HH, clip_lo=clip_lo, clip_hi=clip_hi, cap_id=_cap_id,
        rail_margin=rail_margin,
        newton_maxit=newton_maxit, newton_step_limit=newton_step_limit,
        newton_vtol=newton_vtol, fallback_full_jacobian=fallback_full_jacobian,
        fallback_least_squares=fallback_least_squares, fallback_tol=fallback_tol,
        max_step=max_step, flat_max_step=flat_max_step,
        max_retry_subdivisions=max_retry_subdivisions,
        edge_mask_arr=edge_mask_arr, gear2_be_fallback=gear2_be_fallback,
        integration_method=integration_method,
        adaptive=adaptive, adaptive_config=adaptive_config)
    runtime = _RuntimeCaches(
        op_cache_valid=op_cache_valid, op_cache_vs1=op_cache_vs1,
        op_cache_vd1=op_cache_vd1, stats=_NewtonStats())
    ctx = _TransientCtx(
        topology=topology_ctx, devices=device_ctx, passives=passive_ctx,
        sources=source_ctx, solver=solver_opts, runtime=runtime)
    return _TransientMarshal(
        ctx=ctx, tgrid=tgrid, input_keys=input_keys, input_values=input_values,
        inputs=inputs, node_inputs=node_inputs, V0=V0, Vhist=Vhist,
        edge_mask_arr=edge_mask_arr, profile=bool(profile),
        gear2_retry_requested=gear2_retry_requested)


def osdi_model_names(model_types):
    """Names in a ``model_types`` map bound to OSDI (compiled-VA) devices."""
    if not model_types:
        return ()
    from .device_model import get_model_class
    names = []
    for name, mt in model_types.items():
        cls = get_model_class(mt)
        if cls is not None and getattr(cls, "TRANSIENT_BACKEND", None) == "osdi":
            names.append(name)
    return tuple(names)


def ngspice_model_names(model_types):
    """Names bound to a direct-ngspice full-charge transient backend."""
    if not model_types:
        return ()
    from .device_model import get_model_class
    names = []
    for name, mt in model_types.items():
        cls = get_model_class(mt)
        if cls is not None and getattr(cls, "TRANSIENT_BACKEND", None) == "ngspice":
            names.append(name)
    return tuple(names)


def freepdk45_model_names(model_types):
    """Compatibility alias for callers that previously queried ngspice routing."""
    return tuple(name for name in ngspice_model_names(model_types)
                 if str(model_types[name]).startswith("freepdk45."))


def transient(sizes: Mapping[str, tuple[float, float]], bias: Mapping[str, float],
              tgrid: np.ndarray, vip: Any = None, vin: Any = None,
              nf: int | Mapping[str, int] | None = None, V0: Any = None,
              topo: Any = None, inputs: Mapping[str, Any] | None = None,
              node_inputs: Mapping[str, str] | None = None,
              current_inputs: Sequence[Any] | None = None,
              corner: str | Mapping[str, Any] | None = None,
              model_types: Mapping[str, str] | None = None,
              device_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
              max_step: float | None = None, flat_max_step: float | None = None,
              max_retry_subdivisions: int = 0, newton_maxit: int = 30,
              newton_step_limit: float = 5.0, newton_vtol: float = 1e-8,
              fallback_full_jacobian: bool = False,
              fallback_least_squares: bool = False, fallback_tol: float = 1e-9,
              signed_devices: Any = None, profile: bool = False,
              edge_mask: Any = None,
              rail_margin: float | None = None, integration_method: str = "be",
              gear2_be_fallback: bool = True, cap_mode: Any = None,
              cap_mode_id: Any = None,
              adaptive: bool = False, adaptive_reltol: float = 1e-4,
              adaptive_vabstol: float = 1e-6,
              adaptive_iabstol: float = 1e-12, adaptive_max_steps: int = 200000,
              adaptive_h0: float | None = None, adaptive_config: Any = None, *,
              binding: CircuitBinding | None = None,
              mismatch: Mapping[str, float] | None = None) -> dict:
    """Backward-Euler (default) or gear2/BDF2 transient.

      integration_method : "be" (backward-Euler, 1st order; the default for the
               raw transient because its numba grid keeps substep subdivision +
               retry, which hard standalone transients rely on) or "gear2"
               (variable-step BDF2, 2nd order, numba-accelerated, with maxstep
               subdivision/retry support and step-ratio limiting). The PSS/chopper periodic path
               defaults to gear2 (it closes the chopper PAC switch-edge error to
               <1% and its grid is well-conditioned); raw transient callers can
               opt in on uniform/well-conditioned grids.
      tgrid : (N,) time points [s]
      vip,vin : legacy AFE M7/M8 gate waveforms [V]
      inputs : generic mapping {input_key: waveform}; device gates are mapped by
               topo.transient_inputs, e.g. {"M1": "in"}.
      node_inputs : mapping {node_name: input_key} to drive a (rail) NODE with a
               waveform — used for a testbench where the stimulus enters at source
               nodes and propagates through a front-end network, e.g.
               {"VINP": "vip", "VINN": "vin"}.
      current_inputs : time-varying ideal current sources. Each entry can be
               {"p": nplus, "q": nminus, "input": key} or (p, q, key).
               The waveform current flows p -> q, matching topology.isources.
      max_step : optional maximum internal step. Intervals larger than this are
               split linearly between adjacent input samples.
      flat_max_step : optional maximum internal step for intervals not marked by
               edge_mask. If omitted, max_step is used everywhere.
      max_retry_subdivisions : if Newton fails on a step, recursively bisect that
               step up to this depth before recording a failure.
      fallback_least_squares : if true, a failed Newton step is retried with a
               rail-bounded least-squares solve before substepping/failing.
      fallback_full_jacobian : if true, a failed Newton step is retried with an
               expensive finite-difference Jacobian of the full residual at the
               smallest retry subdivision.
      signed_devices : retained for compatibility. All devices now use the
               signed Verilog-A drain-terminal current; this argument no longer
               changes per-device behavior.
      profile : if true, include transient_profile counters in the result.
      edge_mask : optional boolean mask over tgrid points; intervals touching a
               true point are counted as edge work in transient_profile.
      rail_margin : optional voltage margin around numeric rails for topologies
               that need physical branch selection. If omitted, topologies with
               require_dc_in_box use a 2 V margin; other topologies are unbounded.
      V0    : optional initial solved-node vector.
      model_types / device_kwargs : optional per-device model binding (silicon).
               Circuits whose transistors are OSDI (compiled Verilog-A) devices
               are routed to :func:`circuitopt.osdi_transient.transient_osdi` — the
               default (OTFT) path is untouched when these are None.
    Returns dict: t, output, vout, nfail, and per-node arrays. AFE legacy vop/von
    fields are included when those nodes exist.

    binding : optional :class:`CircuitBinding` supplying defaults for
        topo/nf/corner/model_types/device_kwargs; explicit non-None kwargs override
        it. ``dc_seed`` is not consumed here (transient seeds from ``V0``, a
        solved-node vector, not the DC-op dict). binding=None reproduces the legacy
        path exactly.

    mismatch : optional ``{device: delta_vth[V]}`` per-instance threshold-voltage
        offset map on the model-card ngspice path. Each process adapter emits its
        supported instance parameter (``delvto`` for FreePDK45, ``_delvto`` for
        TSMC28HPC+; see :mod:`circuitopt.ngspice_transient`).
        Rejected for the local-solver paths, which have no per-instance Vth knob.
        ``None`` reproduces the legacy netlist exactly.
    """
    topo, nf, corner, model_types, device_kwargs, _ = resolve_binding(
        binding, topo=topo, nf=nf, corner=corner, model_types=model_types,
        device_kwargs=device_kwargs)
    if topo is None:
        topo = AFE_TOPO
    if ngspice_model_names(model_types):
        from .ngspice_transient import transient_ngspice
        _inputs = inputs
        if _inputs is None:
            _inputs = {}
            if vip is not None:
                _inputs["vip"] = vip
            if vin is not None:
                _inputs["vin"] = vin
        return transient_ngspice(
            sizes, bias, tgrid, topo=topo, nf=nf, V0=V0, inputs=_inputs,
            node_inputs=node_inputs, current_inputs=current_inputs,
            corner=corner, model_types=model_types, device_kwargs=device_kwargs,
            integration_method=integration_method, max_step=max_step,
            mismatch=mismatch,
        )
    if mismatch:
        raise NotImplementedError(
            "per-instance threshold mismatch is only supported on the model-card "
            "ngspice transient path")
    if osdi_model_names(model_types):
        from .osdi_transient import transient_osdi
        _inputs = inputs
        if _inputs is None:
            _inputs = {}
            if vip is not None:
                _inputs["vip"] = vip
            if vin is not None:
                _inputs["vin"] = vin
        _acfg = resolve_adaptive_config(
            adaptive_config, adaptive_reltol=adaptive_reltol,
            adaptive_vabstol=adaptive_vabstol, adaptive_iabstol=adaptive_iabstol,
            adaptive_max_steps=adaptive_max_steps, adaptive_h0=adaptive_h0)
        return transient_osdi(
            sizes, bias, tgrid, topo=topo, nf=nf, V0=V0, inputs=_inputs,
            node_inputs=node_inputs, current_inputs=current_inputs,
            corner=corner, model_types=model_types, device_kwargs=device_kwargs,
            integration_method=integration_method, newton_maxit=newton_maxit,
            newton_vtol=newton_vtol, newton_step_limit=newton_step_limit,
            adaptive=bool(adaptive), adaptive_reltol=_acfg.reltol,
            adaptive_vabstol=_acfg.vabstol, adaptive_iabstol=_acfg.iabstol,
            adaptive_max_steps=_acfg.max_steps)
    marshalled = _marshal_transient(
        sizes, bias, tgrid, vip=vip, vin=vin, nf=nf, V0=V0, topo=topo,
        inputs=inputs, node_inputs=node_inputs, current_inputs=current_inputs,
        corner=corner, model_types=model_types, device_kwargs=device_kwargs,
        max_step=max_step, flat_max_step=flat_max_step,
        max_retry_subdivisions=max_retry_subdivisions,
        newton_maxit=newton_maxit, newton_step_limit=newton_step_limit,
        newton_vtol=newton_vtol,
        fallback_full_jacobian=fallback_full_jacobian,
        fallback_least_squares=fallback_least_squares, fallback_tol=fallback_tol,
        signed_devices=signed_devices, profile=profile, edge_mask=edge_mask,
        rail_margin=rail_margin, integration_method=integration_method,
        gear2_be_fallback=gear2_be_fallback, cap_mode=cap_mode,
        cap_mode_id=cap_mode_id, adaptive=adaptive,
        adaptive_reltol=adaptive_reltol, adaptive_vabstol=adaptive_vabstol,
        adaptive_iabstol=adaptive_iabstol,
        adaptive_max_steps=adaptive_max_steps, adaptive_h0=adaptive_h0,
        adaptive_config=adaptive_config)
    ctx = marshalled.ctx
    tgrid = marshalled.tgrid
    input_keys = marshalled.input_keys
    input_values = marshalled.input_values
    inputs = marshalled.inputs
    node_inputs = marshalled.node_inputs
    V0 = marshalled.V0
    Vhist = marshalled.Vhist
    edge_mask_arr = marshalled.edge_mask_arr
    profile = marshalled.profile
    N = len(tgrid)
    adaptive = ctx.adaptive
    integration_method = ctx.integration_method
    gear2_be_fallback = ctx.gear2_be_fallback
    max_retry_subdivisions = ctx.max_retry_subdivisions
    max_step = ctx.max_step
    flat_max_step = ctx.flat_max_step
    newton_maxit = ctx.newton_maxit
    newton_step_limit = ctx.newton_step_limit
    newton_vtol = ctx.newton_vtol
    fallback_full_jacobian = ctx.fallback_full_jacobian
    fallback_least_squares = ctx.fallback_least_squares
    fallback_tol = ctx.fallback_tol
    signed_devices = ctx.signed_devices
    rail_margin = ctx.rail_margin

    nfail = 0
    nretry = 0
    nsubsteps = 0
    used_grid_numba = False
    partial_grid_numba = False
    profile_wall_s = 0.0
    profile_stats = None
    numba_grid_error = None
    numba_grid_failed_index = None
    numba_grid_failed_substeps = 0
    numba_grid_failed_profile = None
    numba_grid_failed_intervals = None
    gear2_done = False
    gear2_numba_used = False
    gear2_python_retry_used = False
    gear2_retry_requested = marshalled.gear2_retry_requested

    fixed_g2 = _solve_fixed_gear2_numba(
        ctx, V0, tgrid, input_values, edge_mask_arr, profile,
        gear2_retry_requested)
    if fixed_g2.numba_grid_error is not None:
        numba_grid_error = fixed_g2.numba_grid_error
    if fixed_g2.profile_wall_s is not None:
        profile_wall_s = fixed_g2.profile_wall_s
    if fixed_g2.numba_grid_failed_index is not None:
        numba_grid_failed_index = fixed_g2.numba_grid_failed_index
        numba_grid_failed_substeps = int(fixed_g2.numba_grid_failed_substeps)
        numba_grid_failed_profile = fixed_g2.numba_grid_failed_profile
        numba_grid_failed_intervals = fixed_g2.numba_grid_failed_intervals
    if fixed_g2.handled:
        Vhist = fixed_g2.Vhist
        nsubsteps = int(fixed_g2.nsubsteps)
        nretry = int(fixed_g2.nretry)
        nfail = int(fixed_g2.nfail)
        gear2_done = True
        gear2_numba_used = True
        profile_stats = fixed_g2.profile_stats

    adaptive_used = False
    adaptive_numba_used = False
    if adaptive and integration_method == "gear2" and not gear2_done:
        adaptive_nb = _solve_adaptive_gear2_numba(ctx, V0, tgrid, input_values, profile)
        if adaptive_nb.numba_grid_error is not None:
            numba_grid_error = adaptive_nb.numba_grid_error
        if adaptive_nb.handled:
            tgrid = adaptive_nb.tgrid
            Vhist = adaptive_nb.Vhist
            input_values = adaptive_nb.input_values
            N = int(adaptive_nb.N)
            nsubsteps = int(adaptive_nb.nsubsteps)
            nretry = int(adaptive_nb.nretry)
            profile_stats = adaptive_nb.profile_stats
            profile_wall_s = adaptive_nb.profile_wall_s
            gear2_done = True
            adaptive_used = True
            adaptive_numba_used = True
            gear2_numba_used = True

    # Graceful fallback: gear2's single-step Newton stalls on stiff transients
    # (e.g. the chopper switch edges), where it can fail a large fraction of
    # steps and drift.  When too many steps fail, the gear2 result is unreliable,
    # so re-run with the robust backward-Euler path (recursive bisection + LS).
    # The PSS/periodic path opts out (gear2_be_fallback=False): shooting manages
    # its own convergence and must not mix a BE orbit into the gear2 iteration.
    if ((not adaptive) and integration_method == "gear2" and gear2_done and gear2_be_fallback and
            nfail > max(8, int(0.10 * (N - 1)))):
        be_result = transient(
            sizes, bias, tgrid, vip=vip, vin=vin, nf=nf, V0=V0, topo=topo,
            inputs=inputs, node_inputs=node_inputs, current_inputs=current_inputs,
            corner=corner, max_step=max_step, flat_max_step=flat_max_step,
            max_retry_subdivisions=max_retry_subdivisions,
            newton_maxit=newton_maxit, newton_step_limit=newton_step_limit,
            newton_vtol=newton_vtol,
            fallback_full_jacobian=fallback_full_jacobian,
            fallback_least_squares=fallback_least_squares, fallback_tol=fallback_tol,
            signed_devices=signed_devices, profile=profile, edge_mask=edge_mask,
            rail_margin=rail_margin, integration_method="be",
            gear2_be_fallback=False)
        be_result["gear2_be_fallback_used"] = True
        be_result["gear2_nfail_before_fallback"] = int(nfail)
        return be_result

    if (not adaptive) and not gear2_done:
        be_nb = _solve_be_numba(ctx, V0, tgrid, input_values, edge_mask_arr, profile)
        if be_nb.numba_grid_error is not None:
            numba_grid_error = be_nb.numba_grid_error
            used_grid_numba = False
        if be_nb.profile_wall_s is not None:
            profile_wall_s = be_nb.profile_wall_s
        if be_nb.numba_grid_failed_intervals is not None:
            numba_grid_failed_intervals = be_nb.numba_grid_failed_intervals
        if be_nb.handled:
            Vhist = be_nb.Vhist
            nsubsteps = int(be_nb.nsubsteps)
            nfail = int(be_nb.nfail)
            nretry = int(be_nb.nretry)
            used_grid_numba = True
            profile_stats = be_nb.profile_stats
        elif be_nb.numba_grid_failed_index is not None:
            # Numba BE could not converge an interval; accept its partial trajectory.
            # (The Python mid-solve resume was retired with the OO transient path; it
            # rescued ~1 in 345 solves and never a calibration/production case.)
            numba_grid_failed_index = be_nb.numba_grid_failed_index
            numba_grid_failed_substeps = int(be_nb.numba_grid_failed_substeps)
            numba_grid_failed_profile = be_nb.numba_grid_failed_profile
            if be_nb.Vhist is not None:
                Vhist = be_nb.Vhist
                nsubsteps = int(be_nb.nsubsteps)

    return _assemble_result(
        ctx, tgrid, Vhist, input_values, input_keys,
        nfail, nretry, nsubsteps,
        adaptive_used, adaptive_numba_used,
        used_grid_numba, gear2_numba_used,
        gear2_python_retry_used,
        profile, profile_stats, profile_wall_s,
        partial_grid_numba, numba_grid_error,
        numba_grid_failed_index, numba_grid_failed_substeps,
        numba_grid_failed_profile, numba_grid_failed_intervals)



# ── self-consistency check vs the validated DC / AC solvers ──
if __name__ == "__main__":
    sizes = {"M6": (2264, 78), "M7": (61365, 61), "M8": (61365, 61),
             "M9": (3175, 468), "M10": (3175, 468), "M11": (465, 66),
             "M12": (894, 85), "M13": (894, 85), "M14": (5224, 46), "M15": (5224, 46)}
    bias = {"VDD": 40.0, "VCM": 30.65, "VB": 9.84, "VC": 16.0}

    # AC reference gain/BW (the transient must reproduce both)
    ac = ac_solve(sizes, bias, np.logspace(0, 4, 80))
    gain = ac["gains"].max(); bw = ac["bw_Hz"]; tau = 1 / (2 * np.pi * bw)
    print(f"AC ref: gain={gain:.4f} ({20*np.log10(gain):.2f} dB), BW={bw:.0f} Hz, tau={tau*1e3:.3f} ms")

    T = 0.004; N = 200; t = np.linspace(0, T, N)        # h≈20 µs (physically fine)
    vcm = np.full(N, bias["VCM"])

    # (1) steady state: hold vip=vin=VCM -> must sit at the DC op (no CM run-away)
    r0 = transient(sizes, bias, t, vcm, vcm)
    print(f"(1) quiescent drift = {r0['vout'][-1]*1e6:+.4f} µV  nfail={r0['nfail']}/{N-1}  (期望 0)")

    # (2) differential step vip-vin=1 mV at 0.5 ms -> settles to the small-signal gain
    dstep = 0.5e-3; ts = 0.5e-3
    vp = vcm + np.where(t >= ts, +dstep, 0.0)
    vn = vcm - np.where(t >= ts, +dstep, 0.0)
    r = transient(sizes, bias, t, vp, vn); vo = r["vout"]; settled = vo[-1]
    post = np.where(t >= ts)[0]; hit = post[np.abs(vo[post]) >= abs(settled) * (1 - np.exp(-1))]
    tau_tr = (t[hit[0]] - ts) if len(hit) else float("nan")
    print(f"(2) step 1mV: settled={settled*1e3:.4f} mV  gain={settled/(2*dstep):+.4f} (AC {gain:.4f})  "
          f"nfail={r['nfail']}/{N-1}")
    print(f"    tau(63%)={tau_tr*1e3:.3f} ms vs AC tau={tau*1e3:.3f} ms (multi-pole, ~order match)")
