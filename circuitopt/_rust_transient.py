"""Private Python-to-Rust transient topology bridge.

The public transient marshal remains the single place that resolves device
models, rails, waveform inputs, and MNA branch indices.  This module only
serializes that compiled context into the immutable representation consumed by
``circuitopt_core``.
"""
from __future__ import annotations


_PARAMETER_FIELDS = (
    "p_Vfb",
    "p_Vss",
    "p_Lc",
    "p_lambda",
    "p_contact_scale",
    "p_exponent",
    "p_current_scale",
    "p_inv_Rleak",
    "p_two_over_pi",
    "p_cap_cgs1",
    "p_cap_cgd1",
    "p_cap_half_wl_ci",
    "p_cap_cgs3_base",
    "p_cap_cgd3_base",
    "p_k1",
    "p_gate_leak_g",
)


def _optional_index(value):
    return -1 if value is None else int(value)


def _term_record(value):
    kind, reference = value
    if int(kind) in (0, 1):
        return int(kind), int(reference), 0.0
    return int(kind), 0, float(reference)


def passive_problem_spec(plan, dynamic_sources=()):
    """Serialize linear/source MNA data shared by OTFT and BSIM4 grids."""
    return {
        "node_count": int(plan.n),
        "size": int(plan.n_aug),
        "devices": [],
        "resistors": [
            (_term_record(item.a), _term_record(item.b),
             _optional_index(item.ai), _optional_index(item.bi), float(item.g))
            for item in plan.resistors
        ],
        "capacitors": [
            (_term_record(item.a), _term_record(item.b),
             _optional_index(item.ai), _optional_index(item.bi), float(item.value))
            for item in plan.capacitors
        ],
        "current_sources": [
            (_optional_index(item.pi), _optional_index(item.qi), float(item.value))
            for item in plan.isources
        ],
        "dynamic_sources": [
            (_optional_index(pi), _optional_index(qi), int(input_index))
            for pi, qi, input_index in dynamic_sources
        ],
        "vccs": [
            (_optional_index(item.pi), _optional_index(item.qi),
             _term_record(item.cp), _term_record(item.cn),
             _optional_index(item.cpi), _optional_index(item.cni), float(item.gm))
            for item in plan.vccs
        ],
        "voltage_sources": [
            (_term_record(item.p), _term_record(item.q),
             _optional_index(item.pi), _optional_index(item.qi), int(item.bi),
             float(item.e_const), int(item.e_input_idx))
            for item in plan.vsources
        ],
        "vcvs": [
            (_term_record(item.p), _term_record(item.q),
             _term_record(item.cp), _term_record(item.cn),
             _optional_index(item.pi), _optional_index(item.qi),
             _optional_index(item.cpi), _optional_index(item.cni),
             int(item.bi), float(item.mu))
            for item in plan.vcvs
        ],
        "cccs": [
            (_optional_index(item.pi), _optional_index(item.qi),
             int(item.ctrl_bi), float(item.beta))
            for item in plan.cccs
        ],
        "ccvs": [
            (_term_record(item.p), _term_record(item.q),
             _optional_index(item.pi), _optional_index(item.qi), int(item.bi),
             int(item.ctrl_bi), float(item.gamma))
            for item in plan.ccvs
        ],
    }


def transient_problem_spec(ctx):
    """Return the canonical Rust topology record for a ``_TransientCtx``."""
    plan = ctx.plan
    params = tuple(getattr(ctx, name) for name in _PARAMETER_FIELDS)

    devices = []
    for pos, item in enumerate(plan.devices):
        devices.append((
            _term_record(item.d),
            _term_record(item.g),
            _term_record(item.s),
            _optional_index(item.di),
            _optional_index(item.gi),
            _optional_index(item.si),
            bool(ctx.dev_use_abs[pos]),
            [float(values[pos]) for values in params],
        ))

    dynamic_sources = zip(ctx.dyn_pi, ctx.dyn_qi, ctx.dyn_input_idx, strict=True)
    spec = passive_problem_spec(plan, dynamic_sources)
    spec["devices"] = devices
    return spec


def build_otft_transient_problem(ctx):
    """Construct the compiled Rust transient problem for ``ctx``."""
    import circuitopt_core

    return circuitopt_core.OtftTransientProblem(transient_problem_spec(ctx))
