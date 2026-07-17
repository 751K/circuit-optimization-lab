"""Marshalling for the native BSIM4 fixed-grid Rust transient kernel."""
from __future__ import annotations

import numpy as np

from ..._rust_transient import (
    _optional_index,
    _term_record,
    passive_problem_spec,
)
from ...compiled_topology import TERM_RAIL
from .native import Bsim4NativeError, _NativeDevice


def build_bsim4_problem(plan, devices, handles, dynamic_sources=()):
    """Build the shared Rust four-terminal circuit problem around owned handles."""
    import circuitopt_core

    wrappers = [devices[item.name] for item in plan.devices]
    circuit = circuitopt_core.OtftTransientProblem(
        passive_problem_spec(plan, dynamic_sources))
    device_records = []
    for item, wrapper in zip(plan.devices, wrappers, strict=True):
        terms = [
            _term_record(item.d),
            _term_record(item.g),
            _term_record(item.s),
            _term_record((TERM_RAIL, wrapper.vb)),
        ]
        rows = [
            _optional_index(item.di),
            _optional_index(item.gi),
            _optional_index(item.si),
            -1,
        ]
        device_records.append((terms, rows))
    return circuitopt_core.Bsim4TransientProblem(
        circuit, device_records, [handle.pointer for handle in handles])


def solve_bsim4_rust(
    plan,
    devices,
    x0,
    tgrid,
    input_values,
    dynamic_sources,
    *,
    method,
    newton_maxit,
    newton_vtol,
    newton_step_limit,
    gmin,
):
    """Run BSIM4 model evaluation, MNA stamp, Newton, and grid in Rust."""
    wrappers = [devices[item.name] for item in plan.devices]
    if not wrappers:
        raise ValueError("native BSIM4 transient requires at least one device")
    handles = [
        _NativeDevice(
            wrapper.model_card,
            wrapper.instance_card,
            wrapper.temperature,
            backend="rust",
        )
        for wrapper in wrappers
    ]
    try:
        problem = build_bsim4_problem(
            plan, devices, handles, dynamic_sources)
        completed, states, failures, first_failure = problem.solve_fixed_grid(
            np.asarray(x0, dtype=float),
            np.asarray(tgrid, dtype=float),
            np.asarray(input_values, dtype=float),
            integration_method=method,
            max_iterations=int(newton_maxit),
            voltage_tolerance=float(newton_vtol),
            step_limit=float(newton_step_limit),
            gmin=float(gmin),
        )
        if not completed:
            raise Bsim4NativeError(
                f"Rust BSIM4 transient failed at step {int(first_failure)}")
        return np.asarray(states, dtype=float), int(failures), int(first_failure)
    finally:
        for handle in handles:
            handle.close()
