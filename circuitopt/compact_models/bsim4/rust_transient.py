"""Marshalling for the native BSIM4 fixed-grid Rust transient kernel."""
from __future__ import annotations

import time

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
    profile=False,
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
        started = time.perf_counter() if profile else 0.0
        (
            completed,
            states,
            device_currents,
            device_charges,
            failures,
            first_failure,
            newton_iterations,
            bsim_evaluations,
            bsim_batches,
            failed_steps,
        ) = problem.solve_fixed_grid(
            np.asarray(x0, dtype=float),
            np.asarray(tgrid, dtype=float),
            np.asarray(input_values, dtype=float),
            integration_method=method,
            max_iterations=int(newton_maxit),
            voltage_tolerance=float(newton_vtol),
            step_limit=float(newton_step_limit),
            gmin=float(gmin),
            profile=bool(profile),
        )
        wall_time_s = time.perf_counter() - started if profile else 0.0
        if not completed:
            raise Bsim4NativeError(
                f"Rust BSIM4 transient failed at step {int(first_failure)}")
        states = np.asarray(states, dtype=float)
        device_currents = np.asarray(device_currents, dtype=float)
        device_charges = np.asarray(device_charges, dtype=float)
        expected = (len(tgrid), len(wrappers), 4)
        if device_currents.shape != expected or device_charges.shape != expected:
            raise Bsim4NativeError(
                "Rust BSIM4 transient returned invalid device-history shapes: "
                f"currents={device_currents.shape}, charges={device_charges.shape}, "
                f"expected={expected}")
        if (
            not np.all(np.isfinite(device_currents))
            or not np.all(np.isfinite(device_charges))
        ):
            raise Bsim4NativeError(
                "Rust BSIM4 transient returned non-finite device history")
        return (
            states,
            device_currents,
            device_charges,
            int(failures),
            int(first_failure),
            {
                "wall_time_s": float(wall_time_s),
                "newton_iters_total": int(newton_iterations),
                "bsim_evaluations": int(bsim_evaluations),
                "bsim_batch_calls": int(bsim_batches),
                "failed_step_indices": [int(index) for index in failed_steps],
            } if profile else None,
        )
    finally:
        for handle in handles:
            handle.close()
