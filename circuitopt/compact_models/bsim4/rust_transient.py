"""Marshalling for the native BSIM4 Rust transient kernels."""
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
    adaptive=False,
    adaptive_config=None,
    max_step=None,
    profile=False,
):
    """Run BSIM4 model evaluation, MNA stamp, Newton, and time integration."""
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
        if adaptive:
            if adaptive_config is None:
                raise ValueError("adaptive BSIM4 transient requires adaptive_config")
            (
                completed,
                accepted_times,
                states,
                accepted_inputs,
                device_currents,
                device_charges,
                integration_coefficients,
                adaptive_stats,
            ) = problem.solve_adaptive_gear2(
                np.asarray(x0, dtype=float),
                np.asarray(tgrid, dtype=float),
                np.asarray(input_values, dtype=float),
                max_step=-1.0 if max_step is None else float(max_step),
                reltol=float(adaptive_config.reltol),
                voltage_abstol=float(adaptive_config.vabstol),
                current_abstol=float(adaptive_config.iabstol),
                max_steps=int(adaptive_config.max_steps),
                initial_step=(
                    -1.0
                    if adaptive_config.h0 is None
                    else float(adaptive_config.h0)
                ),
                max_iterations=int(newton_maxit),
                voltage_tolerance=float(newton_vtol),
                step_limit=float(newton_step_limit),
                gmin=float(gmin),
                profile=bool(profile),
            )
            (
                accepted_steps,
                rejected_steps,
                trial_solves,
                newton_iterations,
                bsim_evaluations,
                bsim_batches,
                gear2_predictor_steps,
                lte_estimates,
                lte_linear_solves,
                lte_rejections,
                newton_rejections,
            ) = adaptive_stats
            failures = 0
            first_failure = -1
            failed_steps = []
            accepted_times = np.asarray(accepted_times, dtype=float)
            accepted_inputs = np.asarray(accepted_inputs, dtype=float).T
            integration_coefficients = np.asarray(
                integration_coefficients, dtype=float)
        else:
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
                gear2_predictor_steps,
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
            accepted_times = np.asarray(tgrid, dtype=float)
            accepted_inputs = np.asarray(input_values, dtype=float)
            integration_coefficients = None
            accepted_steps = len(accepted_times) - 1
            rejected_steps = 0
            trial_solves = accepted_steps
            lte_estimates = 0
            lte_linear_solves = 0
            lte_rejections = 0
            newton_rejections = 0
        wall_time_s = time.perf_counter() - started if profile else 0.0
        if not completed:
            location = (
                f"step {int(first_failure)}"
                if int(first_failure) >= 0
                else f"t={accepted_times[-1]:.6g}"
                if len(accepted_times)
                else "initialization"
            )
            raise Bsim4NativeError(f"Rust BSIM4 transient failed at {location}")
        states = np.asarray(states, dtype=float)
        device_currents = np.asarray(device_currents, dtype=float)
        device_charges = np.asarray(device_charges, dtype=float)
        expected = (len(accepted_times), len(wrappers), 4)
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
            accepted_times,
            accepted_inputs,
            integration_coefficients,
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
                "gear2_predictor_steps": int(gear2_predictor_steps),
                "failed_step_indices": [int(index) for index in failed_steps],
                "adaptive": bool(adaptive),
                "accepted_steps": int(accepted_steps),
                "rejected_steps": int(rejected_steps),
                "trial_solves": int(trial_solves),
                "lte_estimates": int(lte_estimates),
                "lte_linear_solves": int(lte_linear_solves),
                "lte_rejections": int(lte_rejections),
                "newton_rejections": int(newton_rejections),
            },
        )
    finally:
        for handle in handles:
            handle.close()
