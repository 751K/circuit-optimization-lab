"""Charge-conserving circuit transient for native four-terminal BSIM4 devices."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ...adaptive_config import resolve_adaptive_config
from ...compiled_topology import CompiledTopology, TERM_INPUT, TERM_SOLVED
from ...device_factory import build_devices


_STEP_RATIO_RTOL = 1e-12


def _subdivision_count(interval, max_step):
    """Ceil an interval/limit ratio without splitting roundoff-only excess."""
    ratio = float(interval) / float(max_step)
    nearest = round(ratio)
    tolerance = _STEP_RATIO_RTOL * max(1.0, abs(ratio))
    if nearest >= 1 and abs(ratio - nearest) <= tolerance:
        return nearest
    return max(1, int(np.ceil(ratio)))


def _subdivision_counts(intervals, max_step):
    """:func:`_subdivision_count` over a whole interval array."""
    ratio = np.asarray(intervals, dtype=float) / float(max_step)
    nearest = np.round(ratio)
    tolerance = _STEP_RATIO_RTOL * np.maximum(1.0, np.abs(ratio))
    snapped = (nearest >= 1.0) & (np.abs(ratio - nearest) <= tolerance)
    return np.where(
        snapped, nearest, np.maximum(1.0, np.ceil(ratio))
    ).astype(int)


def _integration_coefficient_columns(tgrid, _method, provided):
    """Validate and split integration coefficients returned by the native solver."""
    if provided is None:
        raise RuntimeError(
            "native BSIM4 transient did not return its integration coefficients")
    coefficients = np.asarray(provided, dtype=float)
    if coefficients.shape != (len(tgrid), 3):
        raise RuntimeError(
            "native BSIM4 transient returned invalid integration coefficients: "
            f"{coefficients.shape}, expected {(len(tgrid), 3)}")
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError(
            "native BSIM4 transient returned non-finite integration coefficients")
    return coefficients[1:, 0], coefficients[1:, 1], coefficients[1:, 2]


def _shift_two(values):
    """``values[sample - 2]``, with sample 1 reusing ``values[0]``.

    Mirrors the scalar ``charges[sample - 2] if sample > 1 else
    charges[sample - 1]`` selection for the whole ``1 ..`` slice at once.
    """
    shifted = np.empty_like(values[1:])
    shifted[0] = values[0]
    shifted[1:] = values[:-2]
    return shifted


def _expanded_grid(tgrid, inputs, max_step):
    if max_step is None:
        return tgrid, inputs, np.arange(len(tgrid))
    max_step = float(max_step)
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    counts = _subdivision_counts(np.diff(np.asarray(tgrid, dtype=float)), max_step)
    if not np.any(counts > 1):
        # Every interval already satisfies max_step, so the expansion is the
        # identity and the requested grid carries the waveforms unchanged.
        return tgrid, inputs, np.arange(len(tgrid))
    times = [float(tgrid[0])]
    requested = [0]
    for k in range(1, len(tgrid)):
        count = counts[k - 1]
        times.extend(np.linspace(tgrid[k - 1], tgrid[k], count + 1)[1:])
        requested.append(len(times) - 1)
    expanded = np.asarray(times, dtype=float)
    waveforms = {
        key: np.interp(expanded, tgrid, value)
        for key, value in inputs.items()
    }
    return expanded, waveforms, np.asarray(requested, dtype=int)


def assemble_native_bsim4_result(
    *,
    plan,
    devices,
    topo,
    bias,
    tgrid,
    input_matrix,
    requested_t,
    requested_index,
    integration_coefficients,
    xhist,
    device_currents,
    device_charges,
    nfail,
    first_fail,
    method,
    node_inputs=None,
    adaptive=False,
    adaptive_config=None,
    native_profile=None,
    profile=False,
    nnear=0,
    failed_residuals=(),
    near_residuals=(),
):
    """Assemble the public transient payload from one accepted native trajectory.

    Both ordinary BSIM transients and the compiled SAR continuation use this
    single reconstruction path, so terminal-charge derivatives, source-current
    signs, legacy aliases, and power-facing branch currents stay identical.
    """
    tgrid = np.asarray(tgrid, dtype=float)
    input_matrix = np.asarray(input_matrix, dtype=float)
    requested_t = np.asarray(requested_t, dtype=float)
    requested_index = np.asarray(requested_index, dtype=int)
    xhist = np.asarray(xhist, dtype=float)
    device_currents = np.asarray(device_currents, dtype=float)
    device_charges = np.asarray(device_charges, dtype=float)
    node_inputs = dict(node_inputs or {})
    native_profile = dict(native_profile or {})
    a0_col, a1_col, a2_col = _integration_coefficient_columns(
        tgrid, method, integration_coefficients)

    def term_series(term):
        """The whole-transient sample series for one compiled terminal."""
        kind, ref = term
        if kind == TERM_SOLVED:
            return xhist[:, ref]
        if kind == TERM_INPUT:
            return input_matrix[ref]
        return np.full(len(tgrid), float(ref))

    rail_values = topo.rail_values(bias)
    rail_currents = {
        name: np.zeros(len(tgrid), dtype=float)
        for name, value in rail_values.items()
        if value != 0.0 and name not in node_inputs
    }
    waveform_currents = {
        f"node:{node}": np.zeros(len(tgrid), dtype=float)
        for node in node_inputs
    }

    def rail_for_node(node):
        return node if node in rail_currents else None

    def bulk_rail(dev):
        matches = [
            name for name in rail_currents
            if np.isclose(rail_values[name], dev.vb, rtol=0.0, atol=1e-15)
        ]
        return matches[0] if matches else None

    device_totals = device_currents.copy()
    if len(tgrid) > 1:
        device_totals[1:] += (
            a0_col[:, None, None] * device_charges[1:]
            + a1_col[:, None, None] * device_charges[:-1]
            + a2_col[:, None, None] * _shift_two(device_charges)
        )

    for position, item in enumerate(plan.devices):
        total = device_totals[:, position, :]
        terminals = (
            rail_for_node(item.d_node),
            rail_for_node(item.g_node),
            rail_for_node(item.s_node),
            bulk_rail(devices[item.name]),
        )
        for terminal_index, rail in enumerate(terminals):
            if rail is not None:
                rail_currents[rail] -= total[:, terminal_index]
        if item.name in topo.transient_inputs:
            waveform_currents[f"gate:{item.name}"] = -total[:, 1].copy()
        else:
            for terminal_index, node in enumerate(
                (item.d_node, item.g_node, item.s_node)
            ):
                key = f"node:{node}"
                if key in waveform_currents:
                    waveform_currents[key] -= total[:, terminal_index]

    for item in plan.resistors:
        current = (term_series(item.a) - term_series(item.b)) * item.g
        rail_a = rail_for_node(item.a_node)
        rail_b = rail_for_node(item.b_node)
        if rail_a is not None:
            rail_currents[rail_a] -= current
        if rail_b is not None:
            rail_currents[rail_b] += current
        if f"node:{item.a_node}" in waveform_currents:
            waveform_currents[f"node:{item.a_node}"] -= current
        if f"node:{item.b_node}" in waveform_currents:
            waveform_currents[f"node:{item.b_node}"] += current

    for item in plan.capacitors:
        voltage = term_series(item.a) - term_series(item.b)
        current = np.zeros(len(tgrid), dtype=float)
        if len(tgrid) > 1:
            current[1:] = item.value * (
                a0_col * voltage[1:]
                + a1_col * voltage[:-1]
                + a2_col * _shift_two(voltage)
            )
        rail_a = rail_for_node(item.a_node)
        rail_b = rail_for_node(item.b_node)
        if rail_a is not None:
            rail_currents[rail_a] -= current
        if rail_b is not None:
            rail_currents[rail_b] += current
        if f"node:{item.a_node}" in waveform_currents:
            waveform_currents[f"node:{item.a_node}"] -= current
        if f"node:{item.b_node}" in waveform_currents:
            waveform_currents[f"node:{item.b_node}"] += current

    for item in plan.isources:
        rail_p = rail_for_node(item.p_node)
        rail_q = rail_for_node(item.q_node)
        if rail_p is not None:
            rail_currents[rail_p] -= item.value
        if rail_q is not None:
            rail_currents[rail_q] += item.value
    for item in plan.vsources:
        branch = xhist[:, item.bi]
        rail_p = rail_for_node(item.p_node)
        rail_q = rail_for_node(item.q_node)
        if rail_p is not None:
            rail_currents[rail_p] -= branch
        if rail_q is not None:
            rail_currents[rail_q] += branch
    for item in plan.vcvs:
        branch = xhist[:, item.bi]
        rail_p = rail_for_node(item.p_node)
        rail_q = rail_for_node(item.q_node)
        if rail_p is not None:
            rail_currents[rail_p] -= branch
        if rail_q is not None:
            rail_currents[rail_q] += branch
    for item in plan.ccvs:
        branch = xhist[:, item.bi]
        rail_p = rail_for_node(item.p_node)
        rail_q = rail_for_node(item.q_node)
        if rail_p is not None:
            rail_currents[rail_p] -= branch
        if rail_q is not None:
            rail_currents[rail_q] += branch

    sampled = xhist[requested_index]
    nodes = {name: sampled[:, plan.idx[name]] for name in plan.solved}
    output = np.zeros(len(requested_t), dtype=float)
    for node, weight in plan.output_weights.items():
        output += weight * nodes[node]
    result = {
        "t": requested_t,
        "output": output,
        "vout": output,
        "nodes": nodes,
        "nfail": int(nfail),
        "nnear": int(nnear),
        "failed_residual_max": (
            float(max(failed_residuals)) if failed_residuals else 0.0
        ),
        "near_residual_max": (
            float(max(near_residuals)) if near_residuals else 0.0
        ),
        "nretry": 0,
        "nsubsteps": 0 if adaptive else int(len(tgrid) - len(requested_t)),
        "adaptive": bool(adaptive),
        "bsim4_native_transient": True,
        "bsim4_rust_transient": True,
        "backend": "bsim4_native",
        "integration_method": "gear2" if method in {"gear2", "bdf2"} else "be",
        "bsim_final_load_tolerance": float(
            native_profile.get("final_load_tolerance_v", 0.0)),
        "bsim_model_bypass_tolerance": float(
            native_profile.get("model_bypass_tolerance_v", 0.0)),
        "X_final": sampled[-1].copy(),
        "branch_currents": {
            name: sampled[:, index]
            for name, index in topo.vsource_index.items()
        } | {
            f"rail:{name}": values[requested_index]
            for name, values in rail_currents.items()
        } | {
            name: values[requested_index]
            for name, values in waveform_currents.items()
        },
    }
    if adaptive:
        result.update({
            "adaptive_reltol": float(adaptive_config.reltol),
            "adaptive_vabstol": float(adaptive_config.vabstol),
            "adaptive_iabstol": float(adaptive_config.iabstol),
            "adaptive_accepted_steps": int(native_profile["accepted_steps"]),
            "adaptive_rejected_steps": int(native_profile["rejected_steps"]),
        })
    if profile:
        solver_steps = (
            native_profile["trial_solves"]
            if adaptive
            else native_profile["accepted_steps"]
        )
        failed_steps = native_profile["failed_step_indices"]
        result["transient_profile"] = {
            "enabled": True,
            "backend": "bsim4_native",
            "rust_grid_solver": True,
            "wall_time_s": native_profile["wall_time_s"],
            "intervals": len(requested_t) - 1,
            "solver_steps": solver_steps,
            "nsubsteps": 0 if adaptive else int(len(tgrid) - len(requested_t)),
            "adaptive": bool(adaptive),
            "accepted_steps": native_profile["accepted_steps"],
            "rejected_steps": native_profile["rejected_steps"],
            "trial_solves": native_profile["trial_solves"],
            "lte_estimates": native_profile["lte_estimates"],
            "lte_linear_solves": native_profile["lte_linear_solves"],
            "lte_rejections": native_profile["lte_rejections"],
            "newton_rejections": native_profile["newton_rejections"],
            "newton_iters_total": native_profile["newton_iters_total"],
            "newton_iters_avg": (
                native_profile["newton_iters_total"] / solver_steps
                if solver_steps
                else 0.0
            ),
            "bsim_evaluations": native_profile["bsim_evaluations"],
            "bsim_batch_calls": native_profile["bsim_batch_calls"],
            "gear2_predictor_steps": native_profile["gear2_predictor_steps"],
            "final_load_tolerance_v": native_profile[
                "final_load_tolerance_v"],
            "model_bypass_tolerance_v": native_profile[
                "model_bypass_tolerance_v"],
            "bsim_evaluations_avg_per_solver_step": (
                native_profile["bsim_evaluations"] / solver_steps
                if solver_steps
                else 0.0
            ),
            "failed_steps": len(failed_steps),
            "failed_step_indices": failed_steps,
            "first_failed_step": int(first_fail) if first_fail >= 0 else None,
        }
    for legacy in ("VOP", "VON"):
        if legacy in nodes:
            result[legacy.lower()] = nodes[legacy]
    return result


def transient_native_bsim4(
    sizes,
    bias,
    tgrid,
    *,
    topo,
    nf=None,
    V0=None,
    inputs=None,
    node_inputs=None,
    current_inputs: Sequence | None = None,
    corner=None,
    model_types=None,
    device_kwargs=None,
    integration_method="be",
    newton_maxit=30,
    newton_vtol=1e-8,
    newton_step_limit=0.25,
    max_step=None,
    gmin=1e-12,
    bsim_final_load_tolerance=0.0,
    bsim_model_bypass_tolerance=0.0,
    adaptive=False,
    adaptive_reltol=1e-4,
    adaptive_vabstol=1e-6,
    adaptive_iabstol=1e-12,
    adaptive_max_steps=200000,
    adaptive_h0=None,
    adaptive_config=None,
    profile=False,
):
    """Integrate native BSIM4 terminal currents and conserved terminal charges.

    The nonlinear residual uses the compact model's full ``(d, g, s, b)``
    current and charge vectors. Backward Euler and variable-step BDF2 are
    supported. BSIM internal drain/source resistance nodes are reduced by the
    native kernel at each Newton point; their poles are therefore treated
    quasi-statically while all external terminal charge is integrated.
    """
    method = str(integration_method).lower()
    if method not in {"be", "gear2", "bdf2"}:
        raise ValueError(
            f"integration_method must be 'be' or 'gear2', got {integration_method!r}")
    if adaptive and method not in {"gear2", "bdf2"}:
        raise ValueError("adaptive transient requires integration_method='gear2'")
    adaptive_config = resolve_adaptive_config(
        adaptive_config,
        adaptive_reltol=adaptive_reltol,
        adaptive_vabstol=adaptive_vabstol,
        adaptive_iabstol=adaptive_iabstol,
        adaptive_max_steps=adaptive_max_steps,
        adaptive_h0=adaptive_h0,
    )
    requested_t = np.asarray(tgrid, dtype=float)
    if requested_t.ndim != 1 or len(requested_t) < 2:
        raise ValueError("tgrid must contain at least two time points")
    if not np.all(np.diff(requested_t) > 0.0):
        raise ValueError("tgrid must be strictly increasing")
    if max_step is not None and (
        not np.isfinite(float(max_step)) or float(max_step) <= 0.0
    ):
        raise ValueError("max_step must be positive and finite")
    bsim_final_load_tolerance = float(bsim_final_load_tolerance)
    if (
        not np.isfinite(bsim_final_load_tolerance)
        or bsim_final_load_tolerance < 0.0
        or bsim_final_load_tolerance > 1e-12
    ):
        raise ValueError(
            "bsim_final_load_tolerance must be finite and within [0, 1e-12] V")
    bsim_model_bypass_tolerance = float(bsim_model_bypass_tolerance)
    if (
        not np.isfinite(bsim_model_bypass_tolerance)
        or bsim_model_bypass_tolerance < 0.0
        or bsim_model_bypass_tolerance > float(newton_vtol)
    ):
        raise ValueError(
            "bsim_model_bypass_tolerance must be finite and within "
            "[0, newton_vtol] V")

    raw_inputs = {
        key: np.asarray(value, dtype=float)
        for key, value in (inputs or {}).items()
    }
    for key, value in raw_inputs.items():
        if value.shape != requested_t.shape:
            raise ValueError(
                f"Input waveform {key!r} shape {value.shape} != tgrid shape "
                f"{requested_t.shape}")
    if adaptive:
        tgrid = requested_t
        inputs = raw_inputs
        requested_index = np.arange(len(requested_t))
    else:
        tgrid, inputs, requested_index = _expanded_grid(
            requested_t, raw_inputs, max_step)
    input_keys = tuple(inputs)
    input_matrix = (
        np.vstack([inputs[key] for key in input_keys])
        if input_keys
        else np.empty((0, len(tgrid)), dtype=float)
    )
    node_inputs = dict(node_inputs or {})
    plan = CompiledTopology(
        topo,
        bias,
        input_keys=input_keys,
        node_inputs=node_inputs,
        transient_inputs=True,
    )
    devices = build_devices(
        sizes,
        nf=nf,
        corner=corner,
        topo=topo,
        model_types=model_types,
        device_kwargs=device_kwargs,
    )
    unsupported = [
        name for name, dev in devices.items()
        if getattr(dev, "TRANSIENT_BACKEND", None) != "bsim4_native"
    ]
    if unsupported:
        raise NotImplementedError(
            "native BSIM4 transient requires every transistor to use the native "
            f"backend; unsupported devices: {', '.join(sorted(unsupported))}")

    n_aug = plan.n_aug
    if V0 is None:
        from ...dc_solver import DC_FALLBACK_TOL
        from .rust_transient import solve_bsim4_dc_rust

        dc_tolerance = getattr(topo, "dc_tol", None) or DC_FALLBACK_TOL
        rail_span = max(
            (abs(float(value)) for value in bias.values()),
            default=1.0,
        )
        V0 = solve_bsim4_dc_rust(
            CompiledTopology(topo, bias),
            devices,
            topo.dc_guess_vectors(bias),
            dc_tolerance=dc_tolerance,
            step_limit=max(0.25, rail_span / 4.0),
        )
        if V0 is not None and getattr(topo, "require_dc_in_box", False):
            if not topo.in_voltage_box(topo.node_vals(V0), bias):
                V0 = None
        if V0 is None:
            from ...ac_solver import ac_solve

            ac = ac_solve(
                sizes,
                bias,
                np.asarray([1.0]),
                topo=topo,
                nf=nf,
                corner=corner,
                model_types=model_types,
                device_kwargs=device_kwargs,
            )
            if ac is None:
                raise RuntimeError(
                    "native BSIM4 transient could not find a DC initial point")
            V0 = np.asarray(
                [ac["dc_op"][name] for name in topo.solved], dtype=float)
    else:
        V0 = np.asarray(V0, dtype=float)
    if len(V0) < n_aug:
        V0 = np.concatenate((V0, np.zeros(n_aug - len(V0))))
    elif len(V0) > n_aug:
        V0 = V0[:n_aug]

    dynamic_sources = []
    for pos, entry in enumerate(current_inputs or ()):
        if isinstance(entry, Mapping):
            p_node, q_node, key = entry["p"], entry["q"], entry["input"]
        else:
            p_node, q_node, key = entry
        if key not in plan.input_index:
            raise ValueError(
                f"current_inputs[{pos}] references missing waveform {key!r}")
        dynamic_sources.append((
            plan.solved_index(plan.compile_term(p_node)),
            plan.solved_index(plan.compile_term(q_node)),
            plan.input_index[key],
        ))

    def add_derivative(matrix, row, term, value):
        if row is not None and term[0] == TERM_SOLVED:
            matrix[row, term[1]] += value

    nnear = 0
    failed_residuals = []
    near_residuals = []
    from .rust_transient import solve_bsim4_rust

    (
        solved_times,
        solved_inputs,
        integration_coefficients,
        xhist,
        device_currents,
        device_charges,
        nfail,
        first_fail,
        native_profile,
    ) = solve_bsim4_rust(
        plan,
        devices,
        V0,
        tgrid,
        input_matrix,
        dynamic_sources,
        method=method,
        newton_maxit=newton_maxit,
        newton_vtol=newton_vtol,
        newton_step_limit=newton_step_limit,
        gmin=gmin,
        final_load_tolerance=bsim_final_load_tolerance,
        model_bypass_tolerance=bsim_model_bypass_tolerance,
        adaptive=adaptive,
        adaptive_config=adaptive_config,
        max_step=max_step,
        profile=profile,
    )
    if adaptive:
        tgrid = solved_times
        input_matrix = solved_inputs
        requested_t = solved_times
        requested_index = np.arange(len(solved_times))
    return assemble_native_bsim4_result(
        plan=plan,
        devices=devices,
        topo=topo,
        bias=bias,
        tgrid=tgrid,
        input_matrix=input_matrix,
        requested_t=requested_t,
        requested_index=requested_index,
        integration_coefficients=integration_coefficients,
        xhist=xhist,
        device_currents=device_currents,
        device_charges=device_charges,
        nfail=nfail,
        first_fail=first_fail,
        method=method,
        node_inputs=node_inputs,
        adaptive=adaptive,
        adaptive_config=adaptive_config,
        native_profile=native_profile,
        profile=profile,
        nnear=nnear,
        failed_residuals=failed_residuals,
        near_residuals=near_residuals,
    )
