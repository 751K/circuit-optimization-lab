"""Charge-conserving circuit transient for native four-terminal BSIM4 devices."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ...compiled_topology import CompiledTopology, TERM_SOLVED
from ...device_factory import build_devices


def _expanded_grid(tgrid, inputs, max_step):
    if max_step is None:
        return tgrid, inputs, np.arange(len(tgrid))
    max_step = float(max_step)
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    times = [float(tgrid[0])]
    requested = [0]
    for k in range(1, len(tgrid)):
        count = max(1, int(np.ceil((tgrid[k] - tgrid[k - 1]) / max_step)))
        times.extend(np.linspace(tgrid[k - 1], tgrid[k], count + 1)[1:])
        requested.append(len(times) - 1)
    expanded = np.asarray(times, dtype=float)
    waveforms = {
        key: np.interp(expanded, tgrid, value)
        for key, value in inputs.items()
    }
    return expanded, waveforms, np.asarray(requested, dtype=int)


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
    requested_t = np.asarray(tgrid, dtype=float)
    if requested_t.ndim != 1 or len(requested_t) < 2:
        raise ValueError("tgrid must contain at least two time points")
    if not np.all(np.diff(requested_t) > 0.0):
        raise ValueError("tgrid must be strictly increasing")

    raw_inputs = {
        key: np.asarray(value, dtype=float)
        for key, value in (inputs or {}).items()
    }
    for key, value in raw_inputs.items():
        if value.shape != requested_t.shape:
            raise ValueError(
                f"Input waveform {key!r} shape {value.shape} != tgrid shape "
                f"{requested_t.shape}")
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
            raise RuntimeError("native BSIM4 transient could not find a DC initial point")
        V0 = np.asarray([ac["dc_op"][name] for name in topo.solved], dtype=float)
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

    def term_value(term, x, sample):
        return plan.term_value(term, x, input_matrix[:, sample])

    def add_derivative(matrix, row, term, value):
        if row is not None and term[0] == TERM_SOLVED:
            matrix[row, term[1]] += value

    def device_state(item, x, sample):
        dev = devices[item.name]
        vs = term_value(item.s, x, sample)
        vd = term_value(item.d, x, sample)
        vg = term_value(item.g, x, sample)
        currents = dev.get_terminal_currents(vs, vd, vg)
        charges = dev.get_terminal_charges(vs, vd, vg)
        conductance, capacitance = dev.get_terminal_linearization(vs, vd, vg)
        return currents, charges, conductance, capacitance

    def coefficients(sample):
        h = float(tgrid[sample] - tgrid[sample - 1])
        if method == "be" or sample == 1:
            return (1.0 / h, -1.0 / h, 0.0)
        h_prev = float(tgrid[sample - 1] - tgrid[sample - 2])
        rho = h / h_prev
        if rho > 2.0:
            return (1.0 / h, -1.0 / h, 0.0)
        return (
            (1.0 + 2.0 * rho) / ((1.0 + rho) * h),
            -(1.0 + rho) / h,
            (rho * rho) / ((1.0 + rho) * h),
        )

    nnear = 0
    failed_residuals = []
    near_residuals = []
    from .rust_transient import solve_bsim4_rust

    xhist, nfail, first_fail = solve_bsim4_rust(
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
    )
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

    for item in plan.devices:
        currents = np.zeros((len(tgrid), 4), dtype=float)
        charges = np.zeros((len(tgrid), 4), dtype=float)
        for sample in range(len(tgrid)):
            currents[sample], charges[sample], _, _ = device_state(
                item, xhist[sample], sample)
        total = currents.copy()
        for sample in range(1, len(tgrid)):
            a0, a1, a2 = coefficients(sample)
            previous2 = charges[sample - 2] if sample > 1 else charges[sample - 1]
            total[sample] += (
                a0 * charges[sample]
                + a1 * charges[sample - 1]
                + a2 * previous2
            )
        terminals = (
            rail_for_node(item.d_node),
            rail_for_node(item.g_node),
            rail_for_node(item.s_node),
            bulk_rail(devices[item.name]),
        )
        for terminal_index, rail in enumerate(terminals):
            if rail is not None:
                # BSIM terminal currents are positive into the device. Branch
                # currents reported for ideal sources are positive into the
                # source, so source-delivered current has the opposite sign.
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
        a = np.asarray([
            term_value(item.a, xhist[sample], sample)
            for sample in range(len(tgrid))
        ])
        b = np.asarray([
            term_value(item.b, xhist[sample], sample)
            for sample in range(len(tgrid))
        ])
        current = (a - b) * item.g
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
        voltage = np.asarray([
            term_value(item.a, xhist[sample], sample)
            - term_value(item.b, xhist[sample], sample)
            for sample in range(len(tgrid))
        ])
        current = np.zeros(len(tgrid), dtype=float)
        for sample in range(1, len(tgrid)):
            a0, a1, a2 = coefficients(sample)
            previous2 = voltage[sample - 2] if sample > 1 else voltage[sample - 1]
            current[sample] = item.value * (
                a0 * voltage[sample]
                + a1 * voltage[sample - 1]
                + a2 * previous2
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
        "nsubsteps": int(len(tgrid) - len(requested_t)),
        "bsim4_native_transient": True,
        "bsim4_rust_transient": True,
        "backend": "bsim4_native",
        "integration_method": "gear2" if method in {"gear2", "bdf2"} else "be",
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
    for legacy in ("VOP", "VON"):
        if legacy in nodes:
            result[legacy.lower()] = nodes[legacy]
    return result
