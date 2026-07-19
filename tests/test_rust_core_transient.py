"""Circuit-level tests for the compiled OTFT transient core.

History: these started as R3 rust-vs-numba differential tests. The Python/numba
`_impl` drivers were removed in v2.0.0 (R7) — the golden corpus
(``tests/golden/engine_parity``) is the frozen numerical oracle now — so the
tests assert the compiled core directly: stamp self-consistency against a
finite-difference Jacobian, analytic solutions where the circuit has one, and
frozen behavioral pins (substep/rejection counts) that were originally
established by the retired A/B comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

from circuitopt import transient_solver as transient_module
from circuitopt._rust_transient import build_otft_transient_problem
from circuitopt.topology import Topology
from circuitopt.transient_profile import PROFILE_INTERVALS
from circuitopt.transient_solver import _marshal_transient

try:
    import circuitopt_core
except ImportError:  # pragma: no cover - optional compiled wheel
    circuitopt_core = None


requires_rust_transient = pytest.mark.skipif(
    circuitopt_core is None
    or not hasattr(circuitopt_core, "OtftTransientProblem"),
    reason="circuitopt_core transient extension is not installed",
)


def _device_context(cap_mode):
    topo = Topology(
        solved=["OUT"],
        devices=[("M1", "OUT", "IN", "VDD")],
        rails={"VDD": "VDD", "GND": 0.0, "IN": "VIN"},
        transient_inputs={"M1": "vin"},
        resistors=[("RL", "OUT", "GND", 2e6)],
        capacitors=[("CL", "OUT", "GND", 3e-12)],
        isources=[("IB", "VDD", "OUT", 2e-6)],
        outputs=("OUT",),
    )
    tgrid = np.array([0.0, 1e-6])
    waveform = np.array([25.0, 25.2])
    marshalled = _marshal_transient(
        {"M1": (2000, 80)}, {"VDD": 40.0, "VIN": 25.0}, tgrid,
        topo=topo, V0=np.array([20.0]), inputs={"vin": waveform},
        cap_mode_id=cap_mode)
    return marshalled.ctx, marshalled.input_values


def _stamp_fd_jacobian(problem, state, previous, input_now, input_previous, h,
                       *, cap_mode=0, bdf=(1.0, -1.0, 0.0), previous2=None,
                       input_previous2=None, step=1e-7):
    """Central finite differences of the stamped residual w.r.t. the state."""
    state = np.asarray(state, float)
    n = len(state)
    jac = np.zeros((n, n))
    for j in range(n):
        hi = state.copy()
        lo = state.copy()
        hi[j] += step
        lo[j] -= step
        _, r_hi, _, _, _ = problem.stamp(
            hi, previous, input_now, input_previous, h, cap_mode=cap_mode,
            bdf=bdf, previous2_state=previous2,
            input_previous2=input_previous2)
        _, r_lo, _, _, _ = problem.stamp(
            lo, previous, input_now, input_previous, h, cap_mode=cap_mode,
            bdf=bdf, previous2_state=previous2,
            input_previous2=input_previous2)
        jac[:, j] = (np.asarray(r_hi) - np.asarray(r_lo)) / (2 * step)
    return jac


@requires_rust_transient
@pytest.mark.parametrize("cap_mode", [0, 1])
@pytest.mark.parametrize("use_bdf2", [False, True])
def test_rust_otft_stamp_newton_converges(cap_mode, use_bdf2):
    """The stamped residual/Jacobian pair must drive Newton to convergence.

    Replaces the retired rust-vs-`_impl` stamp comparison: the residual and
    Jacobian are exercised together — an inconsistent pair cannot contract the
    residual by orders of magnitude in a handful of steps. (The stamped device
    Jacobian is an hh-smoothed terminal derivative by design, so a raw
    finite-difference identity is not applicable to the OTFT arm.)"""
    ctx, inputs = _device_context(cap_mode)
    problem = build_otft_transient_problem(ctx)
    state = np.array([19.7])
    previous = np.array([20.0])
    previous2 = np.array([20.1]) if use_bdf2 else None
    bdf = (1.5, -2.0, 0.5) if use_bdf2 else (1.0, -1.0, 0.0)
    input_previous2 = inputs[:, 0] - 0.1 if use_bdf2 else None

    ok, residual, jacobian, _caches, _stats = problem.stamp(
        state, previous, inputs[:, 1], inputs[:, 0], 1e-6,
        cap_mode=cap_mode, bdf=bdf, previous2_state=previous2,
        input_previous2=input_previous2)
    assert ok is True
    assert np.all(np.isfinite(residual))
    assert np.all(np.isfinite(np.asarray(jacobian)))
    r0 = float(np.max(np.abs(residual)))

    x = state.copy()
    r_last = r0
    for _ in range(20):
        step_ok, r, jac, _caches, _stats = problem.stamp(
            x, previous, inputs[:, 1], inputs[:, 0], 1e-6,
            cap_mode=cap_mode, bdf=bdf, previous2_state=previous2,
            input_previous2=input_previous2)
        assert step_ok is True
        r = np.asarray(r, float)
        r_last = float(np.max(np.abs(r)))
        if r_last < 1e-13:
            break
        x = x - np.linalg.solve(np.asarray(jac, float), r)
    assert r_last < 1e-9 and r_last < r0 * 1e-3


@requires_rust_transient
def test_rust_augmented_branch_stamp_is_consistent():
    """Controlled sources (vsource/vcvs/cccs/ccvs) stamp a linear system whose
    Jacobian must match FD, and the residual must be affine in the state."""
    topo = Topology(
        solved=["OUT", "CTRL"], devices=[], rails={"GND": 0.0},
        resistors=[("RO", "OUT", "GND", 1e3),
                   ("RC", "CTRL", "GND", 2e3)],
        capacitors=[("CO", "OUT", "GND", 1e-9)],
        vsources=[("VIN", "CTRL", "GND", "vin")],
        vcvs=[("E1", "OUT", "GND", "CTRL", "GND", 2.0)],
        cccs=[("F1", "OUT", "GND", "VIN", 0.25)],
        ccvs=[("H1", "OUT", "GND", "VIN", 10.0)],
        outputs=("OUT",),
    )
    tgrid = np.array([0.0, 1e-6])
    marshalled = _marshal_transient(
        {}, {}, tgrid, topo=topo, V0=np.zeros(topo.n_aug),
        inputs={"vin": np.array([0.1, 0.2])})
    ctx = marshalled.ctx
    problem = build_otft_transient_problem(ctx)
    state = np.array([0.15, 0.2, 1e-4, -2e-4, 3e-4])
    previous = np.array([0.1, 0.1, 0.0, 0.0, 0.0])

    ok, residual, jacobian, _caches, _stats = problem.stamp(
        state, previous, marshalled.input_values[:, 1],
        marshalled.input_values[:, 0], 1e-6)
    assert ok is True
    fd = _stamp_fd_jacobian(
        problem, state, previous, marshalled.input_values[:, 1],
        marshalled.input_values[:, 0], 1e-6)
    np.testing.assert_allclose(np.asarray(jacobian), fd, rtol=1e-6, atol=1e-6)
    # Linear system: residual(state) == residual(0) + J @ state, bitwise-tight.
    ok0, residual0, _, _, _ = problem.stamp(
        np.zeros_like(state), previous, marshalled.input_values[:, 1],
        marshalled.input_values[:, 0], 1e-6)
    assert ok0 is True
    np.testing.assert_allclose(
        np.asarray(residual),
        np.asarray(residual0) + np.asarray(jacobian) @ state,
        rtol=1e-9, atol=1e-12)


@requires_rust_transient
@pytest.mark.parametrize("integration_method", ["be", "gear2"])
@pytest.mark.parametrize("max_step", [None, 4e-7])
def test_rust_fixed_grid_solves_and_reports_profile(integration_method, max_step):
    topo = Topology(
        solved=["OUT"],
        devices=[("M1", "OUT", "IN", "VDD")],
        rails={"VDD": "VDD", "GND": 0.0, "IN": "VIN"},
        transient_inputs={"M1": "vin"},
        resistors=[("RL", "OUT", "GND", 2e6)],
        capacitors=[("CL", "OUT", "GND", 3e-12)],
        isources=[("IB", "VDD", "OUT", 2e-6)],
        outputs=("OUT",),
    )
    tgrid = np.linspace(0.0, 3e-6, 7)
    waveform = 25.0 + 0.15 * np.sin(2.0 * np.pi * tgrid / tgrid[-1])
    edge_mask = np.array([False, True, True, False, False, True, False])
    marshalled = _marshal_transient(
        {"M1": (2000, 80)}, {"VDD": 40.0, "VIN": 25.0}, tgrid,
        topo=topo, V0=np.array([20.0]), inputs={"vin": waveform},
        integration_method=integration_method, max_step=max_step,
        edge_mask=edge_mask, profile=True)
    ctx = marshalled.ctx
    problem = build_otft_transient_problem(ctx)
    got = problem.solve_fixed_grid(
        marshalled.V0, tgrid, marshalled.input_values, edge_mask.tolist(),
        integration_method=integration_method,
        max_step=-1.0 if max_step is None else max_step,
        flat_max_step=-1.0, max_retry_subdivisions=0,
        max_iterations=ctx.newton_maxit, step_limit=ctx.newton_step_limit,
        voltage_tolerance=ctx.newton_vtol,
        fallback_accept=False, fallback_tolerance=ctx.fallback_tol,
        clip_lo=ctx.clip_lo, clip_hi=ctx.clip_hi,
        gmin=ctx.gmin, hh=ctx.HH, cap_mode=ctx.cap_id, profile=True)
    got_ok, got_states, got_substeps, got_failed, got_indices, got_profile = got

    assert got_ok is True
    assert got_failed == -1
    assert got_indices == []
    states = np.asarray(got_states)
    assert states.shape == (len(tgrid), 1)
    assert np.all(np.isfinite(states))
    # max_step subdivides every interval beyond the base one-per-interval count.
    base_substeps = len(tgrid) - 1
    if max_step is None:
        assert got_substeps == base_substeps
    else:
        assert got_substeps > base_substeps
    profile = np.asarray(got_profile)
    assert np.all(np.isfinite(profile))
    assert profile[PROFILE_INTERVALS] == float(len(tgrid) - 1)


@requires_rust_transient
def test_rust_gear2_binary_retry_recovers_with_slices():
    """Behavioral pin: the full Newton step fails, and 8 binary slices recover.

    The 9-substep count and the retry-profile slot were originally established
    by the retired numba A/B comparison; they are the frozen contract now."""
    topo = Topology(
        solved=["OUT"],
        devices=[("M1", "OUT", "IN", "VDD")],
        rails={"VDD": "VDD", "GND": 0.0, "IN": "VIN"},
        transient_inputs={"M1": "vin"},
        resistors=[("RL", "OUT", "GND", 2e6)],
        capacitors=[("CL", "OUT", "GND", 3e-12)],
        isources=[("IB", "VDD", "OUT", 2e-6)],
        outputs=("OUT",),
    )
    times = np.array([0.0, 1e-7, 2e-7])
    waveform = np.array([25.0, 25.2, 24.8])
    marshalled = _marshal_transient(
        {"M1": (2000, 80)}, {"VDD": 40.0, "VIN": 25.0}, times,
        topo=topo, V0=np.array([20.0]), inputs={"vin": waveform},
        integration_method="gear2", max_retry_subdivisions=3,
        newton_maxit=3, newton_step_limit=0.05, profile=True)
    ctx = marshalled.ctx

    got = build_otft_transient_problem(ctx).solve_fixed_grid(
        marshalled.V0, times, marshalled.input_values, [False] * len(times),
        integration_method="gear2", max_step=-1.0, flat_max_step=-1.0,
        max_retry_subdivisions=3, max_iterations=3, step_limit=0.05,
        voltage_tolerance=ctx.newton_vtol, fallback_accept=False,
        fallback_tolerance=ctx.fallback_tol, clip_lo=ctx.clip_lo,
        clip_hi=ctx.clip_hi, gmin=ctx.gmin, hh=ctx.HH,
        cap_mode=ctx.cap_id, profile=True)
    got_ok, got_states, got_substeps, got_failed, got_indices, got_profile = got

    assert got_ok is True
    assert got_substeps == 9
    assert got_failed == -1
    assert got_indices == []
    assert got_profile[10] == 1.0  # one interval went through the retry path
    assert np.all(np.isfinite(np.asarray(got_states)))


@requires_rust_transient
@pytest.mark.parametrize("integration_method", ["be", "gear2"])
def test_public_transient_uses_rust_grid(integration_method):
    """The public transient always runs the compiled fixed-grid core and the
    linear RC result matches the analytic step response."""
    topo = Topology(
        solved=["OUT"], devices=[], rails={"VDD": 1.0, "GND": 0.0},
        resistors=[("R", "OUT", "GND", 1e3)],
        capacitors=[("C", "OUT", "GND", 1e-9)],
        isources=[("I", "VDD", "OUT", 1e-3)], outputs=("OUT",))
    times = np.linspace(0.0, 10e-6, 31)
    got = transient_module.transient(
        {}, {}, times, topo=topo, V0=np.array([0.0]),
        integration_method=integration_method, profile=True)

    assert got["rust_grid_solver"] is True
    assert got["rust_adaptive_solver"] is False
    assert got["transient_profile"]["rust_grid_solver"] is True
    assert got["nfail"] == 0
    assert got["nsubsteps"] == len(times) - 1
    # Analytic: V(t) = I*R*(1 - exp(-t/RC)), tau = 1 us, endpoint ~ 1 V.
    # BE is first order (discretization error ~ h/tau ~ 5.5e-2 on this grid);
    # gear2/BDF2 is second order and much tighter.
    analytic = 1e-3 * 1e3 * (1.0 - np.exp(-times / 1e-6))
    np.testing.assert_allclose(got["output"][-1], analytic[-1], rtol=1e-3)
    # gear2 starts with one BE step (BDF2 needs history), so its worst point
    # is the first interval: ~3.4e-2 on this grid vs BE's ~5.4e-2.
    atol = 7e-2 if integration_method == "be" else 4e-2
    np.testing.assert_allclose(got["output"], analytic, atol=atol)


@requires_rust_transient
def test_rust_adaptive_gear2_matches_analytic_rc():
    """Adaptive gear2 on a linear RC current step: accepted grid is monotone,
    endpoints preserved, and the trajectory matches the analytic response."""
    topo = Topology(
        solved=["OUT"], devices=[], rails={"GND": 0.0},
        resistors=[("R", "OUT", "GND", 1e3)],
        capacitors=[("C", "OUT", "GND", 1e-9)], outputs=("OUT",))
    times = np.array([0.0, 2e-6, 2.1e-6, 5e-6])
    current = np.array([0.0, 0.0, 1e-3, 1e-3])
    marshalled = _marshal_transient(
        {}, {}, times, topo=topo, V0=np.array([0.0]),
        inputs={"iin": current}, current_inputs=[("GND", "OUT", "iin")],
        integration_method="gear2", adaptive=True, profile=True,
        adaptive_reltol=1e-5, adaptive_vabstol=1e-8,
        adaptive_max_steps=5000)
    ctx = marshalled.ctx

    problem = build_otft_transient_problem(ctx)
    got = problem.solve_adaptive_gear2(
        marshalled.V0, times, marshalled.input_values,
        max_step=-1.0, reltol=ctx.adaptive_config.reltol,
        voltage_abstol=ctx.adaptive_config.vabstol,
        current_abstol=ctx.adaptive_config.iabstol,
        max_steps=ctx.adaptive_config.max_steps, initial_step=-1.0,
        max_iterations=ctx.newton_maxit, step_limit=ctx.newton_step_limit,
        voltage_tolerance=ctx.newton_vtol, fallback_accept=False,
        fallback_tolerance=ctx.fallback_tol, clip_lo=ctx.clip_lo,
        clip_hi=ctx.clip_hi, gmin=ctx.gmin, hh=ctx.HH,
        cap_mode=ctx.cap_id, profile=True)
    (got_ok, got_times, got_states, got_inputs, got_substeps,
     got_rejected, got_profile) = got

    assert got_ok is True
    assert all(isinstance(value, np.ndarray) for value in (
        got_times, got_states, got_inputs, got_profile))
    t = np.asarray(got_times)
    assert t[0] == times[0] and t[-1] == times[-1]
    assert np.all(np.diff(t) > 0)
    assert got_substeps >= len(t) - 1
    assert got_rejected >= 0
    # Analytic RC response to the 1 mA step ramped over 2.0-2.1us; use the
    # ramp midpoint 2.05us as the effective step time.
    v = np.asarray(got_states)[:, 0]
    tau = 1e-6
    late = t >= 2.1e-6
    analytic_late = 1.0 - np.exp(-(t[late] - 2.05e-6) / tau)
    np.testing.assert_allclose(v[t <= 2e-6], 0.0, atol=1e-12)
    np.testing.assert_allclose(v[late], analytic_late, atol=6e-2)


@requires_rust_transient
def test_rust_nonlinear_adaptive_rejection_pins():
    """Behavioral pin for the OTFT adaptive controller: accepted-point count,
    substeps, and the single LTE rejection. Originally established by the
    retired numba A/B comparison; frozen as the direct rust contract."""
    topo = Topology(
        solved=["OUT"],
        devices=[("M1", "OUT", "IN", "VDD")],
        rails={"VDD": "VDD", "GND": 0.0, "IN": "VIN"},
        transient_inputs={"M1": "vin"},
        resistors=[("RL", "OUT", "GND", 2e6)],
        capacitors=[("CL", "OUT", "GND", 3e-12)],
        isources=[("IB", "VDD", "OUT", 2e-6)],
        outputs=("OUT",),
    )
    times = np.array([0.0, 2.5e-6, 5e-6, 1e-5])
    waveform = np.array([25.0, 25.0, 25.2, 24.8])
    marshalled = _marshal_transient(
        {"M1": (2000, 80)}, {"VDD": 40.0, "VIN": 25.0}, times,
        topo=topo, V0=np.array([20.0]), inputs={"vin": waveform},
        integration_method="gear2", adaptive=True, profile=True,
        adaptive_reltol=1e-4, adaptive_vabstol=1e-6,
        adaptive_max_steps=10000)
    ctx = marshalled.ctx

    got = build_otft_transient_problem(ctx).solve_adaptive_gear2(
        marshalled.V0, times, marshalled.input_values,
        max_step=-1.0, reltol=ctx.adaptive_config.reltol,
        voltage_abstol=ctx.adaptive_config.vabstol,
        current_abstol=ctx.adaptive_config.iabstol,
        max_steps=ctx.adaptive_config.max_steps, initial_step=-1.0,
        max_iterations=ctx.newton_maxit, step_limit=ctx.newton_step_limit,
        voltage_tolerance=ctx.newton_vtol, fallback_accept=False,
        fallback_tolerance=ctx.fallback_tol, clip_lo=ctx.clip_lo,
        clip_hi=ctx.clip_hi, gmin=ctx.gmin, hh=ctx.HH,
        cap_mode=ctx.cap_id, profile=True)
    (got_ok, got_times, got_states, got_inputs, got_substeps,
     got_rejected, got_profile) = got

    assert got_ok is True
    assert len(got_times) == 14
    assert got_substeps == 42
    assert got_rejected == 1
    assert np.all(np.diff(np.asarray(got_times)) > 0)
    assert np.all(np.isfinite(np.asarray(got_states)))


@requires_rust_transient
def test_rust_transient_rejects_invalid_topology_and_state_lengths():
    invalid = {
        "node_count": 1,
        "size": 1,
        "devices": [],
        "resistors": [((0, 0, 0.0), (2, 0, 0.0), 2, -1, 1.0)],
        "capacitors": [],
        "current_sources": [],
        "dynamic_sources": [],
        "vccs": [],
        "voltage_sources": [],
        "vcvs": [],
        "cccs": [],
        "ccvs": [],
    }
    with pytest.raises(ValueError, match="invalid OTFT transient topology"):
        circuitopt_core.OtftTransientProblem(invalid)

    ctx, inputs = _device_context(0)
    problem = build_otft_transient_problem(ctx)
    with pytest.raises(ValueError, match="state lengths"):
        problem.stamp([], [20.0], inputs[:, 1], inputs[:, 0], 1e-6)
    with pytest.raises(ValueError, match="h must be positive"):
        problem.newton_step(
            [20.0], [20.0], inputs[:, 1], inputs[:, 0], 0.0)

    noncontiguous_inputs = np.empty((inputs.shape[0], inputs.shape[1] * 2))[:, ::2]
    noncontiguous_inputs[:] = inputs
    assert not noncontiguous_inputs.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        problem.solve_fixed_grid(
            np.array([20.0]), np.array([0.0, 1e-6]),
            noncontiguous_inputs, [False, False])
    with pytest.raises(ValueError, match="fixed-grid transient input"):
        problem.solve_fixed_grid(
            np.array([20.0]), np.array([1e-6, 0.0]), inputs,
            [False, False])
    with pytest.raises(ValueError, match="adaptive transient input"):
        problem.solve_adaptive_gear2(
            np.array([20.0]), np.array([0.0]), inputs[:, :1])


@requires_rust_transient
def test_public_adaptive_transient_uses_rust():
    """The public adaptive transient runs the compiled adaptive core and the
    linear RC result matches the analytic step response at the endpoints."""
    topo = Topology(
        solved=["OUT"], devices=[], rails={"GND": 0.0},
        resistors=[("R", "OUT", "GND", 1e3)],
        capacitors=[("C", "OUT", "GND", 1e-9)], outputs=("OUT",))
    times = np.array([0.0, 2e-6, 2.1e-6, 5e-6])
    current = np.array([0.0, 0.0, 1e-3, 1e-3])
    got = transient_module.transient(
        {}, {}, times, topo=topo, V0=np.array([0.0]),
        inputs={"iin": current}, current_inputs=[("GND", "OUT", "iin")],
        integration_method="gear2", adaptive=True, profile=True,
        adaptive_reltol=1e-5, adaptive_vabstol=1e-8,
        adaptive_max_steps=5000)

    assert got["rust_adaptive_solver"] is True
    assert got["rust_grid_solver"] is False
    assert got["adaptive"] is True
    assert got["adaptive_accepted_steps"] == len(got["t"]) - 1
    assert got["adaptive_rejected_steps"] >= 0
    t = np.asarray(got["t"])
    assert t[0] == times[0] and t[-1] == times[-1]
    # Endpoint: 1 mA * 1 kOhm step ramped over 2.0-2.1us; at t=5us the RC has
    # settled for ~2.95 tau -> V = 1 - exp(-2.95).
    np.testing.assert_allclose(
        got["output"][-1], 1.0 - np.exp(-(5e-6 - 2.05e-6) / 1e-6), rtol=1e-2)
