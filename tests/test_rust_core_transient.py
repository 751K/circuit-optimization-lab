"""R3 circuit-level parity tests for the Rust OTFT transient core."""
from __future__ import annotations

import numpy as np
import pytest

from circuitopt import transient_solver as transient_module
from circuitopt._rust_transient import build_otft_transient_problem
from circuitopt.numba_kernels import (
    _fill_prev_terms_impl,
    _stamp_transient_system_impl,
    _transient_solve_adaptive_gear2_impl,
    _transient_solve_grid_gear2_impl,
    _transient_solve_grid_impl,
)
from circuitopt.topology import Topology
from circuitopt.transient_solver import (
    _marshal_transient,
    _numba_adaptive_gear2_kernel_args,
    _numba_grid_kernel_args,
    _numba_shared_kernel_arg_groups,
)

try:
    import circuitopt_core
except ImportError:  # pragma: no cover - optional compiled wheel
    circuitopt_core = None


requires_rust_transient = pytest.mark.skipif(
    circuitopt_core is None
    or not hasattr(circuitopt_core, "OtftTransientProblem"),
    reason="R3 circuitopt_core transient extension is not installed",
)


def _reference_stamp(ctx, state, previous, input_now, input_previous, h,
                     cap_mode, bdf, previous2=None, input_previous2=None):
    groups = _numba_shared_kernel_arg_groups(ctx)
    (solver, device_terms, device_nodes, model_params, op_cache, passives,
     sources, cap_clip, vsources, vcvs, cccs, ccvs) = groups
    n, _maxit, _step_limit, _vtol, gmin, _fallback, _fallback_tol, hh = solver
    (dev_d_kind, dev_d_ref, dev_d_val,
     dev_g_kind, dev_g_ref, dev_g_val,
     dev_s_kind, dev_s_ref, dev_s_val) = device_terms
    dev_di, dev_gi, dev_si, dev_use_abs = device_nodes
    (p_vfb, p_vss, p_lc, p_lambda, p_contact_scale, p_exponent,
     p_current_scale, p_inv_rleak, p_two_over_pi, p_cap_cgs1,
     p_cap_cgd1, p_cap_half_wl_ci, p_cap_cgs3_base, p_cap_cgd3_base,
     p_k1, p_gate_leak_g) = model_params
    op_cache_valid, op_cache_vs1, op_cache_vd1 = op_cache
    (res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref, res_b_val,
     res_ai, res_bi, res_g, cap_a_kind, cap_a_ref, cap_a_val,
     cap_b_kind, cap_b_ref, cap_b_val, cap_ai, cap_bi, cap_value) = passives
    isrc_pi, isrc_qi, isrc_value, dyn_pi, dyn_qi, dyn_input_idx = sources

    count = len(dev_di)
    prev_vs = np.empty(count)
    prev_vd = np.empty(count)
    prev_vg = np.empty(count)
    prev_cgs = np.empty(count)
    prev_cgd = np.empty(count)
    cap_prev_dv = np.empty(len(cap_value))
    assert _fill_prev_terms_impl(
        previous, input_previous,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        p_vfb, p_vss, p_lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1,
        op_cache_valid, op_cache_vs1, op_cache_vd1,
        cap_a_kind, cap_a_ref, cap_a_val,
        cap_b_kind, cap_b_ref, cap_b_val, cap_mode,
        prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd, cap_prev_dv)

    if previous2 is None:
        prev2_cgs = prev_cgs.copy()
        prev2_cgd = prev_cgd.copy()
        cap_prev2_dv = cap_prev_dv.copy()
    else:
        prev2_cgs = np.empty(count)
        prev2_cgd = np.empty(count)
        cap_prev2_dv = np.empty(len(cap_value))
        scratch_vs = np.empty(count)
        scratch_vd = np.empty(count)
        scratch_vg = np.empty(count)
        cache2_valid = np.zeros(count, dtype=np.bool_)
        cache2_vs1 = np.zeros(count)
        cache2_vd1 = np.zeros(count)
        assert _fill_prev_terms_impl(
            previous2,
            input_previous if input_previous2 is None else input_previous2,
            dev_d_kind, dev_d_ref, dev_d_val,
            dev_g_kind, dev_g_ref, dev_g_val,
            dev_s_kind, dev_s_ref, dev_s_val,
            p_vfb, p_vss, p_lc, p_lambda, p_contact_scale, p_exponent,
            p_current_scale, p_inv_rleak,
            p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
            p_cap_cgs3_base, p_cap_cgd3_base, p_k1,
            cache2_valid, cache2_vs1, cache2_vd1,
            cap_a_kind, cap_a_ref, cap_a_val,
            cap_b_kind, cap_b_ref, cap_b_val, cap_mode,
            scratch_vs, scratch_vd, scratch_vg, prev2_cgs, prev2_cgd,
            cap_prev2_dv)

    residual = np.empty(ctx.n_aug)
    jacobian = np.empty((ctx.n_aug, ctx.n_aug))
    profile_stats = np.zeros(26)
    ok = _stamp_transient_system_impl(
        state, previous, input_now, input_previous, h, n, gmin, hh,
        dev_d_kind, dev_d_ref, dev_d_val,
        dev_g_kind, dev_g_ref, dev_g_val,
        dev_s_kind, dev_s_ref, dev_s_val,
        dev_di, dev_gi, dev_si, dev_use_abs,
        p_vfb, p_vss, p_lc, p_lambda, p_contact_scale, p_exponent,
        p_current_scale, p_inv_rleak,
        p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
        p_cap_cgs3_base, p_cap_cgd3_base, p_k1, p_gate_leak_g,
        op_cache_valid, op_cache_vs1, op_cache_vd1,
        res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref,
        res_b_val, res_ai, res_bi, res_g,
        cap_a_kind, cap_a_ref, cap_a_val, cap_b_kind, cap_b_ref,
        cap_b_val, cap_ai, cap_bi, cap_value,
        isrc_pi, isrc_qi, isrc_value, dyn_pi, dyn_qi, dyn_input_idx,
        cap_mode, prev_vs, prev_vd, prev_vg, prev_cgs, prev_cgd,
        cap_prev_dv, residual, jacobian, False, profile_stats,
        *bdf, prev2_cgs, prev2_cgd, cap_prev2_dv,
        vsources, vcvs, cccs, ccvs)
    return ok, residual, jacobian


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


@requires_rust_transient
@pytest.mark.parametrize("cap_mode", [0, 1])
@pytest.mark.parametrize("use_bdf2", [False, True])
def test_rust_otft_stamp_matches_numba(cap_mode, use_bdf2):
    ctx, inputs = _device_context(cap_mode)
    problem = build_otft_transient_problem(ctx)
    state = np.array([19.7])
    previous = np.array([20.0])
    previous2 = np.array([20.1]) if use_bdf2 else None
    bdf = (1.5, -2.0, 0.5) if use_bdf2 else (1.0, -1.0, 0.0)
    input_previous2 = inputs[:, 0] - 0.1 if use_bdf2 else None

    ref_ok, ref_residual, ref_jacobian = _reference_stamp(
        ctx, state, previous, inputs[:, 1], inputs[:, 0], 1e-6,
        cap_mode, bdf, previous2, input_previous2)
    got_ok, got_residual, got_jacobian, _caches, _stats = problem.stamp(
        state, previous, inputs[:, 1], inputs[:, 0], 1e-6,
        cap_mode=cap_mode, bdf=bdf, previous2_state=previous2,
        input_previous2=input_previous2)

    assert got_ok is ref_ok is True
    np.testing.assert_allclose(got_residual, ref_residual, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(got_jacobian, ref_jacobian, rtol=1e-12, atol=1e-18)


@requires_rust_transient
def test_rust_augmented_branch_stamp_matches_numba():
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

    ref_ok, ref_residual, ref_jacobian = _reference_stamp(
        ctx, state, previous, marshalled.input_values[:, 1],
        marshalled.input_values[:, 0], 1e-6, 0, (1.0, -1.0, 0.0))
    got_ok, got_residual, got_jacobian, _caches, _stats = problem.stamp(
        state, previous, marshalled.input_values[:, 1],
        marshalled.input_values[:, 0], 1e-6)

    assert got_ok is ref_ok is True
    np.testing.assert_allclose(got_residual, ref_residual, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(got_jacobian, ref_jacobian, rtol=1e-12, atol=1e-18)


@requires_rust_transient
@pytest.mark.parametrize("integration_method", ["be", "gear2"])
@pytest.mark.parametrize("max_step", [None, 4e-7])
def test_rust_fixed_grid_matches_numba(integration_method, max_step):
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
    args = _numba_grid_kernel_args(
        ctx, marshalled.V0, tgrid, marshalled.input_values, edge_mask, True)
    kernel = (_transient_solve_grid_gear2_impl
              if integration_method == "gear2" else _transient_solve_grid_impl)
    ref_ok, ref_states, ref_substeps, ref_failed, ref_profile, ref_indices = kernel(*args)

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

    assert isinstance(got_states, np.ndarray)
    assert isinstance(got_profile, np.ndarray)
    assert got_ok is bool(ref_ok)
    assert got_substeps == ref_substeps
    assert got_failed == ref_failed
    assert got_indices == [int(value) for value in ref_indices if value >= 0]
    np.testing.assert_allclose(got_states, ref_states, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(
        np.asarray(got_profile)[:16], np.asarray(ref_profile)[:16],
        rtol=0.0, atol=0.0)


@requires_rust_transient
def test_rust_gear2_binary_retry_matches_numba_exactly():
    """Exercise the path where the full Newton step fails but 8 slices recover."""
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
    args = _numba_grid_kernel_args(
        ctx, marshalled.V0, times, marshalled.input_values,
        np.zeros(len(times), dtype=bool), True)
    ref = _transient_solve_grid_gear2_impl(*args)
    ref_ok, ref_states, ref_substeps, ref_failed, ref_profile, ref_indices = ref

    got = build_otft_transient_problem(ctx).solve_fixed_grid(
        marshalled.V0, times, marshalled.input_values, [False] * len(times),
        integration_method="gear2", max_step=-1.0, flat_max_step=-1.0,
        max_retry_subdivisions=3, max_iterations=3, step_limit=0.05,
        voltage_tolerance=ctx.newton_vtol, fallback_accept=False,
        fallback_tolerance=ctx.fallback_tol, clip_lo=ctx.clip_lo,
        clip_hi=ctx.clip_hi, gmin=ctx.gmin, hh=ctx.HH,
        cap_mode=ctx.cap_id, profile=True)
    got_ok, got_states, got_substeps, got_failed, got_indices, got_profile = got

    assert got_ok is bool(ref_ok) is True
    assert got_substeps == ref_substeps == 9
    assert got_failed == ref_failed == -1
    assert got_indices == [int(value) for value in ref_indices if value >= 0]
    assert got_profile[10] == ref_profile[10] == 1.0
    np.testing.assert_allclose(got_states, ref_states, rtol=1e-12, atol=1e-16)
    np.testing.assert_array_equal(got_profile, ref_profile)


@requires_rust_transient
@pytest.mark.parametrize("integration_method", ["be", "gear2"])
def test_public_transient_dispatches_to_rust(monkeypatch, integration_method):
    topo = Topology(
        solved=["OUT"], devices=[], rails={"VDD": 1.0, "GND": 0.0},
        resistors=[("R", "OUT", "GND", 1e3)],
        capacitors=[("C", "OUT", "GND", 1e-9)],
        isources=[("I", "VDD", "OUT", 1e-3)], outputs=("OUT",))
    times = np.linspace(0.0, 10e-6, 31)
    kwargs = dict(topo=topo, V0=np.array([0.0]),
                  integration_method=integration_method, profile=True)

    monkeypatch.setattr(transient_module, "current_engine", lambda: "numba")
    reference = transient_module.transient({}, {}, times, **kwargs)
    monkeypatch.setattr(transient_module, "current_engine", lambda: "rust")
    got = transient_module.transient({}, {}, times, **kwargs)

    assert got["rust_grid_solver"] is True
    assert got["numba_grid_solver"] is False
    assert got["transient_profile"]["rust_grid_solver"] is True
    assert got["nfail"] == reference["nfail"]
    assert got["nsubsteps"] == reference["nsubsteps"]
    np.testing.assert_allclose(got["output"], reference["output"],
                               rtol=1e-12, atol=1e-16)


@requires_rust_transient
def test_rust_adaptive_gear2_matches_numba_behavior():
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
    args = _numba_adaptive_gear2_kernel_args(
        ctx, marshalled.V0, times, marshalled.input_values, True)
    ref = _transient_solve_adaptive_gear2_impl(*args)
    (ref_ok, ref_times, ref_states, ref_inputs, ref_count, ref_substeps,
     ref_rejected, ref_profile) = ref

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

    assert all(isinstance(value, np.ndarray) for value in (
        got_times, got_states, got_inputs, got_profile))
    assert got_ok is bool(ref_ok)
    assert len(got_times) == ref_count
    assert got_substeps == ref_substeps
    assert got_rejected == ref_rejected
    np.testing.assert_allclose(got_times, ref_times[:ref_count], rtol=0.0, atol=1e-18)
    np.testing.assert_allclose(got_states, ref_states[:ref_count], rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(got_inputs, ref_inputs[:ref_count], rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(np.asarray(got_profile)[:6],
                               np.asarray(ref_profile)[:6], rtol=0.0, atol=0.0)


@requires_rust_transient
def test_rust_nonlinear_adaptive_rejection_matches_numba_exactly():
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
    ref = _transient_solve_adaptive_gear2_impl(
        *_numba_adaptive_gear2_kernel_args(
            ctx, marshalled.V0, times, marshalled.input_values, True))
    (ref_ok, ref_times, ref_states, ref_inputs, ref_count, ref_substeps,
     ref_rejected, ref_profile) = ref

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

    assert got_ok is bool(ref_ok) is True
    assert len(got_times) == ref_count == 14
    assert got_substeps == ref_substeps == 42
    assert got_rejected == ref_rejected == 1
    np.testing.assert_array_equal(got_times, ref_times[:ref_count])
    np.testing.assert_allclose(got_states, ref_states[:ref_count],
                               rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(got_inputs, ref_inputs[:ref_count],
                               rtol=1e-12, atol=1e-16)
    np.testing.assert_array_equal(got_profile, ref_profile)


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
def test_public_adaptive_transient_dispatches_to_rust(monkeypatch):
    topo = Topology(
        solved=["OUT"], devices=[], rails={"GND": 0.0},
        resistors=[("R", "OUT", "GND", 1e3)],
        capacitors=[("C", "OUT", "GND", 1e-9)], outputs=("OUT",))
    times = np.array([0.0, 2e-6, 2.1e-6, 5e-6])
    current = np.array([0.0, 0.0, 1e-3, 1e-3])
    kwargs = dict(
        topo=topo, V0=np.array([0.0]), inputs={"iin": current},
        current_inputs=[("GND", "OUT", "iin")],
        integration_method="gear2", adaptive=True, profile=True,
        adaptive_reltol=1e-5, adaptive_vabstol=1e-8,
        adaptive_max_steps=5000)

    monkeypatch.setattr(transient_module, "current_engine", lambda: "numba")
    reference = transient_module.transient({}, {}, times, **kwargs)
    monkeypatch.setattr(transient_module, "current_engine", lambda: "rust")
    got = transient_module.transient({}, {}, times, **kwargs)

    assert got["rust_adaptive_solver"] is True
    assert got["numba_adaptive_solver"] is False
    assert got["adaptive_accepted_steps"] == reference["adaptive_accepted_steps"]
    assert got["adaptive_rejected_steps"] == reference["adaptive_rejected_steps"]
    np.testing.assert_allclose(got["t"], reference["t"], rtol=0.0, atol=1e-18)
    np.testing.assert_allclose(got["output"], reference["output"],
                               rtol=1e-12, atol=1e-16)
