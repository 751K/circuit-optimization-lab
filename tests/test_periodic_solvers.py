import numpy as np
import pytest

from core.numba_kernels import (pac_hb_blocks_numba, pac_linearize_orbit_numba,
                                transient_solve_adaptive_gear2_numba)
from core.adaptive_config import AdaptiveConfig, adaptive_lte_wrms, adaptive_next_h
import core.numba_kernels as nk
from core.pac_solver import pac_solve
from core.pnoise_solver import pnoise_solve
from core.pss_solver import pss_solve
from core.transient_solver import transient
from core.topology import Topology


_KB = 1.380649e-23
_TEMP = 300.15


def _rc_lowpass_topology(R=1e5, C=1e-9):
    return Topology(
        solved=["OUT"],
        devices=[],
        rails={"VIN": "VIN", "GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "VIN", "OUT", R)],
        capacitors=[("C1", "OUT", "GND", C)],
    )


def test_generic_pac_solves_non_chopper_rc_lowpass():
    R = 1e5
    C = 1e-9
    period = 1e-3
    t = np.linspace(0.0, period, 401)
    topo = _rc_lowpass_topology(R, C)
    pss = pss_solve(
        {}, {"VIN": 0.0}, period, topo=topo, tgrid=t,
        inputs={"vin": np.zeros_like(t)}, node_inputs={"VIN": "vin"},
        V0=np.array([0.0]), residual_tol=1e-12, max_shooting_iters=2,
    )

    freqs = np.array([100.0, 500.0])
    pac = pac_solve(
        {}, {"VIN": 0.0}, freqs, pss_result=pss,
        input_drive={"vin": 1.0},
        transient_kwargs={"max_retry_subdivisions": 0},
    )

    expected = 1.0 / (1.0 + 2j * np.pi * freqs * R * C)
    np.testing.assert_allclose(np.abs(pac["response"]), np.abs(expected), rtol=1e-6)
    assert pac["method"] == "lti_ac_fast_path"
    assert pac["pac_period_runs"] == 0


def test_pss_analytic_jacobian_matches_fd_jacobian():
    # The analytic-monodromy shooting Jacobian must converge to the same orbit as
    # the finite-difference Jacobian (it only changes the Newton path), and the
    # history should record that the analytic Jacobian was used.
    period = 1e-3
    t = np.linspace(0.0, period, 201)
    topo = _rc_lowpass_topology()
    kw = dict(topo=topo, tgrid=t, inputs={"vin": np.zeros_like(t)},
              node_inputs={"VIN": "vin"}, V0=np.array([10.0]),
              residual_tol=1e-10, max_shooting_iters=12)
    ana = pss_solve({}, {"VIN": 0.0}, period, analytic_jacobian=True, **kw)
    fd = pss_solve({}, {"VIN": 0.0}, period, analytic_jacobian=False, **kw)
    assert ana["converged"] and fd["converged"]
    np.testing.assert_allclose(ana["x0"], fd["x0"], atol=1e-8)
    assert any(h.get("jacobian") == "analytic_monodromy"
               for h in ana["shooting_history"])


def test_generic_analytic_pac_matches_rc_transfer():
    # The analytic-adjoint kernel reduces to the exact RC transfer on an LTI orbit,
    # with no per-frequency transient runs (O(1) linear solve each).
    R = 1e5
    C = 1e-9
    period = 1e-3
    t = np.linspace(0.0, period, 101)
    topo = _rc_lowpass_topology(R, C)
    pss = pss_solve(
        {}, {"VIN": 0.0}, period, topo=topo, tgrid=t,
        inputs={"vin": np.zeros_like(t)}, node_inputs={"VIN": "vin"},
        V0=np.array([0.0]), residual_tol=1e-12, max_shooting_iters=2,
    )
    freqs = np.array([100.0, 500.0, 1000.0])
    pac = pac_solve(
        {}, {"VIN": 0.0}, freqs, pss_result=pss, input_drive={"vin": 1.0},
        lti_fast_path=False, analytic=True, max_sideband=8, n_period_samples=40,
    )
    expected = 1.0 / (1.0 + 2j * np.pi * freqs * R * C)
    assert pac["method"] == "pss_analytic_adjoint"
    assert pac["pac_period_runs"] == 0
    assert pac["pac_condition_computed"] is False
    if pac_linearize_orbit_numba is not None:
        assert pac["pac_numba_linearization_used"] is True
    if pac_hb_blocks_numba is not None:
        assert pac["pac_numba_hb_used"] is True
    np.testing.assert_allclose(pac["response"], expected, rtol=1e-6)

    with_condition = pac_solve(
        {}, {"VIN": 0.0}, freqs, pss_result=pss, input_drive={"vin": 1.0},
        lti_fast_path=False, analytic=True, max_sideband=8, n_period_samples=40,
        compute_condition=True,
    )
    assert with_condition["pac_condition_computed"] is True
    np.testing.assert_allclose(with_condition["response"], pac["response"],
                               rtol=0, atol=0)

    profiled = pac_solve(
        {}, {"VIN": 0.0}, freqs, pss_result=pss, input_drive={"vin": 1.0},
        lti_fast_path=False, analytic=True, max_sideband=8, n_period_samples=40,
        profile=True,
    )
    assert profiled["pac_condition_computed"] is True


def test_generic_pac_reuses_pss_attached_linearization_cache():
    period = 1e-3
    t = np.linspace(0.0, period, 101)
    topo = _rc_lowpass_topology()
    pss = pss_solve(
        {}, {"VIN": 0.0}, period, topo=topo, tgrid=t,
        inputs={"vin": np.zeros_like(t)}, node_inputs={"VIN": "vin"},
        V0=np.array([0.0]), residual_tol=1e-12, max_shooting_iters=2,
    )

    freqs = np.array([100.0, 500.0])
    first = pac_solve({}, {"VIN": 0.0}, freqs, pss_result=pss,
                      input_drive={"vin": 1.0}, lti_fast_path=False, analytic=False)
    second = pac_solve({}, {"VIN": 0.0}, freqs, pss_result=pss,
                       input_drive={"vin": 1.0}, lti_fast_path=False, analytic=False)
    overlap = pac_solve({}, {"VIN": 0.0}, np.array([500.0, 1000.0]),
                        pss_result=pss, input_drive={"vin": 1.0},
                        lti_fast_path=False, analytic=False)

    assert first["pac_period_runs"] == 1 + 2 * len(freqs)
    assert first["pac_state_cache_hit"] is False
    assert second["pac_period_runs"] == 0
    assert second["pac_state_cache_hit"] is True
    assert second["pac_input_cache_hits"] == len(freqs)
    assert overlap["pac_state_cache_hit"] is True
    assert overlap["pac_input_cache_hits"] == 1
    assert overlap["pac_input_period_runs"] == 2
    np.testing.assert_allclose(second["response"], first["response"], rtol=0, atol=0)


def test_generic_pnoise_includes_resistor_thermal_noise():
    R = 1e5
    C = 1e-9
    period = 1e-3
    t = np.linspace(0.0, period, 101)
    topo = _rc_lowpass_topology(R, C)
    pss = pss_solve(
        {}, {"VIN": 0.0}, period, topo=topo, tgrid=t,
        inputs={"vin": np.zeros_like(t)}, node_inputs={"VIN": "vin"},
        V0=np.array([0.0]), residual_tol=1e-12, max_shooting_iters=2,
    )

    freqs = np.array([10.0, 100.0, 1000.0])
    pnoise = pnoise_solve(
        {}, {"VIN": 0.0}, freqs, pss_result=pss, max_sideband=0,
        n_period_samples=32, gains=np.ones_like(freqs),
    )

    z = 1.0 / (1.0 / R + 2j * np.pi * freqs * C)
    expected = np.abs(z) ** 2 * (4.0 * _KB * _TEMP / R)
    np.testing.assert_allclose(pnoise["out_psd"], expected, rtol=1e-5)
    assert pnoise["method"] == "lti_noise_fast_path"
    assert pnoise["pnoise_hb_solve_count"] == 0


def test_generic_pnoise_reuses_hb_and_adjoint_cache():
    R = 1e5
    C = 1e-9
    period = 1e-3
    t = np.linspace(0.0, period, 101)
    topo = _rc_lowpass_topology(R, C)
    pss = pss_solve(
        {}, {"VIN": 0.0}, period, topo=topo, tgrid=t,
        inputs={"vin": np.zeros_like(t)}, node_inputs={"VIN": "vin"},
        V0=np.array([0.0]), residual_tol=1e-12, max_shooting_iters=2,
    )

    freqs = np.array([10.0, 100.0, 1000.0])
    first = pnoise_solve(
        {}, {"VIN": 0.0}, freqs, pss_result=pss, max_sideband=1,
        n_period_samples=32, gains=np.ones_like(freqs), lti_fast_path=False,
    )
    second = pnoise_solve(
        {}, {"VIN": 0.0}, freqs, pss_result=pss, max_sideband=1,
        n_period_samples=32, gains=np.ones_like(freqs), lti_fast_path=False,
    )

    assert first["method"] == "pss_harmonic_balance_conversion_matrix"
    assert first["pnoise_linearization_cache_hit"] is False
    assert first["pnoise_hb_cache_hit"] is False
    assert first["pnoise_hb_solve_count"] == len(freqs)
    assert second["pnoise_linearization_cache_hit"] is True
    assert second["pnoise_hb_cache_hit"] is True
    assert second["pnoise_adjoint_cache_hits"] == len(freqs)
    assert second["pnoise_hb_solve_count"] == 0
    np.testing.assert_allclose(second["out_psd"], first["out_psd"], rtol=0, atol=0)


def test_generic_pnoise_sparse_and_iterative_solvers_match_dense():
    R = 1e5
    C = 1e-9
    period = 1e-3
    t = np.linspace(0.0, period, 101)
    topo = _rc_lowpass_topology(R, C)
    pss = pss_solve(
        {}, {"VIN": 0.0}, period, topo=topo, tgrid=t,
        inputs={"vin": np.zeros_like(t)}, node_inputs={"VIN": "vin"},
        V0=np.array([0.0]), residual_tol=1e-12, max_shooting_iters=2,
    )

    freqs = np.array([10.0, 100.0, 1000.0])
    common = dict(
        max_sideband=2, n_period_samples=32, gains=np.ones_like(freqs),
        lti_fast_path=False, cache_linearization=False,
    )
    dense = pnoise_solve(
        {}, {"VIN": 0.0}, freqs, pss_result=pss,
        hb_solver="dense", **common)
    sparse = pnoise_solve(
        {}, {"VIN": 0.0}, freqs, pss_result=pss,
        hb_solver="sparse", **common)
    iterative = pnoise_solve(
        {}, {"VIN": 0.0}, freqs, pss_result=pss,
        hb_solver="iterative", iterative_tol=1e-12, **common)

    assert dense["pnoise_hb_solver"] == "dense"
    assert sparse["pnoise_hb_solver"] == "sparse"
    assert iterative["pnoise_hb_solver"] == "iterative"
    assert sparse["pnoise_hb_sparse_density"] < 1.0
    assert iterative["pnoise_hb_preconditioner"] == "block_jacobi"
    assert iterative["pnoise_hb_block_preconditioner_count"] == len(freqs)
    assert iterative["pnoise_hb_iterative_fallbacks"] == 0
    assert max(iterative["pnoise_hb_iterative_iterations"]) <= 2
    np.testing.assert_allclose(sparse["out_psd"], dense["out_psd"],
                               rtol=1e-8, atol=1e-30)
    np.testing.assert_allclose(iterative["out_psd"], dense["out_psd"],
                               rtol=1e-8, atol=1e-30)


def test_gear2_is_second_order_on_rc_lowpass():
    # BDF2/gear2 transient must converge ~2nd order (error ~h^2) on a linear RC
    # low-pass, vs backward-Euler's 1st order. This guards the gear2 integration
    # path used to close the chopper PAC switch-edge error.
    from core.transient_solver import transient
    R, C = 1e6, 1e-9                       # RC = 1 ms
    topo = Topology(solved=["OUT"], devices=[], rails={"VIN": "VIN", "GND": 0.0},
                    outputs=("OUT",), resistors=[("R1", "VIN", "OUT", R)],
                    capacitors=[("C1", "OUT", "GND", C)])
    f = 100.0
    w = 2 * np.pi * f
    RC = R * C
    Hmag = 1.0 / np.sqrt(1 + (w * RC) ** 2)
    phi = -np.arctan(w * RC)

    def max_err(method, ppp):
        t = np.linspace(0.0, 6.0 / f, 6 * ppp + 1)
        vin = np.sin(w * t)
        out = transient({}, {"VIN": 0.0}, t, topo=topo, inputs={"vin": vin},
                        node_inputs={"VIN": "vin"}, V0=np.array([0.0]),
                        integration_method=method)["nodes"]["OUT"]
        mask = t >= t[-1] - 1.0 / f
        ana = Hmag * np.sin(w * t[mask] + phi)
        return float(np.max(np.abs(out[mask] - ana)))

    be_coarse, be_fine = max_err("be", 40), max_err("be", 80)
    g2_coarse, g2_fine = max_err("gear2", 40), max_err("gear2", 80)
    # backward-Euler ~1st order (error halves), gear2 ~2nd order (error quarters)
    assert 1.7 < be_coarse / be_fine < 2.3
    assert 3.3 < g2_coarse / g2_fine < 4.6
    # gear2 is far more accurate at the same step
    assert g2_fine < be_fine / 5.0


def test_adaptive_gear2_rc_lowpass_uses_nonuniform_grid():
    R = 1e5
    C = 1e-9
    f = 100.0
    w = 2 * np.pi * f
    period = 1.0 / f
    t_stop = 6.0 * period
    topo = _rc_lowpass_topology(R, C)
    t = np.linspace(0.0, t_stop, 1201)
    vin = np.sin(w * t)
    tr = transient(
        {}, {"VIN": 0.0}, t, topo=topo, inputs={"vin": vin},
        node_inputs={"VIN": "vin"}, V0=np.array([0.0]),
        integration_method="gear2", adaptive=True,
        adaptive_config=AdaptiveConfig(reltol=1e-4, vabstol=1e-6,
                                       h0=period / 20),
        max_step=period / 5)

    tt = tr["t"]
    assert tr["nfail"] == 0
    assert tr["adaptive"] is True
    assert len(tt) < 500
    assert np.all(np.diff(tt) > 0.0)
    assert np.std(np.diff(tt)) > 0.0
    assert tt[-1] == pytest.approx(t_stop)
    assert len(tr["nodes"]["OUT"]) == len(tt)
    assert len(tr["inputs"]["vin"]) == len(tt)

    RC = R * C
    hmag = 1.0 / np.sqrt(1.0 + (w * RC) ** 2)
    phi = -np.arctan(w * RC)
    mask = tt >= t_stop - period
    expected = hmag * np.sin(w * tt[mask] + phi)
    np.testing.assert_allclose(tr["nodes"]["OUT"][mask], expected, atol=3e-4)


def test_adaptive_requires_gear2():
    topo = _rc_lowpass_topology()
    t = np.linspace(0.0, 1e-3, 11)
    with pytest.raises(ValueError, match="adaptive transient requires"):
        transient({}, {"VIN": 0.0}, t, topo=topo, inputs={"vin": np.zeros_like(t)},
                  node_inputs={"VIN": "vin"}, integration_method="be", adaptive=True)


def test_transient_rejects_removed_cap_modes():
    topo = _rc_lowpass_topology()
    t = np.linspace(0.0, 1e-3, 11)
    inputs = {"vin": np.zeros_like(t)}
    common = dict(topo=topo, inputs=inputs, node_inputs={"VIN": "vin"},
                  V0=np.array([0.0]))
    for mode in ("endpoint", "veriloga", "branch", "self", "self-charge"):
        with pytest.raises(ValueError, match="unknown cap_mode"):
            transient({}, {"VIN": 0.0}, t, cap_mode=mode, **common)
    for mode_id in (2, 3):
        with pytest.raises(ValueError, match="cap_mode_id must be 0 .* or 1"):
            transient({}, {"VIN": 0.0}, t, cap_mode_id=mode_id, **common)


def test_adaptive_step_policy_matches_numba_helpers():
    v_half = np.array([1.0, -2.0, 3.0])
    v_full = np.array([1.0002, -1.9997, 2.9995])
    err_py = adaptive_lte_wrms(v_half, v_full, 2, 1e-4, 1e-6, 1e-12)
    err_nb = nk._adaptive_error_impl(v_half, v_full, 2, 1e-4, 1e-6, 1e-12)
    assert err_py == pytest.approx(err_nb)
    for err in (0.0, 1e-12, 0.2, 1.0, 7.0, np.inf):
        assert adaptive_next_h(1e-6, err) == pytest.approx(
            nk._adaptive_next_h_impl(1e-6, err))
    assert adaptive_next_h(1e-6, 0.0) > 1e-6
    assert adaptive_next_h(1e-6, np.inf) < 1e-6


def test_adaptive_pss_inputs_match_orbit_grid():
    period = 1e-3
    t = np.linspace(0.0, period, 101)
    topo = _rc_lowpass_topology()
    pss = pss_solve(
        {}, {"VIN": 0.0}, period, topo=topo, tgrid=t,
        inputs={"vin": np.zeros_like(t)}, node_inputs={"VIN": "vin"},
        V0=np.array([0.0]), residual_tol=1e-10, max_shooting_iters=2,
        integration_method="gear2", adaptive=True, adaptive_reltol=1e-4)
    assert pss["converged"]
    assert pss["adaptive"] is True
    assert len(pss["inputs"]["vin"]) == len(pss["t"])
    pac = pac_solve({}, {"VIN": 0.0}, np.array([100.0]), pss_result=pss,
                    input_drive={"vin": 1.0}, lti_fast_path=False,
                    analytic=True, max_sideband=1, n_period_samples=16)
    assert np.isfinite(pac["gains"][0])


@pytest.mark.skipif(transient_solve_adaptive_gear2_numba is None,
                    reason="numba adaptive gear2 kernel unavailable")
def test_numba_adaptive_gear2_matches_python(monkeypatch):
    import core.transient_solver as ts

    R = 1e5
    C = 1e-9
    f = 100.0
    w = 2 * np.pi * f
    period = 1.0 / f
    t_stop = 2.0 * period
    topo = _rc_lowpass_topology(R, C)
    t = np.linspace(0.0, t_stop, 401)
    vin = np.sin(w * t)
    kw = dict(
        topo=topo, inputs={"vin": vin}, node_inputs={"VIN": "vin"},
        V0=np.array([0.0]), integration_method="gear2", adaptive=True,
        adaptive_reltol=1e-4, adaptive_vabstol=1e-6,
        adaptive_h0=period / 20, max_step=period / 5,
    )
    nb = transient({}, {"VIN": 0.0}, t, **kw)
    assert nb["numba_adaptive_solver"] is True

    monkeypatch.setattr(ts, "transient_solve_adaptive_gear2_numba", None)
    py = transient({}, {"VIN": 0.0}, t, **kw)
    assert py["numba_adaptive_solver"] is False
    np.testing.assert_allclose(nb["t"], py["t"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(nb["nodes"]["OUT"], py["nodes"]["OUT"], rtol=0, atol=1e-8)


@pytest.mark.skipif(transient_solve_adaptive_gear2_numba is None,
                    reason="numba adaptive gear2 kernel unavailable")
def test_numba_adaptive_gear2_matches_python_at_input_kinks(monkeypatch):
    import core.transient_solver as ts

    topo = _rc_lowpass_topology(1e5, 1e-9)
    t = np.array([0.0, 1e-3, 2e-3, 4e-3])
    vin = np.array([0.0, 1.0, 0.0, 0.0])
    kw = dict(
        topo=topo, inputs={"vin": vin}, node_inputs={"VIN": "vin"},
        V0=np.array([0.0]), integration_method="gear2", adaptive=True,
        adaptive_config=AdaptiveConfig(reltol=1e-4, vabstol=1e-6, h0=0.8e-3),
        max_step=2e-3,
    )
    nb = transient({}, {"VIN": 0.0}, t, **kw)
    assert nb["numba_adaptive_solver"] is True

    monkeypatch.setattr(ts, "transient_solve_adaptive_gear2_numba", None)
    py = transient({}, {"VIN": 0.0}, t, **kw)
    assert py["numba_adaptive_solver"] is False
    np.testing.assert_allclose(nb["t"], py["t"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(nb["nodes"]["OUT"], py["nodes"]["OUT"], rtol=0, atol=1e-8)


def test_reverse_biased_pass_switch_restores_not_pumps():
    # A pass-gate switch whose drain is driven ABOVE its source must DISCHARGE the
    # drain back toward the source. The signed Verilog-A drain current does this;
    # the old abs(Idc) flipped a reverse-biased switch into an anti-restoring pump
    # (the SC-LPF runaway: VMID ran 20 -> 333 V). Start the cap node above the
    # source and require it to relax back, never run away.
    from core.transient_solver import transient
    topo = Topology(
        solved=["MID"],
        devices=[("M1", "MID", "VG", "VIN")],          # (name, drain, gate, source)
        rails={"VIN": 20.0, "VG": 0.0, "GND": 0.0},    # source 20, gate 0 -> PMOS on
        capacitors=[("C1", "MID", "GND", 1e-9)],
        outputs=("MID",),
    )
    t = np.linspace(0.0, 5e-3, 2001)
    tr = transient({"M1": (5000.0, 30.0)}, {}, t, topo=topo,
                   V0=np.array([25.0]), integration_method="be")
    mid = tr["nodes"]["MID"]
    assert mid.max() < 25.6, f"reverse switch pumped MID up to {mid.max():.2f}"
    assert abs(mid[-1] - 20.0) < 1.0, f"MID did not restore to source (got {mid[-1]:.2f})"


def test_pss_reports_stiffness_and_honest_status():
    # The solver now reports a Floquet-multiplier stiffness diagnostic and an
    # honest status, and never flags an out-of-bounds orbit as converged.
    period = 1e-3
    t = np.linspace(0.0, period, 201)
    topo = _rc_lowpass_topology()
    pss = pss_solve({}, {"VIN": 5.0}, period, topo=topo, tgrid=t,
                    inputs={"vin": np.full_like(t, 5.0)}, node_inputs={"VIN": "vin"},
                    V0=np.array([0.0]), residual_tol=1e-9, max_shooting_iters=12)
    assert pss["converged"] and not pss["diverged"]
    assert pss["pss_status"] in ("converged_shooting", "converged_stabilization")
    # Stable RC (tau = RC = 0.1*period): dominant multiplier well inside the unit circle.
    assert 0.0 <= pss["dominant_multiplier"] < 1.0
