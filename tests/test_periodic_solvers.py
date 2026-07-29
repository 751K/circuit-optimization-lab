import numpy as np
import pytest

from circuitopt.adaptive_config import AdaptiveConfig, adaptive_lte_wrms, adaptive_next_h
from circuitopt.pac_solver import pac_solve
from circuitopt._engine import current_engine
from circuitopt.pnoise_solver import (
    _fold_terminal_noise_source,
    _psd_matrix_sqrt,
    pnoise_solve,
)
from circuitopt.pss_solver import pss_solve
from circuitopt.transient_solver import transient
from circuitopt.topology import Topology


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
    # v2.0.0: rust is the only engine, so the rust linearization/HB path always runs.
    assert current_engine() == "rust"
    assert pac["pac_rust_linearization_used"] is True
    assert pac["pac_rust_hb_used"] is True
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


def test_correlated_terminal_noise_fold_matches_stationary_quadratic_form():
    white = np.array([
        [4.0, 1.0 - 0.5j, -2.0, 0.0],
        [1.0 + 0.5j, 3.0, -0.5, 0.0],
        [-2.0, -0.5, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]) * 1e-24
    flicker = np.array([
        [2.0, 0.4, -1.0, 0.0],
        [0.4, 1.5, -0.2, 0.0],
        [-1.0, -0.2, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]) * 1e-20
    frequency = 1e3
    z = np.array([2.0 + 0.5j, -1.0j, 0.25 - 0.5j, 0.0])
    adj = z[:3].copy()
    terminal_indices = (
        np.array([0]),
        np.array([1]),
        np.array([2]),
        None,
    )
    actual = _fold_terminal_noise_source(
        adj,
        terminal_indices,
        white[None, None, :, :],
        _psd_matrix_sqrt(flicker)[None, :, :],
        frequency,
        np.array([0]),
        fundamental=1e6,
        max_sideband=0,
    )
    expected = float(np.real(
        z @ (white + flicker / frequency) @ z.conj()))
    assert actual == pytest.approx(expected, rel=1e-12)


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


def test_generic_pnoise_method_matches_time_domain_flag():
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

    freqs = np.array([10.0, 100.0])
    common = dict(
        max_sideband=1, n_period_samples=32, gains=np.ones_like(freqs),
        lti_fast_path=False, cache_linearization=False,
    )
    td = pnoise_solve({}, {"VIN": 0.0}, freqs, pss_result=pss,
                      time_domain=True, **common)
    hb = pnoise_solve({}, {"VIN": 0.0}, freqs, pss_result=pss,
                      time_domain=False, **common)

    assert td["pnoise_time_domain_used"] is True
    assert td["pnoise_conversion"] == "time_domain"
    assert td["method"] == "pss_time_domain_floquet_adjoint"
    assert hb["pnoise_time_domain_used"] is False
    assert hb["pnoise_conversion"] == "harmonic_balance"
    assert hb["method"] == "pss_harmonic_balance_conversion_matrix"


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


def test_pnoise_reports_sparse_solver_degradation(monkeypatch):
    import circuitopt.pnoise_solver as pns

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
    monkeypatch.setattr(pns, "_sp", None)
    monkeypatch.setattr(pns, "_spla", None)

    with pytest.warns(RuntimeWarning, match="falling back to dense"):
        pn = pnoise_solve(
            {}, {"VIN": 0.0}, np.array([100.0]), pss_result=pss,
            max_sideband=1, n_period_samples=32, gains=np.ones(1),
            lti_fast_path=False, cache_linearization=False, hb_solver="sparse",
        )
    assert pn["pnoise_hb_solver"] == "dense"
    assert pn["pnoise_degraded"] is True
    assert any(w["code"] == "hb_sparse_unavailable" for w in pn["pnoise_warnings"])


def test_pnoise_rejects_device_noise_degradation(monkeypatch):
    from circuitopt.pmos_tft_model import PMOS_TFT

    period = 1e-3
    t = np.linspace(0.0, period, 33)
    topo = Topology(
        solved=["OUT"],
        devices=[("M1", "OUT", "VG", "VDD")],
        rails={"VDD": 40.0, "VG": 0.0, "GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "OUT", "GND", 1e6)],
        model_types={"M1": "at4000tg.pmos"},
        device_kwargs={"M1": {"section": "inherit", "bin": "auto"}},
    )
    pss = {
        "topology": topo,
        "t": t,
        "period": period,
        "nodes": {"OUT": np.full_like(t, 20.0)},
        "inputs": {},
        "bias": {},
    }

    def fail_noise(self, Vs, Vd, Vg, frequency):
        raise NotImplementedError("noise hook missing")

    monkeypatch.setattr(PMOS_TFT, "get_noise_psd", fail_noise)
    from circuitopt.run_contract import ModelEvaluationError

    with pytest.raises(ModelEvaluationError, match="periodic noise-PSD"):
        pnoise_solve(
            {"M1": (5000.0, 30.0)}, {}, np.array([100.0]), pss_result=pss,
            max_sideband=0, n_period_samples=16, gains=np.ones(1),
            lti_fast_path=False, cache_linearization=False,
        )


def test_gear2_is_second_order_on_rc_lowpass():
    # BDF2/gear2 transient must converge ~2nd order (error ~h^2) on a linear RC
    # low-pass, vs backward-Euler's 1st order. This guards the gear2 integration
    # path used to close the chopper PAC switch-edge error.
    from circuitopt.transient_solver import transient
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


def test_adaptive_step_policy_sanity():
    # The numba `_impl` twins of these helpers were removed in v2.0.0;
    # circuitopt.adaptive_config is the single source. Lock the formula shape.
    v_half = np.array([1.0, -2.0, 3.0])
    v_full = np.array([1.0002, -1.9997, 2.9995])
    err = adaptive_lte_wrms(v_half, v_full, 2, 1e-4, 1e-6, 1e-12)
    assert np.isfinite(err) and err > 0.0
    assert adaptive_next_h(1e-6, 0.0) > 1e-6          # tiny error -> grow (capped)
    assert adaptive_next_h(1e-6, np.inf) < 1e-6       # huge error -> shrink
    assert adaptive_next_h(1e-6, 7.0) < adaptive_next_h(1e-6, 0.2)  # monotone


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reltol": np.nan},
        {"vabstol": np.inf},
        {"iabstol": np.nan},
        {"h0": np.inf},
        {"freeze_factor": np.nan},
    ],
)
def test_adaptive_config_rejects_nonfinite_values(kwargs):
    with pytest.raises(ValueError, match="finite"):
        AdaptiveConfig(**kwargs)


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


def test_adaptive_pss_warmup_returns_deterministic_final_grid():
    period = 1e-3
    warm = np.array([0.0, 0.2e-3, 0.5e-3, 0.8e-3, period])
    final = np.unique(np.concatenate([
        np.linspace(0.0, period, 257),
        np.array([0.2e-3, 0.5e-3, 0.8e-3]),
    ]))
    topo = _rc_lowpass_topology()
    pss = pss_solve(
        {}, {"VIN": 0.0}, period, topo=topo, tgrid=warm,
        final_tgrid=final, inputs={"vin": np.zeros_like(warm)},
        node_inputs={"VIN": "vin"}, V0=np.array([1.0]),
        tstab_periods=2, residual_tol=1e-8, max_shooting_iters=4,
        integration_method="gear2", adaptive=True,
        adaptive_reltol=1e-4, adaptive_h0=period / 1000)

    assert pss["converged"]
    assert pss["pss_adaptive_requested"] is True
    assert pss["pss_adaptive_warmup"] is True
    assert pss["pss_adaptive_warmup_grid_frozen"] is False
    assert pss["pss_warmup_period_runs"] > 0
    assert pss["pss_final_period_runs"] > 0
    assert pss["pss_final_grid_used"] is True
    assert pss["pss_orbit_grid"] == "deterministic_final"
    assert pss["adaptive"] is False
    assert pss["adaptive_grid_frozen"] is False
    np.testing.assert_array_equal(pss["t"], final)
    assert len(pss["inputs"]["vin"]) == len(final)

    pac = pac_solve(
        {}, {"VIN": 0.0}, np.array([100.0]), pss_result=pss,
        input_drive={"vin": 1.0}, lti_fast_path=False,
        analytic=True, max_sideband=1, n_period_samples=16)
    assert pac["pss"] is pss
    assert np.isfinite(pac["gains"][0])
    pnoise = pnoise_solve(
        {}, {"VIN": 0.0}, np.array([100.0]), pss_result=pss,
        fundamental=1.0 / period, input_drive={"vin": 1.0},
        lti_fast_path=False, max_sideband=1, n_period_samples=16)
    assert pnoise["pss"] is pss
    assert np.isfinite(pnoise["out_psd"][0])


def test_adaptive_pss_final_grid_requires_warmup():
    period = 1e-3
    grid = np.linspace(0.0, period, 11)
    topo = _rc_lowpass_topology()
    common = dict(
        topo=topo, tgrid=grid, final_tgrid=grid,
        inputs={"vin": np.zeros_like(grid)}, node_inputs={"VIN": "vin"},
        V0=np.array([0.0]), integration_method="gear2")
    with pytest.raises(ValueError, match="requires adaptive=True"):
        pss_solve({}, {"VIN": 0.0}, period, adaptive=False, **common)
    with pytest.raises(ValueError, match="tstab_periods >= 1"):
        pss_solve({}, {"VIN": 0.0}, period, adaptive=True,
                  tstab_periods=0, **common)


def test_reverse_biased_pass_switch_restores_not_pumps():
    # A pass-gate switch whose drain is driven ABOVE its source must DISCHARGE the
    # drain back toward the source. The signed Verilog-A drain current does this;
    # the old abs(Idc) flipped a reverse-biased switch into an anti-restoring pump
    # (the SC-LPF runaway: VMID ran 20 -> 333 V). Start the cap node above the
    # source and require it to relax back, never run away.
    from circuitopt.transient_solver import transient
    topo = Topology(
        solved=["MID"],
        devices=[("M1", "MID", "VG", "VIN")],          # (name, drain, gate, source)
        rails={"VIN": 20.0, "VG": 0.0, "GND": 0.0},    # source 20, gate 0 -> PMOS on
        capacitors=[("C1", "MID", "GND", 1e-9)],
        outputs=("MID",),
        model_types={"M1": "at4000tg.pmos"},
        device_kwargs={"M1": {"section": "inherit", "bin": "auto"}},
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


def test_adaptive_transient_stays_inside_the_requested_window():
    # A transient with a periodic block asked for its own window. Adding the
    # waveform breakpoints used to close the whole period, so a 5 ns run of a
    # 10 ns period solved twice as far and handed back an accepted grid
    # reaching past tstop. A PSS orbit still needs the period closed.
    import numpy as np

    from circuitopt.analysis_dispatch import _with_adaptive_waveform_breakpoints

    period = 1e-8
    periodic = {"inputs": {"clk": {"type": "square", "delay": 2e-11, "duty": 0.5}}}
    requested = np.linspace(0.0, 5e-9, 501)

    windowed = _with_adaptive_waveform_breakpoints(
        periodic, requested, period, close_period=False)
    assert windowed[0] == 0.0
    assert windowed[-1] <= requested[-1]
    assert np.all(np.diff(windowed) >= 0.0)
    # Breakpoints inside the window are still added.
    assert len(windowed) >= len(requested)

    closed = _with_adaptive_waveform_breakpoints(periodic, requested, period)
    assert closed[-1] == pytest.approx(period)


def test_adaptive_breakpoints_preserve_nonzero_absolute_time_window():
    import numpy as np

    from circuitopt.analysis_dispatch import _with_adaptive_waveform_breakpoints

    period = 10e-9
    periodic = {
        "inputs": {
            "clk": {
                "type": "square",
                "delay": 3e-9,
                "duty": 0.2,
            }
        }
    }
    requested = np.linspace(12e-9, 16e-9, 17)

    merged = _with_adaptive_waveform_breakpoints(
        periodic,
        requested,
        period,
        close_period=False,
    )

    assert merged[0] == requested[0]
    assert merged[-1] == requested[-1]
    assert np.all(np.diff(merged) > 0.0)
    # The 3 ns and 5 ns phase edges repeat at 13 ns and 15 ns in this window.
    assert np.any(np.isclose(merged, 13e-9, rtol=0.0, atol=1e-18))
    assert np.any(np.isclose(merged, 15e-9, rtol=0.0, atol=1e-18))


def test_adaptive_breakpoints_merge_near_duplicate_edge_times():
    # The MDAC deck's clock edge literal 2e-11 lands 3.2e-27 s away from the
    # 501-point grid's own sample there. Keeping both left a sub-ULP interval
    # in the merged grid: harmless to the solver's startup step (filtered in
    # the core), but a zero-rise edge sampled across such a pair swings full
    # scale over the gap and poisons the slope scale that selects critical
    # times. The merger now collapses each near-duplicate cluster onto its
    # earliest member.
    import numpy as np

    from circuitopt.analysis_dispatch import _with_adaptive_waveform_breakpoints

    period = 1e-8
    periodic = {"inputs": {"clk": {"type": "square", "delay": 2e-11, "duty": 0.5}}}
    requested = np.linspace(0.0, 5e-9, 501)
    assert requested[2] != 2e-11  # the near-duplicate pair this test is about

    merged = _with_adaptive_waveform_breakpoints(
        periodic, requested, period, close_period=False)

    gaps = np.diff(merged)
    assert gaps.min() > 1e-18, f"degenerate interval survived: {gaps.min():.3e}"
    assert np.any(merged == 2e-11)
    assert merged[0] == 0.0
    assert merged[-1] == 5e-9


def _sky130_chopper_pss(n_points=41):
    """A real BSIM4 PSS orbit on a small grid, or skip if SKY130 is absent."""
    pytest.importorskip("circuitopt_core")
    from pathlib import Path

    from circuitopt.pdk.sky130.library import _BUNDLED_CARD_DIR

    if not _BUNDLED_CARD_DIR.exists() or not any(_BUNDLED_CARD_DIR.iterdir()):
        pytest.skip("SKY130 bundled cards not present")

    from circuitopt import analysis_dispatch as ad
    from circuitopt.circuit_loader import load_circuit_json

    deck = Path(__file__).resolve().parents[1] / "examples" / "sky130_chopper.json"
    spec = load_circuit_json(deck)
    cfg = {k: (dict(v) if isinstance(v, dict) else v)
           for k, v in (spec.analyses or {}).items()}
    pss_cfg, periodic = ad._pss_config(spec, cfg, cfg.get("pss", {}))
    pss_cfg["n_points"] = n_points
    return spec, ad._run_pss(spec, spec.binding(), pss_cfg, periodic)


def test_pnoise_solves_the_operating_point_only_for_gated_devices():
    """The periodic-noise loop must not evaluate the small-signal operating
    point for devices it does not conductance-gate.

    ``gds_noise_devices`` is the only consumer of that solve: the terminal-noise
    and noise-PSD branches read the compact model directly. Computing it for
    every device would be a second full BSIM4 solve per (orbit sample, device)
    whose result is then discarded -- on the chopper deck that was 9984 wasted
    evaluations, 17% of the whole analysis.
    """
    spec, pss = _sky130_chopper_pss()
    names = [name for name, *_ in pss["topology"].devices]
    freqs = np.array([1e3, 1e4])

    calls = []

    from circuitopt.pdk.sky130 import device as sky130_device

    original = sky130_device._Sky130NativeFet.get_ss_params

    def counted(self, Vs, Vd, Vg):
        calls.append((Vs, Vd, Vg))
        return original(self, Vs, Vd, Vg)

    def run(**kwargs):
        calls.clear()
        sky130_device._Sky130NativeFet.get_ss_params = counted
        try:
            return pnoise_solve(
                spec.sizes, spec.bias, freqs, pss_result=pss,
                binding=spec.binding(), fundamental=250e3,
                max_sideband=1, n_period_samples=8, cache_linearization=False,
                input_drive={"vinp": 0.5, "vinn": -0.5},
                **kwargs,
            ), len(calls)
        finally:
            sky130_device._Sky130NativeFet.get_ss_params = original

    ungated, ungated_calls = run()
    assert ungated_calls == 0, (
        f"{ungated_calls} operating-point solves for devices nothing gates")

    gated, gated_calls = run(gds_noise_devices=[names[0]])
    # 8 orbit samples for the one gated device -- the gate is still wired up.
    assert gated_calls == 8, gated_calls
    # Gating replaces that device's terminal noise with 4kT*gds, so the two
    # runs must not agree; a vacuous gate would make this test pass trivially.
    assert not np.allclose(
        gated["out_psd"], ungated["out_psd"], rtol=1e-9, atol=0.0)


def _pnoise_on(spec, pss, samples=8, **kwargs):
    return pnoise_solve(
        spec.sizes, spec.bias, np.array([1e3, 1e4]), pss_result=pss,
        binding=spec.binding(), fundamental=250e3, max_sideband=1,
        n_period_samples=samples, cache_linearization=False,
        input_drive={"vinp": 0.5, "vinn": -0.5}, **kwargs,
    )


def _device_batch_width():
    from circuitopt._device_batch import ORBIT_BATCH_SAMPLES

    return int(ORBIT_BATCH_SAMPLES)


def _count_scalar_noise_evaluations(monkeypatch):
    """Count scalar compact-model noise evaluations, live, into a list."""
    from circuitopt.pdk.sky130 import device as sky130_device

    seen = []
    original = sky130_device._Sky130NativeFet.get_terminal_noise

    def counted(self, Vs, Vd, Vg, frequency):
        seen.append(float(frequency))
        return original(self, Vs, Vd, Vg, frequency)

    monkeypatch.setattr(
        sky130_device._Sky130NativeFet, "get_terminal_noise", counted)
    return seen


def test_batched_terminal_noise_matches_the_scalar_adapter(monkeypatch):
    """Reading the orbit through the batch ABI must not change the answer.

    The batched path re-biases one dedicated native handle per device and reads
    both 1/f probe frequencies back in a single call, instead of three scalar
    compact-model solves per (orbit sample, device). BSIM4 device state is
    path-dependent, so "same formula" is not enough -- the two paths must agree
    on the actual orbit.
    """
    spec, pss = _sky130_chopper_pss()
    scalar_calls = _count_scalar_noise_evaluations(monkeypatch)

    batched = _pnoise_on(spec, pss)
    assert scalar_calls == [], (
        f"the batched run made {len(scalar_calls)} scalar noise evaluations")

    import circuitopt.pnoise_solver as pns

    monkeypatch.setattr(pns, "open_orbit_batch", lambda *a, **k: None)
    scalar = _pnoise_on(spec, pss)

    # Not vacuous: the scalar path really does evaluate every device at every
    # sample, at both probe frequencies, and the batched run did none of that.
    assert sorted(set(scalar_calls)) == [1.0, 10.0]
    assert len(scalar_calls) == 2 * 8 * len(pss["topology"].devices)

    np.testing.assert_allclose(
        batched["out_psd"], scalar["out_psd"], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        batched["irn_psd"], scalar["irn_psd"], rtol=1e-12, atol=0.0)


def test_terminal_noise_batch_failure_falls_back_without_partial_grids(monkeypatch):
    """A batch that dies mid-orbit must not leave half-filled noise grids.

    The fallback re-runs the whole set through the scalar adapter; if it kept
    the samples the batch had already written, the analysis would silently
    report noise from a partly zeroed orbit instead of failing or recovering.
    """
    # Several blocks wide, so the failure genuinely lands mid-orbit: the first
    # block is written before the second one dies.
    samples = 4 * _device_batch_width()
    spec, pss = _sky130_chopper_pss()
    reference = _pnoise_on(spec, pss, samples=samples)

    import circuitopt.pnoise_solver as pns
    from circuitopt import _device_batch

    original_noise = _device_batch.NativeOrbitBatch.noise
    state = {"calls": 0}

    def dying_noise(self, frequencies):
        state["calls"] += 1
        if state["calls"] > 1:      # one block lands, then it breaks
            raise RuntimeError("simulated native noise-batch failure")
        return original_noise(self, frequencies)

    monkeypatch.setattr(_device_batch.NativeOrbitBatch, "noise", dying_noise)
    recovered = _pnoise_on(spec, pss, samples=samples)

    assert state["calls"] > 1, "the batch never reached the failure point"
    assert pns.open_orbit_batch is not None
    np.testing.assert_allclose(
        recovered["out_psd"], reference["out_psd"], rtol=1e-12, atol=0.0)


def test_batched_orbit_linearization_matches_the_scalar_adapter(monkeypatch):
    """The batched orbit G/C tensors must equal the scalar adapter's.

    The C batch entry point hands back unreduced kernel output: the scalar
    adapter's bulk-terminal KCL/charge closure lives in Python. A batch that
    skipped it would feed PAC and PNoise conductance and capacitance matrices
    whose columns do not sum to zero.
    """
    spec, pss = _sky130_chopper_pss()
    freqs = np.array([1e3, 1e4])

    def run_pac():
        return pac_solve(
            spec.sizes, spec.bias, freqs, pss_result=pss, binding=spec.binding(),
            input_drive={"vinp": 0.5, "vinn": -0.5}, time_domain=True,
            td_n_period_samples=16, cache_linearization=False,
            cache_forcing=False,
        )

    batched = run_pac()

    import circuitopt.pac_solver as pacs

    calls = []
    original = pacs._fill_dense_linearization_batched
    monkeypatch.setattr(
        pacs, "_fill_dense_linearization_batched",
        lambda *a, **k: (calls.append(1), False)[1])
    scalar = run_pac()

    assert original is not None and calls, "the scalar run never reached the hook"
    np.testing.assert_allclose(
        batched["gains"], scalar["gains"], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        batched["response"], scalar["response"], rtol=1e-12, atol=0.0)


def test_orbit_batch_reduces_terminals_like_the_scalar_adapter():
    """One batched evaluation must equal `get_terminal_linearization` exactly.

    This is the contract the PAC/PNoise orbit tensors rest on, checked against
    the scalar adapter device by device rather than through an analysis.
    """
    spec, pss = _sky130_chopper_pss()

    from circuitopt._device_batch import open_orbit_batch
    from circuitopt.device_factory import build_devices

    topo = pss["topology"]
    dev_inst = build_devices(
        spec.sizes, nf=spec.nf, topo=topo, model_types=spec.model_types,
        device_kwargs=spec.device_kwargs)
    names = [name for name, *_ in topo.devices]
    devices = [dev_inst[name] for name in names]

    batch = open_orbit_batch(devices)
    assert batch is not None, "SKY130 devices must be batchable"

    vs = np.zeros(len(devices))
    vd = np.linspace(0.3, 1.5, len(devices))
    vg = np.linspace(0.2, 1.6, len(devices))
    with batch:
        _i, conductance, _q, capacitance = batch.evaluate(vs, vd, vg)

    for position, dev in enumerate(devices):
        G4, C4 = dev.get_terminal_linearization(
            float(vs[position]), float(vd[position]), float(vg[position]))
        np.testing.assert_allclose(
            conductance[position], np.asarray(G4, float), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            capacitance[position], np.asarray(C4, float), rtol=0.0, atol=0.0)


def test_orbit_batch_closes_the_terminal_residual_at_the_bulk_node(monkeypatch):
    """Every block the batch returns must carry the scalar adapter's reduction.

    The C batch entry point returns raw kernel output; closing BSIM's
    abstol/gmin-scale four-terminal remainder at the reference terminal is the
    Python adapter's job, and PAC stamps the returned conductance and
    capacitance straight into the orbit tensors. On the bundled SKY130 cards
    the kernel's own residual measures exactly zero across a bias sweep, so the
    reduction is injected here rather than waited for.
    """
    from circuitopt import _device_batch
    from circuitopt.compact_models.bsim4 import NativeBsim4Backend

    count = 3
    residual_i = np.array([1e-13, -2e-13, 4e-13])
    residual_q = np.array([3e-22, -1e-22, 5e-22])

    def fake_batch(handles, terminals):
        currents = np.zeros((count, 4))
        currents[:, 0] = 1e-3
        currents[:, 1] = -1e-3
        currents[:, 2] = residual_i           # column sum is the residual
        conductance = np.zeros((count, 4, 4))
        conductance[:, 0, 0] = 1e-4
        conductance[:, 1, 0] = -1e-4
        conductance[:, 2, 1] = residual_i
        charges = np.zeros((count, 4))
        charges[:, 0] = residual_q
        capacitance = np.zeros((count, 4, 4))
        capacitance[:, 0, 2] = residual_q
        return currents, conductance, charges, capacitance

    monkeypatch.setattr(NativeBsim4Backend, "evaluate_batch",
                        staticmethod(fake_batch))

    batch = _device_batch.NativeOrbitBatch([None] * count, np.zeros(count), count)
    currents, conductance, charges, capacitance = batch.evaluate(
        np.zeros(count), np.zeros(count), np.zeros(count))

    # The residual is absorbed at the bulk terminal, index 3, and nowhere else.
    np.testing.assert_allclose(currents[:, 3], -residual_i, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(charges[:, 3], -residual_q, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(currents[:, 0], 1e-3, rtol=0.0, atol=0.0)
    for block in (currents, conductance, charges, capacitance):
        np.testing.assert_allclose(
            block.sum(axis=1), 0.0, rtol=0.0, atol=1e-30)


@pytest.mark.parametrize("width", [1, 3, 8])
def test_orbit_batch_result_is_independent_of_the_block_width(monkeypatch, width):
    """Reshaping a multi-sample block must not scramble sample/device order.

    A block carries ``width`` orbit samples laid out device-fastest; getting the
    stride or the reshape wrong would silently attribute one sample's noise to
    another. Compared against the scalar adapter rather than against another
    width, because the two agree exactly only when the mapping is right.
    """
    from circuitopt import _device_batch

    monkeypatch.setattr(_device_batch, "ORBIT_BATCH_SAMPLES", width)
    spec, pss = _sky130_chopper_pss()
    batched = _pnoise_on(spec, pss, samples=8)

    import circuitopt.pnoise_solver as pns

    monkeypatch.setattr(pns, "open_orbit_batch", lambda *a, **k: None)
    scalar = _pnoise_on(spec, pss, samples=8)

    np.testing.assert_allclose(
        batched["out_psd"], scalar["out_psd"], rtol=1e-12, atol=0.0)


def test_psd_matrix_sqrt_stacks_without_changing_a_single_matrix():
    """The stacked square root must equal the one-at-a-time result exactly.

    `_psd_matrix_sqrt` is applied to a whole orbit of 4x4 flicker matrices in
    one `eigh` call; a transpose that mixes the leading axes into the matrix
    axes would still produce plausible Hermitian output.
    """
    rng = np.random.default_rng(7)
    stack = (rng.normal(size=(5, 3, 4, 4)) + 1j * rng.normal(size=(5, 3, 4, 4)))
    stack = stack + np.conjugate(np.swapaxes(stack, -1, -2))

    stacked = _psd_matrix_sqrt(stack)
    assert stacked.shape == stack.shape
    for i in range(stack.shape[0]):
        for j in range(stack.shape[1]):
            np.testing.assert_allclose(
                stacked[i, j], _psd_matrix_sqrt(stack[i, j]),
                rtol=0.0, atol=0.0)


def test_pac_forcing_solve_is_shared_across_frequencies():
    """Hoisting the forcing solve out of the frequency loop must not change it.

    Every frequency's per-sample right-hand side is now solved in one
    multi-right-hand-side call per orbit sample instead of one call per
    (sample, frequency). Same factorization, same LAPACK routine -- the
    response must be bit-identical to the per-frequency form.
    """
    spec, pss = _sky130_chopper_pss()
    freqs = np.array([1e3, 3e3, 1e4, 3e4])

    def run():
        return pac_solve(
            spec.sizes, spec.bias, freqs, pss_result=pss, binding=spec.binding(),
            input_drive={"vinp": 0.5, "vinn": -0.5}, time_domain=True,
            td_n_period_samples=16, cache_linearization=False, cache_forcing=False,
        )

    batched = run()
    one_at_a_time = np.empty(len(freqs), dtype=complex)
    for index, frequency in enumerate(freqs):
        single = pac_solve(
            spec.sizes, spec.bias, np.array([frequency]), pss_result=pss,
            binding=spec.binding(), input_drive={"vinp": 0.5, "vinn": -0.5},
            time_domain=True, td_n_period_samples=16,
            cache_linearization=False, cache_forcing=False,
        )
        one_at_a_time[index] = single["response"][0]

    np.testing.assert_allclose(
        batched["response"], one_at_a_time, rtol=0.0, atol=0.0)
