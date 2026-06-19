import numpy as np

from core.pac_solver import pac_solve
from core.pnoise_solver import pnoise_solve
from core.pss_solver import pss_solve
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
                      input_drive={"vin": 1.0}, lti_fast_path=False)
    second = pac_solve({}, {"VIN": 0.0}, freqs, pss_result=pss,
                       input_drive={"vin": 1.0}, lti_fast_path=False)
    overlap = pac_solve({}, {"VIN": 0.0}, np.array([500.0, 1000.0]),
                        pss_result=pss, input_drive={"vin": 1.0},
                        lti_fast_path=False)

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
