"""FreePDK45 native-BSIM4 tests that do not require ngspice."""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from circuitopt.toolchain import pdk_root


_ROOT = os.path.dirname(os.path.dirname(__file__))
_CARD = os.path.join(pdk_root(), "freepdk45", "models_nom", "NMOS_VTG.inc")
_CFG = os.path.join(_ROOT, "examples", "freepdk45_5t_ota.json")
pytestmark = [
    pytest.mark.skipif(
        not os.path.isfile(_CARD), reason="FreePDK45 cards not present"),
    pytest.mark.skipif(
        not any(shutil.which(name) for name in ("clang", "cc", "gcc")),
        reason="native BSIM4 tests require a C99 compiler"),
]


def _spec(*, driven=False):
    from circuitopt.circuit_loader import circuit_from_dict

    with open(_CFG, encoding="utf-8") as handle:
        config = json.load(handle)
    if driven:
        config["transient_inputs"] = {"M1": "vip", "M2": "vin"}
    return circuit_from_dict(config), config


def test_native_devices_load_flat_version_4_cards_without_ngspice(monkeypatch):
    from circuitopt.device_model import create_transistor, get_model_class, list_pdks

    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/ngspice")
    assert "freepdk45" in list_pdks()
    assert get_model_class("freepdk45.nmos").TRANSIENT_BACKEND == "bsim4_native"
    assert get_model_class("freepdk45.pmos").TRANSIENT_BACKEND == "bsim4_native"
    nmos = create_transistor(
        "nmos", pdk="freepdk45", W=0.09, L=0.05, corner="nom")
    pmos = create_transistor(
        "pmos", pdk="freepdk45", W=0.09, L=0.05, corner="nom", vb=1.0)
    assert type(nmos).__name__ == "Fp45Nfet"
    assert type(pmos).__name__ == "Fp45Pfet"
    assert nmos.TRANSIENT_BACKEND == pmos.TRANSIENT_BACKEND == "bsim4_native"
    assert nmos.model_card.version == pmos.model_card.version == 4.0
    assert nmos._evaluate(0.0, 0.5, 0.7).operating_point["internal_nodes"] == 4
    assert pmos._evaluate(1.0, 0.5, 0.3).operating_point["internal_nodes"] == 4


def test_native_card_bundle_cache_reuses_immutable_cards():
    from circuitopt.pdk.freepdk45.library import load_freepdk45_library

    library = load_freepdk45_library("nmos", "nom")
    library.clear_card_cache()
    request = {
        "pdk": "freepdk45",
        "model": "nmos",
        "section": "inherit",
        "bin_selector": "auto",
        "width_um": 1.0,
        "length_um": 0.05,
        "nf": 2,
        "mult": 1,
        "corner": "nom",
        "temperature_c": 27.0,
        "mismatch_v": 0.0,
    }
    first = library.device_cards(**request)
    second = library.device_cards(**request)
    assert all(left is right for left, right in zip(first, second, strict=True))
    info = library.card_cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (
        1, 1, 1024, 1)
    with pytest.raises(TypeError):
        first[0].instance_parameters["nf"] = 4
    with pytest.raises(TypeError):
        first[1].parameters["vth0"] = 0.0
    with pytest.raises(TypeError):
        first[2].parameters["nf"] = 4

    hot = library.device_cards(**(request | {"temperature_c": 125.0}))
    extra = library.device_cards(
        **(request | {"instance_parameters": {"rgeo": 1.0}}))
    assert hot[0] is not first[0]
    assert extra[0] is not first[0]


def test_native_single_devices_are_finite_conservative_and_noisy(monkeypatch):
    from circuitopt.device_model import create_transistor

    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/ngspice")
    devices = (
        (create_transistor("nmos", pdk="freepdk45", W=0.09, L=0.05),
         (0.0, 0.5, 0.7), 1),
        (create_transistor(
            "pmos", pdk="freepdk45", W=0.09, L=0.05, vb=1.0),
         (1.0, 0.5, 0.3), -1),
    )
    for device, bias, drain_sign in devices:
        result = device._evaluate(*bias, frequency_hz=1e6)
        assert np.sign(result.terminal_currents[0]) == drain_sign
        np.testing.assert_allclose(result.terminal_currents.sum(), 0.0, atol=1e-18)
        np.testing.assert_allclose(result.terminal_charges.sum(), 0.0, atol=1e-24)
        assert result.operating_point["gm"] > 0
        assert result.noise is not None
        assert result.noise.spectral_density[0, 0].real > 0


def test_native_batch_abi_matches_individual_evaluation(monkeypatch):
    from circuitopt.compact_models.bsim4 import NativeBsim4Backend
    from circuitopt.device_model import create_transistor

    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/ngspice")
    assert NativeBsim4Backend.abi_version == 2
    nmos = create_transistor(
        "nmos", pdk="freepdk45", W=0.09, L=0.05)
    pmos = create_transistor(
        "pmos", pdk="freepdk45", W=0.09, L=0.05, vb=1.0)
    handles = [
        nmos.create_native_solver_handle(),
        pmos.create_native_solver_handle(),
    ]
    try:
        terminals = np.asarray((
            (0.5, 0.7, 0.0, 0.0),
            (0.5, 0.3, 1.0, 1.0),
        ))
        currents, conductance, charges, capacitance = (
            NativeBsim4Backend.evaluate_batch(handles, terminals))
    finally:
        for handle in handles:
            handle.close()
    expected = (
        nmos._evaluate(0.0, 0.5, 0.7),
        pmos._evaluate(1.0, 0.5, 0.3),
    )
    for index, result in enumerate(expected):
        np.testing.assert_allclose(currents[index], result.terminal_currents)
        np.testing.assert_allclose(conductance[index], result.conductance)
        np.testing.assert_allclose(charges[index], result.terminal_charges)
        np.testing.assert_allclose(capacitance[index], result.capacitance)


def test_native_5t_ota_dc_ac_and_noise_without_ngspice(monkeypatch):
    from circuitopt.ac_solver import ac_solve
    from circuitopt.noise_solver import noise_analysis

    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/ngspice")
    spec, config = _spec()
    frequencies = np.logspace(3, 11, 81)
    ac = ac_solve(
        spec.sizes,
        spec.bias,
        frequencies,
        topo=spec.topology,
        nf=spec.nf,
        x0_guess=dict(config["dc_guesses"][0]),
        model_types=spec.model_types,
        device_kwargs=spec.device_kwargs,
    )
    assert ac is not None
    assert 25.0 < 20 * np.log10(np.max(ac["gains"])) < 40.0
    noise = noise_analysis(
        spec.sizes,
        spec.bias,
        frequencies,
        topo=spec.topology,
        nf=spec.nf,
        x0_guess=dict(ac["dc_op"]),
        model_types=spec.model_types,
        device_kwargs=spec.device_kwargs,
    )
    assert noise is not None
    assert np.all(np.isfinite(noise["out_psd"]))
    assert np.all(noise["out_psd"] > 0)


def test_native_5t_ota_transient_without_ngspice(monkeypatch):
    from circuitopt.transient_solver import transient

    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/ngspice")
    spec, _ = _spec(driven=True)
    time = np.linspace(0.0, 20e-9, 101)
    vip = np.where(time < 5e-9, 0.55, 0.56)
    vin = np.where(time < 5e-9, 0.55, 0.54)
    result = transient(
        spec.sizes,
        spec.bias,
        time,
        binding=spec.binding(),
        inputs={"vip": vip, "vin": vin},
        V0=np.asarray((0.1, 0.45, 0.45)),
        integration_method="gear2",
        max_step=0.2e-9,
    )
    assert result["backend"] == "bsim4_native"
    from circuitopt._engine import current_engine
    # v2.0.0: rust is the only engine; the retired numba flags are gone (R7).
    assert current_engine() == "rust"
    assert "numba_grid_solver" not in result
    assert "bsim4_numba_transient" not in result
    assert result["bsim4_rust_transient"] is True
    assert "transient_profile" not in result
    assert result["nfail"] == 0
    assert result["nodes"]["vout"][-1] > result["nodes"]["vout"][0] + 0.2
    assert np.all(np.isfinite(result["nodes"]["vout"]))


def test_native_5t_ota_adaptive_gear2_uses_nonuniform_grid(monkeypatch):
    from circuitopt.transient_solver import transient

    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/ngspice")
    spec, _ = _spec(driven=True)
    source_time = np.linspace(10e-9, 30e-9, 101)
    vip = np.where(source_time < 15e-9, 0.55, 0.56)
    vin = np.where(source_time < 15e-9, 0.55, 0.54)
    result = transient(
        spec.sizes,
        spec.bias,
        source_time,
        binding=spec.binding(),
        inputs={"vip": vip, "vin": vin},
        V0=np.asarray((0.1, 0.45, 0.45)),
        integration_method="gear2",
        adaptive=True,
        adaptive_reltol=1e-4,
        adaptive_vabstol=1e-6,
        max_step=1e-9,
        profile=True,
    )

    accepted_time = result["t"]
    assert result["backend"] == "bsim4_native"
    assert result["adaptive"] is True
    assert result["nfail"] == 0
    assert accepted_time[0] == pytest.approx(source_time[0])
    assert accepted_time[-1] == pytest.approx(source_time[-1])
    assert np.all(np.diff(accepted_time) > 0.0)
    assert not np.allclose(
        np.diff(accepted_time),
        np.diff(accepted_time)[0],
        rtol=1e-10,
        atol=0.0,
    )
    assert np.any(np.isclose(accepted_time, 15e-9, rtol=0.0, atol=1e-18))
    assert result["adaptive_accepted_steps"] == len(accepted_time) - 1
    # A trial with no BDF history (the solve start, and every restart at an
    # input breakpoint) is error-controlled by solving the step once whole and
    # once as two halves, then accepting both halves. Such a trial therefore
    # costs three solves and yields two accepted samples, so trials exceed the
    # accept/reject decision count instead of matching it.
    assert result["transient_profile"]["trial_solves"] >= (
        result["adaptive_accepted_steps"] + result["adaptive_rejected_steps"]
    )
    assert result["transient_profile"]["solver_steps"] == (
        result["transient_profile"]["trial_solves"]
    )
    assert result["transient_profile"]["lte_estimates"] > 0
    # A Richardson restart estimate needs no defect linear solve, so estimates
    # lead linear solves by exactly the number of restart trials.
    assert result["transient_profile"]["lte_estimates"] >= (
        result["transient_profile"]["lte_linear_solves"]
    )
    assert result["nodes"]["vout"][-1] > result["nodes"]["vout"][0] + 0.2
    assert np.all(np.isfinite(result["nodes"]["vout"]))


def test_native_mdac_adaptive_lte_ignores_algebraic_branch_currents(monkeypatch):
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.transient_solver import transient

    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/ngspice")
    spec = load_circuit_json(
        os.path.join(_ROOT, "examples", "freepdk45_mdac_ota.json"))
    seed = spec.topology.dc_guesses[0]
    initial = np.asarray([
        seed.get(node, 0.0) for node in spec.topology.solved
    ])
    source_time = np.linspace(0.0, 5e-9, 501)
    common_mode = spec.bias["VDD"] / 2.0
    residue = -0.9 / 16.0
    bp1 = np.full_like(source_time, common_mode + residue / 2.0)
    bp2 = np.full_like(source_time, common_mode - residue / 2.0)
    bp1[0] = common_mode
    bp2[0] = common_mode

    result = transient(
        spec.sizes,
        spec.bias,
        source_time,
        binding=spec.binding(),
        V0=initial,
        inputs={"bp1": bp1, "bp2": bp2},
        integration_method="gear2",
        adaptive=True,
        max_step=0.5e-9,
        profile=True,
    )

    steps = np.diff(result["t"])
    assert result["nfail"] == 0
    assert result["adaptive_accepted_steps"] < 200
    assert result["adaptive_rejected_steps"] > 0
    assert steps.min() < 2e-12
    assert steps.max() > 100e-12
    assert result["output"][-1] == pytest.approx(0.45, abs=2e-3)


# (v2.0.0) test_native_5t_ota_rust_grid_matches_numba was removed: it did a live
# rust-grid vs numba-grid A/B, and the numba grid solver (compact_models/bsim4/
# numba_transient.py) was deleted with the numba engine. The rust BSIM4 transient
# is covered by tests/golden/engine_parity (freepdk45_5t_ota transient circuit
# golden + devices.npz device grids) and the contract test below.


def test_native_5t_ota_rust_grid_transient(monkeypatch):
    """The BSIM4 transient runs the compiled (rust) grid; quiescent OTA holds.

    (R7: the numba bsim transient and the import-sabotage scaffolding that
    guarded against silently falling back to it were removed with the engine.)
    """
    from circuitopt.transient_solver import transient

    spec, _ = _spec(driven=True)
    monkeypatch.setenv("CIRCUIT_BSIM4_BACKEND", "rust")
    time = np.linspace(0.0, 1e-9, 6)
    result = transient(
        spec.sizes,
        spec.bias,
        time,
        binding=spec.binding(),
        inputs={
            "vip": np.full_like(time, 0.55),
            "vin": np.full_like(time, 0.55),
        },
        V0=np.asarray((0.1, 0.45, 0.45)),
        integration_method="be",
        max_step=0.2e-9,
        profile=True,
    )

    assert result["bsim4_rust_transient"] is True
    assert "bsim4_numba_transient" not in result
    assert result["nfail"] == 0
    profile = result["transient_profile"]
    assert profile["enabled"] is True
    assert profile["backend"] == "bsim4_native"
    assert profile["rust_grid_solver"] is True
    assert profile["solver_steps"] == len(time) - 1
    assert profile["nsubsteps"] == 0
    assert profile["newton_iters_total"] > 0
    assert profile["bsim_batch_calls"] == (
        profile["newton_iters_total"] + 1
    )
    assert profile["bsim_evaluations"] == (
        profile["bsim_batch_calls"]
    ) * len(spec.sizes)
    assert profile["gear2_predictor_steps"] == 0
    assert profile["failed_steps"] == 0
    assert profile["failed_step_indices"] == []
    assert profile["first_failed_step"] is None
    assert profile["wall_time_s"] > 0.0


def test_native_transient_uses_rust_terminal_history_without_scalar_replay(monkeypatch):
    """Accepted Rust states carry I/Q history; Python must not re-evaluate MOS."""
    from circuitopt.pdk.freepdk45.device import _Fp45NativeFet
    from circuitopt.transient_solver import transient

    def unexpected_scalar_replay(*_args, **_kwargs):
        raise AssertionError("native transient replayed a scalar BSIM evaluation")

    monkeypatch.setattr(
        _Fp45NativeFet, "get_terminal_currents", unexpected_scalar_replay)
    monkeypatch.setattr(
        _Fp45NativeFet, "get_terminal_charges", unexpected_scalar_replay)
    monkeypatch.setattr(
        _Fp45NativeFet, "get_terminal_linearization", unexpected_scalar_replay)

    spec, _ = _spec(driven=True)
    time = np.linspace(0.0, 1e-9, 6)
    result = transient(
        spec.sizes,
        spec.bias,
        time,
        binding=spec.binding(),
        inputs={
            "vip": np.full_like(time, 0.55),
            "vin": np.full_like(time, 0.55),
        },
        V0=np.asarray((0.1, 0.45, 0.45)),
        integration_method="be",
        max_step=0.2e-9,
    )

    expected_vdd = np.asarray((
        -3.6835149759650734e-05,
        -3.310677162408139e-05,
        -3.3344422110229174e-05,
        -3.335697816084398e-05,
        -3.3357580817985198e-05,
        -3.3357399055487818e-05,
    ))
    assert result["nfail"] == 0
    assert result["nsubsteps"] == 0
    np.testing.assert_allclose(
        result["branch_currents"]["rail:VDD"], expected_vdd, rtol=1e-12, atol=1e-15)
    assert np.all(np.isfinite(result["branch_currents"]["gate:M1"]))
    assert np.all(np.isfinite(result["branch_currents"]["gate:M2"]))


def test_native_5t_ota_pss_without_ngspice(monkeypatch):
    from circuitopt.pss_solver import pss_solve

    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/ngspice")
    spec, _ = _spec()
    period = 10e-9
    result = pss_solve(
        spec.sizes,
        spec.bias,
        period,
        binding=spec.binding(),
        tgrid=np.linspace(0.0, period, 21),
        V0=np.asarray((0.05066, 0.46044, 0.46033)),
        max_shooting_iters=3,
        residual_tol=1e-7,
        max_step=0.5e-9,
    )
    assert result["backend"] == "bsim4_native"
    assert result["converged"] is True
    assert result["nfail"] == 0
    assert result["residual_norm"] < 1e-7
