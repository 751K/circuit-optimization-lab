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
    assert NativeBsim4Backend.abi_version == 1
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
    assert result["numba_grid_solver"] is True
    assert result["bsim4_numba_transient"] is True
    assert result["nfail"] == 0
    assert result["nodes"]["vout"][-1] > result["nodes"]["vout"][0] + 0.2
    assert np.all(np.isfinite(result["nodes"]["vout"]))


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
