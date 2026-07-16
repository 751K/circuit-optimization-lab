"""TSMC28 native BSIM4 versus explicit ngspice oracle on a compact 5T OTA."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from circuitopt.ngspice_char import ngspice_binary
from circuitopt.toolchain import tsmc28_model_dir


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "tsmc28hpcp_5t_ota.json"
MODEL = os.path.join(
    tsmc28_model_dir(), "cln28hpcp_1d8_elk_v1d0_2p2.l")
needs_model = pytest.mark.skipif(
    not os.path.isfile(MODEL), reason="TSMC28 model deck not configured")
needs_oracle = pytest.mark.skipif(
    not os.path.isfile(MODEL) or ngspice_binary() is None,
    reason="TSMC28 model deck or ngspice oracle unavailable",
)


def test_benchmark_is_a_native_five_transistor_ota():
    from circuitopt import load_circuit_json

    spec = load_circuit_json(CONFIG)
    assert len(spec.topology.devices) == 5
    assert set(spec.model_types.values()) == {
        "tsmc28hpcp.nmos",
        "tsmc28hpcp.pmos",
    }
    assert not spec.adc


@needs_model
def test_native_5t_ota_runs_without_ngspice(monkeypatch):
    from circuitopt import ac_solve, load_circuit_json, noise_analysis

    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/an/executable")
    spec = load_circuit_json(CONFIG)
    freqs = np.logspace(3, 10, 41)
    ac = ac_solve(
        spec.sizes, spec.bias, freqs, binding=spec.binding(), corner="tt")
    noise = noise_analysis(
        spec.sizes, spec.bias, freqs, binding=spec.binding(), corner="tt",
        ac_result=ac)
    assert ac is not None and noise is not None
    assert 30.0 < ac["Av_dc_dB"] < 40.0
    assert 0.4 < ac["dc_op"]["vout"] < 0.5
    assert np.all(np.isfinite(noise["out_psd"]))


@needs_model
def test_native_5t_ota_periodic_noise_uses_terminal_matrix_path():
    from circuitopt import ac_solve, load_circuit_json, noise_analysis
    from circuitopt.pnoise_solver import pnoise_solve

    spec = load_circuit_json(CONFIG)
    freqs = np.array([1e3, 1e5, 1e7])
    ac = ac_solve(
        spec.sizes, spec.bias, freqs,
        binding=spec.binding(), corner="tt")
    noise = noise_analysis(
        spec.sizes, spec.bias, freqs,
        binding=spec.binding(), corner="tt", ac_result=ac)
    period = 1e-6
    t = np.array([0.0, period])
    dc_vector = np.array([
        ac["dc_op"][name] for name in spec.topology.solved])
    pss = {
        "topology": spec.topology,
        "t": t,
        "period": period,
        "nodes": {
            name: np.full(2, ac["dc_op"][name])
            for name in spec.topology.solved
        },
        "inputs": {},
        "bias": spec.bias,
        "model_types": spec.model_types,
        "device_kwargs": spec.device_kwargs,
        "all_sizes": spec.sizes,
        "all_nf": spec.nf,
        "x0": dc_vector,
        "corner": "tt",
    }
    pnoise = pnoise_solve(
        spec.sizes,
        spec.bias,
        freqs,
        pss_result=pss,
        nf=spec.nf,
        corner="tt",
        max_sideband=0,
        n_period_samples=8,
        gains=np.ones_like(freqs),
        lti_fast_path=False,
        cache_linearization=False,
    )
    assert pnoise["pnoise_terminal_noise_source_count"] == 5
    assert pnoise["pnoise_numba_fold_used"] is False
    np.testing.assert_allclose(
        pnoise["out_psd"], noise["out_psd"], rtol=2e-6)


@needs_oracle
def test_native_5t_ota_matches_ngspice_ac_and_noise():
    from circuitopt import ac_solve, load_circuit_json, noise_analysis
    from circuitopt.ngspice_ac import ac_ngspice, ac_response, noise_ngspice

    spec = load_circuit_json(CONFIG)
    oracle_models = {
        name: model.replace("tsmc28hpcp.", "tsmc28hpcp_ngspice.")
        for name, model in spec.model_types.items()
    }
    shared = dict(
        topo=spec.topology,
        nf=spec.nf,
        model_types=oracle_models,
        device_kwargs=spec.device_kwargs,
        corner="tt",
        x0_guess=spec.topology.dc_guesses[0],
    )
    oracle_ac = ac_ngspice(
        spec.sizes, spec.bias,
        acmag={"vinp": (0.5, 0.0), "vinn": (0.5, 180.0)},
        fstart=1e3, fstop=1e10, points=8, out_nodes=["vout"], **shared)
    native_ac = ac_solve(
        spec.sizes, spec.bias, oracle_ac["freq"],
        binding=spec.binding(), corner="tt")
    native_h = native_ac["response"]
    oracle_h = ac_response(oracle_ac, "vout", vin=1.0)
    np.testing.assert_allclose(
        np.abs(native_h), np.abs(oracle_h), rtol=0.01, atol=1e-8)

    oracle_noise = noise_ngspice(
        spec.sizes, spec.bias, out="vout", src="vinp",
        fstart=1e3, fstop=1e10, points=8, band=(1e3, 1e10), **shared)
    native_noise = noise_analysis(
        spec.sizes, spec.bias, oracle_noise["freq"],
        binding=spec.binding(), corner="tt")
    native_rms = np.sqrt(np.trapezoid(
        native_noise["out_psd"], oracle_noise["freq"]))
    assert native_rms == pytest.approx(oracle_noise["onoise_rms"], rel=0.02)
