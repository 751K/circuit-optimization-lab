from __future__ import annotations

import os

import numpy as np
import pytest


def test_expanded_grid_does_not_split_mdac_exact_step_multiples():
    from circuitopt.compact_models.bsim4.transient import _expanded_grid

    requested = np.linspace(0.0, 5e-9, 501)
    waveform = np.linspace(0.1, 0.9, len(requested))
    expanded, inputs, indices = _expanded_grid(
        requested, {"vin": waveform}, 1e-11)

    np.testing.assert_array_equal(expanded, requested)
    np.testing.assert_array_equal(inputs["vin"], waveform)
    np.testing.assert_array_equal(indices, np.arange(len(requested)))


def test_expanded_grid_preserves_integer_subdivision_and_real_excess():
    from circuitopt.compact_models.bsim4.transient import _expanded_grid

    exact, _, exact_indices = _expanded_grid(
        np.asarray([0.0, 3e-11]), {}, 1e-11)
    assert len(exact) == 4
    assert exact_indices.tolist() == [0, 3]

    within_tolerance, _, _ = _expanded_grid(
        np.asarray([0.0, 1.0 + 5e-13]), {}, 1.0)
    above_tolerance, _, above_indices = _expanded_grid(
        np.asarray([0.0, 1.0 + 2e-12]), {}, 1.0)
    assert len(within_tolerance) == 2
    assert len(above_tolerance) == 3
    assert above_indices.tolist() == [0, 2]


def _model_available():
    from circuitopt.toolchain import tsmc28_model_dir

    return os.path.isfile(os.path.join(
        tsmc28_model_dir(),
        "cln28hpcp_1d8_elk_v1d0_2p2.l",
    ))


@pytest.mark.skipif(not _model_available(), reason="TSMC28 model deck not configured")
@pytest.mark.parametrize("adaptive", [False, True])
def test_native_inverter_charge_transient_without_ngspice(monkeypatch, adaptive):
    from circuitopt.circuit_loader import circuit_from_dict
    from circuitopt.transient_solver import transient

    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/an/executable")
    spec = circuit_from_dict({
        "name": "native_tsmc28_inverter",
        "solved": ["OUT"],
        "rails": {"VDD": "VDD", "GND": 0.0, "IN": 0.0},
        "devices": [
            {"name": "MN", "drain": "OUT", "gate": "IN", "source": "GND",
             "W": 1.0, "L": 0.03},
            {"name": "MP", "drain": "OUT", "gate": "IN", "source": "VDD",
             "W": 2.0, "L": 0.03},
        ],
        "models": {
            "MN": {"pdk": "tsmc28hpcp", "model": "nmos", "section": "inherit", "bin": "auto"},
            "MP": {"pdk": "tsmc28hpcp", "model": "pmos", "section": "inherit", "bin": "auto", "vb": 0.9},
        },
        "bias": {"VDD": 0.9},
        "outputs": ["OUT"],
        "load_caps": [["OUT", "GND", 2e-15]],
        "transient_inputs": {"MN": "vin", "MP": "vin"},
        "dc_guesses": [{"OUT": 0.9}],
    })
    tgrid = np.linspace(0.0, 0.4e-9, 81)
    vin = np.where(tgrid < 0.1e-9, 0.0, 0.9)
    result = transient(
        spec.sizes,
        spec.bias,
        tgrid,
        binding=spec.binding(),
        inputs={"vin": vin},
        corner="tt",
        integration_method="gear2",
        max_step=1e-12,
        adaptive=adaptive,
    )
    output = result["nodes"]["OUT"]
    assert result["backend"] == "bsim4_native"
    assert result["adaptive"] is adaptive
    assert result["nfail"] == 0
    assert output[0] > 0.85
    assert output[-1] < 0.05
    assert np.all(np.isfinite(output))
