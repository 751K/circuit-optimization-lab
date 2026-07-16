from __future__ import annotations

import os

import numpy as np
import pytest


def _model_available():
    from circuitopt.toolchain import tsmc28_model_dir

    return os.path.isfile(os.path.join(
        tsmc28_model_dir(),
        "cln28hpcp_1d8_elk_v1d0_2p2.l",
    ))


@pytest.mark.skipif(not _model_available(), reason="TSMC28 model deck not configured")
def test_native_inverter_charge_transient_without_ngspice(monkeypatch):
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
            "MN": {"type": "tsmc28hpcp.nmos"},
            "MP": {"type": "tsmc28hpcp.pmos", "vb": 0.9},
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
    )
    output = result["nodes"]["OUT"]
    assert result["backend"] == "bsim4_native"
    assert result["nfail"] == 0
    assert output[0] > 0.85
    assert output[-1] < 0.05
    assert np.all(np.isfinite(output))
