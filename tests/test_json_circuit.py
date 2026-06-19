import json
from pathlib import Path

import numpy as np
import pytest

from core.ac_solver import ac_solve
from core.circuit_loader import circuit_from_dict, load_circuit_json
from core.noise_solver import band_rms, noise_analysis
from core.topology import AFE_TOPO
from core.transient_solver import transient


ROOT = Path(__file__).resolve().parents[1]


def test_example_json_matches_schema_when_jsonschema_available():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schemas" / "circuit.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    for name in ("single_stage.json", "resistor_load_stage.json", "afe_explore.json"):
        data = json.loads((ROOT / "examples" / name).read_text())
        jsonschema.validate(data, schema)


def test_load_single_stage_json_runs_all_analyses():
    spec = load_circuit_json("examples/single_stage.json")
    freqs = np.logspace(0, 4, 21)

    ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
    assert ac is not None
    assert np.isfinite(ac["dc_op"]["OUT"])
    assert np.isfinite(ac["gains"]).all()
    assert ac["gains"][0] > 0.0

    noise = noise_analysis(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
    assert noise is not None
    assert np.isfinite(noise["out_psd"]).all()
    assert band_rms(freqs, noise["out_psd"], 1.0, 100.0) > 0.0

    t = np.linspace(0, 1e-3, 50)
    vin = np.full_like(t, spec.bias["VIN"]) + np.where(t >= 2e-4, 1e-3, 0.0)
    tr = transient(spec.sizes, spec.bias, t, topo=spec.topology,
                   nf=spec.nf, inputs={"vin": vin})
    assert tr["nfail"] == 0
    assert np.isfinite(tr["output"]).all()
    assert abs(tr["output"][-1] - tr["output"][0]) > 1e-8


def test_afe_json_matches_builtin_topology_ac():
    spec = load_circuit_json("examples/afe_explore.json")
    freqs = np.logspace(0, 4, 21)

    json_ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
    builtin_ac = ac_solve(spec.sizes, spec.bias, freqs, topo=AFE_TOPO, nf=spec.nf)

    assert json_ac is not None
    assert builtin_ac is not None
    np.testing.assert_allclose(json_ac["gains"], builtin_ac["gains"], rtol=1e-10, atol=1e-12)
    assert json_ac["bw_Hz"] == pytest.approx(builtin_ac["bw_Hz"], rel=1e-10)


def test_loader_rejects_unknown_device_node():
    bad = {
        "solved": ["OUT"],
        "rails": {"VDD": "VDD", "GND": 0.0},
        "devices": [
            {"name": "M1", "drain": "OUT", "gate": "MISSING", "source": "VDD",
             "W": 1000, "L": 80}
        ],
        "bias": {"VDD": 40.0},
        "outputs": ["OUT"],
    }
    try:
        circuit_from_dict(bad)
    except ValueError as exc:
        assert "unknown node" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown node")
