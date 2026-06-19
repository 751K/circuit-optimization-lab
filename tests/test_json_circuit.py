import json
from pathlib import Path

import numpy as np
import pytest

import core.analysis_dispatch as dispatch_mod
from core.ac_solver import ac_solve
from core.analysis_dispatch import run_analysis_suite
from core.circuit_loader import circuit_from_dict, load_circuit_json
from core.noise_solver import _KB, _TEMP, band_rms, noise_analysis
from core.topology import AFE_TOPO
from core.transient_solver import transient


ROOT = Path(__file__).resolve().parents[1]


def test_example_json_matches_schema_when_jsonschema_available():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schemas" / "circuit.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    for name in ("single_stage.json", "resistor_load_stage.json", "afe_explore.json",
                 "periodic_rc.json"):
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


def test_periodic_json_dispatch_runs_generic_pss_pac_pnoise():
    spec = load_circuit_json("examples/periodic_rc.json")
    results = run_analysis_suite(spec)

    assert set(results) == {"ac", "noise", "pss", "pac", "pnoise"}
    assert results["pss"]["converged"]
    assert results["pss"]["nfail"] == 0
    assert results["pac"]["pac_condition_computed"] is False

    freqs = np.array([100.0, 1000.0])
    R = 1e5
    C = 1e-9
    expected_h = 1.0 / (1.0 + 2j * np.pi * freqs * R * C)
    np.testing.assert_allclose(results["ac"]["gains"], np.abs(expected_h), rtol=1e-6)
    np.testing.assert_allclose(results["pac"]["gains"], np.abs(expected_h), rtol=2e-2)

    z = 1.0 / (1.0 / R + 2j * np.pi * freqs * C)
    expected_noise = np.abs(z) ** 2 * (4.0 * _KB * _TEMP / R)
    np.testing.assert_allclose(results["pnoise"]["out_psd"], expected_noise, rtol=1e-5)
    assert results["pnoise"]["method"] == "lti_noise_fast_path"
    assert results["pnoise"]["pnoise_hb_solve_count"] == 0
    assert results["pnoise"]["irn_uV_band"] > 0.0


def test_dispatch_reuses_ac_dc_op_as_noise_seed(monkeypatch):
    spec = load_circuit_json("examples/single_stage.json")
    freqs = np.array([1.0, 10.0])
    ac_dc = {"OUT": 12.0}
    ac_result = {"dc_op": ac_dc, "gains": np.ones_like(freqs), "freqs": freqs.copy()}

    def fake_ac_solve(*_args, **_kwargs):
        return ac_result

    def fake_noise_analysis(*_args, **kwargs):
        assert kwargs["x0_guess"] is ac_dc
        assert kwargs["ac_result"] is ac_result
        return {
            "out_psd": np.ones_like(freqs),
            "irn_psd": np.ones_like(freqs),
        }

    monkeypatch.setattr(dispatch_mod, "ac_solve", fake_ac_solve)
    monkeypatch.setattr(dispatch_mod, "noise_analysis", fake_noise_analysis)

    results = run_analysis_suite(
        spec,
        analyses={"ac": {"freqs": freqs.tolist()},
                  "noise": {"freqs": freqs.tolist()}},
    )

    assert set(results) == {"ac", "noise"}


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
