"""Strict multi-testbench PVT campaign behavior."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import circuitopt.signoff_campaign as campaign_module
from circuitopt.noise_solver import _resistor_noise_temperature
from circuitopt.run_contract import SimulationInvalid
from circuitopt.run_contract import validate_signoff_config
from circuitopt.circuit_loader import circuit_from_dict
from circuitopt.signoff_campaign import (
    CampaignConfigurationError,
    load_campaign_json,
    prepare_case_dict,
    run_signoff_campaign,
)


def _passive_circuit():
    return {
        "name": "campaign_fixture",
        "solved": ["OUT"],
        "rails": {"VDD": "VDD", "GND": 0.0},
        "bias": {"VDD": 0.9},
        "devices": [],
        "resistors": [
            {"name": "R1", "a": "VDD", "b": "OUT", "R": 1e3},
            {"name": "R2", "a": "OUT", "b": "GND", "R": 1e3},
        ],
        "outputs": ["OUT"],
        "dc_guesses": [{"OUT": 0.45}],
        "analyses": {
            "ac": {
                "freqs": {"start": 1.0, "stop": 10.0, "num": 2, "scale": "log"}
            }
        },
        "signoff": {
            "measurements": {},
            "constraints": {"gain": {"min": 1.0}},
        },
    }


def _manifest(circuit_name: str):
    return {
        "name": "fixture_45",
        "pvt": {
            "corners": ["tt", "ss", "ff", "sf", "fs"],
            "temperatures_c": [-40, 27, 125],
            "supplies_v": [0.85, 0.9, 0.95],
            "nominal_supply_v": 0.9,
            "supply_bias_key": "VDD",
        },
        "cases": [
            {"name": "open_loop", "circuit": circuit_name, "overrides": {}},
            {"name": "closed_loop", "circuit": circuit_name, "overrides": {}},
        ],
    }


def _write_fixture(tmp_path: Path):
    circuit_path = tmp_path / "fixture.json"
    circuit_path.write_text(json.dumps(_passive_circuit()), encoding="utf-8")
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(_manifest(circuit_path.name)), encoding="utf-8")
    return campaign_path


def _fake_signoff(spec, _results):
    margin = spec.bias["VDD"] - 0.8
    return {
        "status": "pass",
        "measurements": {
            "gain": {"value": 10.0, "unit": "dB", "status": "valid"},
        },
        "constraints": {},
        "passed": True,
        "worst_case": {
            "measurement": "gain",
            "passed": True,
            "normalized_margin": margin,
        },
    }


def test_campaign_runs_exact_45_point_grid_with_deterministic_parallel_order(
    tmp_path, monkeypatch,
):
    path = _write_fixture(tmp_path)
    monkeypatch.setattr(
        campaign_module, "run_analysis_suite", lambda _spec: {"ac": {}})
    monkeypatch.setattr(campaign_module, "evaluate_signoff", _fake_signoff)

    serial = run_signoff_campaign(path, workers=1)
    parallel = run_signoff_campaign(path, workers=4)

    assert serial == parallel
    assert serial["status"] == "pass"
    assert serial["grid"]["total_points"] == 45
    assert len(serial["points"]) == 45
    assert serial["summary"]["points"] == {
        "pass": 45, "fail": 0, "invalid": 0, "total": 45}
    assert serial["summary"]["cases"]["open_loop"]["pass"] == 45
    assert serial["points"][0]["pvt"] == {
        "corner": "tt", "temperature_c": -40.0, "supply_v": 0.85}
    assert serial["points"][-1]["pvt"] == {
        "corner": "fs", "temperature_c": 125.0, "supply_v": 0.95}
    assert serial["worst_case"]["case"] == "open_loop"
    assert serial["worst_case"]["corner"] == "tt"
    assert serial["worst_case"]["temperature_c"] == -40.0
    assert serial["worst_case"]["supply_v"] == 0.85


def test_campaign_retains_invalid_case_and_promotes_point_and_global_status(
    tmp_path, monkeypatch,
):
    path = _write_fixture(tmp_path)

    def run(spec):
        if spec.bias["VDD"] == pytest.approx(0.95):
            raise SimulationInvalid(
                "not_converged", "synthetic solve failure", analysis="ac")
        return {"ac": {}}

    monkeypatch.setattr(campaign_module, "run_analysis_suite", run)
    monkeypatch.setattr(campaign_module, "evaluate_signoff", _fake_signoff)
    result = run_signoff_campaign(path)

    assert result["status"] == "invalid"
    assert result["passed"] is False
    assert result["summary"]["points"]["invalid"] == 15
    invalid = next(point for point in result["points"]
                   if point["pvt"]["supply_v"] == 0.95)
    assert invalid["status"] == "invalid"
    assert invalid["cases"]["open_loop"]["error"] == {
        "code": "not_converged",
        "message": "synthetic solve failure",
        "analysis": "ac",
    }
    assert result["worst_case"]["status"] == "invalid"
    assert result["worst_case"]["case"] == "open_loop"
    assert result["worst_case"]["supply_v"] == 0.95


def test_campaign_cooperative_stop_returns_explicit_partial_result(
    tmp_path, monkeypatch,
):
    path = _write_fixture(tmp_path)
    monkeypatch.setattr(
        campaign_module, "run_analysis_suite", lambda _spec: {"ac": {}})
    monkeypatch.setattr(campaign_module, "evaluate_signoff", _fake_signoff)
    checks = 0

    def should_stop():
        nonlocal checks
        checks += 1
        return checks > 3

    result = run_signoff_campaign(
        path, workers=1, should_stop=should_stop)

    assert result["status"] == "cancelled"
    assert result["passed"] is False
    assert result["stopped_early"] is True
    assert result["grid"]["total_points"] == 45
    assert result["summary"]["points"]["total"] == 3
    assert len(result["points"]) == 3


def test_prepare_case_bakes_exact_corner_temperature_supply_and_pvt_expression():
    base = {
        "bias": {"VDD": 0.9},
        "models": {
            "MN": {
                "pdk": "tsmc28hpcp", "model": "nmos",
                "section": "inherit", "bin": "auto",
            },
            "MP": {
                "pdk": "tsmc28hpcp", "model": "pmos",
                "section": "inherit", "bin": "auto", "vb": 0.9,
            },
        },
        "dc_guesses": [{"OUT": 0.45}],
        "vsources": [["V1", "OUT", "GND", 0.45]],
    }
    deck = prepare_case_dict(
        base,
        {"bias": {
            "VCM": {"$pvt": {"vdd": 0.5}},
            "LEVEL": {"$pvt": {"vdd": 0.5, "constant": 0.225}},
        }},
        corner="sf",
        temperature_c=125.0,
        supply_v=0.95,
        nominal_supply_v=0.9,
        supply_bias_key="VDD",
    )
    assert deck["bias"]["VDD"] == pytest.approx(0.95)
    assert deck["bias"]["VCM"] == pytest.approx(0.475)
    assert deck["bias"]["LEVEL"] == pytest.approx(0.7)
    assert deck["models"]["MN"]["section"] == "sf"
    assert deck["models"]["MN"]["temperature"] == pytest.approx(398.15)
    assert deck["models"]["MP"]["vb"] == pytest.approx(0.95)
    assert deck["dc_guesses"][0]["OUT"] == pytest.approx(0.475)
    assert deck["vsources"][0][3] == pytest.approx(0.475)
    assert base["models"]["MN"]["section"] == "inherit"


def test_resistor_noise_uses_bound_pvt_temperature_and_rejects_mixed_values():
    devices = {
        "M1": type("Device", (), {"temperature": 233.15})(),
        "M2": type("Device", (), {"temperature": 233.15})(),
    }
    assert _resistor_noise_temperature(devices) == pytest.approx(233.15)
    devices["M2"].temperature = 398.15
    with pytest.raises(SimulationInvalid, match="one shared circuit temperature"):
        _resistor_noise_temperature(devices)


def test_campaign_rejects_absolute_and_escaping_circuit_paths(tmp_path):
    absolute = _manifest(str(tmp_path / "fixture.json"))
    path = tmp_path / "absolute.json"
    path.write_text(json.dumps(absolute), encoding="utf-8")
    with pytest.raises(
        CampaignConfigurationError, match="must be relative",
    ):
        load_campaign_json(path)

    child = tmp_path / "child"
    child.mkdir()
    escaping = _manifest("../fixture.json")
    path = child / "escaping.json"
    path.write_text(json.dumps(escaping), encoding="utf-8")
    with pytest.raises(CampaignConfigurationError, match="escapes"):
        run_signoff_campaign(path)


def test_campaign_schema_accepts_fixture_and_rejects_unknown_fields(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "schemas" / "signoff_campaign.schema.json").read_text())
    manifest = _manifest("fixture.json")
    jsonschema.validate(manifest, schema)
    manifest["pvt"]["mystery"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, schema)


def test_tsmc28_mdac_manifest_defines_11_cases_over_45_valid_pvt_points():
    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "tsmc28hpcp_mdac_ota_signoff.json"
    config, manifest = load_campaign_json(path)
    assert len(config["cases"]) == 11
    assert (
        len(config["pvt"]["corners"])
        * len(config["pvt"]["temperatures_c"])
        * len(config["pvt"]["supplies_v"])
    ) == 45

    for corner in config["pvt"]["corners"]:
        for case in config["cases"]:
            base = json.loads(
                (manifest.parent / case["circuit"]).read_text(encoding="utf-8"))
            deck = prepare_case_dict(
                base,
                case["overrides"],
                corner=corner,
                temperature_c=27.0,
                supply_v=0.9,
                nominal_supply_v=0.9,
                supply_bias_key="VDD",
            )
            validate_signoff_config(circuit_from_dict(deck))


def test_tsmc28_mdac_manifest_scopes_relaxed_newton_tolerance_to_transient():
    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "tsmc28hpcp_mdac_ota_signoff.json"
    config, _ = load_campaign_json(path)

    transient_cases = []
    for case in config["cases"]:
        analyses = case["overrides"].get("analyses", {})
        transient = analyses.get("transient")
        if transient is None:
            assert "newton_vtol" not in analyses
            continue
        transient_cases.append(case["name"])
        assert transient["newton_vtol"] == pytest.approx(3e-8)
        assert transient["newton_vtol"] < transient["adaptive_vabstol"]
        assert transient["bsim_model_bypass_tolerance"] == pytest.approx(3e-9)
        assert (
            transient["bsim_model_bypass_tolerance"]
            < transient["newton_vtol"]
        )

    assert transient_cases == [
        "residue_minus_fs16",
        "residue_minus_fs32",
        "residue_zero",
        "residue_plus_fs32",
        "residue_plus_fs16",
        "major_carry_0111_to_1000",
    ]
