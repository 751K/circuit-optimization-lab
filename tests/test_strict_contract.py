"""Behavior gates for explicit model binding and invalid-result propagation."""
from types import SimpleNamespace

import numpy as np
import pytest

import circuitopt  # noqa: F401 - register built-in PDKs
import circuitopt.explore as explore_module
from circuitopt.ac_solver import ac_solve
from circuitopt.circuit_loader import circuit_from_dict, load_circuit_json
from circuitopt.explore import explore, parse_explore
from circuitopt.run_contract import (
    ModelBindingError,
    SignoffConfigurationError,
    SimulationInvalid,
    ensure_analysis_valid,
    evaluate_signoff,
    summarize_design_metrics,
)
from circuitopt.topology import Topology


def _one_mos(models=None):
    return {
        "solved": ["OUT"],
        "rails": {"VDD": 40.0, "GND": 0.0},
        "devices": [
            {
                "name": "M1", "drain": "OUT", "gate": "GND",
                "source": "VDD", "W": 1000.0, "L": 80.0,
            }
        ],
        **({"models": models} if models is not None else {}),
    }


@pytest.mark.parametrize(
    ("models", "message"),
    [
        (None, "models is required"),
        ({"M1": {"pdk": "at4000tg", "model": "pmos",
                 "section": "inherit"}}, "missing required binding key.*bin"),
        ({"M1": {"type": "at4000tg.pmos"}}, "type is obsolete"),
    ],
)
def test_every_mos_requires_complete_explicit_binding(models, message):
    with pytest.raises(ValueError, match=message):
        circuit_from_dict(_one_mos(models))


def test_fixed_section_rejects_conflicting_corner():
    spec = circuit_from_dict(_one_mos({
        "M1": {
            "pdk": "sky130", "model": "pmos",
            "section": "tt", "bin": "auto", "vb": 40.0,
        }
    }))
    with pytest.raises(ValueError, match="fixed section.*conflicts"):
        spec.binding().at_corner("ss")


def test_nonfinite_analysis_is_invalid():
    with pytest.raises(SimulationInvalid, match="non-finite"):
        ensure_analysis_valid(
            "noise", {"out_psd": np.array([1.0, np.nan]),
                      "irn_psd": np.ones(2)})


def test_explore_keeps_failed_candidate_as_invalid(monkeypatch):
    topo = Topology(
        solved=["OUT"], devices=[],
        rails={"VIN": "VIN", "GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "VIN", "OUT", 1e3),
                   ("R2", "OUT", "GND", 1e3)],
    )
    cfg = parse_explore({
        "variables": {"VIN": {"min": 0.9, "max": 1.1}},
        "objectives": {"power_uW": "min"},
        "freqs": {"start": 0, "stop": 1, "num": 2},
    })

    def invalid(*_args, **_kwargs):
        raise SimulationInvalid(
            "model_evaluation_failed", "device model failed", analysis="ac")

    monkeypatch.setattr(explore_module, "evaluate", invalid)
    result = explore(topo, {}, {"VIN": 1.0}, None, cfg, n=2)
    assert all(row["status"]["state"] == "invalid"
               for row in result["candidates"])
    assert all(row["metrics"] is None for row in result["candidates"])


def test_runtime_model_bin_failure_marks_candidate_invalid():
    topo = Topology(
        solved=["OUT"],
        devices=[("M1", "OUT", "GND", "VDD")],
        rails={"VDD": 40.0, "GND": 0.0},
        outputs=("OUT",),
        model_types={"M1": "at4000tg.pmos"},
        device_kwargs={"M1": {"section": "inherit", "bin": "nonexistent"}},
        dc_guesses=[{"OUT": 20.0}],
    )
    cfg = parse_explore({
        "variables": {"M1.W": {"min": 1000.0, "max": 1100.0}},
        "objectives": {"power_uW": "min"},
        "freqs": {"start": 1.0, "stop": 10.0, "num": 2},
    })
    result = explore(
        topo,
        {"M1": (1000.0, 80.0)},
        {},
        None,
        cfg,
        n=1,
        model_types=topo.model_types,
        device_kwargs=topo.device_kwargs,
    )
    row = result["candidates"][0]
    assert row["status"]["state"] == "invalid"
    assert row["status"]["code"] == "model_binding_failed"
    assert row["metrics"] is None

    with pytest.raises(
        (ModelBindingError, ValueError),
        match="(requested bin|cannot verify explicit bin)",
    ):
        ac_solve(
            {"M1": (1000.0, 80.0)},
            {},
            np.array([1.0]),
            topo=topo,
        )


def test_source_power_uses_solved_rail_current():
    topo = Topology(
        solved=["OUT"], devices=[],
        rails={"VDD": "VDD", "GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "VDD", "OUT", 1e3),
                   ("R2", "OUT", "GND", 3e3)],
    )
    result = ac_solve({}, {"VDD": 10.0}, np.array([1.0]), topo=topo)
    assert result["source_power"]["source_currents_a"]["VDD"] == pytest.approx(
        2.5e-3)
    assert result["source_power"]["per_source_w"]["VDD"] == pytest.approx(
        25e-3)
    assert result["source_power"]["total_w"] == pytest.approx(25e-3)


def test_ac_reports_resolved_device_bindings():
    spec = load_circuit_json("examples/single_stage.json")
    result = ac_solve(
        spec.sizes, spec.bias, np.array([1.0]), binding=spec.binding())
    binding = result["device_bindings"]["MPU"]
    assert binding["pdk"] == "at4000tg"
    assert binding["model"] == "pmos"
    assert binding["section_selector"] == "inherit"
    assert binding["bin_selector"] == "auto"


def test_ac_exposes_raw_node_voltage_and_loop_injection_metadata():
    topo = Topology(
        solved=["RET"],
        devices=[],
        rails={"GND": 0.0},
        outputs=("RET",),
        vsources=[("Vinj", "RET", "GND", 0.0)],
        ac_drives={"Vinj": 1.0},
    )
    result = ac_solve(
        {}, {}, np.array([1.0, 10.0]), topo=topo,
        record_node_voltages=True,
    )
    np.testing.assert_allclose(result["node_voltages"]["RET"], 1.0 + 0.0j)
    assert result["ac_stimulus"]["drives"] == {"Vinj": 1.0}
    assert result["ac_stimulus"]["normalization_v"] == pytest.approx(1.0)


def _signoff_fixture(*, final_output=1.0):
    freqs = np.array([1.0, 10.0, 100.0, 1000.0])
    loop_gain = 10.0 / (1.0 + 1j * freqs / 10.0) ** 2
    response = np.array([10.0 + 0j, 1.0 - 1j, 0.1 - 0.2j, 0.01j])
    results = {
        "ac": {
            "Av_dc_dB": 20.0,
            "freqs": freqs,
            "response": response,
            "node_voltages": {
                "INJ": np.ones(4, dtype=complex),
                "RET": -loop_gain,
                "OUTP": response / 2.0,
                "OUTN": -response / 2.0,
            },
            "ac_stimulus": {
                "drives": {"Vinj": 1.0},
                "normalization_v": 1.0,
            },
            "source_power": {
                "total_w": 1.2e-3,
                "per_source_w": {"VDD": 1.2e-3},
                "source_currents_a": {"VDD": 1.0e-3},
                "source_voltages_v": {"VDD": 1.2},
            },
            "operating_regions": {
                "M1": {
                    "status": "valid", "saturated": True,
                    "vds_v": 0.8, "vdsat_v": 0.2, "headroom_v": 0.6,
                }},
        },
        "noise": {
            "freqs": freqs,
            "out_psd": np.full(4, 4e-18),
            "irn_psd": np.full(4, 1e-18),
        },
        "transient": {
            "t": np.array([0.0, 1e-9, 2e-9, 3e-9]),
            "output": np.array([0.0, 0.9, 0.9995, final_output]),
            "nodes": {
                "OUTP": np.array([0.0, 0.45, 0.49975, final_output / 2.0]),
                "OUTN": np.array([0.0, -0.45, -0.49975, -final_output / 2.0]),
            },
        },
    }
    topology = SimpleNamespace(
        vsource_index={"Vinj": 3},
        vsources=[("Vinj", "INJ", "RET", 0.0)],
        ac_drives={"Vinj": 1.0},
        solved=["INJ", "RET", "OUTP", "OUTN"],
        devices=[("M1", "OUTP", "RET", "GND")],
    )
    spec = SimpleNamespace(
        topology=topology,
        analyses={"ac": {}, "noise": {}, "transient": {}},
        signoff={
            "measurements": {
                "phase_margin": {
                    "analysis": "ac",
                    "injection_source": "Vinj",
                    "return_signal": {"RET": 1.0},
                    "polarity": -1,
                },
                "settling_time": {
                    "analysis": "transient",
                    "signal": {"OUTP": 1.0, "OUTN": -1.0},
                    "target": 1.0,
                    "start_time": 0.0,
                    "tolerance": {"relative": 1e-3, "reference": 1.0},
                },
                "noise": {
                    "analysis": "noise",
                    "band": [2.0, 50.0],
                    "references": ["input", "output"],
                },
                "saturation": {
                    "analysis": "ac",
                    "devices": ["M1"],
                    "minimum_headroom": 0.05,
                },
            },
            "constraints": {
                "phase_margin": {"min": 20.0},
                "settling_time": {"max": 3e-9},
                "integrated_input_noise": {"max": 1e-6},
                "saturation": {"equals": True},
            },
        },
    )
    return spec, results


def test_unified_signoff_is_explicit_unit_bearing_and_passes():
    spec, results = _signoff_fixture()
    results["ac"]["ac_stimulus"]["drives"]["Vinj"] = 1.0 + 0.0j
    signoff = evaluate_signoff(spec, results)
    metrics = signoff["measurements"]
    assert metrics["gain"]["unit"] == "dB"
    assert metrics["phase_margin"]["unit"] == "deg"
    assert metrics["phase_margin"]["response_kind"] == "loop_gain"
    assert metrics["phase_margin"]["injection_source"] == "Vinj"
    assert metrics["settling_time"]["unit"] == "s"
    assert metrics["settling_time"]["target"]["value"] == pytest.approx(1.0)
    assert metrics["integrated_input_noise"]["unit"] == "V_rms"
    assert metrics["integrated_output_noise"]["integration_band_hz"] == [
        2.0, 50.0]
    assert metrics["integrated_input_noise"]["value"] == pytest.approx(
        np.sqrt(48e-18))
    assert metrics["saturation"]["value"] is True
    assert metrics["dc_source_power"]["value"] == pytest.approx(1.2e-3)
    branch = metrics["dc_source_power"]["branches"]["VDD"]
    assert branch["voltage"]["unit"] == "V"
    assert branch["current"]["unit"] == "A"
    assert branch["power"]["unit"] == "W"
    assert metrics["saturation"]["devices"]["M1"]["headroom"]["unit"] == "V"
    assert signoff["status"] == "pass"
    assert signoff["passed"] is True
    assert set(signoff["constraints"]) == {
        "phase_margin", "settling_time", "integrated_input_noise", "saturation",
    }
    assert signoff["worst_case"]["measurement"] in signoff["constraints"]


def test_transient_saturation_recomputes_regions_at_declared_checkpoints(
    monkeypatch,
):
    seen = []

    def regions(_topology, _bias, node_values, _devices):
        seen.append(node_values["OUTP"])
        return {
            "M1": {
                "status": "valid",
                "saturated": True,
                "vds_v": node_values["OUTP"],
                "vdsat_v": 0.2,
                "headroom_v": node_values["OUTP"] - 0.2,
            }
        }

    monkeypatch.setattr("circuitopt.dc_measurements.operating_regions", regions)
    topology = SimpleNamespace(
        vsource_index={},
        ac_drives={},
        solved=["OUTP"],
        devices=[("M1", "OUTP", "OUTP", "GND")],
    )
    spec = SimpleNamespace(
        topology=topology,
        sizes={"M1": (1.0, 1.0)},
        bias={},
        binding=lambda: SimpleNamespace(
            build=lambda _sizes: {"M1": object()}),
        analyses={"transient": {}},
        signoff={
            "measurements": {
                "saturation": {
                    "analysis": "transient",
                    "devices": ["M1"],
                    "minimum_headroom": 0.05,
                    "checkpoints": [
                        {"name": "static", "time": 0.0},
                        {"name": "settled", "time": 2e-9},
                    ],
                },
            },
            "constraints": {"saturation": {"equals": True}},
        },
    )
    results = {
        "transient": {
            "t": np.array([0.0, 1e-9, 2e-9]),
            "output": np.array([0.4, 0.5, 0.6]),
            "nodes": {"OUTP": np.array([0.4, 0.5, 0.6])},
        }
    }

    signoff = evaluate_signoff(spec, results)
    saturation = signoff["measurements"]["saturation"]
    assert seen == pytest.approx([0.4, 0.6])
    assert saturation["analysis"] == "transient"
    assert saturation["value"] is True
    assert saturation["checkpoints"]["static"]["time"] == {
        "value": 0.0, "unit": "s", "status": "valid"}
    assert saturation["checkpoints"]["settled"]["devices"]["M1"][
        "headroom"]["value"] == pytest.approx(0.4)


def test_transient_saturation_checkpoint_must_lie_inside_simulation():
    spec, results = _signoff_fixture()
    spec.signoff["measurements"]["saturation"] = {
        "analysis": "transient",
        "devices": ["M1"],
        "minimum_headroom": 0.05,
        "checkpoints": [{"name": "late", "time": 10e-9}],
    }
    with pytest.raises(SignoffConfigurationError, match="outside transient range"):
        evaluate_signoff(spec, results)


def test_pm_saturation_noise_and_settling_are_not_inferred():
    spec, results = _signoff_fixture()
    spec.signoff = None
    metrics = summarize_design_metrics(spec, results)
    assert "phase_margin" not in metrics
    assert "settling_time" not in metrics
    assert "integrated_input_noise" not in metrics
    assert "integrated_output_noise" not in metrics
    assert "saturation" not in metrics
    signoff = evaluate_signoff(spec, results)
    assert signoff["status"] == "not_configured"
    assert signoff["passed"] is None


def test_phase_margin_rejects_ordinary_ac_drive():
    spec, results = _signoff_fixture()
    spec.topology.ac_drives = {"OUTP": 1.0}
    with pytest.raises(SignoffConfigurationError, match="exactly injection source"):
        evaluate_signoff(spec, results)


def test_settling_uses_declared_target_not_last_sample():
    spec, results = _signoff_fixture(final_output=0.99)
    signoff = evaluate_signoff(spec, results)
    settling = signoff["measurements"]["settling_time"]
    assert settling["status"] == "not_settled"
    assert settling["value"] is None
    assert signoff["status"] == "fail"
    assert signoff["constraints"]["settling_time"]["passed"] is False
    assert signoff["constraints"]["settling_time"][
        "normalized_margin"] == pytest.approx(-9.0)
