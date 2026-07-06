import json
import inspect
from pathlib import Path

import numpy as np
import pytest

import circuitopt.analysis_dispatch as dispatch_mod
from circuitopt.ac_solver import ac_solve
from circuitopt.analysis_options import (
    known_keys,
    option_names,
    schema_properties,
    validate_analysis_cfg,
)
from circuitopt.analysis_dispatch import run_analysis_suite
from circuitopt.circuit_loader import circuit_from_dict, load_circuit_json
from circuitopt.corners import CORNERS
from circuitopt.noise_solver import _KB, _TEMP, band_rms, noise_analysis
from circuitopt.pac_solver import pac_solve
from circuitopt.pnoise_solver import pnoise_solve
from circuitopt.pss_solver import pss_solve
from circuitopt.topology import AFE_TOPO
from circuitopt.transient_solver import transient


ROOT = Path(__file__).resolve().parents[1]


def test_example_json_matches_schema_when_jsonschema_available():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schemas" / "circuit.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    for name in ("single_stage.json", "resistor_load_stage.json", "afe_explore.json",
                 "periodic_rc.json"):
        data = json.loads((ROOT / "examples" / name).read_text())
        jsonschema.validate(data, schema)


def test_analysis_option_registry_matches_solver_signatures_and_schema():
    solver_by_analysis = {
        "transient": transient,
        "pss": pss_solve,
        "pac": pac_solve,
        "pnoise": pnoise_solve,
    }
    for analysis, solver in solver_by_analysis.items():
        signature_names = set(inspect.signature(solver).parameters)
        assert option_names(analysis, forwarded_only=True) <= signature_names

    schema = json.loads((ROOT / "schemas" / "circuit.schema.json").read_text())
    def_name = {
        "transient": "transientAnalysis",
        "pss": "pssAnalysis",
        "pac": "pacAnalysis",
        "pnoise": "pnoiseAnalysis",
    }
    for analysis, name in def_name.items():
        props = set(schema["$defs"][name]["properties"])
        assert set(schema_properties(analysis)) <= props


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


def test_dispatch_resolves_and_keeps_process_corner_consistent(monkeypatch):
    spec = load_circuit_json("examples/periodic_rc.json")
    freqs = np.array([100.0, 1000.0])
    seen = []

    def fake_ac_solve(*_args, **kwargs):
        seen.append(("ac", kwargs.get("corner")))
        return {
            "dc_op": {"OUT": 0.0},
            "freqs": freqs.copy(),
            "gains": np.ones_like(freqs),
            "response": np.ones_like(freqs, dtype=complex),
            "corner": kwargs.get("corner"),
        }

    def fake_noise_analysis(*_args, **kwargs):
        seen.append(("noise", kwargs.get("corner"), kwargs.get("ac_result") is not None))
        return {"out_psd": np.ones_like(freqs), "irn_psd": np.ones_like(freqs)}

    def fake_pss_solve(*args, **kwargs):
        seen.append(("pss", kwargs.get("corner")))
        return {"converged": True, "period": args[2], "corner": kwargs.get("corner")}

    def fake_pac_solve(*_args, **kwargs):
        seen.append(("pac", kwargs.get("corner"), kwargs["pss_result"].get("corner")))
        return {"gains": np.ones_like(freqs), "response": np.ones_like(freqs, dtype=complex)}

    def fake_pnoise_solve(*_args, **kwargs):
        seen.append(("pnoise", kwargs.get("corner"), kwargs["pss_result"].get("corner")))
        return {"out_psd": np.ones_like(freqs), "irn_psd": np.ones_like(freqs)}

    monkeypatch.setattr(dispatch_mod, "ac_solve", fake_ac_solve)
    monkeypatch.setattr(dispatch_mod, "noise_analysis", fake_noise_analysis)
    monkeypatch.setattr(dispatch_mod, "pss_solve", fake_pss_solve)
    monkeypatch.setattr(dispatch_mod, "pac_solve", fake_pac_solve)
    monkeypatch.setattr(dispatch_mod, "pnoise_solve", fake_pnoise_solve)

    analyses = {
        "ac": {"freqs": freqs.tolist(), "corner": "slow"},
        "noise": {"freqs": freqs.tolist()},
        "pss": {"max_shooting_iters": 0},
        "pac": {"freqs": freqs.tolist(), "input_drive": {"vin": 1.0}, "corner": "slow"},
        "pnoise": {"freqs": freqs.tolist(), "input_drive": {"vin": 1.0}, "corner": "slow"},
    }
    run_analysis_suite(spec, analyses=analyses)

    slow = CORNERS["slow"]
    assert seen == [
        ("ac", slow),
        ("noise", slow, True),
        ("pss", slow),
        ("pac", slow, slow),
        ("pnoise", slow, slow),
    ]


def test_dispatch_forwards_integration_method(monkeypatch):
    """analyses.pss.integration_method must reach pss_solve — and hence the
    shared orbit that PAC/PNoise linearize around.  When unset, the dispatch
    injects nothing and pss_solve's own default (gear2) governs."""
    spec = load_circuit_json("examples/periodic_rc.json")
    seen = {}

    def fake_pss_solve(*args, **kwargs):
        seen["method"] = kwargs.get("integration_method")
        return {"converged": True, "period": args[2], "corner": kwargs.get("corner")}

    monkeypatch.setattr(dispatch_mod, "pss_solve", fake_pss_solve)

    run_analysis_suite(spec, analyses={"pss": {"integration_method": "be",
                                               "max_shooting_iters": 0}})
    assert seen["method"] == "be"

    seen.clear()
    run_analysis_suite(spec, analyses={"pss": {"max_shooting_iters": 0}})
    assert seen.get("method") is None  # not forwarded -> pss_solve default applies


def test_dispatch_forwards_adaptive_pss_options(monkeypatch):
    spec = load_circuit_json("examples/periodic_rc.json")
    seen = {}

    def fake_pss_solve(*args, **kwargs):
        seen.update(kwargs)
        return {"converged": True, "period": args[2], "corner": kwargs.get("corner")}

    monkeypatch.setattr(dispatch_mod, "pss_solve", fake_pss_solve)

    run_analysis_suite(
        spec,
        analyses={
            "pss": {
                "integration_method": "gear2",
                "adaptive": True,
                "adaptive_reltol": 1e-4,
                "adaptive_vabstol": 1e-6,
                "adaptive_iabstol": 1e-12,
                "adaptive_max_steps": 1234,
                "adaptive_h0": 1e-6,
                "adaptive_freeze_factor": 5.0,
                "cap_mode": "average",
                "max_shooting_iters": 0,
            }
        },
    )

    assert seen["adaptive"] is True
    assert seen["adaptive_config"].reltol == 1e-4
    assert seen["adaptive_config"].vabstol == 1e-6
    assert seen["adaptive_config"].iabstol == 1e-12
    assert seen["adaptive_config"].max_steps == 1234
    assert seen["adaptive_config"].h0 == 1e-6
    assert seen["adaptive_config"].freeze_factor == 5.0
    assert seen["cap_mode"] == "average"


def test_dispatch_forwards_pss_registry_options_and_top_level_grid(monkeypatch):
    spec = load_circuit_json("examples/periodic_rc.json")
    seen = {}

    def fake_pss_solve(*args, **kwargs):
        seen.update(kwargs)
        return {"converged": True, "period": args[2], "corner": kwargs.get("corner")}

    monkeypatch.setattr(dispatch_mod, "pss_solve", fake_pss_solve)

    run_analysis_suite(
        spec,
        analyses={
            "pss": {
                "n_points": 17,
                "analytic_jacobian": False,
                "physical_factor": 3.0,
                "max_stabilization_periods": 11,
                "levenberg_marquardt": False,
                "max_shooting_iters": 0,
            }
        },
    )

    assert len(seen["tgrid"]) == 17
    assert seen["analytic_jacobian"] is False
    assert seen["physical_factor"] == 3.0
    assert seen["max_stabilization_periods"] == 11
    assert seen["levenberg_marquardt"] is False


def test_dispatch_forwards_pac_algorithm_options(monkeypatch):
    spec = load_circuit_json("examples/periodic_rc.json")
    seen = {}

    def fake_pss_solve(*args, **kwargs):
        return {"converged": True, "period": args[2], "corner": kwargs.get("corner")}

    def fake_pac_solve(*_args, **kwargs):
        seen.update(kwargs)
        return {"gains": np.ones(1), "response": np.ones(1, dtype=complex)}

    monkeypatch.setattr(dispatch_mod, "pss_solve", fake_pss_solve)
    monkeypatch.setattr(dispatch_mod, "pac_solve", fake_pac_solve)

    run_analysis_suite(
        spec,
        analyses={
            "pac": {
                "freqs": [100.0],
                "input_drive": {"vin": 1.0},
                "analytic": False,
                "max_sideband": 7,
                "n_period_samples": 96,
                "time_domain": True,
                "td_integration": "be",
                "td_n_period_samples": 128,
                "pacmag": 2.0,
            }
        },
    )

    assert seen["analytic"] is False
    assert seen["max_sideband"] == 7
    assert seen["n_period_samples"] == 96
    assert seen["time_domain"] is True
    assert seen["td_integration"] == "be"
    assert seen["td_n_period_samples"] == 128
    assert seen["pacmag"] == 2.0


def test_dispatch_rejects_mixed_pss_and_pac_corners(monkeypatch):
    spec = load_circuit_json("examples/periodic_rc.json")

    def fake_pss_solve(*args, **kwargs):
        return {"converged": True, "period": args[2], "corner": kwargs.get("corner")}

    monkeypatch.setattr(dispatch_mod, "pss_solve", fake_pss_solve)

    with pytest.raises(ValueError, match="corner mismatch"):
        run_analysis_suite(
            spec,
            analyses={
                "pss": {"corner": "typical", "max_shooting_iters": 0},
                "pac": {
                    "freqs": [100.0],
                    "input_drive": {"vin": 1.0},
                    "corner": "slow",
                },
            },
        )


def test_dispatch_rejects_unknown_analysis_option(monkeypatch):
    # A typo'd option key (max_sidebands for max_sideband) must hard-error instead
    # of being silently ignored and run with the default.
    spec = load_circuit_json("examples/periodic_rc.json")

    def fake_pss_solve(*args, **kwargs):
        return {"converged": True, "period": args[2], "corner": kwargs.get("corner")}

    monkeypatch.setattr(dispatch_mod, "pss_solve", fake_pss_solve)

    with pytest.raises(ValueError, match="Unknown option") as exc:
        run_analysis_suite(
            spec,
            analyses={
                "pss": {"max_shooting_iters": 0},
                "pnoise": {
                    "freqs": [100.0],
                    "input_drive": {"vin": 1.0},
                    "max_sidebands": 5,   # typo: should be max_sideband
                },
            },
        )
    msg = str(exc.value)
    assert "max_sidebands" in msg          # names the offending residual key
    assert "'pnoise'" in msg               # names the analysis
    assert "max_sideband" in msg           # lists the correct key to help fix it


def test_dispatch_accepts_mixed_dispatch_and_solver_keys(monkeypatch):
    # A full cfg mixing dispatch-consumed keys (freqs/input_drive/corner) with
    # forwarded solver options must pass validation unchanged.
    spec = load_circuit_json("examples/periodic_rc.json")

    def fake_pss_solve(*args, **kwargs):
        return {"converged": True, "period": args[2], "corner": kwargs.get("corner")}

    def fake_pnoise_solve(*_args, **kwargs):
        return {"out_psd": np.ones(1), "irn_psd": np.ones(1)}

    monkeypatch.setattr(dispatch_mod, "pss_solve", fake_pss_solve)
    monkeypatch.setattr(dispatch_mod, "pnoise_solve", fake_pnoise_solve)

    # Should not raise.
    run_analysis_suite(
        spec,
        analyses={
            "pss": {"corner": "typical", "max_shooting_iters": 0,
                    "jacobian_reuse": True},
            "pnoise": {
                "freqs": [100.0],
                "input_drive": {"vin": 1.0},
                "corner": "typical",
                "band": [1.0, 100.0],
                "max_sideband": 4,
                "n_period_samples": 32,
                "lti_fast_path": True,
            },
        },
    )


def test_known_keys_cover_dispatch_and_solver_options():
    # ac/noise carry no solver registry -> their legal keys are the dispatch set.
    assert known_keys("ac") == frozenset({"freqs", "corner"})
    assert known_keys("noise") == frozenset({"freqs", "corner", "band"})
    # transient's registry keys plus the dispatch-only signed_devices.
    tr = known_keys("transient")
    assert {"tgrid", "duration", "tstop", "n_points", "corner",
            "signed_devices"} <= tr
    # pnoise: every consumed key lives in the registry (no dispatch extras).
    pn = known_keys("pnoise")
    assert {"freqs", "input_drive", "pss", "corner", "band",
            "max_sideband"} <= pn


def test_validate_analysis_cfg_direct():
    validate_analysis_cfg("ac", {"freqs": [1.0, 10.0], "corner": "slow"})  # no raise
    with pytest.raises(ValueError, match="Unknown option"):
        validate_analysis_cfg("ac", {"freq": [1.0]})  # typo: freq -> freqs


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
