"""Full-charge FreePDK45 transient through the native BSIM4 backend."""
import json
import os
import shutil

import numpy as np
import pytest

from circuitopt.toolchain import pdk_root


PDK_ROOT = pdk_root()
_CARD = os.path.join(PDK_ROOT, "freepdk45", "models_nom", "NMOS_VTG.inc")
pytestmark = [
    pytest.mark.skipif(
        not os.path.isfile(_CARD), reason="FreePDK45 cards not present"),
    pytest.mark.skipif(
        not any(shutil.which(name) for name in ("clang", "cc", "gcc")),
        reason="native BSIM4 tests require a C99 compiler"),
]

_ROOT = os.path.dirname(os.path.dirname(__file__))
_CFG = os.path.join(_ROOT, "examples", "freepdk45_5t_ota.json")


def _spec(*, driven=False, analyses=False):
    from circuitopt.circuit_loader import circuit_from_dict
    with open(_CFG, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if driven:
        cfg["transient_inputs"] = {"M1": "vip", "M2": "vin"}
    if analyses:
        cfg["periodic"] = {
            "period": 100e-9,
            "inputs": {"vip": 0.56, "vin": 0.54},
        }
        cfg["analyses"]["transient"] = {"tstop": 100e-9, "n_points": 101}
    return circuit_from_dict(cfg)


@pytest.mark.ngspice_oracle
def test_render_contains_full_bsim4_devices_and_charge_capable_options(tmp_path):
    from circuitopt.ngspice_transient import render_freepdk45_transient_netlist
    spec = _spec(driven=True)
    tgrid = np.linspace(0.0, 10e-9, 11)
    rendered = render_freepdk45_transient_netlist(
        spec.sizes, spec.bias, tgrid, topo=spec.topology,
        output_path=str(tmp_path / "wave.dat"),
        nf=spec.nf, inputs={"vip": np.full(11, 0.56), "vin": np.full(11, 0.54)},
        model_types=spec.model_types, device_kwargs=spec.device_kwargs,
    )
    deck = rendered.netlist
    assert 'models_nom/NMOS_VTG.inc"' in deck
    assert 'models_nom/PMOS_VTG.inc"' in deck
    assert ".options temp=27 method=gear maxord=1" in deck
    assert "M1 n_n1 n_gate_M1 n_tail 0 NMOS_VTG" in deck
    assert "M3 n_n1 n_n1 n_VDD n_VDD PMOS_VTG" in deck
    assert "Vgate_M1" in deck and "PWL(" in deck
    assert rendered.node_names == ("tail", "n1", "vout")


@pytest.mark.ngspice_oracle
def test_render_accepts_explicit_ngspice_oracle_aliases(tmp_path):
    from circuitopt.ngspice_transient import render_freepdk45_transient_netlist

    spec = _spec()
    model_types = {
        name: model_type.replace("freepdk45.", "freepdk45_ngspice.")
        for name, model_type in spec.model_types.items()
    }
    rendered = render_freepdk45_transient_netlist(
        spec.sizes,
        spec.bias,
        np.asarray((0.0, 1e-9)),
        topo=spec.topology,
        output_path=str(tmp_path / "wave.dat"),
        nf=spec.nf,
        model_types=model_types,
        device_kwargs=spec.device_kwargs,
    )
    assert "NMOS_VTG.inc" in rendered.netlist
    assert "PMOS_VTG.inc" in rendered.netlist


def test_ota_dc_hold_routes_to_native_bsim4():
    from circuitopt.transient_solver import transient
    spec = _spec()
    tgrid = np.linspace(0.0, 20e-9, 21)
    result = transient(spec.sizes, spec.bias, tgrid, binding=spec.binding())
    assert result["backend"] == "bsim4_native"
    assert result["bsim4_native_transient"] is True
    assert result["nfail"] == 0
    assert np.ptp(result["nodes"]["vout"]) < 1e-10
    assert 0.4 < result["nodes"]["vout"][0] < 0.6
    assert "rail:VDD" in result["branch_currents"]


def test_ota_differential_step_and_supply_current():
    from circuitopt.transient_solver import transient
    spec = _spec(driven=True)
    tgrid = np.linspace(0.0, 100e-9, 201)
    vip = np.where(tgrid < 20e-9, 0.55, 0.56)
    vin = np.where(tgrid < 20e-9, 0.55, 0.54)
    result = transient(
        spec.sizes, spec.bias, tgrid, binding=spec.binding(),
        inputs={"vip": vip, "vin": vin}, integration_method="gear2",
    )
    vout = result["nodes"]["vout"]
    ivdd = result["branch_currents"]["rail:VDD"]
    assert vout[-1] - vout[0] > 0.3
    assert np.all(np.isfinite(vout)) and np.all(np.isfinite(ivdd))
    assert np.mean(ivdd) < 0.0  # ideal-source convention: delivered current is negative


def test_nmos_rc_step_has_finite_charge_settling():
    from circuitopt.circuit_loader import circuit_from_dict
    from circuitopt.transient_solver import transient
    spec = circuit_from_dict({
        "name": "fp45_nmos_rc",
        "solved": ["OUT"],
        "rails": {"VDD": "VDD", "GND": 0.0, "GATE": 0.0},
        "devices": [{
            "name": "M1", "drain": "OUT", "gate": "GATE", "source": "GND",
            "W": 0.5, "L": 0.05,
        }],
        "models": {"M1": {"type": "freepdk45.nmos"}},
        "bias": {"VDD": 1.0},
        "outputs": ["OUT"],
        "resistors": [["RLOAD", "VDD", "OUT", 10e3]],
        "capacitors": [["CLOAD", "OUT", "GND", 1e-12]],
        "transient_inputs": {"M1": "gate"},
        "dc_guesses": [{"OUT": 1.0}],
    })
    tgrid = np.linspace(0.0, 100e-9, 501)
    gate = np.where(tgrid < 10e-9, 0.0, 0.8)
    result = transient(
        spec.sizes, spec.bias, tgrid, binding=spec.binding(), inputs={"gate": gate},
        integration_method="gear2", max_step=0.2e-9,
    )
    out = result["nodes"]["OUT"]
    edge = int(np.searchsorted(tgrid, 10e-9))
    assert out[0] > 0.99 and out[-1] < 0.2
    assert out[edge + 1] > out[-1] + 0.2  # finite C prevents an instantaneous jump
    assert np.all(np.diff(out[edge:]) <= 2e-5)


def test_json_transient_dispatch_uses_native_bsim4_backend():
    from circuitopt.analysis_dispatch import run_analysis_suite
    spec = _spec(driven=True, analyses=True)
    result = run_analysis_suite(spec, selected=["transient"])["transient"]
    assert result["backend"] == "bsim4_native"
    assert len(result["t"]) == 101
    assert result["nodes"]["vout"][-1] > 0.8


def test_native_backend_supports_all_controlled_sources():
    from circuitopt.circuit_loader import circuit_from_dict
    from circuitopt.transient_solver import transient
    spec = circuit_from_dict({
        "name": "fp45_controlled_sources",
        "solved": ["VIN", "V2", "V3", "V4", "V5", "DUMMY"],
        "rails": {"VDD": "VDD", "GND": 0.0},
        "devices": [{
            "name": "M0", "drain": "DUMMY", "gate": "GND", "source": "GND",
            "W": 0.1, "L": 0.05,
        }],
        "models": {"M0": {"type": "freepdk45.nmos"}},
        "bias": {"VDD": 1.0},
        "outputs": ["V2"],
        "vsources": [["V1", "VIN", "GND", "vin"]],
        "vcvs": [["E1", "V2", "GND", "VIN", "GND", 2.0]],
        "vccs": [["G1", "V3", "GND", "VIN", "GND", 1e-3]],
        "cccs": [["F1", "V4", "GND", "V1", 2.0]],
        "ccvs": [["H1", "V5", "GND", "V1", -1000.0]],
        "resistors": [
            ["RIN", "VIN", "GND", 1000.0],
            ["R3", "V3", "GND", 1000.0],
            ["R4", "V4", "GND", 500.0],
            ["RD", "VDD", "DUMMY", 1e6],
        ],
    })
    tgrid = np.linspace(0.0, 10e-9, 21)
    vin = np.linspace(0.1, 0.2, len(tgrid))
    result = transient(
        spec.sizes, spec.bias, tgrid, binding=spec.binding(), inputs={"vin": vin})
    # Dynamic ideal sources are enforced from the first integration step; sample
    # zero remains the DC seed supplied by the circuit initializer.
    np.testing.assert_allclose(result["nodes"]["V2"][1:], 2.0 * vin[1:], atol=1e-8)
    np.testing.assert_allclose(result["nodes"]["V3"][1:], vin[1:], atol=1e-8)
    np.testing.assert_allclose(result["nodes"]["V4"][1:], -vin[1:], atol=1e-8)
    np.testing.assert_allclose(result["nodes"]["V5"][1:], vin[1:], atol=1e-8)
