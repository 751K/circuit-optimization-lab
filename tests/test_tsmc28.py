"""TSMC28HPC+ adapter semantics and optional licensed-PDK integration."""
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pytest


def _inverter_spec():
    from circuitopt.circuit_loader import circuit_from_dict

    return circuit_from_dict({
        "name": "tsmc28_inverter",
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


def _fake_model_dir(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "hspice"
    model_dir.mkdir(parents=True)
    (model_dir / "cln28hpcp_1d8_elk_v1d0_2p2.l").write_text("* fake\n", encoding="ascii")
    monkeypatch.setenv("TSMC28_MODEL_DIR", str(model_dir))
    return model_dir


def test_pdk_is_registered():
    from circuitopt import list_pdks, registered_models

    assert "tsmc28hpcp" in list_pdks()
    assert {"tsmc28hpcp.nmos", "tsmc28hpcp.pmos"} <= set(registered_models())


def test_model_root_resolution_handles_outer_delivery(tmp_path, monkeypatch):
    from circuitopt.toolchain import tsmc28_model_dir

    monkeypatch.delenv("TSMC28_MODEL_DIR", raising=False)
    root = tmp_path / "delivery"
    model_dir = root / "iPDK_version" / "models" / "hspice"
    model_dir.mkdir(parents=True)
    (model_dir / "cln28hpcp_1d8_elk_v1d0_2p2.l").write_text("* fake\n", encoding="ascii")
    monkeypatch.setenv("TSMC28_PDK_ROOT", str(root))
    assert tsmc28_model_dir() == str(model_dir)


def test_adapter_renders_library_closure_and_macro_instances(tmp_path, monkeypatch):
    _fake_model_dir(tmp_path, monkeypatch)
    from circuitopt.ngspice_transient import render_ngspice_transient_netlist

    spec = _inverter_spec()
    rendered = render_ngspice_transient_netlist(
        spec.sizes, spec.bias, np.array([0.0, 1e-9]), topo=spec.topology,
        output_path=str(tmp_path / "wave.dat"), nf={"MN": 2, "MP": 4},
        inputs={"vin": np.array([0.0, 0.9])}, corner="SF",
        model_types=spec.model_types, device_kwargs=spec.device_kwargs,
        mismatch={"MN": 0.01},
    )
    deck = rendered.netlist
    assert [line.rsplit(" ", 1)[-1] for line in deck.splitlines() if line.startswith(".lib")] == [
        "setup", "sf", "global", "total", "stat"]
    assert "XMN n_OUT n_gate_MN 0 0 nch_mac w=1u l=0.029999999999999999u nf=2 _delvto=0.01" in deck
    assert "XMP n_OUT n_gate_MP n_VDD n_VDD pch_mac" in deck
    assert rendered.command_args == ("-D", "ngbehavior=hsa")
    assert rendered.process == "TSMC28HPC+"


def test_adapter_rejects_mixed_process_deck(tmp_path, monkeypatch):
    _fake_model_dir(tmp_path, monkeypatch)
    from circuitopt.ngspice_transient import render_ngspice_transient_netlist

    spec = _inverter_spec()
    models = dict(spec.model_types)
    models["MP"] = "freepdk45.pmos"
    with pytest.raises(NotImplementedError, match="one ngspice process adapter"):
        render_ngspice_transient_netlist(
            spec.sizes, spec.bias, np.array([0.0, 1e-9]), topo=spec.topology,
            output_path=str(tmp_path / "wave.dat"), inputs={"vin": np.array([0.0, 0.9])},
            model_types=models, device_kwargs=spec.device_kwargs)


def test_tsmc_capacitance_signs_are_normalized():
    from circuitopt.tsmc28_model import TSMC28HPCP_ADAPTER

    raw = np.array([-2e-16, 3e-17])
    np.testing.assert_array_equal(
        TSMC28HPCP_ADAPTER.normalize_op_data("cgd", raw),
        np.array([-2e-16, -3e-17]))


def test_real_tsmc28_inverter_transient_when_pdk_is_configured():
    from circuitopt.ngspice_char import ngspice_binary
    from circuitopt.toolchain import tsmc28_model_dir

    model = os.path.join(tsmc28_model_dir(), "cln28hpcp_1d8_elk_v1d0_2p2.l")
    if not os.path.isfile(model) or ngspice_binary() is None:
        pytest.skip("set TSMC28_MODEL_DIR and install ngspice for licensed-PDK integration")

    from circuitopt.transient_solver import transient

    spec = _inverter_spec()
    tgrid = np.linspace(0.0, 0.4e-9, 81)
    vin = np.where(tgrid < 0.1e-9, 0.0, 0.9)
    result = transient(
        spec.sizes, spec.bias, tgrid, binding=spec.binding(), inputs={"vin": vin},
        corner="tt", integration_method="gear2", max_step=1e-12)
    out = result["nodes"]["OUT"]
    assert result["process"] == "TSMC28HPC+"
    assert out[0] > 0.85 and out[-1] < 0.05
    assert np.all(np.isfinite(out))


def test_real_tsmc28_sar_adc_smoke_when_pdk_is_configured():
    from circuitopt.ngspice_char import ngspice_binary
    from circuitopt.toolchain import tsmc28_model_dir

    model = os.path.join(tsmc28_model_dir(), "cln28hpcp_1d8_elk_v1d0_2p2.l")
    if not os.path.isfile(model) or ngspice_binary() is None:
        pytest.skip("set TSMC28_MODEL_DIR and install ngspice for licensed-PDK integration")

    from circuitopt.circuit_loader import circuit_from_dict
    from circuitopt.sar import run_sar_conversion

    source = Path(__file__).resolve().parents[1] / "examples" / "freepdk45_sar3.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["name"] = "tsmc28_sar3_smoke"
    for model_config in config["models"].values():
        polarity = model_config["type"].rsplit(".", 1)[-1]
        model_config["type"] = f"tsmc28hpcp.{polarity}"
        if polarity == "pmos":
            model_config["vb"] = 0.9
    config["bias"].update(VDD=0.9, VCM=0.45, VBIAS=0.45)
    config["adc"].update(
        vref=0.9, input_common_mode=0.45, comparator_threshold=0.45,
        points_per_period=20, edge_time=1e-9)
    spec = circuit_from_dict(config)
    result = run_sar_conversion(spec, 0.45, corner="tt")
    assert 0 <= result["code"] < 8
    assert result["transient"]["process"] == "TSMC28HPC+"
    assert np.isfinite(result["total_power_w"])
