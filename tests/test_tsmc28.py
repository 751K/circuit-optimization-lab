"""TSMC28HPC+ adapter semantics and optional licensed-PDK integration."""
from __future__ import annotations

import os
import numpy as np
import pytest


def _inverter_spec(*, oracle=False):
    from circuitopt.circuit_loader import circuit_from_dict

    if oracle:
        import circuitopt.tsmc28_model  # noqa: F401 - registers oracle adapter
    prefix = "tsmc28hpcp_ngspice" if oracle else "tsmc28hpcp"
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
            "MN": {"type": f"{prefix}.nmos"},
            "MP": {"type": f"{prefix}.pmos", "vb": 0.9},
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


@pytest.mark.ngspice_oracle
def test_adapter_renders_library_closure_and_macro_instances(tmp_path, monkeypatch):
    _fake_model_dir(tmp_path, monkeypatch)
    from circuitopt.ngspice_transient import render_ngspice_transient_netlist

    spec = _inverter_spec(oracle=True)
    rendered = render_ngspice_transient_netlist(
        spec.sizes, spec.bias, np.array([0.0, 1e-9]), topo=spec.topology,
        output_path=str(tmp_path / "wave.dat"), nf={"MN": 2, "MP": 4},
        inputs={"vin": np.array([0.0, 0.9])}, corner="SF",
        model_types=spec.model_types, device_kwargs=spec.device_kwargs,
        mismatch={"MN": 0.01}, op_devices=("MN", "MP"),
    )
    deck = rendered.netlist
    assert [line.rsplit(" ", 1)[-1] for line in deck.splitlines() if line.startswith(".lib")] == [
        "setup", "sf", "global", "total", "stat"]
    assert "XMN n_OUT n_gate_MN 0 0 nch_mac w=1u l=0.029999999999999999u nf=2 _delvto=0.01" in deck
    assert "XMP n_OUT n_gate_MP n_VDD n_VDD pch_mac" in deck
    assert rendered.command_args == ("-D", "ngbehavior=hsa")
    assert rendered.process == "TSMC28HPC+"
    assert rendered.op_vectors == tuple(
        (name, variable)
        for name in ("MN", "MP")
        for variable in ("vds", "vgs", "vdsat", "id", "gm", "gds")
    )
    assert "@m.xmn.main[vds]" in deck
    assert "@m.xmp.main[gds]" in deck


@pytest.mark.ngspice_oracle
def test_device_multiplicity_renders_m_parameter(tmp_path, monkeypatch):
    """A device dict "M" field must reach the deck as ``m=<int>`` (via
    Topology.device_mult -> render_devices), and M=1/absent must stay
    byte-identical to a deck that never heard of multiplicity."""
    _fake_model_dir(tmp_path, monkeypatch)
    from circuitopt.circuit_loader import circuit_from_dict
    from circuitopt.ngspice_transient import render_ngspice_transient_netlist

    def render(with_m):
        raw = {
            "name": "tsmc28_mult", "solved": ["OUT"],
            "rails": {"VDD": "VDD", "GND": 0.0, "IN": 0.0},
            "devices": [
                {"name": "MN", "drain": "OUT", "gate": "IN", "source": "GND",
                 "W": 1.0, "L": 0.03, **({"M": 3} if with_m else {})},
                {"name": "MP", "drain": "OUT", "gate": "IN", "source": "VDD",
                 "W": 2.0, "L": 0.03, **({"M": 1} if with_m else {})},
            ],
            "models": {"MN": {"type": "tsmc28hpcp_ngspice.nmos"},
                       "MP": {"type": "tsmc28hpcp_ngspice.pmos", "vb": 0.9}},
            "bias": {"VDD": 0.9}, "outputs": ["OUT"],
            "load_caps": [["OUT", "GND", 2e-15]],
            "transient_inputs": {"MN": "vin", "MP": "vin"},
            "dc_guesses": [{"OUT": 0.9}],
        }
        spec = circuit_from_dict(raw)
        return render_ngspice_transient_netlist(
            spec.sizes, spec.bias, np.array([0.0, 1e-9]), topo=spec.topology,
            output_path=str(tmp_path / "wave.dat"),
            inputs={"vin": np.array([0.0, 0.9])},
            model_types=spec.model_types, device_kwargs=spec.device_kwargs,
        ).netlist

    with_m, without_m = render(True), render(False)
    assert "XMN n_OUT n_gate_MN 0 0 nch_mac w=1u l=0.029999999999999999u nf=1 m=3" in with_m
    mp_lines = [ln for ln in with_m.splitlines() if ln.startswith("XMP")]
    assert mp_lines and " m=" not in mp_lines[0]          # M=1 renders bare
    assert without_m == with_m.replace(" m=3", "")        # only delta is the m=


@pytest.mark.ngspice_oracle
def test_transient_rejects_unknown_operating_point_device(tmp_path, monkeypatch):
    _fake_model_dir(tmp_path, monkeypatch)
    from circuitopt.ngspice_transient import render_ngspice_transient_netlist

    spec = _inverter_spec(oracle=True)
    with pytest.raises(ValueError, match="op_devices references unknown devices: MISSING"):
        render_ngspice_transient_netlist(
            spec.sizes, spec.bias, np.array([0.0, 1e-9]), topo=spec.topology,
            output_path=str(tmp_path / "wave.dat"),
            inputs={"vin": np.array([0.0, 0.9])}, model_types=spec.model_types,
            device_kwargs=spec.device_kwargs, op_devices=("MISSING",),
        )


@pytest.mark.ngspice_oracle
def test_transient_uic_renders_initial_conditions(tmp_path, monkeypatch):
    _fake_model_dir(tmp_path, monkeypatch)
    from circuitopt.ngspice_transient import render_ngspice_transient_netlist

    spec = _inverter_spec(oracle=True)
    rendered = render_ngspice_transient_netlist(
        spec.sizes, spec.bias, np.array([0.0, 1e-9]), topo=spec.topology,
        output_path=str(tmp_path / "wave.dat"),
        inputs={"vin": np.array([0.0, 0.9])}, model_types=spec.model_types,
        device_kwargs=spec.device_kwargs, V0=np.array([0.3]), uic=True,
    )
    assert ".ic v(n_OUT)=0.29999999999999999" in rendered.netlist
    assert " uic\n" in rendered.netlist
    assert ".nodeset " not in rendered.netlist


@pytest.mark.ngspice_oracle
def test_adapter_rejects_mixed_process_deck(tmp_path, monkeypatch):
    _fake_model_dir(tmp_path, monkeypatch)
    from circuitopt.ngspice_transient import render_ngspice_transient_netlist

    spec = _inverter_spec(oracle=True)
    models = dict(spec.model_types)
    models["MP"] = "freepdk45.pmos"
    with pytest.raises(NotImplementedError, match="one ngspice process adapter"):
        render_ngspice_transient_netlist(
            spec.sizes, spec.bias, np.array([0.0, 1e-9]), topo=spec.topology,
            output_path=str(tmp_path / "wave.dat"), inputs={"vin": np.array([0.0, 0.9])},
            model_types=models, device_kwargs=spec.device_kwargs)


@pytest.mark.ngspice_oracle
def test_tsmc_capacitance_signs_are_normalized():
    from circuitopt.tsmc28_model import TSMC28HPCP_ADAPTER

    raw = np.array([-2e-16, 3e-17])
    np.testing.assert_array_equal(
        TSMC28HPCP_ADAPTER.normalize_op_data("cgd", raw),
        np.array([-2e-16, -3e-17]))


def test_real_tsmc28_inverter_transient_when_pdk_is_configured(monkeypatch):
    from circuitopt.toolchain import tsmc28_model_dir

    model = os.path.join(tsmc28_model_dir(), "cln28hpcp_1d8_elk_v1d0_2p2.l")
    if not os.path.isfile(model):
        pytest.skip("set TSMC28_MODEL_DIR for licensed-PDK integration")

    from circuitopt.transient_solver import transient

    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/an/executable")
    spec = _inverter_spec()
    tgrid = np.linspace(0.0, 0.4e-9, 81)
    vin = np.where(tgrid < 0.1e-9, 0.0, 0.9)
    result = transient(
        spec.sizes, spec.bias, tgrid, binding=spec.binding(), inputs={"vin": vin},
        corner="tt", integration_method="gear2", max_step=1e-12)
    out = result["nodes"]["OUT"]
    assert result["backend"] == "bsim4_native"
    assert result["bsim4_native_transient"] is True
    # v2.0.0 (R7): rust is the only engine; the retired numba flags are gone.
    assert "numba_grid_solver" not in result
    assert "bsim4_numba_transient" not in result
    assert result["bsim4_rust_transient"] is True
    assert result["nfail"] == 0
    assert out[0] > 0.85 and out[-1] < 0.05
    assert np.all(np.isfinite(out))
