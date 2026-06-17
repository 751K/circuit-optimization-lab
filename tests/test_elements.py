"""Resistor / capacitor / ideal-current-source element tests.

Each two-terminal element is checked against a closed-form result where possible
(resistor divider, current-source load, resistor thermal noise, RC relaxation),
plus a JSON round-trip that exercises all four analyses, and loader validation.
"""
import numpy as np
import pytest

from core.ac_solver import ac_solve
from core.circuit_loader import circuit_from_dict, load_circuit_json
from core.noise_solver import _KB, _TEMP, band_rms, noise_analysis
from core.topology import Topology
from core.transient_solver import transient


def test_resistor_divider_dc():
    # OUT between R1 (VDD-OUT) and R2 (OUT-GND): Vout = VDD * R2/(R1+R2).
    topo = Topology(solved=["OUT"], devices=[], rails={"VDD": "VDD", "GND": 0.0},
                    outputs=("OUT",),
                    resistors=[("R1", "VDD", "OUT", 1e3), ("R2", "OUT", "GND", 3e3)])
    ac = ac_solve({}, {"VDD": 10.0}, np.array([1.0]), topo=topo)
    assert ac["dc_op"]["OUT"] == pytest.approx(10.0 * 3e3 / 4e3, rel=1e-6)


def test_current_source_into_resistor_dc():
    # IB injects I into OUT, RL drains it to ground: Vout = I * RL.
    topo = Topology(solved=["OUT"], devices=[], rails={"VDD": "VDD", "GND": 0.0},
                    outputs=("OUT",), resistors=[("RL", "OUT", "GND", 2e3)],
                    isources=[("IB", "VDD", "OUT", 1e-3)])
    ac = ac_solve({}, {"VDD": 10.0}, np.array([1.0]), topo=topo)
    assert ac["dc_op"]["OUT"] == pytest.approx(1e-3 * 2e3, rel=1e-6)


def test_capacitor_element_matches_load_cap_in_ac():
    common = dict(solved=["OUT"], devices=[("MPU", "OUT", "IN", "VDD")],
                  rails={"VDD": "VDD", "GND": 0.0, "IN": "VIN"}, outputs=("OUT",),
                  input_drives={"MPU": 1.0}, resistors=[("RL", "OUT", "GND", 4e6)])
    topo_lc = Topology(load_caps=[("OUT", "GND", 2e-12)], **common)
    topo_cap = Topology(capacitors=[("CL", "OUT", "GND", 2e-12)], **common)
    sizes, bias = {"MPU": (2000, 80)}, {"VDD": 40.0, "VIN": 25.0}
    freqs = np.logspace(0, 6, 41)
    g_lc = ac_solve(sizes, bias, freqs, topo=topo_lc)["gains"]
    g_cap = ac_solve(sizes, bias, freqs, topo=topo_cap)["gains"]
    assert np.allclose(g_lc, g_cap, rtol=1e-12, atol=0)


@pytest.mark.filterwarnings("ignore:divide by zero")  # passive net: no gain -> IRN is inf
def test_resistor_thermal_noise_psd():
    # Single resistor OUT-GND is the output: output PSD = |Z|^2 * 4kT/R = R^2*(4kT/R) = 4kTR.
    R = 1e3
    topo = Topology(solved=["OUT"], devices=[], rails={"GND": 0.0},
                    outputs=("OUT",), resistors=[("R1", "OUT", "GND", R)])
    nz = noise_analysis({}, {}, np.array([1.0, 10.0, 100.0]), topo=topo)
    expected = 4.0 * _KB * _TEMP * R
    assert nz["out_psd"][0] == pytest.approx(expected, rel=1e-9)
    assert np.allclose(nz["out_psd"], expected, rtol=1e-9)   # white (frequency-flat)
    assert "R1" in nz["dev_psd"]


def test_rc_current_source_transient_relaxes():
    # IB drives an RC node from 0 to I*R with time constant RC.
    R, C, I = 1e6, 1e-9, 2e-6
    topo = Topology(solved=["OUT"], devices=[], rails={"VDD": "VDD", "GND": 0.0},
                    outputs=("OUT",), resistors=[("R1", "OUT", "GND", R)],
                    capacitors=[("C1", "OUT", "GND", C)],
                    isources=[("IB", "VDD", "OUT", I)])
    tg = np.linspace(0, 1e-2, 400)          # ~10 RC
    tr = transient({}, {"VDD": 10.0}, tg, topo=topo, V0=np.array([0.0]))
    assert tr["nfail"] == 0
    assert tr["output"][0] == pytest.approx(0.0, abs=1e-9)
    assert tr["output"][-1] == pytest.approx(I * R, rel=2e-3)
    # rises monotonically toward the steady value
    assert np.all(np.diff(tr["output"]) > 0)


def test_resistor_load_example_runs_all_analyses():
    spec = load_circuit_json("examples/resistor_load_stage.json")
    freqs = np.logspace(0, 5, 41)

    ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
    assert ac is not None and np.isfinite(ac["dc_op"]["OUT"])
    assert 0.0 < ac["dc_op"]["OUT"] < spec.bias["VDD"]

    nz = noise_analysis(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
    assert "RL" in nz["dev_psd"]                       # resistor noise is accounted for
    assert np.all(np.isfinite(nz["out_psd"])) and nz["out_psd"][0] > 0.0

    t = np.linspace(0, 2e-3, 120)
    vin = np.full_like(t, spec.bias["VIN"]) + np.where(t >= 5e-4, 0.1, 0.0)
    tr = transient(spec.sizes, spec.bias, t, topo=spec.topology, nf=spec.nf,
                   inputs={"vin": vin})
    assert tr["nfail"] == 0
    assert abs(tr["output"][-1] - tr["output"][0]) > 1e-6


def test_loader_rejects_unknown_resistor_node():
    bad = {"solved": ["OUT"], "rails": {"VDD": "VDD", "GND": 0.0},
           "devices": [{"name": "M1", "drain": "OUT", "gate": "VDD", "source": "VDD",
                        "W": 1000, "L": 80}],
           "resistors": [{"name": "RL", "a": "OUT", "b": "NOPE", "R": 1e3}],
           "outputs": ["OUT"]}
    with pytest.raises(ValueError, match="unknown node"):
        circuit_from_dict(bad)


def test_loader_rejects_nonpositive_resistor():
    bad = {"solved": ["OUT"], "rails": {"VDD": "VDD", "GND": 0.0},
           "devices": [{"name": "M1", "drain": "OUT", "gate": "VDD", "source": "VDD",
                        "W": 1000, "L": 80}],
           "resistors": [["RL", "OUT", "GND", 0.0]],
           "outputs": ["OUT"]}
    with pytest.raises(ValueError, match="must be positive"):
        circuit_from_dict(bad)


def test_loader_accepts_tuple_current_source():
    data = {"solved": ["OUT"], "rails": {"VDD": "VDD", "GND": 0.0},
            "devices": [{"name": "M1", "drain": "OUT", "gate": "VDD", "source": "VDD",
                         "W": 1000, "L": 80}],
            "current_sources": [["IB", "OUT", "GND", -1e-6]],   # negative current allowed
            "outputs": ["OUT"]}
    spec = circuit_from_dict(data)
    assert spec.topology.isources == [("IB", "OUT", "GND", -1e-6)]
