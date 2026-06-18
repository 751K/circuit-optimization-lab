import numpy as np

from core.compiled_topology import CompiledTopology, TERM_INPUT, TERM_SOLVED
from core.topology import Topology


def test_compiled_dc_residuals_match_topology_passive_kcl():
    topo = Topology(
        solved=["OUT"],
        devices=[],
        rails={"VDD": "VDD", "GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "VDD", "OUT", 1e3), ("R2", "OUT", "GND", 3e3)],
        isources=[("IB", "VDD", "OUT", 2e-6)],
    )
    bias = {"VDD": 10.0}
    x = np.array([2.5])
    plan = CompiledTopology(topo, bias)

    got = plan.dc_residuals(x, lambda *args: 0.0, gmin=1e-12)
    ref = topo.dc_residuals(x, bias, lambda *args: 0.0, gmin=1e-12)

    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-18)


def test_compiled_ac_metadata_respects_device_and_node_drives():
    topo = Topology(
        solved=["IN", "OUT"],
        devices=[("M1", "OUT", "VG", "VDD")],
        rails={"VDD": "VDD", "VG": "VG", "VIN": "VIN", "GND": 0.0},
        outputs=("OUT",),
        input_drives={"M1": 1.0},
        ac_drives={"VIN": 0.5},
        resistors=[("RIN", "VIN", "IN", 1e3)],
        capacitors=[("CL", "OUT", "GND", 2e-12)],
    )
    plan = CompiledTopology(topo, {"VDD": 10.0, "VG": 3.0, "VIN": 0.0})

    name, d, g, s = plan.ac_devices(drive=topo.input_drives)[0]
    assert name == "M1"
    assert d == ("n", topo.idx["OUT"])
    assert g == ("v", 1.0)
    assert s == ("v", 0.0)

    rname, ra, rb, _, gval = plan.ac_resistors(topo.ac_drives)[0]
    assert rname == "RIN"
    assert ra == ("v", 0.5)
    assert rb == ("n", topo.idx["IN"])
    assert gval == 1e-3

    ca, cb, cap = plan.ac_capacitors()[0]
    assert ca == ("n", topo.idx["OUT"])
    assert cb == ("v", 0.0)
    assert cap == 2e-12


def test_compiled_transient_input_and_node_input_tokens():
    topo = Topology(
        solved=["OUT"],
        devices=[("M1", "OUT", "VG", "VDD")],
        rails={"VDD": "VDD", "VG": "VG", "VIN": "VIN"},
        outputs=("OUT",),
        transient_inputs={"M1": "gate"},
        resistors=[("RIN", "VIN", "OUT", 1e3)],
    )
    plan = CompiledTopology(
        topo,
        {"VDD": 10.0, "VG": 3.0, "VIN": 0.0},
        input_keys=("gate", "vin"),
        node_inputs={"VIN": "vin"},
        transient_inputs=True,
    )

    dev = plan.devices[0]
    assert dev.d[0] == TERM_SOLVED
    assert dev.g == (TERM_INPUT, 0)
    assert dev.si is None

    rin = plan.resistors[0]
    assert rin.a == (TERM_INPUT, 1)
    assert rin.bi == topo.idx["OUT"]

    node_v = np.array([2.0])
    input_v = np.array([0.7, 1.2])
    assert plan.term_value(dev.g, node_v, input_v) == 0.7
    assert plan.term_value(rin.a, node_v, input_v) == 1.2
