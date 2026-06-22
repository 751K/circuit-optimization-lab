"""VCVS / CCCS / CCVS controlled-source tests.

Each case uses a small linear testbench with exact closed-form results.
"""
import numpy as np
import pytest

from core.ac_solver import ac_solve
from core.circuit_loader import circuit_from_dict
from core.noise_solver import noise_analysis
from core.topology import Topology
from core.transient_solver import transient

# ── helpers ─────────────────────────────────────────────────────────────────

def _vcvs_topology(mu=10.0, R1=1e3, R2=2e3):
    """E1: V_OUT = mu * V_IN (VCVS from IN→GND to OUT→GND, both rails).
    R1 IN-MID, R2 MID-GND. V_IN = 1.0 (driven)."""
    return Topology(
        solved=["IN", "MID", "OUT"],
        devices=[],
        rails={"GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "IN", "MID", R1), ("R2", "MID", "GND", R2)],
        vsources=[("V1", "IN", "GND", 1.0)],
        vcvs=[("E1", "OUT", "GND", "IN", "GND", mu)],
    )


def _cccs_topology(beta=2.0, R=1e3):
    """F1: I_OUT = beta * I_ctrl.  I_ctrl = I(V1) (branch current of V1).
    V1 drives R1 from IN to GND.  CCCS drives R2 from OUT to GND.
    I_V1 = 1.0 / R = 1 mA.  I_OUT = beta * 1 mA = 2 mA.  V_OUT = I_OUT * R = beta*R*(1/R) = beta."""
    return Topology(
        solved=["IN", "OUT"],
        devices=[],
        rails={"GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "IN", "GND", R), ("R2", "OUT", "GND", R)],
        vsources=[("V1", "IN", "GND", 1.0)],
        cccs=[("F1", "OUT", "GND", "V1", beta)],
    )


def _ccvs_topology(gamma=100.0, R=1e3):
    """H1: V_OUT = gamma * I_ctrl.  I_ctrl = I(V1), V_IN=1V, R=1k -> I=1mA.
    V_OUT = 100 * 0.001 = 0.1 V."""
    return Topology(
        solved=["IN", "OUT"],
        devices=[],
        rails={"GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "IN", "GND", R), ("R2", "OUT", "GND", R)],
        vsources=[("V1", "IN", "GND", 1.0)],
        ccvs=[("H1", "OUT", "GND", "V1", gamma)],
    )


# ── DC ──────────────────────────────────────────────────────────────────────

def test_vcvs_dc_exact():
    topo = _vcvs_topology(mu=10.0)
    ac = ac_solve({}, {}, np.array([1.0]), topo=topo)
    assert ac is not None
    # V_IN pinned by V1 constraint
    assert ac["dc_op"]["IN"] == pytest.approx(1.0, abs=1e-9)
    # V_OUT = mu * V_IN = 10.0
    assert ac["dc_op"]["OUT"] == pytest.approx(10.0, abs=1e-6)
    # V_MID = V_IN * R2/(R1+R2) = 2/3
    assert ac["dc_op"]["MID"] == pytest.approx(1.0 * 2e3 / 3e3, abs=1e-6)


def test_cccs_dc_exact():
    """CCCS: I_out = beta * I_ctrl. MNA convention: branch current I_V1 from IN→GND
    is −1mA (KCL at IN: −I_V1 − I_R = 0, I_R = 1mA → I_V1 = −1mA).
    Then I_out = 2*(−1mA) = −2mA, V_OUT = −2V."""
    topo = _cccs_topology(beta=2.0)
    ac = ac_solve({}, {}, np.array([1.0]), topo=topo)
    assert ac is not None
    # |I_V1| = 1.0 / 1e3 = 1 mA (branch current magnitude)
    assert abs(ac["branch_currents"]["V1"]) == pytest.approx(1e-3, rel=1e-6)
    # I_OUT = beta * I_V1 = −2mA → V_OUT = −2V
    assert ac["dc_op"]["OUT"] == pytest.approx(-2.0, abs=1e-4)


def test_ccvs_dc_exact():
    """CCVS: V_OUT = gamma * I_ctrl. I_V1 = −1mA, V_OUT = 100*(−1e-3) = −0.1V."""
    topo = _ccvs_topology(gamma=100.0)
    ac = ac_solve({}, {}, np.array([1.0]), topo=topo)
    assert ac is not None
    assert abs(ac["branch_currents"]["V1"]) == pytest.approx(1e-3, rel=1e-6)
    assert ac["dc_op"]["OUT"] == pytest.approx(-0.1, abs=1e-4)


def test_vcvs_dimensions():
    topo = _vcvs_topology()
    assert topo.n == 3
    assert topo.n_branches == 2  # V1 + E1
    assert topo.n_aug == 5
    assert topo.vsource_index["V1"] == 3
    assert topo.vsource_index["E1"] == 4


def test_cccs_dimensions():
    topo = _cccs_topology()
    assert topo.n == 2
    assert topo.n_branches == 1  # only V1; CCCS has no branch current
    assert topo.n_aug == 3


def test_ccvs_dimensions():
    topo = _ccvs_topology()
    assert topo.n == 2
    assert topo.n_branches == 2  # V1 + H1
    assert topo.n_aug == 4


# ── AC ──────────────────────────────────────────────────────────────────────

def test_vcvs_ac_gain_flat():
    """VCVS gain is mu at all frequencies. Drive V1 with AC=1 → V_IN_ac=1."""
    topo = _vcvs_topology(mu=10.0)
    topo.ac_drives = {"V1": 1.0}
    ac = ac_solve({}, {}, np.logspace(0, 4, 5), topo=topo)
    assert ac is not None
    gains = ac["gains"]
    expected = 10.0  # V_OUT / V_IN = mu
    for g in gains:
        assert g == pytest.approx(expected, rel=0.01)


def test_cccs_ac_gain_flat():
    """CCCS: I_OUT = beta*I_ctrl. Drive V1 with AC=1 → I_V1_ac = 1/R."""
    beta, R = 2.0, 1e3
    topo = _cccs_topology(beta=beta, R=R)
    topo.ac_drives = {"V1": 1.0}
    ac = ac_solve({}, {}, np.logspace(0, 4, 5), topo=topo)
    assert ac is not None
    # |I_V1_ac| = 1/R = 1e-3, |I_OUT| = beta*1e-3, |V_OUT| = beta*1e-3*R = beta
    for g in ac["gains"]:
        assert g == pytest.approx(beta, rel=0.01)


def test_ccvs_ac_gain_flat():
    """CCVS: V_OUT = gamma*I_ctrl. Drive V1 with AC=1 → I_V1_ac = 1/R."""
    gamma = 100.0
    topo = _ccvs_topology(gamma=gamma)
    topo.ac_drives = {"V1": 1.0}
    ac = ac_solve({}, {}, np.logspace(0, 4, 5), topo=topo)
    assert ac is not None
    expected = gamma * 1e-3  # = 0.1
    for g in ac["gains"]:
        assert g == pytest.approx(expected, rel=0.01)


# ── Noise ───────────────────────────────────────────────────────────────────

def test_vcvs_is_noiseless():
    """VCVS is noiseless — noise analysis runs without crash. OUT is clamped by
    the VCVS so resistor noise doesn't propagate there in this simple circuit."""
    topo = _vcvs_topology(mu=10.0, R1=1e3, R2=1e3)
    topo.ac_drives = {"V1": 1.0}                   # needed for gain computation
    n = noise_analysis({}, {}, np.logspace(0, 4, 3), topo=topo)
    assert n is not None
    # IRN should be finite (no crash / no NaN)
    assert np.all(np.isfinite(n["irn_psd"]))


def test_cccs_is_noiseless():
    topo = _cccs_topology(beta=1.0, R=1e3)
    topo.ac_drives = {"V1": 1.0}
    n = noise_analysis({}, {}, np.logspace(0, 4, 3), topo=topo)
    assert n is not None
    assert np.all(np.isfinite(n["irn_psd"]))


def test_ccvs_is_noiseless():
    topo = _ccvs_topology(gamma=100.0, R=1e3)
    topo.ac_drives = {"V1": 1.0}
    n = noise_analysis({}, {}, np.logspace(0, 4, 3), topo=topo)
    assert n is not None
    assert np.all(np.isfinite(n["irn_psd"]))


# ── Transient ───────────────────────────────────────────────────────────────

def test_vcvs_transient_step():
    """VCVS transient: step V_IN from 0→1 → V_OUT=mu*V_IN."""
    topo = _vcvs_topology(mu=5.0, R1=1e3, R2=2e3)
    t = np.linspace(0, 0.01, 100)
    tr = transient({}, {}, t, topo=topo, inputs={}, node_inputs={})
    assert tr is not None
    # V_OUT should settle to ~5.0 (mu*1.0 since V1=1.0 always)
    assert tr["vout"][-1] == pytest.approx(5.0, abs=0.05)


def test_cccs_transient_step():
    """CCCS transient: I_V1 = −1mA (MNA convention), I_OUT = beta*(−1mA) = −1mA,
    V_OUT = −1V."""
    topo = _cccs_topology(beta=1.0, R=1e3)
    t = np.linspace(0, 0.01, 100)
    tr = transient({}, {}, t, topo=topo, inputs={}, node_inputs={})
    assert tr is not None
    assert tr["vout"][-1] == pytest.approx(-1.0, abs=0.05)


def test_ccvs_transient_step():
    """CCVS transient: I_V1 = −1mA, V_OUT = gamma*(−1mA) = −0.1V."""
    topo = _ccvs_topology(gamma=100.0, R=1e3)
    t = np.linspace(0, 0.01, 100)
    tr = transient({}, {}, t, topo=topo, inputs={}, node_inputs={})
    assert tr is not None
    assert tr["vout"][-1] == pytest.approx(-0.1, abs=0.05)


# ── JSON loader round-trip ─────────────────────────────────────────────────

def test_json_vcvs_roundtrip():
    data = {
        "name": "vcvs_test",
        "solved": ["A", "B"],
        "rails": {"GND": 0.0},
        "devices": [],
        "bias": {},
        "sizes": {},
        "resistors": [{"name": "R1", "a": "A", "b": "GND", "R": 1e3}],
        "vsources": [{"name": "V1", "p": "A", "q": "GND", "value": 5.0}],
        "vcvs": [{"name": "E1", "p": "B", "q": "GND", "cp": "A", "cn": "GND", "mu": 10.0}],
        "outputs": ["B"],
    }
    spec = circuit_from_dict(data)
    t = spec.topology
    assert len(t.vcvs) == 1
    assert t.vcvs[0] == ("E1", "B", "GND", "A", "GND", 10.0)
    assert t.n_branches == 2  # V1 + E1


def test_json_cccs_roundtrip():
    data = {
        "name": "cccs_test",
        "solved": ["A", "B"],
        "rails": {"GND": 0.0},
        "devices": [],
        "bias": {},
        "sizes": {},
        "resistors": [{"name": "R1", "a": "A", "b": "GND", "R": 1e3},
                       {"name": "R2", "a": "B", "b": "GND", "R": 1e3}],
        "vsources": [{"name": "V1", "p": "A", "q": "GND", "value": 1.0}],
        "cccs": [{"name": "F1", "p": "B", "q": "GND", "ctrl_name": "V1", "beta": 3.0}],
        "outputs": ["B"],
    }
    spec = circuit_from_dict(data)
    t = spec.topology
    assert len(t.cccs) == 1
    assert t.cccs[0][0] == "F1"
    assert t.cccs[0][3] == "V1"
    assert t.cccs[0][4] == 3.0


def test_json_ccvs_roundtrip():
    data = {
        "name": "ccvs_test",
        "solved": ["A", "B"],
        "rails": {"GND": 0.0},
        "devices": [],
        "bias": {},
        "sizes": {},
        "resistors": [{"name": "R1", "a": "A", "b": "GND", "R": 1e3},
                       {"name": "R2", "a": "B", "b": "GND", "R": 1e3}],
        "vsources": [{"name": "V1", "p": "A", "q": "GND", "value": 1.0}],
        "ccvs": [{"name": "H1", "p": "B", "q": "GND", "ctrl_name": "V1", "gamma": 50.0}],
        "outputs": ["B"],
    }
    spec = circuit_from_dict(data)
    t = spec.topology
    assert len(t.ccvs) == 1
    assert t.ccvs[0][0] == "H1"
    assert t.ccvs[0][3] == "V1"  # ctrl_name


# ── Validation: CCCS/CCVS reference check ──────────────────────────────────

def test_cccs_unknown_branch_source_rejected():
    data = {
        "name": "bad",
        "solved": ["A", "B"],
        "rails": {"GND": 0.0},
        "devices": [],
        "bias": {},
        "sizes": {},
        "cccs": [{"name": "F1", "p": "B", "q": "GND", "ctrl_name": "NONEXISTENT", "beta": 1.0}],
    }
    with pytest.raises(ValueError, match="unknown branch source"):
        circuit_from_dict(data)


def test_ccvs_unknown_branch_source_rejected():
    data = {
        "name": "bad",
        "solved": ["A", "B"],
        "rails": {"GND": 0.0},
        "devices": [],
        "bias": {},
        "sizes": {},
        "ccvs": [{"name": "H1", "p": "B", "q": "GND", "ctrl_name": "NONEXISTENT", "gamma": 1.0}],
    }
    with pytest.raises(ValueError, match="unknown branch source"):
        circuit_from_dict(data)


def test_vcvs_identical_terminals_rejected():
    data = {
        "name": "bad",
        "solved": ["A"],
        "rails": {"GND": 0.0},
        "devices": [],
        "bias": {},
        "sizes": {},
        "vcvs": [{"name": "E1", "p": "A", "q": "A", "cp": "GND", "cn": "GND", "mu": 1.0}],
    }
    with pytest.raises(ValueError, match="identical output terminals"):
        circuit_from_dict(data)


# ── CCCS cascading (CCCS controlling on CCVS branch current) ───────────────

def test_cccs_cascaded_on_ccvs():
    """CCCS controlling on the branch current of a CCVS.
    I_V1 = −1mA (MNA). V_MID = 500*(−1mA) = −0.5V.
    KCL at MID: I_H1 = −I_R2 = −(V_MID/R) = +0.5mA.
    I_OUT = 2*(+0.5mA) = +1mA, V_OUT = +1V."""
    topo = Topology(
        solved=["IN", "MID", "OUT"],
        devices=[],
        rails={"GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "IN", "GND", 1e3), ("R2", "MID", "GND", 1e3),
                    ("R3", "OUT", "GND", 1e3)],
        vsources=[("V1", "IN", "GND", 1.0)],
        ccvs=[("H1", "MID", "GND", "V1", 500.0)],
        cccs=[("F1", "OUT", "GND", "H1", 2.0)],
    )
    ac = ac_solve({}, {}, np.array([1.0]), topo=topo)
    assert ac is not None
    assert ac["dc_op"]["MID"] == pytest.approx(-0.5, abs=1e-4)
    assert abs(ac["branch_currents"]["H1"]) == pytest.approx(0.5e-3, rel=1e-4)
    assert ac["dc_op"]["OUT"] == pytest.approx(1.0, abs=1e-4)
