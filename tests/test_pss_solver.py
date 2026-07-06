import numpy as np
import pytest

from circuitopt.pss_solver import pss_solve
from circuitopt.topology import Topology


def test_pss_rejects_nonperiodic_input_boundary():
    topo = Topology(
        solved=["OUT"],
        devices=[],
        rails={"VIN": "VIN", "GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "VIN", "OUT", 1e3)],
    )
    t = np.linspace(0.0, 1e-3, 11)

    with pytest.raises(ValueError, match="not periodic"):
        pss_solve(
            {}, {"VIN": 0.0}, 1e-3, topo=topo, tgrid=t,
            inputs={"vin": np.linspace(0.0, 1.0, len(t))},
            node_inputs={"VIN": "vin"},
            V0=np.array([0.0]),
        )


def test_pss_periodic_rc_converges_to_same_boundary_state():
    period = 1e-3
    t = np.linspace(0.0, period, 81)
    vin = 0.25 * np.sin(2.0 * np.pi * t / period)
    topo = Topology(
        solved=["OUT"],
        devices=[],
        rails={"VIN": "VIN", "GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "VIN", "OUT", 1e5)],
        capacitors=[("C1", "OUT", "GND", 1e-9)],
    )

    result = pss_solve(
        {}, {"VIN": 0.0}, period, topo=topo, tgrid=t,
        inputs={"vin": vin}, node_inputs={"VIN": "vin"},
        V0=np.array([0.0]), residual_tol=1e-9, max_shooting_iters=4,
    )

    assert result["converged"]
    assert result["nfail"] == 0
    assert result["residual_norm"] < 1e-9
    assert result["output"][0] == pytest.approx(result["output"][-1], abs=1e-9)
    assert np.ptp(result["output"]) > 1e-3


def test_pss_constant_passive_network_uses_dc_seed():
    period = 2e-3
    t = np.linspace(0.0, period, 21)
    topo = Topology(
        solved=["OUT"],
        devices=[],
        rails={"VDD": "VDD", "GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "VDD", "OUT", 3e3), ("R2", "OUT", "GND", 1e3)],
        capacitors=[("C1", "OUT", "GND", 1e-9)],
    )

    result = pss_solve({}, {"VDD": 10.0}, period, topo=topo, tgrid=t,
                       residual_tol=1e-10, max_shooting_iters=2)

    assert result["converged"]
    assert result["nfail"] == 0
    assert result["x0"][0] == pytest.approx(2.5, rel=1e-9)
    assert result["residual_norm"] < 1e-10


def test_pss_reuses_converged_stabilization_period(monkeypatch):
    period = 1e-3
    t = np.linspace(0.0, period, 5)
    topo = Topology(
        solved=["A"],
        devices=[],
        rails={"GND": 0.0},
        outputs=("A",),
    )
    calls = {"n": 0}

    def fake_transient(_sizes, _bias, tgrid, V0=None, **_kwargs):
        calls["n"] += 1
        x0 = np.asarray(V0, float)
        vals = np.full(len(tgrid), x0[0])
        return {"t": tgrid, "nodes": {"A": vals}, "output": vals, "vout": vals,
                "nfail": 0}

    monkeypatch.setattr("circuitopt.pss_solver.transient", fake_transient)

    result = pss_solve(
        {}, {}, period, topo=topo, tgrid=t, V0=np.array([0.42]),
        tstab_periods=3, residual_tol=1e-12, max_shooting_iters=2,
    )

    assert result["converged"]
    assert result["shooting_period_runs"] == 1
    assert calls["n"] == 1
    assert result["x0"][0] == pytest.approx(0.42)


def test_pss_reuses_broyden_jacobian_after_first_fd_build(monkeypatch):
    period = 1e-3
    t = np.linspace(0.0, period, 5)
    topo = Topology(
        solved=["A", "B"],
        devices=[],
        rails={"GND": 0.0},
        outputs=("A",),
    )
    root = np.array([0.3, -0.2])
    amat = np.array([[0.35, 0.08], [-0.04, 0.28]])

    def fake_transient(_sizes, _bias, tgrid, V0=None, **_kwargs):
        x0 = np.asarray(V0, float)
        dx = x0 - root
        residual = amat @ dx + 0.08 * dx * dx
        x1 = x0 + residual
        nodes = {
            "A": np.linspace(x0[0], x1[0], len(tgrid)),
            "B": np.linspace(x0[1], x1[1], len(tgrid)),
        }
        return {"t": tgrid, "nodes": nodes, "output": nodes["A"], "vout": nodes["A"],
                "nfail": 0}

    monkeypatch.setattr("circuitopt.pss_solver.transient", fake_transient)

    common = dict(
        topo=topo,
        tgrid=t,
        V0=np.array([0.0, 0.0]),
        residual_tol=1e-30,
        max_shooting_iters=2,
        fd_step=1e-5,
        rail_margin=None,
        analytic_jacobian=False,   # this test exercises the FD + Broyden-reuse path
    )
    reused = pss_solve({}, {}, period, jacobian_reuse=True, **common)
    rebuilt = pss_solve({}, {}, period, jacobian_reuse=False, **common)

    assert reused["shooting_jacobian_evals"] == 1
    assert reused["shooting_jacobian_reuses"] == 1
    assert rebuilt["shooting_jacobian_evals"] == 2
    assert reused["shooting_period_runs"] < rebuilt["shooting_period_runs"]
    assert reused["residual_norm"] < 1e-3
