import numpy as np
import pytest

from core.pss_solver import pss_solve
from core.topology import Topology


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
