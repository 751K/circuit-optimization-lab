"""R3 parity tests for Rust AC/noise LTI MNA assembly."""
from __future__ import annotations

import numpy as np
import pytest

from circuitopt import ac_solver, noise_solver
from circuitopt.topology import Topology

try:
    import circuitopt_core
except ImportError:  # pragma: no cover - optional compiled wheel
    circuitopt_core = None


requires_rust_lti = pytest.mark.skipif(
    circuitopt_core is None or not hasattr(circuitopt_core, "LtiProblem"),
    reason="R3 circuitopt_core LTI extension is not installed")


def _node(index):
    return 0, index, 0.0


def _known(value):
    return 2, 0, value


@requires_rust_lti
def test_rust_lti_rc_assembly_and_solve():
    problem = circuitopt_core.LtiProblem({
        "size": 1,
        "dense_devices": [],
        "mos_devices": [],
        "capacitors": [(_node(0), _known(0.0), 1e-9)],
        "resistors": [(_node(0), _known(1.0), 1e-3)],
        "vccs": [], "voltage_sources": [], "vcvs": [], "cccs": [], "ccvs": [],
    })
    conductance, capacitance, rhs_g, rhs_c = problem.matrices()
    np.testing.assert_array_equal(conductance, [[1e-3]])
    np.testing.assert_array_equal(capacitance, [[1e-9]])
    np.testing.assert_array_equal(rhs_g, [1e-3])
    np.testing.assert_array_equal(rhs_c, [0.0])

    frequencies = np.array([1.0, 1e3, 1e6])
    pairs = problem.solve(frequencies)
    assert isinstance(pairs, np.ndarray)
    got = pairs[..., 0] + 1j * pairs[..., 1]
    expected = 1e-3 / (1e-3 + 2j * np.pi * frequencies * 1e-9)
    np.testing.assert_allclose(got[:, 0], expected, rtol=1e-13, atol=1e-15)

    with pytest.raises(ValueError, match="contiguous"):
        problem.solve(np.arange(8.0)[::2])
    with pytest.raises(ValueError, match="frequencies must be finite"):
        problem.solve(np.array([np.nan]))
    with pytest.raises(ValueError, match="sense length must match"):
        problem.solve_transpose(np.array([1.0]), np.empty(0))


@requires_rust_lti
@pytest.mark.parametrize("field,record", [
    ("resistors", (_node(4), _known(0.0), 1.0)),
    ("voltage_sources", (_node(0), _known(0.0), 4, 1.0, 0.0)),
])
def test_rust_lti_rejects_out_of_bounds_topology(field, record):
    spec = {
        "size": 1,
        "dense_devices": [],
        "mos_devices": [],
        "capacitors": [],
        "resistors": [],
        "vccs": [],
        "voltage_sources": [],
        "vcvs": [],
        "cccs": [],
        "ccvs": [],
    }
    spec[field] = [record]

    with pytest.raises(ValueError, match="invalid LTI MNA problem"):
        circuitopt_core.LtiProblem(spec)


@requires_rust_lti
def test_public_ac_and_noise_dispatch_to_rust(monkeypatch):
    topo = Topology(
        solved=["OUT"], devices=[], rails={"IN": 0.0, "GND": 0.0},
        resistors=[("R1", "IN", "OUT", 1e3),
                   ("R2", "OUT", "GND", 1e3)],
        capacitors=[("C", "OUT", "GND", 1e-9)],
        ac_drives={"IN": 1.0}, outputs=("OUT",))
    frequencies = np.logspace(0, 6, 21)

    monkeypatch.setattr(ac_solver, "current_engine", lambda: "numba")
    monkeypatch.setattr(noise_solver, "current_engine", lambda: "numba")
    ac_reference = ac_solver.ac_solve({}, {}, frequencies, topo=topo)
    noise_reference = noise_solver.noise_analysis(
        {}, {}, frequencies, topo=topo, ac_result=ac_reference)

    monkeypatch.setattr(ac_solver, "current_engine", lambda: "rust")
    monkeypatch.setattr(noise_solver, "current_engine", lambda: "rust")
    ac_got = ac_solver.ac_solve({}, {}, frequencies, topo=topo)
    noise_got = noise_solver.noise_analysis(
        {}, {}, frequencies, topo=topo, ac_result=ac_got)

    assert ac_got["rust_lti_solver"] is True
    assert noise_got["rust_lti_solver"] is True
    np.testing.assert_allclose(ac_got["response"], ac_reference["response"],
                               rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(noise_got["out_psd"], noise_reference["out_psd"],
                               rtol=1e-13, atol=1e-30)


@requires_rust_lti
def test_rust_ac_preserves_complex_source_phase(monkeypatch):
    topology = Topology(
        solved=["IN", "OUT"],
        devices=[],
        rails={"GND": 0.0},
        resistors=[("R1", "IN", "OUT", 1e3),
                   ("R2", "OUT", "GND", 1e3)],
        vsources=[("V1", "IN", "GND", 0.0)],
        ac_drives={"V1": 1j},
        outputs=("OUT",),
    )
    monkeypatch.setattr(ac_solver, "current_engine", lambda: "rust")

    result = ac_solver.ac_solve({}, {}, np.array([1.0]), topo=topology)

    np.testing.assert_allclose(result["response"], [0.5j], rtol=1e-13, atol=1e-15)
