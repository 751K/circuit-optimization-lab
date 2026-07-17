"""Engine-neutral transient API contracts.

The Numba marshalling signature is an implementation detail.  R3 pins the
public result/profile schema and numerical behavior shared by every engine.
"""
from __future__ import annotations

import numpy as np
import pytest

from circuitopt import transient_profile as tp
from circuitopt import transient_solver as ts
from circuitopt.topology import Topology


PUBLIC_RESULT_KEYS = {
    "t",
    "output",
    "vout",
    "nodes",
    "nfail",
    "nretry",
    "nsubsteps",
    "numba_grid_solver",
    "numba_adaptive_solver",
    "rust_grid_solver",
    "rust_adaptive_solver",
    "transient_cap_mode",
    "transient_cap_mode_id",
    "transient_profile",
}


def _rc_transient(engine, monkeypatch):
    topology = Topology(
        solved=["OUT"],
        devices=[],
        rails={"GND": 0.0},
        resistors=[("R", "OUT", "GND", 1e3)],
        capacitors=[("C", "OUT", "GND", 1e-9)],
        outputs=("OUT",),
    )
    times = np.linspace(0.0, 2e-6, 9)
    current = np.where(times == 0.0, 0.0, 1e-3)
    monkeypatch.setattr(ts, "current_engine", lambda: engine)
    return ts.transient(
        {},
        {},
        times,
        topo=topology,
        V0=np.array([0.0]),
        inputs={"iin": current},
        current_inputs=[("GND", "OUT", "iin")],
        integration_method="gear2",
        profile=True,
    )


def test_transient_profile_slots_are_dense_and_named():
    assert len(tp.TRANSIENT_PROFILE_FIELDS) == tp.PROFILE_LEN
    assert tuple(
        tp.PROFILE_SLOT_BY_NAME[name] for name in tp.TRANSIENT_PROFILE_FIELDS
    ) == tuple(range(tp.PROFILE_LEN))
    assert tp.PROFILE_NEWTON_ITERS == tp.PROFILE_SLOT_BY_NAME["newton_iters_total"]
    assert tp.PROFILE_FAILED_INTERVALS == tp.PROFILE_SLOT_BY_NAME["failed_intervals"]
    assert tp.PROFILE_STALLED_RESIDUAL_ACCEPTS == tp.PROFILE_LEN - 1


def test_transient_result_contract_is_engine_neutral(monkeypatch):
    try:
        import circuitopt_core  # noqa: F401
    except ImportError:
        pytest.skip("circuitopt_core is not installed")

    reference = _rc_transient("numba", monkeypatch)
    rust = _rc_transient("rust", monkeypatch)

    assert PUBLIC_RESULT_KEYS <= reference.keys()
    assert PUBLIC_RESULT_KEYS <= rust.keys()
    assert reference["transient_profile"].keys() == rust["transient_profile"].keys()
    assert reference["numba_grid_solver"] is True
    assert reference["rust_grid_solver"] is False
    assert rust["numba_grid_solver"] is False
    assert rust["rust_grid_solver"] is True
    assert rust["nfail"] == reference["nfail"]
    assert rust["nretry"] == reference["nretry"]
    assert rust["nsubsteps"] == reference["nsubsteps"]
    np.testing.assert_allclose(rust["t"], reference["t"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        rust["output"], reference["output"], rtol=1e-12, atol=1e-16
    )
