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


def _rc_transient_rust():
    # Rust is the only engine (v2.0.0), so no engine override is needed.
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


def test_transient_result_contract():
    # v2.0.0: rust is the only engine. The former engine-neutral A/B (vs the
    # removed numba/Python transient) is now a rust-only contract check: the
    # public result/profile schema is pinned, and the retired numba flags stay
    # present-but-False while the rust flags report True.
    try:
        import circuitopt_core  # noqa: F401
    except ImportError:
        pytest.skip("circuitopt_core is not installed")

    rust = _rc_transient_rust()
    assert PUBLIC_RESULT_KEYS <= rust.keys()
    assert rust["numba_grid_solver"] is False
    assert rust["numba_adaptive_solver"] is False
    assert rust["rust_grid_solver"] is True
    assert rust["nfail"] == 0
    assert np.all(np.isfinite(rust["output"]))
    assert np.all(np.isfinite(rust["t"]))
