"""Solver-fallback diagnostics: the counter mechanics + a real zeroed-device hit.

``circuitopt.diagnostics`` makes the solvers' deliberate exception-fallback paths
observable without changing any numerical result. These tests pin the counter
API and prove that the model-eval "return zeroed gm/gds" path -- the one that can
turn a diverged device into a plausible-but-wrong solve -- actually records a
``model.ss_params_zeroed`` event when it fires.
"""
import logging

import pytest

from circuitopt import diagnostics
from circuitopt.device_model import create_device


@pytest.fixture(autouse=True)
def _clean_counters():
    diagnostics.reset()
    yield
    diagnostics.reset()


def test_note_counts_and_snapshot_is_a_copy():
    diagnostics.note("dc.fsolve_guess_fail", ValueError("singular"))
    diagnostics.note("dc.fsolve_guess_fail")
    diagnostics.note("pss.dc_seed_fail")
    snap = diagnostics.snapshot()
    assert snap == {"dc.fsolve_guess_fail": 2, "pss.dc_seed_fail": 1}
    assert diagnostics.total() == 3
    # snapshot() hands back a copy -- mutating it must not corrupt the registry.
    snap["dc.fsolve_guess_fail"] = 999
    assert diagnostics.snapshot()["dc.fsolve_guess_fail"] == 2


def test_last_details_records_exception_repr_and_detail():
    diagnostics.note("dc.residual_eval_fail", RuntimeError("no convergence"))
    diagnostics.note_critical("model.ss_params_zeroed", detail="gm/gds -> 0/1e-12")
    details = diagnostics.last_details()
    assert details["dc.residual_eval_fail"] == "RuntimeError: no convergence"
    assert details["model.ss_params_zeroed"] == "gm/gds -> 0/1e-12"


def test_reset_clears_everything():
    diagnostics.note("a")
    diagnostics.note_critical("b")
    diagnostics.reset()
    assert diagnostics.snapshot() == {}
    assert diagnostics.total() == 0
    assert diagnostics.summary() == "diagnostics: no solver-fallback events recorded"


def test_summary_orders_by_frequency_then_name():
    for _ in range(3):
        diagnostics.note("transient.gear2_step_raised")
    diagnostics.note("dc.box_guess_fail")
    lines = diagnostics.summary().splitlines()
    assert lines[0].startswith("diagnostics:")
    assert "transient.gear2_step_raised" in lines[1]   # most frequent first
    assert "dc.box_guess_fail" in lines[2]


def test_note_never_raises_even_on_bad_input():
    # Diagnostics must never add a second failure mode to an already-failing path.
    class Nasty:
        def __repr__(self):
            raise ValueError("repr blew up")

    diagnostics.note("weird", Nasty())          # exc whose repr raises
    diagnostics.note(None)                       # non-str category
    diagnostics.note("ok")
    assert diagnostics.snapshot().get("ok") == 1


def test_note_critical_first_sighting_logs_at_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="circuitopt.diagnostics"):
        diagnostics.note_critical("model.device_state_zeroed",
                                  RuntimeError("op solve failed"))
        diagnostics.note_critical("model.device_state_zeroed",
                                  RuntimeError("again"))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    # First sighting surfaces at WARNING; the repeat stays counter-only.
    assert len(warnings) == 1
    assert "model.device_state_zeroed" in warnings[0].getMessage()
    assert diagnostics.snapshot()["model.device_state_zeroed"] == 2


def test_model_zeroed_ss_params_path_is_recorded():
    """The real hazard: a device whose evaluation fails returns fabricated
    gm=0 / gds=1e-12 so the circuit still "solves". That must be counted."""
    dev = create_device("pmos_tft", W=1000, L=20)

    def _boom(*_a, **_k):
        raise RuntimeError("forced device-eval failure")

    # Kill both the numba and the finite-difference small-signal branches so the
    # fabricated-zero fallback (pmos_tft_model.get_ss_params) is the only exit.
    dev.get_op = _boom
    dev.get_Idc = _boom
    dev._eval_currents = _boom

    params = dev.get_ss_params(40.0, 0.0, 20.0)

    # Behaviour is unchanged: the fabricated zeroed small-signal dict.
    assert params == {"gm": 0.0, "gds": 1e-12, "Cgs": 0.0, "Cgd": 0.0, "Ich": 0.0}
    # ...but it is no longer silent: a human-written detail is recorded (it takes
    # precedence over the raw exception repr).
    assert diagnostics.snapshot().get("model.ss_params_zeroed", 0) >= 1
    assert "fabricated" in diagnostics.last_details()["model.ss_params_zeroed"]
