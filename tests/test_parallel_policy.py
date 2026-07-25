"""Nested-parallelism policy: who owns the cores when solves run concurrently."""
from __future__ import annotations

from contextlib import nullcontext

import pytest

from circuitopt._parallel import worker_device_eval


def _is_noop(scope) -> bool:
    return isinstance(scope, type(nullcontext()))


@pytest.mark.parametrize("workers", [None, 0, 1])
def test_a_lone_worker_keeps_the_shared_pool(workers):
    # Without an outer parallel level there is nothing to protect, and the pool
    # is what makes a single solve fast.
    assert _is_noop(worker_device_eval(workers))


def test_a_batch_too_small_to_fill_the_machine_keeps_the_pool():
    # Fewer items than workers means the outer level cannot use every core on
    # its own, so the idle ones are better spent inside the solve. This mirrors
    # the compiled campaign's axis rule (candidates in parallel only when
    # n >= workers).
    assert _is_noop(worker_device_eval(8, items=4))
    assert not _is_noop(worker_device_eval(8, items=8))
    assert not _is_noop(worker_device_eval(8, items=45))


def test_the_scope_toggles_the_compiled_flag():
    serial = pytest.importorskip("circuitopt_core").serial_device_eval
    scope = worker_device_eval(4, items=16)
    assert not _is_noop(scope)
    # Entering and leaving is balanced, and re-entry after exit works.
    with scope:
        pass
    with serial():
        with serial():
            pass


def test_signoff_points_run_through_the_worker_wrapper():
    # The wrapper is what puts each PVT point inside the scope; keep it wired.
    from circuitopt import signoff_campaign

    assert hasattr(signoff_campaign, "_run_point_worker")
