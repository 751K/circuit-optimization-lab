"""Nested-parallelism policy for the compiled device evaluator.

The compiled core evaluates a transient's device batch through one process-wide
pool. That is the right choice for a single isolated solve, but every driver
that runs solves concurrently — signoff PVT points, a ramp of SAR conversions,
Monte-Carlo trials, corner sweeps — submits into the *same* pool, so the outer
workers end up queueing behind each other instead of running.

Measured on the reference machine with sixteen TSMC28 MDAC residue transients:
the pool finished one thread's worth in 4.90 s while keeping 7.8 cores busy, and
eight threads only reached 4.03 s because no idle core was left. Evaluating the
batches inline instead took 10.54 s on one thread but 3.19 s on eight, at 6.0
cores. Whoever owns the outer loop should own the parallelism.

Only the schedule changes: batch slots are written independently, so results are
identical either way.
"""
from __future__ import annotations

from contextlib import nullcontext


def worker_device_eval(workers: int, items: int | None = None):
    """Scope for one worker of a ``workers``-wide parallel region.

    Returns a context manager that makes device batches evaluate inline for the
    calling thread. With ``workers <= 1`` there is no outer parallelism to
    protect, so the pool keeps the single solve fast and this is a no-op. The
    same applies when ``items`` is given and there is less outer work than there
    are workers: the outer level cannot fill the machine on its own, so the pool
    is still the better use of the idle cores. This mirrors the compiled
    campaign's own axis rule (parallel over candidates only when
    ``n >= workers``).
    """
    if workers is None or workers <= 1:
        return nullcontext()
    if items is not None and items < workers:
        return nullcontext()
    try:
        from circuitopt_core import serial_device_eval
    except ImportError:  # pragma: no cover - extension always present in-tree
        return nullcontext()
    return serial_device_eval()
