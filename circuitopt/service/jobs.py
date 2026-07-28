"""In-process background-job manager for long solver tasks (explore / MC / PVT).

The service layer runs three kinds of work that are too slow for a synchronous
request: design-space *explore*, per-device mismatch *Monte-Carlo*, and the
*PVT* corner sweep. This module owns those as background **jobs** so the HTTP
layer can submit one, poll its status, stream its progress over a WebSocket, and
request cancellation.

Design (all deliberate, for a **local single-user** service — no persistence,
no multi-tenant isolation):

* **Threading.** Jobs run on a :class:`~concurrent.futures.ThreadPoolExecutor`.
  The solvers are CPU-bound (NumPy / Numba, releasing the GIL in the hot loops),
  so the default is a **single worker** — conservative, no oversubscription of
  the BLAS/Numba thread pools. The ``serve`` CLI exposes ``--job-workers`` to
  raise it. The solver call itself is plain blocking Python; the async event
  loop is never blocked because the work happens on the executor thread and the
  WebSocket bridges to it through a thread-safe queue.

* **Storage.** Jobs live in an in-memory ``dict`` (this is a local service; a
  restart drops history, which is fine). At most :data:`MAX_JOBS` are retained —
  when the cap is exceeded the **oldest already-terminal** job is evicted
  (a running/queued job is never dropped).

* **State machine.** ``queued -> running -> {done, failed, cancelled}``. The
  three terminal states are final. ``result`` is set on ``done``; ``error`` (a
  ``{"stage", "message"}`` envelope, matching the app's 422 detail shape) on
  ``failed``.

* **Progress.** The worker pushes events onto a per-job thread-safe
  :class:`queue.Queue`; the WebSocket endpoint drains it. Every event payload is
  run through :func:`circuitopt.service.serialize.to_jsonable` so it is strict
  JSON (NaN/complex/ndarray safe). ``GET`` polling reads the latest snapshot
  instead of the queue, so polling and streaming can coexist.

* **Cancellation.** Best-effort and cooperative. Each job carries a
  :class:`threading.Event`; the worker passes a ``should_stop`` closure into the
  solver driver, which checks it between candidates/samples. A candidate already
  in flight runs to completion before the job stops — cancellation is not a hard
  kill (there is no safe way to interrupt a NumPy solve mid-call). A cancelled
  job that had done partial work still returns whatever the driver produced,
  flagged ``stopped_early`` by the core hooks.

This module imports **no fastapi** — it is pure threading/queue plumbing, so it
is unit-testable on its own and the app layer just wires it in.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..analysis_dispatch import _frequency_grid
from ..explore import explore_from_dict
from ..corners import (circuit_corner_names, corner_table_from_dict,
                       mismatch_mc_from_dict)
from .serialize import to_jsonable

# How many jobs to retain in memory before evicting the oldest terminal one.
MAX_JOBS = 50

# The kinds of job the manager knows how to run (also surfaced in capabilities).
JOB_KINDS = ("explore", "mc", "pvt")

# Terminal states — a job in any of these never changes again.
_TERMINAL = frozenset({"done", "failed", "cancelled"})


@dataclass
class Job:
    """One background task and its live state.

    All mutation happens under the owning :class:`JobManager`'s lock except the
    per-job progress :class:`queue.Queue`, which is itself thread-safe and is the
    one hand-off point between the worker thread and the WebSocket consumer.
    """
    id: str
    kind: str
    params: dict[str, Any]
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    progress: Optional[dict[str, Any]] = None       # latest progress snapshot
    result: Optional[dict[str, Any]] = None         # set when status == "done"
    error: Optional[dict[str, Any]] = None          # {stage, message} when failed
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _events: "queue.Queue" = field(default_factory=queue.Queue, repr=False)

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL

    def snapshot(self) -> dict[str, Any]:
        """A JSON-safe status view (no result/error — those are added by the app)."""
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "progress": self.progress,
        }


# ── the two job bodies ────────────────────────────────────────────────────────
# Each returns a JSON-safe result dict and takes (params, emit, should_stop):
#   * emit(event: dict)   — push a progress event to the WS/queue (see _run).
#   * should_stop() -> bool — cooperative-cancel check threaded to the core hook.

def _run_explore(params: dict, emit: Callable[[dict], None],
                 should_stop: Callable[[], bool]) -> dict:
    circuit = params["circuit"]
    n = int(params.get("n", 32))
    seed = int(params.get("seed", 0))
    corner = params.get("corner")
    workers = int(params.get("workers", 1))

    def progress(done: int, total: int) -> None:
        emit({"type": "progress", "done": done, "total": total,
              "frac": (done / total) if total else 1.0})

    results = explore_from_dict(circuit, n=n, seed=seed, corner=corner,
                               progress=progress, should_stop=should_stop,
                               workers=workers)
    return to_jsonable(results)


def _run_mc(params: dict, emit: Callable[[dict], None],
            should_stop: Callable[[], bool]) -> dict:
    circuit = params["circuit"]
    n = int(params.get("n", 64))
    seed = int(params.get("seed", 0))
    corner = params.get("corner", "typical")
    workers = int(params.get("workers", 1))

    def progress(done: int, total: int, partial) -> None:
        emit({"type": "progress", "done": done, "total": total,
              "frac": (done / total) if total else 1.0,
              "partial": to_jsonable(partial)})

    freqs, band, freq_source = _sweep_ranges(circuit, params)
    kwargs = {} if band is None else {"band": band}
    results = mismatch_mc_from_dict(circuit, n=n, seed=seed, corner=corner,
                                    freqs=freqs, progress=progress,
                                    should_stop=should_stop, workers=workers,
                                    **kwargs)
    payload = to_jsonable(results)
    payload["freq_range_hz"] = ([float(freqs[0]), float(freqs[-1])]
                                if freqs is not None else None)
    payload["freq_source"] = freq_source
    payload["noise_band_hz"] = list(band) if band is not None else None
    return payload


def _sweep_ranges(circuit: dict, params: dict):
    """Resolve the ``(freqs, band, source)`` a corner or MC sweep should measure over.

    Both :func:`~circuitopt.corners.corner_table` and
    :func:`~circuitopt.corners.mismatch_mc` default to the AFE ranges — a
    0.01–10 kHz grid and a 0.05–100 Hz noise band. Those are wrong for a silicon
    amplifier by four decades, and wrong *silently*: the sweep reports the top of its
    own grid as ``bw_Hz``, so a 79 kHz OTA comes back as "BW = 10 kHz". A number that
    reads as a measurement but is really a grid ceiling is the worst failure mode a
    result view can have — and in an MC summary it is worse still, because every
    sample hits the same ceiling and the spread collapses to σ = 0, which reads as a
    remarkably robust design rather than as a measurement that never happened.

    So the resolution order is: an explicit request range, else the circuit's own
    ``analyses.ac.freqs`` / ``analyses.noise.band`` (the ranges the same circuit's AC
    and noise runs already use, so the sweep agrees with the single-point run the
    user just did), else ``None`` — which leaves the frozen defaults in place for
    the AFE circuits they were written for.

    ``source`` names which of the three won, so the result can say what it measured
    over rather than leaving the reader to guess."""
    analyses = circuit.get("analyses") or {}
    if not isinstance(analyses, dict):
        analyses = {}

    freqs = None
    source = "corner_table default (AFE 0.01 Hz - 10 kHz)"
    if params.get("freqs") is not None:
        freqs = _frequency_grid(params["freqs"])
        source = "request"
    else:
        ac_cfg = analyses.get("ac")
        if isinstance(ac_cfg, dict) and ac_cfg.get("freqs") is not None:
            freqs = _frequency_grid(ac_cfg["freqs"])
            source = "circuit analyses.ac.freqs"

    band = None
    if params.get("band") is not None:
        band = tuple(float(x) for x in params["band"])
    else:
        noise_cfg = analyses.get("noise")
        if isinstance(noise_cfg, dict) and noise_cfg.get("band") is not None:
            band = tuple(float(x) for x in noise_cfg["band"])
    return freqs, band, source


def _run_pvt(params: dict, emit: Callable[[dict], None],
             should_stop: Callable[[], bool]) -> dict:
    """PVT corner sweep (semantics of ``circuit-opt corners``).

    ``corner_table`` has no per-candidate callback and no cooperative-cancel hook —
    it dispatches whole corner batches into the compiled campaign, where a Python
    callback between candidates is exactly what the batching exists to avoid. So
    ``should_stop`` is checked once before the sweep starts and progress is reported
    per completed PVT grid *slice*. A sweep with no temperature/supply axis is a
    single slice; those are the fast case (seconds), so the coarser granularity costs
    nothing in practice.

    The result carries the ``corners``/``temps``/``vdd_scale`` axes and the measured
    frequency range alongside the table, so a consumer can label the grid without
    re-deriving the circuit's family or guessing what the numbers span."""
    circuit = params["circuit"]
    corners = params.get("corners")
    temps = params.get("temps")
    vdd_scale = params.get("vdd_scale")
    workers = int(params.get("workers", 1))

    if should_stop():
        return to_jsonable({"table": {}, "stopped_early": True})

    # Resolve the defaulted corner set here so the result always names its own axes;
    # a consumer plotting the grid should not have to re-derive the circuit's family.
    names, silicon = circuit_corner_names(circuit)
    if corners:
        names = tuple(corners)
    freqs, band, freq_source = _sweep_ranges(circuit, params)
    slices = (len(temps) if temps else 1) * (len(vdd_scale) if vdd_scale else 1)

    def progress(done: int, total: int) -> None:
        emit({"type": "progress", "done": done, "total": total,
              "frac": (done / total) if total else 1.0,
              "unit": "slice"})

    emit({"type": "progress", "done": 0, "total": slices, "frac": 0.0,
          "unit": "slice"})
    kwargs = {} if band is None else {"band": band}
    table = corner_table_from_dict(
        circuit, corners=corners, temps=temps, vdd_scale=vdd_scale,
        freqs=freqs, workers=workers, progress=progress, **kwargs)
    return to_jsonable({
        "table": table,
        "corners": list(names),
        "temps": list(temps) if temps else None,
        "vdd_scale": list(vdd_scale) if vdd_scale else None,
        "silicon": silicon,
        "slices": slices,
        "freq_range_hz": ([float(freqs[0]), float(freqs[-1])]
                          if freqs is not None else None),
        "freq_source": freq_source,
        "noise_band_hz": list(band) if band is not None else None,
    })


_RUNNERS: dict[str, Callable] = {
    "explore": _run_explore, "mc": _run_mc, "pvt": _run_pvt}


class JobManager:
    """Owns the job table and the worker pool. One instance per app, on
    ``app.state.jobs`` so tests can inject a custom-sized manager."""

    def __init__(
        self,
        workers: int = 1,
        max_jobs: int = MAX_JOBS,
        runners: Optional[dict[str, Callable]] = None,
    ):
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(workers)),
                                        thread_name_prefix="cktopt-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs
        self._runners = {**_RUNNERS, **(runners or {})}

    # ── submission ────────────────────────────────────────────────────────────
    def submit(self, kind: str, params: dict) -> Job:
        """Create a queued job of *kind* and schedule it on the pool."""
        if kind not in self._runners:
            raise ValueError(
                f"unknown job kind {kind!r}; known: {sorted(self._runners)}")
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()
        self._pool.submit(self._run, job)
        return job

    def _evict_locked(self) -> None:
        """Drop the oldest terminal jobs until under the cap. Caller holds lock."""
        if len(self._jobs) <= self._max_jobs:
            return
        terminal = sorted((j for j in self._jobs.values() if j.terminal),
                          key=lambda j: j.finished or j.created)
        for job in terminal:
            if len(self._jobs) <= self._max_jobs:
                break
            self._jobs.pop(job.id, None)

    # ── the worker body ───────────────────────────────────────────────────────
    def _run(self, job: Job) -> None:
        """Executed on a pool thread: run the job body, capture terminal state."""
        # A cancel requested while still queued short-circuits before any work.
        if job._cancel.is_set():
            job.status = "cancelled"
            job.finished = time.time()
            job._events.put({"type": "terminal", "status": "cancelled"})
            return
        job.status = "running"
        job.started = time.time()

        def emit(event: dict) -> None:
            safe = to_jsonable(event)
            job.progress = safe
            job._events.put(safe)

        try:
            result = self._runners[job.kind](
                job.params, emit, job._cancel.is_set)
        except Exception as exc:  # any solver/parse failure -> failed, no traceback leak
            job.status = "failed"
            job.error = {"stage": "solve", "message": str(exc)}
            job.finished = time.time()
            job._events.put({"type": "terminal", "status": "failed", "error": job.error})
            return

        job.finished = time.time()
        # A cooperative stop leaves the result flagged; report it as cancelled so
        # the client sees the requested outcome (the partial result is still kept).
        if job._cancel.is_set() or (isinstance(result, dict) and result.get("stopped_early")):
            job.status = "cancelled"
            job.result = result
            job._events.put({"type": "terminal", "status": "cancelled"})
        else:
            job.status = "done"
            job.result = result
            job._events.put({"type": "terminal", "status": "done"})

    # ── queries ───────────────────────────────────────────────────────────────
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        """Newest-first list of status snapshots (no result/error payloads)."""
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
        return [j.snapshot() for j in jobs]

    def cancel(self, job_id: str) -> str:
        """Request cancellation. Returns 'requested' if it was set, or 'terminal'
        if the job has already finished (the caller maps that to HTTP 409)."""
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.terminal:
            return "terminal"
        job._cancel.set()
        return "requested"

    def shutdown(self) -> None:
        """Stop the pool (cancels queued futures; running jobs finish their
        in-flight candidate). Called on app shutdown."""
        for job in list(self._jobs.values()):
            if not job.terminal:
                job._cancel.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
