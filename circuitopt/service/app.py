"""FastAPI application factory for the local circuit-solver service.

``create_app()`` builds the ``/api/v1`` app. Every route is a thin adapter onto
the existing solver stack — no numerical logic lives here:

* ``GET  /api/v1/health``        — liveness + version.
* ``GET  /api/v1/capabilities``  — self-description (models/PDKs, analyses and
  their legal option keys, process-corner names) so a GUI builds its dropdowns
  from the server, never from hardcoded front-end lists.
* ``POST /api/v1/validate``      — parse a circuit + validate its ``analyses``
  block; the validation *outcome* is the payload (always HTTP 200).
* ``POST /api/v1/solve``         — run ``run_analysis_suite`` and return
  JSON-safe results (or a structured 422 on parse/solve failure).

``fastapi`` is an optional dependency (the ``serve`` extra); importing this
module requires it. Callers that only need ``import circuitopt`` never reach
here — see :mod:`circuitopt.service` for the lazy-import contract.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from queue import Empty as queue_Empty
from typing import Any, Optional

try:
    from fastapi import (Body, FastAPI, HTTPException, Response, WebSocket,
                         WebSocketDisconnect)
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as exc:  # optional dependency (serve extra)
    raise ImportError(
        'the service layer needs FastAPI; pip install "circuit-optimization[serve]"'
    ) from exc

from .. import __version__
from ..analysis_dispatch import ANALYSIS_ORDER, run_analysis_suite
from ..analysis_options import known_keys, validate_analysis_cfg
from ..circuit_loader import circuit_from_dict
from ..device_factory import CORNERS, SKY130_CORNERS
from ..device_model import registered_models
from ..freepdk45_model import FREEPDK45_CORNERS
from .jobs import JOB_KINDS, JobManager
from .serialize import serialize_results, to_jsonable


# ── request envelopes ─────────────────────────────────────────────────────────
# pydantic is used ONLY for the thin request wrapper, never to re-describe the
# circuit schema — the single source of truth for a circuit is circuit_from_dict.
# ``circuit`` is therefore an opaque ``dict`` passed straight through.

class SolveRequest(BaseModel):
    """Body of ``POST /api/v1/solve``.

    ``circuit`` is a raw circuit-JSON object (the line format in
    ``docs/json_circuit_format.md``) forwarded verbatim to the loader.
    ``selected`` optionally restricts execution to a subset of analyses;
    ``corner`` optionally overrides the process corner for the whole suite.
    """
    circuit: dict[str, Any] = Field(..., description="Circuit JSON object (line format)")
    selected: Optional[list[str]] = Field(
        None, description="Subset of analyses to run, e.g. ['ac', 'noise']")
    corner: Optional[str] = Field(
        None, description="Process-corner override (OTFT typical/slow/fast or a silicon corner)")


class ExploreJobRequest(BaseModel):
    """Body of ``POST /api/v1/jobs/explore`` — same semantics as ``circuit-opt explore``.

    ``circuit`` is a full circuit-JSON object carrying an ``explore`` block; ``n``
    is the candidate count, ``seed`` the RNG seed, ``corner`` an optional process
    corner. All but ``circuit`` are optional (server defaults apply)."""
    circuit: dict[str, Any] = Field(..., description="Circuit JSON with an 'explore' block")
    n: Optional[int] = Field(None, description="Number of candidates to sample")
    seed: Optional[int] = Field(None, description="RNG seed")
    corner: Optional[str] = Field(None, description="Process-corner override")


class McJobRequest(BaseModel):
    """Body of ``POST /api/v1/jobs/mc`` — same semantics as ``circuit-opt mc``.

    ``circuit`` is a full circuit-JSON object; ``n`` is the MC sample count,
    ``seed`` the RNG seed, ``corner`` the base process corner (typical/slow/fast)."""
    circuit: dict[str, Any] = Field(..., description="Circuit JSON object")
    n: Optional[int] = Field(None, description="Number of MC samples")
    seed: Optional[int] = Field(None, description="RNG seed")
    corner: Optional[str] = Field(None, description="Base process corner (typical/slow/fast)")


# ── capabilities assembly ─────────────────────────────────────────────────────

def _build_capabilities() -> dict:
    """Assemble the self-description payload from the authoritative registries.

    Pulls model/PDK names from the device-model registry, analysis names + legal
    option keys from :mod:`circuitopt.analysis_options`, and the three corner
    families from their defining modules. Nothing here is hardcoded editorial
    content — it all reflects what the current build actually supports.
    """
    analyses = {name: sorted(known_keys(name)) for name in ANALYSIS_ORDER}
    return {
        "version": __version__,
        "api": "v1",
        "models": registered_models(),
        "analyses": analyses,
        "corners": {
            # OTFT continuous-PVT shift names (device_factory.CORNERS).
            "otft": sorted(CORNERS),
            # Silicon discrete corners baked into extracted device cards.
            "sky130": sorted(SKY130_CORNERS),
            "freepdk45": sorted(FREEPDK45_CORNERS),
        },
        # Long-running background job kinds a GUI can submit (see /api/v1/jobs/*).
        "jobs": list(JOB_KINDS),
    }


# ── app factory ───────────────────────────────────────────────────────────────

_LOCALHOST_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def create_app(job_workers: int = 1) -> FastAPI:
    """Build and return the ``/api/v1`` FastAPI application.

    CORS is restricted to ``localhost`` / ``127.0.0.1`` on any port (the Tauri
    dev front-end runs Vite on 5173, but the port is not pinned) — this is a
    local-only service, so no other origins are allowed.

    ``job_workers`` sizes the background-job thread pool (see :mod:`.jobs`); the
    conservative default of 1 keeps CPU-bound solves from oversubscribing the
    BLAS/Numba thread pools. The ``serve`` CLI's ``--job-workers`` sets it. The
    :class:`~circuitopt.service.jobs.JobManager` is placed on ``app.state.jobs`` so
    tests can inject their own.
    """
    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        # The JobManager (and its worker-thread pool) lives for the app's lifetime;
        # shutdown drains it so no worker thread outlives the server.
        app_.state.jobs = JobManager(workers=job_workers)
        try:
            yield
        finally:
            app_.state.jobs.shutdown()

    app = FastAPI(
        title="circuit-optimization service",
        version=__version__,
        summary="Local HTTP layer over the Cadence-calibrated circuit solvers.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_LOCALHOST_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "api": "v1"}

    @app.get("/api/v1/capabilities")
    def capabilities() -> dict:
        return _build_capabilities()

    @app.post("/api/v1/validate")
    def validate(circuit: dict[str, Any] = Body(...)) -> dict:
        """Parse a circuit + validate its analyses block.

        The validation *result* is the payload: both outcomes return HTTP 200,
        as ``{"valid": true}`` or ``{"valid": false, "errors": [...]}``. Error
        strings are the raw exception messages so a GUI can display them as-is.
        """
        errors: list[str] = []
        try:
            spec = circuit_from_dict(circuit)
        except Exception as exc:  # loader raises ValueError/TypeError on bad JSON
            return {"valid": False, "errors": [str(exc)]}

        # Per-analysis option-key validation (a residual/typo'd key is an error).
        for name, cfg in (spec.analyses or {}).items():
            if isinstance(cfg, dict):
                try:
                    validate_analysis_cfg(name, cfg)
                except Exception as exc:
                    errors.append(str(exc))

        if errors:
            return {"valid": False, "errors": errors}
        return {"valid": True}

    @app.post("/api/v1/solve")
    def solve(req: SolveRequest) -> dict:
        """Run the analysis suite and return JSON-safe results.

        Failures are surfaced as HTTP 422 with a ``{"stage": ..., "message": ...}``
        detail distinguishing a ``parse`` error (bad circuit structure) from a
        ``solve`` error (e.g. DC non-convergence). Tracebacks are never leaked.
        """
        # ── parse stage ──
        try:
            spec = circuit_from_dict(req.circuit)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail={"stage": "parse", "message": str(exc)},
            ) from exc

        # ── solve stage ──
        t0 = time.perf_counter()
        try:
            results = run_analysis_suite(spec, selected=req.selected, corner=req.corner)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail={"stage": "solve", "message": str(exc)},
            ) from exc
        elapsed = time.perf_counter() - t0

        return {"results": serialize_results(results), "elapsed_s": to_jsonable(elapsed)}

    # ── background jobs (explore / mismatch MC) ───────────────────────────────

    def _submit(kind: str, params: dict, response) -> dict:
        job = app.state.jobs.submit(kind, params)
        response.status_code = 202
        return {"job_id": job.id, "kind": job.kind, "status": job.status}

    @app.post("/api/v1/jobs/explore", status_code=202)
    def submit_explore(req: ExploreJobRequest, response: Response) -> dict:
        """Queue a design-space exploration (semantics of ``circuit-opt explore``).

        Returns 202 with ``{"job_id", "kind", "status"}``; poll ``GET
        /api/v1/jobs/{id}`` or stream ``WS /api/v1/jobs/{id}/events`` for progress
        and the final result. Only supplied fields override server defaults."""
        params: dict[str, Any] = {"circuit": req.circuit}
        if req.n is not None:
            params["n"] = req.n
        if req.seed is not None:
            params["seed"] = req.seed
        if req.corner is not None:
            params["corner"] = req.corner
        return _submit("explore", params, response)

    @app.post("/api/v1/jobs/mc", status_code=202)
    def submit_mc(req: McJobRequest, response: Response) -> dict:
        """Queue a mismatch Monte-Carlo (semantics of ``circuit-opt mc``). Returns
        202 with ``{"job_id", "kind", "status"}``; see ``submit_explore``."""
        params: dict[str, Any] = {"circuit": req.circuit}
        if req.n is not None:
            params["n"] = req.n
        if req.seed is not None:
            params["seed"] = req.seed
        if req.corner is not None:
            params["corner"] = req.corner
        return _submit("mc", params, response)

    @app.get("/api/v1/jobs")
    def list_jobs() -> dict:
        """List job status snapshots (newest first; no result/error payloads)."""
        return {"jobs": app.state.jobs.list()}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        """Full status of one job: snapshot + (once terminal) ``result`` or
        ``error``. Unknown id -> 404."""
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404,
                                detail={"stage": "job", "message": f"unknown job {job_id!r}"})
        out = job.snapshot()
        if job.result is not None:
            out["result"] = job.result
        if job.error is not None:
            out["error"] = job.error
        return out

    @app.delete("/api/v1/jobs/{job_id}")
    def cancel_job(job_id: str) -> dict:
        """Request cooperative cancellation. Unknown id -> 404; an
        already-terminal job -> 409 (nothing to cancel)."""
        try:
            state = app.state.jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail={"stage": "job", "message": f"unknown job {job_id!r}"}) from exc
        if state == "terminal":
            job = app.state.jobs.get(job_id)
            raise HTTPException(
                status_code=409,
                detail={"stage": "job",
                        "message": f"job {job_id!r} already terminal ({job.status})"})
        return {"job_id": job_id, "status": "cancelling"}

    @app.websocket("/api/v1/jobs/{job_id}/events")
    async def job_events(websocket: WebSocket, job_id: str) -> None:
        """Stream progress events for a job, then a terminal frame, then close.

        The job runs on a worker *thread* and pushes events onto a thread-safe
        ``queue.Queue``; this async endpoint bridges to it by draining the queue
        in a thread executor (``run_in_executor`` with a short timeout), so the
        event loop is never blocked by the solver. An unknown id closes
        immediately with an error frame. Events already emitted before the client
        connected are still in the queue, so a late subscriber does not miss them.
        """
        await websocket.accept()
        job = app.state.jobs.get(job_id)
        if job is None:
            await websocket.send_json({"type": "error", "message": f"unknown job {job_id!r}"})
            await websocket.close()
            return

        loop = asyncio.get_event_loop()
        q = job._events
        try:
            while True:
                try:
                    # Block the *worker* thread (not the loop) for up to 0.5s waiting
                    # for the next event; the timeout lets us re-check job.terminal
                    # even if the queue has drained (e.g. events consumed by polling).
                    event = await loop.run_in_executor(None, _drain_one, q, 0.5)
                except queue_Empty:
                    if job.terminal and q.empty():
                        # Terminal frame is always enqueued by the worker; if we
                        # got here the client connected after it was drained, so
                        # synthesize a final frame from the job's recorded state.
                        await websocket.send_json(_terminal_frame(job))
                        break
                    continue
                await websocket.send_json(event)
                if isinstance(event, dict) and event.get("type") == "terminal":
                    break
        except WebSocketDisconnect:
            return
        await websocket.close()

    return app


def _drain_one(q, timeout: float):
    """Blocking ``queue.get`` with a timeout — run in an executor by the WS route
    so a slow worker never stalls the event loop. Raises ``queue.Empty`` on
    timeout (imported as ``queue_Empty`` in the route)."""
    return q.get(timeout=timeout)


def _terminal_frame(job) -> dict:
    """A terminal WS frame reconstructed from a job's recorded state, for clients
    that subscribe after the worker already drained its own terminal event."""
    frame = {"type": "terminal", "status": job.status}
    if job.error is not None:
        frame["error"] = job.error
    return frame
