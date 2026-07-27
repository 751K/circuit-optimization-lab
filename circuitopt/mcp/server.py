"""Model Context Protocol server over the circuit solver application layer."""
from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

try:
    from mcp.server.fastmcp import Context, FastMCP
    from mcp.server.fastmcp.exceptions import ToolError
    from mcp.server.session import ServerSession
except ImportError as exc:
    raise ImportError(
        'the MCP server needs the optional SDK; '
        'pip install "circuit-optimization[mcp]"'
    ) from exc

from ..service.jobs import JOB_KINDS, JobManager
from ..service.operations import (
    OperationError,
    build_capabilities,
    solve_circuit,
    validate_circuit,
)
from ..passive_bom import DEFAULTS as DEFAULT_PROCESS
from ..passive_bom import build_report, load_decks
from ..service.serialize import to_jsonable
from ..signoff_campaign import run_signoff_campaign, summarize_margins
from .workspace import Workspace, WorkspaceError, write_json_atomic


MCP_JOB_KINDS = (*JOB_KINDS, "signoff")
_WORKFLOW = """\
Circuit Optimization MCP workflow:
1. Call get_capabilities and validate_circuit before simulation.
2. Call run_analysis for a bounded DC/AC/noise/transient result summary.
3. Use submit_exploration or submit_mismatch_mc for long candidate sweeps.
4. Use submit_signoff for a workspace-relative campaign manifest.
5. Poll get_job until done/failed/cancelled; cancel_job is cooperative.
6. Signoff details are written under results/mcp and can be queried with
   inspect_signoff_result. Invalid models or non-convergence are never replaced.
7. To decide what to change next, read signoff_margins: it ranks every spec by
   remaining margin, which worst_case alone does not tell you.
   inspect_signoff_result answers the separate question of where a spec failed.
8. passive_inventory sizes the passive area from deck JSON alone, no solver.

All paths are workspace-relative; absolute paths and '..' are rejected.
"""


@dataclass
class McpApplication:
    workspace: Workspace
    jobs: JobManager


McpContext = Context[ServerSession, McpApplication]


def _tool_error(exc: Exception) -> ToolError:
    if isinstance(exc, OperationError):
        payload = exc.as_dict()
    else:
        payload = {
            "stage": "request",
            "message": str(exc),
        }
    return ToolError(json.dumps(to_jsonable(payload), ensure_ascii=False))


def _compact(value: Any, *, max_items: int = 12, depth: int = 0) -> Any:
    """Bound nested numerical payloads while retaining scalar measurements."""
    if depth >= 7:
        return {"omitted": True, "reason": "maximum summary depth"}
    if isinstance(value, Mapping):
        return {
            str(key): _compact(child, max_items=max_items, depth=depth + 1)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        count = len(value)
        if count <= max_items:
            return [
                _compact(child, max_items=max_items, depth=depth + 1)
                for child in value
            ]
        half = max(1, max_items // 2)
        return {
            "sequence_summary": True,
            "count": count,
            "head": [
                _compact(child, max_items=max_items, depth=depth + 1)
                for child in value[:half]
            ],
            "tail": [
                _compact(child, max_items=max_items, depth=depth + 1)
                for child in value[-half:]
            ],
        }
    return to_jsonable(value)


def _solve_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": payload["status"],
        "elapsed_s": payload["elapsed_s"],
        "analyses": {
            name: _compact(result)
            for name, result in payload.get("results", {}).items()
        },
        "signoff": payload["signoff"],
    }


def _signoff_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: to_jsonable(payload.get(key))
        for key in (
            "schema_version",
            "name",
            "status",
            "passed",
            "stopped_early",
            "grid",
            "cases",
            "summary",
            "worst_case",
        )
        if key in payload
    }


def _run_signoff_job(
    params: dict[str, Any],
    emit: Callable[[dict], None],
    should_stop: Callable[[], bool],
) -> dict[str, Any]:
    def progress(done: int, total: int) -> None:
        emit({
            "type": "progress",
            "done": done,
            "total": total,
            "frac": (done / total) if total else 1.0,
        })

    result = run_signoff_campaign(
        params["campaign_path"],
        workers=int(params.get("workers", 1)),
        progress=progress,
        should_stop=should_stop,
    )
    write_json_atomic(Path(params["output_path"]), to_jsonable(result))
    return {
        **_signoff_summary(result),
        "result_path": params["result_path"],
    }


def _job_payload(
    app: McpApplication,
    job_id: str,
    *,
    include_result: bool,
    save_result: bool,
) -> dict[str, Any]:
    job = app.jobs.get(job_id)
    if job is None:
        raise WorkspaceError(f"unknown job {job_id!r}")
    payload = job.snapshot()
    if job.error is not None:
        payload["error"] = job.error
    if job.result is not None and include_result:
        payload["result"] = _compact(job.result)
    if job.result is not None and save_result:
        existing = (
            job.result.get("result_path")
            if isinstance(job.result, Mapping) else None
        )
        payload["result_path"] = existing or app.workspace.write_json_artifact(
            f"{job.kind}-{job.id}", job.result
        )
    return to_jsonable(payload)


def create_mcp_server(
    *,
    workspace: str | Path = ".",
    job_workers: int = 1,
    host: str = "127.0.0.1",
    port: int = 8342,
) -> FastMCP:
    """Build an MCP server bound to one local workspace."""
    workspace_root = Workspace(workspace)

    @asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncIterator[McpApplication]:
        jobs = JobManager(
            workers=job_workers,
            runners={"signoff": _run_signoff_job},
        )
        try:
            yield McpApplication(workspace=workspace_root, jobs=jobs)
        finally:
            jobs.shutdown()

    server = FastMCP(
        name="circuit-optimization",
        instructions=(
            "Local transistor-level simulation, optimization, and PVT signoff. "
            "Validate before solving; treat invalid results as failed candidates."
        ),
        host=host,
        port=port,
        json_response=True,
        stateless_http=False,
        lifespan=lifespan,
    )

    @server.resource(
        "circuitopt://capabilities",
        name="capabilities",
        mime_type="application/json",
    )
    def capabilities_resource() -> str:
        return json.dumps(
            build_capabilities(jobs=MCP_JOB_KINDS),
            indent=2,
            ensure_ascii=False,
        )

    @server.resource(
        "circuitopt://workflow",
        name="workflow",
        mime_type="text/plain",
    )
    def workflow_resource() -> str:
        return _WORKFLOW

    @server.tool()
    def get_capabilities() -> dict[str, Any]:
        """List installed models, PDK corners, analyses, options, and job kinds."""
        return build_capabilities(jobs=MCP_JOB_KINDS)

    @server.tool(name="validate_circuit")
    def validate_circuit_tool(circuit: dict[str, Any]) -> dict[str, Any]:
        """Validate circuit JSON, analysis options, and strict signoff contracts."""
        return validate_circuit(circuit)

    @server.tool()
    def run_analysis(
        circuit: dict[str, Any],
        ctx: McpContext,
        selected: list[str] | None = None,
        corner: str | None = None,
        save_result: bool = False,
    ) -> dict[str, Any]:
        """Run analyses synchronously and return a bounded, unit-bearing summary.

        Set save_result to write the complete JSON-safe result under results/mcp.
        """
        try:
            result = solve_circuit(circuit, selected=selected, corner=corner)
            summary = _solve_summary(result)
            if save_result:
                summary["result_path"] = (
                    ctx.request_context.lifespan_context.workspace
                    .write_json_artifact("analysis", result)
                )
            return summary
        except (OperationError, WorkspaceError, OSError, ValueError) as exc:
            raise _tool_error(exc) from exc

    @server.tool()
    def submit_exploration(
        circuit: dict[str, Any],
        ctx: McpContext,
        n: int = 32,
        seed: int = 0,
        corner: str | None = None,
        workers: int = 1,
    ) -> dict[str, Any]:
        """Submit design-space exploration as a cancellable background job."""
        if n < 1 or workers < 1:
            raise ToolError("n and workers must be at least 1")
        params: dict[str, Any] = {
            "circuit": circuit, "n": n, "seed": seed, "workers": workers
        }
        if corner is not None:
            params["corner"] = corner
        job = ctx.request_context.lifespan_context.jobs.submit("explore", params)
        return job.snapshot()

    @server.tool()
    def submit_mismatch_mc(
        circuit: dict[str, Any],
        ctx: McpContext,
        n: int = 64,
        seed: int = 0,
        corner: str = "typical",
        workers: int = 1,
    ) -> dict[str, Any]:
        """Submit mismatch Monte Carlo as a cancellable background job."""
        if n < 1 or workers < 1:
            raise ToolError("n and workers must be at least 1")
        job = ctx.request_context.lifespan_context.jobs.submit(
            "mc",
            {
                "circuit": circuit,
                "n": n,
                "seed": seed,
                "corner": corner,
                "workers": workers,
            },
        )
        return job.snapshot()

    @server.tool()
    def submit_signoff(
        campaign_path: str,
        ctx: McpContext,
        workers: int = 1,
    ) -> dict[str, Any]:
        """Submit a workspace-relative multi-testbench PVT signoff campaign."""
        if workers < 1:
            raise ToolError("workers must be at least 1")
        app = ctx.request_context.lifespan_context
        try:
            campaign = app.workspace.resolve_input(
                campaign_path, suffixes=(".json",)
            )
            output, result_path = app.workspace.artifact_path("signoff")
        except WorkspaceError as exc:
            raise _tool_error(exc) from exc
        job = app.jobs.submit(
            "signoff",
            {
                "campaign_path": str(campaign),
                "workers": workers,
                "output_path": str(output),
                "result_path": result_path,
            },
        )
        return {**job.snapshot(), "result_path": result_path}

    @server.tool()
    def list_jobs(ctx: McpContext) -> dict[str, Any]:
        """List newest-first background job status snapshots."""
        return {"jobs": ctx.request_context.lifespan_context.jobs.list()}

    @server.tool()
    def get_job(
        job_id: str,
        ctx: McpContext,
        include_result: bool = False,
        save_result: bool = False,
    ) -> dict[str, Any]:
        """Poll a job; optionally include a bounded result or save full JSON."""
        try:
            return _job_payload(
                ctx.request_context.lifespan_context,
                job_id,
                include_result=include_result,
                save_result=save_result,
            )
        except (WorkspaceError, OSError, ValueError) as exc:
            raise _tool_error(exc) from exc

    @server.tool()
    def cancel_job(job_id: str, ctx: McpContext) -> dict[str, Any]:
        """Request cooperative cancellation of a queued or running job."""
        jobs = ctx.request_context.lifespan_context.jobs
        try:
            state = jobs.cancel(job_id)
        except KeyError as exc:
            raise ToolError(f"unknown job {job_id!r}") from exc
        if state == "terminal":
            job = jobs.get(job_id)
            raise ToolError(
                f"job {job_id!r} is already terminal ({job.status})"
            )
        return {"job_id": job_id, "status": "cancelling"}

    @server.tool()
    def inspect_signoff_result(
        result_path: str,
        ctx: McpContext,
        case: str | None = None,
        corner: str | None = None,
        temperature_c: float | None = None,
        supply_v: float | None = None,
        max_points: int = 10,
    ) -> dict[str, Any]:
        """Inspect selected failures from a saved signoff result without loading it all."""
        if max_points < 1 or max_points > 50:
            raise ToolError("max_points must be between 1 and 50")
        try:
            path = ctx.request_context.lifespan_context.workspace.resolve_input(
                result_path, suffixes=(".json",)
            )
            with path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except (WorkspaceError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise _tool_error(exc) from exc

        matches = []
        matched_total = 0
        for point in result.get("points", []):
            pvt = point.get("pvt", {})
            if corner is not None and pvt.get("corner") != corner.lower():
                continue
            if (
                temperature_c is not None
                and pvt.get("temperature_c") != temperature_c
            ):
                continue
            if supply_v is not None and pvt.get("supply_v") != supply_v:
                continue
            selected_point = point
            if case is not None:
                case_result = point.get("cases", {}).get(case)
                if case_result is None:
                    continue
                selected_point = {
                    "pvt": pvt,
                    "case": case_result,
                }
            matched_total += 1
            if len(matches) < max_points:
                matches.append(_compact(selected_point))
        return {
            **_signoff_summary(result),
            "matched_points": matches,
            "match_count": matched_total,
            "match_count_returned": len(matches),
            "truncated": matched_total > len(matches),
        }

    @server.tool()
    def signoff_margins(
        result_path: str,
        ctx: McpContext,
        max_rows: int = 20,
    ) -> dict[str, Any]:
        """Per-constraint margin table from a saved signoff result.

        ``worst_case`` names the single tightest measurement, which ranks runs
        but does not describe a design: a spec sitting just inside its limit is
        still a constraint when some other spec is the formal worst. Rows come
        back tightest-margin first, each with the PVT point holding that
        margin, the observed range, and every failing point.
        """
        if max_rows < 1 or max_rows > 100:
            raise ToolError("max_rows must be between 1 and 100")
        try:
            path = ctx.request_context.lifespan_context.workspace.resolve_input(
                result_path, suffixes=(".json",)
            )
            with path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            rows = summarize_margins(result)
        except (WorkspaceError, OSError, ValueError, KeyError,
                json.JSONDecodeError) as exc:
            raise _tool_error(exc) from exc
        return {
            **_signoff_summary(result),
            "constraints": [_compact(row) for row in rows[:max_rows]],
            "constraint_count": len(rows),
            "truncated": len(rows) > max_rows,
        }

    @server.tool()
    def passive_inventory(
        paths: list[str],
        ctx: McpContext,
        sheet_ohm_sq: float = DEFAULT_PROCESS["sheet_ohm_sq"],
        resistor_width_um: float = DEFAULT_PROCESS["resistor_width_um"],
        cap_ff_um2: float = DEFAULT_PROCESS["cap_ff_um2"],
    ) -> dict[str, Any]:
        """Inventory every resistor and capacitor with a silicon-area estimate.

        Pass one signoff manifest, which already names every testbench wrapped
        around one DUT, or two or more circuit files. DUT-vs-testbench
        membership is derived from which decks an element appears in, so a
        single deck cannot be classified. Reads JSON only; it runs no solver.
        """
        if not paths:
            raise ToolError("paths must name at least one deck or manifest")
        workspace = ctx.request_context.lifespan_context.workspace
        try:
            resolved = [
                str(workspace.resolve_input(item, suffixes=(".json",)))
                for item in paths
            ]
            decks = load_decks(resolved)
            report = build_report(
                decks,
                sheet_ohm_sq=sheet_ohm_sq,
                resistor_width_um=resistor_width_um,
                cap_ff_um2=cap_ff_um2,
            )
        except SystemExit as exc:
            # load_decks reports its usage errors the CLI way.
            raise ToolError(str(exc)) from exc
        except (WorkspaceError, OSError, ValueError, KeyError,
                json.JSONDecodeError) as exc:
            raise _tool_error(exc) from exc
        return _compact(report, max_items=40)

    return server
