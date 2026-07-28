"""Protocol-level tests for the optional Circuit Optimization MCP server."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import (  # noqa: E402
    create_connected_server_and_client_session,
)

from circuitopt.mcp.server import create_mcp_server  # noqa: E402
from circuitopt.mcp.workspace import Workspace, WorkspaceError  # noqa: E402


_ROOT = Path(__file__).resolve().parent.parent


def _periodic_rc() -> dict:
    return json.loads((_ROOT / "examples" / "periodic_rc.json").read_text())


def _call(coro):
    return asyncio.run(coro)


def _write_signoff_fixture(root: Path) -> None:
    circuit = {
        "name": "mcp_signoff_fixture",
        "solved": ["OUT"],
        "rails": {"VIN": "VIN", "GND": 0.0},
        "bias": {"VIN": 1.0},
        "devices": [],
        "resistors": [
            {"name": "R1", "a": "VIN", "b": "OUT", "R": 1e3},
            {"name": "R2", "a": "OUT", "b": "GND", "R": 1e3},
        ],
        "outputs": ["OUT"],
        "ac_drives": {"VIN": 1.0},
        "analyses": {
            "ac": {
                "freqs": {
                    "start": 1.0,
                    "stop": 10.0,
                    "num": 2,
                    "scale": "log",
                }
            }
        },
        "signoff": {
            "measurements": {},
            # Two constraints, so a margin table has something to order. With
            # one row the "tightest first" assertion is vacuous: a one-element
            # list is sorted whichever way the rows came out.
            "constraints": {
                "gain": {"min": -20.0},
                "dc_source_power": {"max": 1.0},
            },
        },
    }
    manifest = {
        "name": "mcp_campaign",
        "pvt": {
            "corners": ["tt"],
            "temperatures_c": [27.0],
            "supplies_v": [1.0],
            "nominal_supply_v": 1.0,
            "supply_bias_key": "VIN",
        },
        "cases": [
            {"name": "ac", "circuit": "circuit.json", "overrides": {}},
        ],
    }
    (root / "circuit.json").write_text(json.dumps(circuit), encoding="utf-8")
    (root / "campaign.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_workspace_rejects_absolute_and_parent_paths(tmp_path):
    workspace = Workspace(tmp_path)
    with pytest.raises(WorkspaceError, match="absolute"):
        workspace.resolve_input(str((tmp_path / "x.json").resolve()))
    with pytest.raises(WorkspaceError, match="parent path"):
        workspace.resolve_input("../x.json")


def test_protocol_lists_tools_resources_and_runs_analysis(tmp_path):
    async def scenario():
        server = create_mcp_server(workspace=tmp_path)
        async with create_connected_server_and_client_session(server) as session:
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {
                "get_capabilities",
                "validate_circuit",
                "run_analysis",
                "submit_signoff",
                "get_job",
                "cancel_job",
                "inspect_signoff_result",
            } <= names

            resources = await session.list_resources()
            assert {str(item.uri) for item in resources.resources} == {
                "circuitopt://capabilities",
                "circuitopt://workflow",
            }

            capabilities = await session.call_tool("get_capabilities", {})
            assert not capabilities.isError
            advertised = capabilities.structuredContent["jobs"]
            assert advertised == ["explore", "mc", "pvt", "signoff"]
            # Every advertised kind must have a tool that can submit it. A kind
            # with no submit tool reads as available and is not.
            submitters = {"explore": "submit_exploration", "mc": "submit_mismatch_mc",
                          "pvt": "submit_pvt", "signoff": "submit_signoff"}
            assert {submitters[kind] for kind in advertised} <= names

            validation = await session.call_tool(
                "validate_circuit", {"circuit": _periodic_rc()}
            )
            assert not validation.isError
            assert validation.structuredContent["valid"] is True
            # A parseable circuit also reports the corner set it admits.
            assert validation.structuredContent["corners"] == [
                "typical", "slow", "fast",
            ]

            solved = await session.call_tool(
                "run_analysis",
                {"circuit": _periodic_rc(), "selected": ["ac"]},
            )
            assert not solved.isError
            payload = solved.structuredContent
            assert payload["status"] == "valid"
            assert set(payload["analyses"]) == {"ac"}
            assert payload["signoff"]["status"] == "not_configured"
            assert len(payload["analyses"]["ac"]["freqs"]) == 2

    _call(scenario())


def test_signoff_job_writes_artifact_and_is_inspectable(tmp_path):
    _write_signoff_fixture(tmp_path)

    async def scenario():
        server = create_mcp_server(workspace=tmp_path)
        async with create_connected_server_and_client_session(server) as session:
            submitted = await session.call_tool(
                "submit_signoff",
                {"campaign_path": "campaign.json", "workers": 1},
            )
            assert not submitted.isError
            job_id = submitted.structuredContent["job_id"]

            terminal = None
            for _ in range(200):
                polled = await session.call_tool(
                    "get_job",
                    {"job_id": job_id, "include_result": True},
                )
                assert not polled.isError
                terminal = polled.structuredContent
                if terminal["status"] in {"done", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.01)
            assert terminal is not None
            assert terminal["status"] == "done"
            result_path = terminal["result"]["result_path"]
            assert result_path.startswith("results/mcp/signoff-")
            assert (tmp_path / result_path).is_file()

            inspected = await session.call_tool(
                "inspect_signoff_result",
                {"result_path": result_path, "case": "ac"},
            )
            assert not inspected.isError
            details = inspected.structuredContent
            assert details["status"] == "pass"
            assert details["match_count"] == 1
            assert details["matched_points"][0]["pvt"]["corner"] == "tt"

    _call(scenario())


def test_signoff_path_cannot_escape_workspace(tmp_path):
    async def scenario():
        server = create_mcp_server(workspace=tmp_path)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "submit_signoff",
                {"campaign_path": "../campaign.json"},
            )
            assert result.isError
            assert "parent path" in result.content[0].text

    _call(scenario())


def test_signoff_margins_ranks_every_constraint(tmp_path):
    """An agent decides what to change next from margins, not from pass/fail.

    ``inspect_signoff_result`` answers "where did it fail". This answers the
    question that comes first: which spec is closest to failing at all.
    """
    _write_signoff_fixture(tmp_path)

    async def scenario():
        server = create_mcp_server(workspace=tmp_path)
        async with create_connected_server_and_client_session(server) as session:
            submitted = await session.call_tool(
                "submit_signoff", {"campaign_path": "campaign.json"})
            job_id = submitted.structuredContent["job_id"]
            for _ in range(200):
                polled = await session.call_tool(
                    "get_job", {"job_id": job_id, "include_result": True})
                terminal = polled.structuredContent
                if terminal["status"] in {"done", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.01)
            assert terminal["status"] == "done"
            result_path = terminal["result"]["result_path"]

            margins = await session.call_tool(
                "signoff_margins", {"result_path": result_path})
            assert not margins.isError
            payload = margins.structuredContent
            assert payload["status"] == "pass"
            assert payload["constraint_count"] >= 1
            row = payload["constraints"][0]
            assert {"case", "constraint", "worst_margin", "worst_point"} <= set(row)
            # Tightest first is the whole point of the ordering.
            margin_values = [
                item["worst_margin"] for item in payload["constraints"]
                if item.get("worst_margin") is not None
            ]
            assert margin_values == sorted(margin_values)

    _call(scenario())


def test_signoff_margins_rejects_paths_outside_the_workspace(tmp_path):
    async def scenario():
        server = create_mcp_server(workspace=tmp_path)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "signoff_margins", {"result_path": "../escape.json"})
            assert result.isError

    _call(scenario())


def test_passive_inventory_needs_more_than_one_deck(tmp_path):
    """DUT-vs-testbench membership is derived from several decks around one DUT.

    A single deck cannot be classified, so the tool must say so rather than
    silently calling every element a DUT component.
    """
    _write_signoff_fixture(tmp_path)
    # A second testbench around the same DUT: R1/R2 are in both decks and so
    # are DUT parts, RPROBE appears only here and so is testbench.
    second = json.loads((tmp_path / "circuit.json").read_text())
    second["resistors"] = [*second["resistors"],
                           {"name": "RPROBE", "a": "OUT", "b": "GND", "R": 1e6}]
    (tmp_path / "circuit_probe.json").write_text(json.dumps(second),
                                                 encoding="utf-8")

    async def scenario():
        server = create_mcp_server(workspace=tmp_path)
        async with create_connected_server_and_client_session(server) as session:
            single = await session.call_tool(
                "passive_inventory", {"paths": ["circuit.json"]})
            assert single.isError
            assert "two decks" in single.content[0].text

            both = await session.call_tool(
                "passive_inventory",
                {"paths": ["circuit.json", "circuit_probe.json"]})
            assert not both.isError
            report = both.structuredContent
            names = {row["name"]: row for row in report["rows"]}
            assert names["R1"]["dut"] is True
            assert names["RPROBE"]["dut"] is False

    _call(scenario())
