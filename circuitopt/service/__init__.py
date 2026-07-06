"""Local FastAPI service layer over the circuit solver stack (S1 base).

A **thin adapter**: it carries no numerical logic and makes no design judgement.
Requests hand circuit JSON straight through to the existing single sources of
truth — :func:`circuitopt.circuit_loader.circuit_from_dict` for the schema,
:mod:`circuitopt.analysis_options` for option validation, and
:func:`circuitopt.analysis_dispatch.run_analysis_suite` for execution. This is
the base the Tauri desktop GUI and the MCP server (S2+) sit on top of.

``fastapi`` / ``uvicorn`` are **optional** dependencies (the ``serve`` extra).
This ``__init__`` deliberately imports **neither** them nor :mod:`.app`, so
``import circuitopt`` and every existing module stay fastapi-free — the web
framework is only pulled in on the ``serve`` CLI path or by the tests that
exercise it. Reach the app factory lazily::

    from circuitopt.service.app import create_app   # imports fastapi here

The CLI glue (:func:`add_cli_args` / :func:`run_cli`) is re-exported below but
its ``run_cli`` imports fastapi/uvicorn only when actually invoked, so merely
touching this module still does not require the extra.

Install the extra with::

    pip install "circuit-optimization[serve]"
"""
from __future__ import annotations

from .cli import add_cli_args, run_cli

__all__ = ["add_cli_args", "run_cli"]
