"""Optional Model Context Protocol adapter for Circuit Optimization.

Importing this package does not import the optional ``mcp`` dependency.  The
SDK is loaded only when the server is constructed or its CLI is started.
"""
from __future__ import annotations

from .cli import add_cli_args, run_cli

__all__ = ["add_cli_args", "run_cli"]
