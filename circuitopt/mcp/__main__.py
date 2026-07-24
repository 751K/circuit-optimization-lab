"""Run the Circuit Optimization MCP server."""
from __future__ import annotations

import argparse

from .cli import add_cli_args, run_cli


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m circuitopt.mcp",
        description="Start the local Circuit Optimization MCP server.",
    )
    add_cli_args(parser)
    run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    main()
