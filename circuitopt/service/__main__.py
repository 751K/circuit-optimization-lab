"""``python -m circuitopt.service`` — start the local solver HTTP service.

Standalone entry point mirroring the ``circuitopt serve`` subcommand; both wire
their args from :func:`circuitopt.service.cli.add_cli_args` and dispatch through
:func:`circuitopt.service.cli.run_cli`, so they can't drift.
"""
from __future__ import annotations

import argparse

from .cli import add_cli_args, run_cli


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m circuitopt.service",
        description="Start the local FastAPI service over the circuit solvers.",
    )
    add_cli_args(parser)
    run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    main()
