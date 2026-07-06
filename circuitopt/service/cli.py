"""CLI glue for the ``serve`` subcommand (single source for arg wiring).

Both ``python -m circuitopt.service`` (see :mod:`circuitopt.service.__main__`) and the
``circuitopt serve`` subcommand in :mod:`circuitopt.__main__` add their arguments from
:func:`add_cli_args` and dispatch through :func:`run_cli`, so the two entry
points can never drift — mirroring the explore/dataset single-source pattern.

``fastapi`` / ``uvicorn`` are optional (the ``serve`` extra); they are imported
lazily *inside* :func:`run_cli`, so importing this module (which ``__init__``
does eagerly) never requires the extra.
"""
from __future__ import annotations

import argparse


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the ``serve`` arguments on *parser*."""
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1, loopback only). Setting 0.0.0.0 "
             "exposes the solver on the network — at your own risk.")
    parser.add_argument(
        "--port", type=int, default=8341,
        help="TCP port to listen on (default: 8341)")
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable uvicorn auto-reload (development only; default: off)")
    parser.add_argument(
        "--job-workers", type=int, default=1,
        help="Background-job worker threads for explore/mc jobs (default: 1). "
             "Solves are CPU-bound; raise only if you have spare cores and want "
             "concurrent jobs.")
    return parser


def run_cli(args) -> None:
    """Start the uvicorn server programmatically from parsed *args*."""
    try:
        import uvicorn
    except ImportError as exc:  # optional dependency (serve extra)
        raise SystemExit(
            'the serve command needs uvicorn; pip install "circuit-optimization[serve]"'
        ) from exc

    from .app import create_app

    if args.reload:
        # uvicorn's reloader re-imports the app from an import string, so hand it
        # the factory path rather than a live app instance. The import-string
        # factory takes no args, so --job-workers is ignored here (reload is a
        # dev-only mode and defaults to a single job worker).
        uvicorn.run(
            "circuitopt.service.app:create_app",
            factory=True,
            host=args.host, port=args.port, reload=True,
        )
    else:
        uvicorn.run(create_app(job_workers=args.job_workers),
                    host=args.host, port=args.port)
