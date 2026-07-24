"""CLI wiring for the optional MCP server."""
from __future__ import annotations

import argparse
from pathlib import Path


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Root allowed for MCP file access (default: current directory)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Streamable HTTP bind address (loopback only; default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8342,
        help="Streamable HTTP port (default: 8342)",
    )
    parser.add_argument(
        "--job-workers",
        type=int,
        default=1,
        help="Concurrent background explore/MC/signoff jobs (default: 1)",
    )
    return parser


def run_cli(args) -> None:
    if args.job_workers < 1:
        raise SystemExit("--job-workers must be at least 1")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.transport == "streamable-http" and args.host not in _LOOPBACK_HOSTS:
        raise SystemExit(
            "the unauthenticated MCP server only binds to loopback "
            "(127.0.0.1, localhost, or ::1)"
        )
    try:
        from .server import create_mcp_server
    except ImportError as exc:
        raise SystemExit(
            'the mcp command needs the optional SDK; '
            'pip install "circuit-optimization[mcp]"'
        ) from exc

    server = create_mcp_server(
        workspace=Path(args.workspace),
        job_workers=args.job_workers,
        host=args.host,
        port=args.port,
    )
    server.run(transport=args.transport)
