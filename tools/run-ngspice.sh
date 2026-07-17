#!/usr/bin/env bash
# Run the configured ngspice binary with any args.
#   tools/run-ngspice.sh -b netlist.cir
#
# Resolution order: NGSPICE_BIN, active venv, project .venv, then PATH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -n "${NGSPICE_BIN:-}" ]; then
    BIN="$(command -v "$NGSPICE_BIN" 2>/dev/null || true)"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/ngspice/bin/ngspice" ]; then
    BIN="$VIRTUAL_ENV/ngspice/bin/ngspice"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/ngspice" ]; then
    BIN="$VIRTUAL_ENV/bin/ngspice"
elif [ -x "$REPO_ROOT/.venv/ngspice/bin/ngspice" ]; then
    BIN="$REPO_ROOT/.venv/ngspice/bin/ngspice"
else
    BIN="$(command -v ngspice 2>/dev/null || true)"
fi

if [ -z "${BIN:-}" ] || [ ! -x "$BIN" ]; then
    echo "run-ngspice: ngspice not found (checked NGSPICE_BIN, active/project venv, PATH)." >&2
    exit 1
fi

exec "$BIN" "$@"
