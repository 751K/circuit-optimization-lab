#!/usr/bin/env bash
# Run an OSDI-enabled ngspice with any args.
#   tools/run-ngspice.sh -b netlist.cir
#
# ngspice here is a from-source (OSDI-enabled) build that is not on PATH; this
# wrapper is the single place that resolves its location. Override with
# NGSPICE_BIN if it moves. The default points at this machine's external-drive
# build; other checkouts should set NGSPICE_BIN.
set -euo pipefail

: "${NGSPICE_BIN:=/Volumes/MacoutDsik/ngspice/install/bin/ngspice}"

if [ ! -x "$NGSPICE_BIN" ]; then
    echo "run-ngspice: ngspice not found at $NGSPICE_BIN" >&2
    echo "run-ngspice: is the external drive mounted? set NGSPICE_BIN to override." >&2
    exit 1
fi

exec "$NGSPICE_BIN" "$@"
