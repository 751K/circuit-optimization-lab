#!/usr/bin/env bash
# Compile a Verilog-A model to .osdi with an already-built openvaf-r.
#
#   tools/vacompile.sh model.va -o out.osdi
#
# The openvaf-r compiler itself lives outside this repo (it is large and
# machine-specific). Resolution order:
#   1. $OPENVAF_BIN  — explicit path to the openvaf-r binary, if set.
#   2. $OPENVAF_ROOT/target/release/openvaf-r
#   3. active/project virtual environment, then PATH.
#
# Runs the compiler with a clean PATH so the generated .osdi is linked by
# Apple's system clang (which knows the macOS SDK). Prepending LLVM's bin to
# PATH here causes `ld: library 'System' not found` — a macOS-specific gotcha,
# so the clean-PATH exec below must be preserved.
set -euo pipefail

if [ -n "${OPENVAF_BIN:-}" ]; then
    BIN="$(command -v "$OPENVAF_BIN" 2>/dev/null || true)"
elif [ -n "${OPENVAF_ROOT:-}" ]; then
    BIN="$OPENVAF_ROOT/target/release/openvaf-r"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/openvaf-r" ]; then
    BIN="$VIRTUAL_ENV/bin/openvaf-r"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/openvaf/bin/openvaf-r" ]; then
    BIN="$VIRTUAL_ENV/openvaf/bin/openvaf-r"
elif [ -x "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin/openvaf-r" ]; then
    BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin/openvaf-r"
elif [ -x "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/openvaf/bin/openvaf-r" ]; then
    BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/openvaf/bin/openvaf-r"
else
    BIN="$(command -v openvaf-r 2>/dev/null || true)"
fi

if [ -z "${BIN:-}" ] || [ ! -x "$BIN" ]; then
    echo "vacompile: openvaf-r not found (checked OPENVAF_BIN/ROOT, venv, PATH)." >&2
    exit 1
fi

# Force Apple's toolchain by using a standard system PATH (no LLVM bin dir).
exec env PATH="/usr/bin:/bin:/usr/sbin:/sbin" "$BIN" "$@"
