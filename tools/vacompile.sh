#!/usr/bin/env bash
# Compile a Verilog-A model to .osdi with an already-built openvaf-r.
#
#   tools/vacompile.sh model.va -o out.osdi
#
# The openvaf-r compiler itself lives outside this repo (it is large and
# machine-specific). Resolution order:
#   1. $OPENVAF_BIN  — explicit path to the openvaf-r binary, if set.
#   2. $OPENVAF_ROOT/target/release/openvaf-r
#      ($OPENVAF_ROOT defaults to the checkout on this machine's external drive).
#
# Runs the compiler with a clean PATH so the generated .osdi is linked by
# Apple's system clang (which knows the macOS SDK). Prepending LLVM's bin to
# PATH here causes `ld: library 'System' not found` — a macOS-specific gotcha,
# so the clean-PATH exec below must be preserved.
set -euo pipefail

if [ -n "${OPENVAF_BIN:-}" ]; then
    BIN="$OPENVAF_BIN"
else
    OPENVAF_ROOT="${OPENVAF_ROOT:-/Volumes/MacoutDsik/Code/VAF/OpenVAF-Reloaded}"
    BIN="$OPENVAF_ROOT/target/release/openvaf-r"
fi

if [ ! -x "$BIN" ]; then
    echo "vacompile: openvaf-r not found at $BIN" >&2
    echo "vacompile: set OPENVAF_BIN to the binary, or OPENVAF_ROOT to its" >&2
    echo "vacompile: checkout (expects \$OPENVAF_ROOT/target/release/openvaf-r)." >&2
    exit 1
fi

# Force Apple's toolchain by using a standard system PATH (no LLVM bin dir).
exec env PATH="/usr/bin:/bin:/usr/sbin:/sbin" "$BIN" "$@"
