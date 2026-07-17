"""Compute-engine selection switch.

CircuitOpt can run its numerical hot paths on one of three engines:

* ``"numba"``  — JIT-compiled scalar kernels (the default).
* ``"python"`` — the pure-Python fallback (numba disabled); handy for
  debugging and for environments without numba.
* ``"rust"``   — the compiled Rust core (the ``circuitopt_core`` extension).

Selection precedence is **argv > ``CIRCUIT_ENGINE`` env > default ("numba")**.
``apply_engine_env()`` runs once, at the very top of ``circuitopt/__init__.py``
— before any solver import pulls in ``circuitopt.numba_kernels`` (which bakes
its ``USE_NUMBA`` flag from ``CIRCUIT_USE_NUMBA`` at import time). It resolves
the requested engine, writes the result back to ``CIRCUIT_ENGINE`` for child
processes, and — for the pure-Python engine — reuses the existing
``CIRCUIT_USE_NUMBA=0`` kill-switch.

R4 dispatches OTFT scalar evaluation, fixed/adaptive transient, BSIM4 transient,
AC/noise MNA, periodic HB/PAC linearization, and scalar PNoise folding into the
Rust core. SciPy sparse solves and FFT orchestration remain in Python. Requesting
``"rust"`` warns and falls back to ``"numba"`` only when the extension cannot be
imported.
"""
from __future__ import annotations

import os
import sys
import warnings

# The complete set of legal engine names.
_VALID_ENGINES = ("rust", "numba", "python")

# Resolved once by apply_engine_env(); read by current_engine()/engine_info().
# ``_requested_engine`` keeps the *asked-for* value (e.g. "rust") even after a
# fallback, so engine_info() can report both what was requested and what runs.
_resolved_engine: str | None = None
_requested_engine: str | None = None


def _fail(message: str) -> None:
    """Print a clear error and exit with status 2 (argparse's usage-error code)."""
    print(f"circuitopt: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _scan_argv_engine(argv: list[str]) -> str | None:
    """Return the ``--engine`` value in ``argv`` (last one wins), else ``None``.

    Accepts both ``--engine VALUE`` and ``--engine=VALUE``. Raises
    ``SystemExit(2)`` if ``--engine`` appears with no following value. The value
    itself is validated by the caller against ``_VALID_ENGINES``.
    """
    value: str | None = None
    i = 0
    n = len(argv)
    while i < n:
        token = argv[i]
        if token == "--engine":
            if i + 1 >= n:
                _fail("argument --engine: expected one argument")
            value = argv[i + 1]
            i += 2
            continue
        if token.startswith("--engine="):
            value = token[len("--engine="):]
        i += 1
    return value


def _resolve(requested: str) -> str:
    """Map the requested engine to the one that will actually run.

    Only ``"rust"`` can differ from the request: it falls back to ``"numba"``
    (with exactly one ``RuntimeWarning``) when ``circuitopt_core`` is missing.
    """
    if requested != "rust":
        return requested
    try:
        import circuitopt_core  # noqa: F401  (import is the availability probe)
    except Exception:
        warnings.warn(
            "CIRCUIT_ENGINE=rust requested but circuitopt_core is not "
            "installed; falling back to numba",
            RuntimeWarning,
            stacklevel=3,
        )
        return "numba"
    return "rust"


def apply_engine_env(argv: list[str] | None = None) -> str:
    """Resolve the engine from argv/env and wire the process for it.

    Called once from ``circuitopt/__init__.py`` before the solver imports. Steps
    (matching the documented precedence argv > CIRCUIT_ENGINE > "numba"):

    a. Pre-scan argv for ``--engine X`` / ``--engine=X``; an illegal value exits 2.
    b. ``--no-numba`` forces the ``"python"`` engine; combining it with an argv
       ``--engine`` other than ``python`` is a conflict and exits 2.
    c. Write the *resolved* engine back to ``CIRCUIT_ENGINE`` (children inherit it
       and thus never re-attempt — or re-warn about — an unavailable rust core).
    d. The ``"python"`` engine reuses the existing ``CIRCUIT_USE_NUMBA=0`` switch.
    e. A ``"rust"`` request resolves to ``"rust"`` when ``circuitopt_core`` imports,
       otherwise warns once and resolves to ``"numba"``.

    Returns the resolved engine name.
    """
    global _resolved_engine, _requested_engine
    if argv is None:
        argv = sys.argv

    argv_engine = _scan_argv_engine(argv)
    no_numba = "--no-numba" in argv

    if no_numba:
        # --no-numba is an argv-level request for the pure-Python engine; it
        # outranks CIRCUIT_ENGINE but must not silently contradict an explicit
        # argv --engine.
        if argv_engine is not None and argv_engine != "python":
            _fail(
                f"--no-numba conflicts with --engine {argv_engine}: --no-numba "
                "selects the pure-Python engine, so --engine must be 'python' "
                "or omitted"
            )
        requested = "python"
    elif argv_engine is not None:
        requested = argv_engine
    else:
        env_engine = os.environ.get("CIRCUIT_ENGINE")
        requested = env_engine if env_engine else "numba"

    if requested not in _VALID_ENGINES:
        _fail(
            f"invalid engine {requested!r}; choose from "
            f"{', '.join(_VALID_ENGINES)}"
        )

    _requested_engine = requested
    resolved = _resolve(requested)
    _resolved_engine = resolved

    # (c) resolved result written back for child processes.
    os.environ["CIRCUIT_ENGINE"] = resolved
    # (d) pure-Python engine reuses the numba kill-switch that numba_kernels reads.
    if resolved == "python":
        os.environ["CIRCUIT_USE_NUMBA"] = "0"
    return resolved


def current_engine() -> str:
    """Return the engine that is actually active (``"numba"`` after a fallback)."""
    if _resolved_engine is None:
        # Defensive: any ``import circuitopt`` runs apply_engine_env() first, so
        # this only triggers if _engine is driven in isolation.
        apply_engine_env()
    return _resolved_engine  # type: ignore[return-value]


def engine_info() -> dict:
    """Return ``{"engine": <active>, "requested": <asked-for>, "core": <dict|None>}``.

    ``core`` carries ``circuitopt_core.engine_info()`` when the rust core is live,
    otherwise ``None``.
    """
    engine = current_engine()
    info: dict = {"engine": engine, "requested": _requested_engine, "core": None}
    if engine == "rust":
        try:
            import circuitopt_core

            info["core"] = circuitopt_core.engine_info()
        except Exception:
            info["core"] = None
    return info
