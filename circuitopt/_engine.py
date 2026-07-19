"""Compute-engine selection switch.

As of v2.0.0 CircuitOpt runs its numerical work on a single engine — ``"rust"``,
the compiled ``circuitopt_core`` core. It is the only supported value; there is
no Python or numba fallback (see ``docs/rust_core_rewrite_plan.md`` §R6 and the
v2.0.0 CHANGELOG entry).

``apply_engine_env()`` runs once, at the very top of ``circuitopt/__init__.py``
— before any solver import — and resolves the engine to ``"rust"``. The
``--engine`` CLI flag and ``CIRCUIT_ENGINE`` env var are retained (the §7
compatibility contract keeps the flag names), but their value domain has shrunk
to ``{rust}``: any other value is a hard error that points at the CHANGELOG.

Removed in v2.0.0 (each is now a hard error, not a silent no-op):

* ``--engine python`` / ``--engine numba`` / ``CIRCUIT_ENGINE=python|numba``
  — the pure-Python reference engine and the numba JIT engine were both removed;
  the frozen golden corpus (``tests/golden/engine_parity``) is the reference
  oracle now.
* ``--no-numba``            — the numba engine it disabled no longer exists.
* ``CIRCUIT_USE_NUMBA``     — same.
"""
from __future__ import annotations

import os
import sys
import warnings

# v2.0.0: rust is the sole engine.
_VALID_ENGINES = ("rust",)
_DEFAULT_ENGINE = "rust"

# Engine names that used to be valid and are now removed, with a one-line reason.
_REMOVED_ENGINES = {
    "python": "the pure-Python reference engine was removed",
    "numba": "the numba JIT engine was removed",
}

_CHANGELOG_HINT = (
    "rust is the only engine in v2.0.0; see the v2.0.0 entry in CHANGELOG.md "
    "(\"Python reference engine removed\")"
)

# Resolved once by apply_engine_env(); read by current_engine()/engine_info().
_resolved_engine: str | None = None
_requested_engine: str | None = None


def _fail(message: str) -> None:
    """Print a clear error and exit with status 2 (argparse's usage-error code)."""
    print(f"circuitopt: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _removed(what: str, reason: str) -> None:
    """Exit 2 with a removal message that points at the CHANGELOG."""
    _fail(f"{what} was removed in v2.0.0: {reason}. {_CHANGELOG_HINT}.")


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


def _validate_engine(value: str, *, source: str) -> str:
    """Return ``value`` if it is the sole supported engine, else exit 2.

    A retired engine name (``python``/``numba``) gets a removal message; any
    other unknown value gets a generic invalid-choice message.
    """
    if value in _VALID_ENGINES:
        return value
    if value in _REMOVED_ENGINES:
        _removed(f"{source}={value}", _REMOVED_ENGINES[value])
    _fail(f"invalid engine {value!r}; the only supported engine is 'rust'")


def apply_engine_env(argv: list[str] | None = None) -> str:
    """Resolve the engine from argv/env and wire the process for it.

    Called once from ``circuitopt/__init__.py`` before the solver imports. The
    engine is always ``"rust"``; this function's job is to reject the retired
    switches loudly (rather than silently ignoring them) and to write the
    resolved engine back to ``CIRCUIT_ENGINE`` for child processes.
    """
    global _resolved_engine, _requested_engine
    if argv is None:
        argv = sys.argv

    # Retired switches → hard errors (they used to select a now-removed engine).
    if "--no-numba" in argv:
        _removed("--no-numba", "the numba engine it disabled no longer exists")
    if os.environ.get("CIRCUIT_USE_NUMBA") is not None:
        _removed("the CIRCUIT_USE_NUMBA env var",
                 "the numba engine it toggled no longer exists")

    argv_engine = _scan_argv_engine(argv)
    if argv_engine is not None:
        requested = _validate_engine(argv_engine, source="--engine")
    else:
        env_engine = os.environ.get("CIRCUIT_ENGINE")
        if env_engine:
            requested = _validate_engine(env_engine, source="CIRCUIT_ENGINE")
        else:
            requested = _DEFAULT_ENGINE

    _requested_engine = requested
    _resolved_engine = requested  # rust is the only value; nothing to fall back to

    # Write it back so child processes inherit the resolved engine.
    os.environ["CIRCUIT_ENGINE"] = requested

    # Early, non-fatal heads-up when the compiled core is absent: `import
    # circuitopt` still succeeds (version/tooling paths do not touch the core),
    # but any solver call will fail. Warning here surfaces the cause sooner.
    try:
        import circuitopt_core  # noqa: F401
    except Exception:
        warnings.warn(
            "circuitopt_core (the compiled rust core) is not importable; rust is "
            "the only engine, so solver calls will fail until it is installed "
            "(pip install circuitopt-core, or maturin develop rust/crates/co-py)",
            RuntimeWarning,
            stacklevel=2,
        )
    return requested


def current_engine() -> str:
    """Return the active engine (always ``"rust"`` in v2.0.0)."""
    if _resolved_engine is None:
        # Defensive: any ``import circuitopt`` runs apply_engine_env() first, so
        # this only triggers if _engine is driven in isolation.
        apply_engine_env()
    return _resolved_engine  # type: ignore[return-value]


def engine_info() -> dict:
    """Return ``{"engine": <active>, "requested": <asked-for>, "core": <dict|None>}``.

    ``core`` carries ``circuitopt_core.engine_info()`` when the rust core is
    importable, otherwise ``None``.
    """
    engine = current_engine()
    info: dict = {"engine": engine, "requested": _requested_engine, "core": None}
    try:
        import circuitopt_core

        info["core"] = circuitopt_core.engine_info()
    except Exception:
        info["core"] = None
    return info
