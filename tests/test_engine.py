"""Subprocess matrix for the ``CIRCUIT_ENGINE`` compute-engine switch.

Mirrors ``tests/test_cli_numba_flag.py``'s out-of-process style. The engine is
resolved at *import time* by ``circuitopt._engine.apply_engine_env`` (wired into
``circuitopt/__init__.py``), and the numba kill-switch it drives is baked when
``circuitopt.numba_kernels`` is first imported — neither is observable after the
fact in the current interpreter. Each case therefore runs in a fresh subprocess
with a controlled environment.

The rust cases rig ``sys.modules['circuitopt_core']`` before importing circuitopt
so the fallback ("missing") and success ("stub") paths are deterministic whether
or not the compiled extension is actually installed; a separate probe-skipped
case exercises the real wheel (installed only in CI).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CIRCUIT = "examples/periodic_rc.json"

# Import circuitopt inside a warnings recorder, then report the resolved engine,
# the baked numba flag, and how many rust-fallback warnings fired. sys.argv[1]
# optionally rigs circuitopt_core *before* the import:
#   poison → force `import circuitopt_core` to fail (None in sys.modules)
#   stub   → inject a stand-in module with engine_info()
_DRIVER = r"""
import json, sys, types, warnings
mode = sys.argv[1] if len(sys.argv) > 1 else "plain"
if mode == "poison":
    sys.modules["circuitopt_core"] = None            # -> ImportError on import
elif mode == "stub":
    fake = types.ModuleType("circuitopt_core")
    fake.__version__ = "0.0.stub"
    fake.engine_info = lambda: {"stub": True}
    sys.modules["circuitopt_core"] = fake
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import circuitopt
    from circuitopt import _engine, numba_kernels
warns = [w for w in caught
         if issubclass(w.category, RuntimeWarning)
         and "circuitopt_core" in str(w.message)]
print("RESULT " + json.dumps({
    "engine": _engine.current_engine(),
    "info": _engine.engine_info(),
    "use_numba": bool(numba_kernels.USE_NUMBA),
    "warn_count": len(warns),
}))
"""


def _child_env(**overrides):
    """A clean child environment: the two engine vars are cleared, then set."""
    env = os.environ.copy()
    env.pop("CIRCUIT_ENGINE", None)
    env.pop("CIRCUIT_USE_NUMBA", None)
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _probe(mode="plain", **env_overrides):
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, mode],
        cwd=str(_REPO_ROOT), capture_output=True, text=True,
        env=_child_env(**env_overrides),
    )
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    return proc, result


def _core_installed():
    try:
        import circuitopt_core  # noqa: F401
        return True
    except Exception:
        return False


# ── env-driven resolution ────────────────────────────────────────────────────

def test_engine_python_disables_numba():
    proc, r = _probe(CIRCUIT_ENGINE="python")
    assert r is not None, proc.stderr
    assert r["engine"] == "python"
    assert r["use_numba"] is False
    assert r["warn_count"] == 0


def test_engine_numba_keeps_numba():
    proc, r = _probe(CIRCUIT_ENGINE="numba")
    assert r is not None, proc.stderr
    assert r["engine"] == "numba"
    assert r["use_numba"] is True
    assert r["warn_count"] == 0


def test_engine_unset_defaults_to_numba():
    proc, r = _probe()  # CIRCUIT_ENGINE and CIRCUIT_USE_NUMBA both cleared
    assert r is not None, proc.stderr
    assert r["engine"] == "numba"
    assert r["use_numba"] is True
    assert r["warn_count"] == 0


def test_engine_rust_missing_falls_back_to_numba_with_one_warning():
    proc, r = _probe(mode="poison", CIRCUIT_ENGINE="rust")
    assert r is not None, proc.stderr
    assert r["engine"] == "numba"
    assert r["warn_count"] == 1
    assert r["info"]["requested"] == "rust"
    assert r["info"]["core"] is None


def test_engine_rust_stub_resolves_rust_without_warning():
    proc, r = _probe(mode="stub", CIRCUIT_ENGINE="rust")
    assert r is not None, proc.stderr
    assert r["engine"] == "rust"
    assert r["warn_count"] == 0
    assert r["info"]["requested"] == "rust"
    assert r["info"]["core"] == {"stub": True}


@pytest.mark.skipif(
    not _core_installed(),
    reason="circuitopt_core not installed; build+install rust/crates/co-py to cover it",
)
def test_engine_rust_present_resolves_rust():
    # The real compiled extension (CI installs it via `pip install rust/crates/co-py`).
    proc, r = _probe(CIRCUIT_ENGINE="rust")
    assert r is not None, proc.stderr
    assert r["engine"] == "rust"
    assert r["warn_count"] == 0
    assert r["info"]["core"] is not None
    assert r["info"]["core"]["version"]


def test_engine_illegal_value_exits_two():
    proc, r = _probe(CIRCUIT_ENGINE="cuda")
    assert proc.returncode == 2
    assert r is None
    assert "invalid engine" in proc.stderr


# ── CLI ──────────────────────────────────────────────────────────────────────

def _run_cli(env, *extra):
    return subprocess.run(
        [sys.executable, "-m", "circuitopt", "run", _CIRCUIT, "-a", "ac", "--quiet", *extra],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, env=env,
    )


def test_cli_engine_python_succeeds(tmp_path):
    out = tmp_path / "ac.json"
    proc = _run_cli(_child_env(), "--engine", "python", "-o", str(out))
    assert proc.returncode == 0, proc.stderr


def test_cli_no_numba_engine_rust_conflicts():
    proc = _run_cli(_child_env(), "--no-numba", "--engine", "rust")
    assert proc.returncode != 0
    assert "conflict" in proc.stderr.lower()
