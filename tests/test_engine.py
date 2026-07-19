"""Subprocess matrix for the ``CIRCUIT_ENGINE`` compute-engine switch (v2.0.0).

As of v2.0.0 rust is the only engine. The switch is resolved at *import time* by
``circuitopt._engine.apply_engine_env`` (wired into ``circuitopt/__init__.py``),
so each case runs in a fresh subprocess with a controlled environment. The
retired ``python``/``numba`` engine values and the ``--no-numba`` /
``CIRCUIT_USE_NUMBA`` switches are hard errors (exit 2) that point at the
CHANGELOG.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CIRCUIT = "examples/periodic_rc.json"

# Import circuitopt and report the resolved engine + engine_info.
_DRIVER = r"""
import json
import circuitopt
from circuitopt import _engine
print("RESULT " + json.dumps({
    "engine": _engine.current_engine(),
    "info": _engine.engine_info(),
}))
"""


def _child_env(**overrides):
    """A clean child environment: the engine vars are cleared, then set."""
    env = os.environ.copy()
    env.pop("CIRCUIT_ENGINE", None)
    env.pop("CIRCUIT_USE_NUMBA", None)
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _probe(**env_overrides):
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER],
        cwd=str(_REPO_ROOT), capture_output=True, text=True,
        env=_child_env(**env_overrides),
    )
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    return proc, result


# ── env-driven resolution ────────────────────────────────────────────────────

def test_engine_unset_defaults_to_rust():
    proc, r = _probe()  # CIRCUIT_ENGINE and CIRCUIT_USE_NUMBA both cleared
    assert r is not None, proc.stderr
    assert r["engine"] == "rust"
    assert r["info"]["core"] is not None  # the compiled core is installed in CI/dev


def test_engine_rust_explicit_ok():
    proc, r = _probe(CIRCUIT_ENGINE="rust")
    assert r is not None, proc.stderr
    assert r["engine"] == "rust"


def test_engine_python_value_removed_errors():
    proc, r = _probe(CIRCUIT_ENGINE="python")
    assert proc.returncode == 2, proc.stdout
    assert r is None
    assert "removed in v2.0.0" in proc.stderr
    assert "CHANGELOG" in proc.stderr


def test_engine_numba_value_removed_errors():
    proc, r = _probe(CIRCUIT_ENGINE="numba")
    assert proc.returncode == 2, proc.stdout
    assert r is None
    assert "removed in v2.0.0" in proc.stderr


def test_use_numba_env_removed_errors():
    proc, r = _probe(CIRCUIT_USE_NUMBA="1")
    assert proc.returncode == 2, proc.stdout
    assert r is None
    assert "CIRCUIT_USE_NUMBA" in proc.stderr
    assert "removed in v2.0.0" in proc.stderr


def test_use_numba_env_zero_also_removed_errors():
    # The old CIRCUIT_USE_NUMBA=0 kill-switch is gone too: it errors, not no-ops.
    proc, r = _probe(CIRCUIT_USE_NUMBA="0")
    assert proc.returncode == 2, proc.stdout
    assert "CIRCUIT_USE_NUMBA" in proc.stderr


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


def test_cli_engine_rust_succeeds(tmp_path):
    out = tmp_path / "ac.json"
    proc = _run_cli(_child_env(), "--engine", "rust", "-o", str(out))
    assert proc.returncode == 0, proc.stderr


def test_cli_engine_python_removed_errors():
    proc = _run_cli(_child_env(), "--engine", "python")
    assert proc.returncode == 2
    assert "removed in v2.0.0" in proc.stderr


def test_cli_no_numba_removed_errors():
    proc = _run_cli(_child_env(), "--no-numba")
    assert proc.returncode == 2
    assert "--no-numba" in proc.stderr
    assert "removed in v2.0.0" in proc.stderr
