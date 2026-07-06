"""Tests for the ``--no-numba`` CLI flag wiring (regression: dead-flag bug).

``circuitopt.numba_kernels`` bakes ``USE_NUMBA``/``NUMBA_AVAILABLE`` from
``CIRCUIT_USE_NUMBA`` at *import time*. Under ``python -m circuitopt …`` the package
``circuitopt/__init__.py`` runs first and its solver imports pull ``numba_kernels`` in
transitively, so setting the env var inside a ``_cmd_*`` handler was too late and
``--no-numba`` silently no-oped. The fix is an argv pre-scan in
``circuitopt/__init__.py`` (before those imports) plus a ``_assert_numba_flag`` guard
that fails loudly if the flag is requested but Numba is still active.

These tests pin:
  * a subprocess ``run --no-numba`` returns 0 *and* actually forces
    ``USE_NUMBA is False`` inside that process (pre-scan works),
  * the same command without the flag also returns 0 (guard is not trigger-happy),
  * the guard raises ``SystemExit`` when handed ``no_numba=True`` while
    ``numba_kernels.USE_NUMBA`` is still ``True`` (tripwire fires).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from circuitopt.__main__ import _assert_numba_flag

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CIRCUIT = "examples/periodic_rc.json"


def _run_cli(*extra):
    """Run ``python -m circuitopt run <circuit> --analysis ac --quiet [extra]``."""
    return subprocess.run(
        [sys.executable, "-m", "circuitopt", "run", _CIRCUIT,
         "--analysis", "ac", "--quiet", *extra],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )


def test_no_numba_run_succeeds_and_forces_pure_python():
    # A tiny driver that reproduces the CLI invocation in-process and prints the
    # baked flag — proves the pre-scan set CIRCUIT_USE_NUMBA=0 *before*
    # numba_kernels was imported, not just that the run exited 0.
    driver = (
        "import sys; "
        f"sys.argv = ['prog','run','{_CIRCUIT}','--analysis','ac','--no-numba','--quiet']; "
        "import circuitopt.__main__ as m; m.main(); "
        "from circuitopt import numba_kernels as nk; "
        "print('USE_NUMBA', nk.USE_NUMBA)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=str(_REPO_ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "USE_NUMBA False" in proc.stdout, proc.stdout


def test_run_with_no_numba_returncode_zero():
    proc = _run_cli("--no-numba")
    assert proc.returncode == 0, proc.stderr


def test_run_without_no_numba_returncode_zero():
    # Guard must not misfire on the ordinary (Numba-on) path.
    proc = _run_cli()
    assert proc.returncode == 0, proc.stderr


def test_assert_numba_flag_fires_when_numba_still_active():
    # In *this* process Numba was never disabled, so numba_kernels.USE_NUMBA is
    # whatever the default is. Skip if this environment has no Numba (then the
    # guard would correctly stay silent and there is nothing to trip).
    from circuitopt import numba_kernels
    if not numba_kernels.USE_NUMBA:
        pytest.skip("Numba already disabled in this process; guard has nothing to trip")

    args = argparse.Namespace(no_numba=True)
    with pytest.raises(SystemExit) as exc:
        _assert_numba_flag(args)
    assert "no-numba" in str(exc.value)


def test_assert_numba_flag_silent_when_flag_not_set():
    # no_numba=False must never raise, regardless of Numba state.
    _assert_numba_flag(argparse.Namespace(no_numba=False))
    _assert_numba_flag(argparse.Namespace())  # attribute absent → treated as False
