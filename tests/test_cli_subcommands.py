"""End-to-end smoke tests for the ``corners`` / ``mc`` / ``chopper`` subcommands.

The ``run`` / ``explore`` / ``dataset`` / ``plot`` subcommands already have CLI
coverage; these three did not. They are exercised here the same way a user runs
them — as a real ``python -m circuitopt …`` subprocess from the repo root — so the full
path (argv routing, spec load, solver call, summary print, exit code) is checked,
not just arg parsing. Scale is pinned to the minimum that still produces output
(``--freqs-num`` tiny, ``-n 2``, chopper ``ideal`` level with a small harmonic
count) so all three combined run in ~2 s of wall time.

Each subcommand gets a happy-path case (returncode 0 + an expected stdout anchor)
and a bad-input case (missing JSON or an illegal flag value) asserting a non-zero
returncode and a readable stderr message.

Style follows tests/test_cli_numba_flag.py: subprocess, repo-root cwd, captured
text output.
"""
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CIRCUIT = "examples/afe_explore.json"
_MISSING = "examples/__does_not_exist__.json"


def _run(*args, timeout=90):
    """Run ``python -m circuitopt <args>`` from the repo root, capturing output."""
    return subprocess.run(
        [sys.executable, "-m", "circuitopt", *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ── corners ───────────────────────────────────────────────────────────────────

def test_corners_smoke():
    # freqs-num 5 keeps the three-corner sweep to well under a second.
    proc = _run("corners", _CIRCUIT, "--freqs-num", "5")
    assert proc.returncode == 0, proc.stderr
    assert "Corner sweep" in proc.stdout
    # All three process corners are named in the table.
    for corner in ("typical", "slow", "fast"):
        assert corner in proc.stdout, proc.stdout
    # Each row reports gain/BW/IRN.
    assert "gain=" in proc.stdout and "BW=" in proc.stdout and "IRN=" in proc.stdout


def test_corners_missing_file_fails():
    proc = _run("corners", _MISSING, "--freqs-num", "5")
    assert proc.returncode != 0
    assert "not found" in (proc.stderr + proc.stdout).lower()


# ── mc (mismatch Monte Carlo) ─────────────────────────────────────────────────

def test_mc_smoke():
    # -n 2 is the minimum that still yields a mean/std summary.
    proc = _run("mc", _CIRCUIT, "-n", "2", "--seed", "1", "--freqs-num", "5")
    assert proc.returncode == 0, proc.stderr
    assert "Mismatch MC" in proc.stdout
    # The latch-rate statistic line is always printed.
    assert "latch_rate:" in proc.stdout, proc.stdout


def test_mc_illegal_corner_choice_fails():
    # --corner is a fixed choice set; 'bogus' must be rejected by argparse (rc 2).
    proc = _run("mc", _CIRCUIT, "-n", "2", "--corner", "bogus")
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr and "corner" in proc.stderr


# ── chopper ───────────────────────────────────────────────────────────────────

def test_chopper_ideal_smoke():
    # The 'ideal' square-wave LPTV level is the fastest; small harmonic + freq
    # counts keep it sub-second.
    proc = _run("chopper", _CIRCUIT, "--level", "ideal",
                "--freqs-num", "11", "--max-harmonic", "3")
    assert proc.returncode == 0, proc.stderr
    assert "Chopper analysis (ideal)" in proc.stdout
    # ideal level prints a peak-gain / IRN summary line.
    assert "peak:" in proc.stdout and "IRN:" in proc.stdout, proc.stdout


def test_chopper_illegal_level_fails():
    # --level is a fixed choice set; 'bogus' must be rejected by argparse (rc 2).
    proc = _run("chopper", _CIRCUIT, "--level", "bogus")
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr and "level" in proc.stderr


def test_chopper_missing_file_fails():
    proc = _run("chopper", _MISSING, "--level", "ideal",
                "--freqs-num", "5", "--max-harmonic", "3")
    assert proc.returncode != 0
    assert "not found" in (proc.stderr + proc.stdout).lower()
