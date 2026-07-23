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

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CIRCUIT = "examples/afe_explore.json"
_MISSING = "examples/__does_not_exist__.json"


def _freepdk45_ready():
    try:
        from circuitopt.toolchain import pdk_root
        return (Path(pdk_root()) / "freepdk45" / "models_nom" / "NMOS_VTG.inc").is_file()
    except Exception:
        return False


_FREEPDK45 = _freepdk45_ready()


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


def test_corners_default_output_format_frozen(tmp_path):
    """Default corners (no PVT axes) keeps the frozen flat output + CSV header.

    The default surface is a red line: no ``temps:``/``vdd_scale:`` header lines, no
    ``[T=…]`` slice grouping, and the CSV header stays exactly the four frozen columns
    — opting into an axis is the only thing that changes the shape."""
    out = tmp_path / "corners.csv"
    proc = _run("corners", _CIRCUIT, "--freqs-num", "5", "-o", str(out))
    assert proc.returncode == 0, proc.stderr
    assert "temps:" not in proc.stdout and "vdd_scale:" not in proc.stdout
    assert "[T=" not in proc.stdout and "Vdd×" not in proc.stdout
    assert out.read_text().splitlines()[0] == "corner,gain_peak_dB,bw_Hz,irn_uV"


def test_corners_pvt_grid_groups_and_adds_csv_columns(tmp_path):
    """--temps/--vdd-scale group the print by slice and add the CSV axis columns."""
    if not _FREEPDK45:
        pytest.skip("FreePDK45 cards not present")
    out = tmp_path / "grid.csv"
    proc = _run("corners", "examples/freepdk45_5t_ota.json", "--freqs-num", "5",
                "--temps=27,125", "--vdd-scale=1.0,1.1", "-o", str(out))
    assert proc.returncode == 0, proc.stderr
    assert "temps: 27, 125 °C" in proc.stdout
    assert "vdd_scale: 1, 1.1" in proc.stdout
    assert "[T=27 °C  Vdd×1]" in proc.stdout          # grouped slice header
    assert out.read_text().splitlines()[0] == \
        "corner,temp_c,vdd_scale,gain_peak_dB,bw_Hz,irn_uV"


def test_corners_pvt_axis_rejects_otft():
    """--temps on an OTFT circuit fails cleanly (non-zero, no partial table)."""
    proc = _run("corners", _CIRCUIT, "--freqs-num", "5", "--temps=27")
    assert proc.returncode != 0
    assert "silicon" in (proc.stderr + proc.stdout).lower()


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
