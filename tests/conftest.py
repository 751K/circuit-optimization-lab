"""Shared pytest hooks.

Central assignment of the ``heavy_e2e`` marker (see pyproject.toml).

These tests run complete SAR/ADC conversions on the native silicon BSIM4
backend. They cost minutes each on the v1.4.0 tree (2026-07-17: the 25 slowest
accounted for ~1200 s of a 1312 s run), which is why they were once excluded
from the default suite; the compiled kernels have since taken the whole set to
~23 s and the exclusion is gone. The marker remains for selecting them::

    pytest -m heavy_e2e

Keeping the list here (rather than per-file ``pytestmark``) gives one
reviewable inventory of the heavyweight set.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Keep Matplotlib's font cache writable and reusable across pytest subprocesses.
# Without this, sandboxed/home-read-only runs rebuild it in a temporary
# directory on every process start.
_MPLCONFIGDIR = Path(__file__).resolve().parent.parent / ".pytest_cache" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

# Whole files whose tests are all heavyweight end-to-end conversions.
_HEAVY_E2E_FILES = {
    "test_freepdk45_sar6.py",
    "test_plot_adc.py",
    "test_plot_adc_semantics.py",
    "test_sar6_clock_semantics.py",
    "test_sar_explore.py",
    "test_sar_mc.py",
    "test_sar_mc_semantics.py",
    "test_sar_parallel.py",
    "test_sar_rust.py",
    "test_sar_wp2_semantics.py",
}

# Individual heavyweight tests inside otherwise-fast files.
_HEAVY_E2E_TESTS = {
    ("test_sar.py", "test_sar_code_center_sweep_has_every_code"),
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        name = item.path.name
        if name in _HEAVY_E2E_FILES or (name, item.name) in _HEAVY_E2E_TESTS:
            item.add_marker(pytest.mark.heavy_e2e)
