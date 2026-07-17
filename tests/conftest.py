"""Shared pytest hooks.

Central assignment of the ``heavy_e2e`` marker (see pyproject.toml).

These tests run complete SAR/ADC conversions on the native silicon BSIM4
backend — legitimate end-to-end regressions, but minutes each on a machine
with PDK cards installed. Measured 2026-07-17 on the v1.4.0 tree: the 25
slowest of them accounted for ~1200 s of a 1312 s default run. They are
excluded from the default suite via ``addopts`` and run explicitly with::

    pytest -m heavy_e2e

Keeping the list here (rather than per-file ``pytestmark``) gives one
reviewable inventory of everything the default suite skips.
"""
from __future__ import annotations

import pytest

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
