"""Tests for ``tools/workers.py`` — the per-machine --workers advisor.

Detection and the recommendation rules are pure and run everywhere; the
calibration path needs FreePDK45 cards and is exercised by the heavy CLI
smoke only.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.workers import detect_topology, pick_knee, recommend

ROOT = Path(__file__).resolve().parents[1]


def test_detect_topology_reports_this_machine_sanely():
    t = detect_topology()
    assert t["total"] >= 1
    assert t["p_cores"] >= 1 and t["e_cores"] >= 0
    if t["hybrid"]:
        assert t["p_cores"] + t["e_cores"] == t["total"]
    else:
        assert t["e_cores"] == 0


def test_recommend_rules_are_topology_driven():
    hybrid = {"total": 10, "p_cores": 4, "e_cores": 6, "hybrid": True}
    uniform = {"total": 8, "p_cores": 8, "e_cores": 0, "hybrid": False}
    r = recommend(hybrid)
    assert r["sweep/ramp"] == r["sine"] == r["explore"] == r["transitions"] == 10
    assert r["signoff"] == 10 and r["single_conversion"] == 1
    assert recommend(uniform)["sweep/ramp"] == 8
    # MC: one task per trial while trials are few, core count once they amortise
    assert recommend(hybrid, mc_trials=16)["mc"] == 16
    assert recommend(hybrid, mc_trials=40)["mc"] == 40
    assert recommend(hybrid, mc_trials=41)["mc"] == 10
    assert recommend(uniform, mc_trials=200)["mc"] == 8
    with pytest.raises(ValueError):
        recommend(hybrid, mc_trials=0)


def test_pick_knee_prefers_the_smallest_count_within_tolerance():
    timings = {1: 3.0, 4: 1.0, 8: 0.68, 10: 0.60, 16: 0.59}
    # 0.60 <= 0.59 * 1.05, so 10 wins over the marginally faster 16
    assert pick_knee(timings) == 10
    # a tighter tolerance flips it to the true argmin
    assert pick_knee(timings, tolerance=0.01) == 16
    with pytest.raises(ValueError):
        pick_knee({})


def test_cli_json_output_is_machine_readable():
    proc = subprocess.run(
        [sys.executable, "tools/workers.py", "--json", "--mc-trials", "12"],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["topology"]["total"] >= 1
    assert data["recommended"]["single_conversion"] == 1
    assert data["recommended"]["mc"] in (12, data["topology"]["total"])
