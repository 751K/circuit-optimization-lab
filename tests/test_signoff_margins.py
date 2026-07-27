"""Contracts for the margin table and the passive-tolerance sweep."""
from __future__ import annotations

import pytest

from circuitopt.signoff_campaign import (
    CampaignConfigurationError,
    _scale_passives,
    format_margin_table,
    run_tolerance_sweep,
    summarize_margins,
)


def _constraint(value, unit, margin, passed):
    return {
        "observed": {"value": value, "unit": unit, "status": "valid"},
        "checks": {},
        "passed": passed,
        "normalized_margin": margin,
    }


def _result(points):
    return {"points": points}


def _point(corner, temperature, supply, constraints):
    return {
        "pvt": {"corner": corner, "temperature_c": temperature,
                "supply_v": supply},
        "cases": {
            case: {"name": case, "status": "pass", "passed": True,
                   "signoff": {"constraints": detail}}
            for case, detail in constraints.items()
        },
    }


# ── margin table ────────────────────────────────────────────────────────────────
def test_margin_table_reports_every_constraint_not_just_the_worst():
    """The campaign's worst_case names one measurement; designing needs them all.

    A spec sitting at +0.02 of its limit is a design constraint even when some
    other spec is at -0.5, and it disappears entirely from a worst-case view."""
    result = _result([
        _point("tt", 27.0, 0.9, {
            "loop": {"phase_margin": _constraint(70.0, "deg", 0.17, True)},
            "tran": {"settling_time": _constraint(4.0e-9, "s", 0.20, True)},
        }),
        _point("ss", 125.0, 0.85, {
            "loop": {"phase_margin": _constraint(61.0, "deg", 0.017, True)},
            "tran": {"settling_time": _constraint(4.98e-9, "s", 0.004, True)},
        }),
    ])
    rows = summarize_margins(result)
    assert [(r["case"], r["constraint"]) for r in rows] == [
        ("tran", "settling_time"), ("loop", "phase_margin")]   # tightest first
    settling = rows[0]
    assert settling["worst_margin"] == pytest.approx(0.004)
    assert settling["worst_point"] == "ss/125/0.85"
    assert settling["min"] == pytest.approx(4.0e-9)
    assert settling["max"] == pytest.approx(4.98e-9)
    assert settling["unit"] == "s"
    assert settling["total"] == 2 and settling["fail_points"] == []


def test_margin_table_collects_every_failing_corner():
    result = _result([
        _point("ff", -40.0, 0.85, {
            "tran": {"settling_time": _constraint(5.4e-9, "s", -0.08, False)}}),
        _point("ff", 27.0, 0.85, {
            "tran": {"settling_time": _constraint(5.2e-9, "s", -0.04, False)}}),
        _point("tt", 27.0, 0.9, {
            "tran": {"settling_time": _constraint(3.9e-9, "s", 0.22, True)}}),
    ])
    (row,) = summarize_margins(result)
    assert row["fail_points"] == ["ff/-40/0.85", "ff/27/0.85"]
    assert row["worst_point"] == "ff/-40/0.85"
    assert row["total"] == 3


def test_margin_table_survives_a_missing_margin():
    result = _result([_point("tt", 27.0, 0.9, {
        "tran": {"settling_time": _constraint(None, "s", None, False)}})])
    (row,) = summarize_margins(result)
    assert row["worst_margin"] is None and row["min"] is None
    assert "n/a" in format_margin_table([row])


def test_margin_table_formats_one_line_per_constraint():
    result = _result([_point("tt", 27.0, 0.9, {
        "loop": {"phase_margin": _constraint(70.0, "deg", 0.17, True)},
        "tran": {"settling_time": _constraint(4.0e-9, "s", 0.20, True)}})])
    text = format_margin_table(summarize_margins(result))
    assert len(text.splitlines()) == 3          # header + two constraints
    assert "loop/phase_margin" in text and "tran/settling_time" in text


# ── passive scaling ─────────────────────────────────────────────────────────────
def test_scale_passives_applies_class_factors():
    deck = {
        "resistors": [{"name": "RZ1", "R": 400.0}, {"name": "RS1", "R": 1e5}],
        "capacitors": [{"name": "CC1", "C": 840e-15}],
    }
    _scale_passives(deck, {"R": 1.2, "C": 0.9})
    assert deck["resistors"][0]["R"] == pytest.approx(480.0)
    assert deck["resistors"][1]["R"] == pytest.approx(1.2e5)
    assert deck["capacitors"][0]["C"] == pytest.approx(756e-15)


def test_scale_passives_lets_a_named_element_override_its_class():
    """Per-element perturbation is how a single sensitive component is found.

    The MDAC's nulling resistor drives the whole compensation; scaling it alone
    is a different question from scaling every resistor together."""
    deck = {"resistors": [{"name": "RZ1", "R": 400.0},
                          {"name": "RS1", "R": 1e5}]}
    _scale_passives(deck, {"R": 1.2, "RZ1": 0.8})
    assert deck["resistors"][0]["R"] == pytest.approx(320.0)   # named wins
    assert deck["resistors"][1]["R"] == pytest.approx(1.2e5)   # class elsewhere


def test_scale_passives_ignores_decks_without_passives():
    deck = {"devices": [{"name": "M1"}]}
    _scale_passives(deck, {"R": 1.2})          # must not raise
    assert deck == {"devices": [{"name": "M1"}]}


# ── tolerance validation ────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1])
def test_tolerance_fraction_outside_the_unit_interval_is_rejected(bad):
    with pytest.raises(CampaignConfigurationError):
        run_tolerance_sweep({"name": "x", "pvt": {}, "cases": []}, {"R": bad})
