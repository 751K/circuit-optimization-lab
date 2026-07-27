"""Contracts for the passive bill of materials (circuitopt/passive_bom.py)."""
from __future__ import annotations

import json

import pytest

from circuitopt import passive_bom


def _decks():
    """Two testbenches around one amplifier.

    ``RCORE``/``CCORE`` are in both (amplifier); ``RPROBE`` is only in the loop
    deck and ``CLOAD_TB`` only in the transient deck (testbench furniture)."""
    core_r = {"name": "RCORE", "a": "A", "b": "B", "R": 1000.0}
    core_c = {"name": "CCORE", "a": "A", "b": "B", "C": 1e-12}
    return {
        "loop.json": {"resistors": [core_r, {"name": "RPROBE", "R": 1e6}],
                      "capacitors": [core_c]},
        "tran.json": {"resistors": [core_r],
                      "capacitors": [core_c, {"name": "CLOAD_TB", "C": 5e-13}]},
    }


# ── DUT vs testbench classification ─────────────────────────────────────────────
def test_elements_in_every_deck_are_dut_others_are_testbench():
    """Membership across the deck set IS the classification rule.

    It needs no annotation, so it stays correct when a testbench is added --
    which is the whole reason for deriving it instead of declaring it."""
    classified = passive_bom.classify(_decks())
    assert classified["RCORE"] == ("R", 1000.0, True)
    assert classified["CCORE"] == ("C", 1e-12, True)
    assert classified["RPROBE"][2] is False
    assert classified["CLOAD_TB"][2] is False


def test_adding_a_testbench_does_not_reclassify_the_amplifier():
    decks = _decks()
    decks["ac.json"] = {"resistors": [{"name": "RCORE", "R": 1000.0}],
                        "capacitors": [{"name": "CCORE", "C": 1e-12},
                                       {"name": "CAC", "C": 1e-10}]}
    classified = passive_bom.classify(decks)
    assert classified["RCORE"][2] is True and classified["CCORE"][2] is True
    assert classified["CAC"][2] is False


# ── area model ──────────────────────────────────────────────────────────────────
def test_resistor_area_is_squares_times_width_squared():
    # 30 kohm at 300 ohm/sq = 100 squares; at 0.8 um wide that is 64 um2.
    area = passive_bom.resistor_area_um2(
        30e3, sheet_ohm_sq=300.0, resistor_width_um=0.8)
    assert area == pytest.approx(100 * 0.8 * 0.8)


def test_capacitor_area_follows_the_declared_density():
    # 40 pF at 2 fF/um2 is 20 000 um2 -- the number that makes an oversized
    # compensation capacitor visible at review time.
    assert passive_bom.capacitor_area_um2(40e-12, cap_ff_um2=2.0) == pytest.approx(20000)
    assert passive_bom.capacitor_area_um2(40e-12, cap_ff_um2=4.0) == pytest.approx(10000)


def test_gate_area_counts_multiplicity_and_each_instance_once():
    devices = [{"name": "M1", "W": 100.0, "L": 0.2},
               {"name": "M2", "W": 100.0, "L": 0.2},
               {"name": "M0", "W": 300.0, "L": 0.2, "M": 3}]
    decks = {"a.json": {"devices": devices}, "b.json": {"devices": devices}}
    # 20 + 20 + 3*60 = 220 um2; both halves of a differential pair are their
    # own instances and must not be doubled again.
    assert passive_bom.transistor_gate_area_um2(decks) == pytest.approx(220.0)


def test_gate_area_counts_only_devices_present_in_every_deck():
    """A testbench switch is not the amplifier's silicon."""
    decks = {
        "ac.json": {"devices": [{"name": "M1", "W": 100.0, "L": 0.2}]},
        "tran.json": {"devices": [{"name": "M1", "W": 100.0, "L": 0.2},
                                  {"name": "MSW", "W": 10.0, "L": 0.05}]},
    }
    assert passive_bom.transistor_gate_area_um2(decks) == pytest.approx(20.0)


def test_gate_area_is_none_without_devices():
    assert passive_bom.transistor_gate_area_um2({"a.json": {}, "b.json": {}}) is None


# ── report assembly ─────────────────────────────────────────────────────────────
def test_report_totals_count_dut_elements_only():
    report = passive_bom.build_report(_decks(), **passive_bom.DEFAULTS)
    assert report["resistor_area_um2"] == pytest.approx(
        passive_bom.resistor_area_um2(1000.0, sheet_ohm_sq=300.0,
                                      resistor_width_um=0.8))
    assert report["capacitor_area_um2"] == pytest.approx(500.0)   # 1 pF @ 2 fF/um2
    names = {row["name"] for row in report["rows"] if row["counted"]}
    assert names == {"RCORE", "CCORE"}


def test_excluded_elements_stay_listed_but_leave_the_totals():
    """A specified external load is real silicon elsewhere, not this block's."""
    report = passive_bom.build_report(
        _decks(), exclude=["CCORE"], **passive_bom.DEFAULTS)
    assert report["capacitor_area_um2"] == pytest.approx(0.0)
    row = next(r for r in report["rows"] if r["name"] == "CCORE")
    assert row["dut"] is True and row["counted"] is False
    assert report["excluded"] == ["CCORE"]


def test_rows_are_ordered_largest_area_first():
    report = passive_bom.build_report(_decks(), **passive_bom.DEFAULTS)
    areas = [row["area_um2"] for row in report["rows"]]
    assert areas == sorted(areas, reverse=True)


def test_report_records_the_process_constants_it_used():
    report = passive_bom.build_report(
        _decks(), sheet_ohm_sq=1000.0, resistor_width_um=0.5, cap_ff_um2=4.0)
    assert report["process"] == {"sheet_ohm_sq": 1000.0,
                                 "resistor_width_um": 0.5, "cap_ff_um2": 4.0}
    assert report["capacitor_area_um2"] == pytest.approx(250.0)


# ── deck loading ────────────────────────────────────────────────────────────────
def test_a_signoff_manifest_expands_to_its_case_circuits(tmp_path):
    """The manifest already names every testbench around one DUT.

    That is exactly the deck set the membership rule needs, so pointing the
    inventory at a campaign requires no second list to keep in sync."""
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps(
            {"resistors": [{"name": "R1", "R": 1000.0}]}), encoding="utf-8")
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({
        "name": "c", "pvt": {}, "cases": [{"name": "x", "circuit": "a.json"},
                                          {"name": "y", "circuit": "b.json"}]}),
        encoding="utf-8")
    decks = passive_bom.load_decks([manifest])
    assert set(decks) == {"a.json", "b.json"}


def test_a_single_deck_cannot_be_classified(tmp_path):
    """One deck gives no membership signal: everything would look like DUT."""
    deck = tmp_path / "only.json"
    deck.write_text(json.dumps({"resistors": []}), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        passive_bom.load_decks([deck])
    assert "at least two decks" in str(excinfo.value)
