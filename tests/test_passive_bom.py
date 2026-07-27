"""Contracts for the passive bill of materials (tools/passive_bom.py)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "passive_bom", ROOT / "tools" / "passive_bom.py")
passive_bom = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(passive_bom)


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


def test_gate_area_counts_multiplicity_and_each_sizing_entry_once():
    generator = types.SimpleNamespace(
        SZ={"M1": (100.0, 0.2), "M2": (100.0, 0.2), "M0": (300.0, 0.2)},
        MULT={"M0": 3})
    # 20 + 20 + 3*60 = 220 um2; both halves of a differential pair are their
    # own SZ entries and must not be doubled again.
    assert passive_bom.transistor_gate_area_um2(generator) == pytest.approx(220.0)


def test_gate_area_is_none_without_a_sizing_table():
    assert passive_bom.transistor_gate_area_um2(types.SimpleNamespace()) is None


# ── report assembly ─────────────────────────────────────────────────────────────
def _generator():
    return types.SimpleNamespace(
        all_testbenches=_decks,
        SZ={"M1": (100.0, 0.2)}, MULT={})


def test_report_totals_count_dut_elements_only():
    report = passive_bom.build_report(
        _generator(), **passive_bom.DEFAULTS)
    assert report["resistor_area_um2"] == pytest.approx(
        passive_bom.resistor_area_um2(1000.0, sheet_ohm_sq=300.0,
                                      resistor_width_um=0.8))
    assert report["capacitor_area_um2"] == pytest.approx(500.0)   # 1 pF @ 2 fF/um2
    names = {row["name"] for row in report["rows"] if row["counted"]}
    assert names == {"RCORE", "CCORE"}


def test_excluded_elements_stay_listed_but_leave_the_totals():
    """A specified external load is real silicon elsewhere, not this block's."""
    report = passive_bom.build_report(
        _generator(), exclude=["CCORE"], **passive_bom.DEFAULTS)
    assert report["capacitor_area_um2"] == pytest.approx(0.0)
    row = next(r for r in report["rows"] if r["name"] == "CCORE")
    assert row["dut"] is True and row["counted"] is False
    assert report["excluded"] == ["CCORE"]


def test_rows_are_ordered_largest_area_first():
    report = passive_bom.build_report(_generator(), **passive_bom.DEFAULTS)
    areas = [row["area_um2"] for row in report["rows"]]
    assert areas == sorted(areas, reverse=True)


def test_report_records_the_process_constants_it_used():
    report = passive_bom.build_report(
        _generator(), sheet_ohm_sq=1000.0, resistor_width_um=0.5, cap_ff_um2=4.0)
    assert report["process"] == {"sheet_ohm_sq": 1000.0,
                                 "resistor_width_um": 0.5, "cap_ff_um2": 4.0}
    assert report["capacitor_area_um2"] == pytest.approx(250.0)


def test_cli_rejects_a_generator_without_testbenches(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "fake_bom_gen", types.ModuleType("fake_bom_gen"))
    with pytest.raises(SystemExit) as excinfo:
        passive_bom.main(["--generator", "fake_bom_gen"])
    assert "all_testbenches" in str(excinfo.value)
