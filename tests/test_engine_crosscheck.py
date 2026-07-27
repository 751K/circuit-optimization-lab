"""Contracts for the two-engine cross-check (circuitopt/engine_crosscheck.py)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from circuitopt.engine_crosscheck import (
    CrosscheckError,
    _edge_mask,
    driven_nodes,
    format_crosscheck,
    oracle_deck,
)


# ── oracle binding rewrite ──────────────────────────────────────────────────────
def test_oracle_deck_repoints_models_without_touching_the_input():
    """Committed netlists stay engine-neutral; the rewrite happens in memory.

    The registry split that gives ``<pdk>`` the native classes and
    ``<pdk>_ngspice`` the model-card ones silently severed a whole oracle
    campaign once, precisely because a deck bound the wrong one."""
    deck = {"models": {
        "M1": {"pdk": "freepdk45", "model": "nmos"},
        "M2": {"pdk": "freepdk45", "model": "pmos", "vb": 1.0}}}
    out = oracle_deck(deck)
    assert {m["pdk"] for m in out["models"].values()} == {"freepdk45_ngspice"}
    assert out["models"]["M2"]["vb"] == 1.0          # everything else survives
    assert {m["pdk"] for m in deck["models"].values()} == {"freepdk45"}


def test_already_oracle_bound_models_are_left_alone():
    deck = {"models": {"M1": {"pdk": "freepdk45_ngspice", "model": "nmos"},
                       "M2": {"pdk": "freepdk45", "model": "nmos"}}}
    out = oracle_deck(deck)
    assert out["models"]["M1"]["pdk"] == "freepdk45_ngspice"
    assert out["models"]["M2"]["pdk"] == "freepdk45_ngspice"


def test_pdk_without_an_oracle_path_is_named_in_the_error():
    deck = {"models": {"M1": {"pdk": "sky130", "model": "nmos"}}}
    with pytest.raises(CrosscheckError) as excinfo:
        oracle_deck(deck)
    assert "sky130_ngspice" in str(excinfo.value)


def test_deck_with_nothing_to_rewrite_is_rejected():
    with pytest.raises(CrosscheckError):
        oracle_deck({"models": {}})


# ── stimulus nodes ──────────────────────────────────────────────────────────────
def test_driven_nodes_finds_waveform_sources_and_node_inputs():
    """Stimulus nodes reproduce their own waveform in both engines.

    Comparing them tests the testbench, and a 20 ps clock edge landing on two
    different step grids otherwise outranks every real node in the report."""
    spec = SimpleNamespace(topology=SimpleNamespace(vsources=[
        ("VBP1", "BP1", "GND", "bp1"),      # waveform-valued -> driven
        ("VDC", "REF", "GND", 0.45),        # constant -> not stimulus
        ("Vinj", "A", "B", "unknown_key"),  # not a declared waveform
    ]))
    context = {"inputs": {"bp1": None, "dch": None}, "node_inputs": {"DCH": "dch"}}
    assert driven_nodes(spec, context) == {"BP1", "GND", "DCH"}


# ── edge masking ────────────────────────────────────────────────────────────────
def test_edge_mask_covers_the_samples_straddling_a_step():
    t = np.linspace(0.0, 1e-9, 11)
    square = np.array([0.0] * 5 + [1.0] * 6)
    mask = _edge_mask({"inputs": {"clk": square}}, t)
    assert mask[4] and mask[5] and mask[6]      # the jump and its neighbours
    assert not mask[0] and not mask[-1]


def test_constant_and_slow_inputs_mask_nothing():
    t = np.linspace(0.0, 1e-9, 11)
    inputs = {"dc": np.full(11, 0.45), "ramp": np.linspace(0.0, 1.0, 11)}
    assert not _edge_mask({"inputs": inputs}, t).any()


def test_edge_mask_ignores_waveforms_off_the_grid():
    t = np.linspace(0.0, 1e-9, 11)
    assert not _edge_mask({"inputs": {"other": np.zeros(5)}}, t).any()


# ── verdict semantics ───────────────────────────────────────────────────────────
def _report(final_mv, peak_mv):
    return {
        "case": "c", "pvt": {"corner": "tt", "temperature_c": 27.0, "supply_v": 0.9},
        "samples": 100, "compared_samples": 96, "skipped_edge_samples": 4,
        "excluded_driven_nodes": [],
        "nodes": [{"node": "OUTP", "max_abs_delta_v": peak_mv * 1e-3,
                   "at_time_s": 1e-10, "native_v": 0.5, "oracle_v": 0.5,
                   "final_delta_v": final_mv * 1e-3}],
        "worst_abs_delta_v": peak_mv * 1e-3, "worst_node": "OUTP",
        "worst_peak_time_s": 1e-10, "worst_final_delta_v": final_mv * 1e-3,
    }


def test_verdict_keys_on_the_settled_deviation_not_the_peak():
    """A big peak at a fast transition is step placement; a settled gap is not.

    The defect this tool exists for drifted the common mode to +0.42 V and
    LEFT it there, so the settled column is what has to decide."""
    text = format_crosscheck(_report(final_mv=0.4, peak_mv=118.0), tolerance_v=5e-3)
    assert "AGREE" in text and "118" in text        # peak reported, not fatal

    text = format_crosscheck(_report(final_mv=6.4, peak_mv=26.0), tolerance_v=5e-3)
    assert "DIVERGE" in text


def test_peak_flag_uses_its_own_looser_bound():
    row_flagged = format_crosscheck(
        _report(final_mv=0.1, peak_mv=118.0), tolerance_v=5e-3,
        peak_tolerance_v=50e-3).splitlines()[2]
    assert row_flagged.rstrip().endswith("<-")
    row_clean = format_crosscheck(
        _report(final_mv=0.1, peak_mv=118.0), tolerance_v=5e-3,
        peak_tolerance_v=200e-3).splitlines()[2]
    assert not row_clean.rstrip().endswith("<-")


def test_report_states_how_many_samples_were_actually_compared():
    text = format_crosscheck(_report(final_mv=0.1, peak_mv=1.0), tolerance_v=5e-3)
    assert "96/100 samples compared" in text
    assert "4 straddle an input edge" in text
