"""Adversarial semantic tests for the adc.clock extension + the 6-bit SAR design.

Reviewer-side verification, third round. These attack the *contracts* the
decision scheme relies on: a strobe must be low while the CDAC switches, the
strobe pattern must be replay-invariant, validation must be airtight, and the
WP1 mismatch hook must genuinely reach the comparator. The shipped sar6 example
now uses a static comparator, so the strobe tests inject an ``adc.clock`` block
through the config override -- the machinery under test is the waveform
builder, which does not care whether a device consumes the row.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from circuitopt.circuit_loader import load_circuit_json
from circuitopt.sar import _sar_config, sar_input_waveforms, sar_time_grid
from circuitopt.toolchain import pdk_root


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "freepdk45_sar6.json"
CLOCK = {"clock": {"input": "clk"}}
_HAVE = (Path(pdk_root()) / "freepdk45" / "models_nom" / "NMOS_VTG.inc").is_file()
needs_freepdk45 = pytest.mark.skipif(
    not _HAVE, reason="FreePDK45 cards not present")


def _spec():
    return load_circuit_json(EXAMPLE)


# ── strobe timing contracts (pure Python) ─────────────────────────────────────
def test_clock_low_while_trial_cap_switches():
    """StrongARM must be in reset (clk low) at every trial_start — the instant the
    CDAC bit cap flips — and stay low until the differential has settled. A strobe
    that overlaps the switching edge would latch a stale/incompletely-settled value."""
    spec = _spec()
    cfg = _sar_config(spec, CLOCK)
    tgrid = sar_time_grid(spec, cfg)
    wave = sar_input_waveforms(spec, 0.7, [None] * 6, 5, config=cfg, tgrid=tgrid)
    clk = wave["clk"]
    for bit in range(cfg["n_bits"]):
        trial_start = cfg["sample_end"] + (bit + 0.5) * cfg["bit_period"]
        assert np.interp(trial_start, tgrid, clk) == 0.0, f"clk high at bit {bit} switch"
        # still low one edge after the cap has finished slewing
        assert np.interp(trial_start + cfg["edge_time"], tgrid, clk) == 0.0


def test_clock_pattern_is_replay_invariant():
    """The whole decision-extraction scheme rests on one fixed per-bit strobe that
    is identical for every replayed trial and decision history."""
    spec = _spec()
    cfg = _sar_config(spec, CLOCK)
    tgrid = sar_time_grid(spec, cfg)
    a = sar_input_waveforms(spec, 0.2, [None] * 6, 0, config=cfg, tgrid=tgrid)["clk"]
    b = sar_input_waveforms(spec, 0.9, [1, 0, 1, 0, 1, None], 5,
                            config=cfg, tgrid=tgrid)["clk"]
    np.testing.assert_array_equal(a, b)


def test_clock_returns_low_between_decisions():
    """After decision_time + reset_hold the latch must be back in precharge before
    the next bit's cap switches (reset between consecutive regenerations)."""
    spec = _spec()
    cfg = _sar_config(spec, CLOCK)
    ck = cfg["clock"]
    tgrid = sar_time_grid(spec, cfg)
    clk = sar_input_waveforms(spec, 0.5, [None] * 6, 0, config=cfg, tgrid=tgrid)["clk"]
    for bit in range(cfg["n_bits"] - 1):
        decision = cfg["sample_end"] + (bit + 1.0) * cfg["bit_period"]
        probe = decision + ck["reset_hold"] + 2 * cfg["edge_time"]
        assert np.interp(probe, tgrid, clk) == 0.0, f"clk not reset after bit {bit}"


def test_clock_validation_rejects_degenerate_configs():
    spec = _spec()
    with pytest.raises(ValueError):
        _sar_config(spec, {"clock": {"input": "clk", "high": 0.0, "low": 0.0}})
    with pytest.raises(ValueError):
        _sar_config(spec, {"clock": {"input": "clk", "high": 0.4, "low": 0.5}})
    with pytest.raises(ValueError):
        _sar_config(spec, {"clock": {"input": "clk", "eval_before": 0.0}})
    with pytest.raises(ValueError):
        _sar_config(spec, {"clock": {"input": "clk", "reset_hold": -1e-9}})
    with pytest.raises(ValueError):
        _sar_config(spec, {"clock": {"input": "clk", "eval_before": 4e-9,
                                     "reset_hold": 7e-9}})   # sum >= bit_period
    with pytest.raises(ValueError):
        _sar_config(spec, {"clock": {}})                     # input is required


def test_schema_rejects_unknown_clock_key():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schemas" / "circuit.schema.json").read_text())
    data = json.loads(EXAMPLE.read_text())
    jsonschema.validate(data, schema)                        # sanity
    data["adc"]["clock"] = {"input": "clk", "strobe_width": 1e-9}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, schema)


# ── physical interaction (ngspice) ────────────────────────────────────────────
@needs_freepdk45
def test_independent_pinned_code():
    """Second opinion at a different code than the sibling's pinned one.

    0.2890625 is the ideal code-center of code 18 (18.5/64). The static
    three-stage comparator resolves every nominal code center to its ideal
    code (see the full-ramp pin in test_freepdk45_sar6.py), so this reads 18.
    The retired StrongARM example read 19 here: its ±1 LSB INL was real
    converter behavior, but its origin -- charge-share races in a latch whose
    input pair sat at the VTG threshold -- also produced wild codes and a
    deterministic LSB inversion elsewhere on the ramp.
    """
    from circuitopt.sar import run_sar_conversion
    result = run_sar_conversion(_spec(), 0.2890625)
    assert result["code"] == 18
    np.testing.assert_array_equal(result["bits"], [0, 1, 0, 0, 1, 0])


@needs_freepdk45
def test_delvto_reaches_the_comparator():
    """A +50 mV Vth offset on one preamp input device is several LSB of
    input-referred offset — codes on a short sweep must shift coherently. This
    proves the WP1 mismatch hook reaches the comparator front end."""
    from circuitopt.sar import run_sar_sweep
    vin = (np.array([16, 32, 48]) + 0.5) / 64.0
    nominal = run_sar_sweep(_spec(), vin, workers=3)
    shifted = run_sar_sweep(_spec(), vin, workers=3, mismatch={"MPP": 0.05})
    assert not np.array_equal(nominal["codes"], shifted["codes"])
    # offset polarity: higher Vth on the P-side preamp input weakens it, biasing
    # decisions consistently in one direction, never randomly
    diffs = shifted["codes"] - nominal["codes"]
    assert np.all(diffs <= 0) or np.all(diffs >= 0), f"incoherent shift {diffs}"
