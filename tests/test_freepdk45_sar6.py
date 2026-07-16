"""6-bit differential FreePDK45 SAR with a clocked StrongARM comparator."""
import json
from pathlib import Path

import numpy as np
import pytest

from circuitopt.circuit_loader import load_circuit_json
from circuitopt.sar import _sar_config, sar_input_waveforms, sar_time_grid
from circuitopt.toolchain import pdk_root


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "freepdk45_sar6.json"
EXAMPLE3 = ROOT / "examples" / "freepdk45_sar3.json"
_HAVE = (Path(pdk_root()) / "freepdk45" / "models_nom" / "NMOS_VTG.inc").is_file()
needs_freepdk45 = pytest.mark.skipif(
    not _HAVE, reason="FreePDK45 cards not present")


# ── pure-Python: schema + clock machinery + backward compatibility ─────────────
def test_sar6_matches_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schemas" / "circuit.schema.json").read_text())
    jsonschema.validate(json.loads(EXAMPLE.read_text()), schema)


def test_clock_config_present_for_sar6_absent_for_sar3():
    cfg6 = _sar_config(load_circuit_json(EXAMPLE))
    ck = cfg6["clock"]
    assert ck is not None and ck["input"] == "clk"
    assert ck["high"] == 1.0 and ck["low"] == 0.0
    # defaults derive from bit_period=1e-8
    assert ck["eval_before"] == pytest.approx(3e-9)
    assert ck["reset_hold"] == pytest.approx(1e-9)
    # the static-comparator 3-bit example must resolve to no clock (backward compat)
    assert _sar_config(load_circuit_json(EXAMPLE3))["clock"] is None


def test_clock_waveform_strobes_each_decision_time():
    spec = load_circuit_json(EXAMPLE)
    cfg = _sar_config(spec)
    tgrid = sar_time_grid(spec, cfg)
    wave = sar_input_waveforms(spec, 0.7, [None] * 6, 0, config=cfg, tgrid=tgrid)
    clk = wave["clk"]
    assert clk.min() == 0.0 and clk.max() == 1.0
    # low during the sampling phase, high at every bit's decision instant
    assert np.interp(cfg["sample_end"] * 0.5, tgrid, clk) == 0.0
    for bit in range(cfg["n_bits"]):
        dt = cfg["sample_end"] + (bit + 1.0) * cfg["bit_period"]
        assert np.interp(dt, tgrid, clk) == pytest.approx(1.0)


def test_clock_waveform_absent_and_keys_stable_for_sar3():
    """3-bit static-comparator SAR renders the same waveform key set as before."""
    spec = load_circuit_json(EXAMPLE3)
    wave = sar_input_waveforms(spec, 0.7, [1, 0, None], 2)
    assert "clk" not in wave
    assert sorted(wave) == ["b0n", "b0p", "b1n", "b1p", "b2n", "b2p",
                            "bdn", "bdp", "sample", "sample_b"]


def test_invalid_clock_eval_before_rejected():
    spec = load_circuit_json(EXAMPLE)
    # eval_before must stay below bit_period/2 - edge_time
    with pytest.raises(ValueError):
        _sar_config(spec, {"clock": {"input": "clk", "eval_before": 6e-9}})


# ── Native C BSIM4 physical conversion through the StrongARM comparator ────────
@needs_freepdk45
def test_sar6_conversion_pinned_code():
    from circuitopt.sar import run_sar_conversion
    spec = load_circuit_json(EXAMPLE)
    result = run_sar_conversion(spec, 0.7109375)
    assert result["code"] == 44
    np.testing.assert_array_equal(result["bits"], [1, 0, 1, 1, 0, 0])
    assert result["transient"]["backend"] == "bsim4_native"
    assert len(result["decisions"]) == 6
    assert np.isfinite(result["total_power_w"]) and result["total_power_w"] > 0.0


@needs_freepdk45
def test_sar6_subsampled_sweep_monotonic():
    from circuitopt.sar import run_sar_sweep
    spec = load_circuit_json(EXAMPLE)
    ideal_centers = np.array([8, 24, 40, 56])
    vin = (ideal_centers + 0.5) / 64.0
    result = run_sar_sweep(spec, vin, workers=4)
    np.testing.assert_array_equal(result["codes"], [8, 24, 41, 56])
    assert np.all(np.diff(result["codes"]) > 0)
