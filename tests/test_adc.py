import numpy as np
import pytest

from circuitopt.adc import (average_supply_power, average_waveform_source_power,
                            code_density_metrics,
                            decode_bit_waveforms, dynamic_metrics,
                            static_ramp_metrics)


def test_decode_bit_waveforms_msb_first():
    t = np.arange(8.0)
    expected = np.arange(8, dtype=np.int64)
    nodes = {
        "B2": ((expected >> 2) & 1).astype(float),
        "B1": ((expected >> 1) & 1).astype(float),
        "B0": (expected & 1).astype(float),
    }
    result = decode_bit_waveforms(
        t, nodes, ["B2", "B1", "B0"], t, threshold=0.5)
    np.testing.assert_array_equal(result["codes"], expected)
    np.testing.assert_array_equal(result["bits"][5], [1, 0, 1])


def test_static_ramp_metrics_ideal_quantizer():
    n_bits = 4
    vin = (np.arange(16000) + 0.5) / 16000
    codes = np.minimum((vin * (1 << n_bits)).astype(int), (1 << n_bits) - 1)
    result = static_ramp_metrics(vin, codes, n_bits, vmin=0.0, vmax=1.0)
    assert len(result["missing_codes"]) == 0
    assert result["max_abs_dnl"] < 2e-3
    assert result["max_abs_inl"] < 2e-3


def test_code_density_reports_missing_code():
    codes = np.tile(np.array([0, 1, 3]), 100)
    result = code_density_metrics(codes, 2)
    np.testing.assert_array_equal(result["missing_codes"], [2])
    assert result["dnl"][2] == -1.0


def test_dynamic_metrics_matches_ideal_8bit_sine():
    n = 4096
    tone_bin = 37
    phase = 2 * np.pi * tone_bin * np.arange(n) / n
    codes = np.clip(np.floor(128.0 + 126.0 * np.sin(phase)), 0, 255)
    result = dynamic_metrics(codes, 10e6, fundamental_bin=tone_bin)
    assert 47.0 < result["sndr_db"] < 52.0
    assert 7.5 < result["enob"] < 8.4
    assert result["fundamental_frequency"] == pytest.approx(tone_bin * 10e6 / n)


def test_average_supply_power_uses_ngspice_source_sign():
    t = np.linspace(0.0, 1e-6, 101)
    currents = {"rail:VDD": np.full_like(t, -25e-6)}
    result = average_supply_power(t, currents, {"VDD": 1.0})
    assert result["total_w"] == pytest.approx(25e-6)


def test_average_waveform_source_power():
    t = np.linspace(0.0, 1e-6, 101)
    voltage = np.linspace(0.0, 1.0, len(t))
    current = np.full_like(t, -10e-6)
    result = average_waveform_source_power(t, {"VDRV": current}, {"VDRV": voltage})
    assert result["total_w"] == pytest.approx(5e-6)
