import numpy as np
import pytest

from circuitopt.adc import (average_supply_power, average_waveform_source_power,
                            code_density_metrics,
                            decode_bit_waveforms, dynamic_metrics,
                            sampled_transfer_metrics, static_ramp_metrics)


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


# ── sampled_transfer_metrics ──────────────────────────────────────────────────
def test_sampled_transfer_metrics_full_density_perfect_staircase():
    n_bits = 4
    levels = 1 << n_bits
    vin = (np.arange(levels) + 0.5) / levels
    codes = np.arange(levels)
    m = sampled_transfer_metrics(vin, codes, n_bits, vmin=0.0, vmax=1.0)
    np.testing.assert_array_equal(m["ideal_codes"], codes)
    np.testing.assert_array_equal(m["code_errors"], 0)
    assert m["max_abs_code_err"] == 0.0
    assert m["monotonic"]


def test_sampled_transfer_metrics_subsample_scores_a_perfect_converter_clean():
    """The motivating aliasing bug: a perfect 6-bit converter sampled at every
    fourth code center reads ``max_abs_dnl = 3.5`` through the transition
    metrics (four boundaries alias onto one midpoint), while the code-error
    metrics correctly score it clean."""
    n_bits = 6
    levels = 1 << n_bits
    idx = np.arange(0, levels, 4)
    vin = (idx + 0.5) / levels
    codes = idx.copy()
    aliased = static_ramp_metrics(vin, codes, n_bits, vmin=0.0, vmax=1.0)
    assert aliased["max_abs_dnl"] > 1.0                   # wrong, by construction
    m = sampled_transfer_metrics(vin, codes, n_bits, vmin=0.0, vmax=1.0)
    assert m["max_abs_code_err"] == 0.0 and m["monotonic"]


def test_sampled_transfer_metrics_offset_and_missing():
    n_bits = 4
    levels = 1 << n_bits
    idx = np.arange(0, levels, 2)
    vin = (idx + 0.5) / levels
    codes = np.minimum(idx + 1, levels - 1)      # global +1 code offset
    m = sampled_transfer_metrics(vin, codes, n_bits, vmin=0.0, vmax=1.0)
    assert m["max_abs_code_err"] == 1.0
    np.testing.assert_array_equal(m["code_errors"][:-1], 1)
    # No missing-code field: a sparse ramp cannot measure missing codes, and an
    # expected-vs-produced count would misfire on exactly this offset case.
    assert "sampled_missing_codes" not in m


def test_sampled_transfer_metrics_accepts_non_monotonic_codes():
    vin = np.array([0.1, 0.3, 0.5, 0.7])
    codes = np.array([0, 5, 3, 11])
    m = sampled_transfer_metrics(vin, codes, 4, vmin=0.0, vmax=1.0)
    assert not m["monotonic"]
    assert np.isfinite(m["max_abs_code_err"])
    with pytest.raises(ValueError):
        static_ramp_metrics(vin, codes, 4, vmin=0.0, vmax=1.0)


# ── transition bisection ──────────────────────────────────────────────────────
def _ideal_probe(n_bits, offset_lsb=0.0):
    levels = 1 << n_bits
    lsb = 1.0 / levels

    def probe(vins):
        vins = np.asarray(vins, float)
        return np.clip(np.floor(vins / lsb + offset_lsb), 0, levels - 1
                       ).astype(np.int64)
    return probe


def test_bisect_transitions_locates_the_ideal_quantizer():
    from circuitopt.adc import bisect_code_transitions
    n_bits = 4
    lsb = 1.0 / 16
    targets = list(range(1, 16))
    r = bisect_code_transitions(_ideal_probe(n_bits), targets, n_bits,
                                vmin=0.0, vmax=1.0, tol_lsb=0.05)
    assert not r["unmeasured"]
    np.testing.assert_allclose(r["transitions"],
                               np.array(targets) * lsb, atol=0.05 * lsb)
    # cost: 2T bracket probes + <= ceil(log2(4/0.05)) rounds of <= T probes
    assert r["conversions"] <= 2 * 15 + 8 * 15
    assert r["rounds"] <= 8


def test_bisect_transitions_recovers_from_a_missed_bracket():
    """A +5-LSB offset pushes every transition outside the default 2-LSB
    bracket: the search must widen to the full range and still land, and the
    transitions driven below the input range must come back NaN, never a
    made-up number."""
    from circuitopt.adc import bisect_code_transitions
    n_bits = 4
    lsb = 1.0 / 16
    targets = [2, 8, 12]
    r = bisect_code_transitions(_ideal_probe(n_bits, offset_lsb=5.0), targets,
                                n_bits, vmin=0.0, vmax=1.0, tol_lsb=0.05)
    # T(k) = (k-5)*lsb; k=2 lies below 0 -> unmeasurable
    assert r["unmeasured"] == [2]
    assert np.isnan(r["transitions"][0])
    np.testing.assert_allclose(r["transitions"][1:],
                               np.array([3.0, 7.0]) * lsb, atol=0.05 * lsb)


def test_bisect_transitions_rejects_bad_probe_shape():
    from circuitopt.adc import bisect_code_transitions
    with pytest.raises(ValueError):
        bisect_code_transitions(lambda v: np.array([0]), [1, 2], 4,
                                vmin=0.0, vmax=1.0)


def test_carry_transition_codes_cover_both_bins_of_every_carry():
    from circuitopt.adc import carry_transition_codes
    assert carry_transition_codes(3) == [1, 2, 3, 4, 5]
    six = carry_transition_codes(6)
    assert six == [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33]
    for j in range(1, 6):                # both DNL bins around each carry
        m = 1 << j
        assert {m - 1, m, m + 1} <= set(six)
    assert carry_transition_codes(1) == [1]


def test_transition_dnl_inl_reduces_measured_boundaries_only():
    from circuitopt.adc import transition_dnl_inl
    lsb = 1.0 / 16
    targets = [3, 4, 5, 8]
    # code 3 compressed by 0.2 LSB, code 4 widened by 0.2; T(8) unmeasured
    transitions = [3 * lsb, 3.8 * lsb, 5 * lsb, np.nan]
    r = transition_dnl_inl(targets, transitions, 4, vmin=0.0, vmax=1.0)
    np.testing.assert_array_equal(r["dnl_codes"], [3, 4])
    np.testing.assert_allclose(r["dnl"], [-0.2, 0.2], atol=1e-12)
    np.testing.assert_allclose(r["inl"][:3], [0.0, -0.2, 0.0], atol=1e-12)
    assert np.isnan(r["inl"][3])
    assert r["max_abs_dnl"] == pytest.approx(0.2)
