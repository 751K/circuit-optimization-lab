"""Cadence/Spectre calibration regression.

Drives :mod:`circuitopt.calibration` against the archived reference data under
``calibration/`` (fresh Spectre 24.1.0.078). The amp case (DC/AC/noise) is calibrated
to ~machine precision; the chopper PSS/PAC/PNoise cases match Cadence within ~1-2% on
PAC baseband gain and integrated IRN across all three corners. Every case must PASS —
these are the regression guards that catch a model/solver change drifting off Cadence.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from circuitopt.calibration import (
    _sc_lpf_event_tgrid,
    format_report,
    load_reference,
    run_calibration,
    run_local,
)

AMP = "calibration/amp_design3_typical"
CHOPPERS = [
    "calibration/chopper_design3_typical",
    "calibration/chopper_design3_slow",
    "calibration/chopper_design3_fast",
]

pytestmark = [pytest.mark.cadence_regression, pytest.mark.slow_regression]


def test_psf_parses_amp_reference():
    loaded = load_reference(AMP)
    assert loaded["provenance"]["spectre_version"]            # provenance from PSF HEADER
    fr, out, dev = loaded["ref"]["noise"]
    assert out[0] > 0 and dev[next(iter(dev))].shape[1] == 3  # (flicker, thermal, total)
    freqs, sig = loaded["ref"]["ac"]
    assert {"VOP", "VON", "vip", "vin"} <= set(sig)


def test_calibration_amp_passes():
    report = run_calibration(AMP)
    assert report["overall_pass"], format_report(report)
    assert report["results"]["ac"]["metrics"]["gain_dc_dB"]["pass"]
    assert report["results"]["noise"]["metrics"]["irn_uVrms"]["pass"]


def test_calibration_amp_dc_exact():
    # DC operating point matches Spectre to well under a millivolt.
    report = run_calibration(AMP, analyses=["dc"])
    for row in report["results"]["dc"]["metrics"].values():
        assert abs(row["delta"]) < 1e-3


@pytest.mark.parametrize("case", CHOPPERS)
def test_calibration_chopper_matches_cadence(case):
    # Chopper PAC baseband gain + PNoise IRN must stay within the per-case tolerances
    # (~2% PAC, ~3% IRN) of fresh Cadence across typical/slow/fast.
    report = run_calibration(case, analyses=["pac", "pnoise"])
    assert report["overall_pass"], format_report(report)
    assert report["results"]["pac"]["metrics"]["gain_baseband"]["pass"]
    assert report["results"]["pnoise"]["metrics"]["irn_uVrms"]["pass"]


SC_LPF = "calibration/sc_lpf"


def test_sc_lpf_calibration_uses_two_stage_average_gear2_default():
    metadata = json.loads(Path(SC_LPF, "metadata.json").read_text())
    solver = metadata["solver"]
    assert solver["integration_method"] == "gear2"
    assert solver["adaptive"] is True
    assert solver["n_points"] >= 201
    assert solver["final_n_points"] >= 3201
    assert solver["cap_mode"] == "average"
    assert solver["pnoise_n_period_samples"] >= 512
    assert solver["pnoise_max_sideband"] >= 20

    circuit = metadata["circuit"]
    period = 1.0 / circuit["f_clk"]
    final_grid = _sc_lpf_event_tgrid(circuit, solver["final_n_points"])
    expected = []
    for shift in (0.0, 0.5 * period):
        for offset in (
            0.0,
            circuit["edge_time"],
            circuit["duty"] * period,
            circuit["duty"] * period + circuit["edge_time"],
        ):
            expected.append(np.mod(shift + offset, period))
    for event in expected:
        assert np.any(np.isclose(final_grid, event, rtol=0.0, atol=1e-15))


def test_calibration_sc_lpf_matches_cadence():
    # Second periodic calibration case beside the chopper: a single-ended switched-
    # capacitor LPF (vsource clocks, reverse-biased PMOS switches). Adaptive
    # Gear2 supplies only the warm start; deterministic event-aligned shooting,
    # average charge caps, and the PNoise grid match Spectre.
    report = run_calibration(SC_LPF, analyses=["pac", "pnoise"])
    assert report["overall_pass"], format_report(report)
    assert report["results"]["pac"]["metrics"]["gain_baseband"]["pass"]
    assert report["results"]["pac"]["metrics"]["bw_Hz"]["pass"]
    assert report["results"]["pnoise"]["metrics"]["out_uVrms"]["pass"]


def test_sc_lpf_two_stage_result_is_insensitive_to_adaptive_initial_step():
    metadata = json.loads(Path(SC_LPF, "metadata.json").read_text())
    values = []
    for scale in (0.9, 1.1):
        trial = copy.deepcopy(metadata)
        trial["solver"]["adaptive_h0"] *= scale
        values.append(run_local(trial, analyses=["pnoise"])["pnoise"]["out_uVrms"])
    assert values[0] == pytest.approx(values[1], rel=2e-5)
