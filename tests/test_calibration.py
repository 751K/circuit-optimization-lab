"""Cadence/Spectre calibration regression.

Drives :mod:`core.calibration` against the archived reference data under
``calibration/`` (fresh Spectre 24.1.0.078). The amp case (DC/AC/noise) is calibrated
to ~machine precision; the chopper PSS/PAC/PNoise cases match Cadence within ~1-2% on
PAC baseband gain and integrated IRN across all three corners. Every case must PASS —
these are the regression guards that catch a model/solver change drifting off Cadence.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from core import psf
from core.calibration import format_report, load_reference, run_calibration

AMP = "calibration/amp_design3_typical"
CHOPPERS = [
    "calibration/chopper_design3_typical",
    "calibration/chopper_design3_slow",
    "calibration/chopper_design3_fast",
]
_slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW_CHOPPER"),
                           reason="slow chopper PSS/PAC/PNoise; set RUN_SLOW_CHOPPER=1")


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


@_slow
@pytest.mark.parametrize("case", CHOPPERS)
def test_calibration_chopper_matches_cadence(case):
    # Chopper PAC baseband gain + PNoise IRN must stay within the per-case tolerances
    # (~2% PAC, ~3% IRN) of fresh Cadence across typical/slow/fast.
    report = run_calibration(case, analyses=["pac", "pnoise"])
    assert report["overall_pass"], format_report(report)
    assert report["results"]["pac"]["metrics"]["gain_baseband"]["pass"]
    assert report["results"]["pnoise"]["metrics"]["irn_uVrms"]["pass"]


SC_LPF = "calibration/sc_lpf"


def test_sc_lpf_calibration_uses_adaptive_average_gear2_default():
    metadata = json.loads(Path(SC_LPF, "metadata.json").read_text())
    solver = metadata["solver"]
    assert solver["integration_method"] == "gear2"
    assert solver["adaptive"] is True
    assert solver["cap_mode"] == "average"
    assert solver["pnoise_n_period_samples"] >= 512
    assert solver["pnoise_max_sideband"] >= 20


@_slow
def test_calibration_sc_lpf_matches_cadence():
    # Second periodic calibration case beside the chopper: a single-ended switched-
    # capacitor LPF (vsource clocks, reverse-biased PMOS switches). It now also
    # guards the SC-LPF calibration default: gear2 + adaptive + cap_mode="average"
    # with enough PNoise sampling to match the archived Spectre reference.
    report = run_calibration(SC_LPF, analyses=["pac", "pnoise"])
    assert report["overall_pass"], format_report(report)
    assert report["results"]["pac"]["metrics"]["gain_baseband"]["pass"]
    assert report["results"]["pac"]["metrics"]["bw_Hz"]["pass"]
    assert report["results"]["pnoise"]["metrics"]["out_uVrms"]["pass"]
