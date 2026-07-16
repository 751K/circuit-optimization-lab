"""Adversarial semantic tests for the ADC plotting work package.

Reviewer-side verification, fourth round. The agent's tests cover the happy
paths; these attack degenerate data (all-overflow MC, missing codes, single
trial), key derivation on both circuit generations, and the CLI's new --mc
mode. Synthetic tests run without ngspice; physical ones are skip-guarded.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from circuitopt.toolchain import pdk_root                   # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE3 = ROOT / "examples" / "freepdk45_sar3.json"
_HAVE = (Path(pdk_root()) / "freepdk45" / "models_nom" / "NMOS_VTG.inc").is_file()
needs_freepdk45 = pytest.mark.skipif(
    not _HAVE, reason="FreePDK45 cards not present")


def _fake_mc(dnl, inl, offset, *, dnl_thr=0.5, inl_thr=0.5):
    dnl = np.asarray(dnl, float)
    inl = np.asarray(inl, float)
    offset = np.asarray(offset, float)
    passed = ((dnl <= dnl_thr) & (inl <= inl_thr))
    return {
        "arrays": {"max_abs_dnl": dnl, "max_abs_inl": inl, "offset_lsb": offset,
                   "missing_codes": np.zeros_like(dnl)},
        "summary": {"n": len(dnl), "yield": float(np.mean(passed)),
                    "monotonic_rate": float(np.mean(np.isfinite(dnl))),
                    "dnl_threshold": dnl_thr, "inl_threshold": inl_thr},
        "config": {"dnl_threshold": dnl_thr, "inl_threshold": inl_thr},
    }


def test_mc_plot_survives_all_overflow(tmp_path):
    """Every trial non-monotonic (inf DNL/INL, NaN offset): the figure must still
    render with only overflow bins — np.histogram on an all-inf array is a
    classic crash site."""
    from examples.plot_adc import plot_sar_mc
    mc = _fake_mc([np.inf] * 4, [np.inf] * 4, [np.nan] * 4)
    out = plot_sar_mc(mc, out_dir=tmp_path)
    assert Path(out).is_file() and Path(out).stat().st_size > 1024


def test_mc_plot_single_trial(tmp_path):
    """n=1 with zero spread: histogram range collapses to a point."""
    from examples.plot_adc import plot_sar_mc
    out = plot_sar_mc(_fake_mc([0.0], [0.0], [0.0]), out_dir=tmp_path)
    assert Path(out).is_file()


def test_static_plot_with_missing_codes_and_nan_transitions(tmp_path):
    """A ramp with a skipped code produces NaN transitions and a missing-code
    entry — the staircase overlay and DNL bars must tolerate the NaNs."""
    from circuitopt.adc import static_ramp_metrics
    from examples.plot_adc import plot_sar_static
    vin = (np.arange(8) + 0.5) / 8.0
    codes = np.array([0, 1, 3, 3, 4, 5, 6, 7])          # code 2 missing
    metrics = static_ramp_metrics(vin, codes, 3, vmin=0.0, vmax=1.0)
    assert len(metrics["missing_codes"]) > 0            # test data is as intended
    sweep = {"vin": vin, "codes": codes, "metrics": metrics, "n_bits": 3, "vref": 1.0}
    out = plot_sar_static(sweep, out_dir=tmp_path)
    assert Path(out).is_file() and Path(out).stat().st_size > 1024


def test_spectrum_plot_small_record(tmp_path):
    """Shortest legal record (8 samples, 1 cycle): only 5 rfft bins, harmonics all
    alias — annotation placement must not index outside the spectrum."""
    from circuitopt.adc import dynamic_metrics
    from examples.plot_adc import plot_sar_spectrum
    codes = np.round(3.5 + 3.5 * np.sin(2 * np.pi * np.arange(8) / 8)).astype(int)
    sig = {"codes": codes, "metrics": dynamic_metrics(codes, 1e6, fundamental_bin=1),
           "n_bits": 3, "vref": 1.0}
    out = plot_sar_spectrum(sig, out_dir=tmp_path)
    assert Path(out).is_file()


def test_conversion_plot_derives_keys_from_spec_not_hardcoded(tmp_path):
    """Rename every waveform key in a synthetic conversion result: the plot must
    follow the adc block's names (a hardcoded 'clk'/'sample' would KeyError)."""
    from examples.plot_adc import plot_sar_conversion
    n = 64
    t = np.linspace(0.0, 1e-7, n)
    keys = ["strobe_x", "smp", "smp_b", "m2", "m1", "m0"]
    conversion = {
        "vin": 0.5, "code": 3, "bits": np.array([0, 1, 1], np.int8), "n_bits": 3,
        "vref": 1.0, "t": t,
        "input_waveforms": {k: np.linspace(0, 1, n) for k in keys},
        "transient": {"nodes": {"TP": np.full(n, 0.4), "TN": np.full(n, 0.6),
                                "cmp": np.linspace(1, 0, n)}},
        "decisions": [{"bit": b, "decision_time": 2e-8 * (b + 1),
                       "comparator_v": 0.3, "kept": bool(b)} for b in range(3)],
    }
    adc = {"type": "sar", "n_bits": 3, "vref": 1.0,
           "bit_inputs": ["m2", "m1", "m0"], "sample_input": "smp",
           "sample_bar_input": "smp_b", "comparator_node": "cmp",
           "comparator_threshold": 0.5, "sample_end": 1e-8, "bit_period": 2e-8,
           "edge_time": 2e-10, "clock": {"input": "strobe_x"}}
    out = plot_sar_conversion(conversion, adc, out_dir=tmp_path)
    assert Path(out).is_file() and Path(out).stat().st_size > 1024


@needs_freepdk45
def test_cli_mc_mode_with_plot(tmp_path):
    """--mc N end-to-end on the 3-bit example: exits 0, prints a yield, renders."""
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE3),
         "--mc", "2", "--seed", "0", "--workers", "2", "--plot", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert "yield" in proc.stdout.lower()
    pngs = list(tmp_path.glob("*.png"))
    assert pngs and pngs[0].stat().st_size > 1024


@needs_freepdk45
def test_cli_mc_exclusive_with_vin():
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE3),
         "--mc", "2", "--vin", "0.5"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0


def test_plot_results_dir_not_polluted_by_tests(tmp_path):
    """Figure functions honor out_dir strictly — nothing may leak into results/."""
    from examples.plot_adc import plot_sar_mc
    results_dir = ROOT / "results"
    before = set(results_dir.glob("*.png")) if results_dir.is_dir() else set()
    plot_sar_mc(_fake_mc([0.1, 0.2], [0.1, 0.2], [0.0, 0.1]), out_dir=tmp_path)
    after = set(results_dir.glob("*.png")) if results_dir.is_dir() else set()
    assert before == after
