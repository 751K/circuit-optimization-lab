"""Tests for the SAR ADC figures (``examples/plot_adc.py``) and their CLI wiring.

Two layers, mirroring the repo's other plot tests:

* **Pure-Python** (no ngspice / no PDK) — synthetic result dicts matching the real
  key shapes drive ``plot_sar_static`` / ``plot_sar_spectrum`` / ``plot_sar_mc``, and
  a minimal fake conversion drives ``plot_sar_conversion`` (including the
  missing-optional-keys / no-clk path). These run anywhere matplotlib is installed.
* **ngspice-guarded** — the real 3-bit example runs each of the four functions end to
  end, plus a ``python -m circuitopt adc … --plot`` subprocess smoke test.

Every figure assertion checks the PNG exists and is > 1 KB (i.e. actually rendered).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from circuitopt.toolchain import pdk_root
from examples.plot_adc import (plot_sar_conversion, plot_sar_mc, plot_sar_spectrum,
                               plot_sar_static)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE3 = ROOT / "examples" / "freepdk45_sar3.json"
_HAVE = (Path(pdk_root()) / "freepdk45" / "models_nom" / "NMOS_VTG.inc").is_file()
_needs_freepdk45 = pytest.mark.skipif(
    not _HAVE, reason="FreePDK45 cards not present")


def _png_ok(path):
    assert Path(path).is_file()
    assert Path(path).stat().st_size > 1024, f"{path} is suspiciously small"


# ── pure-Python: synthetic result dicts (no ngspice) ──────────────────────────

def _synthetic_sweep(n_bits=3):
    from circuitopt.adc import static_ramp_metrics
    levels = 1 << n_bits
    vin = (np.arange(levels) + 0.5) / levels
    codes = np.arange(levels, dtype=np.int64)
    metrics = static_ramp_metrics(vin, codes, n_bits, vmin=0.0, vmax=1.0)
    return {"vin": vin, "codes": codes, "metrics": metrics, "n_bits": n_bits, "vref": 1.0}


def test_static_synthetic(tmp_path):
    path = plot_sar_static(_synthetic_sweep(), out_dir=tmp_path, note="synthetic")
    _png_ok(path)


def test_spectrum_synthetic(tmp_path):
    from circuitopt.adc import dynamic_metrics
    n = 64
    tone_bin = 5
    phase = 2.0 * np.pi * tone_bin * np.arange(n) / n
    codes = np.round(31.5 + 28 * np.sin(phase)).astype(np.int64)
    metrics = dynamic_metrics(codes, 10e6, fundamental_bin=tone_bin)
    result = {"codes": codes, "metrics": metrics, "n_bits": 6, "vref": 1.0}
    _png_ok(plot_sar_spectrum(result, out_dir=tmp_path))


def test_mc_synthetic_with_overflow(tmp_path):
    # Two clean trials + one non-monotonic (inf DNL/INL, NaN offset) must render, with
    # the inf trials landing in the labeled overflow bin rather than crashing.
    arrays = {
        "max_abs_dnl": np.array([0.20, 0.35, np.inf]),
        "max_abs_inl": np.array([0.40, 0.55, np.inf]),
        "offset_lsb": np.array([0.10, -0.22, np.nan]),
        "missing_codes": np.array([0.0, 0.0, 3.0]),
    }
    summary = {"n": 3, "yield": 1 / 3, "monotonic_rate": 2 / 3,
               "dnl_threshold": 0.5, "inl_threshold": 0.5}
    _png_ok(plot_sar_mc({"arrays": arrays, "summary": summary}, out_dir=tmp_path))


def _fake_conversion(n_bits=3, with_clk=False):
    t = np.linspace(0.0, 9e-8, 60)
    ramp = np.clip((t - 1e-8) / 8e-8, 0, 1)
    waves = {"sample": 1.0 - ramp, "sample_b": ramp}
    bit_keys = [f"b{n_bits - 1 - i}p" for i in range(n_bits)]
    for i, key in enumerate(bit_keys):
        waves[key] = 0.5 + 0.3 * np.sin(2 * np.pi * (i + 1) * t / 9e-8)
    if with_clk:
        waves["clk"] = (np.sin(2 * np.pi * 6 * t / 9e-8) > 0).astype(float)
    nodes = {"TOPP": 0.5 + 0.1 * np.cos(2 * np.pi * t / 9e-8),
             "TOPN": 0.5 - 0.1 * np.cos(2 * np.pi * t / 9e-8),
             "vout": 0.5 + 0.4 * np.sin(2 * np.pi * 3 * t / 9e-8)}
    decisions = [{"bit": i, "decision_time": 1e-8 + (i + 1) * 2e-8,
                  "comparator_v": 0.1 + 0.05 * i, "kept": bool(i % 2)}
                 for i in range(n_bits)]
    return {"t": t, "input_waveforms": waves, "transient": {"nodes": nodes},
            "decisions": decisions, "code": 5, "bits": [1, 0, 1], "vin": 0.7, "vref": 1.0}


def test_conversion_synthetic_no_clk(tmp_path):
    adc = {"sample_input": "sample", "sample_bar_input": "sample_b",
           "bit_inputs": ["b2p", "b1p", "b0p"], "comparator_node": "vout",
           "comparator_threshold": 0.5}
    _png_ok(plot_sar_conversion(_fake_conversion(with_clk=False), adc=adc, out_dir=tmp_path))


def test_conversion_missing_optional_keys(tmp_path):
    # No adc block (keys inferred), no clk, empty decisions, comparator node absent:
    # the function must degrade to whatever it can draw without raising.
    result = _fake_conversion(with_clk=False)
    result["decisions"] = []
    result["transient"]["nodes"].pop("vout")
    _png_ok(plot_sar_conversion(result, adc=None, out_dir=tmp_path,
                                filename="degraded.png"))


# ── ngspice-guarded: real 3-bit example, end to end ───────────────────────────

@_needs_freepdk45
def test_conversion_3bit_real(tmp_path):
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.sar import run_sar_conversion
    spec = load_circuit_json(EXAMPLE3)
    conv = run_sar_conversion(spec, 0.7)
    assert spec.adc.get("clock") is None            # 3-bit is the static-comparator case
    _png_ok(plot_sar_conversion(conv, adc=spec.adc, out_dir=tmp_path))


@_needs_freepdk45
def test_static_3bit_real(tmp_path):
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.sar import run_sar_sweep
    spec = load_circuit_json(EXAMPLE3)
    vin = (np.arange(8) + 0.5) / 8.0
    _png_ok(plot_sar_static(run_sar_sweep(spec, vin), out_dir=tmp_path))


@_needs_freepdk45
def test_spectrum_3bit_real(tmp_path):
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.sar import run_sar_signal
    spec = load_circuit_json(EXAMPLE3)
    n, tone = 16, 1
    phase = 2.0 * np.pi * tone * np.arange(n) / n
    vin = 0.5 + 0.45 * np.sin(phase)
    sig = run_sar_signal(spec, vin, 10e6, fundamental_bin=tone)
    _png_ok(plot_sar_spectrum(sig, out_dir=tmp_path))


@_needs_freepdk45
def test_mc_3bit_real(tmp_path):
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.sar_mc import sar_mismatch_mc
    spec = load_circuit_json(EXAMPLE3)
    mc = sar_mismatch_mc(spec, n=2, seed=1)
    _png_ok(plot_sar_mc(mc, out_dir=tmp_path))


@_needs_freepdk45
def test_cli_adc_plot_subprocess(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE3),
         "--vin", "0.7", "--plot", str(tmp_path)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    pngs = list(Path(tmp_path).glob("*.png"))
    assert pngs, proc.stdout + proc.stderr
    for p in pngs:
        _png_ok(p)
