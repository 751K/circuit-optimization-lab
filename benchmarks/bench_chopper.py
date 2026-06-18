"""Chopper analysis performance benchmarks.

Measures five canonical chopper workloads at increasing complexity:

  - harmonics:    Finite-edge harmonic coefficient computation (pure math)
  - ideal:        Ideal LPTV chopper (frequency-domain sideband folding, no switches)
  - pmos_static:  PMOS-switch static-phase gain/BW/noise (8 switches + AFE)
  - pmos_lptv:    Quasi-static PMOS sideband folding with finite-edge weights
  - pmos_tran:    Hard-switched PMOS chopper transient (finite-edge clocks)

All cases use f_chop=225 Hz (realistic ECG chopper frequency) and the
default locked-design AFE sizes. Numba is enabled by default when installed,
so cold runs include any first-call JIT work unless CIRCUIT_USE_NUMBA=0.

Usage:
  python3 -m benchmarks.bench_chopper --warm-runs 3
  CIRCUIT_USE_NUMBA=0 python3 -m benchmarks.bench_chopper --warm-runs 3 --json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from core.chopper import (
    chopper_analysis,
    finite_edge_chopper_harmonics,
    pmos_chopper_analysis,
    pmos_chopper_lptv_analysis,
    pmos_chopper_transient,
)
from core.numba_kernels import NUMBA_AVAILABLE

# ── fixed AFE design (locked, from bench_afe.py) ──────────────────────
SIZES = {
    "M6": (2264, 78), "M7": (61365, 61), "M8": (61365, 61),
    "M9": (3175, 468), "M10": (3175, 468), "M11": (465, 66),
    "M12": (894, 85), "M13": (894, 85), "M14": (5224, 46), "M15": (5224, 46),
}
BIAS = {"VDD": 40.0, "VCM": 30.65, "VB": 9.84, "VC": 16.0}

# ── chopper parameters ─────────────────────────────────────────────────
F_CHOP = 225.0               # Hz, realistic ECG chopper frequency
SWITCH_SIZE = (5000, 30)     # locked-design switch W/L
EDGE_TIME = 20e-6            # 20 us finite edge
NFREQ = 61                   # baseband frequency points
BAND = (0.05, 100.0)         # IRN integration band (Hz)
MAX_HARMONIC = 31            # odd harmonics up to 31

# ── transient grid: 2 chopper cycles ───────────────────────────────────
_CYCLES = 2
_PERIOD = 1.0 / F_CHOP
_TSTOP = _CYCLES * _PERIOD
_TGRID = np.linspace(0.0, _TSTOP, 201)


def _time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1e3


def _summarize_times(cold_ms, warm_ms):
    warm = np.array(warm_ms, float)
    return {
        "cold_ms": float(cold_ms),
        "warm_runs_ms": [float(x) for x in warm_ms],
        "warm_median_ms": float(np.median(warm)) if len(warm) else None,
        "warm_min_ms": float(np.min(warm)) if len(warm) else None,
    }


def bench_case(name, fn, summarize, warm_runs):
    cold_out, cold_ms = _time_call(fn)
    warm_ms = []
    warm_out = cold_out
    for _ in range(warm_runs):
        warm_out, dt = _time_call(fn)
        warm_ms.append(dt)
    data = {"case": name, **_summarize_times(cold_ms, warm_ms), **summarize(warm_out)}
    return data


def run_benchmarks(warm_runs, skip_tran=False):
    freqs = np.logspace(np.log10(0.1), np.log10(500.0), NFREQ)
    results = []

    # ── harmonics: finite-edge Fourier coefficients ──────────────────
    results.append(bench_case(
        "harmonics",
        lambda: finite_edge_chopper_harmonics(
            MAX_HARMONIC,
            edge_fraction=EDGE_TIME * F_CHOP,
            dead_fraction=0.0,
            samples=4096,
        ),
        lambda r: {"weight_sum": float(np.sum(r[2])), "n_harm": int(len(r[0]))},
        warm_runs,
    ))

    # ── ideal: ideal LPTV chopper ────────────────────────────────────
    results.append(bench_case(
        "ideal",
        lambda: chopper_analysis(
            SIZES, BIAS, freqs, F_CHOP,
            band=BAND, max_harmonic=MAX_HARMONIC,
        ),
        lambda r: {
            "Av_dc_dB": float(r["Av_dc_dB"]),
            "peak_dB": float(r["peak_dB"]),
            "bw_Hz": float(r["bw_Hz"]),
            "irn_uV_band": float(r["irn_uV_band"]),
            "n_sideband": int(len(r["sideband_freqs"])),
        },
        warm_runs,
    ))

    # ── pmos_static: PMOS-switch static-phase analysis ───────────────
    results.append(bench_case(
        "pmos_static",
        lambda: pmos_chopper_analysis(
            SIZES, BIAS, freqs,
            switch_size=SWITCH_SIZE, switch_nf=1,
            band=BAND, phases=("A", "B"),
        ),
        lambda r: {
            "Av_dc_dB": float(r["Av_dc_dB"]),
            "peak_dB": float(r["peak_dB"]),
            "bw_Hz": float(r["bw_Hz"]),
            "irn_uV_band": float(r["irn_uV_band"]),
        },
        warm_runs,
    ))

    # ── pmos_lptv: quasi-static PMOS sideband folding ────────────────
    results.append(bench_case(
        "pmos_lptv",
        lambda: pmos_chopper_lptv_analysis(
            SIZES, BIAS, freqs, F_CHOP,
            switch_size=SWITCH_SIZE, switch_nf=1,
            band=BAND, max_harmonic=MAX_HARMONIC,
            edge_time=EDGE_TIME,
        ),
        lambda r: {
            "Av_dc_dB": float(r["Av_dc_dB"]),
            "peak_dB": float(r["peak_dB"]),
            "bw_Hz": float(r["bw_Hz"]),
            "irn_uV_band": float(r["irn_uV_band"]),
            "n_sideband": int(len(r["sideband_freqs"])),
        },
        warm_runs,
    ))

    # ── pmos_tran: hard-switched PMOS chopper transient ──────────────
    if not skip_tran:
        results.append(bench_case(
            "pmos_tran",
            lambda: pmos_chopper_transient(
                SIZES, BIAS, _TGRID, F_CHOP,
                switch_size=SWITCH_SIZE, switch_nf=1,
                edge_time=EDGE_TIME, refine_edges=True,
                clock_style="pulse",
            ),
            lambda r: {
                "nfail": int(r["nfail"]),
                "npoints": int(r["refined_point_count"]),
                "vout_end": float(r["requested_output"][-1]),
            },
            warm_runs,
        ))

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba_enabled": bool(NUMBA_AVAILABLE),
        "numba_env": os.environ.get("CIRCUIT_USE_NUMBA"),
        "f_chop_Hz": F_CHOP,
        "switch_W_um": SWITCH_SIZE[0],
        "switch_L_um": SWITCH_SIZE[1],
        "edge_time_us": EDGE_TIME * 1e6,
        "n_cycles": _CYCLES,
        "warm_runs": warm_runs,
        "results": results,
    }


def print_text(report):
    def fmt_ms(value):
        return "n/a" if value is None else f"{value:.3f}"

    print(f"python={report['python']} numpy={report['numpy']} "
          f"numba_enabled={report['numba_enabled']} f_chop={report['f_chop_Hz']}Hz "
          f"switch={report['switch_W_um']}/{report['switch_L_um']} "
          f"edge={report['edge_time_us']}us warm_runs={report['warm_runs']}")
    for item in report["results"]:
        print(f"{item['case']}: cold_ms={item['cold_ms']:.3f} "
              f"warm_median_ms={fmt_ms(item['warm_median_ms'])} "
              f"warm_min_ms={fmt_ms(item['warm_min_ms'])}")
        for key, value in item.items():
            if key in {"case", "cold_ms", "warm_runs_ms", "warm_median_ms", "warm_min_ms"}:
                continue
            if isinstance(value, float):
                print(f"  {key}={value:.4g}")
            else:
                print(f"  {key}={value}")


def main():
    parser = argparse.ArgumentParser(
        description="Chopper analysis performance benchmarks")
    parser.add_argument("--warm-runs", type=int, default=3,
                        help="number of warm runs per case (default: 3)")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON")
    parser.add_argument("--skip-tran", action="store_true",
                        help="skip the slow pmos_tran case")
    args = parser.parse_args()

    report = run_benchmarks(args.warm_runs, skip_tran=args.skip_tran)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)


if __name__ == "__main__":
    main()
