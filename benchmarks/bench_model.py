"""Single-device PMOS_TFT micro-benchmark.

Measures hot-path operations of the PMOS_TFT device model across three bias
regions: saturation, subthreshold, and linear.  Reports cold (fresh device,
first-call) and warm (subsequent calls on the same device) timings per
operation. Warm timings batch repeated calls and report per-call time so
sub-microsecond kernels are not dominated by timer quantization. Select the
implementation with CIRCUIT_ENGINE=rust|numba|python. A Numba cold run includes
first-call JIT work.

Usage:
  python3 -m benchmarks.bench_model --warm-runs 3
  CIRCUIT_ENGINE=rust python3 -m benchmarks.bench_model --warm-runs 3 --json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from circuitopt.numba_kernels import NUMBA_AVAILABLE
from circuitopt._engine import current_engine
from circuitopt.pmos_tft_model import PMOS_TFT

# (Vs, Vd, Vg) bias points covering three operating regions.
BIAS_POINTS = {
    "saturation":   (40.0, 0.0,  20.0),
    "subthreshold": (40.0, 0.0,  38.0),
    "linear":       (40.0, 35.0, 20.0),
}

DEFAULT_W = 1000
DEFAULT_L = 20


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


def _fresh_device():
    """Return a new PMOS_TFT with the default W/L and no cached OP state."""
    return PMOS_TFT(W=DEFAULT_W, L=DEFAULT_L)


def bench_case(name, fn, summarize, warm_runs, inner_loops):
    """Time one operation with a fresh device for the cold run, then warm repeats."""
    dev = _fresh_device()
    cold_out, cold_ms = _time_call(lambda: fn(dev))
    warm_ms = []
    warm_out = cold_out
    for _ in range(warm_runs):
        def run_batch():
            result = None
            for _ in range(inner_loops):
                result = fn(dev)
            return result

        warm_out, dt = _time_call(run_batch)
        warm_ms.append(dt / inner_loops)
    data = {"case": name, **_summarize_times(cold_ms, warm_ms), **summarize(warm_out)}
    return data


def run_benchmarks(warm_runs, inner_loops=10_000):
    results = []

    for region, (Vs, Vd, Vg) in BIAS_POINTS.items():

        # --- op: DC operating point (internal-node Newton solve) ---
        results.append(bench_case(
            f"{region}/op",
            lambda d: d.get_op(Vs, Vd, Vg),
            lambda r: {"Vs1_V": float(r[0]), "Vd1_V": float(r[1])},
            warm_runs, inner_loops,
        ))

        # --- idc: drain current ---
        results.append(bench_case(
            f"{region}/idc",
            lambda d: d.get_Idc(Vs, Vd, Vg),
            lambda r: {"Idc_uA": float(r * 1e6)},
            warm_runs, inner_loops,
        ))

        # --- caps: small-signal capacitances ---
        results.append(bench_case(
            f"{region}/caps",
            lambda d: d.get_capacitances(Vs, Vd, Vg),
            lambda r: {"Cgss_pF": float(r[0] * 1e12), "Cgdd_pF": float(r[1] * 1e12)},
            warm_runs, inner_loops,
        ))

        # --- idc_caps: combined Idc + capacitances (shared OP solve) ---
        results.append(bench_case(
            f"{region}/idc_caps",
            lambda d: d.get_Idc_and_capacitances(Vs, Vd, Vg),
            lambda r: {"Idc_uA": float(r[0] * 1e6)},
            warm_runs, inner_loops,
        ))

        # --- noise: noise PSD at a single frequency ---
        results.append(bench_case(
            f"{region}/noise",
            lambda d: d.get_noise_psd(Vs, Vd, Vg, frequency=100.0),
            lambda r: {"S_th_A2Hz": float(r[0]), "S_fl_A2Hz": float(r[1])},
            warm_runs, inner_loops,
        ))

        # --- os: full operating-point dictionary ---
        results.append(bench_case(
            f"{region}/os",
            lambda d: d.get_os(Vs, Vd, Vg),
            lambda r: {
                "Idc_uA": float(r["Idc"] * 1e6),
                "gm_uS": float(r["gm"] * 1e6),
                "rout_MOhm": float(r["rout"] * 1e-6),
            },
            warm_runs, inner_loops,
        ))

        # --- metrics: Cadence-style metrics (gain, ft, gm/Id) ---
        results.append(bench_case(
            f"{region}/metrics",
            lambda d: d.get_cadence_metrics(Vs, Vd, Vg),
            lambda r: {
                "gain_dB": float(r["gain"]),
                "ft_Hz": float(r["ft"]),
                "gm_id": float(r["gm_id"]),
            },
            warm_runs, inner_loops,
        ))

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba_enabled": bool(NUMBA_AVAILABLE),
        "engine": current_engine(),
        "numba_env": os.environ.get("CIRCUIT_USE_NUMBA"),
        "device": {"W_um": DEFAULT_W, "L_um": DEFAULT_L},
        "warm_runs": warm_runs,
        "warm_inner_loops": inner_loops,
        "results": results,
    }


def print_text(report):
    def fmt_ms(value):
        return "n/a" if value is None else f"{value:.3f}"

    print(f"python={report['python']} numpy={report['numpy']} engine={report['engine']} "
          f"numba_enabled={report['numba_enabled']} W={report['device']['W_um']} "
          f"L={report['device']['L_um']} warm_runs={report['warm_runs']}")
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
        description="Single-device PMOS_TFT micro-benchmark")
    parser.add_argument("--warm-runs", type=int, default=3,
                        help="number of warm runs per case (default: 3)")
    parser.add_argument("--inner-loops", type=int, default=10_000,
                        help="calls per warm timing batch (default: 10000)")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON")
    args = parser.parse_args()

    if args.warm_runs < 1 or args.inner_loops < 1:
        parser.error("--warm-runs and --inner-loops must be positive")
    report = run_benchmarks(args.warm_runs, args.inner_loops)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)


if __name__ == "__main__":
    main()
