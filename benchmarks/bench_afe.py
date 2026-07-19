"""Fixed AFE performance benchmark.

Measures the canonical local-solver workloads:
  - ac121:    AC solve on 121 logarithmic frequency points.
  - noise121: noise analysis on the same 121 points.
  - tran200: transient step response on 200 time points.

The first run is reported as "cold"; later runs are reported as "warm". Select
the implementation (rust is the only engine in v2.0.0). A cold run
includes first-call JIT work.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


from circuitopt.ac_solver import ac_solve
from circuitopt._engine import current_engine
from circuitopt.circuit_loader import load_circuit_json
from circuitopt.noise_solver import band_rms, noise_analysis
from circuitopt.transient_solver import transient


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = load_circuit_json(_ROOT / "examples" / "afe_explore.json")
TOPO = _SPEC.topology
SIZES = dict(_SPEC.sizes)
BIAS = dict(_SPEC.bias)
NF = _SPEC.nf


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


def bench_case(label, fn, summarize, warm_runs):
    cold_out, cold_ms = _time_call(fn)
    warm_ms = []
    warm_out = cold_out
    for _ in range(warm_runs):
        warm_out, dt = _time_call(fn)
        warm_ms.append(dt)
    data = {"case": label, **_summarize_times(cold_ms, warm_ms), **summarize(warm_out)}
    return data


def build_inputs():
    freqs = np.logspace(0, 4, 121)
    t = np.linspace(0, 0.004, 200)
    vcm = np.full(len(t), BIAS["VCM"])
    vp = vcm + np.where(t >= 0.5e-3, +0.5e-3, 0.0)
    vn = vcm - np.where(t >= 0.5e-3, +0.5e-3, 0.0)
    return freqs, t, vp, vn


def run_benchmarks(warm_runs, skip_noise=False, skip_tran=False):
    freqs, t, vp, vn = build_inputs()
    results = []

    results.append(bench_case(
        "ac121",
        lambda: ac_solve(SIZES, BIAS, freqs, topo=TOPO, nf=NF),
        lambda r: {
            "gain_db": float(20 * np.log10(r["gains"].max())),
            "bw_Hz": float(r["bw_Hz"]),
            "dc_vop": float(r["dc_op"]["VOP"]),
        },
        warm_runs,
    ))

    if not skip_noise:
        results.append(bench_case(
            "noise121",
            lambda: noise_analysis(SIZES, BIAS, freqs, topo=TOPO, nf=NF),
            lambda r: {
                "irn_uV_1_100Hz": float(band_rms(freqs, r["irn_psd"], 1.0, 100.0) * 1e6),
                "out_uV_1_100Hz": float(band_rms(freqs, r["out_psd"], 1.0, 100.0) * 1e6),
            },
            warm_runs,
        ))

    if not skip_tran:
        results.append(bench_case(
            "tran200",
            lambda: transient(SIZES, BIAS, t, topo=TOPO, inputs={"vip": vp, "vin": vn},
                              nf=NF),
            lambda r: {
                "nfail": int(r["nfail"]),
                "vout_end": float(r["vout"][-1]),
            },
            warm_runs,
        ))

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "engine": current_engine(),
        "warm_runs": warm_runs,
        "results": results,
    }


def print_text(report):
    def fmt_ms(value):
        return "n/a" if value is None else f"{value:.3f}"

    print(f"python={report['python']} numpy={report['numpy']} engine={report['engine']} "
          f"warm_runs={report['warm_runs']}")
    for item in report["results"]:
        print(f"{item['case']}: cold_ms={item['cold_ms']:.3f} "
              f"warm_median_ms={fmt_ms(item['warm_median_ms'])} "
              f"warm_min_ms={fmt_ms(item['warm_min_ms'])}")
        for key, value in item.items():
            if key in {"case", "cold_ms", "warm_runs_ms", "warm_median_ms", "warm_min_ms"}:
                continue
            print(f"  {key}={value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--skip-noise", action="store_true")
    parser.add_argument("--skip-tran", action="store_true")
    args = parser.parse_args()

    report = run_benchmarks(args.warm_runs, skip_noise=args.skip_noise,
                            skip_tran=args.skip_tran)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)


if __name__ == "__main__":
    main()
