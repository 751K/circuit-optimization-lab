"""Periodic chopper PSS/PAC/PNoise performance benchmark.

This benchmark focuses on the production periodic flow:

  - pss:    hard-switched PMOS chopper PSS orbit
  - pac:    PAC on a fixed PSS orbit, with warm cache reuse
  - pnoise: PNoise on a fixed PSS+PAC orbit, with warm cache reuse

The first timed call for each stage is reported as "cold"; repeated calls are
reported as "warm".  The PAC and PNoise stages intentionally reuse the same PSS
object across cold/warm calls so cache-hit fields show whether the stage-level
linearization/adjoint caches are effective.

Usage:
  python3 -m benchmarks.bench_periodic --warm-runs 3
  python3 -m benchmarks.bench_periodic --warm-runs 3 --json
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from circuitopt.chopper import pmos_chopper_pac, pmos_chopper_pnoise, pmos_chopper_pss
from circuitopt.circuit_loader import load_circuit_json
from circuitopt.numba_kernels import NUMBA_AVAILABLE


ROOT = Path(__file__).resolve().parents[1]
SPEC = load_circuit_json(ROOT / "examples" / "afe_explore.json")
SIZES = dict(SPEC.sizes)
BIAS = dict(SPEC.bias)
NF = SPEC.nf

F_CHOP = 225.0
SWITCH_SIZE = (5000.0, 30.0)
SWITCH_NF = 1
EDGE_TIME = 20e-6
CORNER = "typical"
BAND = (0.05, 100.0)
FREQS = np.array([0.05, 0.2, 1.0, 5.0, 20.0, 100.0], dtype=float)

PSS_KWARGS = dict(
    switch_size=SWITCH_SIZE,
    switch_nf=SWITCH_NF,
    edge_time=EDGE_TIME,
    input_diff=0.0,
    charge_injection=False,
    tstab_periods=1,
    n_points=121,
    fallback_least_squares=False,
    corner=CORNER,
)
PAC_KWARGS = dict(
    time_domain=True,
    td_integration="gear2",
    td_n_period_samples=768,
    cache_linearization=True,
    cache_forcing=True,
)
PNOISE_KWARGS = dict(
    time_domain=True,
    max_sideband=32,
    n_period_samples=384,
    band=BAND,
    cache_linearization=True,
)


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
    return {
        "case": label,
        **_summarize_times(cold_ms, warm_ms),
        **summarize(warm_out),
    }


def _cache_counts(pss):
    return {
        "pac_td_cache_entries": len(pss.get("_pac_td_cache", {})),
        "pac_hb_cache_entries": len(pss.get("_pac_analytic_cache", {})),
        "pac_fd_cache_entries": len(pss.get("_pac_cache", {})),
        "pnoise_cache_entries": len(pss.get("_pnoise_cache", {})),
        "pnoise_adjoint_cache_entries": len(pss.get("_pnoise_adjoint_cache", {})),
    }


def run_pss():
    return pmos_chopper_pss(SIZES, BIAS, F_CHOP, nf=NF, **PSS_KWARGS)


def run_pac(pss):
    before = _cache_counts(pss)
    pac = pmos_chopper_pac(
        SIZES, BIAS, FREQS, F_CHOP, pss_result=pss, nf=NF, corner=CORNER,
        **PAC_KWARGS,
    )
    after = _cache_counts(pss)
    cache_hit = bool(
        pac.get("pac_state_cache_hit", False) or
        int(pac.get("pac_input_cache_hits", 0)) > 0 or
        before["pac_td_cache_entries"] > 0 or
        before["pac_hb_cache_entries"] > 0 or
        before["pac_fd_cache_entries"] > 0
    )
    pac["_bench_cache_hit"] = cache_hit
    pac["_bench_cache_before"] = before
    pac["_bench_cache_after"] = after
    return pac


def run_pnoise(pss, pac):
    before = _cache_counts(pss)
    pn = pmos_chopper_pnoise(
        SIZES, BIAS, FREQS, F_CHOP, pss_result=pss, nf=NF, corner=CORNER,
        pac_result=pac, **PNOISE_KWARGS,
    )
    after = _cache_counts(pss)
    cache_hit = bool(
        pn.get("pnoise_linearization_cache_hit", False) or
        pn.get("pnoise_hb_cache_hit", False) or
        int(pn.get("pnoise_adjoint_cache_hits", 0)) > 0 or
        before["pnoise_cache_entries"] > 0 or
        before["pnoise_adjoint_cache_entries"] > 0
    )
    pn["_bench_cache_hit"] = cache_hit
    pn["_bench_cache_before"] = before
    pn["_bench_cache_after"] = after
    return pn


def summarize_pss(pss):
    return {
        "converged": bool(pss.get("converged", False)),
        "pss_status": str(pss.get("pss_status", "")),
        "pss_period_runs": int(pss.get("shooting_period_runs", 0)),
        "shooting_iters": int(pss.get("shooting_iters", 0)),
        "shooting_jacobian_evals": int(pss.get("shooting_jacobian_evals", 0)),
        "shooting_jacobian_reuses": int(pss.get("shooting_jacobian_reuses", 0)),
        "residual_norm": float(pss.get("residual_norm", np.nan)),
        "nfail": int(pss.get("nfail", 0)),
        "orbit_points": int(len(pss.get("t", ()))),
        "numba_grid_solver": bool(pss.get("numba_grid_solver", False)),
        "gear2_python_retry_solver": bool(pss.get("gear2_python_retry_solver", False)),
        "adaptive": bool(pss.get("adaptive", False)),
        "adaptive_grid_frozen": bool(pss.get("adaptive_grid_frozen", False)),
    }


def summarize_pac(pac):
    return {
        "pac_method": str(pac.get("method", "")),
        "pac_cache_enabled": bool(pac.get("pac_cache_enabled", False)),
        "pac_cache_hit": bool(pac.get("_bench_cache_hit", False)),
        "pac_cache_before": pac.get("_bench_cache_before", {}),
        "pac_cache_after": pac.get("_bench_cache_after", {}),
        "pac_period_runs": int(pac.get("pac_period_runs", 0)),
        "pac_state_size": int(pac.get("pac_state_size", 0)),
        "pac_internal_gate1_states": int(pac.get("pac_internal_gate1_states", 0)),
        "pac_td_integration": str(pac.get("pac_td_integration", "")),
        "pac_td_samples": int(pac.get("pac_td_samples", 0)),
        "pac_td_boundary_mode": str(pac.get("pac_td_boundary_mode", "")),
        "pac_td_setup_time_s": float(pac.get("pac_td_setup_time_s", 0.0)),
        "pac_td_boundary_solve_time_s": float(
            pac.get("pac_td_boundary_solve_time_s", 0.0)),
        "gain0": float(pac["gains"][0]),
        "bw_Hz": float(pac.get("bw_Hz", np.nan)),
    }


def summarize_pnoise(pn):
    td_used = bool(pn.get("pnoise_time_domain_used", False))
    return {
        "pnoise_method": str(pn.get("method", "")),
        "pnoise_conversion": str(
            pn.get("pnoise_conversion", "time_domain" if td_used else "harmonic_balance")),
        "pnoise_time_domain_used": td_used,
        "pnoise_cache_enabled": bool(pn.get("pnoise_cache_enabled", False)),
        "pnoise_cache_hit": bool(pn.get("_bench_cache_hit", False)),
        "pnoise_cache_before": pn.get("_bench_cache_before", {}),
        "pnoise_cache_after": pn.get("_bench_cache_after", {}),
        "pnoise_linearization_cache_hit": bool(
            pn.get("pnoise_linearization_cache_hit", False)),
        "pnoise_hb_cache_hit": bool(pn.get("pnoise_hb_cache_hit", False)),
        "pnoise_adjoint_cache_hits": int(pn.get("pnoise_adjoint_cache_hits", 0)),
        "pnoise_hb_solver": str(pn.get("pnoise_hb_solver", "")),
        "pnoise_hb_size": int(pn.get("pnoise_hb_size", 0)),
        "pnoise_hb_solve_count": int(pn.get("pnoise_hb_solve_count", 0)),
        "pnoise_hb_dense_fallbacks": int(pn.get("pnoise_hb_dense_fallbacks", 0)),
        "pnoise_hb_iterative_fallbacks": int(
            pn.get("pnoise_hb_iterative_fallbacks", 0)),
        "pnoise_numba_hb_used": bool(pn.get("pnoise_numba_hb_used", False)),
        "pnoise_numba_fold_used": bool(pn.get("pnoise_numba_fold_used", False)),
        "pnoise_state_size": int(pn.get("pnoise_state_size", 0)),
        "pnoise_internal_gate1_states": int(
            pn.get("pnoise_internal_gate1_states", 0)),
        "irn_uV_band": float(pn.get("irn_uV_band", np.nan)),
        "out_uV_band": float(pn.get("out_uV_band", np.nan)),
        "pnoise_linearization_time_s": float(
            pn.get("pnoise_linearization_time_s", 0.0)),
        "pnoise_hb_assembly_time_s": float(pn.get("pnoise_hb_assembly_time_s", 0.0)),
        "pnoise_hb_solve_time_s": float(pn.get("pnoise_hb_solve_time_s", 0.0)),
        "pnoise_fold_time_s": float(pn.get("pnoise_fold_time_s", 0.0)),
    }


def run_benchmarks(warm_runs):
    results = []

    results.append(bench_case("pss", run_pss, summarize_pss, warm_runs))

    pss_for_pac, setup_pss_pac_ms = _time_call(run_pss)
    pac_case = bench_case(
        "pac",
        lambda: run_pac(pss_for_pac),
        summarize_pac,
        warm_runs,
    )
    pac_case["setup_pss_ms"] = float(setup_pss_pac_ms)
    pac_case["setup_pss_period_runs"] = int(
        pss_for_pac.get("shooting_period_runs", 0))
    results.append(pac_case)

    pss_for_pnoise, setup_pss_pnoise_ms = _time_call(run_pss)
    pac_for_pnoise, setup_pac_pnoise_ms = _time_call(lambda: run_pac(pss_for_pnoise))
    pnoise_case = bench_case(
        "pnoise",
        lambda: run_pnoise(pss_for_pnoise, pac_for_pnoise),
        summarize_pnoise,
        warm_runs,
    )
    pnoise_case["setup_pss_ms"] = float(setup_pss_pnoise_ms)
    pnoise_case["setup_pss_period_runs"] = int(
        pss_for_pnoise.get("shooting_period_runs", 0))
    pnoise_case["setup_pac_ms"] = float(setup_pac_pnoise_ms)
    pnoise_case["setup_pac_method"] = str(pac_for_pnoise.get("method", ""))
    results.append(pnoise_case)

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba_enabled": bool(NUMBA_AVAILABLE),
        "numba_env": os.environ.get("CIRCUIT_USE_NUMBA"),
        "circuit": "examples/afe_explore.json",
        "corner": CORNER,
        "f_chop_Hz": F_CHOP,
        "switch_W_um": SWITCH_SIZE[0],
        "switch_L_um": SWITCH_SIZE[1],
        "switch_nf": SWITCH_NF,
        "edge_time_us": EDGE_TIME * 1e6,
        "freqs_Hz": [float(x) for x in FREQS],
        "band_Hz": [float(BAND[0]), float(BAND[1])],
        "pss_n_points": int(PSS_KWARGS["n_points"]),
        "warm_runs": int(warm_runs),
        "results": results,
    }


def print_text(report):
    def fmt_ms(value):
        return "n/a" if value is None else f"{value:.3f}"

    print(f"python={report['python']} numpy={report['numpy']} "
          f"numba_enabled={report['numba_enabled']} warm_runs={report['warm_runs']}")
    print(f"chopper: f={report['f_chop_Hz']}Hz switch="
          f"{report['switch_W_um']:.0f}/{report['switch_L_um']:.0f} "
          f"edge={report['edge_time_us']:.1f}us corner={report['corner']} "
          f"freqs={len(report['freqs_Hz'])}")
    for item in report["results"]:
        print(f"{item['case']}: cold_ms={item['cold_ms']:.3f} "
              f"warm_median_ms={fmt_ms(item['warm_median_ms'])} "
              f"warm_min_ms={fmt_ms(item['warm_min_ms'])}")
        for key, value in item.items():
            if key in {"case", "cold_ms", "warm_runs_ms", "warm_median_ms", "warm_min_ms"}:
                continue
            if isinstance(value, float):
                print(f"  {key}={value:.6g}")
            else:
                print(f"  {key}={value}")


def main():
    parser = argparse.ArgumentParser(
        description="Periodic chopper PSS/PAC/PNoise performance benchmark")
    parser.add_argument("--warm-runs", type=int, default=3,
                        help="number of warm runs per case (default: 3)")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON")
    args = parser.parse_args()

    report = run_benchmarks(max(0, int(args.warm_runs)))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)


if __name__ == "__main__":
    main()
