"""Batch sweep performance benchmark.

Measures throughput of AC and AC+noise evaluation across many design candidates,
simulating the workload of the explore layer. Uses random perturbations around
the locked-design AFE sizes.

Workloads:
  - ac_only:   AC solve for N candidates (fast pre-filter, no noise)
  - ac_noise:  AC + noise for N candidates (full evaluate)

Reports per-candidate median/mean and overall candidates-per-second throughput.

Usage:
  python3 -m benchmarks.bench_sweep --warm-runs 3
  python3 -m benchmarks.bench_sweep --n-candidates 100 --warm-runs 3
  CIRCUIT_USE_NUMBA=0 python3 -m benchmarks.bench_sweep --warm-runs 3 --json
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np

from circuitopt.ac_solver import ac_solve
from circuitopt.circuit_loader import load_circuit_json
from circuitopt.noise_solver import noise_analysis
from circuitopt.numba_kernels import NUMBA_AVAILABLE

# ── fixed AFE design (locked, loaded from the canonical JSON) ─────────
_ROOT = Path(__file__).resolve().parents[1]
_SPEC = load_circuit_json(_ROOT / "examples" / "afe_explore.json")
BASE_TOPO = _SPEC.topology
BASE_SIZES = dict(_SPEC.sizes)
BASE_BIAS = dict(_SPEC.bias)
BASE_NF = _SPEC.nf
BASE_BINDING = _SPEC.binding()
BENCH_CIRCUIT = str(_ROOT / "examples" / "afe_explore.json")

# ── sweep parameters ───────────────────────────────────────────────────
NFREQ = 61                     # frequency points per candidate
BAND = (0.05, 100.0)           # IRN integration band (Hz)
PERTURB = 0.20                 # ±20% uniform random perturbation on W, L
DEFAULT_N = 50                 # default candidate count

# locked-design keys that must stay symmetric (same perturbation applied
# to both devices in each matched pair)
PAIRED_KEYS = [
    ("M7", "M8"),
    ("M9", "M10"),
    ("M12", "M13"),
    ("M14", "M15"),
]


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


def _make_candidates(n, rng):
    """Generate N size dictionaries by perturbing BASE_SIZES.

    Each W and L is multiplied by a uniform random factor in [1-PERTURB, 1+PERTURB].
    Paired devices (M7=M8, M9=M10, M12=M13, M14=M15) share the same perturbation
    to preserve symmetry — required for the AFE DC solver to stay on the physical
    branch.
    """
    paired_map = {}
    for a, b in PAIRED_KEYS:
        paired_map[a] = a
        paired_map[b] = a

    groups = {}
    for key in sorted(BASE_SIZES):
        group = paired_map.get(key)
        if group is None and BASE_BINDING.model_types:
            # Silicon examples express matched pairs as equal model/geometry
            # instances (M1/M2, M3/M4). Perturb those together so the benchmark
            # measures solver throughput instead of intentional pair mismatch.
            group = (BASE_BINDING.model_types.get(key), BASE_SIZES[key])
        groups.setdefault(group if group is not None else key, []).append(key)

    candidates = []
    for _ in range(n):
        sizes = {}
        for members in groups.values():
            f = rng.uniform(1 - PERTURB, 1 + PERTURB)
            for key in members:
                W0, L0 = BASE_SIZES[key]
                sizes[key] = (W0 * f, L0 * f)
        candidates.append(sizes)
    return candidates


def bench_case(name, fn, summarize, warm_runs):
    cold_out, cold_ms = _time_call(fn)
    warm_ms = []
    warm_out = cold_out
    for _ in range(warm_runs):
        warm_out, dt = _time_call(fn)
        warm_ms.append(dt)
    data = {"case": name, **_summarize_times(cold_ms, warm_ms), **summarize(warm_out)}
    return data


def run_benchmarks(warm_runs, n, seed=42, workers=1):
    if workers is None or workers < 1:
        raise ValueError("workers must be a positive integer")
    rng = np.random.default_rng(seed)
    candidates = _make_candidates(n, rng)
    freqs = np.logspace(np.log10(0.1), np.log10(10e3), NFREQ)
    results = []

    # ── ac_only: N × AC solve ────────────────────────────────────────
    def evaluate_ac(sizes):
        return ac_solve(sizes, BASE_BIAS, freqs, topo=BASE_TOPO, nf=BASE_NF,
                        binding=BASE_BINDING) is not None

    def run_batch(function):
        if workers == 1:
            return [function(sizes) for sizes in candidates]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(function, candidates))

    def run_ac_only():
        t0 = time.perf_counter()
        outcomes = run_batch(evaluate_ac)
        elapsed = (time.perf_counter() - t0) * 1e3
        return {"elapsed_ms": elapsed, "n_ok": sum(outcomes),
                "n_fail": len(outcomes) - sum(outcomes)}

    results.append(bench_case(
        f"ac_only_n{n}",
        run_ac_only,
        lambda r: {
            "elapsed_ms": float(r["elapsed_ms"]),
            "n_ok": int(r["n_ok"]),
            "n_fail": int(r["n_fail"]),
            "ms_per_candidate": float(r["elapsed_ms"] / n),
            "candidates_per_s": float(n / r["elapsed_ms"] * 1000),
        },
        warm_runs,
    ))

    # ── ac_noise: N × (AC + noise) ───────────────────────────────────
    def evaluate_ac_noise(sizes):
        ac = ac_solve(sizes, BASE_BIAS, freqs, topo=BASE_TOPO, nf=BASE_NF,
                      binding=BASE_BINDING)
        if ac is None:
            return False
        noise = noise_analysis(sizes, BASE_BIAS, freqs, topo=BASE_TOPO,
                               nf=BASE_NF, x0_guess=ac["dc_op"],
                               binding=BASE_BINDING)
        return noise is not None

    def run_ac_noise():
        t0 = time.perf_counter()
        outcomes = run_batch(evaluate_ac_noise)
        elapsed = (time.perf_counter() - t0) * 1e3
        return {"elapsed_ms": elapsed, "n_ok": sum(outcomes),
                "n_fail": len(outcomes) - sum(outcomes)}

    results.append(bench_case(
        f"ac_noise_n{n}",
        run_ac_noise,
        lambda r: {
            "elapsed_ms": float(r["elapsed_ms"]),
            "n_ok": int(r["n_ok"]),
            "n_fail": int(r["n_fail"]),
            "ms_per_candidate": float(r["elapsed_ms"] / n),
            "candidates_per_s": float(n / r["elapsed_ms"] * 1000),
        },
        warm_runs,
    ))

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba_enabled": bool(NUMBA_AVAILABLE),
        "numba_env": os.environ.get("CIRCUIT_USE_NUMBA"),
        "n_candidates": n,
        "perturb": PERTURB,
        "nfreq": NFREQ,
        "warm_runs": warm_runs,
        "workers": workers,
        "circuit": BENCH_CIRCUIT,
        "results": results,
    }


def print_text(report):
    def fmt_ms(value):
        return "n/a" if value is None else f"{value:.3f}"

    print(f"python={report['python']} numpy={report['numpy']} "
          f"numba_enabled={report['numba_enabled']} "
          f"n={report['n_candidates']} perturb={report['perturb']} "
          f"nfreq={report['nfreq']} workers={report['workers']} "
          f"warm_runs={report['warm_runs']}")
    print(f"circuit={report['circuit']}")
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
        description="Batch sweep performance benchmark")
    parser.add_argument("--warm-runs", type=int, default=3,
                        help="number of warm runs per case (default: 3)")
    parser.add_argument("--n-candidates", type=int, default=DEFAULT_N,
                        help=f"number of candidates per sweep (default: {DEFAULT_N})")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible candidates")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel candidate workers (default: 1)")
    parser.add_argument("--circuit", default=None,
                        help="circuit JSON to perturb (default: examples/afe_explore.json)")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON")
    args = parser.parse_args()

    if args.circuit is not None:
        global BASE_TOPO, BASE_SIZES, BASE_BIAS, BASE_NF, BASE_BINDING, BENCH_CIRCUIT
        spec = load_circuit_json(args.circuit)
        BASE_TOPO = spec.topology
        BASE_SIZES = dict(spec.sizes)
        BASE_BIAS = dict(spec.bias)
        BASE_NF = spec.nf
        BASE_BINDING = spec.binding()
        BENCH_CIRCUIT = str(Path(args.circuit).resolve())

    report = run_benchmarks(args.warm_runs, args.n_candidates, seed=args.seed,
                            workers=args.workers)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)


if __name__ == "__main__":
    main()
