#!/usr/bin/env python3
"""Recommend ``--workers`` for this machine's ADC/SAR workloads.

Worker counts that are right on one machine are wrong on another — the
numbers in the docs were measured on a 4P+6E Apple M4, where the compiled
SAR kernel saturates at all 10 cores (not the 4 performance cores) and a
16-trial Monte-Carlo is fastest at ``workers=16`` (one trial per task lets
work-stealing even out the P/E asymmetry) while ``workers=10`` is *slower*
than 8 (10 threads sharing 16 large trials leaves a 2-vs-1 tail). This tool
turns those measured rules into per-machine advice:

    python tools/workers.py                  # topology + recommendations
    python tools/workers.py --mc-trials 16   # advice for a specific MC run
    python tools/workers.py --json           # machine-readable
    python tools/workers.py --calibrate      # measure the real knee (needs
                                             # FreePDK45 cards; ~10 s)

Detection is stdlib-only. ``--calibrate`` actually runs the compiled SAR
ramp (codes-only) across a worker grid, picks the smallest count within 5%
of the fastest, and records everything to
``results/workers_calibration.json`` — trust it over the static rules when
the two disagree.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Workloads whose parallel level is the independent conversion (the compiled
# kernel's Rayon chunks): ramp sweeps, sine tests, explore candidates' sweeps,
# and transition-bisection probe rounds. Measured on 4P+6E: throughput keeps
# rising past the P-core count and saturates at the full logical count.
_CONVERSION_MODES = ("sweep/ramp", "sine", "explore", "transitions")


def _sysctl_int(name: str) -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True,
                             text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def _cpu_list_count(path: str) -> int | None:
    """Count CPUs in a sysfs list like ``0-7,16-23`` (Intel hybrid Linux)."""
    try:
        text = Path(path).read_text().strip()
    except OSError:
        return None
    if not text:
        return None
    count = 0
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            count += int(hi) - int(lo) + 1
        else:
            count += 1
    return count


def detect_topology() -> dict:
    """Logical-core topology: total plus the P/E split where the OS exposes it.

    macOS Apple Silicon reports the split through ``hw.perflevel*``; Intel
    hybrid Linux through ``/sys/devices/cpu_core|cpu_atom``. Anywhere else the
    machine is treated as uniform (``e_cores = 0``) — the recommendations only
    need ``total`` to be right.
    """
    total = os.cpu_count() or 1
    p_cores, e_cores, source = total, 0, "uniform (os.cpu_count)"
    system = platform.system()
    if system == "Darwin":
        p = _sysctl_int("hw.perflevel0.logicalcpu")
        e = _sysctl_int("hw.perflevel1.logicalcpu")
        n = _sysctl_int("hw.ncpu")
        if n:
            total = n
        if p and e and p + e == total:
            p_cores, e_cores, source = p, e, "sysctl hw.perflevel*"
    elif system == "Linux":
        p = _cpu_list_count("/sys/devices/cpu_core/cpus")
        e = _cpu_list_count("/sys/devices/cpu_atom/cpus")
        if p and e and p + e == total:
            p_cores, e_cores, source = p, e, "sysfs cpu_core/cpu_atom"
    return {
        "system": system,
        "machine": platform.machine(),
        "total": total,
        "p_cores": p_cores,
        "e_cores": e_cores,
        "hybrid": e_cores > 0,
        "source": source,
    }


def recommend(topology: dict, *, mc_trials: int | None = None) -> dict:
    """Per-workload worker counts from the measured scheduling rules.

    - Conversion-parallel modes saturate at the full logical count (E cores
      carry real load; measured w8 already beat ideal-4P scaling on 4P+6E).
    - MC parallelises whole trials, which are large: with few trials the
      imbalance of splitting them across a non-divisor worker count costs
      more than oversubscription does, so run one trial per task
      (``workers = trials``); with many trials the imbalance amortises and
      the core count wins.
    - Signoff PVT points behave like MC trials but campaigns are wide
      (45 points), so the core count is right.
    - A single conversion has nothing to parallelise.
    """
    total = topology["total"]
    out = {mode: total for mode in _CONVERSION_MODES}
    out["signoff"] = total
    out["single_conversion"] = 1
    if mc_trials is not None:
        if mc_trials < 1:
            raise ValueError("mc trials must be at least 1")
        out["mc"] = mc_trials if mc_trials <= 4 * total else total
    return out


def pick_knee(timings: dict[int, float], *, tolerance: float = 0.05) -> int:
    """The smallest worker count within ``tolerance`` of the fastest run.

    Preferring fewer threads at equal speed keeps the machine responsive and
    avoids pretending noise-level gains are real.
    """
    if not timings:
        raise ValueError("timings must not be empty")
    best = min(timings.values())
    return min(w for w, t in timings.items() if t <= best * (1.0 + tolerance))


def calibrate(circuit: str, *, repeats: int = 1, grid=None) -> dict:
    """Measure the compiled SAR ramp (codes-only) across a worker grid.

    Imports circuitopt lazily so plain detection never needs the package or
    the PDK. Each grid point runs ``repeats`` times (min taken) after one
    warm-up at the widest count; the whole run is ~10 s on a laptop.
    """
    import numpy as np

    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.sar import run_sar_sweep

    topology = detect_topology()
    total = topology["total"]
    if grid is None:
        grid = sorted({1, 2, 4, topology["p_cores"], total,
                       max(total + 2, int(total * 1.6))})
    spec = load_circuit_json(circuit)
    if spec.adc is None:
        raise SystemExit(f"{circuit} has no 'adc' block")
    levels = 1 << int(spec.adc["n_bits"])
    vin = (np.arange(levels) + 0.5) / levels * float(spec.adc["vref"])

    run_sar_sweep(spec, vin, workers=max(grid),
                  include_transients=False)          # warm-up
    timings: dict[int, float] = {}
    for w in grid:
        best = float("inf")
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            run_sar_sweep(spec, vin, workers=w, include_transients=False)
            best = min(best, time.perf_counter() - t0)
        timings[w] = best
    knee = pick_knee(timings)
    return {
        "circuit": circuit,
        "topology": topology,
        "timings_s": {str(w): round(t, 4) for w, t in sorted(timings.items())},
        "recommended_conversion_workers": knee,
        "speedup_vs_serial": round(timings[1] / timings[knee], 2),
    }


def _print_human(topology: dict, rec: dict) -> None:
    hybrid = (f"{topology['p_cores']}P + {topology['e_cores']}E"
              if topology["hybrid"] else "uniform")
    print(f"machine : {topology['system']}/{topology['machine']}  "
          f"{topology['total']} logical cores ({hybrid}; {topology['source']})")
    print("recommended --workers:")
    for mode in (*_CONVERSION_MODES, "signoff", "single_conversion"):
        print(f"  {mode:<18} {rec[mode]}")
    if "mc" in rec:
        print(f"  {'mc':<18} {rec['mc']}")
    else:
        total = topology["total"]
        print(f"  {'mc':<18} trials themselves when trials <= {4 * total} "
              f"(one trial per task), else {total}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Recommend --workers for this machine's SAR/ADC workloads")
    parser.add_argument("--mc-trials", type=int, default=None, metavar="N",
                        help="also print the worker count for an N-trial "
                             "mismatch Monte-Carlo")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    parser.add_argument("--calibrate", action="store_true",
                        help="measure the real saturation knee with the "
                             "compiled SAR ramp (needs FreePDK45 cards)")
    parser.add_argument("--circuit",
                        default=str(ROOT / "examples" / "freepdk45_sar6.json"),
                        help="circuit for --calibrate "
                             "(default: examples/freepdk45_sar6.json)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="timed runs per calibration grid point (min "
                             "taken; default 1)")
    args = parser.parse_args(argv)

    topology = detect_topology()
    rec = recommend(topology, mc_trials=args.mc_trials)
    if args.calibrate:
        result = calibrate(args.circuit, repeats=args.repeats)
        rec = recommend(result["topology"], mc_trials=args.mc_trials)
        for mode in _CONVERSION_MODES:
            rec[mode] = result["recommended_conversion_workers"]
        out_path = ROOT / "results" / "workers_calibration.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**result, "recommended": rec}
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            _print_human(result["topology"], rec)
            print("calibrated timings [s]:",
                  " ".join(f"w{w}={t}" for w, t in result["timings_s"].items()))
            print(f"speedup vs serial: {result['speedup_vs_serial']}x")
            print(f"wrote {out_path.relative_to(ROOT)}")
        return 0
    if args.json:
        print(json.dumps({"topology": topology, "recommended": rec}, indent=2))
    else:
        _print_human(topology, rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
