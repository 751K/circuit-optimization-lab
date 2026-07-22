"""Lightweight profiling of representative solver workloads.

Run: python tools/profile_hotspots.py

Uses cProfile for call-count + cumulative time. Time spent inside the compiled
rust core shows up on the PyO3 entry points (e.g. solve_adaptive_gear2).
"""
from __future__ import annotations
import cProfile, pstats, io, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from circuitopt.circuit_loader import load_circuit_json


def profile(label: str, fn, *, sort_by: str = "cumtime", top: int = 25):
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    result = fn()
    wall = time.perf_counter() - t0
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
    ps.print_stats(top)

    # Also print by call count
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("ncalls")
    ps2.print_stats(top)

    print(f"\n{'='*70}")
    print(f"  {label}  (wall: {wall:.3f}s)  —  sort by: {sort_by}")
    print(f"{'='*70}")
    print(s.getvalue())
    print("\n─── by call count ───")
    print(s2.getvalue())
    return result


def main():
    spec = load_circuit_json("examples/afe_explore.json")
    freqs = np.logspace(-2, 4, 121)
    t = np.linspace(0, 4e-3, 400)
    vip = np.where(t >= 0.5e-3, 30.65 + 0.5e-3, 30.65)
    vin = np.where(t >= 0.5e-3, 30.65 - 0.5e-3, 30.65)

    # ── 1. DC + AC + Noise (warm) ──
    from circuitopt.ac_solver import ac_solve
    from circuitopt.noise_solver import noise_analysis, band_rms

    def _ac_noise():
        ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
        noise = noise_analysis(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
        return band_rms(freqs, noise["irn_psd"], 0.05, 100.0)

    # warm up once (JIT / imports)
    _ac_noise()
    profile("DC+AC+Noise 121pt (warm)", _ac_noise)

    # ── 2. Transient 400 steps (BE, default) ──
    from circuitopt.transient_solver import transient

    def _tran_be():
        return transient(spec.sizes, spec.bias, t, vip, vin, topo=spec.topology, nf=spec.nf)

    _tran_be()
    profile("Transient BE 400 steps (warm)", _tran_be, top=30)

    # ── 3. Transient gear2 ──
    def _tran_gear2():
        return transient(spec.sizes, spec.bias, t, vip, vin, topo=spec.topology, nf=spec.nf,
                         integration_method="gear2")

    _tran_gear2()
    profile("Transient gear2 400 steps (warm)", _tran_gear2, top=30)

    # ── 4. Chopper ideal LPTV ──
    from circuitopt.chopper import chopper_analysis

    def _chop_ideal():
        return chopper_analysis(spec.sizes, spec.bias, freqs, f_chop=225.0,
                                topo=spec.topology, nf=spec.nf, max_harmonic=31)

    _chop_ideal()
    profile("Chopper ideal LPTV 121pt (warm)", _chop_ideal)

    # ── 5. Explore batch (50 candidates, AC only) ──
    from circuitopt.explore import explore

    from circuitopt.explore import ExploreConfig, Variable

    cfg = ExploreConfig(
        variables=[
            Variable("W6", 1000, 4000),
            Variable("L6", 50, 150),
            Variable("VB", 8.0, 12.0),
        ],
        constraints={"gain_dB": {"min": 15.0}},
        objectives={"area": "min"},
        band=(0.05, 100.0),
        freqs=freqs,
    )

    def _explore():
        return explore(spec.topology, spec.sizes, spec.bias, spec.nf, cfg, n=20, seed=1)

    _explore()
    profile("Explore 50 candidates (warm)", _explore)

    # ── 6. Corner + MC summary ──
    from circuitopt.corners import corner_table, mismatch_mc

    def _corners():
        return corner_table(spec.sizes, spec.bias, nf=spec.nf, topo=spec.topology, freqs=freqs)

    _corners()
    profile("Corners typ/slow/fast (warm)", _corners)

    def _mc():
        return mismatch_mc(spec.sizes, spec.bias, nf=spec.nf, topo=spec.topology, freqs=freqs,
                           n=50, base="typical", seed=1)

    _mc()
    profile("Mismatch MC n=50 (warm)", _mc)

    print("\nDone.")
    print("  All paths above ran through the compiled rust core (circuitopt_core);")
    print("  see docs/environment_performance.md for the v1.x/v2.0.0 baselines.")


if __name__ == "__main__":
    main()
