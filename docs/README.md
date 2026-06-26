# Circuit Optimization Flow

[English](README.md) | [中文说明](README_zh.md)

## Overview

Local Python solvers for analog circuit design-space exploration, calibrated against
Cadence/Spectre. The first use case is an **AT4000TG PMOS thin-film transistor
ECG AFE** (analog front-end amplifier with chopper).

What you can do with this:
- **DC/AC/Noise/Transient** — standard circuit analysis without a simulator license.
- **PSS / PAC / PNoise** — periodic steady-state, periodic AC, and periodic noise
  for chopper amplifiers, matched to Spectre RF analyses.
- **Design exploration** — sweep device sizes and bias voltages, filter by
  constraints (gain, BW, noise, power, area), find Pareto-optimal designs.
- **Corners & mismatch** — process corners, per-device mismatch Monte Carlo, latch
  screening.

For solver internals, see [Core Solver Overview](core_overview.md).

---

## Quick Start

```bash
# 1. Install
python3 -m pip install -r requirements.txt

# 2. Optional: Numba acceleration (10-50× faster transient)
python3 -m pip install -r requirements-numba.txt

# 3. Run your first circuit — one command
python3 -m core examples/periodic_rc.json

# 4. Verify — run the AFE benchmark
python3 -m benchmarks.bench_afe --warm-runs 1 --skip-noise
```

The first command above runs AC, noise, PSS, PAC, and PNoise on a passive RC
lowpass and prints a summary. No Python scripting needed. If it prints numbers,
everything works. From there, swap in any circuit JSON or use
`-a ac,noise` to pick specific analyses.

### CLI Reference

`python -m core` uses subcommands (backward compatible — bare `circuit.json` defaults to `run`):

```bash
# ── Analysis dispatch (default: "run") ──
python -m core examples/periodic_rc.json                          # all configured analyses
python -m core examples/periodic_rc.json -a ac,noise,pss          # specific analyses
python -m core run examples/periodic_rc.json -a ac,noise          # explicit subcommand

# ── Design-space exploration ──
python -m core examples/afe_explore.json --explore -n 500         # --explore flag (legacy)
python -m core explore examples/afe_explore.json -n 500 --seed 1  # subcommand

# ── Process corners sweep ──
python -m core corners examples/afe_explore.json                  # typ/slow/fast
python -m core corners examples/afe_explore.json --freqs-num 61

# ── Mismatch Monte Carlo ──
python -m core mc examples/afe_explore.json -n 200 --seed 1      # typical corner
python -m core mc examples/afe_explore.json --corner slow -n 500

# ── Chopper analysis ──
python -m core chopper examples/afe_explore.json --level ideal    # square-wave LPTV
python -m core chopper examples/afe_explore.json --level pmos     # static-phase PMOS
python -m core chopper examples/afe_explore.json --level lptv     # PMOS sideband fold
python -m core chopper examples/afe_explore.json --level pss      # shooting PSS
python -m core chopper examples/afe_explore.json --level pnoise   # PSS→PAC→PNoise
python -m core chopper examples/afe_explore.json --level transient

# Common options for all subcommands:
#   --noise-band LO HI  IRN integration band (default: 0.05 100.0)
#   -o PATH             write results to file
#   --no-numba          disable Numba acceleration
#   --quiet             suppress progress output
```

### How the code is organized

Before diving into the workflows, a one-minute map of the concepts:

| Concept | What it is | Where it lives |
|---------|-----------|----------------|
| **Topology** | Circuit structure — which nodes exist, how devices connect, where inputs/outputs are | `core/topology.py`, or auto-generated from JSON |
| **Sizes** | `{device: (W_µm, L_µm)}` — transistor dimensions | JSON `sizes` field |
| **NF** | Number of fingers (parallel transistor multiplier; increases current) | JSON `nf` field, or per-device in `devices[].NF` |
| **Bias** | `{node: voltage}` — DC operating voltages at rail nodes | JSON `bias` field |
| **Solver** | A function that takes topology + sizes + bias → results (gain, noise, waveforms, …) | `core/ac_solver.py`, `core/transient_solver.py`, etc. |
| **Device Model** | Abstract interface (`TransistorModel`) — solvers call the interface, not a concrete model; swap models via factory | `core/device_model.py`, `core/pmos_tft_model.py` |

Any solver call follows the same pattern:

```python
result = solver(sizes, bias, ..., topo=topology, nf=nf)
```

The JSON file bundles all the inputs together; `load_circuit_json()` unpacks them
into a `CircuitSpec` with `.topology`, `.sizes`, `.bias`, `.nf`, and optionally
`.explore`.

---

## Common Workflows

All examples below are copy-paste ready. They use the locked AFE design from
`examples/afe_explore.json`.

### 1. Load a Circuit and Run DC / AC / Noise

```python
import numpy as np
from core.circuit_loader import load_circuit_json
from core.ac_solver import ac_solve
from core.noise_solver import noise_analysis, band_rms

# Load from JSON — no hard-coded node names in solver code
spec = load_circuit_json("examples/afe_explore.json")
freqs = np.logspace(-2, 4, 121)   # 0.01 Hz to 10 kHz

# DC operating point + AC gain/bandwidth
ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
print(f"Gain: {ac['Av_dc_dB']:.2f} dB,  BW: {ac['bw_Hz']:.1f} Hz")
# → Gain: 22.89 dB,  BW: 549.3 Hz

# Noise analysis (thermal + flicker)
noise = noise_analysis(spec.sizes, spec.bias, freqs,
                       topo=spec.topology, nf=spec.nf)
irn_uv = band_rms(freqs, noise["irn_psd"], 0.05, 100.0) * 1e6
print(f"IRN (0.05–100 Hz): {irn_uv:.2f} µVrms")
# → IRN (0.05–100 Hz): 36.97 µVrms
```

### 2. Run a Transient Simulation

```python
from core.transient_solver import transient

# 4 ms simulation, 0.5 mV step at t=0.5 ms
t = np.linspace(0, 4e-3, 400)
vip = np.where(t >= 0.5e-3, 30.65 + 0.5e-3, 30.65)
vin = np.where(t >= 0.5e-3, 30.65 - 0.5e-3, 30.65)

# Default: backward Euler (BE) — robust, well-validated
tran = transient(spec.sizes, spec.bias, t, vip, vin,
                 topo=spec.topology, nf=spec.nf)
print(f"Transient steps: {len(t)},  nfail: {tran['nfail']}")
# → Transient steps: 400,  nfail: 0

# Optional: gear2/BDF2 — second-order, stiffly stable (chopper PSS/PAC/PNoise default)
tran_gear2 = transient(spec.sizes, spec.bias, t, vip, vin,
                       topo=spec.topology, nf=spec.nf,
                       integration_method="gear2")
```
Gear2 (variable-step BDF2) reduces PAC baseband error from BE's ~−2.5% to <1%
across all corners. On stiff circuits (e.g. chopper), `integration_method="gear2"`
stays in the Numba gear2 grid when `max_step` / `max_retry_subdivisions` request
subdivision/retry; the grid maintains rolling two-step history through accepted
substeps. PSS/PAC/PNoise pipelines default to gear2 for accuracy; bare
`transient()` defaults to BE.

### 3. Chopper Analysis (Three Levels)

#### Level 1 — Ideal LPTV (fast, square-wave model)

```python
from core.chopper import chopper_analysis

chop_ideal = chopper_analysis(
    spec.sizes, spec.bias, freqs, f_chop=225.0,
    topo=spec.topology, nf=spec.nf, max_harmonic=31,
    band=(0.05, 100.0))
print(f"Ideal chop: {chop_ideal['peak_dB']:.2f} dB,  "
      f"IRN: {chop_ideal['irn_uV_band']:.2f} µVrms")
```

#### Level 2 — PMOS Switch (static phases, no PSS needed)

```python
from core.chopper import pmos_chopper_analysis

pmos = pmos_chopper_analysis(
    spec.sizes, spec.bias, freqs,
    switch_size=(20000, 80), band=(0.05, 100.0))
print(f"PMOS static chop: {pmos['peak_dB']:.2f} dB,  "
      f"IRN: {pmos['irn_uV_band']:.2f} µVrms")
```

#### Level 3 — Full PSS / PAC / PNoise (first-principles, matches Spectre)

```python
from core.chopper import (pmos_chopper_pss, pmos_chopper_pac,
                           pmos_chopper_pnoise)

# Step 1: PSS — find the periodic steady-state orbit
pss = pmos_chopper_pss(
    spec.sizes, spec.bias, f_chop=225.0,
    switch_size=(5000, 30), edge_time=20e-6,
    tstab_periods=2, n_points=121)
print(f"PSS converged: {pss['converged']},  "
      f"residual: {pss['residual_norm']:.2e}")

# Step 2: PAC — periodic AC gain on the PSS orbit
pac = pmos_chopper_pac(
    spec.sizes, spec.bias, freqs, f_chop=225.0,
    pss_result=pss)
print(f"PAC gain: {pac['Av_dc_dB']:.2f} dB,  BW: {pac['bw_Hz']:.1f} Hz")

# Step 3: PNoise — periodic noise (harmonic balance, no calibration constants)
pnoise = pmos_chopper_pnoise(
    spec.sizes, spec.bias, freqs, f_chop=225.0,
    pss_result=pss, pac_result=pac, max_sideband=10,
    band=(0.05, 100.0))
print(f"PNoise IRN: {pnoise['irn_uV_band']:.2f} µVrms")
```

The PSS→PAC→PNoise pipeline is the local equivalent of Cadence Spectre
`pss` + `pac` + `pnoise`. PAC has two first-class kernels:

- Default: analytic-adjoint harmonic balance (`method="pss_analytic_adjoint"`).
  It is the most general path and supports bordered MNA cases.
- Fast path: time-domain Floquet PAC (`time_domain=True`,
  `method="pss_time_domain"`). It builds the one-period monodromy once and then
  solves a small quasi-periodic boundary system per frequency, avoiding the large
  `(2K+1)n` HB matrix. Unsupported topologies fall back to HB when
  `analytic=True`. This path remains opt-in: the 2026-06-26 three-corner check
  against the stored Cadence references gives typical −0.44%, fast −0.27%, but
  slow −1.89% baseband gain error, so it is not yet the default.

Set `analytic=False` only for the original finite-difference shooting path
(accurate but costs `n_state+2` transient runs per frequency). PNoise uses
harmonic balance on the PSS orbit — it's a first-principles LPTV noise solve with
no calibration fudge factors.
For the D3 `chop_tb_d3` slow-corner Spectre reference at `f_chop=200 Hz`,
the default HB PAC gain and PNoise IRN are within 1% when run with the matching
PNoise `maxsideband=10` and dec=10 noise grid. Keep the time-domain PAC path
under the same Cadence regression before promoting it to a default for new
chopper cases.
`pmos_chopper_pac` / `pmos_chopper_pnoise` are chopper compatibility wrappers;
generic periodic topologies can call `core.pac_solver.pac_solve` and
`core.pnoise_solver.pnoise_solve` directly using the orbit returned by
`pss_solve` plus an `input_drive` mapping.

**JSON dispatch** — when the circuit JSON has `periodic` and `analyses` blocks,
run everything with one call:

```python
from core.analysis_dispatch import run_analysis_suite
from core.circuit_loader import load_circuit_json

spec = load_circuit_json("examples/periodic_rc.json")
results = run_analysis_suite(spec)
# results["pss"], results["pac"], results["pnoise"] — all ready
```

JSON dispatch supports the same opt-in PAC switch:
`"analyses": {"pac": {"time_domain": true, "td_integration": "gear2"}}`.

### 4. Design-Space Exploration / Optimization

```python
from core.explore import explore
from core.circuit_loader import load_circuit_json

spec = load_circuit_json("examples/afe_explore.json")

# The JSON's "explore" block defines variables, constraints, and objectives.
# explore() samples candidates, evaluates each through the solvers,
# filters by constraints, and returns the Pareto front.
result = explore(spec.topology, spec.sizes, spec.bias, spec.nf,
                 spec.explore, n=500, method="lhs", seed=42)

print(f"Candidates: {result['n_total']},  "
      f"Feasible: {result['n_feasible']},  "
      f"Pareto-optimal: {len(result['pareto'])}")
# → Candidates: 500,  Feasible: 87,  Pareto-optimal: 12
```

Or from the command line:

```bash
python -m core.explore examples/afe_explore.json --n 500 --seed 42
```

Results are exported as CSV and JSONL. The explore config in the JSON file
specifies which variables to sweep (device W/L, bias voltages), what constraints
to enforce (gain > X, IRN < Y, etc.), and which objectives to optimize.

### 5. Process Corners & Mismatch

```python
from core.corners import CORNERS, corner_table, mismatch_mc, latch_screen
import numpy as np

# Corner sweep — one design at typ/slow/fast
table = corner_table(spec.sizes, spec.bias, np.logspace(-2, 4, 121),
                     topo=spec.topology, nf=spec.nf)
for row in table:
    print(f"{row['corner']:>6s}:  gain={row['gain_peak_dB']:.2f} dB,  "
          f"BW={row['bw_Hz']:.0f} Hz,  IRN={row['irn_uV']:.2f} µVrms")
# → typical:  gain=22.89 dB,  BW=549 Hz,  IRN=36.97 µVrms
# →   slow:  gain=20.81 dB,  BW=328 Hz,  IRN=45.72 µVrms
# →   fast:  gain=24.41 dB,  BW=846 Hz,  IRN=28.40 µVrms

# Quick latch screen (deterministic, fast enough for inner-loop use)
rng = np.random.default_rng(0)
latch = latch_screen(spec.sizes, spec.bias, topo=spec.topology,
                     nf=spec.nf, rng=rng, k_sigma=3.0)
print(f"Latch dV: {latch['latch_dV']*1e3:.2f} mV  "
      f"({'LATCHED' if latch['latched'] else 'ok'})")

# Full mismatch Monte Carlo (slower, for final verification)
mc = mismatch_mc(spec.sizes, spec.bias, np.logspace(-2, 4, 61),
                 topo=spec.topology, nf=spec.nf, n=200,
                 corner=CORNERS["typical"], seed=1)
print(f"Latch rate: {mc['latch_rate']*100:.1f}%,  "
      f"IRN: {mc['irn_mean']:.2f} ± {mc['irn_std']:.2f} µVrms")
```

---

## JSON Circuit Format

New circuits are defined in JSON — no solver source edits needed. See
[JSON Circuit Description](json_circuit_format.md) for the full field reference.

Quick example (`examples/single_stage.json`):

```json
{
  "solved": ["OUT"],
  "rails": {"VDD": 40.0, "GND": 0.0},
  "devices": [
    {"name": "M1", "drain": "OUT", "gate": "IN", "source": "VDD",
     "W": 2000, "L": 80, "NF": 1}
  ],
  "bias": {"VDD": 40.0, "VIN": 30.0, "VB": 10.0},
  "outputs": ["OUT"],
  "input_drives": {"IN": 1.0},
  "load_caps": {"OUT": 1e-12}
}
```

Key top-level fields:

| Field | Required | Purpose |
|-------|----------|---------|
| `solved` | yes | Nodes whose voltages the solver must find |
| `rails` | yes | Fixed-voltage nodes: `{"VDD": 40.0, "GND": 0.0, ...}` |
| `devices` | yes | PMOS transistors; passive circuits may use an empty array `[]` |
| `bias` | yes | DC voltage at every rail node: `{"VDD": 40.0, "VIN": 30.0, ...}` |
| `outputs` | yes | Which node(s) to measure gain/noise at |
| `input_drives` | — | Where to inject the AC small-signal stimulus for gain calculation |
| `load_caps` | — | Load capacitance per output node (F): `{"OUT": 1e-12}` |
| `resistors` | — | `[name, node_a, node_b, R_ohm]` |
| `capacitors` | — | `[name, node_a, node_b, C_farad]` |
| `current_sources` | — | Ideal DC current sources: `[name, nplus, nminus, I_amp]` |
| `vccs` | — | Voltage-controlled current sources: `[name, p, q, ctrl_p, ctrl_n, gm]` |
| `vsources` | — | Ideal voltage sources (true MNA): `[name, p, q, value]` — constant EMF or waveform key |
| `nf` | — | Global NF (fingers) applied to all devices; overridden by per-device `NF` |
| `dc_guesses` | — | Initial voltage guesses for DC convergence on tricky circuits |
| `transient_inputs` | — | Maps input waveform names to the nodes they drive |
| `ac_drives` | — | Like `input_drives` but drives a *node* rather than a device gate (used for testbench front-ends) |
| `periodic` | — | Large-signal periodic input description for PSS/PAC/PNoise and periodic transient |
| `analyses` | — | `run_analysis_suite()` dispatch config for `ac/noise/transient/pss/pac/pnoise` |
| `aliases` | — | Shortcuts so tools/sweeps can find key nodes by name (e.g. `"VOP"`, `"VON"`) |
| `explore` | — | Design-space exploration config (variables, constraints, objectives) |

---

## Interactive AFE Tuner

A web-based tuner for real-time exploration:

```bash
python3 -m pip install -r requirements-demo.txt
python3 demo/server.py
# Open http://localhost:5100
```

Adjust device W/L and bias voltages in the browser, see gain/BW/IRN update
live. Includes preset designs and DC warm-start logic.

---

## Benchmarks

Four fixed benchmarks for performance tracking:

```bash
python3 -m benchmarks.bench_afe --warm-runs 3         # AC+noise+transient
python3 -m benchmarks.bench_model --warm-runs 3       # Single-device micro
python3 -m benchmarks.bench_chopper --warm-runs 3     # Chopper: 5 analysis levels
python3 -m benchmarks.bench_sweep --n-candidates 200  # Batch explore workload
```

Set `CIRCUIT_USE_NUMBA=0` for pure-Python comparison. Numba kernels use on-disk
cache by default, so later Python processes can avoid most repeated cold-JIT
startup cost; set `CIRCUIT_NUMBA_CACHE=0` to disable that cache. Typical warm
timings on a modern Mac (Numba enabled):

| Benchmark | Time |
|-----------|------|
| AC 121 points | ~1.5 ms |
| Noise 121 points (standalone) | ~1.7 ms |
| DC+AC+Noise 121 points (AC reused) | ~1.8 ms |
| Transient 200 steps | ~5 ms |
| Ideal chopper (31 harmonics) | ~5 ms |
| PMOS chopper LPTV | ~22 ms |
| Chopper transient (8-PMOS, 225 Hz, 2 cycles, UI sizes) | ~0.15–0.19 s |
| Chopper PSS+PAC(HB)+PNoise (61 points, UI sizes) | ~25.6 s |
| Chopper PSS+PAC(HB)+PNoise (121 points, UI sizes) | ~48.9 s |
| Chopper PAC time-domain only (61 points, same PSS orbit) | ~1.3 s |
| Chopper PAC time-domain only (121 points, same PSS orbit) | ~1.9 s |
| Batch sweep (200 candidates, AC+noise) | ~0.5 s |

Those 25.6 s / 48.9 s full-flow numbers are for the portable HB PAC path. On
rail-driven choppers, `time_domain=True` removes PAC as the dominant bottleneck,
but it remains an opt-in speed path until the slow-corner gain gap is closed.

---

## Example Files

| File | What it is |
|------|-----------|
| `examples/afe_explore.json` | The locked 10-transistor AFE design with sizes, bias, NF, and explore sweep config |
| `examples/single_stage.json` | Minimal single-transistor common-source stage — best starting point for a new circuit |
| `examples/resistor_load_stage.json` | Single transistor with resistive load, demoing `resistors` and `current_sources` fields |
| `examples/periodic_rc.json` | Passive RC lowpass with PSS/PAC/PNoise dispatch — simplest end-to-end periodic example |
| `examples/voltage_divider.json` | Ideal voltage source (true MNA) divider with resistors, capacitors — vsource demo |
| `examples/afe_testbench.py` | Full testbench: dry-electrode front-end (R∥C network) → AFE core → AC + noise + transient |
| `examples/mc_mismatch.py` | Monte Carlo mismatch driver: corner table + 3-corner MC figure |

---

## Troubleshooting

**DC solve doesn't converge.**
Start with `examples/single_stage.json` (one transistor, trivial to converge).
For larger circuits, add `dc_guesses` to the JSON — a dictionary of approximate
node voltages. The locked AFE JSON includes these.

**Transient returns `nfail > 0`.**
Some Newton steps failed. Try: (a) more time points (`np.linspace(0, T, more_steps)`),
(b) tighter `newton_vtol` (default `1e-8`), or (c) enable
`fallback_least_squares=True`. For switched circuits, make sure `max_step` is
smaller than the fastest edge. If using `integration_method="gear2"` on a stiff
circuit with `max_step` / `max_retry_subdivisions`, the hot path stays in the
Numba gear2 grid. Python fallback is only a last resort if the compiled robust
step is rejected. Check `numba_grid_solver`, `gear2_python_retry_solver`, and
`transient_profile.failed_intervals`.

**PSS doesn't converge (`converged=False`).**
Increase `tstab_periods` (extra stabilization cycles before shooting starts)
or reduce `max_shooting_iters`. Check `pss['shooting_history']` to see if the
residual is decreasing. If it stalls, the orbit may be genuinely aperiodic —
check that all input waveforms are periodic with the same period.

**PNoise is slow.**
Reduce `max_sideband` (odd harmonics dominate the fold; 5–7 is often enough)
or `n_period_samples` (trade time-domain resolution for speed). Reuse the same
`pss_result` when sweeping output bands or repeated frequency grids: PNoise now
caches LPTV linearization, HB blocks, and identical-frequency adjoint solves.
With Numba installed, large HB block assembly, noise folding, and gm/gds
linearization also use compiled kernels.
For the current UI chopper case, PNoise is not the full-flow bottleneck: about
0.55 s for 61 points and 0.93 s for 121 points.

**PSS / periodic transient is slow.**
For chopper PSS, first ensure `analytic_jacobian=True` (the default), which builds
the shooting Jacobian in one orbit pass instead of `n_state` finite-difference
period runs. Chopper PSS defaults now use `fallback_least_squares=False`, keeping
the full period in the Numba grid solver and recording failed intervals without
rerunning the period in Python. Use `fallback_least_squares=True` only when
debugging a difficult convergence case. One stabilization period is usually
enough for the PMOS chopper wrappers; extra stabilization cycles are mostly a
throughput tradeoff.

**PAC is slow.**
Leave `compute_condition` unset for normal runs. PAC condition diagnostics are
computed only for `profile=True`, `debug=True`, or explicit
`compute_condition=True`, because the diagnostic runs an SVD of the HB matrix at
every frequency and does not affect gain/BW/noise.
For rail-driven choppers, `time_domain=True` (or JSON `"time_domain": true`) uses
the accelerated time-domain Floquet PAC path, but keep it as a speed/diagnostic
option for now: the current slow-corner Cadence check is about −1.9% at
baseband. If you stay on the default HB path, PAC frequency solves are the
dominant cost: about 24–25 s for 61 points and 47–48 s for 121 points. Further
HB-only speed work should target factorization reuse or batched linear solves.

---

## Further Reading

| Document | When to read it |
|----------|----------------|
| [Core Solver Overview](core_overview.md) | Understand how each solver works, import dependencies, and calibration data |
| [JSON Circuit Format](json_circuit_format.md) | Full field-by-field reference for writing your own circuit JSON |
| [Future Plan](futureplan.md) | What's done, what's next, and the execution roadmap |
| `tests/` directory | Working examples of every API call with expected outputs |
| `benchmarks/` directory | Performance baselines and how the hardware-accelerated paths compare |

---

## Motivation

Analog design needs many simulator runs to tune transistor sizes and bias.
Cadence/Spectre is accurate but slow for sweeping thousands of candidates.

The workflow this project enables:

1. **Cadence/Spectre** = trusted reference.
2. **This repo** = fast local model, calibrated to match Spectre behavior.
3. **Explore locally** — sweep sizes, bias, corners; filter by constraints.
4. **Verify in Cadence** — only the best candidates go back to Spectre.

---

## Contributing

Issues and PRs welcome.

---

## Intended Use

Research and early-stage analog design exploration. **Not** a sign-off simulator
replacement. Use it to understand trade-offs, narrow the search space, and
prepare better candidates for Cadence/Spectre verification.
