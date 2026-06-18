# Core Solver Overview

[Project overview](README.md) | [中文说明](README_zh.md) | [中文版](core_overview_zh.md)

This document introduces the current `core/` solver stack. The code is a compact local implementation of an AT4000TG OTFT ECG AFE solver, calibrated against Cadence/Spectre behavior. It is intended as the first concrete backend of the broader local circuit optimization flow.

## Scope

The current solver stack covers:

- DC operating-point solving.
- AC small-signal gain and bandwidth analysis.
- Noise analysis, including flicker noise and thermal noise.
- Transient response simulation.
- Process-corner and per-device mismatch perturbations.
- Cadence/Spectre-oriented validation for operating point, AC, noise, and transient behavior.

The implementation is intentionally small and self-contained. It currently consists of thirteen Python source files under `core/` (excluding `__init__.py`).

## File Structure

```text
core/
  topology.py          Circuit topology source of truth.
  compiled_topology.py Runtime-compiled topology/index/stamp metadata.
  circuit_loader.py    JSON circuit description loader.
  pmos_tft_model.py    AT4000TG PMOS-OTFT compact-model implementation.
  numba_kernels.py     Optional Numba-accelerated scalar kernels.
  ac_mna.py            MNA stamping primitives.
  ac_solver.py         DC operating point and AC small-signal solver.
  noise_solver.py      Noise propagation and input-referred noise analysis.
  chopper.py           Ideal and PMOS-switch differential chopper analyses.
  transient_solver.py  Time-domain transient solver.
  explore.py           Design-space exploration / optimization driver.
  corners.py           Process corners, mismatch MC, and latch detection.
  mc_corners.py        Cadence PSF Monte-Carlo post-processing helper.
```

## Import Relationship

```text
topology.py          <- no internal dependency
compiled_topology.py <- no internal dependency; consumes Topology-like objects at runtime
circuit_loader.py    <- topology
numba_kernels.py     <- no internal dependency; optional numba at runtime
pmos_tft_model.py    <- optional numba_kernels
ac_mna.py            <- no internal dependency
ac_solver.py         <- topology, compiled_topology, ac_mna, pmos_tft_model
noise_solver.py      <- ac_solver, compiled_topology, topology, ac_mna, pmos_tft_model
chopper.py           <- noise_solver, topology
transient_solver.py  <- ac_solver, compiled_topology, topology, pmos_tft_model
explore.py           <- ac_solver, noise_solver, pmos_tft_model, topology, circuit_loader
corners.py           <- ac_solver, noise_solver, topology
mc_corners.py        <- simulator-side PSF parsing helpers
```

## Main Components

### `pmos_tft_model.py`

Implements the AT4000TG PMOS-OTFT compact model in Python. It provides:

- Terminal current evaluation through `get_Idc`.
- Drain-current noise PSD through `get_noise_psd`.
- Bias-dependent terminal capacitances through `get_capacitances`.
- Geometry area calculation through `g_area`.
- Process and mismatch parameters such as `pvt0`, `mvt0`, `pbeta0`, and `mbeta0`.
- A warm-started internal-node operating point solve.
- Automatic Numba acceleration for hot scalar kernels when Numba is installed
  (`CIRCUIT_USE_NUMBA=0` disables it).

For AC and noise analysis, the solver extracts terminal `gm` and `gds` by finite-differencing `get_Idc`, matching the terminal behavior used by the circuit solver.

### `topology.py`

Defines the circuit topology as the single source of truth. The topology contains the transistor list, solved node list, rail/bias nodes, outputs, AC input drives, load capacitors, transient input mapping, DC guesses, and DC aliases. Solver runtime metadata is derived from this topology instead of being hand-written separately in each solver.

Alongside the PMOS_TFT transistors it also carries two-terminal passive/source elements — `resistors` (a-b, R in ohms), `capacitors` (a-b, C in farads), and `isources` (ideal DC current sources, I from nplus to nminus). These flow through all four analyses: resistor branch currents and current-source injections enter the DC KCL; resistors stamp as `1/R` and capacitors as `jωC` in AC/noise; resistors add `4kT/R` thermal noise; transient adds resistor conductances, capacitor companions, and constant source currents. Current sources are open-circuit (and noiseless) in the small-signal AC system. None of these touch the PMOS_TFT machinery.

The default topology is `AFE_TOPO`, a 10-transistor fully differential AFE core with tail current device, input pair, output stage, and cross-coupled positive-feedback level shifting devices.

### `compiled_topology.py`

Builds a runtime plan from a declarative `Topology` and a bias/input context. It resolves node names once into compact terminal tokens and exposes shared metadata for DC, AC/noise, and transient:

- solved-node indices and rail values;
- per-device drain/gate/source terminal tokens;
- resistor, capacitor, and current-source stamp metadata;
- AC/noise `("n", idx)` / `("v", value)` terminal tables;
- transient input and `node_inputs` mappings.

This keeps AC, noise, and transient on the same indexing/stamping convention while preserving the ability to swap in a different JSON-defined circuit.

### `circuit_loader.py`

Loads JSON circuit descriptions and returns a `CircuitSpec` containing:

- `topology`
- `sizes`
- `bias`
- `nf`

This lets new circuits be added through JSON files such as `examples/single_stage.json` without editing solver source code.

### `numba_kernels.py`

Provides optional Numba kernels for pure scalar hot paths. The module is safe to
import without Numba installed. When Numba is installed, kernels are enabled by
default; force the pure-Python path with:

```bash
CIRCUIT_USE_NUMBA=0
```

`core.explore` and `core.corners` still set `CIRCUIT_USE_NUMBA=1` by default for
long-running workloads, but the general solver path now also uses Numba
automatically when it is available.

At present, the accelerated paths are PMOS current evaluation, internal-node
Newton iterations, bias-dependent capacitance evaluation, terminal derivatives,
and the transient Newton inner loop: topology token lookup, PMOS state solve,
residual/Jacobian stamping, and the small dense Newton solve. If the compiled
path cannot handle a step, `transient_solver.py` falls back to the original
Python Newton / full-Jacobian / least-squares path.

### `ac_mna.py`

Provides the low-level MNA stamping primitives used by the small-signal solvers:

- Admittance stamping.
- VCCS stamping.
- MOS small-signal stamping.

### `ac_solver.py`

Solves the DC operating point and AC response:

- `ac_solve(sizes, bias, freqs, corner=None, x0_guess=None, topo=AFE_TOPO, nf=None)`
- Uses `scipy.fsolve` for the DC node equations.
- Returns gain, bandwidth, node operating point, and extracted small-signal parameters.
- Supports both global process corners and per-device mismatch maps.
- Uses topology metadata for output sensing, load capacitance, and AC input drive.

The DC solve includes robustness handling for physical branch selection, symmetric operating points, and rail-bounded node solutions.

### `noise_solver.py`

Performs noise propagation on the same topology-derived MNA system used by AC analysis. Each transistor drain-current noise source is injected between drain and source, propagated to the configured output, and divided by the signal gain to obtain input-referred noise.

The noise flow supports the same topology-derived terminal mapping and corner/mismatch parameter passing as the AC solver.

### `chopper.py`

Computes gain, bandwidth, and baseband noise for chopper variants around the AFE:

- `chopper_analysis(...)` is the ideal synchronized differential chopper model.
  It treats the eight-switch commutator as +/-1 square-wave multipliers at the
  input and output, then folds sideband gain/noise back to baseband with odd
  harmonic coefficients. This is the linear periodically time-varying (LPTV)
  frequency-domain path for ideal chopping and flicker-noise movement.
- `build_afe_pmos_chopper(...)` inserts the eight switches as real `PMOS_TFT`
  pass devices around the AFE input/output ports.
- `pmos_chopper_analysis(...)` runs static phase A/B AC and noise on that PMOS
  switch topology and averages the phases. It includes switch Ron loading,
  nonlinear capacitance, and PMOS switch noise.
- `finite_edge_clock_pair(...)` and `finite_edge_chopper_harmonics(...)` model
  finite clock edge time and break-before-make dead time in the chopper waveform.
- `pmos_chopper_lptv_analysis(...)` folds the PMOS-switch sideband response/noise
  with those finite-edge harmonic weights. This is a quasi-static LPTV
  approximation for time-varying switch operating points.
- `pmos_chopper_transient(...)` drives the eight-PMOS topology with finite-edge
  clocks. The default clock follows Spectre `type=pulse` timing (`delay=T/2`,
  `width=T/2`, finite `rise/fall`), while the older centered phase waveform is
  still available with `clock_style="phase"` for dead-time experiments. Clock
  feedthrough comes from the PDK `Cgss/Cgdd * ddt()` terms and the long-timescale
  `R_cap2` gate-leak terms stamped by the transient solver; optional
  charge-injection pulses are estimated from the same PDK capacitance equations
  and stamped as time-varying current sources. The helper refines the internal
  time grid around clock edges, uses signed terminal currents for the eight
  bidirectional pass switches, and keeps a tight residual tolerance so slow
  common-mode charge balance is not lost.

The PMOS-switch sideband path is still a quasi-static approximation, not a full
correlated periodic-noise solver. The finite-edge transient path has been checked
against Spectre `tran`; periodic-noise claims should still be validated against
Spectre PSS/PNoise for a clocked testbench.

### `transient_solver.py`

Solves the time-domain response of the topology-defined system using backward Euler integration:

- `transient(sizes, bias, tgrid, vip=None, vin=None, nf=None, V0=None, topo=AFE_TOPO, inputs=None, node_inputs=None)`
- Supports legacy AFE `vip/vin` inputs and generic `inputs={name: waveform}` driven through `topo.transient_inputs`.
- `node_inputs={node: input_key}` drives a (rail) NODE with a waveform — used by a front-end testbench where the stimulus enters at source nodes and propagates through a passive network, rather than driving device gates directly.
- `current_inputs=[{"p": node_a, "q": node_b, "input": key}]` stamps a
  time-varying ideal current source flowing `p -> q`; the PMOS chopper helper uses
  this for charge-injection pulses.
- `max_step`, `max_retry_subdivisions`, `fallback_full_jacobian`, and
  `fallback_least_squares` provide
  controlled step refinement and bounded fallback solving for switched transient
  steps.
- Includes topology-defined load capacitances (and capacitor elements), plus resistor and ideal-current-source branches.
- Re-evaluates nonlinear capacitances during Newton iterations, and includes the
  PMOS `R_cap2` source/drain-to-gate leakage branch used by the PDK Verilog-A
  model.
- Supports `signed_devices` for bidirectional pass switches. The default AFE
  path keeps the historical `abs(Idc)` convention that matches the calibrated
  DC/AC/noise solvers, while switch devices can keep physical drain-current sign
  when source/drain voltages reverse.
- Uses the DC operating point from `ac_solve` as the default initial condition.
- Uses the Numba transient Newton kernel when available. The compiled path
  evaluates PMOS operating points/capacitances, stamps the residual/Jacobian, and
  solves the dense Newton step in one inner loop; it falls back to the Python path
  if a device internal solve or matrix solve cannot be handled.
- Uses implicit differentiation of the PMOS internal nodes for faster transient
  Jacobians, with finite-difference fallback.

### Front-end stimulus (`ac_drives`)

For a testbench, the small-signal AC stimulus can be applied at NODES via `Topology.ac_drives` (e.g. `{"VINP": +0.5, "VINN": -0.5}`) instead of at device gates. The drive propagates through the passive front-end network to the (now solved) amplifier-input nodes, and the gain is normalized by the differential stimulus. In noise analysis these drives are treated as AC ground (inputs carry no signal). `examples/afe_testbench.py` builds the dry-electrode + AC-coupling front end (R_EL∥C_EL, C_AC series, R_AC to VCM) in front of the AFE core and runs AC (bandpass ≈ 0.05 Hz–few-hundred Hz), input-referred noise (including R_EL/R_AC thermal noise), and an in-band transient. Because the AC-coupled input makes the bare AFE DC multistable, the testbench seeds its DC solve from the robust bare-AFE operating point (`dc_seed`).

### `explore.py`

Design-space exploration / optimization driver built on top of the AC and noise
solvers — the "optimization" the project is named for. Given a circuit plus an
`explore` configuration (design variables with ranges, feasibility constraints,
and one or more objectives), it samples candidates, evaluates each through the
solvers, filters by constraints, and Pareto-selects the trade-off front.

- `explore(topo, base_sizes, base_bias, nf, cfg, n=, seed=, method=, corner=)` — run a sweep.
  `corner` applies a process shift (e.g. `CORNERS["slow"]`) to every evaluation, enabling
  corner-aware search without modifying the config.
- `evaluate(topo, sizes, bias, nf, freqs, band, x0_guess=None, corner=None)` — single-candidate
  solver evaluation, now with optional corner/mismatch argument. During `explore`,
  evaluation is AC-first: gain/BW/power/area are computed before noise, failed
  candidates are rejected immediately, and `noise_analysis` runs only when
  `irn_uV` is required by a surviving candidate's constraints or objectives.
- `load_explore_json(path)` — read an `explore` block from a full circuit JSON, or
  from a file naming a `builtin_topology` (e.g. `AFE_TOPO`) plus baseline sizes/bias.
- Sampling is `lhs` (Latin hypercube) or `random`, with a seeded RNG for repeatability.
- Metrics: `gain_dB`, `bw_Hz`, `irn_uV`, `power_uW` (top-rail supply current x rail
  voltage), and `area` (sum of per-device `g_area`).
- A variable's `targets` can drive several keys at once, keeping matched pairs
  (M7=M8, ...) identical so the AFE's symmetric DC continuation stays on the
  physical branch.
- Results export to CSV and JSONL; a CLI runs `python -m core.explore <config.json>`.

Example configs: `examples/afe_explore.json` (built-in AFE topology) and the
`explore` block in `examples/single_stage.json` (generic JSON path).

### `corners.py`

Single source of truth for process-corner and robustness work — the pieces that
otherwise get re-derived in every sweep:

- `CORNERS` — global process shifts (`typical` / `slow` / `fast` as `pvt0`/`pbeta0`,
  from the PDK monte.scs sections; e.g. slow = `{"pvt0": -0.2259, "pbeta0": -0.54}`).
- `mismatch_corner(rng, devices, base)` — per-device random `mvt0`/`mbeta0` on top of
  a process corner.
- `metrics(...)` — one design at one corner → `gain_peak_dB`, `bw_Hz`, `irn_uV`, and
  `latch_dV` (`|out+ - out-|` at the DC op; large ⇒ the cross-coupled positive feedback
  has latched).
- `corner_table(...)` — metrics across typ/slow/fast.
- `latch_screen(...)` — deterministic worst-case latch screen: pushes each symmetric
  pair ±kσ apart over ALL sign patterns and returns the largest output imbalance. A
  single fixed kick has false negatives (the latching sign pattern is design-dependent),
  so the screen scans patterns; cheap enough to use inside a search instead of a full MC.
  It skips noise because only the DC/AC operating point and latch imbalance are needed.
- `mismatch_mc(...)` — per-device mismatch MC at one corner, seeded from the nominal op;
  returns per-metric arrays, a latched mask, and a summary (latch rate + non-latched
  mean/std/P5/P95). Each sample is evaluated AC-first; IRN is computed only for
  non-latched samples included in the final noise statistics.

`ac_solve` / `noise_analysis` accept the same `corner` argument (a flat process dict or a
per-device mismatch map). The driver `examples/mc_mismatch.py` wraps this into a corner
table + 3-corner MC figure. (Distinct from `core/mc_corners.py`, which post-processes
Cadence PSF output — that is the simulator-side flow, this is the local-solver side.)

## Quick Example

```python
import numpy as np

from core.ac_solver import ac_solve
from core.noise_solver import noise_analysis, band_rms
from core.transient_solver import transient

sizes = {
    "M6": (2264, 78),
    "M7": (61365, 61),
    "M8": (61365, 61),
    "M9": (3175, 468),
    "M10": (3175, 468),
    "M11": (465, 66),
    "M12": (894, 85),
    "M13": (894, 85),
    "M14": (5224, 46),
    "M15": (5224, 46),
}

bias = {
    "VDD": 40.0,
    "VCM": 30.65,
    "VB": 9.84,
    "VC": 16.0,
}

freqs = np.logspace(-2, 4, 121)

ac = ac_solve(sizes, bias, freqs)
noise = noise_analysis(sizes, bias, freqs)
irn_uv = band_rms(freqs, noise["irn_psd"], 0.05, 100) * 1e6

t = np.linspace(0, 4e-3, 400)
vip = np.where(t >= 0.5e-3, bias["VCM"] + 0.5e-3, bias["VCM"])
vin = np.where(t >= 0.5e-3, bias["VCM"] - 0.5e-3, bias["VCM"])
tran = transient(sizes, bias, t, vip, vin)
```

## JSON Circuit Example

New circuits can be loaded from JSON. The field-level format is documented in
[JSON 电路描述格式](json_circuit_format_zh.md).

```python
import numpy as np

from core.circuit_loader import load_circuit_json
from core.ac_solver import ac_solve
from core.transient_solver import transient

spec = load_circuit_json("examples/single_stage.json")
freqs = np.logspace(0, 4, 121)

ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)

t = np.linspace(0, 1e-3, 100)
vin = np.full_like(t, spec.bias["VIN"])
tran = transient(spec.sizes, spec.bias, t, topo=spec.topology,
                 nf=spec.nf, inputs={"vin": vin})
```

## Benchmarks

Fixed AFE benchmarks live under `benchmarks/`:

```bash
python3 -m benchmarks.bench_afe --warm-runs 3
CIRCUIT_USE_NUMBA=0 python3 -m benchmarks.bench_afe --warm-runs 3
```

The script reports cold and warm timings separately for `ac121`, `noise121`, and `tran200`. The default run uses Numba when available; the `CIRCUIT_USE_NUMBA=0` run is useful for pure-Python comparison.

## Calibration Status

The current core was calibrated against Cadence Spectre 24.1 for the AT4000TG AFE use case. The observed agreement in the original project included:

- Typical and corner AC behavior within approximately 0.01 dB for gain.
- Input-referred noise within a few percent across validated cases.
- Per-device mismatch Monte Carlo mean and standard deviation matching Cadence trends.
- Transient step and sinusoidal response closely matching Cadence `tran` behavior.
- PMOS eight-switch chopper transient, using UI locked sizes, `f_chop=225 Hz`,
  switch `W/L=5000/30`, and `rise/fall=20 us`, now matches the Spectre
  finite-edge transient scale: last-period output mean about `-11.05 mV`
  vs Spectre `-10.62 mV`, output `21.28 mVpp` vs `21.46 mVpp`, input common-mode
  swing `5.18 Vpp` vs `5.43 Vpp`, and `nfail=0`.
- Final locked design around 22.9 dB gain, 549 Hz bandwidth, and 37 uVrms input-referred noise.

These numbers describe the current AT4000TG validation case. Future PDKs or topologies should be recalibrated against their own simulator references.
