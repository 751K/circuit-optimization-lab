# Core Solver Overview

[Project overview](README.md) | [中文说明](README_zh.md) | [中文版](core_overview_zh.md)

This document introduces the current `core/` solver stack. The code is a compact local implementation of an AT4000TG OTFT ECG AFE solver, calibrated against Cadence/Spectre behavior. It is intended as the first concrete backend of the broader local circuit optimization flow.

## Scope

The current solver stack covers:

- DC operating-point solving.
- AC small-signal gain and bandwidth analysis.
- Noise analysis, including flicker noise and thermal noise.
- Transient response simulation.
- Periodic steady-state (PSS) shooting for periodic transient orbits.
- PSS-assisted PAC (periodic AC) via analytic-adjoint harmonic balance (default),
  opt-in time-domain Floquet shooting, or finite-difference shooting.
- Periodic PNoise, with harmonic-balance and time-domain Floquet-adjoint paths
  plus cyclostationary noise folding.
- Process-corner and per-device mismatch perturbations.
- Cadence/Spectre-oriented validation for operating point, AC, noise, transient,
  PSS, PAC, and PNoise behavior.

The implementation is intentionally small and self-contained. It currently has
22 Python files under `core/`, including `__init__.py`, the CLI entry
`__main__.py`, calibration/PSF/Cadence-netlist helpers, and the main solver stack.

## File Structure

```text
core/
  topology.py          Circuit topology source of truth.
  compiled_topology.py Runtime-compiled topology/index/stamp metadata.
  circuit_loader.py    JSON circuit description loader.
  device_model.py      TransistorModel ABC + NumbaParams + model factory/registry + PDK/polarity layer.
  pmos_tft_model.py    AT4000TG PMOS-OTFT compact-model implementation.
  numba_kernels.py     Optional Numba-accelerated scalar kernels.
  ac_mna.py            MNA stamping primitives.
  ac_solver.py         DC operating point and AC small-signal solver.
  noise_solver.py      Noise propagation and input-referred noise analysis.
  transient_solver.py  Time-domain transient solver.
  pss_solver.py        Transient-shooting periodic steady-state solver.
  pac_solver.py        Generic PSS-assisted PAC solver.
  pnoise_solver.py     Generic PNoise solver (HB + TD adjoint).
  analysis_dispatch.py JSON analysis-configuration dispatch entry point.
  psf.py               PSFASCII parser for Spectre reference data.
  calibration.py       Local-vs-Cadence calibration comparison helpers.
  cadence_netlist.py   Spectre netlist generation helpers for validation runs.
  chopper.py           Ideal and PMOS-switch differential chopper analyses.
  explore.py           Design-space exploration / optimization driver.
  corners.py           Process corners, mismatch MC, and latch detection.
```

## Import Relationship

```text
topology.py          <- no internal dependency
compiled_topology.py <- no internal dependency; consumes Topology-like objects at runtime
circuit_loader.py    <- topology
numba_kernels.py     <- no internal dependency; optional numba at runtime
device_model.py      <- no internal dependency (abc, dataclasses only)
pmos_tft_model.py    <- optional numba_kernels, device_model
ac_mna.py            <- no internal dependency
ac_solver.py         <- topology, compiled_topology, ac_mna, device_model
noise_solver.py      <- ac_solver, compiled_topology, topology, ac_mna, device_model
transient_solver.py  <- ac_solver, compiled_topology, topology, device_model
pss_solver.py        <- ac_solver, ac_mna, device_model, topology, transient_solver
pac_solver.py        <- ac_mna, ac_solver, device_model, transient_solver
pnoise_solver.py     <- ac_solver, noise_solver, pac_solver, device_model, ac_mna
analysis_dispatch.py <- ac_solver, noise_solver, transient_solver, pss_solver, pac_solver, pnoise_solver, circuit_loader
psf.py               <- no internal dependency
calibration.py       <- ac_solver, noise_solver, chopper, psf, circuit_loader
cadence_netlist.py   <- circuit_loader, topology
chopper.py           <- noise_solver, pss_solver, pac_solver, pnoise_solver, device_model, topology
explore.py           <- ac_solver, noise_solver, device_model, topology, circuit_loader
corners.py           <- ac_solver, noise_solver, topology
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
  (`CIRCUIT_USE_NUMBA=0` disables it; `CIRCUIT_NUMBA_CACHE=0` disables the
  default on-disk JIT cache).

For AC and noise analysis, the solver extracts terminal `gm` and `gds` by finite-differencing `get_Idc`, matching the terminal behavior used by the circuit solver.

`PMOS_TFT` inherits from :class:`~device_model.TransistorModel`, the abstract
base class consumed by all solvers. It also provides `get_numba_params()` for the
transient solver’s compiled inner loop and a Numba-accelerated `get_ss_params()`
override.

### `device_model.py`

Defines the abstract device‑model interface that decouples solvers from concrete transistor implementations:

- **`TransistorModel` (ABC)** — seven abstract methods (`get_Idc`, `get_op`, `get_capacitances`, `get_capacitance_charges_from_op`, `get_capacitance_branch_terms_from_op`, `get_noise_psd`, `get_numba_params`); `get_ss_params` provides a finite‑difference default that subclasses can override.
- **`NumbaParams` (frozen dataclass)** — the 16 scalar parameters extracted once per device and passed to numba‑accelerated transient kernels.
- **`register_model()` / `create_device()` + PDK/polarity layer** — factory + registry. Each `(pdk, polarity)` pair registers under a structured key `"<pdk>.<polarity>"` (e.g. `"at4000tg.pmos"`); `register_pdk()` groups one process's polarities and marks the default. Solver files call `create_device(get_default_model_type(), …)` — a single switch point — instead of hardcoding a model name, so a new process or an `nmos` polarity slots in with one `register_pdk` call and no solver edits. `"pmos_tft"` stays a back-compat alias. Generic elements (R/C/ideal V/I/controlled sources) are process-independent topology primitives and are **not** in this registry, so every PDK reuses them unchanged.

### `topology.py`

Defines the circuit topology as the single source of truth. The topology contains the transistor list, solved node list, rail/bias nodes, outputs, AC input drives, load capacitors, transient input mapping, DC guesses, and DC aliases. Solver runtime metadata is derived from this topology instead of being hand-written separately in each solver.

Alongside the transistors it also carries passive/source elements — `resistors` (a-b, R in ohms), `capacitors` (a-b, C in farads), `isources` (ideal DC current sources, I from nplus to nminus), `vccs` (voltage-controlled current sources: p, q, ctrl_p, ctrl_n, gm), `vcvs` (voltage-controlled voltage sources: p, q, cp, cn, mu → Vp−Vq=μ(Vcp−Vcn)), `cccs` (current-controlled current sources: p, q, ctrl_name, beta → Iout=β·Ictrl), `ccvs` (current-controlled voltage sources: p, q, ctrl_name, gamma → Vp−Vq=γ·Ictrl), and `vsources` (ideal voltage sources, true MNA: p, q, value). Each vsource/VCVS/CCVS adds one branch-current unknown and a constraint row, growing the system from `n` to `n_aug = n + m`. These flow through all analyses: resistor branch currents and current-source injections enter the DC KCL; resistors stamp as `1/R`, capacitors as `jωC`, VCCS as ``gm*(Vcp-Vcn)``, VCVS/CCVS/vsource as a bordered ``[[Y,B],[B^T,0]]`` block with the respective constraint rows, and CCCS as a coupling into the KCL rows; resistors add `4kT/R` thermal noise (all controlled sources and ideal voltage sources are noiseless); transient adds resistor conductances, capacitor companions, constant/VCCS/CCCS source currents, and VCVS/CCVS/vsource branch-current unknowns with their constraint equations. Current sources are open-circuit in the small-signal AC system. CCCS and CCVS can cascade: they control on the branch current of any vsource/VCVS/CCVS. None of these touch the transistor model machinery.

The default topology is `AFE_TOPO`, a 10-transistor fully differential AFE core with tail current device, input pair, output stage, and cross-coupled positive-feedback level shifting devices.

### `compiled_topology.py`

Builds a runtime plan from a declarative `Topology` and a bias/input context. It resolves node names once into compact terminal tokens and exposes shared metadata for DC, AC/noise, and transient:

- solved-node indices and rail values;
- per-device drain/gate/source terminal tokens;
- resistor, capacitor, current-source, VCCS, VCVS, CCCS, and CCVS stamp metadata;
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

Compiled Numba kernels are cached on disk by default, so later Python processes
can reuse the generated code instead of paying the full cold JIT cost again.
Disable only the cache with:

```bash
CIRCUIT_NUMBA_CACHE=0
```

`core.explore` and `core.corners` still set `CIRCUIT_USE_NUMBA=1` by default for
long-running workloads, but the general solver path now also uses Numba
automatically when it is available.

At present, the accelerated paths are PMOS current evaluation, internal-node
Newton iterations, bias-dependent capacitance evaluation, AC/PNoise
terminal-derivative small-signal parameters, PNoise HB block assembly/noise fold,
and the transient Newton inner loop: topology token lookup, PMOS state solve,
residual/Jacobian stamping, and the small dense Newton solve. The dense Newton
solve uses an in-place `A*x = -R` path to avoid avoidable per-iteration copies.
If the compiled path cannot handle a step, `transient_solver.py` falls back to
the original Python Newton / full-Jacobian / least-squares path.

### `ac_mna.py`

Provides the low-level MNA stamping primitives used by the small-signal solvers:

- Admittance stamping.
- VCCS, VCVS, CCCS, and CCVS stamping.
- Ideal voltage source stamping (bordered MNA).
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

The noise flow supports the same topology-derived terminal mapping and corner/mismatch parameter passing as the AC solver. In JSON dispatch, when AC and noise use the same frequency grid, noise reuses the prior AC `dc_op`, small-signal parameters, and gains instead of running DC/AC again; with different grids it still uses the AC `dc_op` as a warm seed.

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
  with those finite-edge harmonic weights. It is a fast **first-order** quasi-static
  estimate and underestimates the baseband gain by ~10% (it omits the higher-order
  LPTV conversion). For Cadence-grade gain/noise use the constant-free harmonic-
  balance path (`pmos_chopper_pss` → `pmos_chopper_pac`/`pmos_chopper_pnoise`). The
  old Cadence-fit conversion-phase / noise-PSD-scale constants were retired; the
  `conversion_phase_rad` / `periodic_noise_psd_scale` args remain as manual knobs.
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
- `pmos_chopper_pss(...)` wraps that same hard-switched topology in the generic
  shooting PSS solver and returns one periodic orbit for the selected clock
  period. This is the foundation for native PAC/PNoise. The wrapper defaults to
  `cap_mode="average"` for the orbit: a trapezoidal `0.5*(C_n+C_{n-1})*dV`
  discretization that matches Cadence's commutation feedthrough on the chopper's
  high-impedance internal nodes. Generic transient/PSS calls still default to
  the charge-conservative Q-stamp.
- `pmos_chopper_pac(...)` is a compatibility wrapper over the generic
  `core.pac_solver.pac_solve(...)`. The generic solver still defaults to the
  analytic-adjoint HB path, but the chopper wrapper defaults to the time-domain
  Floquet path (`method="pss_time_domain"`): it builds the one-period monodromy
  once in time domain, then solves a small quasi-periodic boundary system per
  frequency. For PMOS_TFT periodic conversion it retains each device's hidden
  `gate1` small-signal state (`R_cap`, `R_cap2`, `Cgs`, `Cgd`) instead of
  collapsing to terminal `{gm,gds,Cgs,Cgd}` at every orbit sample. The periodic
  conversion linearization uses the Verilog-A-style `C(V)*ddt(V)` operator that
  Spectre PAC folds, not necessarily the transient companion operator used to
  generate the large-signal orbit. When every PMOS device exposes the `gate1`
  network and Numba is enabled, the gate1-retained PAC linearization is assembled
  by the compiled `pac_linearize_orbit_gate1` kernel; mixed topologies fall back
  to the Python assembly. Set `time_domain=False` for the analytic-adjoint HB
  comparison path, or `analytic=False` for the original finite-difference
  shooting path. Static PSS orbits automatically reduce to ordinary `ac_solve`,
  avoiding PAC transient runs.
- `pmos_chopper_pnoise(...)` is a compatibility wrapper over the generic
  `core.pnoise_solver.pnoise_solve(...)`. For chopper verification it defaults to
  the time-domain Floquet adjoint: solve the sparse periodic adjoint BVP
  directly, then reuse the existing cyclostationary device/resistor noise fold.
  This removes the HB-adjoint sideband-truncation error. The harmonic-balance
  PNoise path remains available with `time_domain=False`: sample periodic
  small-signal G(t)/C(t), FFT to Fourier coefficients, assemble
  `Y[kr,kc] = G_{kr-kc} + jω·C_{kr-kc}`, solve one adjoint system per baseband
  frequency, and fold noise to the baseband output. Unlike
  `pmos_chopper_lptv_analysis`, neither path needs a Cadence calibration factor.

The PMOS-switch sideband path was initially validated with
`pmos_chopper_lptv_analysis` paired with Cadence calibration constants. The
native `pmos_chopper_pac` and `pmos_chopper_pnoise` now replace those
calibration-dependent paths with first-principles periodic small-signal and
noise solves. The finite-edge transient path has been checked against Spectre
`tran`. For the D3 `chop_tb_d3` slow-corner PSS/PAC/PNoise reference, native
default time-domain PAC is about +0.03%, and native TD PNoise IRN is about
+0.02%. The old HB-K32 PNoise IRN errors across slow/typical/fast were
+1.81% / +1.05% / +0.66%; the TD-adjoint errors are +0.02% / -0.00% / +0.57%.
This closes the earlier false comfort from sideband truncation while remaining
first-principles.

### `pss_solver.py`

Solves periodic steady state by shooting on top of the existing transient
engine:

- `pss_solve(sizes, bias, period, topo=..., tgrid=..., inputs=..., node_inputs=...)`
- Integrates one period with `transient(...)` and solves `x(T)-x(0)=0`.
- Uses the DC operating point as the default seed, with optional stabilization
  periods before shooting.
- **Shooting Jacobian:** By default (`analytic_jacobian=True`), the first
  Jacobian is built analytically in one orbit pass: sample the small-signal
  G(t)/C(t) stamps at each converged trajectory step, form the per-step map
  A_m = (G_m + C_m/h)^{-1} · (C_m/h), and accumulate the monodromy matrix
  Φ = ∏ A_m. The shooting Jacobian is then Φ - I — O(1) orbit pass instead of
  `n_state` finite-difference period runs. Falls back to finite differences on
  failure. Set `analytic_jacobian=False` to use the original finite-difference
  path.
- After the first Jacobian build (analytic or FD), the solver reuses it with a
  Broyden secant update. This removes repeated Jacobian builds on later shooting
  iterations while still recomputing the true one-period residual for every
  accepted step. Use `jacobian_reuse=False` to rebuild every iteration, or set
  `jacobian_rebuild_interval` for periodic rebuilds.
- Returns the one-period trajectory, `x0`, `x_end`, residual vector/norm,
  convergence flag, iteration history, and performance counters such as
  `shooting_period_runs`, `shooting_jacobian_evals`, and
  `shooting_jacobian_reuses`. The history records the Jacobian kind used
  (`"analytic_monodromy"` or `"finite_difference"`).
- PMOS chopper wrappers default to the Numba-grid transient path
  (`fallback_least_squares=False`), one stabilization period, and the chopper-only
  `cap_mode="average"` orbit when they build a PSS internally for PAC/PNoise.
  This preserves the residual/nfail convergence checks while avoiding Python
  fallback reruns on every period.
- Chopper PSS auto-seeding caches the bare-AFE DC seed for identical
  size/bias/corner inputs. Repeated analyses reuse only the initial guess; the
  real shooting residual is still recomputed, so convergence accuracy is unchanged.

This PSS orbit can feed the generic `pac_solve` and `pnoise_solve` kernels
directly. `pmos_chopper_pac` / `pmos_chopper_pnoise` are wrappers that map the
chopper's differential input to `input_drive={"vip": 0.5, "vin": -0.5}`.

### `pac_solver.py`

- `pac_solve(sizes, bias, freqs, pss_result=..., input_drive=...)`
- Circuit-generic: it only requires the PSS result to carry `topology`, `t`,
  `nodes`, `x0`, `x_end`, `output`, and periodic input-waveform metadata.
- `input_drive` maps small-signal complex amplitudes to transient input keys,
  e.g. `{"vip": 0.5, "vin": -0.5}` for differential input or `{"vin": 1.0}`
  for single-ended input.
- Four performance paths, tried in order:
  1. **LTI fast path** — static PSS orbits reduce to ordinary `ac_solve`.
  2. **Time-domain Floquet PAC** (`time_domain=True`; chopper wrapper default) — samples
     periodic G(t)/C(t) and input coupling on a uniform orbit grid, builds the
     frequency-independent monodromy once, then solves
     `(exp(jωT)I - Ψ)x0 = g` per frequency. This avoids HB sideband truncation
     and the large `(2K+1)n` conversion matrix. PMOS_TFT devices are expanded
     with their internal `gate1` small-signal states during periodic conversion.
     The all-PMOS gate1 case uses a Numba assembly kernel; unsupported mixed,
     bordered, or vsource-driven cases return `None` and continue to the next
     path.
  3. **Analytic-adjoint** (generic default, `analytic=True`) — samples periodic G(t)/C(t)
     and input-coupling columns G_in(t)/C_in(t) on the PSS orbit, FFTs to
     harmonic coefficients G_k/C_k, builds the harmonic-balance conversion
     matrix Y_HB(f), and reads sideband-0 gain from a single adjoint linear
     solve per frequency. Cost is O(1) per frequency with zero extra transient
     runs. Controlled by `n_period_samples` (time resolution) and `max_sideband`
     (sideband count).
  4. **Finite-difference shooting** (`analytic=False`) — finite-differences the
     state transition matrix Φ and the complex input forcing around the PSS
     orbit, then solves `(Φ-γI)dx0=-b`. Costs `n_state+2` transient runs per
     frequency. Cached on `pss_result` for repeated PAC/PNoise calls.
- Results include counters such as `pac_period_runs`, `pac_state_cache_hit`,
  `pac_input_cache_hits`, `pac_td_setup_time_s`, and `method`
  (`"pss_time_domain"`, `"pss_analytic_adjoint"`, or `"pss_fd_shooting"`).
  PAC condition diagnostics are off by default and are enabled only by
  `profile=True`, `debug=True`, or explicit `compute_condition=True`; the
  diagnostic does an SVD per frequency and does not affect gain/BW/noise.

### `pnoise_solver.py`

- `pnoise_solve(sizes, bias, freqs, pss_result=..., fundamental=...)`
- Uses generic `Topology` device/resistor/capacitor stamps. PMOS device noise is
  sampled along the PSS orbit; resistor thermal noise is folded as a stationary
  source.
- Static PSS orbits use the same LTI `noise_analysis` path as normal noise
  analysis. True LPTV runs cache sampled `G(t)/C(t)`, HB blocks, and
  identical-frequency adjoint solves on `pss_result` where the HB path is used.
- With `time_domain=True`, PNoise replaces the K-truncated HB adjoint solve with
  a sparse Floquet adjoint BVP (`pnoise_time_domain_used=True`). This is exact in
  the conversion sideband index; its remaining numerical error is the time-grid
  discretization, so default-like `n_period_samples < 640` is raised to 768.
- HB adjoint solves support `hb_solver="auto" | "dense" | "sparse" |
  "iterative"`. The default keeps small systems on dense BLAS/LAPACK and switches
  large, very sparse HB matrices to SciPy sparse direct solves. Forced
  `iterative` uses block-Jacobi preconditioned GMRES, with per-harmonic diagonal
  block LU factors, and falls back to sparse direct if convergence fails.
- With Numba available, large LPTV PNoise runs use compiled HB block assembly
  and compiled `freq × source × sideband²` noise folding. `get_ss_params()` also
  uses the compiled terminal-derivative path for gm/gds and falls back to the
  original finite difference near small-current/kink regions. The all-PMOS
  `gate1` PAC conversion assembly is also compiled when available. Numba/Rust-
  style compiled code mainly helps the matrix-fill and noise-fold loops; HB
  linear solves are dominated by BLAS/LAPACK, SuperLU, or GMRES rather than
  Python loop overhead.
- If `gains` or `pac_result` are not provided, pass the same `input_drive` and
  the function will call generic `pac_solve` for input-referred noise.

### `transient_solver.py`

Solves the time-domain response of the topology-defined system using backward Euler (default) or variable-step BDF2/gear2 integration:

- `transient(sizes, bias, tgrid, vip=None, vin=None, nf=None, V0=None, topo=AFE_TOPO, inputs=None, node_inputs=None, integration_method="be", adaptive=False)`
- Supports legacy AFE `vip/vin` inputs and generic `inputs={name: waveform}` driven through `topo.transient_inputs`.
- `node_inputs={node: input_key}` drives a (rail) NODE with a waveform — used by a front-end testbench where the stimulus enters at source nodes and propagates through a passive network, rather than driving device gates directly.
- `current_inputs=[{"p": node_a, "q": node_b, "input": key}]` stamps a
  time-varying ideal current source flowing `p -> q`; the PMOS chopper helper uses
  this for charge-injection pulses.
- `cap_mode` / `cap_mode_id` are per-call capacitance-operator overrides.
  `None` uses the module/environment default (`charge`), while chopper PSS passes
  the `average` operator explicitly for Cadence-matched clock feedthrough. This
  override affects only the transient/PSS orbit, not PAC/PNoise conversion
  linearization.
- `adaptive=True` is an opt-in LTE-controlled gear2 path. It treats the supplied
  `tgrid` as the input-sampling grid and `[t0, tstop]` boundary, then returns a
  self-chosen non-uniform accepted grid. It is valid only with
  `integration_method="gear2"` and uses `adaptive_reltol`,
  `adaptive_vabstol`, `adaptive_iabstol`, `adaptive_max_steps`, and
  `adaptive_h0`.
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
  solves the dense Newton step in one inner loop. The linear solve overwrites the
  temporary Jacobian/residual arrays in place, and the Python substep loop reuses
  the previous interpolated input as the next substep's start input.
- For PSS-style non-robust runs (`fallback_least_squares=False` and
  `fallback_full_jacobian=False`), the compiled grid solver stays in Numba across
  the full period. Failed substeps are counted as failed intervals and the
  trajectory continues from the last accepted state, matching the non-throwing
  Python transient behavior without rerunning the whole period in Python.
- Robust fallback modes still return to the Python path so least-squares or full
  finite-difference Jacobian recovery can be applied only when requested.
- Uses implicit differentiation of the PMOS internal nodes for faster transient
  Jacobians, with finite-difference fallback.

**Gear2/BDF2 integration** (`integration_method="gear2"`): The transient solver
also supports variable-step BDF2 (second-order, stiffly stable). Key properties:

- Uses the stable charge-mode capacitor companion `i_n = (α0·Q_n + α1·Q_{n-1} +
  α2·Q_{n-2})/h_n`, same as BE's `(Q_n − Q_{n-1})/h` but with two-step history.
  A per-call cap-mode override exists for calibrated orbit generation; the PMOS
  chopper PSS wrapper uses the `average` operator to match Spectre feedthrough,
  while generic stiff circuits keep the charge default.
- Step-ratio clamp ρ≤2 guarantees zero-stability on non-uniform grids.
- BE self-start on the first step of every interval.
- A compiled Numba gear2 grid solver (`_transient_solve_grid_gear2_impl`) handles
  periodic PSS/PAC/PNoise orbits and raw-transient `max_step`,
  `flat_max_step`, and `max_retry_subdivisions`; the analytic gear2 monodromy
  (augmented 2n-state) feeds the PSS shooting Jacobian.
- The adaptive gear2 path uses a step-doubling LTE estimate and freezes the PSS
  grid near convergence before the final fixed-grid orbit/monodromy. Numba
  accelerates adaptive gear2 for `n_aug == n`; circuits with ideal-voltage-source
  branch unknowns fall back to the Python adaptive loop.
- Raw `transient(integration_method="gear2")` keeps the BE default opt-in
  boundary, but when `max_retry_subdivisions` or `max_step` asks for robustness
  it stays in the Numba gear2 grid. The grid updates rolling two-step BDF2
  history after every accepted internal substep and retries failed substeps with
  fixed `2**max_retry_subdivisions` bisection. The Python gear2 `solve_chunk`
  path remains only as a last-resort fallback if the compiled robust step is
  rejected.
- Chopper PSS/PAC/PNoise default to gear2 — PAC baseband errors drop from BE's
  −2.5% (typ/fast) to <1% across all three corners.
- Raw `transient()` still defaults to BE to preserve the established raw
  transient regression surface and first-order damping behavior. The default BE
  hard-switched chopper transient hot path is also fully Numba now; normal runs
  no longer enter a Python tail or SciPy `least_squares`.

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
- `load_explore_json(path)` — read an `explore` block from a full circuit JSON.
  The topology, device sizes, bias, and optional NF data are all loaded through
  the same JSON path; legacy `builtin_topology` configs are no longer accepted
  in the exploration layer.
- Sampling is `lhs` (Latin hypercube) or `random`, with a seeded RNG for repeatability.
- Metrics: `gain_dB`, `bw_Hz`, `irn_uV`, `power_uW` (top-rail supply current x rail
  voltage), and `area` (sum of per-device `g_area`).
- A variable's `targets` can drive several keys at once, keeping matched pairs
  (M7=M8, ...) identical so the AFE's symmetric DC continuation stays on the
  physical branch.
- Results export to CSV and JSONL; a CLI runs `python -m core.explore <config.json>`.

Example configs: `examples/afe_explore.json` and `examples/single_stage.json`;
both are full circuit JSON files with an `explore` block.

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
table + 3-corner MC figure.

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

Four benchmarks live under `benchmarks/`:

```bash
# Full-AFE benchmark (ac121 / noise121 / tran200)
python3 -m benchmarks.bench_afe --warm-runs 3
CIRCUIT_USE_NUMBA=0 python3 -m benchmarks.bench_afe --warm-runs 3

# Single-device PMOS_TFT micro-benchmark (7 hot-path ops × 3 bias regions)
python3 -m benchmarks.bench_model --warm-runs 3
CIRCUIT_USE_NUMBA=0 python3 -m benchmarks.bench_model --warm-runs 3

# Chopper analysis benchmark (harmonics / ideal / pmos_static / pmos_lptv / pmos_tran)
python3 -m benchmarks.bench_chopper --warm-runs 3
python3 -m benchmarks.bench_chopper --skip-tran --warm-runs 3

# Batch sweep benchmark (N × AC / AC+noise, explore-layer workload)
python3 -m benchmarks.bench_sweep --n-candidates 200 --warm-runs 3
```

`bench_afe.py` reports cold and warm timings for the three canonical full-AFE
workloads. `bench_model.py` measures individual device operations (DC OP, Idc,
capacitances, noise PSD, Cadence metrics) across saturation, subthreshold, and
linear bias regions. `bench_chopper.py` covers the five chopper analysis levels
at f_chop=225 Hz — from fast finite-edge harmonic math (~1 ms) through ideal
LPTV folding, PMOS static-phase, quasi-static PMOS sideband folding, and the
heavy hard-switched PMOS chopper transient. `bench_sweep.py` measures batch
throughput of AC and AC+noise evaluation across randomly perturbed design
candidates, simulating the explore layer's per-candidate workload.  The
default run uses Numba when available; `CIRCUIT_USE_NUMBA=0` is useful for
pure-Python comparison.

The old UI chopper full-flow bottleneck was the portable HB PAC frequency solve:
explicit `PSS+PAC(HB)+PNoise` (`time_domain=False`) takes about 25.6 s for
61 frequency points (PSS≈0.35 s, PAC≈24.7 s, PNoise≈0.55 s) and about 48.9 s for
121 points (PSS≈0.44 s, PAC≈47.6 s, PNoise≈0.93 s). The default chopper
time-domain PAC path keeps the PMOS `gate1` states and uses the Numba gate1
conversion assembly when available; it takes about 1.4 s for 61 points on the
same PSS orbit. A non-chopper AFE `DC+AC+Noise` 121-point run
is about 1.8 ms when noise reuses the AC result.

## Calibration Status

The current core was calibrated against Cadence Spectre 24.1 for the AT4000TG AFE use case. The observed agreement in the original project included:

- Typical and corner AC behavior within approximately 0.01 dB for gain.
- Input-referred noise within a few percent across validated cases.
- Per-device mismatch Monte Carlo mean and standard deviation matching Cadence trends.
- Transient step and sinusoidal response closely matching Cadence `tran` behavior.
- PMOS eight-switch chopper transient, using UI locked sizes, `f_chop=225 Hz`,
  switch `W/L=5000/30`, and `rise/fall=20 us`, now matches the Spectre
  finite-edge transient scale with the default `edge_time/10` internal step:
  last-period output mean about `-10.76 mV` vs Spectre `-10.62 mV`, output
  `21.11 mVpp` vs `21.46 mVpp`, input common-mode swing `5.14 Vpp` vs
  `5.43 Vpp`, and `nfail=0`.
- PMOS eight-switch chopper PSS/PAC/PNoise, using the same UI locked case, is
  matched by `pmos_chopper_lptv_analysis(...)`: gain `21.370 dB` vs Spectre
  `21.369 dB`, bandwidth `738.6 Hz` vs `721.9 Hz`, and IRN `12.592 uVrms` vs
  `12.591 uVrms`.
- Native `pmos_chopper_pac` and `pmos_chopper_pnoise` (first-principles,
  no calibration constants) match the D3 `chop_tb_d3` slow-corner Spectre
  PSS/PAC/PNoise reference at `f_chop=200 Hz`: default time-domain PAC is about
  +0.03%, and TD-adjoint PNoise IRN is about +0.02%. Across slow/typical/fast,
  the old HB-K32 IRN errors were +1.81% / +1.05% / +0.66%; TD PNoise gives
  +0.02% / -0.00% / +0.57%.
- SC-LPF calibration now explicitly uses `gear2 + adaptive + cap_mode="average"`
  with edge breakpoints and enough PNoise resampling (`512` samples,
  `max_sideband=20`). Against the archived Spectre SC-LPF reference it currently
  passes with PAC gain about `-0.32%`, bandwidth `+1.07%`, and output noise
  `+2.82%`.
- Final locked design around 22.9 dB gain, 549 Hz bandwidth, and 37 uVrms
  input-referred noise.

These numbers describe the current AT4000TG validation case. Future PDKs or topologies should be recalibrated against their own simulator references.
