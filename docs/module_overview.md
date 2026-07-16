# Core Solver Overview

[Project overview](README.md) | [中文说明](README_zh.md) | [中文版](module_overview_zh.md)

> **Status: maintained architecture reference.** Module responsibilities and
> data flow are maintained; benchmark and calibration numbers near the end are
> dated snapshots and should be reproduced before use in a new report.

This document introduces the current `circuitopt/` solver stack. The code is a compact local implementation of an AT4000TG OTFT ECG AFE solver, calibrated against Cadence/Spectre behavior. It is intended as the first concrete backend of the broader local circuit optimization flow.

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

The implementation is intentionally small and self-contained. Its modules include
`__init__.py`, the CLI entry `__main__.py`,
calibration/PSF/Cadence-netlist helpers, shared diagnostics/profiling
modules, the main solver stack, an ML-surrogate layer (dataset builder, surrogate
training, screen-and-verify optimizer), three silicon PDKs — SKY130 (via an
OpenVAF/OSDI bridge), FreePDK45 (via a flat-card loader and native Berkeley
BSIM4.5), and TSMC28HPC+ (via the internal HSPICE parser and the same native backend) — plugged into the
same `TransistorModel` interface as the original AT4000TG OTFT model, and an
optional local HTTP service layer (`circuitopt/service/`) over the whole stack.

## File Structure

```text
circuitopt/
  topology.py          Circuit topology source of truth.
  compiled_topology.py Runtime-compiled topology/index/stamp metadata.
  circuit_loader.py    JSON circuit description loader.
  device_model.py      TransistorModel ABC + NumbaParams + model factory/registry + PDK/polarity layer.
  device_factory.py    Device build/resolve layer (build_devices, get_ss_params) + corner routing
                        (OTFT CORNERS, silicon apply_silicon_corner). Leaf module: depends
                        only on device_model.
  pmos_tft_model.py    AT4000TG PMOS-OTFT compact-model implementation.
  numba_kernels.py     Optional Numba-accelerated scalar kernels.
  ac_mna.py            MNA stamping primitives.
  ac_solver.py         Pure AC small-signal solver: DC operating point + AC response (ac_solve).
  dc_solver.py         DC solve fallback (bounded least squares) + AFE-specific symmetric DC
                        seeding/continuation heuristics.
  noise_solver.py      Noise propagation and input-referred noise analysis.
  transient_solver.py  Time-domain transient solver.
  transient_profile.py Shared transient/chopper analysis counter slots.
  pss_solver.py        Transient-shooting periodic steady-state solver.
  pac_solver.py        Generic PSS-assisted PAC solver.
  pnoise_solver.py     Generic PNoise solver (HB + TD adjoint).
  adaptive_config.py   Shared adaptive-step configuration types and helpers.
  analysis_dispatch.py JSON analysis-configuration dispatch entry point.
  analysis_options.py  Central solver-option registry for JSON dispatch.
  diagnostics.py       Thread-safe solver-fallback observer (counters + logging).
  psf.py               PSFASCII parser for Spectre reference data.
  calibration.py       Local-vs-Cadence calibration comparison helpers.
  cadence_netlist.py   Spectre netlist generation helpers for validation runs.
  chopper.py           Ideal and PMOS-switch differential chopper analyses.
  explore.py           Design-space exploration / optimization driver.
  corners.py           Process corners, mismatch MC, and latch detection.
  dataset.py           Labeled surrogate-training dataset builder (provenance + failure-retaining).
  surrogate.py         Baseline metric surrogate (GBT via optional scikit-learn) + region-of-interest filtering.
  surrogate_torch.py   Differentiable surrogate (torch/MPS) + gradient-based design optimization.
  optimize.py          Screen-with-surrogate / Pareto-select / verify-with-solver optimization loop.
  osdi_host.py         OSDI 0.4 ctypes host — loads a compiled Verilog-A (.osdi) model, single-device DC/AC/noise eval.
  osdi_device.py       TransistorModel adapter over an OSDI-hosted compact model (bridges any OSDI PDK into the solver stack).
  osdi_transient.py    Numba fixed-grid/adaptive transient with direct OSDI ABI calls.
  sky130_model.py      SKY130 nfet/pfet PDK: BSIM4 param-card extraction (via ngspice) + PDK registration.
  ngspice_char.py      Model-card evaluator: batch ngspice .dc/.noise characterization → cached (Vsb,Vds,Vgs) grid.
  ngspice_device.py    TransistorModel over a cached ngspice grid (interpolated Id/gm/gds/caps/noise; extract_w + temperature).
  ngspice_process.py   Process-adapter protocol: deck preamble, instance syntax, op vectors, simulator flags.
  freepdk45_model.py   FreePDK45 compatibility exports + explicit ngspice oracle registration.
  pdk/freepdk45/       Flat-card loader and native FreePDK45 nmos/pmos adapter.
  compact_models/bsim4/ Native Berkeley BSIM4 host, ABI, Numba marshal, and transient.
  tsmc28_model.py      TSMC28HPC+ core nmos/pmos binding: nch_mac/pch_mac + HSPICE library closure.
  service/             Optional local FastAPI HTTP service layer (the `serve` extra) — see below.
    __init__.py        Re-exports CLI glue only; never imports fastapi (import circuitopt stays fastapi-free).
    app.py             create_app() — /api/v1 routes (health/capabilities/validate/solve/jobs/*); thin adapter, no numerics.
    jobs.py            JobManager — in-process thread-pool background jobs (explore/mc) with progress queue + cooperative cancel.
    serialize.py        to_jsonable()/serialize_results() — numpy/complex/NaN → strict-JSON conventions.
    cli.py              add_cli_args()/run_cli() — shared `serve` subcommand argument wiring (lazy fastapi/uvicorn import).
```

## Import Relationship

```text
topology.py          <- no internal dependency
compiled_topology.py <- no internal dependency; consumes Topology-like objects at runtime
circuit_loader.py    <- topology
numba_kernels.py     <- no internal dependency; optional numba at runtime
device_model.py      <- no internal dependency (abc, dataclasses only)
device_factory.py    <- device_model only (leaf device layer; no solver/workflow imports)
pmos_tft_model.py    <- optional numba_kernels, device_model
ac_mna.py            <- no internal dependency
ac_solver.py         <- device_factory, dc_solver, topology, compiled_topology, diagnostics
dc_solver.py         <- device_factory, topology, diagnostics
noise_solver.py      <- device_model, ac_mna, ac_solver, device_factory, topology, compiled_topology, diagnostics
transient_solver.py  <- adaptive_config, topology, ac_solver, device_factory, transient_profile, compiled_topology, numba_kernels, diagnostics; lazily imports osdi_transient for OSDI-backed devices
transient_profile.py <- no internal dependency (counter slot constants)
pss_solver.py        <- ac_mna, ac_solver, device_factory, adaptive_config, topology, transient_solver, diagnostics
pac_solver.py        <- ac_mna, ac_solver, device_factory, numba_kernels, topology, transient_solver, diagnostics
pnoise_solver.py     <- ac_mna, device_factory, noise_solver, numba_kernels, pac_solver, diagnostics
adaptive_config.py   <- no internal dependency (dataclass only)
analysis_dispatch.py <- ac_solver, noise_solver, transient_solver, pss_solver, pac_solver, pnoise_solver, circuit_loader, analysis_options
analysis_options.py  <- no internal dependency (registry)
diagnostics.py       <- no internal dependency (thread-safe counters)
psf.py               <- no internal dependency
calibration.py       <- psf, ac_solver, adaptive_config, noise_solver
cadence_netlist.py   <- circuit_loader, topology
chopper.py           <- ac_solver, dc_solver, device_factory, adaptive_config, device_model, noise_solver, pac_solver, pnoise_solver, pss_solver, topology, transient_solver
explore.py           <- ac_solver, device_factory, device_model, noise_solver, circuit_loader, diagnostics
corners.py           <- ac_solver, device_factory, noise_solver, topology, diagnostics
dataset.py           <- diagnostics, circuit_loader, corners, device_model, device_factory, explore, transient_solver
surrogate.py         <- no internal dependency; optional scikit-learn/joblib at runtime
surrogate_torch.py   <- dataset (CLI only); optional torch at runtime
optimize.py          <- surrogate, circuit_loader, dataset, explore
osdi_host.py         <- no internal dependency; ctypes + numpy only
osdi_device.py       <- device_model, osdi_host (lazy import)
osdi_transient.py    <- diagnostics, numba_kernels, osdi_host, compiled_topology (calls the OsdiDevice interface generically; does not import transient_solver)
sky130_model.py      <- device_model, osdi_device
ngspice_char.py      <- no internal dependency; ngspice subprocess + numpy only
ngspice_device.py    <- device_model, ngspice_char; optional scipy at runtime
ngspice_process.py   <- device_model
freepdk45_model.py   <- pdk/freepdk45, device_model, ngspice_device
pdk/freepdk45/*      <- spice parser, compact_models/bsim4, device_model, toolchain
tsmc28_model.py      <- device_model, ngspice_device, ngspice_process, toolchain
service/app.py       <- analysis_dispatch, analysis_options, circuit_loader, device_factory, device_model,
                        freepdk45_model, service/jobs, service/serialize; optional fastapi/pydantic at import time
service/jobs.py      <- explore, corners, service/serialize; no fastapi (pure threading/queue)
service/serialize.py <- no internal dependency; numpy only
service/cli.py       <- service/app (lazy); optional uvicorn at runtime
```

The `service/` subpackage is a pure *consumer* leaf — nothing outside it imports
back into it, and `circuitopt/__init__.py` never imports it, so plain
`import circuitopt` stays fastapi-free even when the `serve` extra is installed.

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
- **Backend-capability class attributes** — generic solvers dispatch on *capabilities*, never on a concrete backend type (no `isinstance(dev, OsdiDevice)`). `HAS_TERMINAL_LINEARIZATION` (default `False`) advertises the full quasi-static 4×4 terminal `(G, C)` stamp used by the periodic PAC/PNoise linearizer; `OsdiDevice` overrides it to `True`. `TRANSIENT_BACKEND` (default `None`, meaning the generic OTFT numba transient path) names a specialised integrator to route to instead; `OsdiDevice` sets it to `"osdi"`, which `transient_solver.py` reads to route to `circuitopt.osdi_transient.transient_osdi`.
- **`register_model()` / `create_device()` + PDK/polarity layer** — factory + registry. Each `(pdk, polarity)` pair registers under a structured key `"<pdk>.<polarity>"` (e.g. `"at4000tg.pmos"`); `register_pdk()` groups one process's polarities and marks the default. Solver files call `create_device(get_default_model_type(), …)` — a single switch point — instead of hardcoding a model name, so a new process or an `nmos` polarity slots in with one `register_pdk` call and no solver edits. `"pmos_tft"` stays a back-compat alias. `get_model_class(model_type)` is a public read-only registry accessor so solvers can inspect a model's capability flags without importing a concrete backend class. `registered_models()` returns a read-only `{model_type: "module.QualName"}` snapshot of the whole registry (insertion order) for a caller that needs to *enumerate* rather than look up one entry — the service layer's `GET /api/v1/capabilities` uses it to list every selectable model key. Generic elements (R/C/ideal V/I/controlled sources) are process-independent topology primitives and are **not** in this registry, so every PDK reuses them unchanged. `register_model()` still *replaces* an existing entry on re-registration (intentional swap-in, e.g. a test stub, keeps working silently), but a genuine collision — a different class (by `__module__.__qualname__`) taking over an already-occupied name, e.g. two PDK modules racing for the same alias — now emits a `RuntimeWarning` before overwriting; a repeat import or `importlib.reload` of the *same* class stays silent.

### `device_factory.py`

The leaf device-build layer: it depends only on `device_model` (never on any solver or workflow
module), so every solver can import it without risking a cycle.

- **`build_devices(sizes, *, nf=None, corner=None, topo, model_types=None, device_kwargs=None)`** /
  **`get_ss_params(...)`** — turn the per-device inputs a solver already carries (sizes, NF, corner,
  `model_types`, `device_kwargs`) into concrete `TransistorModel` instances, migrated here from
  `ac_solver.py` unchanged. `dev_corner`/`dev_nf`/`is_per_device_corner` are the small per-device
  corner/NF resolution helpers behind them.
- **`CORNERS`** — the OTFT continuous-PVT global process shift dict (`typical`/`slow`/`fast` as
  `pvt0`/`pbeta0`), migrated here from `corners.py`; `corners.py` now imports it from here instead of
  defining it.
- **`SKY130_CORNERS`** / **`SILICON_CORNERS`** / **`apply_silicon_corner(model_types, device_kwargs,
  corner)`** — the silicon discrete-corner routing that stamps a corner name (`tt`/`ss`/`ff`/`sf`/`fs`
  for SKY130, plus `nom` for FreePDK45) onto extracted silicon device cards, migrated here from
  `explore.py`; `explore.py` and `dataset.py` now import it from here instead of defining it.
- **`CircuitBinding`** — a frozen dataclass bundling the six per-circuit inputs every solver used to
  thread by hand: `topo` / `model_types` / `device_kwargs` / `nf` / `corner` / `dc_seed`. It exists to
  close a bug class: dropping `model_types`/`device_kwargs` on the way into a solver silently reverted
  the circuit to the default OTFT PDK. A caller now passes `binding=` once instead of re-plumbing the
  cluster. Build one from `CircuitSpec.binding()` (see `circuit_loader.py`). `binding.build(sizes)`
  materializes `{name: TransistorModel}` for those sizes; `binding.at_corner(corner)` returns a binding
  routed to a corner — a silicon corner is baked onto `device_kwargs` (solver corner cleared, via
  `apply_silicon_corner`), an OTFT corner stays on `binding.corner`, and `None` returns `self`.
  **Resolution priority** (via `resolve_binding`): an explicit non-`None` keyword to a solver always
  wins; otherwise the binding field supplies the default (`binding.dc_seed` backs `x0_guess`); with
  `binding=None` the legacy kwargs path is byte-identical. All six solver entry points
  (`ac_solve`/`noise_analysis`/`transient`/`pss_solve`/`pac_solve`/`pnoise_solve`) accept `binding=`,
  and the internal workflows — `analysis_dispatch.run_analysis_suite`, `explore`, `dataset`,
  `optimize` — thread one binding rather than re-forwarding the model cluster to each branch.

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

It also hosts the small marshalling helpers `term_arrays()` (splits `(kind, ref_or_value)` terminal
tokens into parallel `kind`/`ref`/`value` int/float arrays) and `index_array()` (packs optional
integer indices into an int64 array, `None` → `-1`). Both `transient_solver.py`'s raw-transient marshal
and `osdi_transient.py`'s OSDI transient marshal build the same stamp-ready arrays from these, so the
helpers live with the topology tokens they share instead of being duplicated per backend.

### `circuit_loader.py`

Loads JSON circuit descriptions and returns a `CircuitSpec` containing:

- `topology`
- `sizes`
- `bias`
- `nf`

This lets new circuits be added through JSON files such as `examples/single_stage.json` without editing solver source code. `CircuitSpec.binding()` bundles the spec's `topology`, `model_types`, `device_kwargs`, `nf`, and default DC seed (its first dict `dc_guess`) into a `CircuitBinding` (see `device_factory.py`), so a workflow can pass `binding=` to the solvers instead of threading the whole cluster.

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

The solver path uses Numba automatically whenever it is available; no module
sets `CIRCUIT_USE_NUMBA=1` at import time (set `CIRCUIT_USE_NUMBA=0` to opt out).

The flag is **baked at import time**: `USE_NUMBA`/`NUMBA_AVAILABLE` are fixed
constants computed once when `numba_kernels` is first imported, so setting the
env var afterwards is a silent no-op. `circuitopt/__init__.py` pre-scans `sys.argv`
for `--no-numba` and sets `CIRCUIT_USE_NUMBA=0` *before* its solver imports pull
in `numba_kernels` transitively — under `python -m circuitopt …` this `__init__` runs
before `__main__.py`, so the CLI flag takes effect. Each `__main__.py`
subcommand handler then calls `_assert_numba_flag(args)`, which raises
`SystemExit` if `--no-numba` was requested but `numba_kernels.USE_NUMBA` is
still `True` — turning a defeated pre-scan (e.g. calling `main()` programmatically
after something already imported a solver module) into a loud failure instead
of a silently-ignored flag. Disabling Numba from Python code (not the CLI) must
set `CIRCUIT_USE_NUMBA=0` **before** `import circuitopt`.

At present, the accelerated paths are PMOS current evaluation, internal-node
Newton iterations, bias-dependent capacitance evaluation, AC/PNoise
terminal-derivative small-signal parameters, PNoise HB block assembly/noise fold,
and the transient Newton inner loop: topology token lookup, PMOS state solve,
residual/Jacobian stamping, and the small dense Newton solve. The dense Newton
solve uses an in-place `A*x = -R` path to avoid avoidable per-iteration copies.
If the compiled path cannot handle a step, `transient_solver.py` falls back to
the original Python Newton / full-Jacobian / least-squares path.

### `analysis_options.py` / `analysis_dispatch.py`

`analysis_options.py` is the central per-analysis option registry that
`analysis_dispatch.py`'s `run_analysis_suite` (and the JSON schema regression
test) both derive from, so solver kwargs/defaults/schema cannot silently drift
apart. `validate_analysis_cfg(analysis, cfg)` rejects residual keys in a JSON
`analyses` block: `known_keys(analysis)` unions the solver's option registry
with `DISPATCH_KEYS` (the handful of keys — e.g. `freqs`/`corner`/`band` for
`ac`/`noise`, `signed_devices` for `transient` — that `run_analysis_suite`
reads directly out of `cfg` rather than forwarding into `solver_kwargs`); any
key outside that union raises `ValueError` naming the analysis, the offending
key(s), and the sorted list of valid keys. This turns a typo (e.g.
`max_sidebands` for `max_sideband`) into an immediate error instead of a
silently-ignored option running with its default.

`ANALYSIS_ORDER = ("ac", "noise", "transient", "pss", "pac", "pnoise")` in
`analysis_dispatch.py` is the canonical analysis-name tuple and execution order;
`run_analysis_suite` iterates it, and the service layer's `GET
/api/v1/capabilities` builds its `analyses` map by iterating the same tuple, so
the two surfaces list identical analysis names with no separate hardcoded list.

### `psf.py`

`provenance(path)["fundamental"]` reads the PSF HEADER's `"fundamental
frequency"` key (falling back to the bare `"fundamental"` spelling for any
non-standard writer) — periodic analyses (PAC/PNoise/PSS) report their real
drive frequency; DC/AC/noise/tran carry neither key and read back `None`.
`parse_noise(path)`'s per-device noise arrays are **ragged**: the column count
follows each device's TYPE-declared struct width (a MOSFET's `(flicker,
thermal, total)` struct is width 3; a resistor's `(rn, total)` struct is width
2), so callers must read the *last* column (`[:, -1]`) for the total
contribution and check `.shape[1]` before slicing a specific field — never
assume width 3.

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

The DC solve includes robustness handling for physical branch selection, symmetric operating points,
and rail-bounded node solutions — implemented in `dc_solver.py` and called from here. `ac_solver.py`
itself is now a pure AC small-signal module: `ac_solve` plus the `bw_from_gain` bandwidth helper. It
no longer carries the device-factory or DC-seeding code it used to.

### `dc_solver.py`

DC operating-point solving support that used to live inline in `ac_solver.py`, split out because it
mixes two different concerns:

- **`bounded_least_squares_dc(...)`** / **`dc_residual_ok(...)`** — a generic last-resort DC solve:
  a bounded least-squares fallback used when the primary Newton/`fsolve` path fails to converge, plus
  the residual-acceptance check that gates it.
- **`symmetric_seed(...)`** / **`symmetric_continuation(...)`** / **`is_afe_topology(...)`** /
  **`is_pairwise_symmetric_afe(...)`** / **`_AFE_SYMMETRIC_PAIRS`** — a circuit-specific seeding
  heuristic for the AFE topology only, **not** general solver logic. It selects the physical
  (Spectre-matching) symmetric power-up branch for that one circuit. Keeping it in its own module
  keeps `ac_solver.py`'s generic solve free of per-circuit branches.

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
  `circuitopt.pac_solver.pac_solve(...)`. The generic solver still defaults to the
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
  `circuitopt.pnoise_solver.pnoise_solve(...)`. For chopper verification it defaults to
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
  The returned `method` is `pss_time_domain_floquet_adjoint` on this path and
  `pss_harmonic_balance_conversion_matrix` on the HB fallback path.
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
  Production paths only support `charge`/id 0 and `average`/id 1. `None` uses
  the module/environment default (`charge`), while chopper PSS passes the
  `average` operator explicitly for Cadence-matched clock feedthrough. This
  override affects only the transient/PSS orbit, not PAC/PNoise conversion
  linearization.
- `adaptive=True` is an opt-in LTE-controlled gear2 path. It treats the supplied
  `tgrid` as the input-sampling grid and `[t0, tstop]` boundary, then returns a
  self-chosen non-uniform accepted grid. It is valid only with
  `integration_method="gear2"`. Public callers may still pass
  `adaptive_reltol`, `adaptive_vabstol`, `adaptive_iabstol`,
  `adaptive_max_steps`, and `adaptive_h0`; internally these are normalized into
  `AdaptiveConfig` and shared by transient/PSS/chopper paths. PSS additionally
  uses `adaptive_freeze_factor` from the same config.
  The adaptive LTE policy constants and Python helpers live in
  `circuitopt/adaptive_config.py`; Numba keeps a compiled mirror for performance, with
  tests checking the two implementations agree. Newton failure rejects now shrink
  the candidate step, while zero-error accepted steps may grow it.
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
- Results export to CSV and JSONL; a CLI runs `python -m circuitopt.explore <config.json>`.
- Silicon corner routing (`SKY130_CORNERS`/`SILICON_CORNERS`/`apply_silicon_corner`) now lives in
  `device_factory.py`; `explore.py` imports it from there rather than defining it.
- `add_cli_args(parser)` / `run_cli(args)` are the single source of the CLI argument definitions —
  both the `python -m circuitopt explore` subcommand and the standalone `python -m circuitopt.explore` entry
  point call the same two functions, so the two surfaces cannot drift apart (see
  [`cli_reference.md`](cli_reference.md)).
- `explore_from_dict(data, n=, seed=, method=, corner=, progress=None, should_stop=None)` — the
  single shared entry point for the `explore` subcommand and the service layer's
  `POST /api/v1/jobs/explore`: parses the `explore` block, binds any silicon `models`, and calls
  `explore()`. `progress(done, total)` / `should_stop()` are optional hooks threaded straight
  through to `explore()` for a caller that needs live progress or cooperative cancellation (e.g.
  the service's background-job manager, see `service/jobs.py`); both default to `None`, in which
  case behavior is byte-identical to the pre-hook code path. On early stop, `results["stopped_early"]`
  and `results["summary"]["stopped_early"]` are `True` and `summary["evaluated"]` records how many
  candidates actually ran (`summary["n"]` stays the originally requested count).

Example configs: `examples/afe_explore.json` and `examples/single_stage.json`;
both are full circuit JSON files with an `explore` block.

### `corners.py`

Single source of truth for process-corner and robustness *work* — the pieces that
otherwise get re-derived in every sweep. The `CORNERS` data itself (global process shifts
`typical` / `slow` / `fast` as `pvt0`/`pbeta0`, from the PDK monte.scs sections; e.g.
slow = `{"pvt0": -0.2259, "pbeta0": -0.54}`) now lives in `device_factory.py`, the shared
leaf device layer; `corners.py` imports it from there (`from .device_factory import CORNERS`)
so existing `from circuitopt.corners import CORNERS` call sites keep working unchanged. What stays
in `corners.py` is the mismatch/latch machinery built on top of it:

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
  non-latched samples included in the final noise statistics. Accepts optional
  `progress(i, n, partial)` / `should_stop()` hooks (default `None`, byte-identical result to the
  pre-hook code path): `progress` fires after each sample with a lightweight running summary,
  `should_stop` is checked before each sample and, if it returns `True`, ends the run early with
  `"stopped_early": True` on both the top level and `summary` — the count actually evaluated is
  `summary["n"]` (this path has no separate "requested n" field; unlike `explore`'s
  `summary["evaluated"]`, `mismatch_mc`'s `summary["n"]` always reflects samples actually run).
- `mismatch_mc_from_dict(data, n=, seed=, corner="typical", progress=None, should_stop=None)` — the
  shared entry point for the `mc` subcommand and the service layer's `POST /api/v1/jobs/mc`; parses
  the circuit and calls `mismatch_mc()` so the two surfaces can't drift.

`ac_solve` / `noise_analysis` accept the same `corner` argument (a flat process dict or a
per-device mismatch map). The driver `examples/mc_mismatch.py` wraps this into a corner
table + 3-corner MC figure.

### ML surrogate layer (`dataset.py` / `surrogate.py` / `surrogate_torch.py` / `optimize.py`)

Turns the validated solvers into a full **build dataset → train surrogate → optimize →
verify** loop. The solvers stay the ground truth throughout; the surrogate only
accelerates the *screening* of a large candidate pool.

- **`dataset.py`** — like `explore.py`'s sampling/evaluation but with **no** constraint
  or Pareto filtering and **always** evaluates noise, so every sample (including DC
  failures) becomes a labeled training row. Writes `.jsonl` (human-debuggable rows) +
  `.manifest.json` (provenance: schema version, solver git commit/dirty flag, topology
  hash, PDK, `models` binding, corner, sampling seed/method, variable ranges — so a
  consumer can reject out-of-domain designs) + `.npz` (dense `X`/`Y` matrices, NaN where
  a label is missing) + optional `.parquet`. **Label groups** (`--labels`, opt-in beyond
  the default `ac_noise`): `transient` (stimulus-agnostic waveform features from the
  config's validated periodic transient), `pss` (periodic-steady-state quality + orbit
  output), `pac` (baseband conversion gain + PAC-grid −3 dB corner) and `pnoise`
  (band-integrated output / input-referred periodic noise — the chopper figures of
  merit). The `pss`/`pac`/`pnoise` groups run one shared `run_analysis_suite` chain per
  candidate, so the config's validated `analyses` solver settings (`time_domain`,
  drive, band, shooting tolerances) apply exactly; `pac`/`pnoise` require their
  `analyses` blocks. **Design-axis grammar** extends beyond `DEV.W/.L/.NF`/bias:
  `<Cap>.C` / `<Res>.R` (named passive values — *structural*, rebuilds the circuit per
  candidate via `candidate_circuit()`), `periodic.frequency` (clock), and `pvt0`/`pbeta0`
  (continuous global process shift — sampling this turns the discrete corner sweep into
  a continuous-PVT training axis). Like `explore.py`, `dataset.py` imports silicon corner
  routing (`SKY130_CORNERS`, `apply_silicon_corner`) from `device_factory.py` rather than
  defining it, and exposes the same `add_cli_args(parser)` / `run_cli(args)` pair so the
  `python -m circuitopt dataset` subcommand and the standalone `python -m circuitopt.dataset` entry
  point share one argument definition.
- **`surrogate.py`** — `HistGradientBoostingRegressor` (optional `scikit-learn`
  dependency) trained per label, with automatic log-space fitting for labels spanning
  multiple decades (e.g. IRN). `filter_rows()` / CLI `--filter label:lo:hi` restricts
  training to a region of interest (e.g. drop railed/collapsed designs whose extreme
  labels would dominate a squared-error fit, and which get screened out by a constraint
  anyway). `score()` reports median/P95 relative error and R² per label.
- **`surrogate_torch.py`** — a differentiable MLP surrogate (optional `torch`
  dependency; MPS-capable on Apple Silicon) for gradient-based multi-objective design
  optimization with constraint penalties, plus a `--verify` pass back through the solver.
- **`optimize.py`** — the screen-and-verify payoff: predict a large pool (µs/candidate)
  with the surrogate, take the constrained Pareto front, then re-evaluate the top-K on
  the actual calibrated solver. Uses `dataset.candidate_circuit()`/`split_variables()` so
  every variable kind (including structural cap/resistor/clock axes) is honored in the
  verify pass, not just size/bias.

A key operational lesson: no single surrogate is both a
precise *region-of-interest* model and a good *failure-region-aware screener* — training
on the operating region (via `--filter`) gives the tightest metric accuracy but makes the
surrogate blind to railed designs during screening. The screen-and-verify architecture
tolerates this by design: the solver, not the surrogate, has the final word on feasibility.

### Silicon PDK / OSDI layer (`osdi_host.py` / `osdi_device.py` / `osdi_transient.py` / `sky130_model.py`)

Plugs a **second, industry-standard device physics model (BSIM4)** into the same
`TransistorModel` interface the AT4000TG OTFT model implements — so any bulk-BSIM4
PDK (SKY130 today) runs through the *same* DC/AC/noise solver engine, additively
(`default=False`; the OTFT PDK is untouched and remains byte-identical).

A **third PDK, FreePDK45**, now uses the same native Berkeley BSIM4.5 kernel as
TSMC28HPC+. `pdk/freepdk45/library.py` parses each flat level-54 VTG card into a
numeric model card; `pdk/freepdk45/device.py` exposes four-terminal current,
conductance, charge, capacitance, and correlated noise through
`freepdk45.nmos` / `.pmos`. The source cards declare `version=4.0`; the bundled
Berkeley source treats that field as metadata, and regression tests compare
single-device operating points/noise plus a 5T OTA against ngspice.

The native host reduces the complete FreePDK45 internal drain/source, gate, and
body-resistance network with a small pivoted linear solve. Four-terminal charge
aggregation includes distributed body-junction charge and normalizes PMOS charge
signs, so AC capacitance and finite-difference charge derivatives agree for both
polarities. DC, AC, noise, transient, PSS, PAC, and PNoise therefore share one
in-process compact-model path.

The native host also exports a versioned conserved-evaluation ABI, an all-`void *`
entry point, and a batch evaluator. The fixed-grid BSIM4 transient keeps matrix
assembly, Newton iteration, and time stepping inside Numba while calling the C
compact model through a runtime ctypes function pointer. Disabling Numba retains
the Python implementation as a reference path.

- **`pdk/freepdk45/library.py`** — portable card resolution, strict corner and
  polarity validation, numeric parsing, and path/mtime/size caching.
- **`pdk/freepdk45/device.py`** — native `TransistorModel` adapter and default
  `freepdk45.*` registration.
- **`freepdk45_model.py`** — compatibility exports and the optional historical
  `freepdk45_ngspice.*` grid aliases.
- **`ngspice_char.py` / `ngspice_device.py` / `ngspice_transient.py`** — retained
  external regression-oracle infrastructure, not the default FreePDK45 runtime.

### TSMC28HPC+ native adapter (`spice/` / `compact_models/bsim4/` / `pdk/tsmc28/`)

The internal HSPICE frontend resolves nested `.lib`/`.include` closures, parameters,
expressions, subcircuits, foundry MOS macros, and geometry bins. The TSMC28 library
layer selects the 0.9 V `nch_mac` / `pch_mac` core model for each instance and sends
the expanded model and instance parameters to the native Berkeley BSIM4.5 backend.

The native device exposes four-terminal currents, charges, conductance, capacitance,
and correlated noise. DC, AC, noise, transient, PSS, PAC, and PNoise therefore run
without an ngspice subprocess. The old process adapter remains registered only under
the explicit `tsmc28hpcp_ngspice.*` model names as an independent regression oracle.

The default portable model entry is
`PDK/tsmc28hpcp/models/hspice/cln28hpcp_1d8_elk_v1d0_2p2.l`, which is Git-ignored.
Resolution priority is `TSMC28_MODEL_DIR`, `TSMC28_PDK_ROOT`, that project-local
entry, then `PDK_ROOT/tsmc28hpcp`. See [TSMC28HPC+ Local Adapter](tsmc28hpcp.md).

- **`osdi_host.py`** — a ctypes host for the **OSDI 0.4 ABI**, the simulator-independent
  C interface [OpenVAF](https://github.com/pascalkuthe/OpenVAF) compiles Verilog-A
  compact models into (`.osdi`, a native shared library). `load_osdi()` introspects the
  descriptor (nodes/params/opvars) with a struct-size self-check; `Device` sets model/
  instance params via the ABI's `access()`, replicates the simulator-side node collapse,
  runs an internal-node Newton (gmin-regularized for DC-floating internal nodes), and
  exposes `operating_point()` (Id/gm/gds/gmb/capacitances via a Schur complement over
  internal nodes — BSIM4 exposes zero opvars, so small-signal quantities come from the
  Jacobian) and `noise_psd()`. This is a **single-device** DC/AC/noise evaluator; the
  circuit-level MNA/Newton is still owned by the existing `ac_solver`/`noise_solver`.
- **`osdi_device.py`** — `OsdiDevice(TransistorModel)` wraps a `Device`, implementing
  `get_Idc`/`get_ss_params`/`get_capacitances`/`get_noise_psd`. `TransistorModel.kcl_sign`
  (default +1, i.e. source-high — matching PMOS/OTFT) lets `ac_solve`'s DC KCL support
  NMOS (source-low, `kcl_sign=-1`) without changing the OTFT path (byte-identical:
  `1.0 * abs(x) == abs(x)`). `OsdiDevice` overrides the base capability class attributes:
  `HAS_TERMINAL_LINEARIZATION = True` (it provides `get_terminal_linearization`) and
  `TRANSIENT_BACKEND = "osdi"`. The generic OTFT transient-only ABC hooks remain
  separate; OSDI transient uses its own ABI-aware Numba path.
- **`osdi_transient.py`** — `transient_osdi(sizes, bias, tgrid, ...)` is the circuit-level
  entry point. Its fixed-grid and adaptive kernels call OSDI function pointers directly
  inside Numba and include external plus device-internal dynamic nodes in one global
  Newton system. The lower-level `cs_transient()` remains a readable reference. `transient()`
  in `transient_solver.py` checks the device's `TRANSIENT_BACKEND` class attribute and, when
  it is `"osdi"`, lazily imports and routes to `transient_osdi` — a one-way dependency.
  `osdi_transient.py` itself never imports `transient_solver.py`, so the two modules no
  longer form a circular import (they did before this split).
- **`sky130_model.py`** — `Sky130Nfet`/`Sky130Pfet(OsdiDevice)` + `register_pdk("sky130",
  ...)`. SKY130's binned BSIM4 subcircuits (63 bins, 2000+ `.param` expressions) are
  resolved by **letting ngspice do it**: instantiate the subckt, run an `op`, `showmod`
  dumps the fully-resolved flat card (731 params), which is cached under
  `data/pdk/sky130/*.json` and fed to the OpenVAF-compiled `bsim4va`. `EXTRACT_W`/
  `extract_w`: resolve the card once at a reference width and let `bsim4va` scale the
  actual W — avoids a per-candidate ngspice subprocess during a design sweep
  (~2 ms/eval instead). Oracle: **local ngspice loading the same `.osdi`** — since both
  the solver and the oracle run the identical compiled model, correctness is *model==
  oracle* regardless of the SKY130-vs-VA BSIM4 version gap (SKY130's ngspice built-in is
  4.5; the VA source is 4.8 — a realistic 130 nm process, not SkyWater's bit-exact
  sign-off model, which is the right tradeoff for optimizer generalization).

`circuitopt/circuit_loader.py`'s optional `models` block (`{"M1": {"type": "sky130.nmos",
...}}`) binds specific devices in a JSON circuit to a non-default PDK, so a mixed
OTFT+silicon (or all-silicon) circuit is just configuration — see
[JSON Circuit Description](json_circuit_format.md).

### Local service layer (`service/app.py` / `jobs.py` / `serialize.py` / `cli.py`)

An **optional** local FastAPI HTTP layer over the whole solver stack, gated on the
`serve` extra (`pip install -e ".[serve]"`). It is a thin adapter — every route hands
a request straight to an existing single source of truth and carries no numerical
logic of its own. Full endpoint reference: [Service API](service_api.md).

- **`app.py`** — `create_app(job_workers=1) -> FastAPI` builds the `/api/v1` app:
  `GET health`/`capabilities`, `POST validate`/`solve` (synchronous, calling
  `circuit_from_dict`/`validate_analysis_cfg`/`run_analysis_suite` directly), and the
  `jobs/*` background-task routes (`POST jobs/explore`/`jobs/mc`, `GET jobs`/`jobs/{id}`,
  `DELETE jobs/{id}`, `WS jobs/{id}/events`) backed by `jobs.JobManager`. CORS is
  restricted to `localhost`/`127.0.0.1` on any port. `pydantic` request models
  (`SolveRequest`/`ExploreJobRequest`/`McJobRequest`) wrap `circuit` as an opaque
  `dict` — the circuit schema's single source of truth stays `circuit_from_dict`,
  never re-described here.
- **`jobs.py`** — `JobManager`/`Job`: an in-process `ThreadPoolExecutor`-backed
  background-job table for the two long-running drivers, `explore_from_dict` and
  `mismatch_mc_from_dict`. State machine `queued -> running -> {done, failed,
  cancelled}`; progress is pushed onto a per-job `queue.Queue` (drained by the
  WebSocket route) and also cached on `Job.progress` for polling; cancellation sets a
  `threading.Event` consumed as the `should_stop` callback passed into the core driver
  (cooperative — an in-flight candidate/sample always finishes first). Retains at most
  `MAX_JOBS` (50) jobs, evicting the oldest already-terminal one first. Imports no
  `fastapi` — pure threading/queue plumbing, unit-testable standalone.
- **`serialize.py`** — `to_jsonable()`/`serialize_results()`: the numpy/complex/NaN →
  strict-JSON conventions shared by every response (both the synchronous endpoints and
  the job/WebSocket payloads). NaN/±Inf → `null`, `complex` → `{"re", "im"}`,
  `numpy.ndarray` → nested `list`, `_`-prefixed dict keys and callables dropped.
- **`cli.py`** — `add_cli_args(parser)`/`run_cli(args)`, the single source of the
  `serve` subcommand's argument wiring (`--host`/`--port`/`--reload`/`--job-workers`),
  shared by the `circuit-opt serve` subcommand and the standalone
  `python -m circuitopt.service` entry point, mirroring the `explore`/`dataset`
  single-source CLI pattern. Imports `fastapi`/`uvicorn` lazily inside `run_cli`, so
  importing `circuitopt.service` (which `circuitopt/__main__.py` does eagerly for
  subcommand registration) never requires the extra to be installed.

## Quick Example

```python
import numpy as np

from circuitopt.ac_solver import ac_solve
from circuitopt.noise_solver import noise_analysis, band_rms
from circuitopt.transient_solver import transient

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

from circuitopt.circuit_loader import load_circuit_json
from circuitopt.ac_solver import ac_solve
from circuitopt.transient_solver import transient

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

The current solver stack was calibrated against Cadence Spectre 24.1 for the AT4000TG AFE use case. The observed agreement in the original project included:

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
