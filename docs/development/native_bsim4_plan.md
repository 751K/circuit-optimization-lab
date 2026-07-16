# Native TSMC28 BSIM4 Backend

## Objective

Run TSMC28HPC+ core NMOS/PMOS circuits through CircuitOpt's in-process analysis
engines without invoking ngspice at runtime.

The completed data path is:

```text
licensed HSPICE library
  -> circuitopt.spice parser and elaborator
  -> TSMC corner/subcircuit/bin selector
  -> numeric BSIM4.5 model and instance parameters
  -> in-process BSIM4 residual/Jacobian/charge/noise evaluator
  -> existing CircuitOpt DC/AC/noise/transient/PSS/PAC/PNoise solvers
```

The licensed model remains local and Git-ignored. Parsing and elaboration are
in-memory operations; no foundry parameter card is committed or copied into a
portable cache.

## Package Boundaries

```text
circuitopt/spice/
  parser.py          lexical and structural HSPICE parser
  expressions.py     deterministic, sandboxed HSPICE expression evaluator
  elaborator.py      .lib closure, parameter scopes, functions and subcircuits

circuitopt/compact_models/bsim4/
  abi.py             simulator-independent model/instance/result contracts
  native.py          binding to the independent BSIM4.5 numerical implementation
  device.py          TransistorModel adapter for CircuitOpt

circuitopt/pdk/tsmc28/
  library.py         local model resolution and parsed-library lifecycle
  core.py            nch_mac/pch_mac expansion and W/L bin selection
  registry.py        tsmc28hpcp native model registration
```

PDK syntax, compact-model numerics, and circuit solving are separate layers. The
TSMC layer must not call ngspice, and the BSIM layer must not contain TSMC paths
or model parameters.

## Supported Scope

The first complete target is the 0.9 V core `nch_mac` and `pch_mac` pair used by
the ADC/OTA flow:

- process corners: TT, SS, FF, SF and FS;
- temperature and supply sweeps handled by the existing circuit solvers;
- instance geometry: W, L, NF, multiplicity and threshold mismatch;
- DC currents and terminal Jacobian;
- complete quasi-static terminal charges and capacitance Jacobian;
- BSIM4 thermal, flicker and induced-gate noise required by circuit noise;
- transient state required by switched-capacitor MDAC simulation.

Diodes, BJTs, IO devices, reliability checks, statistical Monte Carlo semantics
and layout verification are outside the first core-MOS milestone. Unsupported
constructs must raise an explicit error; they must never silently fall back to
nominal values or ngspice.

## Milestones And Gates

### 1. Syntax

- Parse CRLF files, continuations, inline comments, `.lib/.endl`, `.param`,
  parameter functions, `.model`, `.subckt/.ends`, and element statements.
- Parse the installed TSMC library without writing model contents to disk.
- Gate: every parameter statement is represented, all core model cards and
  `nch_mac`/`pch_mac` are present, and malformed syntax reports source locations.

### 2. Elaboration

- Resolve same-file `.lib` dependencies in declaration order.
- Implement case-insensitive lexical scopes for global, section, subcircuit,
  model and instance parameters.
- Evaluate arithmetic, comparison and the HSPICE functions actually used by the
  delivery, including deterministic nominal behavior for statistical functions.
- Detect cycles, unknown symbols and unsupported simulator-dependent expressions.
- Gate: all parameters needed by one core NMOS and PMOS instance become finite
  numeric values at every process corner.

### 3. Macro And Bin Selection

- Expand `nch_mac`/`pch_mac` to their single internal MOS.
- Apply W/L/NF transformations and select exactly one geometry bin using the
  model-card limits.
- Gate: representative and boundary geometries select one and only one card;
  gaps and overlaps are explicit errors.

### 4. BSIM4.5 Numerics

- Use an independently built open-source/reference BSIM4.5 implementation. Do
  not link libngspice and do not call an ngspice executable.
- Expose terminal residuals, resistive/reactive Jacobians, charges, operating
  point data and noise through a small simulator-independent API.
- Gate: finite-difference Jacobian/charge-conservation tests pass and all outputs
  remain finite over the supported terminal-voltage/temperature domain.

### 5. Solver Integration

- Bind the native model to `TransistorModel`.
- Route all analyses to the existing CircuitOpt engines.
- Gate: tests run with an environment that hides/removes ngspice and still solve
  a transistor, inverter, differential pair and switched-capacitor transient.

### 6. Accuracy And Campaign

- Keep ngspice only as a development oracle while both implementations are
  available; store only non-proprietary numerical tolerances/results.
- Compare DC current, gm/gds, terminal capacitances, noise PSD and transient
  trajectories before circuit-level regression.
- Gate: the MDAC OTA 45-point campaign completes through the native backend and
  every remaining discrepancy is documented with an accepted tolerance or fix.

## Completion Definition

This work is complete only when:

1. setting `NGSPICE_BIN` to an invalid path does not affect native TSMC runs;
2. no native TSMC module imports `circuitopt.ngspice_*`;
3. all five process corners elaborate from the original local model library;
4. DC, AC, noise and transient use the native model and existing solver stack;
5. the representative circuit and OTA/MDAC regression suites pass;
6. unsupported PDK features fail loudly and are documented.
