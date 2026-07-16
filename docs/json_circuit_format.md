# JSON Circuit Description Format

[Project Overview](README.md) | [Core Solver Overview](module_overview.md) | [中文版](json_circuit_format_zh.md)

> **Status: maintained reference.** The loader and
> `schemas/circuit.schema.json` are the source of truth. This document describes
> the user-facing contract and should be updated with either one.

## Purpose

The JSON circuit description separates topology, sizes, bias, and analysis metadata
from Python source code. Swap circuits by changing JSON files instead of editing
`circuitopt/ac_solver.py`, `circuitopt/noise_solver.py`, or `circuitopt/transient_solver.py`.

Schema file:

```text
schemas/circuit.schema.json
```

Example files:

```text
examples/single_stage.json
examples/resistor_load_stage.json
examples/afe_explore.json
examples/periodic_rc.json
```

## Minimal Structure

A valid circuit JSON needs at minimum:

```json
{
  "solved": ["OUT"],
  "rails": {
    "VDD": "VDD",
    "GND": 0.0,
    "IN": "VIN"
  },
  "devices": [
    {"name": "M1", "drain": "OUT", "gate": "IN", "source": "VDD", "W": 2000, "L": 80}
  ],
  "bias": {
    "VDD": 40.0,
    "VIN": 25.0
  },
  "outputs": ["OUT"]
}
```

- `solved` — nodes the solver must find voltages for.
- `rails` — known-voltage nodes. Values can be numeric constants or keys in `bias`.
- `devices` — list of transistor devices (currently PMOS_TFT; model type is determined by the ``device_model`` factory, default ``"pmos_tft"``).
- `bias` — DC voltages for rail references.
- `outputs` — nodes observed for AC/noise/transient results.

## Field Reference

### `name`

Optional. Circuit name for display and logging.

```json
"name": "single_stage_pmos_load"
```

### `solved`

Required. Unknown node list; also defines the MNA/DAE vector ordering.

```json
"solved": ["VOP", "VON", "VFBP", "VFBN", "NET20", "NET2"]
```

- At least one node.
- No duplicate names.
- Every node in `outputs` must be in `solved`.

### `rails`

Required. Known-voltage node map.

```json
"rails": {
  "VDD": "VDD",
  "GND": 0.0,
  "VB": "VB"
}
```

- `"GND": 0.0` — GND is always 0 V.
- `"VDD": "VDD"` — VDD voltage is read from `bias["VDD"]`.
- Every node referenced by a device port must appear in `solved` or `rails`.

### `devices`

Required. Each active device is a three-terminal transistor (drain/gate/source). Model implementation is selected by the ``device_model`` factory (default ``"pmos_tft"``).

Object form (preferred):

```json
{
  "name": "M7",
  "drain": "VOP",
  "gate": "VCM",
  "source": "NET2",
  "W": 61365,
  "L": 61,
  "NF": 1
}
```

Array shorthand:

```json
["M7", "VOP", "VCM", "NET2"]
```

If using array form, W/L must be supplied in `sizes`.

### `sizes`

Optional. Per-device dimensions, useful for separating topology from sizing.

```json
"sizes": {
  "M7": [61365, 61],
  "M8": [61365, 61]
}
```

- If a device object already has `W` and `L`, `sizes` is not needed.
- If both are present, `sizes` overrides the embedded W/L.
- Every device must ultimately have W and L.

### `nf`

Optional. Number of fingers (multiplies drain current).

Global:

```json
"nf": 2
```

Per-device:

```json
"nf": {
  "M7": 4,
  "M8": 4
}
```

Device-level `NF` in the device object is overridden by top-level `nf` when both exist.

### `models`

Optional. Binds specific devices to a non-default PDK model type (e.g. silicon
SKY130) instead of the default `"pmos_tft"` (AT4000TG OTFT). Devices not listed here
use the default PDK — this is purely additive, so an OTFT-only config never needs it.

```json
"models": {
  "M1": {"type": "sky130.nmos", "extract_w": 24.0},
  "M3": {"type": "sky130.pmos", "vb": 1.8, "extract_w": 12.0}
}
```

- `type` — a model-registry key, `"<pdk>.<polarity>"` (e.g. `"sky130.nmos"`,
  `"sky130.pmos"`, `"freepdk45.nmos"`, `"tsmc28hpcp.nmos"`, `"at4000tg.pmos"`). See
  `circuitopt.device_model.register_pdk`.
- Remaining keys are forwarded to the device constructor. For SKY130 devices:
  `vb` (bulk bias, volts; default 0), `corner` (SKY130 process corner —
  `tt`/`ss`/`ff`/`sf`/`fs`; default `tt`), `extract_w` (µm — resolve the SKY130
  parameter card once at this reference width and let the compact model scale the
  actual `W`, avoiding a per-candidate re-extraction during a design sweep),
  `temperature` (kelvin; default 300.15), `NF` (int).
- **FreePDK45** (`"freepdk45.nmos"` / `"freepdk45.pmos"`) directly parses the
  flat BSIM4 level-54 cards and evaluates them with the in-process Berkeley
  BSIM4.5 backend. The cards declare `version=4.0`; this metadata field does not
  select a separate equation path in the bundled kernel, and native device/5T
  OTA results are regression-checked against ngspice. Device keys include `vb`
  (0 for NMOS, normally 1.0 V for PMOS), `corner`
  (`nom`/`tt`/`ss`/`ff`/`sf`/`fs`; default `nom`), `temperature` (kelvin), `NF`,
  `M`, and supported numeric BSIM4 instance parameters. The backend supplies
  full terminal current, conductance, charge, capacitance, and correlated noise
  to DC, AC, noise, transient, PSS, PAC, and PNoise. `extract_w` is accepted as a
  legacy hint but native devices always use their actual geometry. The optional
  `freepdk45_ngspice.nmos` / `.pmos` aliases retain the old cached-grid evaluator,
  while the full-circuit ngspice helpers provide an external oracle. Cards live
  under `PDK_ROOT/freepdk45/`; see
  `examples/freepdk45_5t_ota.json` (simple) and `examples/freepdk45_fd_ota.json`
  (the fully differential OTA design case, [docs/freepdk45_fd_ota_design.md](freepdk45_fd_ota_design.md)).
- **TSMC28HPC+** (`"tsmc28hpcp.nmos"` / `"tsmc28hpcp.pmos"`) binds the 0.9 V
  `nch_mac` / `pch_mac` core wrappers from the licensed 1d8 HSPICE deck. Use
  `vb=0.9` for a PMOS bulk tied to the core supply. Supported corners are
  `tt`/`ss`/`ff`/`sf`/`fs` (`nom` aliases `tt`); `temperature` is in kelvin and
  `NF` is passed natively to the foundry macro. The portable default model entry is
  `PDK/tsmc28hpcp/models/hspice/cln28hpcp_1d8_elk_v1d0_2p2.l`; overrides are
  `TSMC28_MODEL_DIR`, then `TSMC28_PDK_ROOT`. The default model types use the
  internal HSPICE parser plus native Berkeley BSIM4.5 backend for DC, AC, noise,
  transient, PSS, PAC, and PNoise; they do not launch ngspice. Explicit
  `tsmc28hpcp_ngspice.nmos` / `.pmos` aliases retain the complete-circuit ngspice
  oracle for regression comparisons. See
  [TSMC28HPC+ Local Adapter](tsmc28hpcp.md).
- A mixed circuit (some devices OTFT, some silicon) is valid — e.g. a complementary
  silicon OTA binds NMOS/PMOS devices independently. See `examples/sky130_5t_ota.json`.
- SKY130 needs its documented external simulator toolchain. FreePDK45 and
  TSMC28HPC+ native simulation need their model files and a C compiler
  for the first BSIM4 backend build; ngspice is optional and used only by its
  explicit oracle aliases. Missing prerequisites raise a clear error. See the
  "Silicon PDK / OSDI layer" section in [Core Solver Overview](module_overview.md).

### `bias`

Optional but usually needed. Supplies numeric values for `rails` string references.

```json
"bias": {
  "VDD": 40.0,
  "VCM": 30.65,
  "VB": 9.84,
  "VC": 16.0
}
```

If a rail references a key not in `bias`, the solve will fail.

### `adc`

Optional closed-loop SAR workflow configuration. The circuit itself still uses the
ordinary `devices`/`capacitors`/`vsources` fields. Differential `bit_inputs` and
`bit_inputs_bar` name CDAC PWL keys from MSB to LSB; decisions are read from the
physical transient comparator node. Run it with `circuit-opt adc --vin`, `--sweep`,
or `--sine`. See `examples/freepdk45_sar3.json` (static 5T comparator) and
`examples/freepdk45_sar6.json` (6-bit, clocked StrongARM comparator) for complete
configurations.

#### `adc.clock`

Optional comparator strobe for a **clocked dynamic (StrongARM) comparator**. When
present, `sar_input_waveforms` generates the named waveform key: it rests at `low`
and pulses to `high` around every bit's `decision_time`, so a dynamic latch
precharges while the CDAC settles and evaluates at the decision instant. Drive the
clocked tail / output-reset devices from that key via `transient_inputs`. Omitting
the block reproduces the static-comparator behaviour (no clock waveform emitted, so
`examples/freepdk45_sar3.json` renders a byte-identical netlist).

```json
"clock": {"input": "clk", "high": 1.0, "low": 0.0,
          "eval_before": 3e-9, "reset_hold": 1e-9}
```

- `input` — required transient waveform key for the strobe.
- `high` / `low` — asserted (evaluate) / deasserted (reset) levels [V]; default
  `high = adc.vref`, `low = 0`.
- `eval_before` — seconds before each `decision_time` that the strobe rises
  (default `0.3 * bit_period`); must be `< bit_period/2 - edge_time` so the tested
  CDAC bit has already switched when the latch samples.
- `reset_hold` — seconds after each `decision_time` before the strobe resets
  (default `0.1 * bit_period`).

#### `adc.mismatch`

Optional per-instance mismatch Monte-Carlo config for the FreePDK45 SAR
path, consumed by `circuitopt.sar_mismatch_mc`. Every sigma defaults to `0.0`, so
omitting the block (or leaving sigmas zero) reproduces the nominal conversion.

```json
"mismatch": {
  "sigma_vth0": 5e-3, "w0": 1.0, "l0": 0.05,
  "sigma_cu": 0.01, "c_unit": 1e-14,
  "dnl_threshold": 0.5, "inl_threshold": 0.5
}
```

- `sigma_vth0` — transistor threshold-voltage sigma [V] at the reference area
  `w0*l0`; per device it scales as `sigma_vth0 / sqrt(W*L / (w0*l0))` (Pelgrom area
  law) and is injected as the BSIM4 instance parameter `delvto`. `sigma_vth0_nmos`
  / `sigma_vth0_pmos` override it per polarity.
- `sigma_cu` — CDAC unit-capacitor relative sigma at `c_unit`; a cap of value `C`
  gets relative sigma `sigma_cu / sqrt(C / c_unit)` (binary-weighted caps are
  paralleled units and match better).
- `dnl_threshold` / `inl_threshold` — |DNL|/|INL| yield limits in LSB (default 0.5).

### `outputs`

Optional but needed for AC/noise/transient. Supports single-ended or differential.

Single-ended:

```json
"outputs": ["OUT"]
```

Differential:

```json
"outputs": ["VOP", "VON"]
```

Differential output is computed as the first node minus the second (`VOP - VON`).

### `input_drives`

Optional. AC small-signal gate drive, keyed by device name.

```json
"input_drives": {
  "M7": 0.5,
  "M8": -0.5
}
```

- Only meaningful for devices whose gates are on rails.
- Unlisted gates are treated as small-signal ground.
- For differential input, use `+0.5/-0.5` for a unit differential amplitude.

### `load_caps`

Optional. Fixed load capacitors, stamped into AC/noise/transient.

Array form:

```json
"load_caps": [
  ["VOP", "GND", 5e-12],
  ["VON", "GND", 5e-12]
]
```

Object form:

```json
"load_caps": [
  {"a": "OUT", "b": "GND", "C": 2e-12}
]
```

### `resistors`

Optional. Two-terminal resistors between nodes `a` and `b`, resistance `R` (ohms, must
be positive). DC adds branch current `(Va-Vb)/R`. AC/noise stamps conductance `1/R`.
Transient stamps conductance. Thermal noise PSD `4kT/R` is included in `dev_psd`,
keyed by resistor name.

```json
"resistors": [
  {"name": "RL", "a": "OUT", "b": "GND", "R": 4e6}
]
```

Array form: `["RL", "OUT", "GND", 4e6]`.

### `capacitors`

Optional. Two-terminal capacitors between nodes `a` and `b`, value `C` (farads, must
be positive). DC is open-circuit. AC stamps admittance `jωC`. Transient uses backward
Euler companion model. Equivalent to `load_caps`; the difference is that `capacitors`
entries have names and follow netlist convention.

```json
"capacitors": [
  {"name": "CL", "a": "OUT", "b": "GND", "C": 2e-12}
]
```

Array form: `["CL", "OUT", "GND", 2e-12]`.

### `current_sources`

Optional. Ideal DC current sources. Current `I` (amps, can be negative) flows from
`nplus` to `nminus` inside the source — i.e. pulls `I` from `nplus`, injects `I`
into `nminus`. DC enters KCL. Open-circuit (noiseless) in small-signal AC/noise.
Constant current in transient.

```json
"current_sources": [
  {"name": "IB", "nplus": "VDD", "nminus": "OUT", "I": 1e-6}
]
```

Array form: `["IB", "VDD", "OUT", 1e-6]`.

### `vccs`

Optional. Voltage‑controlled current sources. Output current flows ``p → q``:
``I = gm * (Vctrl_p - Vctrl_n)``. DC enters KCL; AC stamps into G matrix; noiseless
(ideal); instantaneous in transient with full Jacobian contribution.

```json
"vccs": [
  {"name": "G1", "p": "OUT", "q": "GND",
   "ctrl_p": "IN", "ctrl_n": "GND", "gm": 1e-4}
]
```

Array form: `["G1", "OUT", "GND", "IN", "GND", 1e-4]`.

### `vsources`

Optional. Ideal voltage sources, solved with **true MNA**: each source adds one
branch‑current unknown and one constraint row ``V_p − V_q = value``, so the system grows
from `n` nodes to `n_aug = n + m`. `value` is a constant EMF (number) or a transient
input‑waveform key (string) for a time‑varying ``E(t)``.

```json
"vsources": [
  {"name": "V1", "p": "IN", "q": "GND", "value": 2.0}
]
```

Array form: `["V1", "IN", "GND", 2.0]`. At least one of `p`, `q` must be a solved node
(a source between two rails is rejected).

- **DC** pins the node voltage exactly (the node stays in the solved set); `ac_solve`
  reports the source currents under `branch_currents` (sign: `p → q` through the source).
- **AC / Noise** treat a DC source as a short (AC ground); the ideal source carries no
  thermal noise. If the source name appears in `ac_drives`, it acts as an AC stimulus.
- **Transient** supports constant or waveform‑keyed `E(t)`. Circuits containing a voltage
  source run on the pure‑Python `n_aug` path (the numba kernels are fixed at `n` nodes).
- **PSS / PAC / PNoise** are supported too: the shooting monodromy and the harmonic‑balance
  matrices are bordered with the branch‑current unknowns (PNoise forces its dense path when
  a source is present).

### `vcvs`

Optional. Voltage‑controlled voltage sources. Output voltage ``V_p − V_q = mu * (V_cp − V_cn)``.
Each VCVS adds a branch‑current unknown (like an ideal voltage source) and a constraint
row with entries for the control nodes. Ideal / noiseless.

```json
"vcvs": [
  {"name": "E1", "p": "OUT", "q": "GND",
   "cp": "INP", "cn": "INN", "mu": 100.0}
]
```

Array form: `["E1", "OUT", "GND", "INP", "INN", 100.0]`. At least one of `p`, `q` must
be a solved node.

### `cccs`

Optional. Current‑controlled current sources. Output current ``I_out = beta * I_ctrl``
flows ``p → q``. The control current ``I_ctrl`` is the branch current of a voltage source
(vsource / VCVS / CCVS) named by `ctrl_name`. Ideal / noiseless. Does NOT add a new
branch‑current unknown — it references an existing one.

```json
"cccs": [
  {"name": "F1", "p": "OUT", "q": "GND",
   "ctrl_name": "V1", "beta": 2.0}
]
```

Array form: `["F1", "OUT", "GND", "V1", 2.0]`. `ctrl_name` must reference a vsource,
VCVS, or CCVS in the same topology.

### `ccvs`

Optional. Current‑controlled voltage sources. Output voltage ``V_p − V_q = gamma * I_ctrl``.
The control current ``I_ctrl`` is the branch current of a voltage source named by
`ctrl_name`. Each CCVS adds a branch‑current unknown. Ideal / noiseless.

```json
"ccvs": [
  {"name": "H1", "p": "OUT", "q": "GND",
   "ctrl_name": "V1", "gamma": 100.0}
]
```

Array form: `["H1", "OUT", "GND", "V1", 100.0]`. At least one of `p`, `q` must be a
solved node. `ctrl_name` must reference a vsource, VCVS, or CCVS (which has a branch
current). CCCS and CCVS can cascade: a CCCS can control on a CCVS's branch current.

### `dc_guesses`

Optional. DC initial guesses. Each entry can specify some or all solved nodes.

```json
"dc_guesses": [
  {"OUT": 20.0},
  {"OUT": 5.0},
  {"OUT": 35.0}
]
```

Provide multiple physically reasonable guesses for circuits with multistability or
positive feedback.

### `aliases`

Optional. Adds aliases to the DC operating point for compatibility with older code
or report fields.

```json
"aliases": {
  "vfb": "VFBP",
  "net2": "NET2"
}
```

The returned `dc_op` includes both the original solved nodes and the aliases.

### `transient_inputs`

Optional. Maps transient input waveform keys to device gates.

```json
"transient_inputs": {
  "M7": "vip",
  "M8": "vin"
}
```

Call transient with:

```python
tran = transient(sizes, bias, t, topo=topology,
                 inputs={"vip": vip_waveform, "vin": vin_waveform})
```

### `ac_drives`

Optional. Like `input_drives`, but drives a *node* instead of a device gate. Used
for testbench front-ends where the AC stimulus enters through a passive network
rather than directly at a transistor gate.

```json
"ac_drives": {
  "VINP": 0.5,
  "VINN": -0.5
}
```

### `periodic`

Optional. Default large-signal periodic excitation for PSS/PAC/PNoise and
periodic transient dispatch.

```json
"periodic": {
  "frequency": 1000.0,
  "n_points": 101,
  "inputs": {
    "vin": {"type": "constant", "value": "VIN"},
    "clk": {"type": "pulse", "low": 0.0, "high": "VDD", "duty": 0.5,
            "rise": 20e-6, "fall": 20e-6}
  },
  "node_inputs": {"VIN": "vin", "CLK": "clk"},
  "current_inputs": [{"p": "VDD", "q": "OUT", "input": "iqinj"}],
  "signed_devices": ["SW1", "SW2"]
}
```

Supported waveform forms:

- Number or bias key: constant waveform, e.g. `"VIN"`.
- `constant` / `dc`: constant waveform.
- `sine` / `sin` / `cosine` / `cos`: fields include `dc`, `amplitude`,
  `phase`, `frequency`, or `harmonic`.
- `square`: ideal square wave with `low`, `high`, `duty`, and `delay`.
- `pulse`: finite-edge periodic pulse with optional `rise` and `fall`.
- `pwl`: periodic piecewise-linear waveform with `times` and `values`.

### `analyses`

Optional. Unified analysis-dispatch configuration. Calling
`circuitopt.analysis_dispatch.run_analysis_suite(spec)` runs configured analyses in
the fixed order `ac -> noise -> transient -> pss -> pac -> pnoise`; PAC/PNoise
automatically reuse or create the required PSS result.

The authoritative option registry for `transient` / `pss` / `pac` / `pnoise`
lives in `circuitopt.analysis_options`. `analysis_dispatch.py` derives forwarded
solver kwargs and defaults from that registry, and the JSON schema is regression
tested against the same registry so new solver options do not silently drift.
Unknown keys in an `analyses` block are rejected with an error (a typo such as
`max_sidebands` for `max_sideband` is not silently ignored).

```json
"analyses": {
  "pss": {
    "corner": "slow",
    "residual_tol": 1e-12,
    "max_shooting_iters": 2,
    "jacobian_reuse": true,
    "analytic_jacobian": true
  },
  "pac": {
    "freqs": [100.0, 1000.0],
    "input_drive": {"vin": 1.0},
    "analytic": true,
    "max_sideband": 10,
    "n_period_samples": 384,
    "time_domain": false,
    "td_integration": "gear2",
    "td_n_period_samples": 768,
    "lti_fast_path": true,
    "cache_linearization": true,
    "cache_forcing": true
  },
  "pnoise": {
    "freqs": [100.0, 1000.0],
    "input_drive": {"vin": 1.0},
    "max_sideband": 0,
    "n_period_samples": 32,
    "lti_fast_path": true,
    "cache_linearization": true,
    "band": [100.0, 1000.0]
  }
}
```

`freqs` can be an explicit list or an object such as
`{"start": 1.0, "stop": 1e4, "num": 41, "scale": "log"}`. `input_drive` is the
PAC/PNoise small-signal complex amplitude map; JSON complex values can be a
number, `[real, imag]`, or `{"real": ..., "imag": ...}`.
Each analysis may set `corner` to `"typical"`, `"slow"`, `"fast"`, or an explicit
model-shift map. For PAC/PNoise, keep the PSS orbit on the same corner; when PSS
does not specify a corner, dispatch inherits the unique PAC/PNoise corner, and it
raises an error if a dependent analysis requests a different corner from an
already-built PSS.
PSS uses the analytic monodromy Jacobian by default (`"analytic_jacobian": true`):
it builds Φ in one orbit pass from the small-signal G(t)/C(t) stamps instead of
`n_state` finite-difference period runs. Set to `false` for the original FD path.
The Jacobian is then reused with a Broyden update; for difficult convergence or
tight reference comparisons, set `"jacobian_reuse": false` or periodically rebuild
with `"jacobian_rebuild_interval": 2`.
For gear2 PSS/transient, set `"adaptive": true` to enable LTE-controlled
adaptive timestepping. The dispatch forwards `"adaptive_reltol"`,
`"adaptive_vabstol"`, `"adaptive_iabstol"`, `"adaptive_max_steps"`,
`"adaptive_h0"`, and `"cap_mode"`; pulse/square periodic inputs get edge
breakpoints inserted before the adaptive run. `cap_mode` is limited to
`"charge"` (id 0) and `"average"` (id 1), plus their documented aliases.
PAC uses analytic-adjoint harmonic balance by default (`"analytic": true`): one
adjoint linear solve per frequency on the orbit conversion matrix, with zero extra
transient runs. `"max_sideband"` and `"n_period_samples"` control the HB resolution.
For rail-driven chopper-like circuits, set `"time_domain": true` to try the
accelerated time-domain Floquet PAC path first; `"td_integration"` and
`"td_n_period_samples"` control that path's BDF/grid settings. Unsupported
topologies fall back to HB when `"analytic": true`. Set `"analytic": false` only
for the original finite-difference shooting path.
PAC and PNoise enable the static-orbit LTI fast path and PSS-attached caches by default.
Set `"lti_fast_path": false`, `"cache_linearization": false`, or
`"cache_forcing": false` to force fresh finite-difference or harmonic-balance
work. PNoise reuses sampled `G(t)/C(t)`, HB blocks, and identical-frequency
adjoint solves from `pss_result`. For large PNoise HB systems, set
`"hb_solver": "sparse"` or `"iterative"` to force sparse direct or block-Jacobi
preconditioned GMRES; the default `"auto"` keeps small matrices dense and
switches only when the HB matrix is large and very sparse. PAC boundary-matrix
condition diagnostics are off by default because they require an SVD at every
frequency; enable them with
`"profile": true`, `"debug": true`, or explicit `"compute_condition": true`.

The JSON dispatch `pnoise` entry is the generic HB path. The chopper helper
`pmos_chopper_pnoise(...)` now defaults to the TD-adjoint PNoise path for Cadence
alignment; use that wrapper, or call `circuitopt.pnoise_solver.pnoise_solve(...,
time_domain=True)` directly, when the truncation-free chopper PNoise path is
required.

### `explore`

Optional. Design-space exploration configuration — variables to sweep with ranges,
feasibility constraints (gain, BW, IRN, power, area), and optimization objectives.
Consumed by `circuitopt.explore` (sample → evaluate → constrain → Pareto-select),
`circuitopt.dataset` (sample → evaluate every candidate, no filtering — a labeled training
set), and `circuitopt.optimize` (screen a trained surrogate → verify the shortlist on the
solver).

```json
"explore": {
  "variables": {
    "in_pair_W": {"min": 40000, "max": 90000, "targets": ["M7.W", "M8.W"]},
    "VCM":       {"min": 28.0,  "max": 33.0}
  },
  "constraints": {"gain_dB": {"min": 20}, "bw_Hz": {"min": 100},
                  "irn_uV": {"max": 44.5}},
  "objectives":  {"area": "min", "power_uW": "min"},
  "band":  [0.05, 100.0],
  "freqs": {"start": -2, "stop": 3, "num": 81}
}
```

- `variables` — each entry needs numeric `min`/`max`. `targets` lets one variable
  drive several keys at once (matched/symmetric device pairs); it defaults to
  `[<variable name>]`. `round` (decimals) snaps sampled values to a grid; `int`
  rounds to a whole number (handy for W/L/NF).
- `constraints` — each metric needs a `min` and/or `max` bound. Known metrics:
  `gain_dB`, `gain_peak_dB`, `bw_Hz`, `irn_uV`, `power_uW`, `area`.
- `objectives` — `{metric: "min" | "max"}`; at least one is required.
- `band` — `[f_lo, f_hi]` (Hz) for the band-integrated `irn_uV` metric.
- `freqs` — the AC/noise analysis grid: `{"start": <log10 Hz>, "stop": <log10 Hz>,
  "num": <points>}` (logarithmic).

**Target syntax** — beyond `"DEV.W"` / `"DEV.L"` / `"DEV.NF"` (device sizes) and a
bare bias key, `targets` supports structural design axes (each rebuilds the circuit
per candidate; consumed by `circuitopt.dataset`/`circuitopt.optimize`, not by `circuitopt.explore`):

| Target | Axis | Notes |
|--------|------|-------|
| `"<CapName>.C"` | a named capacitor's value (F) | the `capacitors` entry needs a `"name"` |
| `"<ResName>.R"` | a named resistor's value (Ω) | the `resistors` entry needs a `"name"` |
| `"periodic.frequency"` | the periodic stimulus clock frequency | needs a `periodic` block |
| `"pvt0"` / `"pbeta0"` | continuous global process shift | routes into `evaluate(corner=...)`; sampling this turns a discrete corner sweep into continuous-PVT training data |

See the `models` field above for binding a variable's target device to a non-default
PDK (e.g. sweeping a SKY130 device's `W`); the `explore` block itself doesn't change
— `models` and `explore.variables` compose freely.

## Complete Examples

```text
examples/single_stage.json        # Single-transistor common-source (PMOS_TFT)
examples/resistor_load_stage.json # PMOS + resistive load + output cap + current source
examples/voltage_divider.json     # Ideal voltage source (true MNA) — resistor divider
examples/vcvs_amplifier.json      # VCVS amplifier — linear gain 100×
examples/sc_lpf.json              # Switched-capacitor LPF (2-phase, PMOS switches + vsource clocks)
examples/afe_explore.json         # 10-transistor AFE with explore config
examples/periodic_rc.json         # Passive RC with PSS/PAC/PNoise dispatch
examples/sky130_5t_ota.json       # Silicon SKY130 complementary 5T OTA — `models` block + explore/dataset/optimize
examples/freepdk45_sar3.json      # FreePDK45 differential 3-bit SAR — full-charge .tran + ADC metrics
```

Load and run:

```python
import numpy as np

from circuitopt.circuit_loader import load_circuit_json
from circuitopt.ac_solver import ac_solve
from circuitopt.noise_solver import noise_analysis
from circuitopt.transient_solver import transient

spec = load_circuit_json("examples/single_stage.json")
freqs = np.logspace(0, 4, 121)

ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
noise = noise_analysis(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)

t = np.linspace(0, 1e-3, 100)
vin = np.full_like(t, spec.bias["VIN"])
tran = transient(spec.sizes, spec.bias, t, topo=spec.topology,
                 nf=spec.nf, inputs={"vin": vin})
```

Or run the analyses configured inside the JSON:

```python
from circuitopt.analysis_dispatch import run_analysis_suite
from circuitopt.circuit_loader import load_circuit_json

spec = load_circuit_json("examples/periodic_rc.json")
results = run_analysis_suite(spec)
pac_gain = results["pac"]["gains"]
pnoise_irn = results["pnoise"]["irn_uV_band"]
```

## Current Limitations

The JSON format is a local-solver circuit description, not a full SPICE netlist.

Supported:

- Three-terminal transistor devices (PMOS_TFT via ``TransistorModel`` interface).
- Resistors, capacitors, ideal DC current sources, VCCS (voltage‑controlled current sources), VCVS (voltage‑controlled voltage sources), CCCS (current‑controlled current sources), CCVS (current‑controlled voltage sources), ideal voltage sources (true MNA).
- DC/AC/noise/transient shared topology (resistors include thermal noise; controlled sources and ideal voltage sources are ideal/noiseless).
- Single-ended or differential outputs.
- Fixed load capacitance.
- AC gate drive and node drive.
- Transient gate waveforms and node waveforms.
- Periodic PSS/PAC/PNoise dispatch from JSON.
- DC initial guesses.

Supported (model abstraction):

- Device model registry (`circuitopt/device_model.py`) — ``TransistorModel`` ABC + factory.
  New model types can be added without modifying solver code.
- NMOS and PMOS across AT4000TG (PMOS-only), SKY130, FreePDK45, and TSMC28HPC+.
- Per-device model binding via the ``models`` field — mixed OTFT/silicon circuits
  (default PDK stays ``"at4000tg.pmos"`` unless overridden).
- Silicon DC/AC/noise; SKY130 uses OSDI transient, while FreePDK45 and
  TSMC28HPC+ use the internal native BSIM4 backend.

Not yet supported:

- ADC transient noise, layout parasitic extraction, and transistor-level SAR digital control.
- Multi-output simultaneous analysis.
- Hierarchical subcircuits.
- Arbitrary user-circuit SPICE netlist import. The internal HSPICE parser is
  currently scoped to supported local model-library elaboration.
