# JSON Circuit Description Format

[Project Overview](README.md) | [Core Solver Overview](core_overview.md) | [中文版](json_circuit_format_zh.md)

## Purpose

The JSON circuit description separates topology, sizes, bias, and analysis metadata
from Python source code. Swap circuits by changing JSON files instead of editing
`core/ac_solver.py`, `core/noise_solver.py`, or `core/transient_solver.py`.

Schema file:

```text
schemas/circuit.schema.json
```

Example files:

```text
examples/single_stage.json
examples/resistor_load_stage.json
examples/afe_explore.json
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
- `devices` — list of PMOS_TFT transistors.
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

Required. Each active device is a three-terminal PMOS_TFT (drain/gate/source).

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

### `explore`

Optional. Design-space exploration configuration — variables to sweep with ranges,
feasibility constraints (gain, BW, IRN, power, area), and optimization objectives.

```json
"explore": {
  "variables": {
    "W6": [500, 5000],
    "L6": [40, 200],
    "VB": [8.0, 12.0]
  },
  "constraints": {
    "gain_dB": {">": 20.0},
    "bw_Hz": {">": 300.0},
    "irn_uV": {"<": 50.0}
  },
  "objectives": ["area", "power_uW"]
}
```

## Complete Examples

```text
examples/single_stage.json        # Pure PMOS_TFT
examples/resistor_load_stage.json # PMOS + resistive load + output cap + current source
examples/afe_explore.json         # 10-transistor AFE with explore config
```

Load and run:

```python
import numpy as np

from core.circuit_loader import load_circuit_json
from core.ac_solver import ac_solve
from core.noise_solver import noise_analysis
from core.transient_solver import transient

spec = load_circuit_json("examples/single_stage.json")
freqs = np.logspace(0, 4, 121)

ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
noise = noise_analysis(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)

t = np.linspace(0, 1e-3, 100)
vin = np.full_like(t, spec.bias["VIN"])
tran = transient(spec.sizes, spec.bias, t, topo=spec.topology,
                 nf=spec.nf, inputs={"vin": vin})
```

## Current Limitations

The JSON format is a local-solver circuit description, not a full SPICE netlist.

Supported:

- PMOS_TFT three-terminal devices.
- Resistors, capacitors, ideal DC current sources.
- DC/AC/noise/transient shared topology (resistors include thermal noise).
- Single-ended or differential outputs.
- Fixed load capacitance.
- AC gate drive and node drive.
- Transient gate waveforms and node waveforms.
- DC initial guesses.

Not yet supported:

- NMOS or other compact models (no NMOS in this PDK).
- Ideal voltage sources, controlled sources, switched/time-varying elements.
- Multi-output simultaneous analysis.
- Hierarchical subcircuits.
- SPICE syntax parsing.

These should be added after a device model registry and additional MNA stamp
elements are in place.
