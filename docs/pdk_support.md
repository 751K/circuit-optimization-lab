# PDK Support Matrix

[Documentation Home](README.md) | [中文](pdk_support_zh.md)

CircuitOpt uses per-device model bindings. A circuit may bind different devices
to different registered model types, although most practical circuits use one
process consistently.

## Capability Matrix

| Process | Model keys | Device backend | DC / AC / Noise | Transient | PSS / PAC / PNoise | External prerequisites |
|---|---|---|---|---|---|---|
| AT4000TG | `at4000tg.pmos` | Built-in calibrated PMOS model | Yes | Native | Yes | None |
| SKY130 | `sky130.nmos`, `sky130.pmos` | Bundled resolved cards + native Berkeley BSIM4.5 | Yes | Compiled Rust core + native C BSIM4 BE/Gear2 | Native terminal backend; validate each periodic topology | None for a released wheel; external tools only for new-card extraction |
| FreePDK45 | `freepdk45.nmos`, `freepdk45.pmos` | Flat-card loader + native Berkeley BSIM4.5 | Yes | Compiled Rust core + native C BSIM4 BE/Gear2 | Native terminal backend; validate each periodic topology | FreePDK45 cards; none else for a released wheel |
| FreePDK45 oracle | `freepdk45_ngspice.nmos`, `freepdk45_ngspice.pmos` | Cached ngspice-C grid / complete-deck oracle | Oracle only | Oracle only | Not the default periodic backend | FreePDK45 cards and ngspice |
| TSMC28HPC+ core | `tsmc28hpcp.nmos`, `tsmc28hpcp.pmos` | Internal HSPICE frontend + native Berkeley BSIM4.5 | Yes | Compiled Rust core + native C BSIM4 BE/Gear2 | Yes | Licensed supported model file; none else for a released wheel |
| TSMC28HPC+ oracle | `tsmc28hpcp_ngspice.nmos`, `tsmc28hpcp_ngspice.pmos` | Explicit ngspice comparison path | Oracle only | Oracle only | Not the default periodic backend | Licensed model file and ngspice |

Building `circuitopt_core` from a source checkout (instead of installing a
released wheel) additionally needs a Rust toolchain (rustup) and a C compiler
for the vendored BSIM4.5 sources (`maturin develop --release -m
rust/crates/co-py/Cargo.toml`); see [Getting Started](getting_started.md).

“Supported” means the backend is connected to the analysis path. It does not
mean foundry sign-off equivalence. Each topology still needs regression against
an appropriate reference.

## Process Details

### AT4000TG

- Default process when a transistor is not listed in the JSON `models` block.
- PMOS-only.
- Calibrated against archived Cadence Spectre references for the included AFE
  and chopper cases.
- Corners: `typical`, `slow`, `fast`.
- The generic `mc` command currently targets this continuous mismatch model.

### SKY130

- Loads bundled, geometry-resolved BSIM4.5 cards and evaluates them directly
  with the same in-process native C backend used by FreePDK45 and TSMC28HPC+.
- Normal DC, AC, noise, transient, PSS, PAC, and PNoise runs do not launch an
  external simulator.
- This is a useful local optimization model, not a bit-exact replacement for
  the official SKY130 simulator model.
- Corners accepted by the circuit flow: `tt`, `ss`, `ff`, `sf`, `fs`.
- The repository includes the resolved cards used by its examples. For a new
  geometry/corner that has no card, explicitly run
  `circuitopt.sky130_model.extract_sky130_card()` with a local SKY130/ngspice
  installation and point `SKY130_CARD_DIR` at the generated card directory.

### FreePDK45

- Parses the flat level-54 VTG cards directly and evaluates them with the
  in-process Berkeley BSIM4.5 kernel.
- The cards declare BSIM4 `version=4.0`. In the bundled Berkeley source the
  version field is metadata rather than an equation switch; single-device
  `Id/gm/gds`, noise, and 5T OTA AC are regression-checked against ngspice.
- The native backend exposes full four-terminal current, conductance, charge,
  capacitance, and correlated noise for DC, AC, noise, transient, and periodic
  analyses.
- The fixed-grid transient hot path runs Newton iteration and matrix stamping
  inside the compiled Rust core (`co-core` calling `co-bsim4` in-process); the
  Python-facing `compact_models/bsim4/native.py` module binds the same compiled
  ABI through `ctypes` (a `void *` function-pointer interface) for single-device
  op-point/AC/noise calls outside the transient loop. As of v2.0.0 `rust` is the
  sole engine; there is no runtime compiler step and no Python numeric fallback.
- `freepdk45_ngspice.*` and the full-circuit ngspice helpers remain available as
  independent regression oracles. Import `circuitopt.freepdk45_model` to
  register the oracle-only model keys. They are optional for normal simulation.
- Corners: `nom`, `tt`, `ss`, `ff`, `sf`, `fs`.
- `tt` aliases `nom`; `sf` selects NMOS slow plus PMOS fast, and `fs` the
  reverse.
- Generic silicon mismatch semantics are not yet implemented through `circuit-opt mc`.

### TSMC28HPC+

- Supports the 0.9 V `nch_mac` and `pch_mac` core wrappers from the documented
  1d8 HSPICE model file.
- The internal parser resolves `.lib` sections, parameters, subcircuits, macros,
  and geometry bins in memory.
- The native backend exposes four-terminal current, conductance, charge,
  capacitance, and correlated noise.
- Transient uses the same compiled Rust-to-C BSIM4 bridge as FreePDK45.
- Corners: `tt`, `ss`, `ff`, `sf`, `fs`; `nom` aliases `tt`.
- Temperature is passed in kelvin at the device binding API.
- The current adapter does not claim support for IO devices, RF devices, SRAM,
  eFuse, reliability models, statistical sections, layout extraction, or
  sign-off checks from the full iPDK.
- Import `circuitopt.tsmc28_model` only when the
  `tsmc28hpcp_ngspice.*` oracle model keys are required.
- See the [TSMC28HPC+ Native Adapter](tsmc28hpcp.md).

## JSON Binding

```json
{
  "devices": [
    {
      "name": "MN",
      "drain": "OUT",
      "gate": "IN",
      "source": "GND",
      "W": 1.0,
      "L": 0.03
    },
    {
      "name": "MP",
      "drain": "OUT",
      "gate": "IN",
      "source": "VDD",
      "W": 2.0,
      "L": 0.03
    }
  ],
  "models": {
    "MN": {"type": "tsmc28hpcp.nmos"},
    "MP": {"type": "tsmc28hpcp.pmos", "vb": 0.9}
  }
}
```

Geometry is expressed in micrometres. Model-specific constructor options are
documented under `models` in the [Circuit JSON Format](json_circuit_format.md).

## Path Resolution

| Input | Resolution |
|---|---|
| General PDK root | `PDK_ROOT`, then active/project virtual-environment `pdk/` |
| Additional SKY130 resolved cards | `SKY130_CARD_DIR`, then bundled package cards |
| TSMC model directory | `TSMC28_MODEL_DIR`, `TSMC28_PDK_ROOT`, project-local ignored entry, then `PDK_ROOT/tsmc28hpcp` |
| ngspice | `NGSPICE_BIN`, virtual-environment locations, then `PATH` |

## What CircuitOpt Does Not Replace

CircuitOpt is a local simulation and optimization framework. It does not replace
official PDK installation, schematic/layout libraries, DRC, LVS, parasitic
extraction, EM/IR, aging, reliability, statistical sign-off, or foundry-approved
tapeout flows.
