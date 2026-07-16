# PDK Support Matrix

[Documentation Home](README.md) | [中文](pdk_support_zh.md)

CircuitOpt uses per-device model bindings. A circuit may bind different devices
to different registered model types, although most practical circuits use one
process consistently.

## Capability Matrix

| Process | Model keys | Device backend | DC / AC / Noise | Transient | PSS / PAC / PNoise | External prerequisites |
|---|---|---|---|---|---|---|
| AT4000TG | `at4000tg.pmos` | Built-in calibrated PMOS model | Yes | Native | Yes | None |
| SKY130 | `sky130.nmos`, `sky130.pmos` | Resolved card + OpenVAF OSDI | Yes | OSDI backend | Available where terminal linearization is supported; validate each topology | SKY130 PDK, ngspice for card resolution, OpenVAF/BSIM4 VA |
| FreePDK45 | `freepdk45.nmos`, `freepdk45.pmos` | Cached ngspice-C characterization grid | Yes | Full-circuit ngspice `.tran` | No direct FreePDK45 PSS/PAC/PNoise backend | FreePDK45 cards and ngspice |
| TSMC28HPC+ core | `tsmc28hpcp.nmos`, `tsmc28hpcp.pmos` | Internal HSPICE frontend + native Berkeley BSIM4.5 | Yes | Native charge-conserving BE/Gear2 | Yes | Licensed supported model file; C compiler on first build |
| TSMC28HPC+ oracle | `tsmc28hpcp_ngspice.nmos`, `tsmc28hpcp_ngspice.pmos` | Explicit ngspice comparison path | Oracle only | Oracle only | Not the default periodic backend | Licensed model file and ngspice |

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

- Uses ngspice to resolve the foundry subcircuit and binned parameter web into a
  flat card, then evaluates an OpenVAF-compiled BSIM4 Verilog-A model through the
  OSDI host.
- This is a useful local optimization model, not a bit-exact replacement for
  the official SKY130 simulator model.
- Corners accepted by the circuit flow: `tt`, `ss`, `ff`, `sf`, `fs`.
- Resolved cards and compiled OSDI artifacts are caches, not source inputs.

### FreePDK45

- Uses ngspice-C as the model evaluator because the supplied BSIM4 version and
  the available OSDI BSIM source do not match closely enough.
- DC, AC, and noise use cached characterization grids.
- Transient routes the complete supported circuit to ngspice so BSIM charge and
  junction capacitance remain in the simulation.
- The fast AC grid omits some drain/source junction capacitance. Whole-circuit
  bandwidth should be cross-checked with the explicit ngspice AC oracle.
- Corners: `nom`, `tt`, `ss`, `ff`, `sf`, `fs`.
- Generic silicon mismatch semantics are not yet implemented through `circuit-opt mc`.

### TSMC28HPC+

- Supports the 0.9 V `nch_mac` and `pch_mac` core wrappers from the documented
  1d8 HSPICE model file.
- The internal parser resolves `.lib` sections, parameters, subcircuits, macros,
  and geometry bins in memory.
- The native backend exposes four-terminal current, conductance, charge,
  capacitance, and correlated noise.
- Corners: `tt`, `ss`, `ff`, `sf`, `fs`; `nom` aliases `tt`.
- Temperature is passed in kelvin at the device binding API.
- The current adapter does not claim support for IO devices, RF devices, SRAM,
  eFuse, reliability models, statistical sections, layout extraction, or
  sign-off checks from the full iPDK.
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
| TSMC model directory | `TSMC28_MODEL_DIR`, `TSMC28_PDK_ROOT`, project-local ignored entry, then `PDK_ROOT/tsmc28hpcp` |
| ngspice | `NGSPICE_BIN`, virtual-environment locations, then `PATH` |
| OpenVAF | `OPENVAF_BIN`, `OPENVAF_ROOT`, virtual-environment locations, then `PATH` |
| Native model cache | `CIRCUITOPT_NATIVE_MODEL_CACHE`, otherwise the selected virtual environment |

## What CircuitOpt Does Not Replace

CircuitOpt is a local simulation and optimization framework. It does not replace
official PDK installation, schematic/layout libraries, DRC, LVS, parasitic
extraction, EM/IR, aging, reliability, statistical sign-off, or foundry-approved
tapeout flows.
