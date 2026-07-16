# Design Cases and Validation Status

[Documentation Home](README.md)

These documents are engineering records for specific circuits. They preserve
derivations, topology decisions, dimensions, and reproduction commands, but
their maturity differs. Use the status column before treating any number as a
project guarantee.

| Design record | Process | What it demonstrates | Current status |
|---|---|---|---|
| [AFE Design Equations](afe_design_equations.md) | AT4000TG | SLiCAP symbolic equations and cross-checks | Reference derivation |
| [SKY130 Fully Differential OTA](sky130_fd_ota_design.md) | SKY130 | Telescopic OTA, CMFB, sizing, PVT workflow | Reproducible with bundled cards and native BSIM; new geometries may need card extraction |
| [FreePDK45 Fully Differential OTA](freepdk45_fd_ota_design.md) | FreePDK45 | Full OTA design and ngspice AC cross-check | Reproducible design snapshot; backend limitations apply |
| [FreePDK45 SAR ADC](freepdk45_sar_design.md) | FreePDK45 | CDAC, StrongARM comparator, static/dynamic/MC workflow | Reproducible experiment; not a sign-off ADC flow |
| [FreePDK45 MDAC OTA Derivation](mdac_ota_derivation.md) | FreePDK45 | ADC-to-OTA requirement derivation and testbench conventions | No complete versioned campaign; the local audit output covered 42 of 45 points |
| [TSMC28HPC+ MDAC OTA](tsmc28_mdac_ota_design.md) | TSMC28HPC+ | 14-bit pipeline first-stage MDAC OTA architecture and tests | No complete versioned campaign; the local audit output covered 7 TT points |

## Reading Rules

- A design case is scoped to its exact topology, model backend, dimensions,
  testbench, and integration bandwidth.
- “Reproducible” means the repository contains the circuit and command path. It
  does not mean the external PDK is redistributable.
- Campaign CSV files under `results/` are local outputs and are not versioned.
  When present, they are the numerical source for a local report; prose must not
  claim corners absent from that output.
- PVT, noise, transient, and loop-stability claims are not interchangeable. Each
  must be tied to the testbench that measured it.
- CircuitOpt results are pre-layout unless a document explicitly supplies
  extracted parasitics.
