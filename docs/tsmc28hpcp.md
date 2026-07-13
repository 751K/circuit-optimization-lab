# TSMC28HPC+ Local Adapter

[English](tsmc28hpcp.md) | [中文说明](tsmc28hpcp_zh.md)

The `tsmc28hpcp` PDK binding lets circuitopt use a locally installed, licensed
TSMC 28HPC+ model delivery without committing any foundry files to Git.
It targets the 0.9 V core `nch_mac` and `pch_mac` devices in the 1.8 V-capable
HSPICE model deck.

## Configure the model

The standard portable entry point is:

```text
PDK/tsmc28hpcp/models/hspice/cln28hpcp_1d8_elk_v1d0_2p2.l
```

This path is relative to the repository root and is checked automatically. The
licensed `models/` payload is ignored by Git. On another computer, place the same
model file at this location; no source-code path needs to change.

Only this main 1d8 model file is required by the current core-device adapter. The
160 GB delivery does not need to be copied into the project. The original file is
referenced unchanged; circuitopt stores no extracted model parameters in source.

Environment variables remain available when the PDK is installed elsewhere:

```bash
# Directly name the directory containing cln28hpcp_1d8_elk_v1d0_2p2.l
export TSMC28_MODEL_DIR=/path/to/iPDK/models/hspice

# Or name an installed/delivery root; circuitopt searches models/hspice below it.
export TSMC28_PDK_ROOT=/path/to/iPDK_delivery
```

`TSMC28_MODEL_DIR` has priority, followed by `TSMC28_PDK_ROOT`, the project-local
path, and then `PDK_ROOT/tsmc28hpcp`. No absolute machine path is stored in a
circuit JSON file or Python module. The adapter checks the model file only when
the PDK is actually used.

Use the project ngspice build (ngspice 46 is verified), or set:

```bash
export NGSPICE_BIN=/path/to/ngspice
```

The adapter starts ngspice with `-D ngbehavior=hsa`, expands the foundry deck's
nested same-file `.lib` dependency closure (`setup`, corner, `global`, `total`,
`stat`), and leaves the delivered model file unchanged.

## Bind devices

```json
{
  "devices": [
    {"name": "MN", "drain": "OUT", "gate": "IN", "source": "GND", "W": 1.0, "L": 0.03},
    {"name": "MP", "drain": "OUT", "gate": "IN", "source": "VDD", "W": 2.0, "L": 0.03}
  ],
  "models": {
    "MN": {"type": "tsmc28hpcp.nmos"},
    "MP": {"type": "tsmc28hpcp.pmos", "vb": 0.9}
  },
  "bias": {"VDD": 0.9}
}
```

Geometry is expressed in micrometres. `NF` is passed to the foundry macro rather
than approximated by multiplying a one-finger result. PMOS bulk should normally be
set explicitly to the 0.9 V core rail.

Supported corners are `tt`, `ss`, `ff`, `sf`, and `fs` (`nom` aliases `tt`). A
shared circuit temperature can be supplied in kelvin through each model's
`temperature` field.

## Analysis coverage

- `transient()` automatically routes a TSMC-bound circuit to full ngspice
  simulation, retaining BSIM4 charge and transient behavior.
- `ac_ngspice()`, `noise_ngspice()`, and `op_ngspice()` render the full circuit
  against the same deck. Hierarchical operating-point vectors from `nch_mac` and
  `pch_mac` are mapped back to circuitopt device names.
- The local DC/AC/noise solvers can use cached ngspice characterization grids for
  optimization loops. Cache files are process-, corner-, geometry-, temperature-,
  and finger-count-specific and are ignored by Git.
- Per-instance threshold mismatch is emitted through the macro's `_delvto`
  parameter on the full transient path.

This adapter covers simulation and optimization. Layout, DRC/LVS, extraction, and
Cadence library setup remain responsibilities of the installed foundry iPDK and are
intentionally outside circuitopt.

PSS/PAC/PNoise are not currently routed to the direct-ngspice model-card backend.
Use full-circuit ngspice `.ac`/`.noise` for model-deck measurements and native
`.tran` for ADC switching/settling. The local characterized grid is intended for
fast optimization loops, not final junction-charge sign-off.

## Verify

```bash
pytest -q tests/test_tsmc28.py
```

When the project-local model or an environment override is available, integration
tests run a real CMOS inverter and a 3-bit SAR ADC. The adapter has also been checked
against hierarchical `.op`, one-device DC/capacitance characterization, and `.noise`.
The SAR smoke test takes about two minutes because it replays a complete transient
for each physical comparator decision. Without the licensed model, adapter unit
tests still run and model-dependent tests are skipped.

The model remains subject to the user's foundry agreement/NDA. Keeping it in an
ignored local directory prevents accidental Git publication; it does not change
the model's licensing terms.
