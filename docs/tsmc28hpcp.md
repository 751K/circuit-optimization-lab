# TSMC28HPC+ Native Adapter

[English](tsmc28hpcp.md) | [中文说明](tsmc28hpcp_zh.md)

> **Status: maintained adapter guide.** Scope is the documented 0.9 V core MOS
> wrappers and the analysis paths listed below, not the complete iPDK.

The `tsmc28hpcp` binding evaluates a locally installed, licensed TSMC 28HPC+
model deck inside circuitopt. The default `tsmc28hpcp.nmos` and
`tsmc28hpcp.pmos` model types do not launch ngspice: circuitopt parses the
HSPICE library closure, resolves the foundry MOS macros and bins, and evaluates
the resulting BSIM4.5 models through its bundled native compact-model backend.

The adapter targets the 0.9 V core `nch_mac` and `pch_mac` wrappers in the 1d8
HSPICE model deck. Foundry files remain local and must not be committed.

## Model Entry

The portable project-local entry point is:

```text
PDK/tsmc28hpcp/models/hspice/cln28hpcp_1d8_elk_v1d0_2p2.l
```

Only this main model file is required for the current core-device adapter; the
complete iPDK delivery does not need to be copied into the repository. The
licensed `models/` payload is ignored by Git.

External installations can use:

```bash
export TSMC28_MODEL_DIR=/path/to/models/hspice
# or
export TSMC28_PDK_ROOT=/path/to/iPDK_delivery
```

Resolution order is `TSMC28_MODEL_DIR`, `TSMC28_PDK_ROOT`, the project-local
entry, then `PDK_ROOT/tsmc28hpcp`. No machine-specific absolute path is stored
in circuit JSON or Python source.

The vendored Berkeley BSIM4.5 device source is compiled into the
`circuitopt_core` extension at build time (the `co-bsim4` Rust crate drives the
build), not on first use at runtime. Installing a released `circuitopt-core`
wheel needs no compiler at all; building it from a source checkout needs a Rust
toolchain (rustup) and a C compiler once, via `maturin develop --release -m
rust/crates/co-py/Cargo.toml`. ngspice is not required for normal simulation.

## Device Binding

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

Geometry is in micrometres. `NF` is passed through the foundry macro rather
than approximated by scaling a one-finger result. PMOS bulk should normally be
set explicitly to the 0.9 V core rail.

Supported corners are `tt`, `ss`, `ff`, `sf`, and `fs`; `nom` aliases `tt`.
`temperature` is expressed in kelvin. Per-instance threshold mismatch is
forwarded through the macro `_delvto` parameter.

## Analysis Coverage

The native path supports:

- nonlinear DC and four-terminal operating-point currents;
- full four-terminal conductance and charge linearization for AC and PAC;
- correlated four-terminal white and flicker noise for noise and PNoise;
- charge-conserving backward-Euler and Gear2 transient integration;
- PSS shooting with an analytic full-terminal monodromy;
- PAC harmonic conversion and PNoise cyclostationary folding.

PNoise retains the terminal covariance matrix and extracts the foundry deck's
flicker exponent instead of assuming every device follows exactly `1/f`.

ngspice remains available only as an independent oracle. Bind every MOS to
`tsmc28hpcp_ngspice.nmos` / `tsmc28hpcp_ngspice.pmos`, or use the helpers in
`circuitopt.ngspice_ac`, when an explicit comparison against ngspice is wanted.
Set `NGSPICE_BIN` only for that oracle path.

This adapter covers circuit simulation and optimization. Cadence library setup,
layout, DRC/LVS, extraction, reliability checks, and tapeout sign-off remain
responsibilities of the official installed iPDK and approved foundry tools.

## Five-Transistor OTA Verification

The compact benchmark contains exactly five MOS devices and no ADC/CDAC logic:

```bash
python experiments/tsmc28_5t_ota_compare.py \
  --output /tmp/tsmc28_5t_compare.json
pytest -q tests/test_tsmc28_5t_ota.py
```

The comparison runs the same TSMC28 5T OTA through the native backend and the
explicit ngspice oracle. It checks device `Id/gm/gds`, differential AC, output
noise integrated from 1 kHz to 10 GHz, and a 2 mV differential input-step
transient. The report contains fixed pass thresholds and a top-level `passed`
flag.

The model remains subject to the user's foundry agreement/NDA. Keeping it in an
ignored local directory prevents accidental Git publication; it does not alter
the licensing terms.
