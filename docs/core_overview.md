# Core Solver Overview

[Project overview](README.md) | [中文说明](README_zh.md)

This document introduces the current `core/` solver stack. The code is a compact local implementation of an AT4000TG OTFT ECG AFE solver, calibrated against Cadence/Spectre behavior. It is intended as the first concrete backend of the broader local circuit optimization flow.

## Scope

The current solver stack covers:

- DC operating-point solving.
- AC small-signal gain and bandwidth analysis.
- Noise analysis, including flicker noise and thermal noise.
- Transient response simulation.
- Process-corner and per-device mismatch perturbations.
- Cadence/Spectre-oriented validation for operating point, AC, noise, and transient behavior.

The implementation is intentionally small and self-contained. It currently consists of six Python source files under `core/`.

## File Structure

```text
core/
  topology.py          Circuit topology source of truth.
  pmos_tft_model.py    AT4000TG PMOS-OTFT compact-model implementation.
  ac_mna.py            MNA stamping primitives.
  ac_solver.py         DC operating point and AC small-signal solver.
  noise_solver.py      Noise propagation and input-referred noise analysis.
  transient_solver.py  Time-domain transient solver.
```

## Import Relationship

```text
topology.py          <- no internal dependency
pmos_tft_model.py    <- no internal dependency
ac_mna.py            <- no internal dependency
ac_solver.py         <- topology, ac_mna, pmos_tft_model
noise_solver.py      <- ac_solver, topology, ac_mna, pmos_tft_model
transient_solver.py  <- ac_solver, topology, pmos_tft_model
```

## Main Components

### `pmos_tft_model.py`

Implements the AT4000TG PMOS-OTFT compact model in Python. It provides:

- Terminal current evaluation through `get_Idc`.
- Drain-current noise PSD through `get_noise_psd`.
- Bias-dependent terminal capacitances through `get_capacitances`.
- Geometry area calculation through `g_area`.
- Process and mismatch parameters such as `pvt0`, `mvt0`, `pbeta0`, and `mbeta0`.

For AC and noise analysis, the solver extracts terminal `gm` and `gds` by finite-differencing `get_Idc`, matching the terminal behavior used by the circuit solver.

### `topology.py`

Defines the circuit topology as the single source of truth. The topology contains the device list, solved node list, and rail/bias nodes. DC KCL equations, bias mapping, and AC/noise terminal tables are derived from this topology instead of being hand-written separately in each solver.

The default topology is `AFE_TOPO`, a 10-transistor fully differential AFE core with tail current device, input pair, output stage, and cross-coupled positive-feedback level shifting devices.

### `ac_mna.py`

Provides the low-level MNA stamping primitives used by the small-signal solvers:

- Admittance stamping.
- VCCS stamping.
- MOS small-signal stamping.

### `ac_solver.py`

Solves the DC operating point and AC response:

- `ac_solve(sizes, bias, freqs, corner=None, x0_guess=None, topo=AFE_TOPO)`
- Uses `scipy.fsolve` for the DC node equations.
- Returns gain, bandwidth, node operating point, and extracted small-signal parameters.
- Supports both global process corners and per-device mismatch maps.

The DC solve includes robustness handling for physical branch selection, symmetric operating points, and rail-bounded node solutions.

### `noise_solver.py`

Performs noise propagation on the same six-node MNA system used by AC analysis. Each transistor drain-current noise source is injected between drain and source, propagated to the differential output, and divided by the signal gain to obtain input-referred noise.

The noise flow supports the same topology-derived terminal mapping and corner/mismatch parameter passing as the AC solver.

### `transient_solver.py`

Solves the time-domain response of the six-node system using backward Euler integration:

- `transient(sizes, bias, tgrid, vip, vin, nf=None, V0=None)`
- Directly drives the M7/M8 input gates through `vip(t)` and `vin(t)`.
- Includes output load capacitance.
- Re-evaluates nonlinear capacitances during Newton iterations.
- Uses the DC operating point from `ac_solve` as the default initial condition.

## Quick Example

```python
import numpy as np

from ac_solver import ac_solve
from noise_solver import noise_analysis, band_rms
from transient_solver import transient

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

## Calibration Status

The current core was calibrated against Cadence Spectre 24.1 for the AT4000TG AFE use case. The observed agreement in the original project included:

- Typical and corner AC behavior within approximately 0.01 dB for gain.
- Input-referred noise within a few percent across validated cases.
- Per-device mismatch Monte Carlo mean and standard deviation matching Cadence trends.
- Transient step and sinusoidal response closely matching Cadence `tran` behavior.
- Final locked design around 22.9 dB gain, 549 Hz bandwidth, and 37 uVrms input-referred noise.

These numbers describe the current AT4000TG validation case. Future PDKs or topologies should be recalibrated against their own simulator references.

