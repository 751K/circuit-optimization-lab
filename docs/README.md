# Circuit Optimization Flow

[English](README.md) | [中文说明](README_zh.md)

## Overview

This project builds a local circuit modeling, simulation, and optimization flow for analog design-space exploration. The core motivation is to reduce the need for exhaustive Cadence/Spectre sweeps during early sizing and biasing iterations, while still using Cadence/Spectre results as the final verification reference.

The first use case is an AT4000TG thin-film transistor amplifier design. In this project, a local Python-based model is used to reproduce and analyze key circuit behavior, including DC operating point, small-signal response, transient response, noise, and design constraints. The repository is intended to grow beyond this initial implementation and become a more general framework for circuit exploration and optimization.

## Current Scope

The current flow covers or is planned to cover:

- Device-level compact-model evaluation.
- DC operating-point solving.
- AC small-signal gain and bandwidth estimation.
- Transient response simulation.
- Noise analysis, including thermal noise and flicker noise.
- Cadence/Spectre comparison and calibration.
- Constraint checking for gain, bandwidth, input-referred noise, power, and area.
- Local design-space exploration without running every candidate directly in Cadence.
- Size and bias optimization using search, greedy shrink, and Pareto selection.
- Process-corner and mismatch Monte Carlo style robustness checks.
- Research-style plot generation for design reports and presentations.

The current implementation details are summarized in [Core Solver Overview](core_overview.md).

## Installation

Requires Python 3.10 or later:

```bash
python3 -m pip install -r requirements.txt
```

`requirements.txt` installs this project in editable mode, so external scripts and notebooks can import directly via `from core...` without manual path adjustments.

Optional Numba-accelerated backend for the PMOS model and transient Newton hot paths. Numba is enabled by default when installed; set `CIRCUIT_USE_NUMBA=0` to disable:

```bash
python3 -m pip install -r requirements-numba.txt
python3 -m benchmarks.bench_afe --skip-noise --warm-runs 3
```

Fixed performance benchmarks:

```bash
python3 -m benchmarks.bench_afe --warm-runs 3
CIRCUIT_USE_NUMBA=0 python3 -m benchmarks.bench_afe --warm-runs 3
```

## JSON Circuit Description

Solvers now support loading generic circuit descriptions from JSON, avoiding hard-coded node and device names in `core/*.py`. Format details in [JSON Circuit Description](json_circuit_format_zh.md), example at `examples/single_stage.json`.

Minimal usage:

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

Key JSON fields include `solved`, `rails`, `devices`, `bias`, `outputs`, `input_drives`, `load_caps`, `dc_guesses`, and `transient_inputs`. `devices` can embed `W/L/NF` directly, or use separate `sizes`/`nf` fields.

## Interactive AFE Tuner

A web-based interactive tuner is available under `demo/`:

```bash
python3 -m pip install -r requirements-demo.txt
python3 demo/server.py
# Open http://localhost:5100
```

The tuner exposes the validated core solvers (DC + AC + noise) through a REST API, with an HTML frontend for real-time exploration of gain, bandwidth, and input-referred noise across device sizes and bias voltages. It includes preset designs (Base, Final Locked, Min Area, First Feasible), DC seed warm-starting and branch rescue logic, and a bounded thread pool with concurrency control.

## Motivation

Analog circuit design often requires repeated simulator runs to tune transistor dimensions, bias currents, and compensation components. These sweeps are accurate but slow, especially when evaluating many design candidates or running corner and mismatch checks.

This project follows a complementary approach:

1. Use Cadence/Spectre as the trusted reference.
2. Build a local model that matches the relevant simulator behavior.
3. Use the local solver for fast exploration and optimization.
4. Send only selected candidates back to Cadence for verification.

This makes it easier to understand design trade-offs and iterate on constraints before committing to expensive simulations.

## Optimization Direction

The current optimization strategy is best described as model-based design-space exploration rather than pure machine learning. It uses a calibrated physics-based surrogate model to evaluate candidate designs quickly.

Existing and planned optimization ideas include:

- Random global search over sizing and bias variables.
- Constraint filtering for feasibility.
- Greedy per-device size shrink to reduce area while preserving specifications.
- Pareto selection for area-power or noise-power trade-offs.
- Future extension to differentiable or machine-learning surrogate models.

## Future Work

Planned extensions include:

- Support for more PDKs and transistor compact models.
- More general topology descriptions.
- More advanced DC, AC, transient, and noise solvers.
- Better calibration workflows against simulator data.
- Automated generation of validation reports.
- Interactive graphical interface for exploring trade-offs.
- Integration of machine-learning based surrogate models for faster optimization.

## Intended Use

This repository is meant for research and early-stage analog design exploration. It is not intended to replace a sign-off simulator. Instead, it provides a fast local environment for understanding trends, narrowing the search space, and preparing better candidates for Cadence/Spectre verification.
