# Circuit Optimization Flow

[English](README.md) | [中文说明](README_zh.md)

## Overview

This project aims to build a local circuit modeling, simulation, and optimization flow for analog design-space exploration. The motivation is to reduce the need for exhaustive Cadence/Spectre sweeps during early sizing and biasing iterations, while still using simulator results as the validation reference.

The first use case is an AT4000TG thin-film transistor amplifier design. In that project, a local Python-based model was used to reproduce and analyze key circuit behavior, including operating point, small-signal response, transient response, noise, and design constraints. The repository is intended to grow beyond this initial implementation and become a more general framework for circuit exploration and optimization.

## Current Scope

The current flow includes or is planned to include the following components:

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
