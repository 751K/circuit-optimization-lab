# Developer Handoff Guide

[Documentation Home](README.md) | [Core Solver Overview](module_overview.md)

This guide is the shortest route from “the project runs” to “I can change it
without breaking another backend.”

## Repository Map

| Path | Responsibility |
|---|---|
| `circuitopt/circuit_loader.py` | JSON loading and `CircuitSpec` construction |
| `circuitopt/topology.py` | Circuit graph and element definitions |
| `circuitopt/device_model.py` | Device-model interface and PDK registry |
| `circuitopt/device_factory.py` | Per-device model construction and binding |
| `circuitopt/dc_solver.py` | Nonlinear operating point |
| `circuitopt/ac_solver.py` / `ac_mna.py` | Small-signal MNA |
| `circuitopt/noise_solver.py` | Stationary noise |
| `circuitopt/transient_solver.py` | Native transient dispatch and integration |
| `circuitopt/pss_solver.py` | Shooting periodic steady state |
| `circuitopt/pac_solver.py` | Periodic small-signal conversion |
| `circuitopt/pnoise_solver.py` | Cyclostationary noise |
| `circuitopt/spice/` | HSPICE library parser and elaborator |
| `circuitopt/compact_models/bsim4/` | Native BSIM4 ABI and implementation |
| `circuitopt/pdk/tsmc28/` | TSMC-specific model resolution and device adapter |
| `circuitopt/ngspice_*.py` | External ngspice characterization and oracle paths |
| `circuitopt/osdi_*.py` | OpenVAF/OSDI compact-model host |
| `circuitopt/dataset.py` | Dataset generation and provenance |
| `circuitopt/surrogate*.py` / `optimize.py` | Surrogate training and optimization |
| `circuitopt/service/` | Optional local HTTP layer |
| `examples/` | Runnable circuits and scripts |
| `experiments/` | Longer campaigns and comparison drivers |
| `tests/` | Unit, regression, and backend tests |
| `calibration/` | Archived Cadence/Spectre reference data |

## Development Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Optional model toolchains are not required for the core test suite. Tests that
need unavailable PDK payloads should skip with an actionable reason.

## Test Strategy

Start with the narrowest relevant tests:

```bash
pytest -q tests/test_json_circuit.py
pytest -q tests/test_cli_subcommands.py
pytest -q tests/spice tests/test_tsmc28.py tests/test_tsmc28_5t_ota.py
```

Then run the full suite:

```bash
pytest -q
ruff check .
```

Documentation:

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

Use `git diff --check` before committing.

## Change Boundaries

### Adding a JSON field

1. Update `schemas/circuit.schema.json`.
2. Update `circuitopt/circuit_loader.py`.
3. Thread the field through `CircuitSpec` or the relevant analysis configuration.
4. Add schema, loader, and behavior tests.
5. Update both JSON format documents.

### Adding an analysis option

1. Define and validate it in `analysis_options.py`.
2. Route it through `analysis_dispatch.py`.
3. Add solver-level and CLI-level tests.
4. Update the CLI and JSON references.

### Adding a PDK

1. Keep process-specific parsing and file resolution outside the numerical
   solver modules.
2. Implement the `TransistorModel` contract and register model keys through
   `register_pdk`.
3. State whether the backend exposes terminal conductance, charge,
   capacitance, and correlated noise.
4. Define corner normalization and fail on unknown corners.
5. Add a small circuit benchmark before an OTA-scale benchmark.
6. Document required local files without embedding machine paths or licensed
   parameters.

### Changing a Solver

Check every backend capability path touched by the change:

- built-in AT4000TG;
- OSDI/SKY130;
- characterized-grid FreePDK45;
- native TSMC28 BSIM4;
- explicit ngspice oracle helpers.

A solver optimization must preserve numerical behavior within an explicit
tolerance. Do not use benchmark speed as a substitute for a regression check.

## Result and Cache Hygiene

- PDK payloads, generated parameter cards, virtual environments, and caches are
  local artifacts.
- `results/` contains experiment outputs, not immutable truth. A design document
  must identify the exact CSV or fixture it summarizes.
- Avoid writing foundry model parameters into logs, docs, fixtures, or committed
  caches.
- Keep path resolution in `circuitopt/toolchain.py`; do not add per-machine
  absolute paths to examples.

## Documentation Maintenance

The maintained public entry points are:

- `docs/README.md` and `docs/README_zh.md`;
- getting-started guides;
- CLI reference;
- JSON format;
- PDK support matrix;
- architecture overview;
- service API;
- TSMC adapter guide.

Design records and dated performance notes must carry a visible status. Completed
implementation plans and speculative roadmaps should live in issue tracking or
version history, not in the formal documentation navigation.

## Release Checks

Before a release:

1. Confirm `pyproject.toml`, the package version, README badge, service health
   version, and changelog agree.
2. Run the full tests and lint.
3. Build the documentation with `mkdocs build --strict`.
4. Re-run representative passive, AT4000TG, and available silicon smoke tests.
5. Verify that ignored PDK payloads are not staged.
6. Mark partial experiment campaigns as partial.
