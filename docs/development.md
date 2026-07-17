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
| `circuitopt/ngspice_*.py` | Explicit external ngspice characterization and oracle paths |
| `circuitopt/osdi_*.py` | Explicit OpenVAF/OSDI compact-model oracle host |
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

The default pytest configuration excludes external simulator oracles. Normal
PDK tests exercise the in-process C BSIM path and should pass even when
`NGSPICE_BIN` points to a nonexistent executable. Run the optional comparison
suite explicitly when changing compact-model equations, card parsing, or an
oracle adapter:

```bash
pytest -q -m ngspice_oracle
```

That marker includes ngspice and OpenVAF/OSDI comparison workflows and may
require local PDK payloads. It is intentionally outside the routine test gate
because full-circuit subprocess campaigns are much slower than native tests.

The default run also excludes `heavy_e2e`: complete SAR/ADC conversions on the
native silicon backend (minutes per test on a machine with FreePDK45 cards —
they made the default suite take ~22 min instead of ~2 min). The inventory
lives in `tests/conftest.py`. Run them explicitly when touching the SAR/ADC
workflow, the transient solver, or the native BSIM4 backend:

```bash
pytest -q -m heavy_e2e
```

Documentation:

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

Use `git diff --check` before committing.

## Rust Core Scaffolding

The `rust/` workspace hosts the in-progress compiled core: `co-core` (solver
kernels, filled in R3), `co-bsim4` (Berkeley BSIM4.5 host, **live as of R2**),
and `co-py` — the `circuitopt_core` PyO3 extension. `co-bsim4` compiles the
*unmodified* vendored Berkeley C (the same translation units `native.py`
builds, minus `host.c`) at build time via the `cc` crate and reimplements the
`host.c` adapter layer in Rust (parameter binding, internal-node reduction,
terminal I/G/Q/C extraction, noise combination); `bindgen` derives the shared
struct layouts so the port keeps an identical ABI with the compiled C. The
solver hot path still runs through numba until R3.

Toolchain: stable Rust from rustup with the `rustfmt` and `clippy` components,
plus `maturin` for the Python bridge. Building `co-bsim4` also needs a C
compiler (clang/gcc) and `libclang` for `bindgen`.

```bash
cd rust
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# Build + install circuitopt_core into the active venv for local testing
maturin develop --release -m crates/co-py/Cargo.toml
```

Engine selection: the solvers run on one of three engines, chosen by the CLI
`--engine` flag or the `CIRCUIT_ENGINE` environment variable (precedence
argv > env > default):

- `numba` (default) — the JIT kernels;
- `python` — the pure-Python fallback, same switch as `--no-numba`;
- `rust` — requires `circuitopt_core`; when it is not installed the run warns
  once and falls back to `numba`.

`circuitopt.current_engine()` reports the resolved engine. CI lints the
workspace (`cargo fmt` + `clippy`), installs `circuitopt_core` in the test
matrix so the rust-present tests run, and the release workflow archives
per-OS `circuitopt_core` wheels as build artifacts (attached to GitHub
Releases only from R6).

### BSIM4.5 backend selector (R2)

Independently of `CIRCUIT_ENGINE`, the native BSIM4.5 compact model chooses
its numerical backend from `CIRCUIT_BSIM4_BACKEND`, read on **every**
evaluation (never baked at import, mirroring `ngspice_chain_enabled`):

- `cc` (default) — the historical path: `native.py` compiles the vendored C
  at runtime with the system compiler and calls it through `ctypes`;
- `rust` — the `co-bsim4` port, reached by loading the compiled
  `circuitopt_core` extension and binding the identical `co_bsim4_*` C ABI.
  Requires `maturin develop` to have installed the extension; otherwise a
  clear `Bsim4NativeError` names the build command.

Both backends produce the same results (currents/conductance/charges
bit-identical to the reference C; the complex AC solve within ~1 ULP). Build
the extension before selecting `rust`:

```bash
maturin develop --release -m rust/crates/co-py/Cargo.toml
CIRCUIT_BSIM4_BACKEND=rust python -m pytest tests/compact_models/bsim4
```

## Version Management

`pyproject.toml` is the canonical source for the project version. Do not edit
the frontend, npm lockfile, or Tauri versions by hand.

```bash
# Show or verify the current version
python tools/version.py show
python tools/version.py check

# Set one version everywhere
python tools/version.py set 1.4.0

# Prepare a release: set all versions and archive Unreleased changelog entries
python tools/version.py release 1.4.0
```

`set` synchronizes `pyproject.toml`, `frontend/package.json`,
`frontend/package-lock.json`, `frontend/src-tauri/Cargo.toml`,
`frontend/src-tauri/tauri.conf.json`, and `rust/Cargo.toml` (the
`[workspace.package]` version every Rust member crate inherits). `release`
also creates the dated changelog heading and comparison links. CI rejects version drift, and the
release workflow rejects a tag that does not match the canonical version.

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
- native SKY130 BSIM4;
- native FreePDK45 BSIM4;
- native TSMC28 BSIM4;
- explicit OSDI and ngspice oracle paths;
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

1. Run `python tools/version.py release X.Y.Z`, then
   `python tools/version.py check --tag vX.Y.Z`.
2. Run the full tests and lint.
3. Build the documentation with `mkdocs build --strict`.
4. Re-run representative passive, AT4000TG, and available silicon smoke tests.
5. Verify that ignored PDK payloads are not staged.
6. Mark partial experiment campaigns as partial.
