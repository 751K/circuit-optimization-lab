# Getting Started

[Documentation Home](README.md) | [中文](getting_started_zh.md)

This guide gets the repository running without requiring any external PDK.

## Requirements

- Python 3.10 or newer.
- `uv` is recommended for environment management; standard `venv` and `pip`
  remain supported.
- A supported compiler only when a native compact-model backend must be built
  for the first time.
- Optional PDK files and external tools only for the corresponding silicon
  process. See the [PDK Support Matrix](pdk_support.md).

## Install With uv

From the repository root:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Install the development and test dependencies when modifying the project:

```bash
uv pip install -e ".[dev]"
```

Useful optional groups:

```bash
uv pip install -e ".[ml]"       # scikit-learn surrogate
uv pip install -e ".[torch]"    # differentiable PyTorch surrogate
uv pip install -e ".[plot]"     # matplotlib plotting
uv pip install -e ".[serve]"    # FastAPI and uvicorn
uv pip install -e ".[parquet]"  # Parquet dataset export
```

## Install With Standard venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Verify the Core Installation

The passive RC example needs no transistor model or PDK:

```bash
circuit-opt run examples/periodic_rc.json --analysis ac,noise
```

Run all analyses configured in that JSON:

```bash
circuit-opt run examples/periodic_rc.json
```

The package module entry is equivalent:

```bash
python -m circuitopt run examples/periodic_rc.json --analysis ac,noise
```

For a test-level check:

```bash
pytest -q tests/test_cli_subcommands.py tests/test_periodic_solvers.py
```

## Run a Transistor Example

The default transistor process is the built-in AT4000TG PMOS model:

```bash
circuit-opt run examples/single_stage.json --analysis ac,noise
```

Other processes are selected per device through the JSON `models` field. They
are not global simulator switches. See the [Circuit JSON Format](json_circuit_format.md)
and [PDK Support Matrix](pdk_support.md).

## Typical Workflows

```bash
# Design-space exploration
circuit-opt explore examples/afe_explore.json -n 200 --seed 1

# Process corners for the circuit's configured process
circuit-opt corners examples/afe_explore.json

# AT4000TG mismatch Monte Carlo
circuit-opt mc examples/afe_explore.json -n 100 --seed 1

# Build a surrogate dataset
circuit-opt dataset examples/single_stage.json -n 500 --out results/datasets/single

# Start the optional local API
circuit-opt serve
```

See the [CLI Reference](cli_reference.md) before applying a workflow to a silicon
PDK: corner and mismatch support differs by backend.

## Paths and Portability

The project does not require machine-specific absolute paths in circuit JSON.
Resolution follows explicit environment variables, the active virtual
environment, the project `.venv`, and documented project-local conventions.

Common variables:

| Variable | Purpose |
|---|---|
| `PDK_ROOT` | Root for SKY130 and FreePDK45 installations |
| `TSMC28_MODEL_DIR` | Directory containing the supported TSMC HSPICE model file |
| `TSMC28_PDK_ROOT` | Outer TSMC iPDK or delivery root |
| `NGSPICE_BIN` | Explicit ngspice executable for backends or oracle comparisons |
| `OPENVAF_BIN` / `OPENVAF_ROOT` | OpenVAF-Reloaded compiler or source checkout |
| `BSIM4_VA` | Explicit BSIM4 Verilog-A source |
| `OSDI_CACHE_DIR` | OSDI build cache |
| `CIRCUITOPT_NATIVE_MODEL_CACHE` | Native compact-model build cache |

Do not commit licensed model files, generated model cards, local virtual
environments, or simulator caches.

## Where to Go Next

- Write a circuit: [Circuit JSON Format](json_circuit_format.md)
- Run analyses: [CLI Reference](cli_reference.md)
- Select a process: [PDK Support Matrix](pdk_support.md)
- Call from another application: [Local Service API](service_api.md)
- Modify the codebase: [Developer Handoff Guide](development.md)
