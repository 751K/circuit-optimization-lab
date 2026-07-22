# CLI Reference

[Documentation Home](README.md) | [Getting Started](getting_started.md) |
[Circuit JSON Format](json_circuit_format.md) | [中文版](cli_reference_zh.md)

This document covers only the currently public commands. Treat the actual
`--help` output as the final source of truth:

```bash
circuit-opt --help
python -m circuitopt --help
```

The two entry points are equivalent. The rest of this document uses
`circuit-opt` throughout.

## Command Overview

| Command | Purpose |
|---|---|
| `run` | Run AC, noise, transient, PSS, PAC, PNoise per the JSON config |
| `explore` | Sample, solve, filter by constraints, and generate the Pareto front from an `explore` block |
| `corners` | Fixed `typical/slow/fast` process-corner sweep for AT4000TG |
| `mc` | Per-device mismatch Monte Carlo for AT4000TG |
| `chopper` | Ideal, static-phase, LPTV, PSS, PAC, PNoise, and transient flows for the AFE chopper |
| `adc` | SAR ADC single conversion, static sweep, sine-wave dynamic test, mismatch MC, and design exploration |
| `plot` | Render the built-in AFE/chopper waveform and Bode plots |
| `dataset` | Build a surrogate dataset with provenance |
| `serve` | Start the optional local FastAPI service |

The legacy no-subcommand form still auto-routes to `run`:

```bash
circuit-opt examples/periodic_rc.json
```

New docs and scripts should spell out `run` explicitly.

## `run`

```bash
circuit-opt run CIRCUIT.json [options]
```

Common examples:

```bash
# Run every analysis in the JSON analyses block
circuit-opt run examples/periodic_rc.json

# Run only AC and noise
circuit-opt run examples/periodic_rc.json --analysis ac,noise

# Override the process corner and write JSON output
circuit-opt run examples/tsmc28hpcp_5t_ota.json \
  --analysis ac,noise --corner ss --output results/tsmc28_ss.json
```

Flags:

| Flag | Description |
|---|---|
| `-a`, `--analysis` | Comma-separated subset of analyses: `ac,noise,transient,pss,pac,pnoise` |
| `--corner` | Process-corner override; `typical/slow/fast` for AT4000TG, silicon PDKs use their own supported corners |
| `--noise-band LO HI` | Noise integration band for the CLI summary, default `0.05 100.0` Hz |
| `-o`, `--output` | Write results to JSON |
| `--engine {rust}` | Compute engine; only `rust` as of v2.0.0 (omitting the flag defaults to `rust`) |
| `--no-numba` | **Removed in v2.0.0 (errors out)**: the numba engine no longer exists; use `--engine rust` instead |
| `--quiet` | Suppress progress and summary output |

`run`'s specific numeric options come from the JSON top-level `analyses`
block. See [Circuit JSON Format](json_circuit_format.md) for the fields.

## `explore`

```bash
circuit-opt explore CONFIG.json [options]
```

The config file must be a complete circuit JSON that includes an `explore`
block.

```bash
circuit-opt explore examples/afe_explore.json -n 500 --seed 1
circuit-opt explore examples/sky130_5t_ota.json -n 200 --corner ss
circuit-opt explore examples/tsmc28hpcp_5t_ota.json -n 200 --corner ff
```

Flags:

| Flag | Default | Description |
|---|---:|---|
| `-n`, `--n` | `200` | Number of candidates |
| `--seed` | `0` | RNG seed |
| `--method` | `lhs` | `lhs` or `random` |
| `--corner` | no override | Process corner used for solving |
| `-o`, `--out`, `--output` | none | Output prefix; writes `<prefix>.csv` and `<prefix>.jsonl` |
| `--quiet` | off | Don't print per-candidate progress |
| `--engine {rust}` | `rust` | Only `rust` as of v2.0.0 (omitting the flag defaults to `rust`) |
| `--no-numba` | — | Removed in v2.0.0 (errors out): the numba engine no longer exists |

See the `explore` field in the JSON documentation for explorable variables,
constraints, and objectives.

## `corners`

```bash
circuit-opt corners CIRCUIT.json [options]
```

The current implementation calls `circuitopt.corners.corner_table`, a fixed
sweep over AT4000TG's `typical/slow/fast`. It is not a general-purpose
silicon PVT campaign driver. For silicon PDKs, use `run --corner`,
`explore --corner`, or a dedicated campaign under `experiments/`.

```bash
circuit-opt corners examples/afe_explore.json \
  --freqs-start 0.01 --freqs-stop 10000 --freqs-num 121 \
  --noise-band 0.05 100 --output results/afe_corners.csv
```

Flags:

| Flag | Default | Description |
|---|---:|---|
| `--freqs-start` | `0.01` | Start frequency, Hz |
| `--freqs-stop` | `10000` | Stop frequency, Hz |
| `--freqs-num` | `121` | Number of log-spaced frequency points |
| `--noise-band LO HI` | `0.05 100.0` | IRN integration band |
| `-o`, `--output` | none | CSV output |
| `--workers` | `1` | Number of parallel corner workers (`ThreadPoolExecutor`; each solve releases the GIL independently); there are only 3 corners, so going beyond 3 has no benefit |
| `--engine {rust}` | `rust` | Only `rust` as of v2.0.0 (omitting the flag defaults to `rust`) |
| `--no-numba` | — | Removed in v2.0.0 (errors out): the numba engine no longer exists |
| `--quiet` | off | Suppress per-corner output |

## `mc`

```bash
circuit-opt mc CIRCUIT.json [options]
```

The current general-purpose `mc` uses AT4000TG's `mvt0`/`mbeta0` continuous
mismatch model and the AFE latch criterion. It is not a general-purpose
foundry mismatch engine.

```bash
circuit-opt mc examples/afe_explore.json \
  -n 300 --seed 1 --corner slow --output results/afe_mc.json
```

Flags:

| Flag | Default | Description |
|---|---:|---|
| `-n`, `--n` | `200` | Number of MC samples |
| `--seed` | `0` | RNG seed |
| `--workers` | `1` | Number of parallel MC workers (`ThreadPoolExecutor`; each solve releases the GIL independently); mismatch draws are pre-sampled before dispatch, so results are byte-identical regardless of worker count |
| `--corner` | `typical` | `typical`, `slow`, or `fast` |
| `--freqs-start/stop/num` | `0.01/10000/121` | AC/noise grid |
| `--noise-band LO HI` | `0.05 100.0` | IRN integration band |
| `-o`, `--output` | none | JSON summary |
| `--engine {rust}` | `rust` | Only `rust` as of v2.0.0 (omitting the flag defaults to `rust`) |
| `--no-numba` | — | Removed in v2.0.0 (errors out): the numba engine no longer exists |
| `--quiet` | off | Suppress progress output |

SAR ADC has its own separate `adc --mc` flow, whose mismatch semantics come
from the JSON `adc.mismatch` block.

## `chopper`

```bash
circuit-opt chopper CIRCUIT.json --level LEVEL [options]
```

`LEVEL`:

| Level | Meaning |
|---|---|
| `ideal` | Ideal square-wave LPTV |
| `pmos` | PMOS switch static phase |
| `lptv` | PMOS sideband folding |
| `pss` | Shooting PSS |
| `pac` | PAC on the PSS orbit |
| `pnoise` | Periodic noise after PSS/PAC |
| `transient` | Hard-switched transient |

```bash
circuit-opt chopper examples/afe_explore.json --level ideal
circuit-opt chopper examples/afe_explore.json --level pnoise \
  --f-chop 225 --max-sideband 10
circuit-opt chopper examples/afe_explore.json --level transient \
  --n-periods 8 --n-points 121
```

Main flags:

| Flag | Default |
|---|---:|
| `--f-chop` | `225` Hz |
| `--switch-w` / `--switch-l` | `5000` / `30` µm |
| `--edge-time` | `20e-6` s |
| `--max-harmonic` | `31` |
| `--max-sideband` | `10` |
| `--tstab-periods` | `2` |
| `--n-points` | `121` |
| `--n-periods` | `8` |
| `--freqs-start/stop/num` | `0.01/10000/121` |
| `--noise-band LO HI` | `0.05 100.0` Hz |
| `-o`, `--output` | Write results to JSON |
| `--engine {rust}` | Only `rust` as of v2.0.0 (omitting the flag defaults to `rust`) |
| `--no-numba` | Removed in v2.0.0 (errors out): the numba engine no longer exists |
| `--quiet` | Suppress summary output |

This command is a wrapper around the project's AFE chopper; it is not the
sole entry point for arbitrary JSON periodic circuits. For general periodic
circuits, configure `periodic` and `analyses` in the JSON first, then use
`run`.

## `adc`

```bash
circuit-opt adc CIRCUIT.json MODE [options]
```

The modes are mutually exclusive:

```bash
# Single conversion
circuit-opt adc examples/freepdk45_sar3.json --vin 0.7

# Static ramp, DNL, INL, and missing codes
circuit-opt adc examples/freepdk45_sar6.json --sweep 64 --workers 8

# Coherent sine wave, SNDR, SFDR, and ENOB
circuit-opt adc examples/freepdk45_sar6.json \
  --sine 128 --tone-bin 13 --sample-rate 10e6 --workers 8

# MC using the adc.mismatch config
circuit-opt adc examples/freepdk45_sar6.json \
  --mc 32 --seed 1 --workers 8

# Design-space exploration
circuit-opt adc examples/freepdk45_sar6.json \
  --explore examples/freepdk45_sar6_explore.json -n 20 --workers 4
```

Main flags:

| Flag | Description |
|---|---|
| `--vin VIN` | Single conversion; defaults to running one conversion at 0.5 V when no mode flag is given |
| `--sweep N` | N uniformly spaced ramp inputs |
| `--sine N` | N coherent sine-wave samples |
| `--mc N` | N per-device mismatch MC trials |
| `--explore CONFIG` | Standalone SAR-explore config JSON; runs ADC design-space exploration; mutually exclusive with `--vin`/`--sweep`/`--sine`/`--mc` |
| `--tone-bin` | Coherent-input FFT bin, default 3 |
| `--sample-rate` | Sample rate reported in the results, default 10 MHz |
| `--amplitude` | Sine peak amplitude, default `0.45*vref` |
| `--offset` | Sine DC offset, default `0.5*vref` |
| `--corner` | The current ADC CLI accepts `nom/ss/ff` |
| `-n`, `--n` | Number of candidates in `--explore` mode, default `50` |
| `--seed` | RNG seed in `--explore` mode, default `0` |
| `--workers` | Concurrency for conversions or candidates; bit decisions within a single conversion stay serial. `--mc` prefers the compiled Rust batch path (`circuitopt_core.CompiledSarConversion.evaluate_batch`, a single Rayon pool, results byte-identical regardless of worker count); when the conditions aren't met (non-native devices, an incomplete DC seed, etc.) it falls back to `ThreadPoolExecutor` per-trial solving; `--sweep`/`--sine`/`--explore` use `ThreadPoolExecutor` per-candidate solving |
| `--plot [DIR]` | Write the corresponding PNG; needs the `plot` extra |
| `--csv` / `--jsonl` | ADC explore output |
| `-o`, `--output` | JSON result |

The ADC control state machine runs in Python; the comparator, CDAC, and
switches are still computed by transistor-level transient analysis. The
current flow is not equivalent to a full transistor-level digital SAR
controller.

## `plot`

```bash
circuit-opt plot [all|transient|bode|afe|chopper|ac|pac] [options]
```

This command renders the project's built-in AFE/chopper examples; it does
not read an arbitrary circuit JSON.

```bash
uv pip install -e ".[plot]"
circuit-opt plot bode --npts 121 --out-dir results
circuit-opt plot chopper --f-chop 225 --input-diff 1e-3
```

Flags:

| Flag | Default | Description |
|---|---:|---|
| `--f0` | `10` Hz | AFE transient sine frequency |
| `--amp` | `5e-4` V | AFE transient differential half-amplitude |
| `--f-chop` | `225` Hz | Chopper frequency used by the chopper/pac plots |
| `--input-diff` | `1e-3` V | Chopper transient DC differential input |
| `--npts` | per-plot default | Number of Bode frequency points |
| `--out-dir` | `results` | Output directory |
| `--engine {rust}` | `rust` | Only `rust` as of v2.0.0 (omitting the flag defaults to `rust`) |
| `--no-numba` | — | Removed in v2.0.0 (errors out): the numba engine no longer exists |
| `--quiet` | off | Suppress summary output |

## `dataset`

```bash
circuit-opt dataset CONFIG.json [options]
```

The input must be a complete circuit JSON with an `explore` block. Each
sample retains its design variables, labels, failure status, and
provenance.

```bash
circuit-opt dataset examples/single_stage.json \
  -n 500 --seed 1 --labels ac_noise --out results/datasets/single

circuit-opt dataset examples/sky130_chopper.json \
  -n 200 --labels pss,pac,pnoise --out results/datasets/sky_chopper
```

Flags:

| Flag | Default | Description |
|---|---:|---|
| `-n`, `--n` | `200` | Number of samples |
| `--seed` | `0` | RNG seed |
| `--workers` | `1` | Number of parallel candidate workers (`ThreadPoolExecutor`; each solve releases the GIL independently) |
| `--method` | `lhs` | `lhs` or `random` |
| `--corner` | `typical` | Solve corner; silicon PDKs may pass their own corner |
| `--labels` | `ac_noise` | Any combination of `ac_noise,transient,pss,pac,pnoise` |
| `--freqs-start` | `-2` | AC start decade |
| `--freqs-stop` | no override | AC stop decade |
| `--freqs-num` | `101` | AC point count |
| `--out` | automatic | Output prefix |
| `--no-npz` | off | Don't write the dense NPZ |
| `--parquet` | off | Also write a Parquet table; needs the `parquet` extra |
| `--quiet` | off | Suppress progress |
| `--engine {rust}` | `rust` | Only `rust` as of v2.0.0 (omitting the flag defaults to `rust`) |
| `--no-numba` | — | Removed in v2.0.0 (errors out): the numba engine no longer exists |

## Surrogate and Optimization

These are standalone module entry points, not subcommands of the main CLI:

```bash
uv pip install -e ".[ml]"

python -m circuitopt.surrogate train \
  results/datasets/single.npz --out results/models/single.pkl

python -m circuitopt.surrogate predict \
  results/models/single.pkl --x 2000,1500,25

python -m circuitopt.optimize \
  examples/single_stage.json results/models/single.pkl \
  --n-screen 100000 --top-k 20
```

PyTorch variant:

```bash
uv pip install -e ".[torch]"
python -m circuitopt.surrogate_torch --help
```

The surrogate is only for screening or gradient-based search; final
feasibility should always be re-verified against the physical solver.

## `serve`

```bash
uv pip install -e ".[serve]"
circuit-opt serve
```

Flags:

| Flag | Default | Description |
|---|---:|---|
| `--host` | `127.0.0.1` | Listen address |
| `--port` | `8341` | TCP port |
| `--reload` | off | uvicorn development auto-reload |
| `--job-workers` | `1` | Number of background workers for `explore`/`mc` |

`0.0.0.0` exposes the unauthenticated service to the network and should not
be used as a default configuration. See [Local Service API](service_api.md)
for the full protocol.

## Calibration and Benchmarks

Calibration regression:

```bash
python -m circuitopt.calibration --all
python -m circuitopt.calibration --all --json
python -m circuitopt.calibration calibration/amp_design3_typical/ --analyses ac,noise
```

Performance benchmarks:

```bash
python -m benchmarks.bench_afe --warm-runs 3
python -m benchmarks.bench_model --warm-runs 3
python -m benchmarks.bench_periodic --warm-runs 3
python -m benchmarks.bench_chopper --warm-runs 3
python -m benchmarks.bench_sweep --n-candidates 200
```

Performance numbers are affected by Python, Numba, CPU, cold-start, and
cache state. See [Environment & Benchmarks](environment_performance.md) for
historical measurements.

## Exit Codes and Output

- Returns 0 on success; returns non-zero for argument errors, missing
  files, analysis failures, or failed calibration.
- `run` outputs JSON; `corners` outputs CSV; `explore` outputs CSV/JSONL;
  `dataset` outputs JSONL/manifest/NPZ, with optional Parquet.
- In numeric results, frequency is in Hz, time is in seconds, voltage is in
  V, and current is in A.
- Noise integration results must always record the integration band
  alongside them.
