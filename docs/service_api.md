# Local Service API

[Project overview](README.md) | [Core Solver Overview](module_overview.md) |
[CLI Reference](cli_reference.md) | [中文版](service_api_zh.md)

> **Status: maintained API reference.** The service is local, in-process, and
> unauthenticated by default.

A local FastAPI HTTP layer over the same solver stack the CLI drives. It is a
**thin adapter** — every route hands a request straight to an existing single
source of truth (`circuit_from_dict`, `analysis_options`, `run_analysis_suite`,
`explore_from_dict`, `mismatch_mc_from_dict`) and carries no numerical logic of
its own. Local client applications can use this layer instead of importing the
solver package directly.

## Quick Start

```bash
# Install the optional serve extra (fastapi + uvicorn)
pip install -e ".[serve]"

# Start the server (defaults: 127.0.0.1:8341, 1 job worker)
circuit-opt serve

# Equivalent module form
python -m circuitopt.service
```

Swagger/OpenAPI docs are served automatically at `http://127.0.0.1:8341/docs`
(FastAPI's built-in UI — no separate schema file to maintain).

```bash
curl http://127.0.0.1:8341/api/v1/health
# {"status":"ok","version":"1.3.0","api":"v1"}
```

### Server flags

| Flag | Default | Notes |
|------|---------|-------|
| `--host` | `127.0.0.1` | Bind address. Loopback only by default; see [Security](#security). |
| `--port` | `8341` | TCP port. |
| `--reload` | off | uvicorn auto-reload for development. `--job-workers` is ignored in this mode (the reloader re-imports the app from `circuitopt.service.app:create_app`, which takes no arguments). |
| `--job-workers` | `1` | Thread-pool size for background jobs (`explore`/`mc`). Solves are CPU-bound (NumPy/Numba); raise only if you have spare cores and want concurrent jobs running. |

All routes are under the `/api/v1` prefix.

## Security

This is a **local, single-user service** — there is no authentication, no
multi-tenant isolation, and no persistence (job history is in-memory and is
lost on restart). `--host` defaults to `127.0.0.1` (loopback only). Passing
`--host 0.0.0.0` exposes the solver — and therefore arbitrary compute — to
your network; do that **at your own risk** and only on a trusted network.

CORS is restricted to `http://localhost:<any port>` and `http://127.0.0.1:<any
port>` (regex `^https?://(localhost|127\.0\.0\.1)(:\d+)?$`) — enough for a
local Vite/Tauri dev front-end without pinning a specific port, and nothing
else is allowed to call the API from a browser context.

## Endpoints at a Glance

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/v1/health` | Liveness + version |
| `GET` | `/api/v1/capabilities` | Self-description: models, analyses, corners, job kinds |
| `POST` | `/api/v1/validate` | Validate a circuit JSON (always 200) |
| `POST` | `/api/v1/solve` | Run the analysis suite synchronously |
| `POST` | `/api/v1/jobs/explore` | Submit a design-space exploration job |
| `POST` | `/api/v1/jobs/mc` | Submit a mismatch Monte-Carlo job |
| `GET` | `/api/v1/jobs` | List jobs (newest first, no result payload) |
| `GET` | `/api/v1/jobs/{id}` | Job status + (once terminal) result/error |
| `DELETE` | `/api/v1/jobs/{id}` | Request cooperative cancellation |
| `WS` | `/api/v1/jobs/{id}/events` | Stream progress, then a terminal frame |

## Synchronous Endpoints

### `GET /api/v1/health`

```bash
curl http://127.0.0.1:8341/api/v1/health
```

```json
{"status": "ok", "version": "1.3.0", "api": "v1"}
```

### `GET /api/v1/capabilities`

The single source of truth for a GUI's dropdowns — nothing here is hardcoded
editorial content, it all reflects what the running build actually supports.

```bash
curl http://127.0.0.1:8341/api/v1/capabilities
```

```json
{
  "version": "1.3.0",
  "api": "v1",
  "models": {"pmos_tft": "circuitopt.pmos_tft_model.PMOS_TFT", "sky130.nmos": "...", "...": "..."},
  "analyses": {
    "ac": ["band", "corner", "freqs", "..."],
    "noise": ["...", "..."],
    "transient": ["...", "..."],
    "pss": ["...", "..."],
    "pac": ["...", "..."],
    "pnoise": ["...", "..."]
  },
  "corners": {
    "otft": ["fast", "slow", "typical"],
    "sky130": ["ff", "fs", "sf", "ss", "tt"],
    "freepdk45": ["ff", "fs", "nom", "sf", "ss", "tt"]
  },
  "jobs": ["explore", "mc"]
}
```

- `models` — a snapshot of the device-model registry (`registered_models()`):
  registered model-type key -> its class's fully qualified name.
- `analyses` — one entry per analysis in `ANALYSIS_ORDER` (`ac`, `noise`,
  `transient`, `pss`, `pac`, `pnoise`); the value is the sorted list of legal
  option keys for that analysis's `analyses.<name>` JSON block
  (`known_keys(name)` from `analysis_options.py`), so a client can validate or
  render a form without duplicating the option registry.
- `corners` — the three process-corner families: `otft` (continuous OTFT PVT
  shift names), `sky130` and `freepdk45` (discrete silicon corners).
- `jobs` — the background job kinds a client can submit (`explore`, `mc`).

### `POST /api/v1/validate`

Parses a circuit and validates its `analyses` block. **The outcome is the
payload — this endpoint always returns HTTP 200.** Errors are collected, not
short-circuited on the first one, so a client can show every problem at once.

Request body: a raw circuit JSON object (the format documented in
[JSON Circuit Description](json_circuit_format.md)) — not wrapped in an
envelope.

```bash
curl -X POST http://127.0.0.1:8341/api/v1/validate \
  -H "Content-Type: application/json" \
  -d @examples/periodic_rc.json
```

```json
{"valid": true}
```

A broken circuit (missing required field, or a typo'd option key inside
`analyses`) still returns 200:

```json
{"valid": false, "errors": ["'solved' is a required property", "..."]}
```

### `POST /api/v1/solve`

Runs `run_analysis_suite` and returns JSON-safe results — this is the
programmatic equivalent of `circuit-opt run`.

Request body:

```json
{
  "circuit": { "...": "circuit JSON object, see json_circuit_format.md" },
  "selected": ["ac", "noise"],
  "corner": "slow"
}
```

`circuit` is required; `selected` (subset of analyses to run) and `corner`
(process-corner override — OTFT `typical`/`slow`/`fast`, or a silicon corner)
are optional. Omit `selected` to run everything the circuit's `analyses` block
configures.

```bash
curl -X POST http://127.0.0.1:8341/api/v1/solve \
  -H "Content-Type: application/json" \
  -d '{"circuit": '"$(cat examples/periodic_rc.json)"', "selected": ["ac"]}'
```

Success (`200`):

```json
{
  "results": {
    "ac": {"Av_dc_dB": 22.90, "bw_Hz": 562.3, "response": [{"re": 1.0, "im": 0.0}, "..."]}
  },
  "elapsed_s": 0.0034
}
```

Failure (`422`) — a parse error (malformed circuit structure) or a solve error
(e.g. DC non-convergence, a bad `analyses` option key) each carry a `stage` so
a client can tell which phase failed. No traceback is ever leaked.

```json
{"detail": {"stage": "parse", "message": "'solved' is a required property"}}
```

```json
{"detail": {"stage": "solve", "message": "unknown option(s) for 'ac': {'bogus_key'}; valid: [...]"}}
```

## Background Jobs

`explore` and mismatch `mc` runs can take seconds to minutes, so they run as
**background jobs**: submit, then poll or stream progress, instead of holding
one HTTP request open. See [Job State Machine](#job-state-machine) below for
lifecycle details.

### `POST /api/v1/jobs/explore`

Same semantics as `circuit-opt explore` — both go through the shared
`explore_from_dict` entry point, so the two surfaces can't drift.

Request body:

```json
{
  "circuit": { "...": "circuit JSON with an 'explore' block" },
  "n": 300,
  "seed": 42,
  "corner": "slow"
}
```

Only `circuit` is required; `n`, `seed`, `corner` fall back to
`explore_from_dict`'s defaults (`n=200`, `seed=0`, no corner) when omitted.

```bash
curl -i -X POST http://127.0.0.1:8341/api/v1/jobs/explore \
  -H "Content-Type: application/json" \
  -d '{"circuit": '"$(cat examples/afe_explore.json)"', "n": 300, "seed": 42}'
```

```
HTTP/1.1 202 Accepted
{"job_id": "a1b2c3d4e5f6", "kind": "explore", "status": "queued"}
```

### `POST /api/v1/jobs/mc`

Same semantics as `circuit-opt mc` (via the shared `mismatch_mc_from_dict`
entry point). Note `corner` here is the **base process corner**
(`typical`/`slow`/`fast`) that the per-device mismatch is layered on top of —
not an OTFT/silicon analysis corner like `jobs/explore`'s `corner`.

```json
{
  "circuit": { "...": "circuit JSON object" },
  "n": 300,
  "seed": 1,
  "corner": "typical"
}
```

```bash
curl -i -X POST http://127.0.0.1:8341/api/v1/jobs/mc \
  -H "Content-Type: application/json" \
  -d '{"circuit": '"$(cat examples/afe_explore.json)"', "n": 300, "seed": 1}'
```

```
HTTP/1.1 202 Accepted
{"job_id": "f6e5d4c3b2a1", "kind": "mc", "status": "queued"}
```

### `GET /api/v1/jobs`

Newest-first list of status snapshots — no `result`/`error` payload (keeps the
listing cheap even with a large `explore`/`mc` result in memory).

```bash
curl http://127.0.0.1:8341/api/v1/jobs
```

```json
{"jobs": [
  {"job_id": "a1b2c3d4e5f6", "kind": "explore", "status": "running",
   "created": 1751000000.0, "started": 1751000000.1, "finished": null,
   "progress": {"type": "progress", "done": 42, "total": 300, "frac": 0.14}}
]}
```

Up to **50 jobs** are retained in memory; once the cap is exceeded, the
oldest already-terminal job is evicted (a running/queued job is never
dropped). A server restart clears all job history — this is a local,
non-persistent service.

### `GET /api/v1/jobs/{id}`

Full status: the same snapshot as the list entry, plus `result` (once
`status == "done"` or `"cancelled"` with partial data) or `error` (once
`status == "failed"`).

```bash
curl http://127.0.0.1:8341/api/v1/jobs/a1b2c3d4e5f6
```

```json
{
  "job_id": "a1b2c3d4e5f6", "kind": "explore", "status": "done",
  "created": 1751000000.0, "started": 1751000000.1, "finished": 1751000012.4,
  "progress": {"type": "progress", "done": 300, "total": 300, "frac": 1.0},
  "result": {"candidates": ["..."], "summary": {"n": 300, "feasible": 87, "pareto": 12}, "objectives": "..."}
}
```

Unknown id -> `404`:

```json
{"detail": {"stage": "job", "message": "unknown job 'deadbeef0000'"}}
```

### `DELETE /api/v1/jobs/{id}`

Request cooperative cancellation. See [Job State Machine](#job-state-machine).

```bash
curl -X DELETE http://127.0.0.1:8341/api/v1/jobs/a1b2c3d4e5f6
```

```json
{"job_id": "a1b2c3d4e5f6", "status": "cancelling"}
```

Unknown id -> `404`; an already-terminal job -> `409` (nothing to cancel), both
with the same `{"stage": "job", "message": ...}` detail shape as above:

```json
{"detail": {"stage": "job", "message": "job 'a1b2c3d4e5f6' already terminal (done)"}}
```

### `WS /api/v1/jobs/{id}/events`

Streams progress frames for a running job, then exactly one terminal frame,
then closes. An unknown job id gets a single error frame and an immediate
close instead.

```bash
# with websocat, or any WS client
websocat ws://127.0.0.1:8341/api/v1/jobs/a1b2c3d4e5f6/events
```

**Frame sequence — `mc` job** (N progress frames, each carrying a running
`partial` summary, then one terminal frame):

```json
{"type": "progress", "done": 1, "total": 300, "frac": 0.0033, "partial": {"n": 1, "latched": 0, "latch_rate": 0.0, "noise_evaluated": 1}}
{"type": "progress", "done": 2, "total": 300, "frac": 0.0067, "partial": {"n": 2, "latched": 0, "latch_rate": 0.0, "noise_evaluated": 2}}
"...": "one frame per completed sample"
{"type": "terminal", "status": "done"}
```

**Frame sequence — `explore` job** (same shape, no `partial` — explore's
progress is just a fraction):

```json
{"type": "progress", "done": 1, "total": 300, "frac": 0.0033}
"...": "one frame per completed candidate"
{"type": "progress", "done": 300, "total": 300, "frac": 1.0}
{"type": "terminal", "status": "done"}
```

**Unknown job id:**

```json
{"type": "error", "message": "unknown job 'nope00000000'"}
```
followed by the socket closing.

**Failed job** terminal frame carries the same `{stage, message}` error shape
used by the synchronous endpoints:

```json
{"type": "terminal", "status": "failed", "error": {"stage": "solve", "message": "..."}}
```

A client connecting after the job already finished (events already drained
from the queue) still gets a synthesized terminal frame reconstructed from the
job's recorded state, so a late subscriber never hangs waiting for a frame
that already happened.

## Job State Machine

```
queued -> running -> { done, failed, cancelled }
```

The three end states are final. `result` is populated on `done` (and on
`cancelled` if partial work was produced); `error` (a `{"stage", "message"}`
envelope, matching the 422 detail shape) is populated on `failed`.

- **Cancellation is cooperative, not a hard kill.** `DELETE
  /api/v1/jobs/{id}` sets a flag; the running candidate/sample already in
  flight always finishes first (there is no safe way to interrupt a NumPy
  solve mid-call). Once the driver notices the flag it stops and returns
  whatever it has produced so far.
- **Partial results are kept.** A cancelled job's `result` (and
  `result.summary`) is flagged `"stopped_early": true`. The actual count
  completed lives at `summary.evaluated` for an `explore` job (`summary.n` is
  the originally *requested* count) and at `summary.n` for an `mc` job (`mc`'s
  `summary.n` is always the count actually evaluated, whether or not the job
  ran to completion — there is no separate "requested" field on that path).
- **A cancel requested while still `queued`** short-circuits before any
  solver work starts and the job goes straight to `cancelled`.
- Job state is entirely **in-memory**; there is no persistence across a
  server restart, and at most 50 jobs are retained (oldest terminal job
  evicted first — see [`GET /api/v1/jobs`](#get-apiv1jobs)).

## Serialization Conventions

Solver results are dicts of NumPy scalars, ndarrays, Python `complex` numbers,
and nested dicts/lists — not strict JSON. Every response is passed through
`circuitopt.service.serialize.to_jsonable`, which applies these rules
recursively:

| Input | Output |
|-------|--------|
| `numpy` scalar (`np.float64`, `np.int64`, `np.bool_`, ...) | native Python scalar |
| `numpy.ndarray` | nested Python `list` (element-wise, so complex/NaN entries are handled too) |
| `complex` (Python or numpy) | `{"re": <float>, "im": <float>}` |
| `NaN`, `+Inf`, `-Inf` (bare or inside a complex/array element) | `null` |
| `dict` key starting with `_`, or any callable value | dropped |
| `bytes` | UTF-8 decoded string (best effort) |
| anything else already JSON-native | passed through unchanged |

The NaN/Inf -> `null` rule exists because strict JSON (RFC 8259) has no
literal for non-finite floats; this guarantees every response parses with a
standard-compliant JSON parser (no reliance on `json.dumps`'s non-standard
`NaN`/`Infinity` tokens). An analysis entry that is `None` (an analysis that
produced no result) is dropped from `POST /solve`'s `results` object entirely.

## CLI Equivalence

Every endpoint here mirrors an existing CLI subcommand — the service layer
adds no new solver behavior, only an HTTP transport over the same entry
points. See [CLI Reference](cli_reference.md) for the full flag reference of
each command.

| HTTP endpoint | CLI equivalent | Shared entry point |
|---------------|-----------------|---------------------|
| `POST /api/v1/solve` | `circuit-opt run` | `run_analysis_suite` |
| `POST /api/v1/jobs/explore` | `circuit-opt explore` | `explore_from_dict` |
| `POST /api/v1/jobs/mc` | `circuit-opt mc` | `mismatch_mc_from_dict` |
| `POST /api/v1/validate` | (no direct CLI equivalent) | `circuit_from_dict` + `validate_analysis_cfg` |
| `GET /api/v1/capabilities` | (no direct CLI equivalent) | `registered_models`, `analysis_options.known_keys`, `device_factory.CORNERS`/`SKY130_CORNERS`, `freepdk45_model.FREEPDK45_CORNERS` |

Because both surfaces call the same underlying functions, a circuit that runs
correctly through `circuit-opt run/explore/mc` on the shell produces identical
results through the corresponding HTTP endpoint (same seed -> same output).

## See Also

- [JSON Circuit Description](json_circuit_format.md) — the schema every
  `circuit` field in this API follows.
- [CLI Reference](cli_reference.md) — the `serve` subcommand and every other
  command line entry point.
- [Core Solver Overview](module_overview.md) — the `service/` subpackage entry
  in the module map, and the solver internals every endpoint calls into.
