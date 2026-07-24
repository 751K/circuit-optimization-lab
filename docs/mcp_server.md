# MCP Server

[Documentation Home](README.md) | [CLI Reference](cli_reference.md) |
[中文版](mcp_server_zh.md)

The optional Model Context Protocol server lets an LLM client validate,
simulate, explore, and sign off circuits through the same application
operations used by the local HTTP service. It is a protocol adapter: numerical
work still runs in the Rust solver and native BSIM backend.

## Install and Start

```bash
uv pip install -e ".[mcp]"

# Recommended local-client transport
circuit-opt mcp --transport stdio --workspace .

# Equivalent dedicated/module entry points
circuit-opt-mcp --workspace .
python -m circuitopt.mcp --workspace .
```

For a local Streamable HTTP client:

```bash
circuit-opt mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8342 \
  --workspace .
```

The HTTP transport intentionally rejects non-loopback bind addresses because
this local server has no authentication.

## Client Configuration

Configure the client to launch the executable from the environment where
Circuit Optimization and `circuitopt-core` are installed:

```json
{
  "mcpServers": {
    "circuit-optimization": {
      "command": "/absolute/path/to/project/.venv/bin/circuit-opt-mcp",
      "args": [
        "--workspace",
        "/absolute/path/to/project"
      ]
    }
  }
}
```

Launch configuration paths are machine-specific. Circuit manifests and
testbench references remain project-relative and portable.

## Tools

| Tool | Behavior |
|---|---|
| `get_capabilities` | List installed models, analyses, option keys, corners, and job kinds |
| `validate_circuit` | Parse circuit JSON and validate analysis/signoff contracts |
| `run_analysis` | Run selected analyses synchronously and return a bounded summary |
| `submit_exploration` | Queue design-space exploration |
| `submit_mismatch_mc` | Queue mismatch Monte Carlo |
| `submit_signoff` | Queue a workspace-relative multi-testbench PVT campaign |
| `list_jobs` | List background jobs |
| `get_job` | Poll state and optionally retrieve/save a result |
| `cancel_job` | Request cooperative cancellation |
| `inspect_signoff_result` | Filter a saved signoff artifact by case and PVT |

`run_analysis(save_result=true)` writes the complete serialized result under
`results/mcp/`; its direct tool response keeps scalar measurements and compacts
long vectors. `submit_signoff` always writes its complete result there. This
keeps protocol responses bounded while preserving waveforms and all PVT rows.

## Resources

- `circuitopt://capabilities`: JSON capability snapshot.
- `circuitopt://workflow`: concise tool-use sequence for an LLM client.

## Jobs and Cancellation

Exploration, mismatch MC, and signoff use the shared in-process `JobManager`.
The states are:

```text
queued -> running -> done | failed | cancelled
```

Cancellation is cooperative. A candidate, MC sample, or PVT point already in
flight completes before the job stops. Signoff writes a partial result with
`stopped_early: true`; invalid model evaluation and non-convergence remain
explicit invalid outcomes.

## File Security

- `--workspace` defines the only file tree MCP tools may access.
- Tool paths must be relative; absolute paths and `..`/symlink escapes fail.
- Signoff campaign files must be JSON.
- Generated artifacts are confined to `results/mcp/`, which is Git-ignored.
- The server does not expose arbitrary shell execution or unrestricted file
  reading.

The MCP and FastAPI adapters share
`circuitopt.service.operations`; neither contains circuit equations or solver
fallbacks.
