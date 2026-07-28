# Agent Guide

[Documentation Home](README.md) | [CLI Reference](cli_reference.md) |
[JSON Circuit Format](json_circuit_format.md) | [MCP Server](mcp_server.md)

This page is written for an autonomous agent driving this codebase, not for a
person reading over its shoulder. It covers what the tools cost, which result
means what, and the failure modes that produce a confident wrong answer rather
than an error.

## What this codebase is

A local analog-circuit simulator and optimizer. A circuit is a JSON file. The
solvers run in a compiled Rust core (`circuitopt_core`); Python describes the
problem, Rust computes it. There is no cloud service and no licensed simulator
in the loop — foundry model files are read from the local machine and never
written into the repository.

Three things follow from that, and they shape every task you will be given:

- **Runs are cheap enough to iterate on.** A full 45-point PVT signoff of a
  40-transistor amplifier takes about 11 seconds. You are expected to measure
  rather than reason about what the circuit will do.
- **Results are unit-bearing and gated.** An analysis either produces a valid
  measurement or raises `SimulationInvalid`. There is no third state.
- **Nothing is inferred.** Phase margin, settling time, integrated noise and
  saturation checks come only from an explicit `signoff` block. The tools will
  not guess them from an AC response or the last transient sample.

## The rule that matters most

**Never convert a failure into a number.**

Non-convergence, a NaN, a missing measurement, an unsupported topology — each
of these is a result. It is `invalid`, and `invalid` is more severe than
`fail`. If you catch `SimulationInvalid` and substitute a default, a fallback,
or a previous run's value, you have produced a report that says the design
works when nobody knows whether it does. The campaign layer is built to make
this hard: an invalid case poisons its PVT point, and an invalid point poisons
the campaign status.

The same applies to specifications. If a design misses a spec, report the miss.
Do not widen the limit in the JSON to make the run go green unless the user
asked you to change the specification, and say so explicitly when you do.

## Two ways in

| Surface | Use it when | Entry point |
| --- | --- | --- |
| CLI | You have a shell and a checkout. This is the fuller surface. | `circuit-opt <subcommand>` |
| MCP | You are a tool-calling agent with a workspace directory. | `circuit-opt mcp` |

The CLI has 13 subcommands; the MCP server exposes 12 tools over a
path-confined workspace. MCP covers the common design loop and refuses
absolute paths and `..`; the CLI covers everything, including the parts that
need arbitrary file access.

### CLI subcommands by cost

| Cost | Subcommands | Notes |
| --- | --- | --- |
| Milliseconds, no solver | `passive-bom`, `plot` | Read JSON or an existing result. |
| Seconds | `run`, `adc`, `chopper` | One circuit, one analysis suite. |
| Seconds to minutes | `signoff`, `corners`, `mc`, `explore`, `dataset` | Scales with grid size × workers. |
| Hours | `verify-engine` on a large case | Spawns ngspice; use for final cross-checks only. |

### MCP tools

| Tool | Cost | Purpose |
| --- | --- | --- |
| `get_capabilities` | free | Installed models, PDK corners, analyses, job kinds. |
| `validate_circuit` | free | Schema and structural check. Call before simulating. |
| `passive_inventory` | free | Resistor/capacitor inventory and area estimate from deck JSON. |
| `run_analysis` | seconds | Bounded DC/AC/noise/transient summary for one circuit. |
| `submit_exploration` | job | Candidate sweep with Pareto selection. |
| `submit_mismatch_mc` | job | Per-device mismatch Monte Carlo. |
| `submit_pvt` | job | Corner sweep of one circuit; temperature and supply axes on silicon. |
| `submit_signoff` | job | Multi-testbench PVT campaign from a manifest. |
| `list_jobs` / `get_job` / `cancel_job` | free | Poll and cooperative cancellation. |
| `signoff_margins` | free | Per-constraint margin table from a saved result. |
| `inspect_signoff_result` | free | Selected failing points from a saved result. |

`submit_pvt` and `submit_signoff` also answer different questions. A PVT sweep
shows how one testbench *moves* across corners — gain, bandwidth, input-referred
noise per grid point. A signoff campaign runs a whole manifest against declared
constraints and returns a verdict. Sweep to understand, sign off to decide.

`signoff_margins` and `inspect_signoff_result` answer different questions.
Margins rank every spec by how much room is left and tell you what constrains
the design. Inspection tells you where a specific spec failed. When deciding
what to change next, read margins first.

## The design loop

```
validate_circuit          structural errors are free to find
   -> run_analysis        one circuit, is it even alive
   -> submit_signoff      the real grid
   -> signoff_margins     which spec is tightest, and where
   -> change ONE thing
   -> repeat
```

Change one thing per iteration. The measured couplings in this codebase are
not intuitive — lengthening one current-source device collapsed an entire
amplifier because it pushed a control node below a bias floor 400 mV away —
and a two-variable change leaves you unable to attribute the result.

### Reading a campaign result

- `status` is `pass`, `fail`, or `invalid`. `invalid` dominates `fail`, which
  dominates `pass`.
- `worst_case` names one measurement. It ranks runs; it does not describe a
  design, because it hides every other spec that is also close to its limit.
- The margin table is the design view. `normalized_margin` is a fraction of
  the limit, negative when violated, so rows with different units compare
  directly. Rows come back tightest first.

### Do not stop at nominal

A signoff that passes only at nominal component values is not a signoff. Poly
resistors carry roughly 20 % absolute tolerance and MOM capacitors roughly
10 %:

```bash
circuit-opt signoff campaign.json --tolerance R=20,C=10
```

The result is `robust` only when every perturbed run also passes. A
compensation network tuned on a knife edge will pass nominal and fail silicon.

## Changing a design

Device sizes live either in the circuit JSON or in a generator script that
emits several testbenches from one source of truth. Prefer the generator when
one DUT appears in several decks, so the decks cannot drift apart.

For a generator, drive it with `tools/design_iterate.py` rather than editing
and re-running by hand:

```bash
python tools/design_iterate.py run --generator my_gen \
    --manifest examples/campaign.json --set "SZ:M9=175/0.25" --set CC=850e-15
```

Every override asserts that its target exists, so a typo aborts instead of
silently evaluating the unmodified design.

If you do edit source text programmatically, **assert that the target string
is present before replacing it**. `str.replace` does not raise when it matches
nothing; it returns the string unchanged. Three full measurement rounds were
once spent on a design that had never been modified.

Never run `git checkout <file>` to undo an edit. It discards uncommitted work
without warning. Snapshot the file first and copy it back.

## Traps that produce a confident wrong answer

These are the failure modes that do not raise. Each cost real debugging time.

**Unknown configuration keys are rejected on purpose.** A misspelled option
that was silently ignored would run with the default and produce a plausible
result for a configuration nobody asked for. If you get an unknown-key error,
fix the key; do not route around the check.

**Geometry has bin limits.** TSMC28 core devices accept a per-finger width of
0.1–2.5 µm and a length of 0.03–1.0 µm. `W` is the total width and `NF` splits
it into fingers, so a wide device needs a proportional `NF`. Out-of-range
geometry raises `0 bins`; it does not silently clamp.

**`NF` and `M` are not the same knob.** `NF` splits one device's width into
fingers: same total current, different gate capacitance. `M` is parallel
instances: current, capacitance and noise power all scale. Confusing them
changes the answer without changing the netlist's plausibility.

**Two outputs mean a difference, not a sum.** `outputs: ["OUTP", "OUTN"]`
senses `OUTP - OUTN`. A single output senses to ground.

**An AC run with no `ac_drives` has no stimulus.** It returns the -180 dB gain
floor rather than an error. If a gain looks impossibly bad, check that a
stimulus is configured before you touch the circuit.

**A measured bandwidth can be the top of the sweep.** `bw_Hz` is found by
scanning the swept response, so a circuit whose corner lies above the grid
reports the last grid point — and it reports it as a number, indistinguishable
from a measurement. `corner_table` (behind `circuit-opt corners`) defaults to
the AFE range of 0.01 Hz – 10 kHz, which makes a 79 kHz silicon OTA read as
"BW = 10 kHz" and its 84 µV input noise read as 2.1 V. Pass `--freqs-*` and
`--noise-band` to match the circuit, and treat a bandwidth sitting exactly on
the grid ceiling as unmeasured. The service's `pvt` and `mc` jobs inherit the
circuit's own `analyses.ac.freqs` and `analyses.noise.band` for this reason and
echo back what they used.

**In an MC summary this same trap reads as robustness.** Every sample hits the
identical ceiling, so the spread collapses to sigma zero — a design that looks
remarkably tight is often one whose metric was never measured. Check the
reported sweep range before believing a small sigma.

**A sweep of a chopper measures nothing unless the chopper has an AC path.**
`corners` and `mc` both measure gain, bandwidth and noise from an AC run. A
testbench driven only through its `periodic` block has no `ac_drives`, so the
sweep returns the -180 dB floor for every corner and every sample, and the
input-referred noise — output noise divided by that floor — comes back around
1e134 V. Nothing raises. Sweep such a circuit with PSS/PAC/PNoise, or give it an
`ac_drives` entry first. A gain near the floor invalidates every other number in
the same result.

**Each analysis reads a different stimulus block, and transient reads none by
default.** `ac`/`noise` are excited by `input_drives` (a device gate) and
`ac_drives` (a node); `transient` is excited only by a `periodic` block, and
`_run_transient` passes no inputs at all when there isn't one. So a deck with
`input_drives` and no `periodic` — `sky130_5t_ota`, and most single-testbench
decks — has a properly excited AC run and a transient in which every source sits
at DC. It converges, every node is a flat line, and the result is reported as a
successful solve: measured `vout` peak-to-peak is exactly `0.0`. Before believing
a transient, check that the circuit has a `periodic` block or that you passed
`inputs=` yourself; a zero-swing output is the signature. The GUI derives one
from the AC ports on request and writes it to `analyses.transient.periodic`,
which merges over the top-level block so PSS/PAC keep their own excitation.

**A background job silently drops a field its request model does not declare.**
The service uses pydantic only as a thin request wrapper, so an undeclared key
is discarded rather than rejected — a caller passing `freqs` to an endpoint that
never declared it gets a successful job measured over the wrong range. When
adding a parameter, assert it reaches the *result*, not merely that the request
returned 202.

**Device state is path-dependent.** BSIM handles keep their warm start and
limiting state between calls. Concurrent campaign points must lease into
isolated cache scopes or results depend on worker count. This is already
handled inside the campaign layer; it matters if you write a new parallel
driver.

**A transient's start point must match its first sample.** The DC operating
point has to be solved with every waveform-driven source pinned at `u(t0)`.
Getting this wrong once drove a common mode 0.42 V into the rail and failed
every PVT point of two unrelated topologies, while the frozen golden corpus
stayed bit-exact. When a whole trajectory diverges, suspect initial conditions
before device models.

**The reported version can be stale.** `get_capabilities` reads the version
from the installed distribution metadata. An editable install made before the
last version bump reports the old number. Check `python tools/version.py
check` against the repository rather than trusting the capabilities payload.

## Verification discipline

**A new test must be shown to fail.** Write the test, then deliberately break
the invariant it claims to protect, confirm the test goes red, and restore.
A test that passes against the broken code asserts nothing. This is not
optional here: one existing test compared object identities across a free, and
passed for the wrong reason roughly four runs in five.

Restore from a file snapshot, not from git.

**Clear `__pycache__` between mutation runs** or you will measure the old
bytecode.

**The golden corpus is the reference oracle.** `tests/golden/engine_parity`
is frozen and reproduced bit-exactly. If a change moves those numbers, that is
the finding — investigate before re-freezing.

**ngspice is the independent second engine.** `circuit-opt verify-engine` runs
one signoff case through both the native BSIM4 path and the foundry model-card
path and diffs node trajectories. It keys its verdict on the settled deviation,
because a large deviation at a fast transition is usually step placement rather
than disagreement. Use it to confirm a result you intend to rely on, not as a
routine step — it is hours, not seconds.

## Gates before you report work as done

```bash
pytest tests -q                          # 935 passed, ~110 s
ruff check circuitopt tests examples
python -m mkdocs build --strict
python tools/version.py check
```

`mkdocs build --strict` aborts on warnings, including broken documentation
links. It cannot catch a syntax error inside a ```mermaid fence — those render
blank in a browser and pass the build.

## Where to look next

| Question | Document |
| --- | --- |
| What does each module do? | `docs/module_overview.md` |
| What happens between the CLI and the Rust core? | `docs/solver_call_flow_zh.html` |
| What fields can a circuit JSON contain? | `docs/json_circuit_format.md` |
| How is a PVT campaign configured? | `docs/signoff_campaign.md` |
| Which PDKs and corners exist? | `docs/pdk_support.md` |
| How do I set up a development environment? | `docs/development.md` |
| What changed recently, and why? | `CHANGELOG.md` |

Worked designs live under "Design Cases" in the navigation. They record the
dead ends as well as the result — a topology that was measured and abandoned
is documented as abandoned, so you do not spend a day rediscovering it.
