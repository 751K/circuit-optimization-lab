# Circuit Optimization Documentation

[English](README.md) | [中文](README_zh.md)

This site is the maintained documentation for Circuit Optimization. It is
organized for three audiences:

- **Users** who want to install the project, run an existing circuit, and create
  circuit JSON files.
- **PDK integrators** who need to understand model bindings, external toolchains,
  process corners, and backend limitations.
- **Developers** who need the solver architecture, extension points, tests, and
  release checks.

## Start Here

1. [Getting Started](getting_started.md): install with `uv` or `venv`, run the
   passive smoke test, and choose optional dependencies.
2. [CLI Reference](cli_reference.md): analysis, exploration, corners, mismatch,
   ADC, plotting, dataset, and service commands.
3. [Circuit JSON Format](json_circuit_format.md): the maintained field-level
   reference for circuit descriptions.
4. [PDK Support Matrix](pdk_support.md): model keys, prerequisites, supported
   analyses, corner names, and accuracy boundaries.

## Learn the Internals

- [Core Solver Overview](module_overview.md): data flow, model abstraction,
  MNA/Newton solvers, transient integration, periodic analyses, optimization,
  and service layers.
- [Developer Handoff Guide](development.md): repository map, test strategy,
  extension workflow, and documentation maintenance rules.
- [Local Service API](service_api.md): FastAPI endpoints, background jobs,
  serialization, and CLI equivalence.
- [ngspice Oracle Helpers](ngspice_oracles.md): explicit external-reference
  paths used by FreePDK45 and TSMC regression workflows.

## PDK Guides

- [TSMC28HPC+ Native Adapter](tsmc28hpcp.md)
- [PDK Support Matrix](pdk_support.md)

Foundry model payloads are local inputs. They are not part of the repository,
must remain Git-ignored, and remain subject to their original license terms.

## Design Records

[Design Cases and Validation Status](design_cases.md) separates reproducible
examples from incomplete campaigns and historical engineering notes. A design
record documents one experiment; it is not automatically a release-wide
accuracy guarantee.

## Reference Documents

| Document | Purpose | Maintenance status |
|---|---|---|
| [Getting Started](getting_started.md) | Installation and first run | Maintained |
| [CLI Reference](cli_reference.md) | Current command-line interface | Maintained |
| [Circuit JSON Format](json_circuit_format.md) | Circuit schema and examples | Maintained |
| [PDK Support Matrix](pdk_support.md) | Backend and process capability boundaries | Maintained |
| [Core Solver Overview](module_overview.md) | Implementation architecture | Maintained |
| [Local Service API](service_api.md) | HTTP and job API | Maintained |
| [TSMC28HPC+ Adapter](tsmc28hpcp.md) | Licensed local model integration | Maintained |
| [Licenses and Third-Party Software](third_party_licenses.md) | MIT scope, BSIM4 attribution, and vendored-code terms | Maintained |
| [Environment and Performance Notes](environment_performance.md) | Dated benchmark snapshots and tuning notes | Reference snapshot |
| [Design Cases](design_cases.md) | Design-specific derivations and results | Per-case status |

The release history is maintained in the repository
[CHANGELOG](https://github.com/751K/circuit-optimization-lab/blob/main/CHANGELOG.md).
Licensing and attribution details are maintained in
[Licenses and Third-Party Software](third_party_licenses.md).

## Documentation Policy

- Commands in maintained guides must correspond to current `--help` output.
- Capability claims must distinguish the local solver, an external oracle, and
  sign-off tooling.
- Result tables must identify their source file and coverage. Partial PVT data
  must not be described as a completed campaign.
- Roadmaps and completed implementation plans do not belong in the user-facing
  documentation. Durable implementation details are folded into architecture or
  PDK guides.
- The JSON schema in `schemas/circuit.schema.json` and the loader are the source
  of truth when prose and code disagree.
