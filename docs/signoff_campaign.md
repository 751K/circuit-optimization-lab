# Signoff Campaigns

[Documentation Home](README.md) | [CLI Reference](cli_reference.md) |
[Circuit JSON Format](json_circuit_format.md) | [中文版](signoff_campaign_zh.md)

`circuit-opt signoff` runs multiple circuit testbenches over one explicit
process/voltage/temperature grid. It is intended for checks that cannot be
represented honestly by one netlist, such as open-loop gain, differential and
common-mode loop stability, closed-loop noise, and large-signal settling.

```bash
circuit-opt signoff examples/tsmc28hpcp_mdac_ota_signoff.json \
  --workers 4 --output results/tsmc28_mdac_signoff.json
```

The example contains 11 testbench cases over
`tt/ss/ff/sf/fs x -40/27/125 degC x 0.85/0.90/0.95 V`, for 45 PVT points and
495 case runs. It covers open-loop gain, the differential loop, both CMFB
loops, closed-loop input/output noise, five residue levels, and the 0111 to
1000 major-carry transition.

## Manifest

Campaign paths are relative to the manifest file. Absolute paths and paths
that escape the manifest directory are rejected, so a repository can be moved
without editing machine-specific locations.

```json
{
  "name": "ota_signoff",
  "pvt": {
    "corners": ["tt", "ss", "ff", "sf", "fs"],
    "temperatures_c": [-40, 27, 125],
    "supplies_v": [0.85, 0.9, 0.95],
    "nominal_supply_v": 0.9,
    "supply_bias_key": "VDD"
  },
  "cases": [
    {
      "name": "open_loop",
      "circuit": "ota_ac.json",
      "overrides": {
        "ac_drives": {"VINP": 0.5, "VINN": -0.5},
        "signoff": {
          "measurements": {},
          "constraints": {"gain": {"min": 80}}
        }
      }
    }
  ]
}
```

Each case deep-merges `overrides` into its base circuit. Objects merge
recursively and arrays replace the base array. A numeric PVT expression is an
explicit affine expression:

```json
{"$pvt": {"vdd": 0.5, "temperature_c": 0.0, "constant": 0.225}}
```

This evaluates to `0.5 * VDD + 0.0 * temperature_c + 0.225`. At each point the
runner also binds `section`, MOS temperature in kelvin, the named supply bias,
PMOS bulk voltage, numeric voltage-source levels, and DC seeds.

## Result Contract

Every case must contain a circuit-level `signoff` block. The campaign stores
that unit-bearing signoff envelope, not the raw waveform arrays. A model error,
non-convergence, non-finite result, or invalid signoff configuration produces
an `invalid` case; it is never converted to a passing fallback value.

Each point reports its case results and `worst_case`. The campaign-level
`worst_case` includes `case`, `corner`, `temperature_c`, `supply_v`,
`measurement`, and normalized margin. `invalid` dominates `fail`, and `fail`
dominates `pass`. Point ordering is deterministic for every worker count.

The manifest schema is
[`schemas/signoff_campaign.schema.json`](https://github.com/751K/circuit-optimization-lab/blob/main/schemas/signoff_campaign.schema.json).
The referenced testbenches continue to use
[`schemas/circuit.schema.json`](https://github.com/751K/circuit-optimization-lab/blob/main/schemas/circuit.schema.json).
