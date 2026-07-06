# Round-trip fixtures

These 12 JSON files are **verbatim copies** of the circuit examples in the repo
root `examples/` directory. They are the source of truth for the graph<->JSON
round-trip tests (`../roundtrip.test.ts`).

## Source & sync

Each `<name>.json` here is an exact copy of `../../../../examples/<name>.json`
(repo root `examples/`). To re-sync after the examples change:

```bash
# from the repo root
for f in voltage_divider resistor_load_stage single_stage periodic_rc \
         sky130_5t_ota sky130_fd_ota freepdk45_5t_ota freepdk45_fd_ota \
         afe_explore sc_lpf vcvs_amplifier sky130_chopper; do
  cp "examples/$f.json" "frontend/src/model/__fixtures__/$f.json"
done
```

They are copied (not imported across the repo boundary) so the frontend package
is self-contained and its tests don't reach outside `frontend/`.
