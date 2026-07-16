# results/

Historical benchmark plots and exploration outputs. Generated files are not
tracked in Git (`.gitignore` excludes `results/`); only this index is versioned.

## File index

| File | Origin | Reproduce |
|------|--------|-----------|
| `max_gain_analysis.png` | `examples/find_max_gain.py` | `python examples/find_max_gain.py` |
| `mc_mismatch.png` | `examples/mc_mismatch.py` | `python examples/mc_mismatch.py [n] [seed]` |
| `vin_vout_sweep.png` | `examples/sweep_vin_vout.py` | `python examples/sweep_vin_vout.py` |
| `afe_testbench_explore.csv` | one-off explore run (AFE testbench) | `core.explore.explore(...)` with testbench spec |
| `afe_testbench_explore.png` | " | " |
| `pmos_gmid_gmro_multi_wl.csv` | one-off device characterization | PMOS_TFT gm/Id sweep over multiple W/L combos |
| `pmos_gmid_gmro_multi_wl.png` | " | " |
| `mdac_ota_pvt45.csv` | FreePDK45 MDAC OTA campaign output; local audit observed 42/45 points | `.venv/bin/python experiments/mdac_ota_pvt_campaign.py` |
| `tsmc28_mdac_ota_pvt45.csv` | TSMC28HPC+ MDAC OTA campaign output; local audit observed 7 TT points | `.venv/bin/python experiments/tsmc28_mdac_pvt_campaign.py --workers 4` |

## Cleanup

To clear accumulated outputs:

```bash
rm results/*.csv results/*.png
```

The `README.md` itself is tracked so the directory always has at least one file.
