# `calibration/` — Cadence/Spectre reference data + closed-loop check

Each subdirectory is one **calibration case**: a `metadata.json` (provenance + circuit
+ analyses + tolerances) next to the Spectre PSFASCII reference files it describes. The
engine in [`core/calibration.py`](../core/calibration.py) loads a case, runs the matching
local analyses with the *same* sizes/bias, and compares each metric against the per-case
tolerance.

```
calibration/
  amp_design3_typical/      # design #3 AFE amplifier (non-chopper): DC / AC / noise
      metadata.json  dcOp.dc  ac.ac  noiseAnal.noise
  chopper_design3_{typical,slow,fast}/   # 8-PMOS chopper: PSS/PAC/PNoise, f_chop=225
      metadata.json  pac.0.pac  pnoise.pnoise
  sc_lpf/                   # 2-phase switched-cap LPF (single-ended LPTV): PSS/PAC/PNoise
      metadata.json  pac.0.pac  pnoise.pnoise  pss.td.pss
```

The two periodic cases exercise different regimes: the **chopper** is a differential
commutated amplifier; the **sc_lpf** is a single-ended switched-capacitor filter whose
PMOS switches go *reverse-biased* (drain above source) — it is the regression guard for
the signed reverse-bias device current (PAC bandwidth) and the cyclostationary-flicker
folding (output noise). Its `metadata.json` carries the full topology (devices / vsource
clocks / caps) so the engine builds it without importing `examples/`.

## Run it

```bash
python -m core.calibration --all                       # every case, text report
python -m core.calibration calibration/amp_design3_typical/
python -m core.calibration calibration/chopper_design3_typical/ --analyses pac,pnoise
python -m core.calibration --all --json                # CI-friendly
python -m core.calibration --all --relaxed             # 3x tolerances
```

Exit code is non-zero if any case fails. `tests/test_calibration.py` drives the same
engine under pytest.

## Regenerating the reference data (Spectre on `flex`)

Netlists are generated from the repo topology by
[`core/cadence_netlist.py`](../core/cadence_netlist.py) — same sizes/bias as the solvers,
so Cadence and Python describe the *same* circuit. The `cadence-server-verify` skill
encodes the server access (csh login, Spectre-only license, `-format psfascii`). Outline:

```python
from core.cadence_netlist import gen_amp_netlist, gen_chopper_netlist
# write .scs -> scp to flex -> ssh 'bash -s' (license + source SPECTRE env + spectre)
# -> pull *.dc/*.ac/*.noise (amp) or pac.0.pac/pnoise.pnoise (chopper) into the case dir
```

Provenance (Spectre version, run date, fundamental) is read straight from the PSF
HEADER by `core.psf.provenance` and copied into `metadata.json`.

## Current status (Spectre 24.1.0.078, 2026-06-21) — all PASS ✅

| case | metric | local | Cadence | Δ |
|------|--------|------:|--------:|----:|
| amp_design3_typical | gain / IRN | 22.90 dB / 38.31 µV | 22.89 dB / 38.31 µV | **+0.00 dB / +0.0%** |
| chopper_design3_typical | PAC gain / IRN | 11.96 / 9.83 µV | 11.83 / 9.81 µV | **+1.11% / +0.18%** |
| chopper_design3_slow | PAC gain / IRN | 8.95 / 9.50 µV | 9.03 / 9.32 µV | **−0.88% / +1.92%** |
| chopper_design3_fast | PAC gain / IRN | 12.00 / 10.81 µV | 11.87 / 10.84 µV | **+1.07% / −0.26%** |
| sc_lpf | PAC gain / BW / out-noise | 1.006 / 16.65 Hz / 3.53 µV | 1.003 / 16.82 Hz / 3.48 µV | **+0.3% / −1.0% / +1.4%** |

The amp (DC/AC/noise) matches Cadence to ~machine precision; the chopper PAC baseband gain
and integrated IRN match within ~1–2% across all three corners; the SC-LPF PAC DC gain,
−3 dB bandwidth, and integrated output noise all match within ~1.4%. The SC-LPF PAC is computed by
the analytic-adjoint harmonic balance (the small-signal drive on the `V_IN` ideal source couples
into the bordered HB branch row), so it is **integration-method independent** — gear2 and BE give
the same ~1.006 gain. (The finite-difference shooting fallback was x0-sensitive: on this stiff τ≫T
circuit a 0.003 V gear2-vs-BE orbit difference fed a near-singular (I−Φ)⁻¹ and produced a spurious
24× baseband gain — fixed 2026-06-22.) The local chopper run must
use the validated solver configuration (gear2 PSS orbit, `switch_size`, `edge_time`,
`output_filter`, settling) — captured per case in `metadata.json`'s `circuit`/`solver`
blocks — otherwise a bare-default call mis-reports the gain by >10%.

The HB chopper path (`pmos_chopper_pss` → `pmos_chopper_pac`/`pmos_chopper_pnoise`,
what this loop validates) carries **no empirical constants**. The two old Cadence-fit
constants (`_CADENCE_PMOS_CHOPPER_CONVERSION_PHASE_RAD`=24.93°,
`_CADENCE_PMOS_CHOPPER_PERIODIC_NOISE_PSD_SCALE`=1.0355) were **retired 2026-06-22** —
they only patched the fast first-order `pmos_chopper_lptv_analysis` quasi-static estimate,
which now honestly reports its ~10% gain underestimate rather than fudging it.
