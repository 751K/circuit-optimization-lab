# Single-source the numerical kernels onto the Numba `_impl` functions

## Problem

Each numerical kernel exists **twice**: an object-oriented Python version (readable
reference + no-Numba fallback) and a flat-array Numba `_impl` (fast). Every math
change must be made in both and kept bit-identical — the dominant maintenance tax.

Three duplicated families:

| Family | OO Python | Numba `_impl` | Parity guard |
|---|---|---|---|
| Device model | `PMOS_TFT._eval_currents / _capacitances_from_op / _capacitance_charges_from_op / _newton_internal` + terminal derivs | `_eval_currents_impl`, `_capacitances_impl`, `_capacitance_charges_impl`, `_newton_internal_impl`, `_terminal_derivatives_impl` | `test_model_kernels` |
| PAC/PNoise fold/HB | `_assemble_pac_linearization_python`, Python fold | `_pac_linearize_orbit_impl`, `_pnoise_fold_psd_impl`, `_pnoise_hb_blocks_impl` | `test_model_kernels` |
| Transient stamp/Newton/drivers | `_k_step_residual / _k_build_jac / _k_terminal_derivatives / _k_newton / _k_*` | `_stamp_transient_system_impl`, `_transient_newton_impl`, the 3 drivers | `test_numba_augmented` + byte-gate |

## Approach — single-source onto `_impl`, keep `.py_func` (NOT a hard Numba dep)

The Numba `_impl` functions are already pure-Python (Numba-subset). When jitted they
expose `.py_func`; when Numba is absent the raw `_impl` *is* pure Python. So one
source (`_impl`) can serve **both** the compiled and the interpreted path:

```python
# numba_kernels: _eval_currents_impl is the jitted kernel when Numba is on,
# the raw function when off — never None.
def _eval_currents(self, Vs, Vd, Vg, Vs1, Vd1):
    return _eval_currents_impl(Vs, Vd, Vg, Vs1, Vd1, self.Vfb, ...)   # the ONE formula
```

We **delete the OO formula bodies** and delegate to `_impl`. We do **not** make Numba a
hard dependency and do **not** delete the interpreted path: `.py_func` / the raw
`_impl` keeps debuggability, portability, and JIT-free smoke — at zero maintenance
cost (it is the same source). Numba stays in `optional-dependencies`.

### Why this is byte-identical (measured, not assumed)

`max |OO(x) − _impl.py_func(x)| = 0.000e+00` across 32 (size, bias) points for
`_eval_currents`, `_capacitances_impl`, `_capacitance_charges_impl`. Production (Numba
on) already runs the jitted `_impl`; delegating the OO body to the same `_impl` cannot
move a single ULP. amp/chopper stay byte-identical; the calibration byte-gate is the
per-stage gate.

## Stages (each ends with the byte-gate: `calibration --all` 5/5 byte-identical + `pytest -q`)

- **1a — Device model, pure formulas. ✅ DONE.** Deleted the OO bodies of
  `_eval_currents`, `_capacitances_from_op`, `_capacitance_charges_from_op` (and the
  now-orphaned `_eval_channel_ich_sorted` helper); each delegates to
  `_eval_currents_impl` / `_capacitances_impl` / `_capacitance_charges_impl`. Dropped the
  unused `*_numba` guard names from the import. **84 lines removed**; calibration 5/5
  **byte-identical**, `pytest` 191 passed, ruff clean, Numba-off interpreted path verified.
- **1b — Device model, `_newton_internal`. ✅ DONE.** `_newton_internal` now delegates to
  `_newton_internal_impl` (adapting its `(ok, Vs1, Vd1)` return to the array/`None`
  contract); the duplicated 2×2 Newton loop + numba-first wrapper deleted, unused
  `newton_internal_numba` import dropped. Byte-identical (5/5), 191 tests, ruff clean.
  Total device-model reduction 1a+1b: **−118 lines** (796→678). Note: the
  `get_ss_params` terminal-derivative path is numba-first (`terminal_derivatives_numba`,
  already the single source) **plus a genuinely distinct finite-difference fallback** —
  not an analytic duplicate — so it is left as-is. The device-model family is now
  effectively single-sourced (remaining OO methods — `_eval_channel`, `_robust_op`
  fsolve cold-start, `_capacitance_branch_terms_from_op`, small helpers — have no Numba
  twin).
- **2 — PAC/PNoise HB blocks. ✅ DONE (partial — by design).** `_hb_blocks` (pnoise) and
  `_pac_hb_blocks` (pac) now delegate to `_pnoise_hb_blocks_impl` through a shared
  `numba_kernels.py_impl()` helper (jitted for `(2K+1)·n ≥ 16`, interpreted `.py_func`
  below) — **3 copies of the HB conversion-block assembly collapsed to 1**. Byte-identical
  (measured 0.0 over 20 combos; calibration 5/5; 191 tests).
  **NOT delegated (intentionally):** `_assemble_pac_linearization_python` and the pnoise
  fold are **supersets** — the Python paths handle cases the numba `_impl` cannot
  (non-`charge_caps` PAC linearization + retained gate1 state; bordered/vsource fold with
  `nbr > 0`). Wholesale delegation would drop those cases, and a partial sub-case split is
  low-value (production already takes the Numba path for the supported cases), so they stay
  as separate implementations. The remaining meaningful periodic duplication is the
  per-orbit stamp/fold *math*, which overlaps with the transient stamp (stage 3).
- **P4 (transient stage-3 prerequisite). ✅ DONE.** Sized the two fixed-grid Numba
  drivers `_transient_solve_grid_impl` (BE) and `_transient_solve_grid_gear2_impl`
  (fixed gear2) at `n_aug` (Vhist/Vwork/R/J + every full-vector copy/store loop; kept `n`
  for the stamp/substep node-count args — same pattern P5 used for the adaptive driver)
  and relaxed the two guards `_solve_fixed_gear2_numba` / `_solve_be_numba` to
  `n_aug >= n`. The stamp/Newton kernels already handled `n_aug` (P2/P3). Effect,
  measured with a solver-path probe across the transient/PSS test suite: **full-Python
  transient solves 160/345 (46%) → 2/345 (0.6%)**; pure-Numba 184 → 342. amp/chopper/
  sc_lpf byte-identical (they are `n_aug==n` or use the adaptive path); the two vsource
  transient tests re-baselined onto the Numba path (`numba_grid_solver` now `True`, RC
  step / divider numerics still pass); 191 tests, ruff clean.
- **3 — Transient stamp/Newton/drivers.** Unlike 1–2 this is a *delete + rewire*, not a
  delegate (the OO path is ctx/object-based, numba is flat-array), done in two gated steps:
  - **3A — Reroute. ✅ DONE.** The three `_solve_*_numba` now call the module-level
    `_transient_solve_{grid,grid_gear2,adaptive_gear2}_impl` directly (jitted when Numba is
    on → **byte-identical**; raw pure-Python when off), instead of the `*_numba` public name
    gated on `is not None`. So the **no-Numba transient path now runs the single `_impl`
    interpreted** (validated: numba-off `test_controlled_sources` 22 + `test_vsource` 19
    pass), making the OO `_k_*`/`_solve_*_python` production-dead (only the debug toggle
    `CIRCUIT_GEAR2_NUMBA=0` + the 1/345 resume still reach them). The two
    `test_numba_adaptive_gear2_matches_python[_at_input_kinks]` parity tests were deleted
    (their monkeypatch-numba-to-None mechanism no longer gates anything after the reroute).
    Byte-gate 5/5, 189 tests, ruff clean.
  - **3B — Delete. ✅ DONE.** Rewired the `transient()` dispatch to drop the three
    `_solve_*_python` fallbacks + the mid-solve resume (numba failure now accepts the
    partial trajectory), then deleted the OO `_k_*` block (578–1384) + the three
    `_solve_*_python` (~230 lines) + the `CIRCUIT_GEAR2_NUMBA` debug toggle + the dead
    `python_start_idx`, and removed `tests/test_numba_augmented.py`.
    **transient_solver.py 2464 → 1412 lines (−1052).** Byte-gate 5/5 **byte-identical**;
    full suite 183 passed (189 − 6 deleted parity tests); numba-off transient validated
    (runs the interpreted `_impl`); ruff clean. The numba transient `_impl` is now the
    single source, validated by the Cadence byte-gate + physics tests (RC step / controlled
    sources) — no independent hand-written Python reference. Follow-up cleanup also
    removed the now-dead `use_numba_newton` ctx field + its computation and the five
    unused `*_numba` public imports (`transient_solver.py` → 1392 lines).

## Result

All three families single-sourced. Device model −118, HB blocks 3→1, transient −1052.
The numba `_impl` kernels are the one source; the interpreted path is their `.py_func` /
raw form (kept, free). Numba stays optional. Cadence byte-gate 5/5 byte-identical
throughout.

## CI / rot guard

Add one fast run of the suite (or a smoke subset) with `CIRCUIT_USE_NUMBA=0` so the
interpreted `_impl` path stays exercised and cannot bit-rot after the OO twins are gone.
(This also fixes the ~2 tests that currently fail without Numba, since the interpreted
`_impl` becomes a complete path.)

## Risks

- **Delegation must hit the same `_impl` production already uses.** Import the module-level
  `_impl` name (jitted when Numba on) — not `.py_func` — so production perf/behaviour is
  unchanged; the interpreted path falls out automatically when Numba is off.
- **numba_kernels becomes a hard intra-package import for the device model** (it already is —
  it is pure-Python importable without Numba). If it ever fails to import, fail loudly
  rather than silently running a divergent OO twin.
- **Transient (stage 3) is gated on P4** and on accepting the loss of Python mid-solve
  recovery; do it last, only if 1–2 don't relieve enough.
