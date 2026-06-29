# LTE-controlled adaptive timestepping (Cadence-faithful) for the gear2 transient + PSS

> Implementation status (2026-06-28): implemented as an opt-in path. Public
> entry points now accept `adaptive=True` plus LTE tolerances; PSS freezes the
> accepted grid near convergence; JSON dispatch/schema and calibration metadata
> forward the new options. A Numba adaptive gear2 kernel is available for
> `n_aug == n`; ideal-voltage-source MNA topologies use the Python adaptive
> fallback. SC-LPF calibration now defaults to
> `gear2 + adaptive + cap_mode="average"` with edge breakpoints in the input
> grid, BDF2-history restart after clock slope discontinuities, and
> `pnoise_n_period_samples=512` / `pnoise_max_sideband=20`. Current local
> SC-LPF calibration is PASS: PAC gain −0.32%, BW +1.07%, PNoise output +2.82%
> versus the archived Spectre reference.

## Context

The multi-session arc closed on *why* Cadence handles both the chopper and the SC-LPF with **one** cap operator while our local solver needs a charge/average split. The server PSS log (`~/afe_corner_typ/spectre.log`) is decisive:

- Cadence's model is the **non-conservative** `Cgss·ddt(V(s,gate1))` (server `pmos_TFT/veriloga/veriloga.va:244`, byte-identical to our `PDK/veriloga.va:244`; `Cgss` depends on `Vd` → multi-variable non-conservative). Cadence uses this one operator everywhere.
- Cadence makes it work via **`method=gear2only`** (it auto-detected *"trapezoidal ringing"* and switched off `traponly`) **+ adaptive LTE step control** (`reltol=1e-4`, `errpreset=conservative`, **487 accepted steps/period**, refined below the 25 µs `maxstep` at switch edges).

Our local solver uses **fixed uniform grids**, so the non-conservative `average` operator (which *is* Cadence's operator) is accurate on the chopper (n=321, already edge-refined) but fails the SC-LPF at n=201 (+19% out-noise). A grid-refinement sweep confirmed this is a **coarse-Δt artifact**, not a structural conflict: `average` SC-LPF noise drops +16%→+3% as N goes 201→1601. So the missing piece is **adaptive Δt** — the same mechanism Cadence uses.

**Goal:** add LTE-controlled adaptive timestepping to the gear2 transient + PSS so the non-conservative (`average`) operator is accurate everywhere (unifying the orbit onto one operator, matching Cadence's method), with the standalone wins of fewer steps per accuracy and robust stiff handling. (User chose the full **Level 2** engine, not just a-priori edge refinement.)

**Two findings that de-risk this sharply:**
1. **The conversion needs NO change.** PAC/PNoise already periodic-interpolate the orbit's own grid `pss_result["t"]` onto a uniform FFT grid ([pnoise_solver.py:500,549,552] `_periodic_interp`→`t_uniform`; same in [pac_solver.py]). A non-uniform/adaptive orbit feeds the existing HB conversion unchanged.
2. **The monodromy already handles non-uniform grids.** The chopper orbit is already edge-refined (non-uniform) via `refine_chopper_tgrid` ([chopper.py:795]) and the gear2 analytic monodromy + shooting work end-to-end. Variable-step BDF2 coeffs already exist (the `rho=h_n/h_prev`, ρ>2→BE block at [numba_kernels.py:1578-1586]).

## Phase 1 — LTE-adaptive gear2 transient driver (Python-first)

Add an adaptive time-marching path to `transient()` ([core/transient_solver.py]); default stays the fixed-grid path (byte-stable).

- New `adaptive=True` (+ `reltol`, `abstol`, `t_end`) on `transient()`. When set with `integration_method="gear2"`, run a **time-marching loop** `t: 0→t_end` that *chooses* `h` instead of consuming a fixed `tgrid`.
- Reuse the existing per-step machinery: the BDF2 coeffs from the step ratio (the `rho`→`a0,a1,a2`, ρ>2→BE block at [numba_kernels.py:1578]), the per-step Newton (`_transient_newton_reuse_impl` / the Python `solve_chunk` gear2 path), the BDF2 history tuple `(x[n-1], x[n-2], h[n-1])`, and the maxstep cap.
- **LTE estimate** (variable-step BDF2, order 2): standard predictor–corrector difference (equivalently the 3rd divided difference of the solution history), weighted by `wrms = ||lte / (reltol·|y| + abstol)||_RMS`. **Accept** if `wrms ≤ 1`; else **reject** and shrink. New step `h_new = h · clamp(0.9·wrms^(-1/3), 0.2, 2.0)`, then clamp to ρ≤2 (BDF2 zero-stability, already enforced) and `maxstep`. Self-start with BE (already the ρ>2 path).
- Reuse the existing Newton-failure recursive subdivision as the reject fallback when LTE rejection alone doesn't converge.
- Returns the orbit on the self-chosen non-uniform grid (`t`, per-node arrays) — same result shape as today.

Standalone test: stiff RC + `pmos_chopper_transient` — adaptive reaches a target accuracy in far fewer steps than the equivalent fine uniform grid, and is 2nd-order.

## Phase 2 — adaptive PSS orbit + monodromy

`pss_solve(adaptive=True)` ([core/pss_solver.py]):

- Each shooting iteration integrates one period with the Phase-1 adaptive driver → orbit + **its** grid; residual `x(T)-x0` as today.
- The analytic gear2 monodromy is built on **that iteration's grid** (the variable-step monodromy already works — the chopper's edge-refined non-uniform grid proves it). **Freeze the grid** once near convergence (use the last accepted iteration's grid for the final orbit + monodromy) so the final Jacobian/orbit are on one stable grid and the shooting Newton isn't perturbed by grid churn. FD-shooting remains valid (re-integrate per perturbation).
- `pss_result["t"]` is non-uniform → PAC/PNoise consume it unchanged (finding #1). No edits to `pac_solver.py` / `pnoise_solver.py`.

Test: SC-LPF adaptive PSS converges (`pss_status`). The PMOS chopper wrapper now
rejects `adaptive=True` with `ValueError` and keeps the validated fixed
edge-refined grid until chopper-specific adaptive stepping is proven.

## Phase 3 — validate Cadence match + operator unification

- **The unification proof:** SC-LPF on the **non-conservative `average`** operator + adaptive → matches Cadence ~3.48 µV (the operator that failed at fixed n=201 now passes because adaptive Δt resolves the clock edges, like Cadence's 487 steps/period). Chopper on `average`+adaptive → unchanged/matches.
- `python -m core.calibration --all` → **5/5** with adaptive opt-in; the fixed-grid default stays **byte-identical** (no chopper regression — the hard gate). Wire an `adaptive` switch into the `metadata["solver"]` block so calibration can run both.
- Only after this passes, consider flipping the default to adaptive and/or unifying globally onto `average` (a separate, gated decision — not required by this plan).

## Phase 4 — numba acceleration (after Python correctness)

Port the adaptive loop to numba: new `_transient_solve_adaptive_gear2_impl` modeled on `_transient_solve_grid_gear2_impl` ([numba_kernels.py:1629]) but time-marching with the LTE accept/reject inside the kernel. Behind a flag (`_GEAR2_NUMBA_GRID`-style), validated to match the Python adaptive to ~1e-8 per node. Clear `core/__pycache__` to force recompile (stale-`.nbc` trap).

## Key files
- [core/transient_solver.py] — Phase 1 adaptive driver + `transient(adaptive=…)`; reuse the gear2 step-coeff + Newton + history.
- [core/numba_kernels.py] — `rho`/BDF2-coeff block (1578), `_transient_newton_reuse_impl`, `_transient_solve_grid_gear2_impl` (1629, the model); Phase 4 adaptive kernel.
- [core/pss_solver.py] — Phase 2 adaptive shooting + grid-freeze; `_make_period_grid` (311), `_shooting_monodromy` (variable-step already).
- [core/chopper.py] — `pmos_chopper_pss(adaptive=True)` is an explicit error; `refine_chopper_tgrid` remains the validated a-priori-edge baseline.
- **Unchanged:** [core/pac_solver.py], [core/pnoise_solver.py] — already resample any orbit grid (`_periodic_interp`→`t_uniform`).
- [tests/test_periodic_solvers.py], [tests/test_chopper.py] — adaptive tests.

## Verification
- `RUN_SLOW_CHOPPER=1 CIRCUIT_USE_NUMBA=1 python -m core.calibration --all` → 5/5; fixed-grid default byte-stable for cases that do not opt into adaptive. SC-LPF intentionally opts into `gear2 + adaptive + cap_mode="average"` in `calibration/sc_lpf/metadata.json`.
- New tests: (a) adaptive 2nd-order + step-economy on a stiff RC (steps ≪ uniform for the same error); (b) chopper wrapper rejects unsupported adaptive mode; (c) **adaptive SC-LPF on `average` matches Cadence 3.48 µV** (the unification result) vs the stored `calibration/sc_lpf` ref.
- Optional server cross-check: rerun SC-LPF/chopper on `flex`, compare our adaptive grid's step distribution to Cadence's (`reltol=1e-4`, ~487 steps/period).

## Risks
- **PSS monodromy on a per-iteration-varying grid** → shooting-Jacobian churn. Mitigation: freeze the grid near convergence (standard); FD-shooting fallback.
- **LTE-estimator tuning** (reltol/abstol/safety/clamps). Mitigation: target Cadence's `reltol=1e-4` + conservative; calibration is the gate.
- **Numba surgery (Phase 4)** on the fragile gear2 driver. Mitigation: Python-first correctness + a ~1e-8 match test; clear `__pycache__`.
- **No regression rule:** adaptive stays opt-in; the fixed-grid path must remain byte-identical on the chopper calibration until adaptive is proven.
