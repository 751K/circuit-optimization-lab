# R5-B parity surface — `circuitopt_core`

The compiled core (`circuitopt_core`, built by `co-py`) exposes the co-spice
deck parser + elaborator and the co-pdk PDK compilers for **differential
verification only**. Production keeps using the Python `circuitopt.spice` /
`circuitopt.pdk` paths; nothing here is wired into the solver. Every entry point
is a 1:1 port of the frozen Python reference, verified to match it bit-for-bit
(or within a relative error of 1e-14 where an expression evaluates through
`libm pow`). All heavy compute runs under `py.detach` (GIL released); Rust errors
map to Python exceptions and never panic across the boundary.

## Exceptions (all `ValueError` subclasses)

| Class | Base | Raised by |
|-------|------|-----------|
| `SpiceExpressionError` | `ValueError` | expression parse/eval |
| `UnknownSymbolError` | `SpiceExpressionError` | missing symbol/function |
| `ParameterCycleError` | `SpiceExpressionError` | lazy-parameter cycle |
| `SpiceSyntaxError` | `ValueError` | malformed deck syntax |
| `SpiceElaborationError` | `ValueError` | section cycle / unknown section / unknown subckt |

PDK-specific model errors (the Python `Freepdk45ModelError` / `Sky130ModelError`
/ `Tsmc28ModelError`, themselves `ValueError` subclasses) surface as plain
`ValueError`. Numericization failures inside TSMC28 elaboration surface as the
matching co-spice class above.

## SPICE deck parser + elaborator functions

```python
spice_parse_number(text: str) -> float
spice_logical_lines(text: str, path: str = "<string>")
    -> list[tuple[str, tuple[str, int, int]]]        # (joined line, (path, first, last))
spice_parse_assignments(text: str) -> list[dict]     # {name, expression, formal_parameters}
spice_parse_library(path: str) -> dict               # canonical tree (reads the file, encoding="ascii")
spice_parse_library_text(text: str, path: str = "<string>") -> dict
spice_select_sections(path: str, sections: list[str]) -> list[str]   # ordered, de-duplicated
spice_elaborate(path: str, sections: list[str], overrides: dict | None = None) -> dict
    # {model_name: {"name", "model_type", "parameters": {name: float}}} for section-level .model
spice_elaborate_instance(path, sections, subckt, params=None, overrides=None) -> dict
    # {"models": [{"name","model_type","parameters"}...], "elements": [{"kind","name","parameters"}...]}
```

`overrides` seeds the elaboration root scope (`initial_values`, e.g.
`{"temper": 27.0}`); `params` are subcircuit instance overrides
(`Mapping[str, float | str]`).

### Canonical tree shape (field names mirror the Python dataclasses 1:1)

```
library    = {"path", "top_level": section, "sections": {name: section}}
section    = {"name", "location", "statements": [statement], "subcircuits": {name: subckt}}
subckt     = {"name", "location", "terminals": [str], "parameters": [assignment], "statements": [statement]}
statement  = {"kind", "location", "text", "name": str|None, "arguments": [str], "parameters": [assignment]}
assignment = {"name", "expression", "formal_parameters": [str]}
location   = (path: str, first_line: int, last_line: int)
```

Sequence fields (`arguments`, `terminals`, `formal_parameters`) are JSON lists;
`sections`/`subcircuits` are dicts keyed by lower-cased name. To compare against
Python, canonicalize its dataclasses with the same shape (tuples → lists,
`SourceLocation` → `(path, first_line, last_line)`).

## PDK compiler

```python
class circuitopt_core.CompiledPdk:
    def __init__(self, pdk: str, root: str | None = None): ...
        # pdk: "freepdk45" | "sky130" | "tsmc28"
        # root: freepdk45 -> PDK_ROOT dir (holds freepdk45/models_*/)
        #       sky130    -> resolved card directory (holds *.json)
        #       tsmc28    -> HSPICE model directory (holds the .l delivery)
    def numeric_card(self, polarity, corner, temp_c,
                     w_um=None, l_um=None, nf=1, mult=1, mismatch=None) -> dict
```

`temp_c` is used only by TSMC28; `w_um`/`l_um` are required (positive µm);
`mismatch` is `None` (no offset) or a `delvto` volts value.

Returned dict:

```
{
  "model_parameters":    {name: float},   # == Python *Card.model_parameters
  "instance_parameters": {name: float},   # == Python *Card.instance_parameters
  "model_name": str,                       # freepdk45: NMOS_VTG/PMOS_VTG; sky130: card stem; tsmc28: bin name
  "model_type": str,                       # freepdk45/sky130: polarity; tsmc28: model_type
  "source_version": float,                 # 4.0 (freepdk45) / 4.5 (sky130, tsmc28)
  "bin": {"name", "lmin", "lmax", "wmin", "wmax"} | None,   # tsmc28 only
  "source": {"pdk","polarity","corner","path","temperature_c","macro_name","bin_name"}
}
```

`source` carries only paths and section/bin identifiers — never card text.

### Reference paths for the differential gate

| PDK | Python reference | Rust `root` |
|-----|------------------|-------------|
| freepdk45 | `load_freepdk45_library(pol, corner).device_card(width_um, length_um, nf, mult, mismatch_v)` | `circuitopt.toolchain.pdk_root()` |
| sky130 | `load_sky130_card(pol, width_um, length_um, nf, mult, corner, mismatch_v)` | `circuitopt.pdk.sky130.library._BUNDLED_CARD_DIR` |
| tsmc28 | `load_tsmc28_core_library().core_card(pol, width_um, length_um, nf, mult, corner, temperature_c, mismatch_v)` | `circuitopt.toolchain.tsmc28_model_dir()` (set `TSMC28_PDK_ROOT`) |

Cache (D12): immutable `CompiledPdk` + a process-local, thread-safe in-memory
cache keyed on the canonical file path + mtime/size, plus (for TSMC28) the
elaborated section set + temperature. No card content is persisted.

### Documented deviation

- TSMC28 `numeric_card` returns the **raw** card parameters (matching
  `Tsmc28CoreCard.model_parameters` / `.instance_parameters`). The `to_bsim4_cards`
  `mulu0 → u0` mobility fold is a downstream co-bsim4 step, not applied here.

## Compiled campaign / candidate executor (R5-C)

`CompiledCampaign` holds an immutable circuit template + analysis plan and
evaluates a candidate matrix through a device-build → DC → AC → noise pipeline
entirely under one `py.detach`, with **no per-candidate Python callback**. The
generic batch machinery lives in `co_core::campaign` (single Rayon pool,
adaptive candidate-vs-frequency axis, candidate-index-ordered write-back, atomic
progress + cooperative cancel that never feed a reduction, and the
`bw_from_gain` / `band_rms` metric reductions). Two device families are wired:
the AFE OTFT evaluator (`co_core::otft_campaign`) and the silicon BSIM4
evaluator (`co-py/src/silicon_campaign.rs`, families freepdk45 / sky130 /
tsmc28). Not connected to any production workflow (that is R5-D).

```python
class circuitopt_core.CompiledCampaign:
    def __init__(self, spec: dict): ...
        # spec = {"family": "afe_otft" | "silicon_bsim4", "template": <template dict>}
    @property
    def family(self) -> str: ...
    def evaluate_batch(self, candidates: list[dict], workers: int = 1,
                       analyses: list[str] = ("dc", "ac", "noise")) -> list[dict]
```

### Template dict (candidate-invariant; marshalled once from `CompiledTopology`)

Built by `circuitopt._rust_campaign.AfeOtftCampaign` from
`CompiledTopology(AFE_TOPO, bias)` + the analysis plan. All terminal tokens are
`(kind, ref, value)` triples: **kind 0** = solved-node index (`ref`), **kind 2**
= fixed rail/AC value (`value`).

```
{
  "family": "afe_otft",
  "template": {
    "n_aug": int, "n_nodes": int,
    "consts": [vt, ci, roff, reg, c1, c2, c3, c4, kv, kh, temperature],  # 11 AT4000TG defaults
    "devices": [ (dc_d, dc_g, dc_s, di, si, ac_d, ac_g, ac_s), ... ],    # 8-tuples per device
    #   dc_* / ac_* are (kind, ref, value) tuples; di/si are int KCL rows (-1 = none)
    "ac_caps": [ (a_term, b_term, value), ... ],
    "output_weights": [ (node_index, weight), ... ],   # e.g. (0, 1.0), (1, -1.0)
    "sense": [floats length n_aug],
    "vin_norm": float,
    "freqs": [floats],
    "band": [f_lo, f_hi],
    "gmin": float, "dc_tol": float,
    "dc_guesses": [[floats length n_aug], ...],         # topo.dc_guess_vectors(bias)
    "latch_nodes": (idx0, idx1) | None,
  },
}
```

### Candidate dict

```
{
  "devices": [ [w, l, nf, pvt0, mvt0, pbeta0, mbeta0], ... ],  # one row per template device
  "seed": [floats length n_aug] | absent,                      # optional DC operating-point seed
  "trust_seed_as_op": bool,                                    # default False
}
```

`trust_seed_as_op=True` uses `seed` verbatim as the DC operating point (no
re-solve) — the mode that isolates bit-exact AC/noise parity from DC-root
behaviour. `False` refines the seed (or the template guesses, cold) with the
Rust circuit Newton. Random mismatch (`mvt0`/`mbeta0`) is drawn **up front** on
the Python side in candidate/device order (same rule as
`corners.mismatch_corner`) and baked into the candidate, so the detached batch
carries no RNG. `analyses` gates the noise stage (drop `"noise"` for the
AC-only prefilter — `irn_uV` becomes NaN).

### Silicon family (`"silicon_bsim4"`, freepdk45 / sky130 / tsmc28)

Template (built by `circuitopt._rust_campaign.SiliconCampaign` from a loaded
circuit JSON; marshalled once):

```
{
  "pdk": "freepdk45" | "sky130" | "tsmc28",
  "root": <CompiledPdk root, see the PDK table above>,
  "circuit": OtftTransientProblem(passive_problem_spec(plan)),  # passive MNA circuit
  "n_aug": int,
  "dc_devices": [ ([d, g, s, (2, 0, vb)], [di, gi, si, -1]), ... ],  # solve_dc records
  "devices": [ (polarity, vb, temperature_k, temp_c, ac_d, ac_g, ac_s), ... ],
  "ac_caps"/"ac_resistors"/"ac_vccs"/"ac_vsources"/"ac_vcvs"/"ac_cccs"/"ac_ccvs":
      LtiProblem-shaped element records (drives applied),
  "resistor_noise": [ (a_term, b_term, R_ohms), ... ],          # 4kT/R at 300.15 K
  "output_weights", "sense", "vin_norm", "freqs", "band",
  "dc_guesses": [[floats length n_aug], ...],
  "dc_options": [max_iterations, voltage_tolerance, step_limit, gmin],
      # ac_solver values: [100, min(dc_tol, 1e-10), max(0.25, rail_span/4), 1e-12]
  "latch_nodes": (idx0, idx1) | None,
}
```

Candidate: `{"devices": [[w_um, l_um, nf, mult, delvto], ...], "corner": str,
"seed"?: [...], "trust_seed_as_op"?: bool}` — the process corner is
per-candidate (the `apply_silicon_corner` semantics: one name stamped on every
device), `delvto` is the per-device mismatch volts (`0.0` mirrors the Python
`mismatch_v=0.0` default).

Per-candidate pipeline (each step 1:1 with the frozen path):
`CompiledPdk::numeric_card` → `Bsim4ModelCard`-equivalent normalization
(lower-cased keys, `level`/`version` dropped) → **TSMC28 only:**
`co_pdk::apply_mulu0_fold` (the `to_bsim4_cards()` fold: always pop `mulu0`
from the instance card; multiply the model `u0` when non-unity; error if `u0`
is absent) → `co_bsim4` `create(polarity, temp_K)` / `set_model*` /
`set_instance*` / `setup` (the `_NativeDevice.__init__` sequence; fresh handles
per candidate, freed on drop) → `bsim_transient::solve_dc` (the exact kernel
Python's rust engine calls through `Bsim4TransientProblem.solve_dc`) → one
`eval_vp` per device at the op (the raw 4×4 G/C are
`get_terminal_linearization`, and the eval biases the handle for the noise
call — the evaluate-then-noise order `noise_solver` requires) → dense-device
LTI AC + transposed noise solve, per-device `max(Re(z·S·z*), 0)` with the 4×4
total spectral density, resistor `4kT/R`, then `bw_from_gain` / `band_rms`.

### Silicon differential gate

Reference = the frozen Python scalar path itself (`ac_solve` +
`noise_analysis` under the rust engine — the `bench_sweep`
`evaluate_ac`/`evaluate_ac_noise` semantics). BSIM4 evaluation is a pure
function of (card, bias) — there is **no** OTFT-style warm/cold split — so one
seeded gate covers parity directly. Observed on the three 5T OTA examples
(geometry × corner matrices, 21 (candidate, corner) cases): gain worst rel
~4e-16, bandwidth ~2e-15, IRN ~1.7e-16 (the `band_rms` summation ULP), and the
same-seed Rust DC Newton reproduces the Python operating point **bit-for-bit**
(1 iteration from a converged seed). Seeded `delvto` mismatch matches to
~1.4e-16. Byte-identical across workers {1, 2, 8}; zero Python
BSIM4-backend/solver callbacks during the batch.

### Silicon documented deviations

- **SKY130 reference width (`extract_w`) is outside the surface.** The frozen
  loader accepts `reference_width_um` (the device wrapper always passes
  `extract_w` / its class default), letting the instance `w` differ from the
  card-stem width. `CompiledPdk::numeric_card` has no reference-width
  parameter (reference = actual width, the loader's `None` branch), so the
  campaign covers `extract_w == W` geometries — bundled card stems. Circuits
  that pin `extract_w != W` (the sky130 explore path) need an R5-B surface
  extension before R5-D can route them through the campaign.
- **TSMC28 `mulu0` delivery values.** The exercised core-library bins/corners
  all carry `mulu0 = 1.0`, so end-to-end tsmc28 parity exercises the
  (load-bearing) *pop* arm — an un-popped `mulu0` would fail `set_instance` —
  while the multiply and missing-`u0` arms are pinned by direct `co-pdk` unit
  tests (`apply_mulu0_fold`, exact IEEE product).
- **TSMC28 nmos ff/sf/fs bins.** Some OTA geometries select zero bins in the
  ff/sf/fs sections of this delivery; both sides reject them identically
  (`ValueError` ↔ per-candidate `{ok: False}`), and the parity matrices use
  tt/ss where all bins resolve.

### Result dicts (candidate-index ordered)

```
{"ok": True, "gain_peak_dB", "bw_Hz", "irn_uV", "latch_dV",
 "dc_op": [floats length n_aug], "dc_iterations": int, "dc_from_seed": bool}
# or, for a candidate that could not be evaluated:
{"ok": False, "error": str}          # a bad candidate never sinks the batch
```

### Differential gate (parity vs the frozen Python scalar path)

The semantically-correct reference is a **cold-consistent** Python evaluation:
fresh (cold) `PMOS_TFT` small-signal params at the shared DC op → the same
`circuitopt_core.LtiProblem` → the same `bw_from_gain` / `band_rms` reductions
(under the rust device engine, which dispatches to the identical `otft` kernels
the campaign uses). Against it the campaign is **bit-for-bit**: gain/bandwidth
worst rel ~1e-15 across sizes × {typical, slow, fast}, IRN ~2e-16 (the
`band_rms` naive-sum-vs-numpy-`np.sum` ULP), DC operating point bit-for-bit when
seeded, and byte-identical across worker counts {1, 2, 8}. Seeded mismatch
samples match the same reference to ~1e-15.

Cache/threads: one Rayon pool per `evaluate_batch`, sized to `workers`; the
frequency-level `lti` par-iter runs on that same pool (no nesting, no second
pool). No PDK/device text or numeric values cross the boundary beyond the
marshalled template the caller already owns (D12).

### Documented deviations

- **`band_rms` summation.** The trapezoid integral accumulates sequentially;
  NumPy's `np.trapezoid` reduces through `np.sum` (8-way unrolled + pairwise).
  The two agree to a relative error well inside `1e-12` (observed ~2e-16 on the
  IRN), not bit-for-bit.
- **AFE OTFT internal-node operating point is seed-path-dependent (flagged).**
  The device's internal 2-node Newton (`PMOS_TFT._solve_internal`) stops at
  `tol=1e-12`, so its `(Vs1, Vd1)` — and the `gm`/`gds` derived from it — depend
  on the Newton seed. The frozen `ac_solver`/`corners.metrics` path warm-starts
  this solve from a per-instance cache; a cold evaluation of the *same* model
  (e.g. `noise_solver.device_psd`, or a fresh instance) lands on an equally
  valid but different point. Python's own warm-vs-cold `gm`/`gds` disagree by up
  to ~6e-8 for the locked AFE design. The campaign is deterministically
  **cold-seed-consistent**, so it reproduces the cold reference bit-for-bit but
  agrees with the warm `corners.metrics` path only to that inherent ~1e-8 floor.
  This is a property of the frozen model, not a port error; wiring `metrics` to
  the campaign (R5-D) would standardise the whole stack on the cold path.
- **Circuit-level DC root.** The campaign's Rust circuit Newton (numeric
  Jacobian + backtracking) is not a bit-for-bit reproduction of the scipy
  `fsolve` (MINPACK hybrd) guard cascade in `ac_solver`. When seeded from a
  converged Python DC op it reproduces that op bit-for-bit (0 iterations); the
  cold multi-guess path can select a different physical branch on multistable
  points and is not exercised by the parity gate (which seeds from the Python
  reference). No silent root substitution occurs — a non-converging candidate is
  flagged `{"ok": False}`.
