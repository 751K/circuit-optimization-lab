# Full-circuit ngspice oracles (FreePDK45 and TSMC28HPC+)

FreePDK45 uses cached ngspice-C characterization in its local solvers. TSMC28HPC+
normally uses circuitopt's native BSIM4.5 backend; ngspice is retained as an
independent regression oracle. The oracle helpers render the **complete** circuit
and let ngspice run the original compact-model deck with full charge. They live in
`circuitopt/ngspice_ac.py` and share the deck renderer in `circuitopt/ngspice_render.py`
with the `.tran` backend, so device M/X lines, R/C, E/G/F/H controlled sources,
rails, per-polarity corner routing, temperature and supply bias render identically.

FreePDK45's local grid carries only `Cgs`/`Cgd`, so its 45 nm FD-OTA reads about 8%
optimistic UGBW versus the complete deck (see `freepdk45_fd_ota_design.md` §4.5).
Use the full-circuit oracle whenever junction charge, exact bandwidth, switching, or
foundry wrapper behavior matters.

All four honor: **temperature** (`temperature=` in Kelvin → `.options temp=`),
**corner** (`corner=` including the mixed `sf`/`fs`, via `binding.at_corner(...)` or
directly), and **supply** (through `bias`). A multistable OTA DC point is seeded with
`x0_guess={node: V}` → `.nodeset` (use the circuit's `dc_guesses[0]`).

## Process selection

The registered model types choose the renderer:

| Model type | SPICE instance | Model setup | ngspice arguments |
|------------|----------------|-------------|-------------------|
| `freepdk45.nmos` / `.pmos` | flat `M` device | per-polarity `.include` cards | default |
| `tsmc28hpcp_ngspice.nmos` / `.pmos` | `X` wrapper using `nch_mac` / `pch_mac` | explicit five-section `.lib` closure | `-D ngbehavior=hsa` |

A complete circuit must use one process adapter for every MOS. Mixed foundry setup
semantics in one full-circuit ngspice deck are rejected instead of silently selecting
the wrong cards.

TSMC28HPC+ resolves its licensed model from `TSMC28_MODEL_DIR`,
`TSMC28_PDK_ROOT`, the portable project-local
`PDK/tsmc28hpcp/models/hspice/` entry, then `PDK_ROOT/tsmc28hpcp`. The adapter reads
hierarchical `@m.x*.main[...]` operating-point vectors and passes `NF` to the foundry
wrapper. See [TSMC28HPC+ Local Adapter](tsmc28hpcp.md).

## Mixed per-polarity corners: `sf` / `fs`

FreePDK45 ships `models_{nom,ss,ff}/` card directories. The corner name now selects a
directory **per polarity** (`circuitopt.freepdk45_model.corner_card_dir`):

| corner | NMOS card dir | PMOS card dir |
|--------|---------------|---------------|
| `nom` / `tt` | `models_nom` | `models_nom` |
| `ss` | `models_ss` | `models_ss` |
| `ff` | `models_ff` | `models_ff` |
| `sf` | `models_ss` | `models_ff` |
| `fs` | `models_ff` | `models_ss` |

`sf` = NMOS slow + PMOS fast; `fs` the reverse; `tt` is an alias of `nom`. Both the
characterisation-grid path and the full-circuit ngspice render honor them; the grid
cache keys on the corner **name**, so an `sf` NMOS grid (built from the `ss` card) is
cached separately from — and never collides with — the `ss` grid. When the two
polarities differ (`sf`/`fs`), the rendered deck `.include`s **both** card files.
nom/ss/ff decks are byte-identical to the pre-change renderer (golden-locked).

Corner names are **case-insensitive** (`"SF"` behaves as `sf`) and **strictly
validated** on both paths: `None`/`""` mean `nom`, but an unknown name (a typo like
`"sx"`) raises `ValueError` naming the valid set — it never silently falls back to
nominal, so a misspelled corner cannot poison a PVT campaign
(`circuitopt.freepdk45_model.normalize_corner`).

## `ac_ngspice` — small-signal transfer

```python
res = ac_ngspice(sizes, bias, topo=topo,
                 acmag={"VINP": (0.5, 0), "VINN": (0.5, 180)},  # differential drive
                 fstart=1e3, fstop=1e11, points=15,             # .ac dec <points> ...
                 out_nodes=["OUTP", "OUTN"], nf=nf,
                 model_types=..., device_kwargs=..., corner="nom",
                 temperature=300.15, x0_guess=seed)
H = ac_response(res, "OUTP", "OUTN", vin=1.0)   # differential transfer / diff-input mag
peak_gain_db(res["freq"], H)     # passband gain (use for AC-coupled / band-pass)
dc_gain_db(res["freq"], H)       # f->0 gain
unity_gain_freq(res["freq"], H)  # UGBW [Hz]
phase_margin(res["freq"], H)     # PM [deg], referenced to the passband phase
gain_margin_db(res["freq"], H)   # GM [dB] at the -180 deg crossing
```

`acmag` maps a stimulus source (a **rail** name or an ideal-**vsource** name) to
`(magnitude, phase_deg)`; a differential drive is two sources with opposite phase.
`res["nodes"]` holds the complex voltage of every recorded solved node.
Validated on the FD-OTA example: 58.9 dB / 119.9 MHz / 84 deg — matching §4.5.

## `noise_ngspice` — output & input-referred noise

```python
n = noise_ngspice(sizes, bias, topo=topo, out="OUTP", ref="OUTN",   # v(outp,outn)
                  src="VINP", fstart=1e3, fstop=1e9, points=20,
                  band=(1e4, 1e8), ...)
n["onoise_psd"], n["inoise_psd"]     # V^2/Hz over n["freq"]
n["onoise_rms"], n["inoise_rms"]     # sqrt(integral PSD df) over n["band"]
```

`src` (the ngspice `.noise` input source) is driven `ac 1` automatically so `inoise`
is meaningful. `out`/`ref` give single-ended `v(out)` or differential `v(out,ref)`.
PSD is the ngspice `*noise_spectrum` amplitude density squared. A bare resistor reads
`4kTR` to <2 %.

## `op_ngspice` — operating point + saturation check

```python
op = op_ngspice(sizes, bias, topo=topo, margin=0.0, ...)
op["M1"]  # {"vds","vgs","vdsat","id","gm","gds","region_ok"}
```

`region_ok = |vds| >= |vdsat| + margin` (absolute values handle NMOS/PMOS uniformly) —
the saturation-region test for a bias-point audit across the PVT grid.

For a charge-transfer circuit, replacing the final transient state with a new DC
solve can discard sampled charge. Pass `op_devices=["M1", ...]` to
`transient_ngspice` instead: `device_op` then contains the same six quantities as
waveforms and `device_op_final` contains their final-sample values. This is the
MDAC campaign's saturation oracle after maximum residue and major-carry settling.

## `loop_gain_ngspice` — loop gain & phase margin

Chosen method: **Middlebrook single voltage injection**. It needs only one
testbench-side ideal voltage source in series in the loop (no loop-breaking inductor —
our renderer already supports ideal vsources and E/G/F/H controlled sources), and it
preserves the DC operating point because the injection source is 0 V at DC.

Recipe: insert an ideal vsource `Vinj` **in series in the loop**, at a high-impedance /
low-impedance boundary — a transistor **gate** is ideal. Its `p` terminal faces the
DUT input (high-Z gate), its `q` terminal faces the driver (low-Z output); DC value 0.

```python
lg = loop_gain_ngspice(sizes, bias, topo=topo, inject="Vinj",
                       fstart=1e3, fstop=1e10, points=30, ...)
lg["loop_gain"]  # complex T(f); lg["ugf"], lg["pm"], lg["gm_db"]
```

The source is driven `ac 1` and `T = -V(q)/V(p)` at the break (exact at a high-Z/low-Z
boundary). Validated against an analytic single-pole feedback loop (PM within a few
degrees). To apply it to the FD-OTA:

- **Differential loop** — split an input-pair gate node into two (`G1` → `G1a`/`G1b`),
  put `Vinj` between them, and close the amplifier in unity feedback in the testbench.
- **CMFB loop** — split the common-mode control gate node (the `CTRL` net driving the
  PMOS-load gates) and inject there; the sense pairs + mirror close the CM loop.

Because PM/UGBW/GM are magnitude/relative-phase quantities, the loop-gain sign
convention (which depends on ngspice's controlled-source current sense) does not affect
them.

### `loop_gain_tian_ngspice` — exact loop gain at a capacitive break

Single voltage injection is exact **only while the `p` side stays high-impedance
relative to the `q` side across the whole sweep**. At a MOS gate this fails at RF:
a large input pair's Cgg (pF-class) falls to hundreds of ohms right around loop
crossover, and the reported PM becomes a probe artifact — we measured a two-stage
MDAC OTA loop reporting PM ≈ 98° while its closed-loop transient rang at ~500 MHz
(true margin ≈ 25°); `gm_db = nan` in the sweep was the tell.

`loop_gain_tian_ngspice` (same signature/return shape) removes the impedance
condition with Tian's double injection (IEEE Circuits & Devices 17(1), 2001): a
second AC run drives a testbench current source into the break's `p` node with the
0 V vsource left in place, and the two runs' `v(p)`/`i(Vinj)` combine as
`T = -1/(1 - 1/(2*(i1*v2 - v1*i2) + v1 + i2))` — exact for arbitrary impedances on
both sides, orientation-symmetric, validated to machine precision against a
two-pole analytic loop (`tests/test_pvt_machinery.py`). **PM sign-off for any loop
broken at a capacitive gate must use this variant.** By default the v- and
i-injection sweeps are CHAINED into one ngspice process (the current sources sit
in the deck at `ac 0`, an `alter @src[acmag]` swaps the drive between sweeps), so
the deck's fixed model-expansion cost is paid once — measured bit-identical to
the two-process path on both analytic and foundry decks. Set
`CIRCUITOPT_NGSPICE_CHAIN=0` (or `chain=False`) to force the historical
one-process-per-sweep behaviour.
