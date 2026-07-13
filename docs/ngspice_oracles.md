# Full-circuit ngspice oracles (FreePDK45)

The local grid solvers (`ac_solve`, `noise_analysis`) evaluate FreePDK45 through a
cached device model that carries only `Cgs`/`Cgd` — no drain/source junction caps —
so on 45 nm they read ~8 % optimistic UGBW (see `freepdk45_fd_ota_design.md` §4.5).
For sign-off numbers, four oracles render the **complete** circuit and let ngspice's
C-BSIM4 (the FreePDK45 oracle) run the analysis with full charge. They live in
`circuitopt/ngspice_ac.py` and share the deck renderer in `circuitopt/ngspice_render.py`
with the existing `.tran` backend, so device M-lines, R/C, E/G/F/H controlled sources,
rails, per-polarity corner routing, temperature and supply bias render identically.

All four honor: **temperature** (`temperature=` in Kelvin → `.options temp=`),
**corner** (`corner=` including the mixed `sf`/`fs`, via `binding.at_corner(...)` or
directly), and **supply** (through `bias`). A multistable OTA DC point is seeded with
`x0_guess={node: V}` → `.nodeset` (use the circuit's `dc_guesses[0]`).

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
