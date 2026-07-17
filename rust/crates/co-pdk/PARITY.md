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
