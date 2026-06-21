# SLiCAP Symbolic Analysis — AFE Half-Circuit

Uses [SLiCAP](https://analog-electronics.eu/slicap/) v5.0.3 for symbolic small-signal
analysis of the 10-T fully-differential AFE.

## Quick Start

```bash
# Numeric validation (fast, for cross-checking)
/opt/miniconda3/envs/daily/bin/python tools/slicap/full_circuit.py
/opt/miniconda3/envs/daily/bin/python tools/slicap/half_circuit.py

# Symbolic derivation (design equations with ~17 free parameters)
/opt/miniconda3/envs/daily/bin/python tools/slicap/half_circuit.py --sym
```

## Files

| File | Purpose |
|------|---------|
| `full_circuit.py` | Full 10-T circuit → SLiCAP Laplace (numeric + symbolic) |
| `half_circuit.py` | Differential half-circuit → SLiCAP Laplace (numeric + symbolic) |
| `cir/` | Generated SPICE netlists (SLiCAP convention, at project root) |

## Workflow

```
Python AC Solver (DC op)
    → gm, gds, Cgs, Cgd per transistor (PMOS_TFT model)
    → Build small-signal equivalent netlist (VCCS + R + C)
    → SLiCAP Laplace analysis → H(s), poles, zeros, design equations
```

## Half-Circuit Mapping

The AFE is fully differential. For pure differential excitation:

| Full Circuit | Half-Circuit Equivalent |
|-------------|------------------------|
| M7/M8 (input pair) | M7_half: source → GND (virtual ground) |
| M9/M10 (output stage) | M9_half: drain → GND |
| M12/M13 (cross-coupled) | VCCS: +gm_x·VOP into VFBP (positive feedback) |
| M14/M15 (diode load) | Conductance gm_l from VFBP to GND |
| M6, M11 (tail) | AC open (both ends at virtual ground) |

**Nodes**: 7 (half) vs 13 (full) → symbolic analysis O(n!) improvement.

## Validation (vs full 10-T circuit)

| Metric | Half-Circuit | Full Circuit | Error |
|--------|-------------|-------------|-------|
| DC Gain | 22.96 dB | 22.96 dB | 0.00% |
| p₁ (dominant pole) | 629.3 Hz | 629.4 Hz | 0.02% |
| p₂ | 791.0 Hz | 808.3 Hz | 2.14% |

The 2.1% p₂ error is from simplified Cgd cross-coupling terms (transcapacitor ignored).

## Environment

- Python: `/opt/miniconda3/envs/daily/bin/python3`
- SLiCAP: v5.0.3
- Project root: `/Volumes/MacoutDsik/Code/Circuit_Optimizaion`
- Netlists: `cir/*.cir` (SLiCAP looks here)

## SLiCAP Syntax Notes

- First non-comment line = title (alphanumeric + underscore only)
- V-source: `V_NAME N+ N- value` (exactly 3 fields)
- Signal source: `.source V_NAME`
- Detector: `.detector V_node` (use V_ prefix for voltage)
- Parameters: `.param name = {name}` (symbolic) or `.param name = value`
- No nested `{}` — use intermediate params for `1/gds` expressions
- `makeCircuit("name.cir")` — include `.cir` extension; looks in `cir/`
