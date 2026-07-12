---
name: slicap
description: Use SLiCAP for symbolic circuit analysis — derive transfer functions, poles/zeros, and design equations from SPICE-like netlists.
allowed-tools: Bash, Read, Write, Edit, WebFetch, WebSearch
model: sonnet
---

# SLiCAP Symbolic Circuit Analysis

You are a skilled SLiCAP user. When invoked, you help the user do **symbolic** circuit analysis on the AFE (or other circuits in this project).

## Environment

- **Python**: `/opt/miniconda3/envs/daily/bin/python3`
- **SLiCAP**: v5.0.3, installed in the `daily` conda env
- **Project root**: `/Volumes/MacoutDsik/Code/Circuit_Optimizaion`
- **SLiCAP project dir**: same as project root (SLiCAP inits there)
- **Circuit files**: place under `cir/*.cir` (SLiCAP looks there)

**Always** run SLiCAP scripts with `os.chdir("/Volumes/MacoutDsik/Code/Circuit_Optimizaion")` BEFORE `initProject`, because `initProject` creates subdirectories and may change cwd.

## Core Workflow

### 1. Obtain Small-Signal Parameters

This project uses a **custom PMOS_TFT model** (not standard SPICE). SLiCAP cannot simulate it directly. The workflow is:

```
Python AFE Solver (DC op)
    → gm, gds, Cgs, Cgd for each transistor
    → Build small-signal equivalent netlist (VCCS + R + C)
    → SLiCAP symbolic Laplace analysis
    → H(s), poles, zeros, design equations
```

To extract small-signal parameters, pipe through the Python solver:

```python
import sys; sys.path.insert(0, "/Volumes/MacoutDsik/Code/Circuit_Optimizaion")
from core.topology import AFE_TOPO
from core.ac_solver import ac_solve
from core.pmos_tft_model import PMOS_TFT
# ... dc = ac_solve(sizes, bias, freqs, topo=AFE_TOPO)["dc_op"]
# ... for each device: tft.get_os(Vs, Vd, Vg) → gm, rout, Cgss, Cgdd
```

### 2. Build SLiCAP-Compatible Netlist

Each transistor becomes:
```
G_NAME <drain> <source> <gate> <source> {gm_param}    ← VCCS: i_d = gm*(Vg-Vs)
R_NAME <drain> <source> {1/gds_param}                 ← output resistance
C_NAME_gs <gate> <source> {Cgs_param}                 ← gate-source cap
C_NAME_gd <gate> <drain> {Cgd_param}                  ← gate-drain cap
```

**CRITICAL SLiCAP syntax rules:**
- FIRST non-comment line must be a title (plain ID string)
- V-source format: `V_NAME N+ N- value` (only 2 nodes + value, NO `dc 0 ac 0.5`)
- Signal source declared via `.source V_NAME` directive
- Detector declared via `.detector V_node` (use `V_<node>` for voltage)
- Parameters: `.param name = {name}` for symbolic, or `.param name = value` for numeric
- Use `{param_name}` in element values to reference parameters
- DO NOT use standard SPICE `dc`, `ac` qualifiers — SLiCAP doesn't understand them
- File extension: `.cir`, placed in `cir/` subdirectory

**Parameter naming:** use meaningful names that reflect transistor groups:
- Input pair: `gm_i, gds_i, Cgs_i, Cgd_i`
- Output stage: `gm_o, gds_o, Cgs_o, Cgd_o`
- Cross-coupled: `gm_x, gds_x, Cgs_x, Cgd_x`
- Diode loads: `gm_l, gds_l, Cgs_l, Cgd_l`
- Load capacitance: `C_L`
- etc.

### 3. Load and Analyze

```python
import os; os.chdir("/Volumes/MacoutDsik/Code/Circuit_Optimizaion")
import SLiCAP

SLiCAP.initProject("AFE_Symbolic")
os.chdir("/Volumes/MacoutDsik/Code/Circuit_Optimizaion")  # initProject may chdir

circuit = SLiCAP.makeCircuit("filename.cir")  # looks in cir/

# Symbolic Laplace analysis
result = SLiCAP.doLaplace(circuit, source='V_INP', detector='V_out', transfer='gain')
H = result.laplace  # sympy expression

# Pole-zero analysis (numeric: substitutes .param defaults)
pz = SLiCAP.doPZ(circuit, source='V_INP', detector='V_out', transfer='gain')
```

### 4. Interpret and Simplify

```python
from sympy import simplify, fraction, degree, factor, collect, Poly
import sympy as sp

H_s = simplify(H)   # may be slow for large expressions!
num, den = fraction(H_s)
s = sp.Symbol('s')
degree_num = degree(num, s)
degree_den = degree(den, s)

# DC gain: H(0)
H0 = H_s.subs(s, 0)

# Extract coefficients
coeffs_den = Poly(den, s).all_coeffs()  # [a2, a1, a0] for 2nd-order
```

## Strategy for Symbolic Analysis

### Full Symbolic (all ~25 params)
- **When**: you want the COMPLETE structural form
- **Risk**: raw expression may be huge (100K+ chars). Use `simplify()` — sympy's simplification handles the 13x13 sparse MNA cancellation well, typically reducing to ~1K chars
- **Expect**: H(s) is 2nd-order after pole-zero cancellation (verified for this AFE)

### Targeted Symbolic (keep 3-7 key params, substitute others numeric)
- **When**: you want practical design equations showing parameter dependencies
- **Approach**: put numeric values directly in the netlist for non-key elements; use `{param}` syntax only for key ones
- **Typical key params**: gm_i (sets gain), C_L (sets pole), gm_x (positive feedback)

### Numeric (all concrete values)
- **When**: you just need the numbers to cross-check with the Python solver
- **Approach**: no `{param}` brackets, just raw numbers in element lines

## Common Pitfalls

1. **Title line**: The netlist must start with a non-comment ID line as the circuit title
2. **V-source syntax**: `V_NAME N+ N- DCvalue` — only 3 fields after the name. No `ac`, `dc` qualifiers
3. **Detector name**: Use `V_out` not `out` (SLiCAP prefixes node voltages with `V_`)
4. **Case sensitivity**: Module is `SLiCAP` (capital S, L, C, A, P). `import SLiCAP`, not `slicap`
5. **initProject chdir**: Always `os.chdir` back to project root after `initProject`
6. **`{1/gds}`**: This works in SLiCAP — it evaluates the expression during substitution
7. **File extension matters**: SLiCAP determines the netlister from the extension. Use `.cir` for SPICE-like syntax

## Files Created During Analysis

- `cir/` — netlist files (`.cir`)
- `lib/` — compiled model libraries
- `html/`, `csv/`, `img/`, `tex/` — SLiCAP output directories
- `SLiCAP.ini` — project configuration (auto-generated)
- `~/SLiCAP.ini` — user configuration (auto-generated)
