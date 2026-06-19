#!/usr/bin/env python3
"""
SLiCAP symbolic analysis of the AFE — derive design equations.

Uses symbolic .param definitions for all small-signal parameters so SLiCAP
returns H(s) as a rational function of gm, gds, Cgs, Cgd, and load C.
"""
import os, sys
os.chdir("/Volumes/MacoutDsik/Code/Circuit_Optimizaion")

import SLiCAP
SLiCAP.initProject("AFE_Symbolic")
os.chdir("/Volumes/MacoutDsik/Code/Circuit_Optimizaion")

# Build the symbolic small-signal netlist
# All gm, gds, C values are symbolic params -> SLiCAP keeps them as symbols in H(s)

lines = ["* AFE Small-Signal Symbolic Analysis"]
lines.append("* Parameters: gm_X, gds_X, Cgs_X, Cgd_X, C_L for each transistor")
lines.append("")

# Define all symbolic parameters
params = []
for name in ["M6","M7","M8","M9","M10","M11","M12","M13","M14","M15"]:
    params.append(f".param gm_{name} = {{gm_{name}}}")
    params.append(f".param gds_{name} = {{gds_{name}}}")
    params.append(f".param Cgs_{name} = {{Cgs_{name}}}")
    params.append(f".param Cgd_{name} = {{Cgd_{name}}}")
params.append(".param C_L = {C_L}")
lines.extend(params)
lines.append("")

# Device connections
devices = [
    ("M6",  "NET2",  "VB",   "VDD"),
    ("M7",  "VOP",   "INP",  "NET2"),
    ("M8",  "VON",   "INN",  "NET2"),
    ("M9",  "0",     "VFBP", "VOP"),
    ("M10", "0",     "VFBN", "VON"),
    ("M11", "NET20", "VC",   "VDD"),
    ("M12", "VFBN",  "VOP",  "NET20"),
    ("M13", "VFBP",  "VON",  "NET20"),
    ("M14", "0",     "0",    "VFBN"),
    ("M15", "0",     "0",    "VFBP"),
]

# Build small-signal equivalent elements
for name, d, g, s in devices:
    lines.append(f"G_{name} {d} {s} {g} {s} {{gm_{name}}}")
    lines.append(f"R_{name} {d} {s} {{1/gds_{name}}}")
    if g != s:
        lines.append(f"C_{name}_gs {g} {s} {{Cgs_{name}}}")
    if g != d:
        lines.append(f"C_{name}_gd {g} {d} {{Cgd_{name}}}")

lines.append("")
lines.append("* Load capacitors")
lines.append("C_LP VOP 0 {C_L}")
lines.append("C_LN VON 0 {C_L}")
lines.append("")

lines.append("* Bias rails (AC ground)")
lines.append("V_VDD VDD 0 40")
lines.append("V_VB  VB  0 9.84")
lines.append("V_VC  VC  0 16")
lines.append("")

lines.append("* Input source")
lines.append("V_INP INP 0 0")
lines.append("V_INN INN 0 0")
lines.append("")

lines.append("* Differential output detector")
lines.append("E_OUT out 0 VOP VON 1.0")
lines.append("")

lines.append(".source V_INP")
lines.append(".detector V_out")
lines.append(".end")

netlist = "\n".join(lines)
with open("cir/afe_symbolic.cir", "w") as f:
    f.write(netlist)
print("Symbolic netlist written to cir/afe_symbolic.cir")

# Load circuit
circuit = SLiCAP.makeCircuit("afe_symbolic.cir")
print(f"Elements: {len(circuit.elements)}")
print(f"Params: {circuit.params}")

# ── Run purely symbolic Laplace analysis ──
print("\n=== Symbolic Laplace Analysis ===")
result = SLiCAP.doLaplace(circuit, source='V_INP', detector='V_out', transfer='gain')
H_sym = result.laplace
print(f"H(s) = {H_sym}")

import sympy as sp
from sympy import simplify, factor, collect, fraction

# Simplify
H_simple = simplify(H_sym)
print(f"\nSimplified H(s):")
print(H_simple)

# Get numerator and denominator
num, den = fraction(H_simple)
print(f"\nNumerator degree in s: {sp.degree(num, sp.Symbol('s'))}")
print(f"Denominator degree in s: {sp.degree(den, sp.Symbol('s'))}")

# Count total free symbols
all_syms = H_simple.free_symbols
print(f"Total free symbols: {len(all_syms)}")
print(f"Symbols: {sorted([str(x) for x in all_syms])[:30]}...")
