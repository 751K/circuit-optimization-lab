#!/usr/bin/env python3
"""
Half-circuit symbolic analysis of the AFE via SLiCAP.

The AFE is fully differential. For pure differential-mode analysis, the symmetry
plane (NET2, NET20) is at virtual ground, halving the node count from ~13 to ~7.

Half-circuit mapping (diff mode, NET2 = NET20 = 0):
  M7_half:  input transistor, source → GND (virtual ground)
  M9_half:  output stage (common-gate), drain → GND
  M12/13:   cross-coupled pair → single VCCS 2·gm_x from VFBP to GND (positive fb)
            plus a VCVS-inverted VFBN node for the cross-coupled Cgd
  M14/15:   diode load → conductance gm_l from VFBP to GND
  M6, M11:  tail current sources → open (virtual ground at both ends)

Usage:
  /opt/miniconda3/envs/daily/bin/python tools/slicap/half_circuit.py          # numeric
  /opt/miniconda3/envs/daily/bin/python tools/slicap/half_circuit.py --sym    # symbolic
"""

import sys, os

PROJECT = "/Volumes/MacoutDsik/Code/Circuit_Optimizaion"
os.chdir(PROJECT)

import numpy as np

SYMBOLIC = "--sym" in sys.argv

# ── AFE design ────────────────────────────────────────────────────────────
SIZES = {
    "M6": (2264, 78), "M7": (61365, 61), "M8": (61365, 61),
    "M9": (3175, 468), "M10": (3175, 468), "M11": (465, 66),
    "M12": (894, 85), "M13": (894, 85), "M14": (5224, 46), "M15": (5224, 46),
}
BIAS = {"VDD": 40.0, "VCM": 30.65, "VB": 9.84, "VC": 16.0}

# ── Step 1: Extract small-signal parameters ───────────────────────────────
print("=" * 70)
print("Step 1: DC operating point & small-signal parameters")
print("=" * 70)

from core.topology import AFE_TOPO
from core.ac_solver import ac_solve
from core.pmos_tft_model import PMOS_TFT

ac_result = ac_solve(SIZES, BIAS, np.array([1.0]), topo=AFE_TOPO)
dc = ac_result["dc_op"]

def _v(node):
    if node in AFE_TOPO.idx:
        return dc[node]
    if node in BIAS:
        return BIAS[node]
    rv = AFE_TOPO.rails.get(node, 0.0)
    return float(rv) if rv is not None else 0.0

ss = {}
for name, d, g, s in AFE_TOPO.devices:
    W, L = SIZES[name]
    tft = PMOS_TFT(W=W, L=L)
    osv = tft.get_os(_v(s), _v(d), _v(g))
    rout = osv["rout"]
    gds = 1.0 / rout if rout != 0 and np.isfinite(rout) else 0.0
    ss[name] = {"gm": osv["gm"], "gds": gds, "Cgs": osv["Cgss"], "Cgd": osv["Cgdd"]}

# ── Parameter groups (symbolic mode) ─────────────────────────────────────
# Using symmetry: M7=M8, M9=M10, M12=M13, M14=M15
p = {
    "gm_i":   ss["M7"]["gm"],   "gds_i":  ss["M7"]["gds"],
    "Cgs_i":  ss["M7"]["Cgs"],  "Cgd_i":  ss["M7"]["Cgd"],
    "gm_o":   ss["M9"]["gm"],   "gds_o":  ss["M9"]["gds"],
    "Cgs_o":  ss["M9"]["Cgs"],  "Cgd_o":  ss["M9"]["Cgd"],
    "gm_x":   ss["M12"]["gm"],  "gds_x":  ss["M12"]["gds"],
    "Cgs_x":  ss["M12"]["Cgs"], "Cgd_x":  ss["M12"]["Cgd"],
    "gm_l":   ss["M14"]["gm"],  "gds_l":  ss["M14"]["gds"],
    "Cgs_l":  ss["M14"]["Cgs"],
}
# Note: M14 Cgd is gate(0) to drain(VFBN); gate=0 so Cgd_l is from 0 to VFBN.
# In half-circuit → Cgd_l from 0 to VFBP (symmetry)

for k, v in p.items():
    print(f"  {k:8s} = {v:.4e}")

# ── Step 2: Build half-circuit netlist ────────────────────────────────────
print(f"\n{'='*70}")
print("Step 2: Build half-circuit SLiCAP netlist")
print("=" * 70)


def _val(key):
    """Return symbolic {param} or numeric value."""
    if SYMBOLIC:
        return f"{{{key}}}"
    return f"{p[key]:.6e}"


def _r_val(gds_key):
    """Resistance value: symbolic via conductance param g_gds_X, numeric as 1/gds."""
    if SYMBOLIC:
        return f"{{g_{gds_key}}}"
    return f"{1.0/p[gds_key]:.6e}"


def _expr(expr_str, numeric_val):
    """Arithmetic expression (symbolic) or numeric value."""
    if SYMBOLIC:
        return f"{{{expr_str}}}"
    return f"{numeric_val:.6e}"


def build_half_netlist():
    """
    Half-circuit nodes (AC): 0, INP, VOP, VFBP

    Devices:
      M7_half  — input pair half     (d=VOP,  g=INP,  s=0)
      M9_half  — output stage half   (d=0,    g=VFBP, s=VOP)
      X_half   — cross-coupled equiv (d=VFBP, g=VOP,  s=0)  gm=+2gm_x
      L_half   — diode load half     (d=0,    g=0,    s=VFBP)
    """
    lines = []
    lines.append("AFE_Half_Circuit")
    lines.append("* Half-circuit small-signal equivalent for differential-mode analysis")
    lines.append("* Symmetry plane NET2=NET20=0 (virtual ground)")
    lines.append(f"* Mode: {'SYMBOLIC' if SYMBOLIC else 'NUMERIC'}")
    lines.append("")

    if SYMBOLIC:
        for key in p:
            lines.append(f".param {key}={{{key}}}")
        # Conductance params (avoid nested {1/{gds}} syntax)
        for g_key in ["gds_i", "gds_o", "gds_x", "gds_l"]:
            lines.append(f".param g_{g_key}={{1/{g_key}}}")
        lines.append(".param C_L={C_L}")
        lines.append("")

    # ── M7_half: input transistor (source at virtual ground) ──────────
    lines.append("* M7_half: input pair (d=VOP, g=INP, s=0)")
    lines.append(f"G_i VOP 0 INP 0 {_val('gm_i')}")
    lines.append(f"R_i VOP 0 {_r_val('gds_i')}")
    lines.append(f"C_i_gs INP 0 {_val('Cgs_i')}")
    lines.append(f"C_i_gd INP VOP {_val('Cgd_i')}")
    lines.append("")

    # ── M9_half: output stage (drain at GND) ──────────────────────────
    lines.append("* M9_half: output stage (d=0, g=VFBP, s=VOP)")
    lines.append(f"G_o 0 VOP VFBP VOP {_val('gm_o')}")
    lines.append(f"R_o 0 VOP {_r_val('gds_o')}")
    lines.append(f"C_o_gs VFBP VOP {_val('Cgs_o')}")
    lines.append(f"C_o_gd VFBP 0 {_val('Cgd_o')}")
    lines.append("")

    # ── M14/15_half: diode load ───────────────────────────────────────
    lines.append("* L_half: diode load (M14/M15 equivalent)")
    lines.append(f"G_l 0 VFBP 0 VFBP {_val('gm_l')}")
    lines.append(f"R_l 0 VFBP {_r_val('gds_l')}")
    lines.append(f"C_l_gs 0 VFBP {_val('Cgs_l')}")
    lines.append("")

    # ── M12/M13_half: positive-feedback cross-coupling ────────────────
    # Full circuit: M12 (d=VFBN, g=VOP, s=NET20) + M13 (d=VFBP, g=VON, s=NET20)
    # Diff mode: NET20=0, VON=-VOP, VFBN=-VFBP
    #
    # KCL at VFBP (M13 side): +gm_x·VOP - VFBP·(g_x + gm_l + g_l) = 0
    # The M12 side is symmetric and does NOT add extra current into VFBP.
    # → G_x injects +gm_x·VOP into VFBP (positive feedback)
    # → Conductance from VFBP to GND: gds_x (from M13 only; M12's gds on VFBN side)
    # → Cgs from VOP to GND: 2·Cgs_x (M12+M13, both connect to VOP or its complement)
    # → Cgd: self-term as 2·Cgd from VFBP to GND (trans-term VOP→VFBP mainly
    #   affects the RHP zero, negligible for DC gain and dominant poles)
    lines.append("* X_half: cross-coupled pair (M12/M13, positive feedback)")
    lines.append(f"G_x 0 VFBP VOP 0 {_val('gm_x')}")  # +gm_x·VOP into VFBP
    lines.append(f"R_x 0 VFBP {_r_val('gds_x')}")
    lines.append(f"C_x_gs VOP 0 {_expr('2*Cgs_x', 2.0*p['Cgs_x'])}")
    lines.append(f"C_x_gd 0 VFBP {_expr('2*Cgd_x', 2.0*p['Cgd_x'])}")
    lines.append("")

    # ── Load capacitor ─────────────────────────────────────────────────
    cl_val = "{C_L}" if SYMBOLIC else "5e-12"
    lines.append("* Load capacitor on VOP")
    lines.append(f"C_L VOP 0 {cl_val}")
    lines.append("")

    # ── Bias rails (AC ground) ─────────────────────────────────────────
    lines.append("* Bias rails (AC ground)")
    lines.append("V_VDD VDD 0 40")
    lines.append("V_VB  VB  0 9.84")
    lines.append("V_VC  VC  0 16")
    lines.append("")

    # ── Stimulus & detector ────────────────────────────────────────────
    lines.append("* Input: V_INP drives the half-circuit input gate")
    lines.append("V_INP INP 0 0")
    lines.append("")
    lines.append("* Output: VOP (single-ended half-circuit output)")
    lines.append("")
    lines.append(".source V_INP")
    lines.append(".detector V_VOP")
    lines.append(".end")

    return "\n".join(lines)


netlist = build_half_netlist()
print(netlist[:3000])
print("...")

os.makedirs("cir", exist_ok=True)
cir_name = "afe_half_symbolic.cir" if SYMBOLIC else "afe_half.cir"
with open(os.path.join("cir", cir_name), "w") as f:
    f.write(netlist)
print(f"\nNetlist → cir/{cir_name}")

# ── Step 3: SLiCAP Laplace analysis ───────────────────────────────────────
print(f"\n{'='*70}")
print("Step 3: SLiCAP Laplace & pole-zero analysis")
print("=" * 70)

import SLiCAP

SLiCAP.initProject("AFE_Half")
os.chdir(PROJECT)

circuit = SLiCAP.makeCircuit(cir_name)
print(f"Title: {circuit.title}")
print(f"Nodes: {circuit.nodes}")
print(f"Elements ({len(circuit.elements)}):")
for ref, elem in sorted(circuit.elements.items()):
    print(f"  {ref}: {elem.type} nodes={elem.nodes} params={elem.params}")

lap = SLiCAP.doLaplace(circuit, source='V_INP', detector='V_VOP', transfer='gain')
H_raw = lap.laplace

# ── Symbolic simplification (SLiCAP raw output is unsimplified) ─────
from sympy import I, pi, simplify, fraction, Poly, degree, symbols as sp_symbols

# Build numeric substitution dict for symbolic mode
subs_dict = {}
if SYMBOLIC:
    print("\n--- Simplifying symbolic H(s) ---")
    # Simplify H(s) to a compact rational function
    H_simple = simplify(H_raw)
    print(f"H(s) simplified ({len(str(H_simple))} chars)")

    # Extract numerator and denominator
    num, den = fraction(H_simple)
    s = sp_symbols('s')
    print(f"Numerator degree: {degree(num, s)}")
    print(f"Denominator degree: {degree(den, s)}")

    # DC gain H(0)
    H0_sym = simplify(H_simple.subs(s, 0))
    print(f"\n=== DC Gain (symbolic) ===")
    print(f"H(0) = {H0_sym}")

    # Collect denominator coefficients for pole expressions
    den_coeffs = Poly(den, s).all_coeffs()
    print(f"\nDenominator coefficients (a2*s^2 + a1*s + a0):")
    for i, c in enumerate(den_coeffs):
        print(f"  a{len(den_coeffs)-1-i} = {c}")

    # Substitute numeric defaults to verify
    subs_dict = {sp_symbols(k): p[k] for k in p}
    subs_dict[sp_symbols('C_L')] = 5e-12
    # Also substitute the conductance params
    for gk in ["gds_i", "gds_o", "gds_x", "gds_l"]:
        subs_dict[sp_symbols(f'g_{gk}')] = 1.0 / p[gk]

    H0_num = float(H0_sym.subs(subs_dict).evalf())
    print(f"\nH(0) numeric (from symbolic): {H0_num:.6f} ({20*np.log10(abs(H0_num)):.2f} dB)")

    # Use simplified H for downstream analysis
    H = H_simple
else:
    H = H_raw

print(f"\nH(s) (raw) length: {len(str(H))} chars")

# Pole-zero
pz = SLiCAP.doPZ(circuit, source='V_INP', detector='V_VOP', transfer='gain')
print(f"\nPoles: {pz.poles}")
print(f"Zeros: {pz.zeros}")

# ── Frequency response ────────────────────────────────────────────────────
print(f"\n--- Frequency response ---")

# For symbolic mode, substitute numeric defaults before frequency sweep
if SYMBOLIC:
    H_num = H.subs(subs_dict)
else:
    H_num = H

for f in [0.01, 1.0, 100.0, 1000.0, 10000.0]:
    try:
        H_jw = complex(H_num.subs('s', I * 2 * pi * f).evalf())
        mag = abs(H_jw)
        print(f"  f={f:8.2f} Hz: |H|={mag:.6f} ({20*np.log10(mag):.2f} dB)")
    except Exception as e:
        print(f"  f={f:8.2f} Hz: eval error: {e}")
        break

# ── Step 4: Cross-validate against full circuit ───────────────────────────
print(f"\n{'='*70}")
print("Step 4: Cross-validation (half vs full circuit)")
print("=" * 70)

H_eval = H.subs(subs_dict) if SYMBOLIC else H
H0_half = abs(complex(H_eval.subs('s', 0).evalf()))
gain_db_half = 20 * np.log10(H0_half)
print(f"Half-circuit DC gain (VOP/Vin): {H0_half:.6f} ({gain_db_half:.2f} dB)")

# Full circuit with diff drive: H(s) = V_out_diff / V_in = (VOP-VON)/V_in ≈ 2*VOP/V_in
# Measured SLiCAP V_out gain = 28.130 (from full_circuit.py with diff drive)
# → VOP/V_in ≈ 28.130/2 = 14.065 (half-circuit should match this)
H0_full_vop = 28.130076 / 2.0  # VOP/V_in in full circuit with diff drive
print(f"Full-circuit VOP/V_in (diff drive): {H0_full_vop:.6f} ({20*np.log10(H0_full_vop):.2f} dB)")
print(f"Ratio half/full: {H0_half/H0_full_vop:.6f}")

# Pole comparison
if pz.poles:
    real_poles = [p for p in pz.poles if abs(np.imag(complex(p))) < 1e-9]
    real_poles_hz = [abs(np.real(complex(p))) / (2*np.pi) for p in real_poles]
    half_poles = sorted(real_poles_hz)
    full_poles = [629.4, 808.3]  # from full_circuit.py
    print(f"Half-circuit poles: {[f'{p:.1f} Hz' for p in half_poles]}")
    print(f"Full-circuit poles:  {[f'{p:.1f} Hz' for p in full_poles]}")
    if len(half_poles) >= 2:
        print(f"p1 error: {abs(half_poles[0]-full_poles[0])/full_poles[0]*100:.2f}%")
        print(f"p2 error: {abs(half_poles[1]-full_poles[1])/full_poles[1]*100:.2f}%")

print("\nDone.")
