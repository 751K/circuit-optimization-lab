#!/usr/bin/env python3
"""
Full 10-T AFE symbolic analysis via SLiCAP small-signal equivalent circuit.

Workflow:
  1. Run the Python AFE solver → DC op + small-signal parameters (gm, gds, Cgs, Cgd)
  2. Build a SLiCAP-compatible SPICE netlist (VCCS + R + C equivalents)
  3. SLiCAP symbolic Laplace → H(s), poles, zeros
  4. Cross-validate against the Python AC solver

Two modes:
  - numeric:  substitute concrete values → fast, for validation
  - symbolic: keep params as {symbols} → design equations

Usage:
  /opt/miniconda3/envs/daily/bin/python tools/slicap/full_circuit.py          # numeric
  /opt/miniconda3/envs/daily/bin/python tools/slicap/full_circuit.py --sym    # symbolic
"""

import sys, os

PROJECT = "/Volumes/MacoutDsik/Code/Circuit_Optimizaion"
os.chdir(PROJECT)

SYMBOLIC = "--sym" in sys.argv

import numpy as np

# ── AFE design (locked) ───────────────────────────────────────────────────
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

from circuitopt.topology import AFE_TOPO
from circuitopt.ac_solver import ac_solve
from circuitopt.device_model import create_device

ac_result = ac_solve(SIZES, BIAS, np.array([1.0]), topo=AFE_TOPO)
dc = ac_result["dc_op"]
print(f"DC solved: {len(dc)} nodes")
print(f"AC gain @ 1Hz: {ac_result['gains'][0]:.6f} ({20*np.log10(ac_result['gains'][0]):.2f} dB)")

ss = {}
for name, d, g, s in AFE_TOPO.devices:
    W, L = SIZES[name]
    def _v(node):
        if node in AFE_TOPO.idx:
            return dc[node]
        if node in BIAS:
            return BIAS[node]
        rv = AFE_TOPO.rails.get(node, 0.0)
        return float(rv) if rv is not None else 0.0
    vd, vg, vs = _v(d), _v(g), _v(s)
    tft = create_device("pmos_tft", W=W, L=L)
    osv = tft.get_os(vs, vd, vg)
    rout = osv["rout"]
    gds = 1.0 / rout if rout != 0 and np.isfinite(rout) else 0.0
    ss[name] = {
        "gm": osv["gm"], "gds": gds,
        "Cgs": osv["Cgss"], "Cgd": osv["Cgdd"],
        "Vs": vs, "Vd": vd, "Vg": vg,
    }

for name in sorted(ss.keys()):
    p = ss[name]
    print(f"  {name}: gm={p['gm']:.4e} gds={p['gds']:.4e} Cgs={p['Cgs']:.4e} Cgd={p['Cgd']:.4e}")

# ── Step 2: Build SLiCAP netlist ──────────────────────────────────────────
print(f"\n{'='*70}")
print("Step 2: Build SLiCAP small-signal netlist")
print("=" * 70)


def build_full_netlist(ss, symbolic=False):
    """Build SLiCAP-compatible small-signal netlist for the full 10-T AFE.

    Each transistor → VCCS(gm) + R(gds) + Cgs + Cgd.
    SLiCAP syntax: title line, V-sources as 'V_NAME N+ N- value',
    .source / .detector directives, {param} for symbolic.

    Drives INP single-ended, INN grounded. Output = VOP - VON via E_OUT.
    """
    lines = []
    lines.append("AFE_Full_10T")  # SLiCAP title: no spaces/hyphens
    lines.append("* PMOS_TFT -> VCCS(gm) + R(gds) + Cgs + Cgd")
    lines.append("* Input: V_INP at INP (M7 gate), INN grounded")
    lines.append("* Output: differential VOP-VON via E_OUT")
    lines.append("")

    # Symbolic parameters (targeted: gm only by group, rest numeric)
    if symbolic:
        lines.append("* Targeted symbolic: gm grouped by symmetry, others numeric")
        # Group symmetric pairs: M7=M8, M9=M10, M12=M13, M14=M15
        lines.append(".param gm_i={gm_i}")     # input pair M7/M8
        lines.append(".param gm_o={gm_o}")     # output stage M9/M10
        lines.append(".param gm_x={gm_x}")     # cross-coupled M12/M13
        lines.append(".param gm_l={gm_l}")     # diode load M14/M15
        lines.append(".param gm_t={gm_t}")     # tail M6
        lines.append(".param gm_k={gm_k}")     # tail M11
        lines.append("")

    # Device connections — v2 topology: M7/M8 gates driven by INP/INN
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

    # Group mapping for symbolic gm
    gm_group = {"M7":"gm_i","M8":"gm_i", "M9":"gm_o","M10":"gm_o",
                "M12":"gm_x","M13":"gm_x", "M14":"gm_l","M15":"gm_l",
                "M6":"gm_t", "M11":"gm_k"}

    for name, d, g, s in devices:
        p = ss[name]
        if symbolic:
            gm_val = f"{{{gm_group[name]}}}"
            r_val = f"{1.0/p['gds']:.6e}"   # numeric
            cgs_val = f"{p['Cgs']:.6e}" if p['Cgs'] > 0 else None
            cgd_val = f"{p['Cgd']:.6e}" if p['Cgd'] > 0 else None
        else:
            gm_val = f"{p['gm']:.6e}"
            r_val = f"{1.0/p['gds']:.6e}" if p['gds'] > 0 else None
            cgs_val = f"{p['Cgs']:.6e}" if p['Cgs'] > 0 else None
            cgd_val = f"{p['Cgd']:.6e}" if p['Cgd'] > 0 else None

        lines.append(f"G_{name} {d} {s} {g} {s} {gm_val}")
        if r_val is not None:
            lines.append(f"R_{name} {d} {s} {r_val}")
        if g != s and cgs_val is not None:
            lines.append(f"C_{name}_gs {g} {s} {cgs_val}")
        if g != d and cgd_val is not None:
            lines.append(f"C_{name}_gd {g} {d} {cgd_val}")

    lines.append("")
    lines.append("* Load capacitors on VOP/VON")
    cl_val = "5e-12"  # always numeric in targeted-symbolic mode
    lines.append(f"C_LP VOP 0 {cl_val}")
    lines.append(f"C_LN VON 0 {cl_val}")
    lines.append("")

    lines.append("* Bias voltage rails (AC ground)")
    lines.append("V_VDD VDD 0 40")
    lines.append("V_VB  VB  0 9.84")
    lines.append("V_VC  VC  0 16")
    lines.append("")

    lines.append("* AC stimulus: differential drive via VCVS (INN = -INP)")
    lines.append("V_INP INP 0 0")
    lines.append("E_INN INN 0 INP 0 -1.0")
    lines.append("")

    lines.append("* Differential output detector")
    lines.append("E_OUT out 0 VOP VON 1.0")
    lines.append("")

    lines.append(".source V_INP")
    lines.append(".detector V_out")
    lines.append(".end")

    return "\n".join(lines)


# Generate both netlists — write to cir/ (SLiCAP convention)
netlist_num = build_full_netlist(ss, symbolic=False)
netlist_sym = build_full_netlist(ss, symbolic=True)

os.makedirs("cir", exist_ok=True)

with open("cir/afe_full.cir", "w") as f:
    f.write(netlist_num)
print("Numeric netlist → cir/afe_full.cir")

with open("cir/afe_full_symbolic.cir", "w") as f:
    f.write(netlist_sym)
print("Symbolic netlist → cir/afe_full_symbolic.cir")

# ── Step 3: SLiCAP analysis ───────────────────────────────────────────────
print(f"\n{'='*70}")
print("Step 3: SLiCAP Laplace & pole-zero analysis")
print("=" * 70)

import SLiCAP

SLiCAP.initProject("AFE_Full")
os.chdir(PROJECT)

# Load netlist (symbolic or numeric)
cir_file = "afe_full_symbolic.cir" if SYMBOLIC else "afe_full.cir"
circuit = SLiCAP.makeCircuit(cir_file)
print(f"Title: {circuit.title}")
print(f"Nodes: {circuit.nodes}")
print(f"Elements ({len(circuit.elements)}):")
for ref, elem in sorted(circuit.elements.items()):
    print(f"  {ref}: {elem.type} nodes={elem.nodes} params={elem.params}")

# Laplace analysis
mode_label = "symbolic" if SYMBOLIC else "numeric"
print(f"\n--- Laplace ({mode_label}) ---")
lap = SLiCAP.doLaplace(circuit, source='V_INP', detector='V_out', transfer='gain')
H_raw = lap.laplace

# ── Symbolic simplification ────────────────────────────────────────────
from sympy import I, pi, symbols as sp_symbols

subs_dict = {}
if SYMBOLIC:
    print("\n--- Symbolic H(s) from SLiCAP ---")
    # Skip sympy simplify() — the full 10-T 6th-order rational with huge
    # coefficients is too heavy. Use half_circuit.py for design equations.
    H = H_raw  # raw SLiCAP output (already pole-zero cancelled to 2nd order)
    print(f"H(s) length: {len(str(H))} chars")

    s_sym = sp_symbols('s')
    print(f"Free symbols: {sorted([str(x) for x in H.free_symbols if str(x) != 's'])}")

    # DC gain closed-form (substitute s=0 without full simplify — fast)
    H0_sym = H.subs(s_sym, 0)
    print("\n=== DC Gain (symbolic) ===")
    print(f"H(0) = {H0_sym}")

    # Build substitution dict for numeric evaluation
    gm_numeric = {
        "gm_i": ss["M7"]["gm"], "gm_o": ss["M9"]["gm"],
        "gm_x": ss["M12"]["gm"], "gm_l": ss["M14"]["gm"],
        "gm_t": ss["M6"]["gm"], "gm_k": ss["M11"]["gm"],
    }
    for k, v in gm_numeric.items():
        sym = sp_symbols(k)
        if sym in H.free_symbols:
            subs_dict[sym] = v

    H0_num = complex(H0_sym.subs(subs_dict).evalf())
    print(f"H(0) numeric: {abs(H0_num):.6f} ({20*np.log10(abs(H0_num)):.2f} dB)")
else:
    H = H_raw

print(f"\nH(s) length: {len(str(H))} chars")

# Pole-zero
print("\n--- Pole-Zero ---")
pz = SLiCAP.doPZ(circuit, source='V_INP', detector='V_out', transfer='gain')
print(f"Poles: {pz.poles}")
print(f"Zeros: {pz.zeros}")

# Frequency sweep
print("\n--- Frequency response ---")
H_eval = H.subs(subs_dict) if SYMBOLIC else H

for f in [0.01, 1.0, 100.0, 1000.0, 10000.0]:
    try:
        H_jw = complex(H_eval.subs('s', I * 2 * pi * f).evalf())
        mag = abs(H_jw)
        print(f"  f={f:8.2f} Hz: |H|={mag:.6f} ({20*np.log10(mag):.2f} dB)")
    except Exception as e:
        print(f"  f={f:8.2f} Hz: eval error: {e}")
        break

# ── Step 4: Cross-validate ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print("Step 4: Cross-validation")
print("=" * 70)

H0_slicap = abs(complex(H_eval.subs('s', 0).evalf()))
gain_db_slicap = 20 * np.log10(H0_slicap)
print(f"SLiCAP DC gain:  {H0_slicap:.6f} ({gain_db_slicap:.2f} dB)")

gain_db_py = 20 * np.log10(ac_result["gains"][0])
print(f"Python AC gain:  {ac_result['gains'][0]:.6f} ({gain_db_py:.2f} dB)")
print(f"Ratio SLiCAP/Python: {H0_slicap/ac_result['gains'][0]:.4f}")
print("(SLiCAP = diff drive → V_out/V_in; Python = diff gain. Ratio ≈2 expected)")

if pz.poles:
    real_poles = [p for p in pz.poles if abs(np.imag(complex(p))) < 1e-9]
    real_poles_hz = [abs(np.real(complex(p))) / (2*np.pi) for p in real_poles]
    print(f"SLiCAP real poles: {[f'{p:.1f} Hz' for p in sorted(real_poles_hz)]}")

print("\nDone.")
