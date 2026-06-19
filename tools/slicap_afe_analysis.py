#!/usr/bin/env python3
"""
SLiCAP symbolic small-signal analysis of the 10-T AFE.
Each PMOS_TFT is replaced by its linearized small-signal equivalent:
  - VCCS (gm) between drain and source, controlled by Vgs
  - Resistor Rds = 1/gds between drain and source
  - Capacitor Cgs between gate and source
  - Capacitor Cgd between gate and drain

Workflow:
  1. Run the Python AFE solver to get DC op + small-signal parameters
  2. Build a SLiCAP netlist from the small-signal equivalents
  3. Run SLiCAP symbolic laplace analysis -> H(s) expression
  4. Extract poles, zeros, bandwidth, gain expressions

Usage:
    /opt/miniconda3/envs/daily/bin/python tools/slicap_afe_analysis.py
"""

import sys, os
import numpy as np

# ── Step 1: Get small-signal parameters from Python AFE solver ──────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.topology import AFE_TOPO
from core.ac_solver import ac_solve
from core.pmos_tft_model import PMOS_TFT

# AFE design
SIZES = {
    "M6": (2264, 78), "M7": (61365, 61), "M8": (61365, 61),
    "M9": (3175, 468), "M10": (3175, 468), "M11": (465, 66),
    "M12": (894, 85), "M13": (894, 85), "M14": (5224, 46), "M15": (5224, 46),
}
BIAS = {"VDD": 40.0, "VCM": 30.65, "VB": 9.84, "VC": 16.0}

print("=" * 70)
print("Step 1: DC operating point & small-signal parameters")
print("=" * 70)

ac_result = ac_solve(SIZES, BIAS, np.array([1.0]), topo=AFE_TOPO)
dc = ac_result["dc_op"]
print(f"DC solved: {len(dc)} nodes")
print(f"AC gain @ 1Hz: {ac_result['gains'][0]:.6f} ({20*np.log10(ac_result['gains'][0]):.2f} dB)")

# Extract small-signal parameters
ss = {}
for name, d, g, s in AFE_TOPO.devices:
    W, L = SIZES[name]
    vd = dc[d] if d in AFE_TOPO.idx else (BIAS[d] if d in BIAS else float(AFE_TOPO.rails[d]))
    vg = dc[g] if g in AFE_TOPO.idx else (BIAS[g] if g in BIAS else float(AFE_TOPO.rails[g]))
    vs = dc[s] if s in AFE_TOPO.idx else (BIAS[s] if s in BIAS else float(AFE_TOPO.rails[s]))
    tft = PMOS_TFT(W=W, L=L)
    osv = tft.get_os(vs, vd, vg)
    rout = osv["rout"]
    gds = 1.0 / rout if rout != 0 and np.isfinite(rout) else 0.0
    ss[name] = {
        "gm": osv["gm"], "gds": gds,
        "Cgs": osv["Cgss"], "Cgd": osv["Cgdd"],
        "Vs": vs, "Vd": vd, "Vg": vg,
    }

# Print summary
for name in sorted(ss.keys()):
    p = ss[name]
    print(f"  {name}: gm={p['gm']:.4e} gds={p['gds']:.4e} Cgs={p['Cgs']:.4e} Cgd={p['Cgd']:.4e}")

# ── Step 2: Build SLiCAP small-signal netlist ──────────────────────────
print(f"\n{'='*70}")
print("Step 2: Build SLiCAP small-signal netlist")
print("=" * 70)

def build_slicap_netlist(ss, symbolic=False):
    """
    Build a SLiCAP-compatible SPICE netlist for the AFE small-signal model.

    Each transistor becomes:
        G_<name>  <drain> <source> <gate> <source>  <gm>
        R_<name>  <drain> <source>  <1/gds>
        C_<name>_gs  <gate> <source>  <Cgs>
        C_<name>_gd  <gate> <drain>  <Cgd>

    If symbolic=True, use symbolic parameter names (e.g. {gm7}) instead of
    numeric values. Otherwise substitute the numeric DC-op values directly.
    """
    def val(key, fmt=".6e"):
        """Return a SPICE value string: either symbolic or numeric."""
        if symbolic:
            return f"{{{key}}}"
        return f"{ss[key]:{fmt}}"

    # Node mapping: use the same names as the Python topology
    # GND = node 0 (SPICE ground)
    # VDD, VB, VC, VCM are AC grounds (DC bias rails)
    lines = []
    lines.append("* AFE Small-Signal Equivalent Circuit")
    lines.append("* Each PMOS_TFT replaced by VCCS(gm) + R(gds) + Cgs + Cgd")
    lines.append("")

    # Parameter definitions for symbolic mode
    if symbolic:
        lines.append("* Symbolic small-signal parameters")
        for name in sorted(ss.keys()):
            p = ss[name]
            lines.append(f".param gm_{name}={p['gm']:.6e}")
            lines.append(f".param gds_{name}={p['gds']:.6e}")
            lines.append(f".param Cgs_{name}={p['Cgs']:.6e}")
            lines.append(f".param Cgd_{name}={p['Cgd']:.6e}")
        lines.append("")

    # Device models
    device_map = {
        # name: (drain, gate, source)
        "M6":  ("NET2", "VB",   "VDD"),
        "M7":  ("VOP",  "VCM",  "NET2"),
        "M8":  ("VON",  "VCM",  "NET2"),
        "M9":  ("GND",  "VFBP", "VOP"),
        "M10": ("GND",  "VFBN", "VON"),
        "M11": ("NET20","VC",   "VDD"),
        "M12": ("VFBN", "VOP",  "NET20"),
        "M13": ("VFBP", "VON",  "NET20"),
        "M14": ("GND",  "GND",  "VFBN"),
        "M15": ("GND",  "GND",  "VFBP"),
    }

    for name, (d, g, s) in device_map.items():
        p = ss[name]
        # Map "GND", "VDD", "VB", "VC", "VCM" to node names
        # SPICE node 0 = GND
        d_node = "0" if d == "GND" else d
        s_node = "0" if s == "GND" else s
        g_node = "0" if g == "GND" else g

        # VCCS: current from drain to source, controlled by V(gate) - V(source)
        lines.append(f"G_{name} {d_node} {s_node} {g_node} {s_node} {p['gm']:.6e}")
        # Output conductance
        if p['gds'] > 0:
            r_val = 1.0 / p['gds']
            lines.append(f"R_{name} {d_node} {s_node} {r_val:.6e}")
        # Gate-source capacitance
        if p['Cgs'] > 0:
            lines.append(f"C_{name}_gs {g_node} {s_node} {p['Cgs']:.6e}")
        # Gate-drain capacitance
        if p['Cgd'] > 0:
            lines.append(f"C_{name}_gd {g_node} {d_node} {p['Cgd']:.6e}")

    lines.append("")
    # Load capacitors on outputs (from the topology)
    lines.append("* Output load capacitors")
    lines.append("CL_VOP VOP 0 5e-12")
    lines.append("CL_VON VON 0 5e-12")
    lines.append("")

    # AC stimulus: drive input pair gates differentially
    # M7 gate (VCM) gets +0.5, M8 gate (VCM) gets -0.5
    # But in small-signal, VCM is an AC ground. We drive through separate sources.
    lines.append("* AC stimulus (differential drive at input pair gates)")
    lines.append("V_DRV_P inp 0 dc 0 ac 0.5")
    lines.append("V_DRV_N inn 0 dc 0 ac -0.5")
    lines.append("")

    # Connect the input drives to the gates
    # M7 gate was VCM (AC ground), now it's driven by inp
    # M8 gate was VCM (AC ground), now it's driven by inn
    # We need to disconnect VCM from the gates and use the drives instead
    # → Re-stamp M7 and M8 VCCS with the driven gates
    lines.append("* NOTE: The VCCS stamps above use VCM as gate for M7,M8.")
    lines.append("* For AC analysis, we override with driven gate voltages.")
    lines.append("* The netlist below has M7_g/M8_g as driven nodes (replacing VCM).")
    lines.append("")

    # Output detector: differential VOP - VON via VCVS
    lines.append("* Differential output (VOP - VON) via VCVS")
    lines.append("E_OUT out 0 VOP VON 1.0")
    lines.append("")

    return "\n".join(lines)


def build_slicap_netlist_v2(ss):
    """
    Version 2: Clean netlist with proper gate drives.

    Uses independent voltage sources to drive the input pair gates,
    replacing the VCM connection for AC analysis.
    All other gate nodes (VB, VC, GND) are AC grounds (0V).
    """
    lines = []
    lines.append("* AFE Small-Signal Equivalent Circuit v2")
    lines.append("* PMOS_TFT → VCCS(gm) + R(gds) + Cgs + Cgd")
    lines.append("* Input: differential drive at INP/INN (M7/M8 gates)")
    lines.append("* Output: differential VOP-VON via E_OUT")
    lines.append("")

    # Device connections: (name, drain, gate, source)
    devices = [
        ("M6",  "NET2",  "VB",   "VDD"),
        ("M7",  "VOP",   "INP",  "NET2"),   # gate driven by INP (was VCM)
        ("M8",  "VON",   "INN",  "NET2"),   # gate driven by INN (was VCM)
        ("M9",  "GND",   "VFBP", "VOP"),
        ("M10", "GND",   "VFBN", "VON"),
        ("M11", "NET20", "VC",   "VDD"),
        ("M12", "VFBN",  "VOP",  "NET20"),
        ("M13", "VFBP",  "VON",  "NET20"),
        ("M14", "GND",   "GND",  "VFBN"),
        ("M15", "GND",   "GND",  "VFBP"),
    ]

    for name, d, g, s in devices:
        p = ss[name]
        d_node = "0" if d == "GND" else d
        s_node = "0" if s == "GND" else s
        g_node = "0" if g == "GND" else g

        lines.append(f"* {name}: d={d} g={g} s={s}")
        # VCCS: i_d = gm * (Vg - Vs), current flows d → s
        lines.append(f"G_{name} {d_node} {s_node} {g_node} {s_node} {p['gm']:.6e}")
        if p['gds'] > 0:
            lines.append(f"R_{name} {d_node} {s_node} {1.0/p['gds']:.6e}")
        if p['Cgs'] > 0:
            lines.append(f"C_{name}_gs {g_node} {s_node} {p['Cgs']:.6e}")
        if p['Cgd'] > 0:
            lines.append(f"C_{name}_gd {g_node} {d_node} {p['Cgd']:.6e}")

    lines.append("")
    lines.append("* Load capacitors on VOP/VON")
    lines.append("C_LP VOP 0 5e-12")
    lines.append("C_LN VON 0 5e-12")
    lines.append("")

    lines.append("* AC stimulus at input gates")
    lines.append("V_INP INP 0 dc 0 ac 0.5")
    lines.append("V_INN INN 0 dc 0 ac -0.5")
    lines.append("")

    lines.append("* Ideal bias voltages at AC ground")
    lines.append("V_VDD VDD 0 dc 40")
    lines.append("V_VB  VB  0 dc 9.84")
    lines.append("V_VC  VC  0 dc 16")
    lines.append("")

    lines.append("* Differential output detector")
    lines.append("E_OUT out 0 VOP VON 1.0")
    lines.append("")

    return "\n".join(lines)

# Build the netlist
netlist = build_slicap_netlist_v2(ss)
print("Netlist:")
print(netlist[:2000])
print("...")

# ── Step 3: Run SLiCAP analysis ────────────────────────────────────────
print(f"\n{'='*70}")
print("Step 3: SLiCAP analysis")
print("=" * 70)

import SLiCAP

# Initialize project (required by SLiCAP)
SLiCAP.initProject("AFE_Symbolic")
# initProject changes cwd to the project dir; switch back
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) or ".")

# Create circuit from netlist string
circuit = SLiCAP.makeCircuit(netlist)
print(f"Circuit title: {circuit.title}")
print(f"Nodes: {circuit.nodes}")
print(f"Elements ({len(circuit.elements)}):")
for ref, elem in sorted(circuit.elements.items()):
    print(f"  {ref}: type={elem.type} nodes={elem.nodes} params={elem.params}")

# ── Run Laplace (symbolic AC) analysis ──────────────────────────────────
print(f"\n--- Laplace analysis ---")
result = SLiCAP.laplace(circuit, gain='vi', source='V_INP', detector='out')
print(f"Transfer function H(s) = Vout / I_in:")

# Simplify the expression
from sympy import simplify, expand, factor, collect

try:
    H = simplify(result.laplace)
    print(f"H(s) = {H}")
except Exception as e:
    print(f"  raw: {result.laplace}")
    print(f"  simplify error: {e}")

# ── Pole-zero analysis ──────────────────────────────────────────────────
print(f"\n--- Pole-Zero analysis ---")
try:
    from SLiCAP import doPZ
    pz_result = doPZ(circuit, gain='vi', source='V_INP', detector='out')
    print(f"Poles: {pz_result.poles}")
    print(f"Zeros: {pz_result.zeros}")
except Exception as e:
    print(f"PZ error: {e}")

# ── Numeric evaluation at key frequencies ───────────────────────────────
print(f"\n--- Numeric evaluation ---")
from sympy import I, pi, N

s = result.laplace.free_symbols
print(f"Free symbols in H(s): {s}")

# Substitute s = j*2*pi*f for frequency evaluation
for f in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    try:
        w = 2 * pi * f
        # H(s) with s = jw
        H_jw = complex(result.laplace.subs('s', I * w).evalf())
        mag = abs(H_jw)
        phase = np.angle(H_jw, deg=True)
        print(f"  f={f:8.2f} Hz: |H|={mag:.6f} ({20*np.log10(mag):.2f} dB), phase={phase:.1f}°")
    except Exception as e:
        print(f"  f={f:8.2f} Hz: eval error: {e}")
        break

print("\nDone.")
