#!/usr/bin/env python3
"""TSMC28HPC+ 14-bit pipeline-ADC first-stage MDAC OTA testbenches.

The switched-capacitor and loop-probe topology is shared with ``mdac_ota_gen``;
all process-dependent geometry, bias, compensation, model bindings, and seeds
are replaced here.  Generated JSON files are complete standalone netlists.
"""
from __future__ import annotations

import json
import os

import mdac_ota_gen as base
import numpy as np


CS = 2.6e-12
CF = CS / 8.0
CL = 500e-15
CC = 840e-15
CSENSE = 100e-15
VDD_NOM = 0.90

# A small ratio trim offsets CMFB2's positive systematic error without moving the
# large-signal common-mode trajectory outside its 5 ns window.
RDIV_TOP = 67.2e3
RDIV_BOTTOM = 71.2e3
RSENSE = 100e3
RSENSE1 = 100e3
# iter5 wide-swing legs: NN1 ~ 20uA*RNC (density-matched MCND), A1 ~ VDD-20uA*RCP
# (density-matched MCPD).  Targets: NN1 ~0.17 (MFN vdsat floor ~0.09, M3 window to
# O1=VREF1 which bottoms at ~0.40 hot/fast), A1 ~ VDD-0.17 (M7 vdsat headroom).
RNC = 8.5e3
RCP = 8.5e3
RCM = 4e3
RZ = 400.0
RDEG2 = 100.0
CSENSE1 = 50e-15
CCMFB1 = 40e-12
RCMFB1 = 5e3
CMILL1 = 10e-12
CCMFB2 = 2e-12
RCMFB2 = 200.0
# iter6b capacitive CM feedforward.  One cap from each output to CTRL2: the
# common-mode component couples into the M11/M12 gate immediately (CM sags ->
# CTRL2 sags -> the PMOS loads source more -> CM recovers) while the
# differential components cancel at CTRL2 by symmetry.  This is the ONLY
# mechanism fast enough for the class-A stage's large-signal CM shift inside a
# 5 ns hold window -- the CMFB2 loop itself crosses near 1 MHz because its
# 2 uA diode drives an 11 pF gate.  Measured droop at the 5 ns checkpoint:
# -82 mV with no feedforward, -38 mV at 2.2 pF, -28 mV at 4 pF (saturating).
# CCMFB2 shrinks to 2 pF so the feedforward divider CFF/(CFF+CCMFB2+Cgate)
# bites; CFF then also supplies CMFB2's phase lead.
CFF = 3e-12

# W/L in um.  The first revision is gm/Id-sized from the local TT model; every
# value is subsequently checked with hierarchical foundry-model operating points.
SZ = {
    "MBN": (6.0, 0.20),
    "MPR": (1.5, 0.20),
    "MPRN": (6.0, 0.20),
    "MPC": (1.5, 0.20),
    # C2b iter5: MCND/MCPD are the M3/M4 and M5/M6 density replicas whose wide-swing
    # legs (VBNC = Vgs(MCND)+I*RNC, VBPC = VDD-I*RCP-|Vgs(MCPD)|) bias the folded
    # cascode gates DIRECTLY (no aux loops).  Near-density matching (~2.2x the main
    # cascodes' current density) keeps the Vgs replica error to tens of mV, so
    # NN1 ~ I*RNC (GND-referenced, tracking M3's own Vth family) and
    # A1 ~ VDD - I*RCP (VDD-referenced, matching M7's VDD-referenced source).
    # The C2 values (4.8/4.9 um at 22x density) put NN1 ~0.35-0.49 V and collapsed
    # M3 at ff/125/0.95 -- measured; the mismatch term was half the level shift.
    "MCND": (48.0, 0.40),
    "MNC": (6.0, 0.20),
    "MCPD": (48.0, 0.30),
    # MREPP mirrors 4 x 20 uA = 80 uA into the M9-replica diode.  C2 narrows MREP
    # (higher density than M9) so VREF1 -> CMFB1 -> O1/O2 CM sits a few tens of mV
    # HIGHER, giving the NMOS cascodes M3/M4 more Vds at fast/hot (ff/125/0.95).
    "MREPP": (6.0, 0.20),
    "MREP": (7.875, 0.20),
    "MCMP": (3.0, 0.20),
    "MCMD": (5.0, 0.20),
    # C2 iter7: full C1 tail restored (noise needs gm1; the AC-coupled aux no
    # longer trades headroom against current).
    "M0": (300.0, 0.20),
    # C2: input pair short-L (0.35 -> 0.18) drops Cgg(M1) ~1.37 -> ~0.55 pF so the
    # noise gain NG_eff = (Cs+Cf+Cgg)/Cf falls 13.2 -> ~10.6; W held for gm1.
    "M1": (260.0, 0.18),
    "M2": (260.0, 0.18),
    # C2b folded cascode: M3/M4 NMOS cascode (source NN1/NN2 -> own bottom branch),
    # M5/M6 PMOS cascode (source A1/A2 = fold node).  Long L carries the intrinsic
    # gain; iter5 drops the regulated-cascode aux loops entirely -- measured, the
    # single-transistor boosters pin NN1 to VDD-|Vgs_p| (supply-referenced) while
    # O1 is GND-referenced (VREF1 = Vgs replica), and across +-5% supply and the
    # mixed corners the N-to-P static window pinches to 0.10 V (ff/125/0.95).
    "M3": (400.0, 0.40),
    "M4": (400.0, 0.40),
    "M5": (400.0, 0.30),
    "M6": (400.0, 0.30),
    # C2b: M7/M8 are the PMOS FOLD current sources (VDD -> A1/A2), gated by CTRL1 so
    # CMFB1 steers the fold current to regulate the stage-1 CM.  L is pinned by the
    # CMFB1 actuator range: lengthening 0.30 -> 0.45 at fixed W needs |Vgs| ~0.60,
    # i.e. CTRL1 ~0.30 -- BELOW the CMFB1 diode's measured output floor (~0.334), so
    # the loop rails, M7 starves the fold node and M5 cuts off (measured full
    # collapse, iter5a).  Any ro7 lever must co-scale W with L (constant density).
    "M7": (225.0, 0.30),
    "M8": (225.0, 0.30),
    # C2b: bottom NMOS cascode-mirror sinks (NN1/NN2 -> GND, gate = IB).  Long L /
    # low gm -- a fold-branch current-source device is a direct input-referred noise
    # contributor; ~57 uA branch at a low-Vov point.
    "MFN1": (60.0, 0.50),
    "MFN2": (60.0, 0.50),
    "M9": (150.0, 0.20),
    "M10": (150.0, 0.20),
    "M11": (139.285714, 0.30),
    "M12": (139.285714, 0.30),
    "MS1": (10.0, 0.20),
    "MS2": (10.0, 0.20),
    "MS3": (10.0, 0.20),
    "MS4": (10.0, 0.20),
    "MT1": (9.3, 0.20),
    "MT2": (3.1, 0.20),
    "MDL1": (2.25, 0.30),
    "MDS1": (2.25, 0.30),
    "MRA": (40.0, 0.20),
    "MRB": (40.0, 0.20),
    "MTB": (2.325, 0.20),
    "MDL2": (0.55, 0.30),
    "MDS2": (0.55, 0.30),
}

# Parallel-instance multiplicity (SPICE ``m=``): one drawn macro instance, M
# identical copies in parallel. Same electrical result as the former explicit
# clones (M0B/M0C, M9B/M10B, M11B/C, M12B/C) at 1/M the per-deck hsa expansion
# cost (~2.9 s per foundry macro instance).  Layout-real: each copy stays within
# the wrapper's characterized per-instance finger geometry.
MULT = {"M0": 3, "M9": 2, "M10": 2, "M11": 3, "M12": 3}

# ── C2b folded-cascode stage-1 rewiring ─────────────────────────────────────────
# The frozen base (mdac_ota_gen) ships a TELESCOPIC stage-1: one stacked branch
# M0(tail)->M1(in)->M3(NMOS casc)->M5(PMOS casc)->M7(PMOS load) per side, with the
# NMOS cascode gate pinned ~70 mV below VDD at the 0.85 V rail (no aux headroom --
# the two measured telescopic-boost dead ends).  ``_port`` re-maps the SAME device
# slots into a FOLDED cascode so every cascode gate sits mid-rail:
#   * M1/M2 (NMOS input) drains stay A1/A2, now the FOLD nodes.
#   * M7/M8 (PMOS) become the fold current sources VDD->A1/A2 (gate still CTRL1, so
#     CMFB1 keeps regulating the stage-1 CM by steering the fold current -- same
#     loop role as the old top load).
#   * M5/M6 (PMOS) cascode A1/A2 -> O1/O2 (boosted gate GBP1/GBP2, source = A1/A2).
#   * M3/M4 (NMOS) cascode O1/O2 -> NN1/NN2 (boosted gate GBN1/GBN2, source moves
#     from the old A1/A2 to the new bottom-branch nodes NN1/NN2).
#   * MFN1/MFN2 (new NMOS) mirror-sink NN1/NN2 -> GND (long-L, low-gm: fold-branch
#     current sources are input-referred noise contributors; gate = 20 uA IB ref).
# The four boosted cascode gates are now the aux OUTPUTS directly (DC-coupled) --
# the AC-coupling caps/gate resistors of the C2 telescopic tip are gone.
FOLD_REWIRE = {
    "M3": {"source": "NN1"},   # NMOS cascode: source A1 -> NN1 (own bottom branch)
    "M4": {"source": "NN2"},
    "M5": {"source": "A1"},     # PMOS cascode: source B1 -> A1 (the fold node)
    "M6": {"source": "A2"},
    "M7": {"drain": "A1"},      # PMOS fold current source: drain B1 -> A1
    "M8": {"drain": "A2"},
}
# new bottom NMOS cascode-mirror sinks (gate = the single 20 uA IB reference)
FOLD_DEVICES = [
    ("MFN1", "NN1", "IB", "GND", "n"),
    ("MFN2", "NN2", "IB", "GND", "n"),
]
FOLD_NEW_NODES = ["NN1", "NN2"]
FOLD_ORPHAN_NODES = ["B1", "B2"]   # telescopic PMOS-casc-source nodes, now unused

# C2b design record -- why NO gain-boost aux loops (iter5 decision, all measured):
# * quad DDA (C2): DC-coupling its output to a telescopic cascode gate collapses the
#   output device to |Vth_p - Vth_n| (triode at sf/fs by 30-60 mV).
# * single-transistor regulated cascode (iter2-iter3): the PMOS-from-VDD booster
#   pins NN1 = VDD - |Vgs_p| (supply-referenced) while O1 = VREF1 = Vgs(MREP) is
#   GND-referenced; across +-5% supply and mixed corners the whole N-to-P static
#   window (A1 - NN1) pinches to 0.10 V at ff/125/0.95 (M3 vds 2.7 mV) and any
#   re-centering pushes NN1 under the MFN vdsat floor at ss/-40/0.85.  Boost gain
#   (+7-19 dB where alive) is not worth an unfixable bias reference.
# * iter5: wide-swing replica legs bias the cascode gates directly (VBNC/VBPC with
#   near-density-matched MCND/MCPD), and the slow-corner intrinsic gain is recovered
#   with longer fold sources (M7/M8 L=0.45) + longer fold sinks (MFN L=0.60).

# Post-collapse saturation-checked core devices (consumed by the PVT campaign).
# The fold sinks MFN1/MFN2 and the Iref diode MBN are region-checked at every PVT
# corner too.
CORE_SAT_DEVICES = ["M0", *[f"M{i}" for i in range(1, 13)],
                    *(name for name, *_ in FOLD_DEVICES),
                    "MBN"]


def _seed(vdd: float) -> dict[str, float]:
    h = vdd / 2.0
    vcm_ref = vdd * RDIV_BOTTOM / (RDIV_TOP + RDIV_BOTTOM)
    return {
        "IB": 0.55,
        "NBC": 0.27,
        "VBNC": 0.77,
        "PB": vdd - 0.57,
        "PX": vdd - 0.20,
        "VBPC": vdd - 0.70,
        "VCM": vcm_ref,
        "VREF1": 0.55,
        "VCMIN": 0.60,
        "CMR": 0.16,
        "TAIL": 0.15,
        # C2b folded cascode (iter5, replica-biased): A1/A2 are the FOLD nodes
        # (input-pair drain = fold PMOS drain = PMOS-cascode source) at
        # ~VDD - 20uA*RCP; NN1/NN2 the bottom NMOS-cascode sources at ~20uA*RNC;
        # O1/O2 the stage-1 outputs at VREF1.
        "A1": vdd - 0.17,
        "A2": vdd - 0.17,
        "NN1": 0.17,
        "NN2": 0.17,
        "O1": 0.50,
        "O2": 0.50,
        "OUTP": h,
        "OUTN": h,
        "MZ1": 0.55,
        "MZ2": 0.55,
        "CTRL1": vdd - 0.57,
        "CMPC1": vdd,
        "CMT1": 0.08,
        "CMT2": 0.08,
        "NSNS1": vdd - 0.55,
        "CMS1": 0.51,
        "CTRL2": vdd - 0.55,
        "CMPC2": vdd,
        "CMB": 0.06,
        "CMRA": 0.07,
        "CMRB": 0.07,
        "CMS": h,
        "NSNS2": vdd - 0.55,
    }


def _nf(width_um: float) -> int:
    """Use roughly 1 um fingers while keeping foundry macro size bounded."""
    return max(1, min(200, int(round(width_um))))


def _port(deck: dict, vdd: float) -> dict:
    deck["name"] = deck["name"].replace("freepdk45", "tsmc28hpcp")
    deck["description"] = deck.get("description", "").replace(
        "FreePDK45", "TSMC28HPC+")
    deck["description"] = deck["description"].replace(
        "docs/mdac_ota_derivation.md", "docs/tsmc28_mdac_ota_design.md")
    deck["bias"]["VDD"] = vdd

    removed = {"MS3", "MS4", "MT2", "MRZ1", "MRZ2"}
    deck["devices"] = [dev for dev in deck["devices"] if dev["name"] not in removed]
    for name in removed:
        deck["models"].pop(name, None)
    deck["solved"] = [node for node in deck["solved"] if node != "CMT2"]

    # ── C2b folded cascode (iter5): re-map the frozen telescopic stage-1 slots into
    # a folded cascode (M3/M4/M5/M6 sources + M7/M8 drains) and add the fold-mirror
    # sinks.  The cascode gates stay on the wide-swing replica legs VBNC/VBPC (the
    # base telescopic wiring), so no gate re-pointing is needed.
    for dev in deck["devices"]:
        if dev["name"] in FOLD_REWIRE:
            dev.update(FOLD_REWIRE[dev["name"]])
    for name, drain, gate, source, kind in FOLD_DEVICES:
        deck["devices"].append(
            {"name": name, "drain": drain, "gate": gate, "source": source})
        deck["models"][name] = {
            "pdk": "tsmc28hpcp", "model": "nmos" if kind == "n" else "pmos",
            "section": "inherit", "bin": "auto",
        }
    # The telescopic PMOS-cascode-source nodes B1/B2 are orphaned by the fold.
    deck["solved"] = [n for n in deck["solved"] if n not in FOLD_ORPHAN_NODES]
    for node in FOLD_NEW_NODES:
        if node not in deck["solved"]:
            deck["solved"].append(node)

    # The high-current mirrors need more total width than one characterized macro
    # allows: tail 3 x 300 um, second-stage NMOS 2 x 300 um, PMOS loads
    # 3 x 371.43 um per side. The MULT table renders each as ONE instance with
    # the SPICE ``m=`` multiplicity - electrically identical to the former
    # explicit clones at a third of the per-instance hsa expansion cost.
    for dev in deck["devices"]:
        width, length = SZ[dev["name"]]
        dev["W"] = width
        dev["L"] = length
        dev["NF"] = _nf(width)
        if dev["name"] in MULT:
            dev["M"] = MULT[dev["name"]]
        if dev["name"] == "MRA":
            dev["source"] = "CMRA"
        elif dev["name"] == "MRB":
            dev["source"] = "CMRB"
        if dev["name"] == "MS1":
            dev["gate"] = "CMS1"

    for node_name in ("CMS1", "CMPC1", "CMPC2", "CMRA", "CMRB"):
        if node_name not in deck["solved"]:
            deck["solved"].append(node_name)
    deck.setdefault("resistors", []).extend([
        {"name": "RSCM1P", "a": "O1", "b": "CMS1", "R": RSENSE1},
        {"name": "RSCM1N", "a": "O2", "b": "CMS1", "R": RSENSE1},
        {"name": "RCMFB1", "a": "CMPC1", "b": "VDD", "R": RCMFB1},
        {"name": "RDEG2A", "a": "CMRA", "b": "CMB", "R": RDEG2},
        {"name": "RDEG2B", "a": "CMRB", "b": "CMB", "R": RDEG2},
        {"name": "RCMFB2", "a": "CMPC2", "b": "VDD", "R": RCMFB2},
        {"name": "RZ1", "a": "MZ1", "b": "O1", "R": RZ},
        {"name": "RZ2", "a": "MZ2", "b": "O2", "R": RZ},
    ])
    deck.setdefault("capacitors", []).extend([
        {"name": "CFCM1P", "a": "O1", "b": "CMS1", "C": CSENSE1},
        {"name": "CFCM1N", "a": "O2", "b": "CMS1", "C": CSENSE1},
        {"name": "CCMFB1", "a": "CTRL1", "b": "CMPC1", "C": CCMFB1},
        {"name": "CMILL1", "a": "CTRL1", "b": "CMS1", "C": CMILL1},
        {"name": "CCMFB2", "a": "CTRL2", "b": "CMPC2", "C": CCMFB2},
        {"name": "CFF1", "a": "OUTP", "b": "CTRL2", "C": CFF},
        {"name": "CFF2", "a": "OUTN", "b": "CTRL2", "C": CFF},
    ])

    for model in deck["models"].values():
        polarity = model["model"]
        model["pdk"] = "tsmc28hpcp"
        if polarity == "pmos":
            model["vb"] = vdd

    cap_values = {
        "CC1": CC, "CC2": CC,
        "CFS1": CSENSE, "CFS2": CSENSE,
        "CL1": CL, "CL2": CL,
        "CF1": CF, "CF2": CF,
        "CS1": CS, "CS2": CS,
    }
    for cap in deck.get("capacitors", []):
        if cap["name"] in cap_values:
            cap["C"] = cap_values[cap["name"]]
    deck["capacitors"] = [
        cap for cap in deck.get("capacitors", [])
        if cap["name"] not in {"COUT1", "COUT2"}
    ]

    res_values = {
        "RVCM1": RDIV_TOP, "RVCM2": RDIV_BOTTOM,
        "RS1": RSENSE, "RS2": RSENSE,
        "RNC": RNC, "RCP": RCP, "RCM": RCM,
    }
    for resistor in deck.get("resistors", []):
        if resistor["name"] in res_values:
            resistor["R"] = res_values[resistor["name"]]

    seed = _seed(vdd)
    for guess in deck.get("dc_guesses", []):
        guess.setdefault("CMS1", seed["CMS1"])
        guess.setdefault("CMPC1", seed["CMPC1"])
        guess.setdefault("CMPC2", seed["CMPC2"])
        guess.setdefault("CMRA", seed["CMRA"])
        guess.setdefault("CMRB", seed["CMRB"])
        for node in FOLD_NEW_NODES:
            guess.setdefault(node, seed[node])
        for node in FOLD_ORPHAN_NODES:
            guess.pop(node, None)
        for node in list(guess):
            if node in seed:
                guess[node] = seed[node]
        if "CTRL1G" in guess:
            guess["CTRL1G"] = seed["CTRL1"]
        if "CMSG" in guess:
            guess["CMSG"] = seed["CMS"]
        if "INP" in guess:
            guess["INP"] = seed["VCMIN"]
        if "INN" in guess:
            guess["INN"] = seed["VCMIN"]
        if "INPD" in guess:
            guess["INPD"] = seed["VCMIN"]
        if "INND" in guess:
            guess["INND"] = seed["VCMIN"]
        if "BP1" in guess:
            guess["BP1"] = vdd / 2.0
        if "BP2" in guess:
            guess["BP2"] = vdd / 2.0
    return deck


def build_ac(vdd: float = VDD_NOM) -> dict:
    return _port(base.build_ac(vdd), vdd)


def build_noise(vdd: float = VDD_NOM) -> dict:
    deck = _port(base.build_noise(vdd), vdd)
    deck["description"] = (
        "Closed-loop hold-phase noise at v(OUTP,OUTN). The ADC Nyquist-band "
        "sign-off integral is 10-50 MHz; the same 10 MHz-20 GHz PSD is retained "
        "for a separately reported wideband stress value."
    )
    deck["analyses"]["noise"] = {
        "freqs": {"start": 1e7, "stop": 2e10, "num": 81, "scale": "log"},
        "band": [1e7, 2e10],
    }
    return deck


def build_dmloop(vdd: float = VDD_NOM) -> dict:
    return _port(base.build_dmloop(vdd), vdd)


def build_cmfb1(vdd: float = VDD_NOM) -> dict:
    deck = _port(base.build_cmfb1(vdd), vdd)
    for dev in deck["devices"]:
        if dev["name"] in {"M7", "M8"}:
            dev["gate"] = "CTRL1"
        elif dev["name"] == "MS1":
            dev["gate"] = "CMS1G"
    deck["solved"] = [node for node in deck["solved"] if node != "CTRL1G"]
    if "CMS1G" not in deck["solved"]:
        deck["solved"].append("CMS1G")
    deck["vsources"] = [source for source in deck.get("vsources", [])
                         if source[0] != "Vinj"]
    deck["vsources"].append(["Vinj", "CMS1G", "CMS1", 0.0])
    for guess in deck.get("dc_guesses", []):
        guess.pop("CTRL1G", None)
        guess["CMS1G"] = guess["CMS1"]
    deck["description"] = (
        "CMFB1 loop gain. O1/O2 are resistor-averaged at CMS1; Vinj breaks the "
        "high-impedance MS1 sense gate (CMS1G/CMS1). CMILL1 stays on the physical "
        "CMS1 node so the probed loop matches the closed-loop compensation exactly."
    )
    return deck


def build_cmfb2(vdd: float = VDD_NOM) -> dict:
    return _port(base.build_cmfb2(vdd), vdd)


def build_transient(vdd: float = VDD_NOM) -> dict:
    deck = _port(base.build_transient(vdd), vdd)
    for resistor in deck["resistors"]:
        if resistor["name"] == "RDC1":
            resistor["b"] = "RDCP"
        elif resistor["name"] == "RDC2":
            resistor["b"] = "RDCN"
    switches = (
        ("MSWPN", "RDCP", "DCH", "VCMIN", "nmos", 0.1),
        ("MSWPP", "RDCP", "DCHB", "VCMIN", "pmos", 0.2),
        ("MSWNN", "RDCN", "DCH", "VCMIN", "nmos", 0.1),
        ("MSWNP", "RDCN", "DCHB", "VCMIN", "pmos", 0.2),
    )
    for name, drain, gate, source, polarity, width in switches:
        deck["devices"].append({
            "name": name, "drain": drain, "gate": gate, "source": source,
            "W": width, "L": 0.05, "NF": _nf(width),
        })
        model = {
            "pdk": "tsmc28hpcp", "model": polarity,
            "section": "inherit", "bin": "auto",
        }
        if polarity == "pmos":
            model["vb"] = vdd
        deck["models"][name] = model
    deck["solved"].extend(["RDCP", "RDCN", "DCH", "DCHB"])
    deck["resistors"].extend([
        {"name": "RFLOATP", "a": "RDCP", "b": "VCMIN", "R": 1e12},
        {"name": "RFLOATN", "a": "RDCN", "b": "VCMIN", "R": 1e12},
    ])
    deck.setdefault("vsources", []).extend([
        ["VDCH", "DCH", "GND", "DCH"],
        ["VDCHB", "DCHB", "GND", "DCHB"],
    ])
    for guess in deck["dc_guesses"]:
        vcm_in = guess["VCMIN"]
        guess.update({"RDCP": vcm_in, "RDCN": vcm_in,
                      "DCH": vdd, "DCHB": 0.0})
    deck["description"] = (
        "Closed-loop hold-phase residue transient. The initial OP sees the 2 Mohm "
        "DC helpers through closed transmission gates; the hold edge opens their "
        "far ends so they cannot leak sampled CDAC charge."
    )
    return deck


def hold_clock_inputs(tgrid, vdd: float = VDD_NOM):
    t = np.asarray(tgrid, float)
    dch = np.zeros_like(t)
    dchb = np.full_like(t, vdd)
    dch[0] = vdd
    dchb[0] = 0.0
    return {"DCH": dch, "DCHB": dchb}


def build_code_transition(vdd: float = VDD_NOM) -> dict:
    """Split-CDAC 0111 -> 1000 major-carry transition testbench."""
    deck = build_transient(vdd)
    deck["name"] = "tsmc28hpcp_mdac_ota_code_transition"
    deck["description"] = (
        "Worst 4-bit CDAC major-carry transition (0111 to 1000). Each side uses "
        "8:4:2:1 binary capacitors plus one dummy unit; all complementary bit edges "
        "switch synchronously. The final differential weighted "
        "bottom-plate step is FS/16 and the ideal residue is -0.45 V differential."
    )
    deck["capacitors"] = [cap for cap in deck["capacitors"]
                           if cap["name"] not in {"CS1", "CS2"}]
    deck["vsources"] = [source for source in deck["vsources"]
                         if source[0] not in {"VBP1", "VBP2"}]
    deck["solved"] = [node for node in deck["solved"] if node not in {"BP1", "BP2"}]

    unit = CS / 16.0
    weights = (8, 4, 2, 1)
    h = vdd / 2.0
    initial_codes = {"P": 0b0111, "N": 0b1000}
    for side, top in (("P", "INP"), ("N", "INN")):
        code = initial_codes[side]
        for bit, weight in zip((3, 2, 1, 0), weights):
            node_name = f"BP{side}{bit}"
            key = f"bp{side.lower()}{bit}"
            deck["solved"].append(node_name)
            deck["capacitors"].append(
                {"name": f"CS{side}{bit}", "a": top, "b": node_name,
                 "C": weight * unit})
            deck["vsources"].append([f"VBP{side}{bit}", node_name, "GND", key])
            is_high = bool(code & (1 << bit))
            deck["dc_guesses"][0][node_name] = h + (0.225 if is_high else -0.225)
        dummy = f"BP{side}D"
        deck["solved"].append(dummy)
        deck["capacitors"].append(
            {"name": f"CS{side}D", "a": top, "b": dummy, "C": unit})
        deck["vsources"].append([f"VBP{side}D", dummy, "GND", h])
        deck["dc_guesses"][0][dummy] = h
    return deck


def code_transition_inputs(tgrid, vdd: float = VDD_NOM, edge_time: float = 20e-12):
    """Waveforms for the synchronous complementary 0111 -> 1000 transition."""
    t = np.asarray(tgrid, float)
    h = vdd / 2.0
    lo, hi = h - 0.225, h + 0.225
    initial = {"P": 0b0111, "N": 0b1000}
    final = {"P": 0b1000, "N": 0b0111}
    waveforms = hold_clock_inputs(t, vdd)
    for side in ("P", "N"):
        for bit in (3, 2, 1, 0):
            v0 = hi if initial[side] & (1 << bit) else lo
            v1 = hi if final[side] & (1 << bit) else lo
            waveforms[f"bp{side.lower()}{bit}"] = np.where(t < edge_time, v0, v1)
    return waveforms


def all_testbenches(vdd: float = VDD_NOM) -> dict[str, dict]:
    return {
        "tsmc28hpcp_mdac_ota.json": build_transient(vdd),
        "tsmc28hpcp_mdac_ota_ac.json": build_ac(vdd),
        "tsmc28hpcp_mdac_ota_dmloop.json": build_dmloop(vdd),
        "tsmc28hpcp_mdac_ota_cmfb1.json": build_cmfb1(vdd),
        "tsmc28hpcp_mdac_ota_cmfb2.json": build_cmfb2(vdd),
        "tsmc28hpcp_mdac_ota_noise.json": build_noise(vdd),
        "tsmc28hpcp_mdac_ota_code_transition.json": build_code_transition(vdd),
    }


def write_all(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for filename, deck in all_testbenches().items():
        path = os.path.join(outdir, filename)
        with open(path, "w", encoding="ascii") as handle:
            json.dump(deck, handle, indent=2)
        print("wrote", path)


if __name__ == "__main__":
    write_all(os.path.dirname(os.path.abspath(__file__)))
