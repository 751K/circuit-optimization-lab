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
CC = 400e-15
CSENSE = 100e-15
VDD_NOM = 0.90

# A small ratio trim offsets CMFB2's positive systematic error without moving the
# large-signal common-mode trajectory outside its 5 ns window.
RDIV_TOP = 67.2e3
RDIV_BOTTOM = 60e3
RSENSE = 100e3
RSENSE1 = 100e3
RNC = 14.6e3
RCP = 11e3
RCM = 4e3
RZ = 420.0
RDEG2 = 100.0
CSENSE1 = 50e-15
CCMFB1 = 40e-12
RCMFB1 = 5e3
CMILL1 = 1.1e-12
CCMFB2 = 40e-12
RCMFB2 = 200.0

# W/L in um.  The first revision is gm/Id-sized from the local TT model; every
# value is subsequently checked with hierarchical foundry-model operating points.
SZ = {
    "MBN": (6.0, 0.20),
    "MPR": (1.5, 0.20),
    "MPRN": (6.0, 0.20),
    "MPC": (1.5, 0.20),
    # C2: MCND/MCPD are the M3/M4 and M5/M6 density replicas that set the fixed
    # cascode-gate bias (VBNC/VBPC) -- the operating-point CM the gain-boost aux
    # amps regulate around.  Re-matched to the short-L cascodes below.
    "MCND": (4.8, 0.40),
    "MNC": (6.0, 0.20),
    "MCPD": (4.9, 0.30),
    # MREPP mirrors 4 x 20 uA = 80 uA into the M9-replica diode.  C2 narrows MREP
    # (higher density than M9) so VREF1 -> CMFB1 -> O1/O2 CM sits a few tens of mV
    # HIGHER, giving the NMOS cascodes M3/M4 more Vds at fast/hot (ff/125/0.95).
    "MREPP": (6.0, 0.20),
    "MREP": (10.5, 0.20),
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
    # M5/M6 PMOS cascode (source A1/A2 = fold node).  Long L carries the cold-corner
    # intrinsic gain; the regulated-cascode boost keeps ro high hot.  Widths trimmed
    # from the C2 telescopic values so the cascode-gate Cgg stays low enough for the
    # single-transistor booster to reach a ~200 MHz loop UGF (fast settling doublet).
    "M3": (400.0, 0.40),
    "M4": (400.0, 0.40),
    "M5": (400.0, 0.30),
    "M6": (400.0, 0.30),
    # C2b: M7/M8 are now the PMOS FOLD current sources (VDD -> A1/A2), still gated by
    # CTRL1 so CMFB1 steers the fold current to regulate the stage-1 CM.  Moderate L
    # keeps their gm (input-referred noise contributor) in check.
    "M7": (225.0, 0.30),
    "M8": (225.0, 0.30),
    # C2b: bottom NMOS cascode-mirror sinks (NN1/NN2 -> GND, gate = IB).  Long L /
    # low gm -- a fold-branch current-source device is a direct input-referred noise
    # contributor; sized for ~0.25 mA branch current at a low-Vov (weak-inv) point.
    "MFN1": (60.0, 0.50),
    "MFN2": (60.0, 0.50),
    "M9": (200.0, 0.20),
    "M10": (200.0, 0.20),
    "M11": (371.428571, 0.40),
    "M12": (371.428571, 0.40),
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
    "MDL2": (0.4875, 0.30),
    "MDS2": (0.4875, 0.30),
    # ── C2b stage-1 gain-boost: single-transistor regulated-cascode boosters ──
    # (see BOOST_DEVICES for the why-not-DDA rationale).  Each booster is one common-
    # source transistor (source at a rail) + one current-source load mirrored off the
    # existing IB/PB bias.  The CS device W pins the cascode source (VDD-Vsg / Vgs);
    # the load current sets the booster gm -> the boost-loop UGF.
    "MBN1": (15.0, 0.30), "MBN2": (15.0, 0.30),      # PMOS CS booster, gate NN1/NN2
    "MBN1L": (40.0, 0.20), "MBN2L": (40.0, 0.20),    # NMOS load (gate IB, ~135 uA)
    "MBP1": (8.0, 0.30), "MBP2": (8.0, 0.30),        # NMOS CS booster, gate A1/A2
    "MBP1L": (14.0, 0.30), "MBP2L": (14.0, 0.30),    # PMOS load (gate PB, ~120 uA)
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

# main-DUT cascode device -> per-side gate node = the booster output (DC-coupled to
# the mid-rail fold cascode gates, the whole point of folding).  Each booster closes
# a local regulated-cascode loop that raises the cascode branch output impedance.
BOOST_CASCODE_GATE = {"M3": "GBN1", "M4": "GBN2", "M5": "GBP1", "M6": "GBP2"}
# (name, drain, gate, source, kind) for the per-side regulated-cascode boosters,
# injected into every DUT deck by ``_port`` (part of the amplifier, not a TB probe).
#
# C2b design note -- why single-transistor common-source, NOT the C2 quad DDA:
# DC-coupling the DDA output to the cascode gate is INFEASIBLE at 0.85 V.  The DDA
# output device shares the differential-pair tail, which sits at (cascode source +-
# Vgs_sense); the cascode gate it must drive sits at (cascode source +- Vgs_cascode)
# -- so the output device's |Vds| collapses to |Vgs_sense - Vgs_cascode| ~ |Vth_p -
# Vth_n|, a threshold-mismatch knife-edge that goes triode at sf (boostn) and fs
# (boostp) by 30-60 mV (measured).  Widening the cascodes to sink Vgs below the
# opposite threshold needs > 4 pF of gate cap, which kills the boost UGF.
# The fold's intrinsic cascode gain is already ~98 dB (the DDA delivered only ~5 dB
# of loop gain), so the booster does NOT need diff-pair gain -- it only has to be a
# stable regulated-cascode loop that keeps every device saturated.  A single common-
# source transistor whose SOURCE is at a rail does exactly that: its output-device
# saturation only needs the cascode gate inside [Vdsat, VDD-Vdsat] (always true mid-
# rail), with no Vth-mismatch term.  It pins the cascode source to VDD-Vsg (boostn)
# / Vgs (boostp), which TRACKS the NMOS-referenced O1 across corners, so the cascode
# Vds stays put.  NMOS-cascode gates -> PMOS booster from VDD; PMOS-cascode -> NMOS
# booster from GND; loads are IB/PB-mirrored current sources (no new ideal source).
BOOST_DEVICES = [
    # boostn: PMOS common-source boosters for the NMOS cascodes M3/M4.  Gate = the
    # sensed cascode source NN1/NN2, source VDD (rail -> output always saturated),
    # drain = cascode gate GBN1/GBN2; NMOS current-source load gated by the 20 uA IB.
    ("MBN1", "GBN1", "NN1", "VDD", "p"),
    ("MBN2", "GBN2", "NN2", "VDD", "p"),
    ("MBN1L", "GBN1", "IB", "GND", "n"),
    ("MBN2L", "GBN2", "IB", "GND", "n"),
    # boostp: NMOS common-source boosters for the PMOS cascodes M5/M6.  Gate = the
    # sensed fold node A1/A2, source GND, drain = cascode gate GBP1/GBP2; PMOS
    # current-source load gated by the PB mirror node.
    ("MBP1", "GBP1", "A1", "GND", "n"),
    ("MBP2", "GBP2", "A2", "GND", "n"),
    ("MBP1L", "GBP1", "PB", "VDD", "p"),
    ("MBP2L", "GBP2", "PB", "VDD", "p"),
]
BOOST_NODES = ["GBN1", "GBN2", "GBP1", "GBP2"]
# No booster reference dividers: each single-transistor booster self-biases its
# cascode source to its own Vgs (Vth-tracking), so there is nothing to divide.
BOOST_RESISTORS = []
# per-side break nodes for the two boost-loop testbenches (build_boostn/p)
BOOST_BREAK = {
    "boostn": ("M3", "M4", "GBN1", "GBN2"),
    "boostp": ("M5", "M6", "GBP1", "GBP2"),
}

# Post-collapse saturation-checked core devices (consumed by the PVT campaign).
# Every MB* aux transistor and the fold sinks MFN1/MFN2 are region-checked at every
# PVT corner too.
CORE_SAT_DEVICES = ["M0", *[f"M{i}" for i in range(1, 13)],
                    *(name for name, *_ in FOLD_DEVICES),
                    "MBN", *(name for name, *_ in BOOST_DEVICES)]


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
        # C2b folded cascode: A1/A2 are the FOLD nodes (input-pair drain = fold PMOS
        # source drain = PMOS-cascode source), pinned by the boostp NMOS booster to
        # its Vgs (~0.6); NN1/NN2 the bottom NMOS-cascode sources, pinned by the
        # boostn PMOS booster to VDD-Vsg (~0.2); O1/O2 the stage-1 outputs mid-rail.
        "A1": 0.60,
        "A2": 0.60,
        "NN1": 0.20,
        "NN2": 0.20,
        "O1": 0.50,
        "O2": 0.50,
        # C2b single-transistor boosters, DC-coupled: the cascode gates ARE the
        # booster drains (GBN* mid-high ~0.6 for the NMOS cascodes; GBP* mid-low ~0.3
        # for the PMOS cascodes).  No DDA tail/reference/coupling nodes.
        "GBN1": 0.60,
        "GBN2": 0.60,
        "GBP1": 0.30,
        "GBP2": 0.30,
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

    # ── C2b folded cascode + DC-coupled gain-boost: re-map the frozen telescopic
    # stage-1 slots into a folded cascode (M3/M4/M5/M6 sources + M7/M8 drains), point
    # the four boosted cascode gates at the aux outputs (GBN*/GBP*, DC-coupled), and
    # add the fold-mirror sinks + aux devices (all part of the amplifier, every deck).
    for dev in deck["devices"]:
        if dev["name"] in BOOST_CASCODE_GATE:
            dev["gate"] = BOOST_CASCODE_GATE[dev["name"]]
        if dev["name"] in FOLD_REWIRE:
            dev.update(FOLD_REWIRE[dev["name"]])
    for name, drain, gate, source, kind in (*FOLD_DEVICES, *BOOST_DEVICES):
        deck["devices"].append(
            {"name": name, "drain": drain, "gate": gate, "source": source})
        deck["models"][name] = {
            "pdk": "tsmc28hpcp", "model": "nmos" if kind == "n" else "pmos",
            "section": "inherit", "bin": "auto",
        }
    deck.setdefault("resistors", []).extend(
        {**res} for res in BOOST_RESISTORS)
    # The telescopic PMOS-cascode-source nodes B1/B2 are orphaned by the fold.
    deck["solved"] = [n for n in deck["solved"] if n not in FOLD_ORPHAN_NODES]
    for node in (*BOOST_NODES, *FOLD_NEW_NODES):
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
        for node in (*BOOST_NODES, *FOLD_NEW_NODES):
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


def _boost_loop(vdd: float, which: str) -> dict:
    """Stage-1 gain-boost loop-gain testbench (Tian double injection).

    The DUT is the open-loop ``build_ac`` bias configuration (inputs held at the
    on-chip VCMIN through 2 Mohm, CL on each output).  ``Vinj`` breaks the +side
    cascode gate between the aux-amp output (GBN1/GBP1, high-Z) and the cascode
    MOS gate (GBN1G/GBP1G, capacitive) -- exactly the high-Z/high-Z boundary where
    Middlebrook lies, so the campaign probes it with the Tian method.  A unity VCVS
    ``Emir`` mirrors the break anti-phase onto the -side gate so the excitation
    stays in the differential subspace (auto-detected by loop_gain_tian_ngspice)."""
    devp, devn, gp, gn = BOOST_BREAK[which]
    gpg, gng = gp + "G", gn + "G"
    deck = _port(base.build_ac(vdd), vdd)
    for dev in deck["devices"]:
        if dev["name"] == devp:
            dev["gate"] = gpg
        elif dev["name"] == devn:
            dev["gate"] = gng
    for node in (gpg, gng):
        if node not in deck["solved"]:
            deck["solved"].append(node)
    deck["vsources"] = [s for s in deck.get("vsources", []) if s[0] != "Vinj"]
    deck["vsources"].append(["Vinj", gpg, gp, 0.0])
    deck["vcvs"] = [{"name": "Emir", "p": gng, "q": gn,
                     "cp": gp, "cn": gpg, "mu": 1.0}]
    seed = _seed(vdd)
    for guess in deck.get("dc_guesses", []):
        guess[gpg] = guess.get(gp, seed[gp])
        guess[gng] = guess.get(gn, seed[gn])
    deck["name"] = f"tsmc28hpcp_mdac_ota_{which}"
    side = "NMOS" if which == "boostn" else "PMOS"
    deck["description"] = (
        f"Stage-1 {side}-cascode gain-boost loop gain. Vinj breaks the +side "
        f"cascode gate ({gpg}/{gp}); Emir mirrors the -side gate anti-phase. Tian "
        "double injection is mandatory (both break terminals are high-Z). PM must "
        "be > 60 deg and the UGF sits between the main-loop UGF and ~2x it.")
    return deck


def build_boostn(vdd: float = VDD_NOM) -> dict:
    return _boost_loop(vdd, "boostn")


def build_boostp(vdd: float = VDD_NOM) -> dict:
    return _boost_loop(vdd, "boostp")


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
        "tsmc28hpcp_mdac_ota_boostn.json": build_boostn(vdd),
        "tsmc28hpcp_mdac_ota_boostp.json": build_boostp(vdd),
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
