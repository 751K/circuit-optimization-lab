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
RNC = 34e3
RCP = 10e3
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
    "MCND": (3.15, 0.30),
    "MNC": (6.0, 0.20),
    "MCPD": (4.35, 0.30),
    # MREPP mirrors 4 x 20 uA = 80 uA into the M9-replica diode; matching M9's
    # 7 uA/um density at the same L makes VREF1 (-> CMFB1 -> O1/O2 CM) track the
    # stage-2 current across corners: W = 80 uA / 7 uA/um = 11.43 um.
    "MREPP": (6.0, 0.20),
    "MREP": (11.43, 0.20),
    "MCMP": (3.0, 0.20),
    "MCMD": (5.0, 0.20),
    "M0": (300.0, 0.20),
    "M1": (315.0, 0.35),
    "M2": (315.0, 0.35),
    "M3": (290.0, 0.40),
    "M4": (290.0, 0.40),
    "M5": (300.0, 0.30),
    "M6": (300.0, 0.30),
    "M7": (225.0, 0.30),
    "M8": (225.0, 0.30),
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
}

# Parallel-instance multiplicity (SPICE ``m=``): one drawn macro instance, M
# identical copies in parallel. Same electrical result as the former explicit
# clones (M0B/M0C, M9B/M10B, M11B/C, M12B/C) at 1/M the per-deck hsa expansion
# cost (~2.9 s per foundry macro instance).  Layout-real: each copy stays within
# the wrapper's characterized per-instance finger geometry.
MULT = {"M0": 3, "M9": 2, "M10": 2, "M11": 3, "M12": 3}

# Post-collapse saturation-checked core devices (consumed by the PVT campaign).
CORE_SAT_DEVICES = ["M0", *[f"M{i}" for i in range(1, 13)]]


def _seed(vdd: float) -> dict[str, float]:
    h = vdd / 2.0
    vcm_ref = vdd * RDIV_BOTTOM / (RDIV_TOP + RDIV_BOTTOM)
    return {
        "IB": 0.55,
        "NBC": 0.30,
        "VBNC": 0.77,
        "PB": vdd - 0.57,
        "PX": vdd - 0.20,
        "VBPC": vdd - 0.70,
        "VCM": vcm_ref,
        "VREF1": 0.55,
        "VCMIN": 0.60,
        "CMR": 0.16,
        "TAIL": 0.15,
        "A1": 0.27,
        "A2": 0.27,
        "O1": 0.55,
        "O2": 0.55,
        "B1": vdd - 0.18,
        "B2": vdd - 0.18,
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
        polarity = model["type"].rsplit(".", 1)[-1]
        model["type"] = f"tsmc28hpcp.{polarity}"
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
        model = {"type": f"tsmc28hpcp.{polarity}"}
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
