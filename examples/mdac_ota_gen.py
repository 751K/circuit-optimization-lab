#!/usr/bin/env python3
"""Single source of truth for the FreePDK45 14-bit-pipeline MDAC OTA and its
testbenches.

Design task: fully-differential two-stage Miller OTA for the first-stage MDAC of a
14-bit / 100 MS/s pipeline ADC (residue gain 8, hold phase ~5 ns).  See
``docs/mdac_ota_derivation.md`` for the ADC->OTA spec derivation, the architecture
decision record and the sizing table; this file MECHANISES that design so every
testbench JSON shares one identical DUT block (change a W here, it changes
everywhere).

Architecture (all-transistor DUT, one ideal 20 uA current reference from the TB):
  * Stage 1: telescopic cascode, NMOS input pair (gain, cascoded ~ >50 dB).
  * Stage 2: complementary common-source, high output swing (~VDD-2Vdsat).
  * Miller compensation Cc with a triode-NMOS nulling resistor Rz per side.
  * CMFB1 regulates the stage-1 cascode output CM (drives the PMOS loads).
  * CMFB2 regulates the output CM (drives the stage-2 NMOS loads).
  * Bias network: a 20 uA reference (ideal, TB-side) mirrored to every tail /
    cascode gate through an all-transistor constant-current bias generator.
  * VCM = VDD/2 reference: matched poly-resistor divider (ratio-based, supply-
    tracking); it only drives high-Z CMFB sense gates.

The testbenches (each reuses ``core()`` verbatim):
  freepdk45_mdac_ota.json         closed-loop MDAC residue transient (Cs/Cf net,
                                  bottom-plate PWL step, 500 fF loads)
  freepdk45_mdac_ota_ac.json      open-loop differential .ac (gain / UGBW / PM)
  freepdk45_mdac_ota_dmloop.json  differential loop gain (Middlebrook Vinj)
  freepdk45_mdac_ota_cmfb1.json   CMFB1 loop gain (Vinj)
  freepdk45_mdac_ota_cmfb2.json   CMFB2 loop gain (Vinj)
  freepdk45_mdac_ota_noise.json   input-referred noise
"""
from __future__ import annotations

import json
import os

# ── derived design constants (see docs/mdac_ota_derivation.md) ──────────────────
CS = 2.6e-12        # CDAC / total sampling cap per side  (noise budget)
CF = CS / 8.0       # feedback cap for closed-loop residue gain 8 (non-flip)
CL = 500e-15        # external single-ended load per output
CC = 1.0e-12        # Miller compensation cap per side (sets loop NBW: noise budget)
COUT = 1.5e-12      # DUT-internal MOM cap per output: filters above-UGF device noise
VDD_NOM = 1.00      # design nominal supply (campaign 0.90/1.00/1.10 V, +-10%)

# ── device sizes: {name: (W_um, L_um)}.  role comments drive the sizing table. ──
# W's are the tuned values; currents/regions are verified by op_ngspice at PVT.
SZ = {
    # -- bias generator (all mirrored from the 20 uA reference at node IB) --------
    "MBN":   (12.0, 0.2),    # NMOS diode: 20 uA -> IB (tail/mirror gate ref)
    "MPR":   (14.0, 0.20),   # PMOS diode: carries 20 uA -> PB (pmos mirror gate)
    "MPRN":  (12.0, 0.2),    # NMOS mirror sets the 20 uA in the PMOS ref leg
    # Cascode-gate and CM references are all "replica Vgs + poly-R level shift"
    # legs (wide-swing style): a density-matched diode tracks the target device's
    # Vgs across PVT, and a poly resistor sets the headroom term.
    "MPC":   (14.0, 0.20),   # PMOS mirror sources 20 uA into the VBNC leg
    "MCND":  (5.9, 0.15),    # NMOS diode, M3-density replica: VBNC = Vgs(M3) + 0.28
    "MNC":   (12.0, 0.2),    # NMOS mirror sinks 20 uA from the VBPC leg
    "MCPD":  (19.0, 0.15),    # PMOS diode, M5-density replica: VBPC = VDD-0.28-Vgs(M5)
    "MREPP": (76.0, 0.20),   # PMOS mirror: ~108 uA into the M9-replica leg
    "MREP":  (17.5, 0.14),   # NMOS diode, M9 replica (same L, same current density):
    #                          VREF1 = Vgs(M9 @ I2-target) -> CMFB1 reference, so the
    #                          stage-2 current is mirror-tracked at every PVT corner.
    "MCMP":  (28.0, 0.20),   # PMOS mirror: ~40 uA into the input-CM reference leg
    "MCMD":  (8.5, 0.16),    # NMOS diode, M1-density replica: VCMIN = Vgs(M1) + 0.12.
    #                          NMOS-tracking input virtual-ground CM keeps
    #                          TAIL = VCMIN - Vgs(M1) ~ 0.12 V constant over PVT so
    #                          the tail M0 stays saturated even at ss/125C/0.9V (a
    #                          VDD/2 input CM leaves TAIL < 40 mV there).  The MDAC
    #                          virtual-ground CM is a free design choice; the OUTPUT
    #                          CM stays at VDD/2 per spec.
    # -- stage 1 telescopic cascode ----------------------------------------------
    "M0":    (3000.0, 0.2),  # NMOS tail (~1.3 mA)
    "M1":    (560.0, 0.16),  # NMOS input pair +
    "M2":    (560.0, 0.16),  # NMOS input pair -
    "M3":    (700.0, 0.15),   # NMOS cascode +
    "M4":    (700.0, 0.15),   # NMOS cascode -
    "M5":    (2250.0, 0.15),  # PMOS cascode +
    "M6":    (2250.0, 0.15),  # PMOS cascode -
    "M7":    (650.0, 0.13),  # PMOS load + (gate = CMFB1 ctrl)
    "M8":    (650.0, 0.13),  # PMOS load - (gate = CMFB1 ctrl)
    # -- stage 2 NMOS-input common-source, high output swing ---------------------
    "M9":    (585.0, 0.14),   # NMOS CS + (gate = O1)
    "M10":   (585.0, 0.14),   # NMOS CS - (gate = O2)
    "M11":   (1870.0, 0.14),  # PMOS load + (gate = CMFB2 ctrl)
    "M12":   (1870.0, 0.14),  # PMOS load - (gate = CMFB2 ctrl)
    "MRZ1":  (48.0, 0.05),    # triode NMOS nulling resistor + (gate = VDD)
    "MRZ2":  (48.0, 0.05),    # triode NMOS nulling resistor -
    # -- CMFB1 (stage-1 CM): NMOS sense pairs steer into diode-PMOS CTRL1 ---------
    # Tail current sized to null the CM offset: I_tail = I_M7 * (Wdiode*Lload)/(Wload*Ldiode).
    "MS1":   (90.0, 0.12),   # sense O1
    "MS2":   (90.0, 0.12),   # ref  VCM
    "MS3":   (90.0, 0.12),   # sense O2
    "MS4":   (90.0, 0.12),   # ref  VCM
    "MT1":   (70.0, 0.2),   # CMFB1 tail (~90 uA, offset null)
    "MT2":   (70.0, 0.2),
    "MDL1":  (20.0, 0.10),   # diode PMOS at CTRL1 (ref-sum load)
    "MDS1":  (20.0, 0.10),   # diode PMOS at NSNS1 (sense-sum load, symmetry)
    # -- CMFB2 (output CM): resistive CM sense (linear over the full swing) into a
    #    single NMOS diff pair; diode-PMOS CTRL2 drives the PMOS loads M11/M12.
    "MRA":   (60.0, 0.12),   # sense CMS (resistor-averaged output CM)
    "MRB":   (60.0, 0.12),   # ref  VCM
    "MTB":   (100.0, 0.2),   # CMFB2 tail (~220 uA, offset null)
    "MDL2":  (35.0, 0.14),   # diode PMOS at CTRL2 (ref-side load)
    "MDS2":  (35.0, 0.14),   # diode PMOS at NSNS2 (sense-side load, symmetry)
}

# (name, drain, gate, source, kind).  kind: 'n' NMOS, 'p' PMOS (bulk=VDD).
CORE_DEVICES = [
    # bias generator
    ("MBN",  "IB",   "IB",   "GND", "n"),
    ("MPRN", "PB",   "IB",   "GND", "n"),
    ("MPR",  "PB",   "PB",   "VDD", "p"),
    # VBNC leg: 20 uA -> M3-replica diode -> RNC to GND: VBNC = Vgs(M3-dens) + 0.28
    ("MPC",  "VBNC", "PB",   "VDD", "p"),
    ("MCND", "VBNC", "VBNC", "NBC", "n"),
    # VBPC leg: RCP from VDD -> M5-replica diode -> 20 uA sink:
    # VBPC = VDD - 0.28 - |Vgs(M5-dens)|
    ("MCPD", "VBPC", "VBPC", "PX",  "p"),
    ("MNC",  "VBPC", "IB",   "GND", "n"),
    ("MREPP", "VREF1", "PB", "VDD", "p"),   # replica-bias leg: PB-mirrored current
    ("MREP", "VREF1", "VREF1", "GND", "n"),  # M9 replica diode -> VREF1
    # VCMIN leg: 40 uA -> M1-replica diode -> RCM to GND: VCMIN = Vgs(M1-dens) + 0.12
    ("MCMP", "VCMIN", "PB", "VDD", "p"),
    ("MCMD", "VCMIN", "VCMIN", "CMR", "n"),
    # stage 1
    ("M0",   "TAIL", "IB",   "GND", "n"),
    ("M1",   "A1",   "INP",  "TAIL", "n"),
    ("M2",   "A2",   "INN",  "TAIL", "n"),
    ("M3",   "O1",   "VBNC", "A1",  "n"),
    ("M4",   "O2",   "VBNC", "A2",  "n"),
    ("M5",   "O1",   "VBPC", "B1",  "p"),
    ("M6",   "O2",   "VBPC", "B2",  "p"),
    ("M7",   "B1",   "CTRL1", "VDD", "p"),
    ("M8",   "B2",   "CTRL1", "VDD", "p"),
    # stage 2 (NMOS CS in from O1/O2; PMOS load from CTRL2).  OUTN driven by O1 side.
    ("M9",   "OUTN", "O1",   "GND", "n"),
    ("M10",  "OUTP", "O2",   "GND", "n"),
    ("M11",  "OUTN", "CTRL2", "VDD", "p"),
    ("M12",  "OUTP", "CTRL2", "VDD", "p"),
    # Zero-nulling triode on the O1/O2 side of the series-RC Miller branch: O1/O2
    # only move ~+-20 mV over the full output swing, so the triode Vgs (VDD-O1)
    # stays constant and Rz never collapses/opens at swing extremes (a triode with
    # its source on OUT starves at Vout=VDD-Vth and leaves a slow doublet).
    ("MRZ1", "MZ1",  "VDD",  "O1", "n"),
    ("MRZ2", "MZ2",  "VDD",  "O2", "n"),
    # CMFB1 -> CTRL1 (PMOS loads M7/M8).  Sense-sum drains -> NSNS1 (diode-loaded
    # for Vds symmetry with the ref-sum node CTRL1 -> low systematic CM offset).
    ("MS1",  "NSNS1", "O1",   "CMT1", "n"),
    ("MS2",  "CTRL1", "VREF1", "CMT1", "n"),
    ("MS3",  "NSNS1", "O2",   "CMT2", "n"),
    ("MS4",  "CTRL1", "VREF1", "CMT2", "n"),
    ("MT1",  "CMT1", "IB",   "GND", "n"),
    ("MT2",  "CMT2", "IB",   "GND", "n"),
    ("MDL1", "CTRL1", "CTRL1", "VDD", "p"),
    ("MDS1", "NSNS1", "NSNS1", "VDD", "p"),
    # CMFB2 -> CTRL2 (PMOS loads M11/M12).  CMS = resistor-averaged output CM
    # (RS1/RS2, linear over the whole output swing); single diff pair MRA/MRB.
    ("MRA",  "NSNS2", "CMS", "CMB", "n"),
    ("MRB",  "CTRL2", "VCM", "CMB", "n"),
    ("MTB",  "CMB",  "IB",   "GND", "n"),
    ("MDL2", "CTRL2", "CTRL2", "VDD", "p"),
    ("MDS2", "NSNS2", "NSNS2", "VDD", "p"),
]

# solved internal nodes of the DUT (everything the OTA computes)
CORE_SOLVED = [
    "IB", "NBC", "VBNC", "PB", "PX", "VBPC", "VCM", "VREF1", "VCMIN", "CMR",
    "TAIL", "A1", "A2", "O1", "O2", "B1", "B2",
    "OUTP", "OUTN", "MZ1", "MZ2",
    "CTRL1", "CMT1", "CMT2", "NSNS1",
    "CTRL2", "CMB", "CMS", "NSNS2",
]

RDIV = 60e3      # VCM = VDD/2 divider leg (matched poly pair)
RSENSE = 100e3   # CMFB2 output-CM sense resistor (poly; >> stage-2 ro ~ 0.3k)
RNC = 12e3       # VBNC leg level shift: 20 uA * 12k = 0.24 V (A1 headroom)
RCP = 14e3       # VBPC leg level shift: 20 uA * 14k = 0.28 V (B1 headroom)
RCM = 5.5e3        # VCMIN leg level shift: 40 uA * 5.5k = 0.22 V (TAIL headroom)
CSENSE = 150e-15  # feedforward cap across each RSENSE (cancels the RS*C(CMS) pole)


def core(vdd=VDD_NOM, gate_rename=None):
    """Return the shared DUT fragment: device dicts + model bindings.

    ``gate_rename`` maps ``device_name -> new_gate_node`` to split a gate for a
    Middlebrook loop-break (e.g. M7/M8 gate CTRL1 -> CTRL1G with Vinj in series)."""
    gate_rename = dict(gate_rename or {})
    devices, models = [], {}
    for name, d, g, s, kind in CORE_DEVICES:
        W, L = SZ[name]
        dv = {"name": name, "drain": d, "gate": gate_rename.get(name, g),
              "source": s, "W": W, "L": L}
        # Multi-finger layout (~2 um fingers) on every wide device: FreePDK45's
        # BSIM4 gate-resistance noise (rgateMod) scales as W/nf^2 and DOMINATES
        # the output noise for single-finger 100+ um devices (measured: m1.rg
        # 566 uV vs m1.id 163 uV at nf=1).  NF is layout-real and mandatory here.
        if W >= 8.0:
            dv["NF"] = max(2, int(round(W / 2.0)))
        devices.append(dv)
        if kind == "n":
            models[name] = {"type": "freepdk45.nmos"}
        else:
            models[name] = {"type": "freepdk45.pmos", "vb": vdd}
    return devices, models


def core_passives():
    """Miller caps + VCM divider (internal to the DUT)."""
    caps = [
        {"name": "CC1", "a": "OUTN", "b": "MZ1", "C": CC},
        {"name": "CC2", "a": "OUTP", "b": "MZ2", "C": CC},
        {"name": "CFS1", "a": "OUTP", "b": "CMS", "C": CSENSE},
        {"name": "CFS2", "a": "OUTN", "b": "CMS", "C": CSENSE},
        {"name": "COUT1", "a": "OUTP", "b": "GND", "C": COUT},
        {"name": "COUT2", "a": "OUTN", "b": "GND", "C": COUT},
    ]
    res = [
        {"name": "RVCM1", "a": "VDD", "b": "VCM", "R": RDIV},
        {"name": "RVCM2", "a": "VCM", "b": "GND", "R": RDIV},
        {"name": "RS1", "a": "OUTP", "b": "CMS", "R": RSENSE},
        {"name": "RS2", "a": "OUTN", "b": "CMS", "R": RSENSE},
        # poly level-shift resistors of the replica bias legs
        {"name": "RNC", "a": "NBC", "b": "GND", "R": RNC},
        {"name": "RCP", "a": "VDD", "b": "PX", "R": RCP},
        {"name": "RCM", "a": "CMR", "b": "GND", "R": RCM},
    ]
    return caps, res


def base_seed(vdd=VDD_NOM):
    h = vdd / 2.0
    return {
        "IB": 0.45, "NBC": 0.28, "VBNC": 0.72, "PB": vdd - 0.45, "PX": vdd - 0.28,
        "VBPC": vdd - 0.70, "VCM": h, "VREF1": 0.48, "VCMIN": 0.56, "CMR": 0.12,
        "TAIL": 0.13, "A1": 0.26, "A2": 0.26, "O1": 0.48, "O2": 0.48,
        "B1": vdd - 0.13, "B2": vdd - 0.13,
        "OUTP": h, "OUTN": h, "MZ1": h, "MZ2": h,
        "CTRL1": vdd - 0.45, "CMT1": 0.11, "CMT2": 0.11, "NSNS1": vdd - 0.42,
        "CTRL2": vdd - 0.42, "CMB": 0.11, "CMS": h, "NSNS2": vdd - 0.42,
    }


def _rails(extra=None):
    r = {"VDD": "VDD", "GND": 0.0}
    if extra:
        r.update(extra)
    return r


def _iref():
    # the ONE ideal reference: 20 uA injected into the bias node IB.
    return [["Iref", "VDD", "IB", 20e-6]]


def build_ac(vdd=VDD_NOM):
    """Open-loop differential .ac testbench.

    The input gates bias to the DUT's own VCMIN reference through 2 Mohm (as in
    the MDAC hold phase), and the differential stimulus AC-couples in through
    100 pF TB caps from 0 V TB vsources VACP/VACN (drive them (0.5,0)/(0.5,180)
    in ac_ngspice).  Coupling corner ~800 Hz, cap divider ~0.99 — read the gain
    as the plateau above ~2 kHz (dc_gain_db at a 1-10 kHz sweep start)."""
    devices, models = core()
    caps, res = core_passives()
    solved = list(CORE_SOLVED) + ["INP", "INN", "ACP", "ACN"]
    caps = caps + [
        {"name": "CL1", "a": "OUTP", "b": "GND", "C": CL},
        {"name": "CL2", "a": "OUTN", "b": "GND", "C": CL},
        {"name": "CAC1", "a": "ACP", "b": "INP", "C": 100e-12},
        {"name": "CAC2", "a": "ACN", "b": "INN", "C": 100e-12},
    ]
    res = res + [
        {"name": "RB1", "a": "INP", "b": "VCMIN", "R": 2e6},
        {"name": "RB2", "a": "INN", "b": "VCMIN", "R": 2e6},
    ]
    seed = base_seed(vdd)
    seed.update({"INP": 0.57, "INN": 0.57, "ACP": 0.0, "ACN": 0.0})
    d = {
        "name": "freepdk45_mdac_ota_ac",
        "description": "Open-loop differential AC of the MDAC two-stage OTA. "
        "Inputs bias to the on-chip VCMIN reference via 2 Mohm; the stimulus "
        "AC-couples through 100 pF from TB vsources VACP/VACN (+-0.5 differential). "
        "Gain/UGBW/PM from v(OUTP,OUTN) above the ~800 Hz coupling corner. "
        "500 fF load per output. One ideal 20 uA reference at IB.",
        "solved": solved,
        "rails": _rails(),
        "bias": {"VDD": vdd},
        "devices": devices,
        "models": models,
        "current_sources": _iref(),
        "capacitors": caps,
        "resistors": res,
        "vsources": [["VACP", "ACP", "GND", 0.0], ["VACN", "ACN", "GND", 0.0]],
        "outputs": ["OUTP", "OUTN"],
        "dc_guesses": [seed],
        "analyses": {"ac": {"freqs": {"start": 1e4, "stop": 5e10,
                                      "num": 121, "scale": "log"}}},
    }
    return d


def build_noise(vdd=VDD_NOM):
    """CLOSED-LOOP noise TB: the full MDAC configuration (Cs/Cf network, hold
    phase) with the bottom plates held at VCM by constant vsources.  Run
    noise_ngspice(out='OUTP', ref='OUTN', src='VBP1'): onoise is the closed-loop
    residue-amplifier output noise (integrate it and divide by the residue gain 8
    to refer to the ADC input); inoise refers it to the bottom-plate signal input
    (|H| = 8 in band)."""
    d = build_transient(vdd)
    d["name"] = "freepdk45_mdac_ota_noise"
    d["description"] = ("Closed-loop noise of the MDAC residue amplifier (hold "
                        "configuration, bottom plates static at VCM). onoise at "
                        "v(OUTP,OUTN) integrated over band, /8 -> ADC-input-referred; "
                        "must sit under the amplifier noise allocation "
                        "(see docs/mdac_ota_derivation.md).")
    d["vsources"] = [["VBP1", "BP1", "GND", vdd / 2],
                     ["VBP2", "BP2", "GND", vdd / 2]]
    d["analyses"] = {"noise": {"freqs": {"start": 1e3, "stop": 2e10, "num": 81,
                                         "scale": "log"}, "band": [1e3, 1e10]}}
    return d


def build_dmloop(vdd=VDD_NOM):
    """Differential feedback loop gain (differential Middlebrook injection).

    The MDAC Cs/Cf network closes the loop (beta ~ 1/9).  Both input gates are
    split from their feedback-network nodes: Vinj (the measured 0 V source) breaks
    INP/INPD, and a TB-side mirror VCVS (Emir, exempt from the all-transistor rule)
    forces the INN/INND break to the exact opposite voltage, so the injection is
    purely differential and ``T = -V(INPD)/V(INP)`` (what loop_gain_ngspice
    computes for inject='Vinj') IS the differential loop gain — a single-ended
    break in a fully-differential loop would otherwise mix in the CM path.
    Large TB resistors set the virtual-ground DC to VCM (open-cap DC path)."""
    devices, models = core(vdd)
    caps, res = core_passives()
    solved = list(CORE_SOLVED) + ["INP", "INN", "INPD", "INND"]
    caps = caps + [
        {"name": "CL1", "a": "OUTP", "b": "GND", "C": CL},
        {"name": "CL2", "a": "OUTN", "b": "GND", "C": CL},
        {"name": "CF1", "a": "OUTP", "b": "INPD", "C": CF},
        {"name": "CS1", "a": "INPD", "b": "VCM", "C": CS},
        {"name": "CF2", "a": "OUTN", "b": "INND", "C": CF},
        {"name": "CS2", "a": "INND", "b": "VCM", "C": CS},
    ]
    # DC-bias resistors: 2 Mohm is large enough that the cap feedback governs above
    # ~30 kHz (loop crossover ~500 MHz), yet small enough that 45 nm gate leakage
    # (~nA) drops only a few mV (1 Gohm here would float the gates ~1 V off VCM).
    res = res + [
        {"name": "RDC1", "a": "INPD", "b": "VCMIN", "R": 2e6},
        {"name": "RDC2", "a": "INND", "b": "VCMIN", "R": 2e6},
    ]
    seed = base_seed(vdd)
    seed.update({"INP": 0.57, "INN": 0.57, "INPD": 0.57, "INND": 0.57})
    d = {
        "name": "freepdk45_mdac_ota_dmloop",
        "description": "Differential-loop gain of the OTA closed by the MDAC Cs/Cf "
        "network (beta~1/9): differential Middlebrook injection (Vinj at the + gate "
        "break, mirror VCVS at the - gate break). PM from "
        "loop_gain_ngspice(inject='Vinj') must be > 60 deg at all PVT.",
        "solved": solved, "rails": _rails(), "bias": {"VDD": vdd},
        "devices": devices, "models": models, "current_sources": _iref(),
        "capacitors": caps, "resistors": res,
        "vsources": [["Vinj", "INP", "INPD", 0.0]],
        "vcvs": [{"name": "Emir", "p": "INN", "q": "INND",
                  "cp": "INPD", "cn": "INP", "mu": 1.0}],
        "outputs": ["OUTP", "OUTN"], "dc_guesses": [seed],
        "analyses": {"ac": {"freqs": {"start": 1e3, "stop": 5e10, "num": 81,
                                      "scale": "log"}}},
    }
    return d


def _cmfb_loop(name, break_devs, ctrl, ctrlg, desc, vdd=VDD_NOM):
    """CMFB loop gain TB: split the loop node (ctrl -> ctrlg) with Vinj; the OTA
    inputs bias to VCMIN through 2 Mohm (as in the hold phase); CL on the outputs."""
    devices, models = core(vdd, gate_rename={dv: ctrlg for dv in break_devs})
    caps, res = core_passives()
    solved = list(CORE_SOLVED) + [ctrlg, "INP", "INN"]
    caps = caps + [
        {"name": "CL1", "a": "OUTP", "b": "GND", "C": CL},
        {"name": "CL2", "a": "OUTN", "b": "GND", "C": CL},
    ]
    res = res + [
        {"name": "RB1", "a": "INP", "b": "VCMIN", "R": 2e6},
        {"name": "RB2", "a": "INN", "b": "VCMIN", "R": 2e6},
    ]
    seed = base_seed(vdd)
    seed[ctrlg] = seed[ctrl]
    seed.update({"INP": 0.57, "INN": 0.57})
    d = {
        "name": name, "description": desc,
        "solved": solved, "rails": _rails(),
        "bias": {"VDD": vdd},
        "devices": devices, "models": models, "current_sources": _iref(),
        "capacitors": caps, "resistors": res,
        "vsources": [["Vinj", ctrlg, ctrl, 0.0]],
        "outputs": ["OUTP", "OUTN"], "dc_guesses": [seed],
        "analyses": {"ac": {"freqs": {"start": 1e3, "stop": 5e10, "num": 81,
                                      "scale": "log"}}},
    }
    return d


def build_cmfb1(vdd=VDD_NOM):
    return _cmfb_loop(
        "freepdk45_mdac_ota_cmfb1", ["M7", "M8"], "CTRL1", "CTRL1G",
        "CMFB1 (stage-1 CM) loop gain: Vinj splits the PMOS-load gate (CTRL1 -> "
        "CTRL1G). PM must be > 60 deg at all PVT.", vdd)


def build_cmfb2(vdd=VDD_NOM):
    # Break at the MRA sense gate (CMS -> CMSG), not at the M11/12 gate: the
    # 600 um load-gate capacitance drops below the CTRL2 diode impedance in the
    # GHz range and invalidates single-injection there (T(f) reads a rising HF
    # artifact); the MRA gate stays high-Z against the RS/CFS sense network.
    return _cmfb_loop(
        "freepdk45_mdac_ota_cmfb2", ["MRA"], "CMS", "CMSG",
        "CMFB2 (output CM) loop gain: Vinj splits the CM-sense gate (CMS -> "
        "CMSG). PM must be > 60 deg at all PVT.", vdd)


def build_transient(vdd=VDD_NOM):
    """Closed-loop MDAC residue hold-phase transient.

    Sampling/phase-1 clocking are OUT of scope (this DUT is the OTA + hold phase).
    The sampled charge is pre-established at t<0 with the bottom plates BP1/BP2 at
    VCM (balanced); at t=0 a differential bottom-plate PWL step (waveform keys
    'bp1'/'bp2', supplied by the driver) injects the residue, and the OTA settles
    v(OUTP,OUTN) -> 8 x residue.  Cs=CDAC, Cf=Cs/8 (non-flip -> gain 8), 500 fF
    external load per output, DC virtual ground set to VCM by large TB resistors."""
    devices, models = core(vdd)
    caps, res = core_passives()
    solved = list(CORE_SOLVED) + ["INP", "INN", "BP1", "BP2"]
    caps = caps + [
        {"name": "CL1", "a": "OUTP", "b": "GND", "C": CL},
        {"name": "CL2", "a": "OUTN", "b": "GND", "C": CL},
        {"name": "CF1", "a": "OUTP", "b": "INP", "C": CF},
        {"name": "CS1", "a": "INP", "b": "BP1", "C": CS},
        {"name": "CF2", "a": "OUTN", "b": "INN", "C": CF},
        {"name": "CS2", "a": "INN", "b": "BP2", "C": CS},
    ]
    # 2 Mohm: compromise between the slew-window charge artifact (~1/RDC, TB-only
    # path, absent in the switched MDAC) and 125C gate-leak IR drop (~20 mV).
    res = res + [
        {"name": "RDC1", "a": "INP", "b": "VCMIN", "R": 2e6},
        {"name": "RDC2", "a": "INN", "b": "VCMIN", "R": 2e6},
    ]
    seed = base_seed(vdd)
    seed.update({"INP": 0.57, "INN": 0.57, "BP1": vdd / 2, "BP2": vdd / 2})
    d = {
        "name": "freepdk45_mdac_ota",
        "description": "Closed-loop MDAC first-stage residue amplifier (hold phase) "
        "on the FreePDK45 two-stage OTA. Cs=2.6pF (CDAC), Cf=Cs/8 -> residue gain 8 "
        "(non-flip), 500 fF external load/side, one ideal 20 uA reference. Bottom "
        "plates BP1/BP2 step differentially at t=0 (waveform keys bp1/bp2) to inject "
        "the residue; OTA settles to 8 x residue within the 5 ns hold to < 0.1%. "
        "Sampling/phase-1 out of scope. See docs/mdac_ota_derivation.md.",
        "solved": solved, "rails": _rails(), "bias": {"VDD": vdd},
        "devices": devices, "models": models, "current_sources": _iref(),
        "capacitors": caps, "resistors": res,
        "vsources": [["VBP1", "BP1", "GND", "bp1"], ["VBP2", "BP2", "GND", "bp2"]],
        "outputs": ["OUTP", "OUTN"],
        "aliases": {"vop": "OUTP", "von": "OUTN"},
        "dc_guesses": [seed],
    }
    return d


def all_testbenches(vdd=VDD_NOM):
    return {
        "freepdk45_mdac_ota.json": build_transient(vdd),
        "freepdk45_mdac_ota_ac.json": build_ac(vdd),
        "freepdk45_mdac_ota_dmloop.json": build_dmloop(vdd),
        "freepdk45_mdac_ota_cmfb1.json": build_cmfb1(vdd),
        "freepdk45_mdac_ota_cmfb2.json": build_cmfb2(vdd),
        "freepdk45_mdac_ota_noise.json": build_noise(vdd),
    }


def write_all(outdir):
    os.makedirs(outdir, exist_ok=True)
    for fn, dct in all_testbenches().items():
        path = os.path.join(outdir, fn)
        with open(path, "w") as fh:
            json.dump(dct, fh, indent=2)
        print("wrote", path)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    write_all(here)
