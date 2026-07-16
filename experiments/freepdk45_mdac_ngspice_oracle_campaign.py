#!/usr/bin/env python3
"""Explicit ngspice-oracle 45-point PVT regression for the FreePDK45 MDAC OTA.

Grid: corners {tt, ss, ff, sf, fs} x temperatures {-40, 27, 125} C x supplies
{0.90, 1.00, 1.10} V = 45 points.  Every point is measured with the ngspice
C-BSIM4 full-circuit oracles, driven from the single source of truth
``examples/mdac_ota_gen.py`` (each supply rebuilds the testbench so the PMOS bulk
``vb`` and the VDD/2 references track the rail).

Per point (see docs/mdac_ota_derivation.md sections 5-6 for the BINDING conventions):

  a. Open-loop differential AC (build_ac): gain_dB at 10 kHz (the AC-coupling
     plateau, section 6.3), open-loop UGBW, and the forward-response phase margin
     (reference only -- the loop PM below is the stability metric).
  b. DM loop (build_dmloop, differential Middlebrook, inject=Vinj): loop UGF, PM,
     GM.  Spec: PM > 60 deg.  The Cs/Cf feedback puts the loop-gain plateau ABOVE
     the ~30 kHz feedback corner, so PM references the plateau -> fstart = 1e5
     (reproduces the designer's nominal DM PM 122.3 deg exactly).
  c. CMFB1 / CMFB2 (build_cmfb1/2): PM each (spec > 60 deg) and a finite UGF (the
     loop is actually active).  These loops have DC plateaus, so PM references DC
     -> fstart = 1e3 (reproduces the designer's nominal CMFB2 PM 81.3 deg).
  d. Closed-loop residue transients (build_transient): residue levels
     s in {-FS/16, -FS/32, 0, +FS/32, +FS/16}, FS = 0.9.  ideal = -8*s,
     err = |Vod(5 ns) - ideal| / 0.45 (FS-normalised, section 6.2); spec < 0.1 %
     for every level.  Solver tolerances tightened per section 6.1
     (reltol 1e-7 / vntol 1e-11 / abstol 1e-15), max_step 0.05 ns, 5 ns hold.
     Overshoot and the 5 ns CM excursion are recorded (report only, section 6.5).
  e. Static output CM error: t=0 DC solution of the transient TB, |CM - VDD/2|;
     spec < 20 mV.
  f. Saturation (M0-M12 region_ok, margin 0): at the static point (op on build_ac)
     AND after settling at s = +-FS/16 (op on the transient TB with the bottom
     plates held DC at h +- s/2, seeded from the transient's final state -- that IS
     the settled DC point).
  g. Closed-loop noise (build_noise): onoise integrated 10 MHz-20 GHz (section 6.4),
     v(OUTP,OUTN); report output-referred rms AND ADC-input-referred (/8).
     Spec: output <= 452 uV rms.
  h. Power: supply current i(Vrail_VDD) from a static .op x VDD (report only).

Output: one row per point to ``results/mdac_ota_pvt45.csv`` (all metrics + pass
flags).  Resumable -- points already in the CSV are skipped.  Points run in
parallel across a ThreadPoolExecutor (each ngspice call is a subprocess that
releases the GIL).

Usage::

    .venv/bin/python experiments/freepdk45_mdac_ngspice_oracle_campaign.py \
        [--workers 8] [--force]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import mdac_ota_gen as G  # noqa: E402
from circuitopt.circuit_loader import circuit_from_dict  # noqa: E402
from circuitopt.ngspice_ac import (  # noqa: E402
    _network_deck,
    _resolve_source_name,
    _run_ngspice_capture,
    ac_ngspice,
    ac_response,
    loop_gain_ngspice,
    noise_ngspice,
    op_ngspice,
    phase_margin,
    unity_gain_freq,
)
from circuitopt.ngspice_char import ngspice_chain_enabled  # noqa: E402
from circuitopt.ngspice_render import _element  # noqa: E402
from circuitopt.ngspice_transient import (  # noqa: E402
    transient_ngspice,
    transient_ngspice_chain,
)

class RunPointError(Exception):
    """Wraps a run_point failure so the collector can log its wall-clock elapsed.

    The original exception is preserved on ``__cause__`` (raised ``from`` it)."""

    def __init__(self, elapsed_s):
        super().__init__()
        self.elapsed_s = float(elapsed_s)


# ── grid ──────────────────────────────────────────────────────────────────────
CORNERS = ["tt", "ss", "ff", "sf", "fs"]
TEMPS_C = [-40.0, 27.0, 125.0]
SUPPLIES = [0.90, 1.00, 1.10]

FS = 0.9                                   # differential full scale (Vpp-diff)
RESIDUE_LEVELS = [-FS / 16, -FS / 32, 0.0, FS / 32, FS / 16]
CORE_DEVS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8",
             "M9", "M10", "M11", "M12"]

# BINDING measurement conventions (docs/mdac_ota_derivation.md section 6)
TIGHT = {"reltol": 1e-7, "vntol": 1e-11, "abstol": 1e-15}

# specs
SPEC_GAIN_DB = 84.0        # design all-PVT floor (section 1.6; hard floor 82 dB)
SPEC_PM_DEG = 60.0
SPEC_SETTLE_PCT = 0.1      # % FS
SPEC_CM_MV = 20.0
SPEC_NOISE_UV = 452.0

OUT_CSV = ROOT / "results" / "mdac_ota_pvt45.csv"

# ── optional module hooks (empty/off => base behaviour unchanged) ────────────────
# A specialisation (e.g. the TSMC28 campaign) may set GRID_PRIORITY to a list of
# (corner, temp_c, vdd) tuples that main() runs FIRST, in the given order, before the
# rest of the nested grid.  Left empty here so the FreePDK45 campaign is unchanged.
GRID_PRIORITY: list = []

# --smoke skips the two slowest per-point measurements.  run_point (base and the
# TSMC28 override) consults these; default False keeps every measurement.  When a
# measurement is skipped its columns are nan and the matching pass flag is False,
# and pass_all is computed only over the specs actually measured (see the ``smoke``
# column).  Skipping is an honest non-sign-off, never a silent pass.
SKIP_CODE_TRANSITION = False
SKIP_NOISE = False

CSV_FIELDS = [
    "corner", "temp_c", "vdd",
    "gain_db", "ac_ugbw_hz", "ac_pm_deg",
    "dm_ugf_hz", "dm_pm_deg", "dm_gm_db",
    "cmfb1_ugf_hz", "cmfb1_pm_deg", "cmfb2_ugf_hz", "cmfb2_pm_deg",
    "settle_n2_pct", "settle_n1_pct", "settle_z_pct", "settle_p1_pct",
    "settle_p2_pct", "settle_worst_pct",
    "overshoot_worst_pct", "cm5_worst_mv",
    "cm_static_mv",
    "sat_static_ok", "sat_settled_ok", "sat_bad",
    "noise_onoise_uv", "noise_adc_uv",
    "isupply_ma", "power_mw",
    "pass_gain", "pass_dmpm", "pass_cmfb1pm", "pass_cmfb2pm",
    "pass_settle", "pass_cm", "pass_sat", "pass_noise", "pass_all",
]


# ── helpers ─────────────────────────────────────────────────────────────────────
def _dk(spec, tk):
    """Per-device kwargs with the point temperature threaded in (Kelvin)."""
    b = spec.binding()
    base = b.device_kwargs or {}
    return b, {name: dict(base.get(name, {}), temperature=tk)
               for name, *_ in spec.topology.devices}


def _supply_current(spec, dk, corner, seed, core_devs=None):
    """Static supply current |i(Vrail_VDD)| [A] from a full-circuit .op deck.

    Returns ``(isupply_A, static_regions)``.  When ``core_devs`` is given AND the
    process renders through an adapter, the SAME .op deck also prints each core
    device's ``vds``/``vdsat`` op-vectors, and ``static_regions`` is a mapping
    ``{device: region_ok}`` (``region_ok = |vds| >= |vdsat|`` — the margin-0
    saturation test, identical to :func:`op_ngspice`).  This folds the static
    saturation check into the power .op so the point costs one foundry-macro
    expansion instead of two.  ``static_regions`` is ``None`` when the merge is not
    available (no adapter, or ``core_devs`` not requested), and the caller must fall
    back to a separate :func:`op_ngspice` static call (keeps FreePDK45 working)."""
    b = spec.binding()
    lines, _nm, _node, adapter = _network_deck(
        spec.topology, spec.sizes, spec.bias, header="* circuitopt pvt power .op",
        nf=spec.nf, model_types=b.model_types, device_kwargs=dk, corner=corner,
        temperature=None, x0_guess=seed)

    merge = bool(core_devs) and adapter is not None
    dev_vectors = {}       # device -> {"vds": vec_no_at, "vdsat": vec_no_at}
    prints = ["print i(vrail_vdd)"]
    if merge:
        for name in core_devs:
            elem = _element("X", name).lower()
            dev_vectors[name] = {}
            for p in ("vds", "vdsat"):
                vector = adapter.op_vector(elem, p)          # e.g. @m.xm0.main[vds]
                dev_vectors[name][p] = vector[1:].lower()    # strip leading '@'
                prints.append(f"print {vector}")

    lines.extend([".control", "op", *prints, ".endc", ".end"])
    with tempfile.TemporaryDirectory(prefix="circuitopt-pvt-pwr-") as td:
        deck = os.path.join(td, "deck.cir")
        with open(deck, "w", encoding="ascii") as fh:
            fh.write("\n".join(lines) + "\n")
        txt = _run_ngspice_capture(
            deck, timeout=900.0, what="MDAC PVT power .op",
            extra_args=adapter.command_args if adapter is not None else ())
    m = re.search(r"i\(vrail_vdd\)\s*=\s*([-+0-9.eE]+)", txt)
    if not m:
        raise RuntimeError("could not read i(vrail_vdd) from op deck")
    isupply = abs(float(m.group(1)))

    if not merge:
        return isupply, None

    # ngspice prints a scalar as "@m.xm0.main[vds] = 3.5e-01"
    pat = re.compile(r"@([^\s=]+)\s*=\s*([-+0-9.eEnaN]+)")
    raw = {}
    for mo in pat.finditer(txt):
        try:
            raw[mo.group(1).lower()] = float(mo.group(2))
        except ValueError:
            continue
    static_regions = {}
    for name, vecs in dev_vectors.items():
        vds = raw.get(vecs["vds"])
        vdsat = raw.get(vecs["vdsat"])
        if vds is None or vdsat is None:
            # A device whose op-vars ngspice did not report cannot be certified in
            # region; mark it failing so the merge is never silently more lenient
            # than the standalone op_ngspice path (which omits such devices).
            static_regions[name] = False
        else:
            static_regions[name] = bool(abs(vds) >= abs(vdsat))
    return isupply, static_regions


def _ac_power_static_merged(spec, dk, corner, seed, *, acmag, fstart, fstop, points,
                            out_nodes, core_devs, timeout=900.0):
    """ONE ngspice process for run_point sections (a) + (h/f-static).

    Both measurements operate on the same ``build_ac`` testbench, so the deck is
    rendered once (AC-stimulus sources included — an ``ac`` magnitude on a source
    does not perturb the operating point) and the control block chains ``op``
    (power + static-saturation ``print``\\ s, exactly as :func:`_supply_current`
    emits them) with the ``ac dec`` sweep + ``wrdata`` exactly as
    :func:`~circuitopt.ngspice_ac.ac_ngspice` emits them.  stdout carries the op
    prints; the wrdata file carries the sweep.  On a foundry process this halves
    the macro-expansion cost of the point's build_ac work.

    Returns ``(ac_result, isupply_A, static_regions)`` where ``ac_result`` has
    :func:`~circuitopt.ngspice_ac.ac_ngspice`'s exact shape and ``isupply_A`` /
    ``static_regions`` follow :func:`_supply_current` (``static_regions`` is
    ``None`` off the adapter path, and the caller falls back to
    :func:`~circuitopt.ngspice_ac.op_ngspice`)."""
    topo = spec.topology
    b = spec.binding()
    acmag = {k: (float(v[0]), float(v[1])) for k, v in dict(acmag or {}).items()}
    ac_stimulus = {_resolve_source_name(topo, k): v for k, v in acmag.items()}
    record = list(out_nodes) if out_nodes is not None else list(topo.solved)
    for name in record:
        if name not in topo.idx:
            raise ValueError(f"ac out node {name!r} is not a solved node")

    lines, node_map, _node, adapter = _network_deck(
        topo, spec.sizes, spec.bias, header="* circuitopt pvt merged power .op + .ac",
        nf=spec.nf, model_types=b.model_types, device_kwargs=dk, corner=corner,
        temperature=None, x0_guess=seed, ac=ac_stimulus)

    merge = bool(core_devs) and adapter is not None
    dev_vectors = {}       # device -> {"vds": vec_no_at, "vdsat": vec_no_at}
    prints = ["print i(vrail_vdd)"]
    if merge:
        for name in core_devs:
            elem = _element("X", name).lower()
            dev_vectors[name] = {}
            for p in ("vds", "vdsat"):
                vector = adapter.op_vector(elem, p)          # e.g. @m.xm0.main[vds]
                dev_vectors[name][p] = vector[1:].lower()    # strip leading '@'
                prints.append(f"print {vector}")

    vecs = []
    for name in record:
        vecs.append(f"real(v({node_map[name]}))")
        vecs.append(f"imag(v({node_map[name]}))")
    with tempfile.TemporaryDirectory(prefix="circuitopt-pvt-acop-") as td:
        out_path = os.path.join(td, "ac.dat")
        deck = os.path.join(td, "deck.cir")
        lines.extend([
            ".control", "set filetype=ascii", "set wr_singlescale", "set wr_vecnames",
            "op", *prints,
            f"ac dec {int(points):d} {float(fstart):.17g} {float(fstop):.17g}",
            f"wrdata {out_path} " + " ".join(vecs),
            ".endc", ".end",
        ])
        with open(deck, "w", encoding="ascii") as fh:
            fh.write("\n".join(lines) + "\n")
        txt = _run_ngspice_capture(
            deck, timeout=timeout, what="MDAC PVT merged .op+.ac",
            extra_args=adapter.command_args if adapter is not None else ())
        if not os.path.exists(out_path):
            raise RuntimeError("merged .op+.ac produced no AC sweep output")
        raw_ac = np.loadtxt(out_path, skiprows=1, ndmin=2)

    m = re.search(r"i\(vrail_vdd\)\s*=\s*([-+0-9.eE]+)", txt)
    if not m:
        raise RuntimeError("could not read i(vrail_vdd) from merged .op+.ac deck")
    isupply = abs(float(m.group(1)))

    static_regions = None
    if merge:
        # ngspice prints a scalar as "@m.xm0.main[vds] = 3.5e-01"
        pat = re.compile(r"@([^\s=]+)\s*=\s*([-+0-9.eEnaN]+)")
        raw = {}
        for mo in pat.finditer(txt):
            try:
                raw[mo.group(1).lower()] = float(mo.group(2))
            except ValueError:
                continue
        static_regions = {}
        for name, vecs_dev in dev_vectors.items():
            vds = raw.get(vecs_dev["vds"])
            vdsat = raw.get(vecs_dev["vdsat"])
            if vds is None or vdsat is None:
                static_regions[name] = False
            else:
                static_regions[name] = bool(abs(vds) >= abs(vdsat))

    freq = raw_ac[:, 0]
    nodes = {}
    for i, name in enumerate(record):
        re_col = raw_ac[:, 1 + 2 * i]
        im_col = raw_ac[:, 2 + 2 * i]
        nodes[name] = re_col + 1j * im_col
    return {"freq": freq, "nodes": nodes, "acmag": acmag}, isupply, static_regions


def _ac_power_static(spec, dk, corner, seed, *, acmag, fstart, fstop, points,
                     out_nodes, core_devs):
    """run_point sections (a) + (h/f-static): dispatch on the chaining toggle.

    Chained (:func:`~circuitopt.ngspice_char.ngspice_chain_enabled`, read at call
    time): one merged process via :func:`_ac_power_static_merged`.  Unchained:
    exactly today's two separate calls, through the module-global
    ``ac_ngspice`` / ``_supply_current`` names so campaign specialisations that
    rebind them (e.g. the TSMC28 timeout wrapper) keep working."""
    if not ngspice_chain_enabled():
        b = spec.binding()
        ac = ac_ngspice(spec.sizes, spec.bias, topo=spec.topology, acmag=acmag,
                        fstart=fstart, fstop=fstop, points=points,
                        out_nodes=out_nodes, nf=spec.nf, model_types=b.model_types,
                        device_kwargs=dk, corner=corner, x0_guess=seed)
        isup, static_regions = _supply_current(spec, dk, corner, seed,
                                               core_devs=core_devs)
        return ac, isup, static_regions
    return _ac_power_static_merged(spec, dk, corner, seed, acmag=acmag,
                                   fstart=fstart, fstop=fstop, points=points,
                                   out_nodes=out_nodes, core_devs=core_devs)


def _transient_batch_loop(spec_args, cases, **shared_kwargs):
    """Per-case fallback: one :func:`transient_ngspice` process per case, through
    the module-global name so per-call wrappers (e.g. the TSMC28 override that
    adds hold clocks / op_devices and records ``_tls.transients``) keep working."""
    return [transient_ngspice(*spec_args, **{**shared_kwargs, **case})
            for case in cases]


def transient_batch(spec_args, cases, **shared_kwargs):
    """Run the residue-transient cases, chained into one process when enabled.

    ``spec_args`` is ``(sizes, bias, tgrid)``; ``cases`` is a sequence of
    ``{"inputs": {...}}`` dicts in measurement order; every other kwarg is
    shared.  With chaining off this loops the module-global
    ``transient_ngspice`` per case (byte-for-byte today's behaviour, wrappers
    included); with chaining on it runs ONE ngspice process via
    :func:`~circuitopt.ngspice_transient.transient_ngspice_chain`.  A campaign
    specialisation overrides this hook the same way it overrides
    ``transient_ngspice``."""
    if not ngspice_chain_enabled():
        return _transient_batch_loop(spec_args, cases, **shared_kwargs)
    return transient_ngspice_chain(*spec_args, cases=cases, **shared_kwargs)


def run_point(corner, temp_c, vdd):
    """Measure every campaign spec at one PVT point.  Returns a CSV row dict."""
    tk = temp_c + 273.15
    h = vdd / 2.0
    row = {"corner": corner, "temp_c": temp_c, "vdd": vdd}

    # ── (a) open-loop differential AC + (h) power + (f-static) saturation ─────
    # All three operate on the same build_ac testbench.  The power .op harvests
    # each core device's vds/vdsat (adapter processes), so the static saturation
    # check rides along instead of paying a second full foundry-macro expansion;
    # with chaining enabled the .op and the .ac sweep share ONE ngspice process
    # too (the AC stimulus does not perturb the operating point).  FreePDK45
    # (no adapter) falls back to a separate op_ngspice call below.
    spec = circuit_from_dict(G.build_ac(vdd))
    b, dk = _dk(spec, tk)
    ac, isup, static_regions = _ac_power_static(
        spec, dk, corner, spec.topology.dc_guesses[0],
        acmag={"VACP": (0.5, 0.0), "VACN": (0.5, 180.0)},
        fstart=1e4, fstop=5e10, points=25, out_nodes=["OUTP", "OUTN"],
        core_devs=CORE_DEVS)
    H = ac_response(ac, "OUTP", "OUTN", vin=1.0)
    row["gain_db"] = 20.0 * np.log10(abs(H[0]))          # 10 kHz plateau
    row["ac_ugbw_hz"] = unity_gain_freq(ac["freq"], H)
    row["ac_pm_deg"] = phase_margin(ac["freq"], H)
    row["isupply_ma"] = isup * 1e3
    row["power_mw"] = isup * vdd * 1e3

    if static_regions is None:
        op0 = op_ngspice(spec.sizes, spec.bias, topo=spec.topology, margin=0.0,
                         nf=spec.nf, model_types=b.model_types, device_kwargs=dk,
                         corner=corner, x0_guess=spec.topology.dc_guesses[0])
        bad_static = [m for m in CORE_DEVS if not op0.get(m, {}).get("region_ok", False)]
    else:
        bad_static = [m for m in CORE_DEVS if not static_regions.get(m, False)]

    # ── (b) DM loop (plateau above the ~30 kHz Cs/Cf corner -> fstart 1e5) ────
    spec = circuit_from_dict(G.build_dmloop(vdd))
    b, dk = _dk(spec, tk)
    lg = loop_gain_ngspice(spec.sizes, spec.bias, topo=spec.topology, inject="Vinj",
                           fstart=1e5, fstop=2e10, points=20, nf=spec.nf,
                           model_types=b.model_types, device_kwargs=dk,
                           corner=corner, x0_guess=spec.topology.dc_guesses[0])
    row["dm_ugf_hz"] = lg["ugf"]
    row["dm_pm_deg"] = lg["pm"]
    row["dm_gm_db"] = lg["gm_db"]

    # ── (c) CMFB1 / CMFB2 (DC plateau -> fstart 1e3) ─────────────────────────
    spec = circuit_from_dict(G.build_cmfb1(vdd))
    b, dk = _dk(spec, tk)
    lg1 = loop_gain_ngspice(spec.sizes, spec.bias, topo=spec.topology, inject="Vinj",
                            fstart=1e3, fstop=2e10, points=20, nf=spec.nf,
                            model_types=b.model_types, device_kwargs=dk,
                            corner=corner, x0_guess=spec.topology.dc_guesses[0])
    row["cmfb1_ugf_hz"] = lg1["ugf"]
    row["cmfb1_pm_deg"] = lg1["pm"]

    spec = circuit_from_dict(G.build_cmfb2(vdd))
    b, dk = _dk(spec, tk)
    lg2 = loop_gain_ngspice(spec.sizes, spec.bias, topo=spec.topology, inject="Vinj",
                            fstart=1e3, fstop=2e10, points=20, nf=spec.nf,
                            model_types=b.model_types, device_kwargs=dk,
                            corner=corner, x0_guess=spec.topology.dc_guesses[0])
    row["cmfb2_ugf_hz"] = lg2["ugf"]
    row["cmfb2_pm_deg"] = lg2["pm"]

    # ── (d) closed-loop residue transients + (e) static CM ───────────────────
    spec = circuit_from_dict(G.build_transient(vdd))
    b, dk = _dk(spec, tk)
    seed = spec.topology.dc_guesses[0]
    V0 = np.array([seed.get(n, 0.0) for n in spec.topology.solved])
    n = 101
    tg = np.linspace(0.0, 5e-9, n)
    settle_pct = {}
    overshoot_worst = 0.0
    cm5_worst = 0.0
    cm_static_mv = None
    settled_finals = {}       # s -> {node: V(5ns)} for the +-FS/16 saturation op
    keys = {-FS / 16: "settle_n2_pct", -FS / 32: "settle_n1_pct", 0.0: "settle_z_pct",
            FS / 32: "settle_p1_pct", FS / 16: "settle_p2_pct"}
    cases = []
    for s in RESIDUE_LEVELS:
        bp1 = np.full(n, h + s / 2); bp1[0] = h
        bp2 = np.full(n, h - s / 2); bp2[0] = h
        cases.append({"inputs": {"bp1": bp1, "bp2": bp2}})
    # One case per residue level, RESIDUE_LEVELS order; only bp1/bp2 differ between
    # levels, so with chaining on the batch hook folds all five into ONE ngspice
    # process (with chaining off it loops transient_ngspice per case as before).
    residues = transient_batch(
        (spec.sizes, spec.bias, tg), cases, topo=spec.topology, nf=spec.nf,
        model_types=b.model_types, device_kwargs=dk, corner=corner, V0=V0,
        extra_options=TIGHT, max_step=0.05e-9)
    for s, r in zip(RESIDUE_LEVELS, residues):
        vop, von = r["nodes"]["OUTP"], r["nodes"]["OUTN"]
        vod = vop - von
        ideal = -8.0 * s
        err_pct = abs(vod[-1] - ideal) / 0.45 * 100.0
        settle_pct[keys[s]] = err_pct
        if cm_static_mv is None:
            cm_static_mv = abs((vop[0] + von[0]) / 2.0 - h) * 1e3
        cm5_worst = max(cm5_worst, abs((vop[-1] + von[-1]) / 2.0 - h) * 1e3)
        if s != 0.0:
            dirn = np.sign(ideal)
            ov = float(np.max(dirn * (vod - ideal))) / abs(ideal) * 100.0
            overshoot_worst = max(overshoot_worst, max(ov, 0.0))
        if abs(s) == FS / 16:
            settled_finals[s] = {node: float(r["nodes"][node][-1])
                                 for node in spec.topology.solved if node in r["nodes"]}
    row.update(settle_pct)
    row["settle_worst_pct"] = max(settle_pct.values())
    row["overshoot_worst_pct"] = overshoot_worst
    row["cm5_worst_mv"] = cm5_worst
    row["cm_static_mv"] = cm_static_mv

    # ── (f-settled) saturation after settling at s = +-FS/16 ─────────────────
    bad_settled = []
    for s in (-FS / 16, FS / 16):
        d = G.build_transient(vdd)
        d["vsources"] = [["VBP1", "BP1", "GND", h + s / 2],
                         ["VBP2", "BP2", "GND", h - s / 2]]
        sp2 = circuit_from_dict(d)
        b2, dk2 = _dk(sp2, tk)
        op = op_ngspice(sp2.sizes, sp2.bias, topo=sp2.topology, margin=0.0,
                        nf=sp2.nf, model_types=b2.model_types, device_kwargs=dk2,
                        corner=corner, x0_guess=settled_finals[s])
        bad_settled += [f"{m}@{s*1e3:+.0f}mV" for m in CORE_DEVS
                        if not op.get(m, {}).get("region_ok", False)]

    # ── (g) closed-loop noise (skipped in --smoke: slowest per-point run) ─────
    if SKIP_NOISE:
        row["noise_onoise_uv"] = float("nan")
        row["noise_adc_uv"] = float("nan")
    else:
        spec = circuit_from_dict(G.build_noise(vdd))
        b, dk = _dk(spec, tk)
        nz = noise_ngspice(spec.sizes, spec.bias, topo=spec.topology, out="OUTP",
                           ref="OUTN", src="VBP1", fstart=1e7, fstop=2e10, points=20,
                           band=(1e7, 2e10), nf=spec.nf, model_types=b.model_types,
                           device_kwargs=dk, corner=corner,
                           x0_guess=spec.topology.dc_guesses[0])
        row["noise_onoise_uv"] = nz["onoise_rms"] * 1e6
        row["noise_adc_uv"] = nz["onoise_rms"] / 8.0 * 1e6

    # ── saturation bookkeeping + pass flags ──────────────────────────────────
    row["sat_static_ok"] = not bad_static
    row["sat_settled_ok"] = not bad_settled
    row["sat_bad"] = ";".join(bad_static + bad_settled)

    row["pass_gain"] = bool(row["gain_db"] > SPEC_GAIN_DB)
    row["pass_dmpm"] = bool(np.isfinite(row["dm_pm_deg"]) and row["dm_pm_deg"] > SPEC_PM_DEG)
    row["pass_cmfb1pm"] = bool(np.isfinite(row["cmfb1_pm_deg"])
                               and np.isfinite(row["cmfb1_ugf_hz"])
                               and row["cmfb1_pm_deg"] > SPEC_PM_DEG)
    row["pass_cmfb2pm"] = bool(np.isfinite(row["cmfb2_pm_deg"])
                               and np.isfinite(row["cmfb2_ugf_hz"])
                               and row["cmfb2_pm_deg"] > SPEC_PM_DEG)
    row["pass_settle"] = bool(row["settle_worst_pct"] < SPEC_SETTLE_PCT)
    row["pass_cm"] = bool(row["cm_static_mv"] < SPEC_CM_MV)
    row["pass_sat"] = bool(row["sat_static_ok"] and row["sat_settled_ok"])
    # Skipped in smoke mode: noise never ran, so it cannot sign off — flag False and
    # exclude it from pass_all (see the smoke handling below).
    row["pass_noise"] = bool((not SKIP_NOISE)
                             and row["noise_onoise_uv"] <= SPEC_NOISE_UV)
    # pass_all covers only the specs that were actually MEASURED: in smoke mode the
    # skipped specs (noise here; code-transition in the TSMC28 override) are omitted
    # so pass_all reflects the measured subset honestly rather than a false green.
    measured = ["pass_gain", "pass_dmpm", "pass_cmfb1pm", "pass_cmfb2pm",
                "pass_settle", "pass_cm", "pass_sat"]
    if not SKIP_NOISE:
        measured.append("pass_noise")
    row["pass_all"] = all(bool(row[k]) for k in measured)
    return row


# ── CSV / resumability ──────────────────────────────────────────────────────────
def _key(corner, temp_c, vdd):
    return f"{corner}/{temp_c:g}/{vdd:g}"


def _parse_points(spec: str, grid):
    """Parse a ``--points`` string ("corner/temp/vdd[,...]") into grid tuples.

    Each token uses the same ``_key`` layout (``corner/temp_c/vdd``).  Every token
    must resolve to a point that exists in ``grid``; a malformed or unknown token
    raises ``ValueError`` so a typo fails loudly instead of silently running nothing.
    Returns the points in the order given, de-duplicated."""
    grid_keys = {_key(*p): p for p in grid}
    out = []
    seen = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split("/")
        if len(parts) != 3:
            raise ValueError(
                f"--points token {token!r} must be 'corner/temp/vdd'")
        corner, temp_s, vdd_s = parts
        try:
            key = _key(corner, float(temp_s), float(vdd_s))
        except ValueError:
            raise ValueError(f"--points token {token!r} has a non-numeric temp/vdd")
        if key not in grid_keys:
            raise ValueError(
                f"--points token {token!r} is not a grid point "
                f"(corners={CORNERS}, temps={TEMPS_C}, supplies={SUPPLIES})")
        if key not in seen:
            seen.add(key)
            out.append(grid_keys[key])
    if not out:
        raise ValueError("--points did not resolve to any grid point")
    return out


def _errors_csv_path(out_path: Path) -> Path:
    """Sibling errors CSV: results/foo.csv -> results/foo.errors.csv."""
    return out_path.with_suffix(".errors.csv")


_ERROR_FIELDS = ["corner", "temp_c", "vdd", "error", "elapsed_s"]


def _append_error_row(path: Path, lock, corner, temp_c, vdd, error, elapsed_s):
    """Thread-safe append of one failed point to the errors CSV (header on create).

    The error string is collapsed to a single line so one failure never spills across
    CSV rows.  Flushed per row so a crash mid-campaign still leaves the record."""
    single_line = " ".join(str(error).splitlines()).strip()
    with lock:
        new_file = not path.is_file()
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_ERROR_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow({
                "corner": corner, "temp_c": _fmt(temp_c), "vdd": _fmt(vdd),
                "error": single_line, "elapsed_s": _fmt(float(elapsed_s)),
            })
            fh.flush()


def _load_done(path):
    done = {}
    if not path.is_file():
        return done
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            done[_key(r["corner"], float(r["temp_c"]), float(r["vdd"]))] = r
    return done


def _fmt(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "nan"
        return f"{v:.6g}"
    return v


def _order_todo(grid, done):
    """The points still to run, GRID_PRIORITY entries first (in that order), then
    the rest of the nested grid in its natural order.  Already-done points drop out.
    Priority entries not on the grid are ignored (a specialisation owns the list)."""
    grid_keys = {_key(*p): p for p in grid}
    ordered = []
    seen = set()
    for p in GRID_PRIORITY:
        key = _key(*p)
        if key in grid_keys and key not in seen:
            ordered.append(grid_keys[key])
            seen.add(key)
    for p in grid:
        key = _key(*p)
        if key not in seen:
            ordered.append(p)
            seen.add(key)
    return [p for p in ordered if _key(*p) not in done]


def main():
    global SKIP_CODE_TRANSITION, SKIP_NOISE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--force", action="store_true", help="ignore existing CSV rows")
    ap.add_argument("--points", type=str, default=None,
                    help="run exactly these grid points, e.g. "
                         "'ss/125/0.85,tt/27/0.9'")
    ap.add_argument("--smoke", action="store_true",
                    help="run only the priority points and skip the two slowest "
                         "per-point measurements (code transition + noise); those "
                         "specs are NOT signed off")
    args = ap.parse_args()

    if args.smoke and args.points:
        ap.error("--smoke and --points are mutually exclusive")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {} if args.force else _load_done(args.out)
    grid = [(c, t, v) for c in CORNERS for t in TEMPS_C for v in SUPPLIES]

    if args.smoke:
        SKIP_CODE_TRANSITION = True
        SKIP_NOISE = True
        print("SMOKE MODE: skipping code-transition and noise; those specs are NOT "
              "signed off (pass_all reflects only the measured specs).")
        # Restrict to the priority points; if none set, fall back to the full grid
        # so --smoke alone still exercises every measured spec once per point.
        prio = [p for p in GRID_PRIORITY if _key(*p) in {_key(*g) for g in grid}]
        smoke_grid = prio if prio else grid
        todo = _order_todo(smoke_grid, done)
    elif args.points:
        selected = _parse_points(args.points, grid)
        todo = _order_todo(selected, done)
    else:
        todo = _order_todo(grid, done)

    print(f"campaign: {len(grid)} points, {len(done)} already done, {len(todo)} to run, "
          f"{args.workers} workers")

    new_file = not args.out.is_file() or args.force
    lock = threading.Lock()
    errors_path = _errors_csv_path(args.out)
    fh = open(args.out, "w" if args.force else "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader(); fh.flush()

    results = list(done.values())
    fails = []

    def _timed_run_point(corner, temp_c, vdd):
        start = time.time()
        try:
            return run_point(corner, temp_c, vdd), time.time() - start
        except Exception as exc:                           # noqa: BLE001
            raise RunPointError(time.time() - start) from exc

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_timed_run_point, *p): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c, t, v = futs[fut]
            try:
                row, _elapsed = fut.result()
            except RunPointError as wrapped:               # noqa: BLE001
                exc = wrapped.__cause__
                print(f"[{i}/{len(todo)}] {_key(c, t, v):16s}  ERROR: {exc}")
                fails.append((_key(c, t, v), str(exc)))
                _append_error_row(errors_path, lock, c, t, v, exc, wrapped.elapsed_s)
                continue
            with lock:
                writer.writerow({k: _fmt(row[k]) for k in CSV_FIELDS})
                fh.flush()
            results.append(row)
            tag = "PASS" if row["pass_all"] else "FAIL"
            print(f"[{i}/{len(todo)}] {_key(c, t, v):16s}  {tag}  "
                  f"gain={float(row['gain_db']):.1f}dB DMpm={float(row['dm_pm_deg']):.0f} "
                  f"settle={float(row['settle_worst_pct']):.3f}% CM={float(row['cm_static_mv']):.1f}mV "
                  f"noise={float(row['noise_onoise_uv']):.0f}uV P={float(row['power_mw']):.1f}mW")
    fh.close()
    _summary(results, fails)


def _f(row, k):
    return float(row[k])


def _pass(row, k):
    v = row[k]
    return v in (True, "1", 1)


def _summary(results, fails):
    print("\n" + "=" * 78)
    print(f"SUMMARY  ({len(results)} points)")
    print("=" * 78)
    if not results:
        return
    npass = sum(_pass(r, "pass_all") for r in results)
    print(f"OVERALL: {npass}/{len(results)} points PASS all specs")

    def worst(metric, cmp, fmt):
        best = None
        for r in results:
            val = _f(r, metric)
            if not np.isfinite(val):
                continue
            if best is None or cmp(val, best[0]):
                best = (val, r)
        if best is None:
            return
        val, r = best
        print(f"  {metric:22s} worst {fmt.format(val):>10s}  "
              f"@ {r['corner']}/{_f(r, 'temp_c'):g}C/{_f(r, 'vdd'):g}V")

    print("per-spec worst point:")
    worst("gain_db", lambda a, b: a < b, "{:.1f}dB")
    worst("dm_pm_deg", lambda a, b: a < b, "{:.0f}deg")
    worst("cmfb1_pm_deg", lambda a, b: a < b, "{:.0f}deg")
    worst("cmfb2_pm_deg", lambda a, b: a < b, "{:.0f}deg")
    worst("settle_worst_pct", lambda a, b: a > b, "{:.3f}%")
    if "settle_time_worst_ns" in results[0]:
        worst("settle_time_worst_ns", lambda a, b: a > b, "{:.2f}ns")
    worst("cm_static_mv", lambda a, b: a > b, "{:.1f}mV")
    worst("noise_onoise_uv", lambda a, b: a > b, "{:.0f}uV")
    worst("power_mw", lambda a, b: a > b, "{:.1f}mW")
    worst("cm5_worst_mv", lambda a, b: a > b, "{:.1f}mV")
    if "code_transition_pct" in results[0]:
        worst("code_transition_pct", lambda a, b: a > b, "{:.3f}%")
        worst("code_settle_ns", lambda a, b: a > b, "{:.2f}ns")
        worst("code_peak_glitch_pct", lambda a, b: a > b, "{:.1f}%")
        if "code_cm5_mv" in results[0]:
            worst("code_cm5_mv", lambda a, b: a > b, "{:.1f}mV")
    if fails:
        print(f"\n{len(fails)} points ERRORED:")
        for k, e in fails:
            print(f"  {k}: {e}")
    nfail = [r for r in results if not _pass(r, "pass_all")]
    if nfail:
        print(f"\n{len(nfail)} points FAIL a spec:")
        for r in nfail:
            flags = [k for k in CSV_FIELDS if k.startswith("pass_") and k != "pass_all"
                     and not _pass(r, k)]
            print(f"  {r['corner']}/{_f(r, 'temp_c'):g}C/{_f(r, 'vdd'):g}V: {flags}  "
                  f"sat_bad={r['sat_bad']}")


if __name__ == "__main__":
    main()
