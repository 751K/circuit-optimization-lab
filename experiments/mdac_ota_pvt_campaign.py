#!/usr/bin/env python3
"""Full 45-point PVT verification campaign for the FreePDK45 MDAC first-stage OTA.

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

    .venv/bin/python experiments/mdac_ota_pvt_campaign.py [--workers 8] [--force]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import mdac_ota_gen as G  # noqa: E402
from circuitopt.circuit_loader import circuit_from_dict  # noqa: E402
from circuitopt.ngspice_ac import (  # noqa: E402
    _network_deck,
    _run_ngspice_capture,
    ac_ngspice,
    ac_response,
    loop_gain_ngspice,
    noise_ngspice,
    op_ngspice,
    phase_margin,
    unity_gain_freq,
)
from circuitopt.ngspice_transient import transient_ngspice  # noqa: E402

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


def _supply_current(spec, dk, corner, seed):
    """Static supply current |i(Vrail_VDD)| [A] from a full-circuit .op deck."""
    b = spec.binding()
    lines, _nm, _node = _network_deck(
        spec.topology, spec.sizes, spec.bias, header="* circuitopt pvt power .op",
        nf=spec.nf, model_types=b.model_types, device_kwargs=dk, corner=corner,
        temperature=None, x0_guess=seed)
    lines.extend([".control", "op", "print i(vrail_vdd)", ".endc", ".end"])
    with tempfile.TemporaryDirectory(prefix="circuitopt-pvt-pwr-") as td:
        deck = os.path.join(td, "deck.cir")
        with open(deck, "w", encoding="ascii") as fh:
            fh.write("\n".join(lines) + "\n")
        txt = _run_ngspice_capture(deck, timeout=120.0, what="FreePDK45 pvt power .op")
    m = re.search(r"i\(vrail_vdd\)\s*=\s*([-+0-9.eE]+)", txt)
    if not m:
        raise RuntimeError("could not read i(vrail_vdd) from op deck")
    return abs(float(m.group(1)))


def run_point(corner, temp_c, vdd):
    """Measure every campaign spec at one PVT point.  Returns a CSV row dict."""
    tk = temp_c + 273.15
    h = vdd / 2.0
    row = {"corner": corner, "temp_c": temp_c, "vdd": vdd}

    # ── (a) open-loop differential AC ────────────────────────────────────────
    spec = circuit_from_dict(G.build_ac(vdd))
    b, dk = _dk(spec, tk)
    ac = ac_ngspice(spec.sizes, spec.bias, topo=spec.topology,
                    acmag={"VACP": (0.5, 0.0), "VACN": (0.5, 180.0)},
                    fstart=1e4, fstop=5e10, points=25, out_nodes=["OUTP", "OUTN"],
                    nf=spec.nf, model_types=b.model_types, device_kwargs=dk,
                    corner=corner, x0_guess=spec.topology.dc_guesses[0])
    H = ac_response(ac, "OUTP", "OUTN", vin=1.0)
    row["gain_db"] = 20.0 * np.log10(abs(H[0]))          # 10 kHz plateau
    row["ac_ugbw_hz"] = unity_gain_freq(ac["freq"], H)
    row["ac_pm_deg"] = phase_margin(ac["freq"], H)

    # ── (f-static) saturation at the static bias point ───────────────────────
    op0 = op_ngspice(spec.sizes, spec.bias, topo=spec.topology, margin=0.0,
                     nf=spec.nf, model_types=b.model_types, device_kwargs=dk,
                     corner=corner, x0_guess=spec.topology.dc_guesses[0])
    bad_static = [m for m in CORE_DEVS if not op0.get(m, {}).get("region_ok", False)]

    # ── (h) power from the static .op ────────────────────────────────────────
    isup = _supply_current(spec, dk, corner, spec.topology.dc_guesses[0])
    row["isupply_ma"] = isup * 1e3
    row["power_mw"] = isup * vdd * 1e3

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
    for s in RESIDUE_LEVELS:
        bp1 = np.full(n, h + s / 2); bp1[0] = h
        bp2 = np.full(n, h - s / 2); bp2[0] = h
        r = transient_ngspice(spec.sizes, spec.bias, tg, topo=spec.topology,
                              nf=spec.nf, model_types=b.model_types,
                              device_kwargs=dk, corner=corner, V0=V0,
                              inputs={"bp1": bp1, "bp2": bp2},
                              extra_options=TIGHT, max_step=0.05e-9)
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

    # ── (g) closed-loop noise ────────────────────────────────────────────────
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
    row["pass_noise"] = bool(row["noise_onoise_uv"] <= SPEC_NOISE_UV)
    row["pass_all"] = bool(row["pass_gain"] and row["pass_dmpm"] and row["pass_cmfb1pm"]
                           and row["pass_cmfb2pm"] and row["pass_settle"] and row["pass_cm"]
                           and row["pass_sat"] and row["pass_noise"])
    return row


# ── CSV / resumability ──────────────────────────────────────────────────────────
def _key(corner, temp_c, vdd):
    return f"{corner}/{temp_c:g}/{vdd:g}"


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--force", action="store_true", help="ignore existing CSV rows")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {} if args.force else _load_done(args.out)
    grid = [(c, t, v) for c in CORNERS for t in TEMPS_C for v in SUPPLIES]
    todo = [p for p in grid if _key(*p) not in done]
    print(f"campaign: {len(grid)} points, {len(done)} already done, {len(todo)} to run, "
          f"{args.workers} workers")

    new_file = not args.out.is_file() or args.force
    lock = threading.Lock()
    fh = open(args.out, "w" if args.force else "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader(); fh.flush()

    results = list(done.values())
    fails = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_point, *p): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c, t, v = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:                       # noqa: BLE001
                print(f"[{i}/{len(todo)}] {_key(c, t, v):16s}  ERROR: {exc}")
                fails.append((_key(c, t, v), str(exc)))
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
    worst("cm_static_mv", lambda a, b: a > b, "{:.1f}mV")
    worst("noise_onoise_uv", lambda a, b: a > b, "{:.0f}uV")
    worst("power_mw", lambda a, b: a > b, "{:.1f}mW")
    worst("cm5_worst_mv", lambda a, b: a > b, "{:.1f}mV")
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
