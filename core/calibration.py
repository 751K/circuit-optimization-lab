"""Cadence/Spectre calibration engine.

Closes the loop between the local solver suite and Spectre reference data: load a
calibration case (``metadata.json`` + co-located PSFASCII reference files), run the
matching local analyses with the *same* sizes/bias, compare each metric against a
per-case tolerance, and emit a structured pass/fail report.

A case directory looks like::

    calibration/amp_design3_typical/
        metadata.json      provenance + circuit + analyses + tolerances
        dcOp.dc  ac.ac  noiseAnal.noise        (Spectre PSFASCII, from cadence_netlist)

``metadata.json`` schema (the engine reads only what it needs)::

    {
      "case": "...", "description": "...",
      "testbench": "amp" | "chopper",
      "corner": "typical" | "slow" | "fast",
      "circuit": {"sizes": {M: [W,L]}, "bias": {...}, "nf": 1, "f_chop": 225.0},
      "analyses": ["dc", "ac", "noise"],          # or ["pss","pac","pnoise"]
      "reference_files": {"dc": "dcOp.dc", ...},
      "tolerances": { ... }                        # overrides the defaults below
    }

The CLI (``python -m core.calibration <case>/``) and ``tests/test_calibration.py``
drive this module; see :mod:`core.psf` for the parsers and
:mod:`core.cadence_netlist` for generating the reference netlists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import psf
    from .ac_solver import ac_solve
    from .noise_solver import band_rms, noise_analysis
    from .topology import AFE_TOPO
except ImportError:  # pragma: no cover - legacy direct module import
    import psf
    from ac_solver import ac_solve
    from noise_solver import band_rms, noise_analysis
    from topology import AFE_TOPO

# Integrated input-referred-noise band (Hz), matching the AFE spec / Cadence ViVA.
_IRN_BAND = (0.05, 100.0)

DEFAULT_TOL = {
    "dc_v_atol": 1e-3,        # node voltage absolute error (V)
    "ac_gain_rtol": 0.01,     # DC gain (dB) relative error
    "ac_bw_rtol": 0.05,       # -3 dB bandwidth relative error
    "noise_irn_rtol": 0.03,   # integrated IRN relative error
    "pac_gain_rtol": 0.02,    # PAC baseband conversion gain
    "pnoise_irn_rtol": 0.03,  # PNoise integrated IRN
    "pac_bw_rtol": 0.05,      # PAC -3 dB bandwidth (SC-LPF single-ended LPTV)
    "pnoise_out_rtol": 0.05,  # PNoise integrated *output* noise (SC-LPF)
}


# ── small helpers ────────────────────────────────────────────────────────────

def _bw_interp(f, mag):
    """-3 dB low-pass bandwidth via log-log interpolation of the peak/√2 crossing."""
    f = np.asarray(f, float)
    mag = np.asarray(mag, float)
    thr = mag.max() / np.sqrt(2.0)
    ip = int(np.argmax(mag))
    for i in range(ip + 1, len(mag)):
        if mag[i] < thr:
            return float(10.0 ** np.interp(np.log10(thr),
                                           [np.log10(mag[i]), np.log10(mag[i - 1])],
                                           [np.log10(f[i]), np.log10(f[i - 1])]))
    return float(f[-1])


def _metric(local, ref, tol, *, rel=True):
    """One comparison row: absolute or relative delta against a tolerance."""
    local = float(local)
    ref = float(ref)
    if rel:
        denom = abs(ref) if abs(ref) > 1e-300 else 1.0
        delta = (local - ref) / denom
    else:
        delta = local - ref
    return {"local": local, "ref": ref, "delta": float(delta), "rel": bool(rel),
            "tol": float(tol), "pass": bool(abs(delta) <= tol)}


def _sizes(metadata):
    return {k: tuple(v) for k, v in metadata["circuit"]["sizes"].items()}


# ── reference loading ────────────────────────────────────────────────────────

def load_reference(case_dir) -> dict:
    """Load ``metadata.json`` and every referenced PSF file (raw parser output),
    plus the provenance pulled from the first PSF HEADER."""
    case_dir = Path(case_dir)
    metadata = json.loads((case_dir / "metadata.json").read_text())
    parsers = {
        "dc": psf.parse_dc, "ac": psf.parse_ac, "noise": psf.parse_noise,
        "tran": psf.parse_tran, "pss": psf.parse_tran,
        "pac": psf.parse_pac, "pnoise": psf.parse_pnoise,
    }
    ref, prov = {}, None
    for key, fname in metadata.get("reference_files", {}).items():
        path = case_dir / fname
        ref[key] = parsers[key](str(path))
        if prov is None:
            prov = psf.provenance(str(path))
    return {"metadata": metadata, "ref": ref, "provenance": prov, "dir": case_dir}


# ── local solver runs ────────────────────────────────────────────────────────

def run_local(metadata, *, analyses=None) -> dict:
    """Run the local analyses named in ``metadata`` (or the subset ``analyses``)."""
    tb = metadata.get("testbench", "amp")
    want = set(analyses or metadata.get("analyses", []))
    sizes = _sizes(metadata)
    bias = dict(metadata["circuit"]["bias"])
    nf = metadata["circuit"].get("nf", 1)
    corner = metadata.get("corner", "typical")
    if tb == "amp":
        return _run_local_amp(sizes, bias, nf, corner, want)
    if tb == "chopper":
        return _run_local_chopper(sizes, bias, nf, corner, metadata, want)
    if tb == "sc_lpf":
        return _run_local_sc_lpf(metadata, want)
    raise ValueError(f"unknown testbench {tb!r}")


def _run_local_amp(sizes, bias, nf, corner, want):
    out = {}
    freqs = np.logspace(-2, 4, 121)             # match Cadence ac/noise 0.01–10k dec=20
    ac = ac_solve(sizes, bias, freqs, nf=nf, corner=corner)
    if "dc" in want:
        out["dc"] = dict(ac["dc_op"])
    if "ac" in want:
        out["ac"] = {"gain_dc_dB": float(ac["Av_dc_dB"]), "bw_Hz": float(ac["bw_Hz"])}
    if "noise" in want:
        nz = noise_analysis(sizes, bias, freqs, nf=nf, corner=corner)
        out["noise"] = {
            "irn_uVrms": float(band_rms(freqs, nz["irn_psd"], *_IRN_BAND) * 1e6),
        }
    return out


def _run_local_chopper(sizes, bias, nf, corner, metadata, want):
    """Chopper PSS/PAC/PNoise via the 8-PMOS wrapper. The PAC/PNoise *must* run on a
    properly-built gear2 PSS orbit (switch sizing, edge time, output RC filter, settling
    — all from ``metadata``), so the local result reproduces the validated solver call
    rather than the bare-default one. The orbit + PAC are shared across both analyses."""
    from .chopper import pmos_chopper_pac, pmos_chopper_pnoise, pmos_chopper_pss
    c = metadata["circuit"]
    s = metadata.get("solver", {})
    f_chop = float(c.get("f_chop", 225.0))
    adaptive_kwargs = {
        key: s[key] for key in (
            "adaptive", "adaptive_reltol", "adaptive_vabstol",
            "adaptive_iabstol", "adaptive_max_steps", "adaptive_h0",
            "adaptive_freeze_factor", "cap_mode",
        ) if key in s
    }
    pss = pmos_chopper_pss(
        sizes, bias, f_chop,
        switch_size=tuple(c.get("switch_size", (5000, 30))),
        switch_nf=int(c.get("switch_nf", 1)), nf=nf,
        edge_time=float(c.get("edge_time", 20e-6)), input_diff=0.0,
        input_common_mode=float(c.get("input_common_mode", bias["VCM"])),
        charge_injection=bool(c.get("charge_injection", False)),
        output_filter=tuple(c.get("output_filter", (1e6, 680e-12))),
        tstab_periods=int(s.get("tstab_periods", 2)),
        n_points=int(s.get("n_points", 321)),
        max_shooting_iters=int(s.get("max_shooting_iters", 5)),
        integration_method=s.get("integration_method", "gear2"),
        analytic_jacobian=bool(s.get("analytic_jacobian", False)),
        fallback_least_squares=False, corner=corner, **adaptive_kwargs)
    out = {}
    pac = (pmos_chopper_pac(sizes, bias, np.array([0.05, 200.0]), f_chop,
                            pss_result=pss, nf=nf, corner=corner)
           if {"pac", "pnoise"} & set(want) else None)
    if "pac" in want:
        out["pac"] = {"gain_baseband": float(pac["gains"][0])}
    if "pnoise" in want:
        band = tuple(c.get("noise_band", _IRN_BAND))
        freqs = np.logspace(np.log10(band[0]), np.log10(band[1]), 37)
        # Input-refer by the converged baseband conversion gain (scalar), the same
        # divisor used on the Cadence side — so the IRN comparison is apples-to-apples.
        pn = pmos_chopper_pnoise(sizes, bias, freqs, f_chop, pss_result=pss,
                                 gains=float(pac["gains"][0]), nf=nf, corner=corner,
                                 band=band)
        out["pnoise"] = {"irn_uVrms": float(pn.get("irn_uV_band", np.nan))}
    return out


def _sc_lpf_topology(c):
    """Build the SC-LPF Topology from the self-describing ``circuit`` block (so the
    engine stays independent of examples/). vsource clocks make it an n_aug circuit."""
    try:
        from .topology import Topology
    except ImportError:  # pragma: no cover
        from topology import Topology
    return Topology(
        devices=[tuple(d) for d in c["devices"]],
        vsources=[tuple(v) for v in c["vsources"]],
        capacitors=[tuple(x) for x in c["capacitors"]],
        rails={"GND": 0.0, "VDD": float(c.get("vdd", 40.0))},
        solved=list(c["solved"]),
        outputs=tuple(c["outputs"]),
    )


def _sc_lpf_clocks(c, tgrid):
    """Two-phase non-overlapping square clocks + DC input, matching examples/sc_lpf:
    trapezoidal edges (edge_time), duty, CLK2 delayed half a period."""
    period = 1.0 / float(c["f_clk"])
    vdd = float(c.get("vdd", 40.0))
    width = float(c.get("duty", 0.45)) * period
    rise = fall = float(c.get("edge_time", 2e-6))
    n = len(tgrid)

    def square(phase):
        out = np.zeros(n)
        for i in range(n):
            p = phase[i]
            if rise > 0 and p < rise:
                out[i] = vdd * p / rise
            elif p < width:
                out[i] = vdd
            elif fall > 0 and p < width + fall:
                out[i] = vdd * (1.0 - (p - width) / fall)
        return out

    return {"clk1": square(np.mod(tgrid, period)),
            "clk2": square(np.mod(tgrid - period / 2.0, period)),
            "vin": np.full(n, float(c.get("vin_dc", 20.0)))}


def _sc_lpf_adaptive_tgrid(c, n_points):
    period = 1.0 / float(c["f_clk"])
    base = np.linspace(0.0, period, int(n_points))
    width = float(c.get("duty", 0.45)) * period
    edge = float(c.get("edge_time", 2e-6))
    pts = [0.0, period]
    for shift in (0.0, 0.5 * period):
        for off in (0.0, edge, width, width + edge):
            pts.append(float(np.mod(shift + off, period)))
    out = np.unique(np.concatenate([base, np.asarray(pts, float)]))
    out = out[(out >= -1e-15) & (out <= period + 1e-15)]
    out[0] = 0.0
    if not np.isclose(out[-1], period, rtol=1e-12, atol=max(1e-18, period * 1e-12)):
        out = np.append(out, period)
    return out


def _run_local_sc_lpf(metadata, want):
    """Switched-capacitor LPF PSS/PAC/PNoise (single-ended LPTV). PSS converges on the
    signed-current device model + LM/best-physical shooting; PAC gives the baseband
    transfer (DC gain + -3 dB BW), PNoise the integrated output noise."""
    try:
        from .pac_solver import pac_solve
        from .pnoise_solver import pnoise_solve
        from .pss_solver import pss_solve
    except ImportError:  # pragma: no cover
        from pac_solver import pac_solve
        from pnoise_solver import pnoise_solve
        from pss_solver import pss_solve
    c = metadata["circuit"]
    s = metadata.get("solver", {})
    sizes = _sizes(metadata)
    topo = _sc_lpf_topology(c)
    f_clk = float(c["f_clk"])
    period = 1.0 / f_clk
    n_points = int(s.get("n_points", 201))
    adaptive = bool(s.get("adaptive", False))
    if adaptive:
        tgrid = _sc_lpf_adaptive_tgrid(c, n_points)
        pss_grid_kwargs = {"tgrid": tgrid}
    else:
        tgrid = np.linspace(0.0, period, n_points + 1)[:-1]
        pss_grid_kwargs = {"n_points": n_points}
    pss = pss_solve(
        sizes, {}, period, topo=topo,
        inputs=_sc_lpf_clocks(c, tgrid),
        tstab_periods=int(s.get("tstab_periods", 60)),
        residual_tol=float(s.get("residual_tol", 2e-2)),
        max_shooting_iters=int(s.get("max_shooting_iters", 20)),
        integration_method=s.get("integration_method", "be"),
        max_stabilization_periods=int(s.get("max_stabilization_periods", 200)),
        adaptive=adaptive,
        adaptive_reltol=float(s.get("adaptive_reltol", 1e-4)),
        adaptive_vabstol=float(s.get("adaptive_vabstol", 1e-6)),
        adaptive_iabstol=float(s.get("adaptive_iabstol", 1e-12)),
        adaptive_max_steps=int(s.get("adaptive_max_steps", 200000)),
        adaptive_h0=s.get("adaptive_h0"),
        adaptive_freeze_factor=float(s.get("adaptive_freeze_factor", 10.0)),
        cap_mode=s.get("cap_mode"),
        **pss_grid_kwargs)
    out = {}
    if {"pac", "pnoise"} & set(want):
        pf = np.logspace(-1, 3, 41)
        pac = pac_solve(sizes, {}, pf, pss_result=pss, input_drive={"vin": 1.0},
                        fd_state_step=1e-4, fd_input_step=1e-4)
        g = np.asarray(pac["gains"], float)
    if "pac" in want:
        out["pac"] = {"gain_baseband": float(g[0]), "bw_Hz": _bw_interp(pf, g)}
    if "pnoise" in want:
        band = tuple(c.get("noise_band", (0.1, 100.0)))
        nf = np.logspace(np.log10(band[0]), np.log10(band[1]), 21)
        pn = pnoise_solve(sizes, {}, nf, pss_result=pss, fundamental=f_clk,
                          input_drive={"vin": 1.0},
                          max_sideband=int(s.get("pnoise_max_sideband", 10)),
                          n_period_samples=int(s.get("pnoise_n_period_samples", 128)),
                          band=band)
        out["pnoise"] = {"out_uVrms": float(pn["out_uV_band"])}
    return out


# ── per-analysis comparison ──────────────────────────────────────────────────

def compare_dc(local, ref_dc, tol):
    """ref_dc: {signal: V}. Compare every node Cadence reports that we also solve."""
    rows = {}
    for name, refv in ref_dc.items():
        if name in local:
            rows[name] = _metric(local[name], refv, tol["dc_v_atol"], rel=False)
    return {"metrics": rows, "pass": all(r["pass"] for r in rows.values())}


def compare_ac(local, ref_ac, tol, *, outputs=("VOP", "VON"), inputs=("vip", "vin")):
    freqs, sig = ref_ac
    op, on = outputs
    ip, inn = inputs
    Hc = np.abs((sig[op] - sig[on]) / (sig[ip] - sig[inn]))
    rows = {
        "gain_dc_dB": _metric(local["gain_dc_dB"], 20 * np.log10(Hc.max()),
                              tol["ac_gain_rtol"]),
        "bw_Hz": _metric(local["bw_Hz"], _bw_interp(freqs, Hc), tol["ac_bw_rtol"]),
    }
    return {"metrics": rows, "pass": all(r["pass"] for r in rows.values())}


def compare_noise(local, ref_noise, ref_ac, tol, *, outputs=("VOP", "VON"),
                  inputs=("vip", "vin")):
    fr, out, _dev = ref_noise
    fac, sig = ref_ac
    op, on = outputs
    ip, inn = inputs
    Hc = np.abs((sig[op] - sig[on]) / (sig[ip] - sig[inn]))
    Hn = np.interp(np.log10(fr), np.log10(fac), Hc)
    irn_c = band_rms(fr, out ** 2 / Hn ** 2, *_IRN_BAND) * 1e6
    rows = {"irn_uVrms": _metric(local["irn_uVrms"], irn_c, tol["noise_irn_rtol"])}
    return {"metrics": rows, "pass": all(r["pass"] for r in rows.values())}


def compare_pac(local, ref_pac, tol, *, outputs=("voutp_f", "voutn_f"),
                inputs=("vinp", "vinn")):
    _f, sig = ref_pac
    op, on = outputs
    ip, inn = inputs
    gain = float(np.abs((sig[op][0] - sig[on][0]) / (sig[ip][0] - sig[inn][0])))
    rows = {"gain_baseband": _metric(local["gain_baseband"], gain, tol["pac_gain_rtol"])}
    return {"metrics": rows, "pass": all(r["pass"] for r in rows.values())}


def compare_pnoise(local, ref_pnoise, ref_pac, tol, *, outputs=("voutp_f", "voutn_f"),
                   inputs=("vinp", "vinn")):
    fr, out, _dev = ref_pnoise
    _f, sig = ref_pac
    op, on = outputs
    ip, inn = inputs
    gain = float(np.abs((sig[op][0] - sig[on][0]) / (sig[ip][0] - sig[inn][0])))
    irn_c = band_rms(fr, (out / gain) ** 2, *_IRN_BAND) * 1e6
    rows = {"irn_uVrms": _metric(local["irn_uVrms"], irn_c, tol["pnoise_irn_rtol"])}
    return {"metrics": rows, "pass": all(r["pass"] for r in rows.values())}


def compare_pac_sc(local, ref_pac, tol, *, output="VOUT", input="VIN"):
    """Single-ended SC-LPF PAC: baseband transfer H(f)=|VOUT/VIN| at sideband 0.
    Checks both DC gain (~1) and the -3 dB bandwidth (the metric the reverse-bias
    runaway used to wreck: 12 Hz vs Cadence ~17 Hz)."""
    freqs, sig = ref_pac
    Hc = np.abs(sig[output] / sig[input])
    rows = {
        "gain_baseband": _metric(local["gain_baseband"], float(Hc[0]),
                                 tol["pac_gain_rtol"]),
        "bw_Hz": _metric(local["bw_Hz"], _bw_interp(freqs, Hc), tol["pac_bw_rtol"]),
    }
    return {"metrics": rows, "pass": all(r["pass"] for r in rows.values())}


def compare_pnoise_sc(local, ref_pnoise, tol, *, band=(0.1, 100.0)):
    """Single-ended SC-LPF PNoise: integrated *output* noise over the band (the
    unambiguous metric — no gain referral; the flicker over-count was 4.7x here)."""
    fr, out, _dev = ref_pnoise
    out_c = band_rms(fr, out ** 2, *band) * 1e6
    rows = {"out_uVrms": _metric(local["out_uVrms"], out_c, tol["pnoise_out_rtol"])}
    return {"metrics": rows, "pass": all(r["pass"] for r in rows.values())}


# ── orchestration ────────────────────────────────────────────────────────────

def run_calibration(case_dir, *, analyses=None, tol_overrides=None, relaxed=False) -> dict:
    """Load reference, run the local solvers, compare per analysis, return a report."""
    loaded = load_reference(case_dir)
    metadata = loaded["metadata"]
    ref = loaded["ref"]
    want = analyses or metadata.get("analyses", [])

    tol = dict(DEFAULT_TOL)
    tol.update(metadata.get("tolerances", {}))
    tol.update(tol_overrides or {})
    if relaxed:
        tol = {k: v * 3.0 for k, v in tol.items()}

    local = run_local(metadata, analyses=want)
    io = {k: metadata["circuit"][k] for k in ("outputs", "inputs")
          if k in metadata["circuit"]}
    out_io = tuple(io.get("outputs", ("VOP", "VON")))
    in_io = tuple(io.get("inputs", ("vip", "vin")))
    if metadata.get("testbench") == "chopper":
        out_io = tuple(io.get("outputs", ("voutp_f", "voutn_f")))
        in_io = tuple(io.get("inputs", ("vinp", "vinn")))

    results = {}
    if metadata.get("testbench") == "sc_lpf":
        c = metadata["circuit"]
        sout, sin = c.get("output", "VOUT"), c.get("input", "VIN")
        if "pac" in want and "pac" in ref:
            results["pac"] = compare_pac_sc(local["pac"], ref["pac"], tol,
                                            output=sout, input=sin)
        if "pnoise" in want and "pnoise" in ref:
            results["pnoise"] = compare_pnoise_sc(
                local["pnoise"], ref["pnoise"], tol,
                band=tuple(c.get("noise_band", (0.1, 100.0))))
        return {
            "case": metadata.get("case", str(case_dir)),
            "testbench": "sc_lpf", "corner": metadata.get("corner"),
            "provenance": loaded["provenance"], "results": results,
            "overall_pass": all(r["pass"] for r in results.values()) if results else False,
        }
    if "dc" in want and "dc" in ref:
        results["dc"] = compare_dc(local["dc"], ref["dc"], tol)
    if "ac" in want and "ac" in ref:
        results["ac"] = compare_ac(local["ac"], ref["ac"], tol,
                                   outputs=out_io, inputs=in_io)
    if "noise" in want and "noise" in ref:
        results["noise"] = compare_noise(local["noise"], ref["noise"], ref["ac"], tol,
                                         outputs=out_io, inputs=in_io)
    if "pac" in want and "pac" in ref:
        results["pac"] = compare_pac(local["pac"], ref["pac"], tol,
                                     outputs=out_io, inputs=in_io)
    if "pnoise" in want and "pnoise" in ref:
        results["pnoise"] = compare_pnoise(local["pnoise"], ref["pnoise"], ref["pac"],
                                           tol, outputs=out_io, inputs=in_io)

    return {
        "case": metadata.get("case", str(case_dir)),
        "testbench": metadata.get("testbench"),
        "corner": metadata.get("corner"),
        "provenance": loaded["provenance"],
        "results": results,
        "overall_pass": all(r["pass"] for r in results.values()) if results else False,
    }


def format_report(report, fmt="text") -> str:
    if fmt == "json":
        return json.dumps(report, indent=2)
    lines = []
    status = "PASS" if report["overall_pass"] else "FAIL"
    prov = report.get("provenance") or {}
    lines.append(f"[{status}] {report['case']}  "
                 f"(Spectre {prov.get('spectre_version')} @ {prov.get('date')})")
    for analysis, res in report["results"].items():
        amark = "ok " if res["pass"] else "XX "
        lines.append(f"  {amark}{analysis}:")
        for metric, row in res["metrics"].items():
            pm = "ok" if row["pass"] else "XX"
            if row.get("rel", True):
                d = f"{row['delta'] * 100:+.2f}% (tol {row['tol'] * 100:g}%)"
            else:
                d = f"{row['delta']:+.4g} (tol {row['tol']:.0e})"
            lines.append(f"      [{pm}] {metric:<18} local={row['local']:.4g} "
                         f"ref={row['ref']:.4g}  Δ={d}")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _iter_cases(paths):
    for p in paths:
        p = Path(p)
        if (p / "metadata.json").exists():
            yield p
        else:
            for sub in sorted(p.iterdir()):
                if (sub / "metadata.json").exists():
                    yield sub


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cadence/Spectre calibration check")
    ap.add_argument("cases", nargs="*", default=["calibration"],
                    help="case dir(s) or a parent dir of cases")
    ap.add_argument("--all", action="store_true", help="run every case under calibration/")
    ap.add_argument("--analyses", help="comma-separated subset, e.g. ac,noise")
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("--relaxed", action="store_true", help="3x tolerances")
    args = ap.parse_args(argv)

    cases = ["calibration"] if args.all else (args.cases or ["calibration"])
    analyses = args.analyses.split(",") if args.analyses else None
    reports, ok = [], True
    for case in _iter_cases(cases):
        report = run_calibration(case, analyses=analyses, relaxed=args.relaxed)
        reports.append(report)
        ok = ok and report["overall_pass"]
        if not args.json:
            print(format_report(report))
    if args.json:
        print(json.dumps(reports, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
