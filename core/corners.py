"""Process corners, mismatch Monte-Carlo, and latch detection — local-solver side.

Single source of truth for the things that were re-derived over and over during
corner / robustness work, so search, verification and MC all agree:

  * CORNERS        — global process shifts (pvt0 = ±3σ, pbeta0 = ±15σ, from the
                     PDK monte.scs sections).
  * mismatch_corner — per-device random mvt0/mbeta0 on top of a process corner.
  * latch_kick_corner — a deterministic ±kσ DIFFERENTIAL mismatch that pushes each
                     symmetric pair apart; a cheap screen for the cross-coupled
                     positive-feedback latch-up (one solve instead of a full MC).
  * metrics        — evaluate one design at one corner -> gain/BW/IRN + latch_dV.
  * corner_table   — metrics across typ/slow/fast.
  * mismatch_mc    — per-device mismatch MC at one corner, seeded from the nominal op.

This module drives the local Python solvers; Cadence/Spectre comparison should
live in dedicated verification scripts instead of the core solver package.
"""
import itertools
import os

# Corner sweeps and mismatch MC are long-running local-solver workloads, so default
# to optional Numba acceleration unless explicitly disabled.
os.environ.setdefault("CIRCUIT_USE_NUMBA", "1")

import numpy as np

from .ac_solver import ac_solve
from .noise_solver import band_rms, noise_analysis
from .topology import AFE_TOPO
from . import diagnostics

# Global process corners (pvt0 = -3·0.0753, pbeta0 = -15·0.036 for slow).
CORNERS = {
    "typical": {"pvt0": 0.0, "pbeta0": 0.0},
    "slow": {"pvt0": -0.2259, "pbeta0": -0.54},
    "fast": {"pvt0": +0.2259, "pbeta0": +0.54},
}
# Per-device mismatch sigmas: Vth (area-scaled inside the model) and beta (flat).
SIGMA_MVT0 = 1.27e-5
SIGMA_MBETA0 = 0.019
# AFE differential pairs — used to drive the latch screen.
AFE_PAIRS = (("M7", "M8"), ("M9", "M10"), ("M12", "M13"), ("M14", "M15"))

_DEFAULT_FREQS = np.logspace(-2, 4, 121)


def _base(corner):
    return CORNERS[corner] if isinstance(corner, str) else dict(corner)


def mismatch_corner(rng, devices, base="typical"):
    """Per-device corner map: process `base` + random mvt0/mbeta0 on each device."""
    b = _base(base)
    return {d: {**b, "mvt0": float(rng.normal(0, SIGMA_MVT0)),
                "mbeta0": float(rng.normal(0, SIGMA_MBETA0))} for d in devices}


def latch_kick_corner(base="slow", pairs=AFE_PAIRS, k=3.0, signs=None):
    """A DIFFERENTIAL ±kσ mismatch on each symmetric pair. `signs` is a ±1 per pair
    (default all +1). NOTE: which sign pattern triggers the cross-coupled latch
    varies by design — screen with `latch_screen`, which scans all patterns, not a
    single kick (a single direction has false negatives)."""
    b = _base(base)
    signs = signs if signs is not None else (1,) * len(pairs)
    c = {d: {**b, "mvt0": 0.0, "mbeta0": 0.0} for p in pairs for d in p}
    for (hi, lo), sg in zip(pairs, signs):
        c[hi] = {**b, "mvt0": +sg * k * SIGMA_MVT0, "mbeta0": +sg * k * SIGMA_MBETA0}
        c[lo] = {**b, "mvt0": -sg * k * SIGMA_MVT0, "mbeta0": -sg * k * SIGMA_MBETA0}
    return c


def latch_screen(sizes, bias, nf=None, base="slow", topo=AFE_TOPO, k=3.0,
                 pairs=AFE_PAIRS, x0_guess=None, freqs=None):
    """Worst-case differential-mismatch latch screen. Pushes each symmetric pair
    ±kσ apart over ALL sign patterns and returns the largest output imbalance
    |out+ - out-|. Small => robust against the regenerative latch; large =>
    latch-prone. Deterministic (2^(P-1) solves) — a cheap, reliable screen to use
    inside a search instead of a full per-candidate mismatch MC."""
    worst = 0.0
    for combo in itertools.product((1, -1), repeat=len(pairs) - 1):
        m = metrics(sizes, bias, nf=nf,
                    corner=latch_kick_corner(base, pairs, k, (1,) + combo),
                    topo=topo, x0_guess=x0_guess, freqs=freqs,
                    include_noise=False)
        if m is not None:
            worst = max(worst, m["latch_dV"])
    return worst


def metrics(sizes, bias, nf=None, corner=None, topo=AFE_TOPO, x0_guess=None,
            freqs=None, band=(0.05, 100.0), include_noise=True,
            noise_gate=None):
    """Evaluate one design at one corner. Returns a dict with:
        gain_peak_dB, bw_Hz, irn_uV, latch_dV (|out+ - out-| at the DC op;
        large => regenerative latch), and dc_op. None if the DC solve fails.

    Noise is optional because latch/gain/BW screens only need the AC/DC result.
    `noise_gate(out)` can defer IRN until after AC/latch checks, e.g. mismatch MC
    skips IRN for latched samples that are excluded from final stats."""
    if freqs is None:
        freqs = _DEFAULT_FREQS
    ac = ac_solve(sizes, bias, freqs, corner=corner, nf=nf, topo=topo, x0_guess=x0_guess)
    if ac is None:
        return None
    out = {"gain_peak_dB": float(ac["peak_dB"]), "bw_Hz": float(ac["bw_Hz"]),
           "dc_op": ac["dc_op"]}
    outs = topo.outputs
    out["latch_dV"] = (abs(ac["dc_op"][outs[0]] - ac["dc_op"][outs[1]])
                       if len(outs) == 2 else 0.0)
    out["irn_uV"] = float("nan")
    out["_noise_evaluated"] = False
    if include_noise and (noise_gate is None or noise_gate(out)):
        try:
            nz = noise_analysis(sizes, bias, freqs, corner=corner, nf=nf, topo=topo,
                                x0_guess=ac["dc_op"])
            out["irn_uV"] = band_rms(freqs, nz["irn_psd"], *band) * 1e6 if nz else float("nan")
            out["_noise_evaluated"] = True
        except Exception as exc:
            diagnostics.note("corners.irn_eval_fail", exc)
            out["irn_uV"] = float("nan")
    return out


def corner_table(sizes, bias, nf=None, topo=AFE_TOPO,
                 corners=("typical", "slow", "fast"), freqs=None, band=(0.05, 100.0),
                 include_noise=True):
    """Evaluate a design across process corners -> {corner: metrics-or-None}."""
    return {c: metrics(sizes, bias, nf=nf, corner=CORNERS[c], topo=topo,
                       freqs=freqs, band=band, include_noise=include_noise)
            for c in corners}


def mismatch_mc(sizes, bias, nf=None, topo=AFE_TOPO, base="slow", n=300, seed=0,
                latch_dV=5.0, freqs=None, band=(0.05, 100.0), include_noise=True):
    """Per-device mismatch MC at one process corner, seeded from the nominal op.

    Returns {"arrays": {metric: ndarray}, "latched": bool ndarray, "summary": ...}.
    A run is "latched" when latch_dV exceeds the threshold; summary stats are over
    the non-latched runs (mean/std/P5/P95) plus the latch_rate."""
    if freqs is None:
        freqs = _DEFAULT_FREQS
    devices = [d for d, *_ in topo.devices]
    rng = np.random.default_rng(seed)
    nom = ac_solve(sizes, bias, freqs, corner=_base(base), nf=nf, topo=topo)
    if nom is None:
        raise RuntimeError(f"nominal {base!r} DC solve failed; cannot seed MC")
    x0 = nom["dc_op"]
    keys = ("gain_peak_dB", "bw_Hz", "irn_uV", "latch_dV")
    rows = {k: [] for k in keys}
    noise_evaluated = 0
    for _ in range(n):
        cm = mismatch_corner(rng, devices, base=base)
        m = metrics(sizes, bias, nf=nf, corner=cm, topo=topo, x0_guess=x0,
                    freqs=freqs, band=band, include_noise=include_noise,
                    noise_gate=lambda out: out["latch_dV"] <= latch_dV)
        if m is None:
            continue
        noise_evaluated += int(m.get("_noise_evaluated", False))
        for k in keys:
            rows[k].append(m[k])
    arr = {k: np.asarray(v, float) for k, v in rows.items()}
    latched = arr["latch_dV"] > latch_dV
    good = ~latched
    summary = {"n": int(arr["gain_peak_dB"].size), "latched": int(latched.sum()),
               "latch_rate": float(latched.mean()) if latched.size else 0.0,
               "noise_evaluated": int(noise_evaluated)}
    for k in ("gain_peak_dB", "bw_Hz", "irn_uV"):
        col = arr[k][good]
        if col.size:
            summary[k] = {"mean": float(col.mean()), "std": float(col.std()),
                          "p5": float(np.percentile(col, 5)),
                          "p95": float(np.percentile(col, 95))}
    return {"arrays": arr, "latched": latched, "summary": summary}
