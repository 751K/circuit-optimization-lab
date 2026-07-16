"""CI-fast guard for the 45-point MDAC OTA PVT campaign.

Two independent checks:

  * If ``results/mdac_ota_pvt45.csv`` contains a complete 45-point campaign,
    every point must pass every spec flag (a regression that shifts any corner
    out of spec fails here without re-running the multi-minute campaign).
    Partial resumable outputs are skipped rather than treated as sign-off data.
    The oracle campaign lives in
    ``experiments/freepdk45_mdac_ngspice_oracle_campaign.py``.

  * One spot PVT point (ss / 125 C / 0.90 V -- the design's worst vertex) is
    re-measured LIVE with the campaign's own conventions so CI catches a
    generator/sizing regression even when the CSV is stale or absent: open-loop
    gain > 84 dB, DM loop PM > 60 deg, static output CM within 20 mV of VDD/2.

Skip-guarded on FreePDK45 cards + ngspice like the other MDAC tests.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

from circuitopt.ngspice_char import ngspice_binary
from circuitopt.toolchain import pdk_root

ROOT = Path(__file__).resolve().parents[1]
FP45 = Path(pdk_root()) / "freepdk45"
_HAVE = (FP45 / "models_nom" / "NMOS_VTG.inc").is_file() and ngspice_binary() is not None
needs_ngspice = pytest.mark.skipif(not _HAVE, reason="FreePDK45 cards / ngspice not present")

CSV = ROOT / "results" / "mdac_ota_pvt45.csv"
PASS_FLAGS = ["pass_gain", "pass_dmpm", "pass_cmfb1pm", "pass_cmfb2pm",
              "pass_settle", "pass_cm", "pass_sat", "pass_noise", "pass_all"]


# ── CSV sign-off gate (no ngspice needed) ────────────────────────────────────────
@pytest.mark.skipif(not CSV.is_file(), reason="campaign CSV not generated")
def test_campaign_csv_all_45_pass():
    with open(CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) < 45:
        pytest.skip(f"campaign CSV is partial: {len(rows)}/45 PVT points")
    assert len(rows) == 45, f"expected 45 PVT points, got {len(rows)}"
    corners = {(r["corner"], float(r["temp_c"]), float(r["vdd"])) for r in rows}
    assert len(corners) == 45, "duplicate / missing PVT points in the campaign CSV"
    bad = [f"{r['corner']}/{r['temp_c']}C/{r['vdd']}V"
           for r in rows if any(r[f] not in ("1", "True") for f in PASS_FLAGS)]
    assert not bad, f"campaign points failing a spec flag: {bad}"


# ── live spot re-measurement at the worst vertex ─────────────────────────────────
@pytest.mark.ngspice_oracle
@needs_ngspice
def test_spot_ss_125c_0v9_live():
    sys.path.insert(0, str(ROOT / "experiments"))
    sys.path.insert(0, str(ROOT / "examples"))
    import mdac_ota_gen as G
    from freepdk45_mdac_ngspice_oracle_campaign import TIGHT, _dk
    from circuitopt.circuit_loader import circuit_from_dict
    from circuitopt.ngspice_ac import (ac_ngspice, ac_response, loop_gain_ngspice)
    from circuitopt.ngspice_transient import transient_ngspice

    corner, tk, vdd = "ss", 125.0 + 273.15, 0.90
    h = vdd / 2.0

    # open-loop gain @ 10 kHz
    spec = circuit_from_dict(G.build_ac(vdd))
    b, dk = _dk(spec, tk)
    ac = ac_ngspice(spec.sizes, spec.bias, topo=spec.topology,
                    acmag={"VACP": (0.5, 0.0), "VACN": (0.5, 180.0)},
                    fstart=1e4, fstop=1e6, points=3, out_nodes=["OUTP", "OUTN"],
                    nf=spec.nf, model_types=b.model_types, device_kwargs=dk,
                    corner=corner, x0_guess=spec.topology.dc_guesses[0])
    gain_db = 20.0 * np.log10(abs(ac_response(ac, "OUTP", "OUTN", vin=1.0)[0]))
    assert gain_db > 84.0, f"ss/125C/0.9V open-loop gain {gain_db:.1f} dB <= 84 dB"

    # DM loop PM (plateau reference -> fstart 1e5, campaign convention)
    spec = circuit_from_dict(G.build_dmloop(vdd))
    b, dk = _dk(spec, tk)
    lg = loop_gain_ngspice(spec.sizes, spec.bias, topo=spec.topology, inject="Vinj",
                           fstart=1e5, fstop=2e10, points=20, nf=spec.nf,
                           model_types=b.model_types, device_kwargs=dk,
                           corner=corner, x0_guess=spec.topology.dc_guesses[0])
    assert np.isfinite(lg["pm"]) and lg["pm"] > 60.0, \
        f"ss/125C/0.9V DM PM {lg['pm']:.1f} deg <= 60 deg"

    # static output CM within 20 mV of VDD/2 (t=0 DC solution of the transient TB)
    spec = circuit_from_dict(G.build_transient(vdd))
    b, dk = _dk(spec, tk)
    seed = spec.topology.dc_guesses[0]
    V0 = np.array([seed.get(n, 0.0) for n in spec.topology.solved])
    nstep = 101
    tg = np.linspace(0.0, 5e-9, nstep)
    bp1 = np.full(nstep, h); bp2 = np.full(nstep, h)
    r = transient_ngspice(spec.sizes, spec.bias, tg, topo=spec.topology, nf=spec.nf,
                          model_types=b.model_types, device_kwargs=dk, corner=corner,
                          V0=V0, inputs={"bp1": bp1, "bp2": bp2},
                          extra_options=TIGHT, max_step=0.05e-9)
    cm = (r["nodes"]["OUTP"][0] + r["nodes"]["OUTN"][0]) / 2.0
    assert abs(cm - h) < 0.020, f"ss/125C/0.9V static CM off VDD/2 by {abs(cm-h)*1e3:.1f} mV"
