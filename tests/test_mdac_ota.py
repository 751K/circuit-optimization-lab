"""CI checks for the FreePDK45 14-bit-pipeline MDAC first-stage OTA.

Skip-guarded on FreePDK45 cards + ngspice (like tests/test_sar.py).  Single
conversions of the heavy analyses only — the mini-PVT / 45-point campaign runs
outside CI (see docs/mdac_ota_derivation.md §8 for the design-time table).

Covers, at nominal (tt / 27 C / 1.0 V):
  * schema validation of every freepdk45_mdac_ota*.json example;
  * generator/JSON consistency (the checked-in JSONs match mdac_ota_gen);
  * op_ngspice: all core gain devices saturated (region_ok);
  * open-loop gain > 80 dB;
  * DM loop PM > 60 deg and both CMFB loop PMs > 60 deg (loop_gain_ngspice);
  * one 5 ns max-residue transient settles < 0.1 % (FS-referred) with the
    output CM within 20 mV of VDD/2.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from circuitopt.ngspice_char import ngspice_binary
from circuitopt.toolchain import pdk_root

ROOT = Path(__file__).resolve().parents[1]
FP45 = Path(pdk_root()) / "freepdk45"
_HAVE = (FP45 / "models_nom" / "NMOS_VTG.inc").is_file() and ngspice_binary() is not None
pytestmark = pytest.mark.skipif(not _HAVE, reason="FreePDK45 cards / ngspice not present")

TB_FILES = [
    "freepdk45_mdac_ota.json",
    "freepdk45_mdac_ota_ac.json",
    "freepdk45_mdac_ota_dmloop.json",
    "freepdk45_mdac_ota_cmfb1.json",
    "freepdk45_mdac_ota_cmfb2.json",
    "freepdk45_mdac_ota_noise.json",
]

CORE_DEVS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8",
             "M9", "M10", "M11", "M12"]

# tight solver tolerances: ngspice's default reltol=1e-3 leaves a ~100 uV
# numerical band that a 0.1 % settling measurement cannot tolerate
TIGHT = {"reltol": 1e-7, "vntol": 1e-11, "abstol": 1e-15}


def _gen():
    sys.path.insert(0, str(ROOT / "examples"))
    import mdac_ota_gen
    return mdac_ota_gen


def _spec(path):
    from circuitopt.circuit_loader import load_circuit_json
    return load_circuit_json(path)


# ── schema + generator consistency ──────────────────────────────────────────────
def test_example_jsons_match_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schemas" / "circuit.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    for fn in TB_FILES:
        data = json.loads((ROOT / "examples" / fn).read_text())
        jsonschema.validate(data, schema)
        _spec(ROOT / "examples" / fn)   # loader accepts it too


def test_checked_in_jsons_match_generator():
    """The six TB JSONs are generated from one source of truth; a stale file
    (a W changed in one TB only) would silently split the DUT."""
    G = _gen()
    for fn, dct in G.all_testbenches().items():
        on_disk = json.loads((ROOT / "examples" / fn).read_text())
        assert on_disk == dct, f"{fn} is stale — regenerate with examples/mdac_ota_gen.py"


# ── DC operating point: saturation ───────────────────────────────────────────────
def test_core_devices_saturated_at_nominal():
    from circuitopt.ngspice_ac import op_ngspice
    spec = _spec(ROOT / "examples" / "freepdk45_mdac_ota_ac.json")
    b = spec.binding()
    op = op_ngspice(spec.sizes, spec.bias, topo=spec.topology, nf=spec.nf,
                    model_types=b.model_types, device_kwargs=b.device_kwargs,
                    corner="nom", x0_guess=spec.topology.dc_guesses[0])
    bad = [n for n in CORE_DEVS if not op[n]["region_ok"]]
    assert not bad, f"core devices out of saturation at nominal: {bad}"


# ── open-loop gain ────────────────────────────────────────────────────────────────
def test_open_loop_gain_above_80db():
    from circuitopt.ngspice_ac import ac_ngspice, ac_response
    spec = _spec(ROOT / "examples" / "freepdk45_mdac_ota_ac.json")
    b = spec.binding()
    res = ac_ngspice(spec.sizes, spec.bias, topo=spec.topology,
                     acmag={"VACP": (0.5, 0.0), "VACN": (0.5, 180.0)},
                     fstart=1e4, fstop=1e6, points=3, out_nodes=["OUTP", "OUTN"],
                     nf=spec.nf, model_types=b.model_types,
                     device_kwargs=b.device_kwargs, corner="nom",
                     x0_guess=spec.topology.dc_guesses[0])
    H = ac_response(res, "OUTP", "OUTN", vin=1.0)
    gain_db = 20.0 * np.log10(abs(H[0]))     # 10 kHz: above the AC-coupling corner
    assert gain_db > 80.0, f"open-loop gain {gain_db:.1f} dB <= 80 dB"


# ── loop phase margins ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("fn", ["freepdk45_mdac_ota_dmloop.json",
                                "freepdk45_mdac_ota_cmfb1.json",
                                "freepdk45_mdac_ota_cmfb2.json"])
def test_loop_phase_margin_above_60deg(fn):
    from circuitopt.ngspice_ac import loop_gain_ngspice
    spec = _spec(ROOT / "examples" / fn)
    b = spec.binding()
    lg = loop_gain_ngspice(spec.sizes, spec.bias, topo=spec.topology, inject="Vinj",
                           fstart=1e5, fstop=2e10, points=10, nf=spec.nf,
                           model_types=b.model_types, device_kwargs=b.device_kwargs,
                           corner="nom", x0_guess=spec.topology.dc_guesses[0])
    assert np.isfinite(lg["pm"]), f"{fn}: no unity crossing found"
    assert lg["pm"] > 60.0, f"{fn}: PM {lg['pm']:.1f} deg <= 60 deg"


# ── closed-loop residue settling + output CM ─────────────────────────────────────
def test_max_residue_settles_and_cm_within_20mv():
    """Max legitimate residue (FS/16 = 56.25 mV diff bottom-plate step ->
    0.45 V diff output) settles < 0.1 % FS-referred within the 5 ns hold, and
    the static output CM sits within 20 mV of VDD/2."""
    from circuitopt.ngspice_transient import transient_ngspice
    spec = _spec(ROOT / "examples" / "freepdk45_mdac_ota.json")
    b = spec.binding()
    seed = spec.topology.dc_guesses[0]
    V0 = np.array([seed.get(n, 0.0) for n in spec.topology.solved])
    n = 101
    tg = np.linspace(0.0, 5e-9, n)
    vdd = spec.bias["VDD"]
    h = vdd / 2
    s = -0.9 / 16                     # max residue: -56.25 mV diff
    bp1 = np.full(n, h + s / 2); bp1[0] = h
    bp2 = np.full(n, h - s / 2); bp2[0] = h
    r = transient_ngspice(spec.sizes, spec.bias, tg, topo=spec.topology, nf=spec.nf,
                          model_types=b.model_types, device_kwargs=b.device_kwargs,
                          corner="nom", V0=V0, inputs={"bp1": bp1, "bp2": bp2},
                          extra_options=TIGHT, max_step=0.05e-9)
    vop, von = r["nodes"]["OUTP"], r["nodes"]["OUTN"]
    vod = vop - von
    ideal = -8.0 * s                  # +0.45 V diff
    err_fs = abs(vod[-1] - ideal) / 0.45
    assert err_fs < 1e-3, f"settling error {err_fs*100:.3f} % FS >= 0.1 %"
    cm_static = (vop[0] + von[0]) / 2
    assert abs(cm_static - h) < 0.020, \
        f"output CM {cm_static:.4f} V off VDD/2 by {abs(cm_static-h)*1e3:.1f} mV"
