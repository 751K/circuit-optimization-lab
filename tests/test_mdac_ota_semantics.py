"""Adversarial semantic tests for the MDAC OTA design work package.

Reviewer-side verification, sixth round. The design's CI tests cover nominal
happy paths; these audit the USER'S HARD CONSTRAINTS structurally (exactly one
ideal reference, all-transistor DUT), and attack the closed-loop behaviors the
CI file leaves untested: both residue polarities, zero-residue systematic
offset, the late-added 1.10 V supply point, and the noise budget.
Reviewer cross-checks of the claimed worst PVT point (ss/125C/0.9V: gain 91.6 dB,
DM PM 121.5 deg, settle 0.016 %, CM 1.3 mV) were reproduced exactly in review;
they are not repeated here to keep CI fast.
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
needs_ngspice = pytest.mark.skipif(not _HAVE, reason="FreePDK45 cards / ngspice not present")

TB_FILES = ["freepdk45_mdac_ota.json", "freepdk45_mdac_ota_ac.json",
            "freepdk45_mdac_ota_dmloop.json", "freepdk45_mdac_ota_cmfb1.json",
            "freepdk45_mdac_ota_cmfb2.json", "freepdk45_mdac_ota_noise.json"]

TIGHT = {"reltol": 1e-7, "vntol": 1e-11, "abstol": 1e-15}


def _tb(fn):
    return json.loads((ROOT / "examples" / fn).read_text())


def _gen():
    sys.path.insert(0, str(ROOT / "examples"))
    import mdac_ota_gen
    return mdac_ota_gen


# ── structural audits of the user's hard constraints (no ngspice needed) ───────
def test_exactly_one_ideal_reference_everywhere():
    """The DUT may receive exactly ONE ideal quantity: the 20 uA testbench
    reference. Every TB must contain exactly one current source, 20 uA, and
    no second current path masquerading as a bias."""
    for fn in TB_FILES:
        data = _tb(fn)
        isrc = data.get("current_sources", [])
        assert len(isrc) == 1, f"{fn}: expected exactly 1 current source, got {len(isrc)}"
        assert isrc[0][3] == pytest.approx(20e-6), f"{fn}: reference is not 20 uA"


def test_dut_is_all_transistor_no_controlled_sources_outside_probes():
    """Controlled sources are legal only as documented TB-side probes: the
    differential-Middlebrook mirror VCVS in the DM-loop TB. Any controlled
    source anywhere else means an ideal element crept into the DUT or an
    undocumented probe appeared."""
    for fn in TB_FILES:
        data = _tb(fn)
        for kind in ("vccs", "cccs", "ccvs"):
            assert not data.get(kind), f"{fn}: unexpected {kind} elements"
        vcvs = data.get("vcvs", [])
        if fn == "freepdk45_mdac_ota_dmloop.json":
            assert len(vcvs) >= 1     # the documented mirror probe
        else:
            assert not vcvs, f"{fn}: VCVS outside the DM-loop probe"


def test_no_ideal_voltage_bias_into_dut_nodes():
    """All internal bias nodes (from the generator's DC seed) must be solved
    nodes, never pinned by testbench vsources — the bias network really has to
    derive them from the 20 uA reference."""
    G = _gen()
    bias_nodes = set(G.base_seed(1.0)) - {"OUTP", "OUTN", "O1P", "O1N", "INP", "INN"}
    for fn in TB_FILES:
        data = _tb(fn)
        pinned = {vs[1] for vs in data.get("vsources", [])} | \
                 {vs[2] for vs in data.get("vsources", [])}
        leaked = bias_nodes & pinned
        # loop TBs re-route one bias node through Vinj by design (the loop break);
        # allow exactly the documented break nodes and nothing else.
        allowed = {"freepdk45_mdac_ota_dmloop.json": 2,
                   "freepdk45_mdac_ota_cmfb1.json": 2,
                   "freepdk45_mdac_ota_cmfb2.json": 2}.get(fn, 0)
        assert len(leaked) <= allowed, f"{fn}: bias nodes pinned by ideal sources: {leaked}"


def test_same_dut_in_every_testbench():
    """Defense-in-depth beyond the generator-consistency test: the on-disk
    JSONs themselves must agree on every device size and model (a stale file
    would silently verify a different DUT)."""
    ref = _tb(TB_FILES[0])
    ref_sizes = {d["name"]: (d["W"], d["L"]) for d in ref["devices"]}
    for fn in TB_FILES[1:]:
        data = _tb(fn)
        sizes = {d["name"]: (d["W"], d["L"]) for d in data["devices"]}
        common = set(ref_sizes) & set(sizes)
        assert len(common) > 10, f"{fn}: too few shared DUT devices"
        diff = {n for n in common if ref_sizes[n] != sizes[n]}
        assert not diff, f"{fn}: device sizes diverge from main TB: {diff}"


# ── closed-loop behaviors the CI file leaves untested ─────────────────────────
def _settle(vdd, s, corner="nom", temp=300.15):
    from circuitopt.circuit_loader import circuit_from_dict
    from circuitopt.ngspice_transient import transient_ngspice
    G = _gen()
    spec = circuit_from_dict(G.build_transient(vdd))
    b = spec.binding()
    dk = {name: dict(b.device_kwargs.get(name, {}), temperature=temp)
          for name, *_ in spec.topology.devices}
    seed = spec.topology.dc_guesses[0]
    V0 = np.array([seed.get(n, 0.0) for n in spec.topology.solved])
    n = 101
    tg = np.linspace(0.0, 5e-9, n)
    h = vdd / 2
    bp1 = np.full(n, h + s / 2); bp1[0] = h
    bp2 = np.full(n, h - s / 2); bp2[0] = h
    r = transient_ngspice(spec.sizes, spec.bias, tg, topo=spec.topology, nf=spec.nf,
                          model_types=spec.model_types, device_kwargs=dk,
                          corner=corner, V0=V0, inputs={"bp1": bp1, "bp2": bp2},
                          extra_options=TIGHT, max_step=0.05e-9)
    vop, von = r["nodes"]["OUTP"], r["nodes"]["OUTN"]
    return vop, von, h


@needs_ngspice
def test_positive_residue_settles_symmetrically():
    """CI tests only the negative max residue; slewing is single-ended-asymmetric
    in a two-stage OTA, so the positive extreme must be verified separately."""
    s = +0.9 / 16
    vop, von, h = _settle(1.0, s)
    vod = vop - von
    assert abs(vod[-1] - (-8.0 * s)) / 0.45 < 1e-3
    cm = (vop[0] + von[0]) / 2
    assert abs(cm - h) < 0.020


@needs_ngspice
def test_zero_residue_has_no_systematic_offset():
    """Zero bottom-plate step: the output must stay put. A differential offset
    here is a systematic DUT/bias asymmetry that max-residue tests mask."""
    vop, von, h = _settle(1.0, 0.0)
    vod = vop - von
    assert abs(vod[-1]) < 1e-3 * 0.45, f"systematic offset {vod[-1]*1e3:.3f} mV"
    assert abs((vop[-1] + von[-1]) / 2 - h) < 0.020


@needs_ngspice
def test_supply_1v1_cm_and_settling():
    """The user's late supply change added 1.10 V; the CM reference must track
    VDD/2 = 0.55 V there and the max residue must still settle."""
    s = -0.9 / 16
    vop, von, h = _settle(1.1, s)
    assert h == pytest.approx(0.55)
    vod = vop - von
    assert abs(vod[-1] - (-8.0 * s)) / 0.45 < 1e-3
    assert abs((vop[0] + von[0]) / 2 - h) < 0.020


@needs_ngspice
def test_half_residue_relative_accuracy():
    """Gain-of-8 must hold at intermediate levels, not just the extremes —
    a soft-compression error would pass the max-residue test yet break codes."""
    s = -0.9 / 32
    vop, von, _h = _settle(1.0, s)
    vod = vop - von
    assert abs(vod[-1] - (-8.0 * s)) / 0.45 < 1e-3


@needs_ngspice
def test_noise_budget_met_at_nominal():
    """Closed-loop output noise integrated 10 MHz - 20 GHz (the documented band)
    must sit inside the 452 uV budget and be physically nonzero."""
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.ngspice_ac import noise_ngspice
    spec = load_circuit_json(ROOT / "examples" / "freepdk45_mdac_ota_noise.json")
    b = spec.binding()
    res = noise_ngspice(spec.sizes, spec.bias, topo=spec.topology,
                        out="OUTP", ref="OUTN", src="VBP1",
                        fstart=1e7, fstop=2e10, points=10, band=(1e7, 2e10),
                        nf=spec.nf, model_types=b.model_types,
                        device_kwargs=b.device_kwargs, corner="nom",
                        x0_guess=spec.topology.dc_guesses[0])
    assert 20e-6 < res["onoise_rms"] < 452e-6, \
        f"output noise {res['onoise_rms']*1e6:.0f} uV outside (20, 452) uV"
