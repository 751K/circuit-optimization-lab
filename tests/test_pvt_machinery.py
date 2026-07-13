"""PVT-campaign machinery: mixed sf/fs corners + full-circuit ngspice oracles.

Skip-guarded on FreePDK45 cards + ngspice like ``tests/test_sar.py`` (the pure
golden-deck backward-compat checks below need only the cards, not a run).

Covers:
  * mixed per-polarity corners sf/fs on the characterisation-grid path AND a tiny
    2-transistor ngspice ``.op`` (NMOS follows ss @ sf / ff @ fs; PMOS the mirror);
  * nom/ss/ff transient decks byte-identical to the pre-change renderer (goldens in
    ``tests/golden/*.deck`` captured from the code BEFORE the sf/fs refactor);
  * ``ac_ngspice`` on an analytic RC low-pass and on the FD-OTA example;
  * ``noise_ngspice`` on a bare resistor (4kTR);
  * ``op_ngspice`` saturation vs triode region check;
  * ``loop_gain_ngspice`` on an analytic single-pole feedback loop;
  * temperature trends on the grid and through ``ac_ngspice``.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from circuitopt.ngspice_char import ngspice_binary
from circuitopt.toolchain import pdk_root

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
FP45 = Path(pdk_root()) / "freepdk45"
_HAVE = (FP45 / "models_nom" / "NMOS_VTG.inc").is_file() and ngspice_binary() is not None
pytestmark = pytest.mark.skipif(not _HAVE, reason="FreePDK45 cards / ngspice not present")


def _norm(deck: str) -> str:
    return deck.replace(str(FP45), "{{FP45}}")


# ── 1a. sf/fs per-polarity current sanity on the grid path ─────────────────────
def test_grid_sf_fs_mix_currents_match_ss_ff_per_polarity():
    from circuitopt.device_model import create_device

    def nmos(corner):
        return create_device("freepdk45.nmos", W=1.0, L=0.05, corner=corner).get_Idc(0.0, 0.6, 0.6)

    def pmos(corner):
        return create_device("freepdk45.pmos", W=1.0, L=0.05, corner=corner,
                             vb=1.0).get_Idc(1.0, 0.4, 0.4)

    # sf = NMOS slow (==ss) + PMOS fast (==ff); fs is the mirror.
    assert nmos("sf") == pytest.approx(nmos("ss"), rel=1e-9)
    assert pmos("sf") == pytest.approx(pmos("ff"), rel=1e-9)
    assert nmos("fs") == pytest.approx(nmos("ff"), rel=1e-9)
    assert pmos("fs") == pytest.approx(pmos("ss"), rel=1e-9)
    # sanity: slow current is strictly below fast for both polarities.
    assert abs(nmos("ss")) < abs(nmos("ff"))
    assert abs(pmos("ss")) < abs(pmos("ff"))


# ── 1a'. corner-name strictness: case-insensitive, typo -> ValueError ──────────
def test_corner_names_case_insensitive_and_typos_rejected_on_both_paths():
    """'SF' must behave as 'sf' and a typo ('sx') must raise ValueError on BOTH the
    grid path and the ngspice render path — never a silent nom fallback."""
    from circuitopt.device_model import create_device
    from circuitopt.freepdk45_model import normalize_corner
    from circuitopt.ngspice_render import resolve_freepdk45_cards
    # grid path: 'SF' == 'sf', 'TT' -> tt (nom card), ''/None -> nom, unknown raises.
    i_sf = create_device("freepdk45.nmos", W=1.0, L=0.05, corner="sf").get_Idc(0.0, 0.6, 0.6)
    i_up = create_device("freepdk45.nmos", W=1.0, L=0.05, corner="SF").get_Idc(0.0, 0.6, 0.6)
    assert i_up == i_sf
    assert normalize_corner("TT") == "tt"
    assert normalize_corner(None) == "nom" and normalize_corner("") == "nom"
    with pytest.raises(ValueError, match="unknown FreePDK45 corner"):
        create_device("freepdk45.nmos", W=1.0, L=0.05, corner="sx")
    # render path: same normalization + same rejection.
    mt = {"MN": "freepdk45.nmos"}
    c, cards, _ = resolve_freepdk45_cards(mt, {"MN": {"corner": "Sf"}}, {"MN"})
    assert c == "sf" and "models_ss" in cards["nmos"]
    with pytest.raises(ValueError, match="unknown FreePDK45 corner"):
        resolve_freepdk45_cards(mt, {"MN": {"corner": "sx"}}, {"MN"})


# ── 1b. sf/fs on a tiny 2-transistor ngspice .op ───────────────────────────────
def _two_fet_spec():
    from circuitopt.circuit_loader import circuit_from_dict
    return circuit_from_dict({
        "name": "sf_fs_pair",
        "solved": ["DN", "DP"],
        "rails": {"VDD": "VDD", "GND": 0.0, "VGN": "VGN", "VGP": "VGP"},
        "bias": {"VDD": 1.0, "VGN": 0.6, "VGP": 0.4},
        "devices": [
            {"name": "MN", "drain": "DN", "gate": "VGN", "source": "GND", "W": 1.0, "L": 0.05},
            {"name": "MP", "drain": "DP", "gate": "VGP", "source": "VDD", "W": 1.0, "L": 0.05}],
        "models": {"MN": {"type": "freepdk45.nmos"},
                   "MP": {"type": "freepdk45.pmos", "vb": 1.0}},
        "resistors": [["RN", "VDD", "DN", 5000.0], ["RP", "DP", "GND", 5000.0]],
        "outputs": ["DN"],
    })


def _op_id(spec, corner):
    from circuitopt.ngspice_ac import op_ngspice
    b = spec.binding().at_corner(corner)
    op = op_ngspice(spec.sizes, spec.bias, topo=spec.topology, nf=spec.nf,
                    model_types=b.model_types, device_kwargs=b.device_kwargs)
    return op["MN"]["id"], op["MP"]["id"]


def test_ngspice_op_sf_fs_currents_track_per_polarity_corner():
    spec = _two_fet_spec()
    n_sf, p_sf = _op_id(spec, "sf")
    n_ss, p_ss = _op_id(spec, "ss")
    n_ff, p_ff = _op_id(spec, "ff")
    n_fs, p_fs = _op_id(spec, "fs")
    # sf: NMOS uses the ss card, PMOS uses the ff card.
    assert n_sf == pytest.approx(n_ss, rel=1e-6)
    assert p_sf == pytest.approx(p_ff, rel=1e-6)
    # fs: the mirror.
    assert n_fs == pytest.approx(n_ff, rel=1e-6)
    assert p_fs == pytest.approx(p_ss, rel=1e-6)


def test_sf_deck_includes_both_card_directories():
    from circuitopt.ngspice_render import resolve_freepdk45_cards
    spec = _two_fet_spec()
    b = spec.binding().at_corner("sf")
    _c, cards, _p = resolve_freepdk45_cards(b.model_types, b.device_kwargs,
                                            {"MN", "MP"})
    assert "models_ss" in cards["nmos"] and "models_ff" in cards["pmos"]


# ── 2. backward-compat: nom/ss/ff transient decks byte-identical ───────────────
def _render_tran(spec, tgrid, corner=None, **kw):
    from circuitopt.ngspice_transient import render_freepdk45_transient_netlist
    return render_freepdk45_transient_netlist(
        spec.sizes, spec.bias, tgrid, topo=spec.topology, output_path="OUT.dat",
        nf=spec.nf, model_types=spec.model_types, device_kwargs=spec.device_kwargs,
        corner=corner, **kw).netlist


def test_transient_decks_byte_identical_to_pre_change_goldens():
    from circuitopt.circuit_loader import circuit_from_dict, load_circuit_json
    spec = load_circuit_json(ROOT / "examples" / "freepdk45_5t_ota.json")
    tgrid = np.linspace(0.0, 10e-9, 11)
    inputs = {"vip": np.full(11, 0.56), "vin": np.full(11, 0.54)}
    got = _render_tran(spec, tgrid, corner="nom", inputs=inputs)
    assert _norm(got) == (GOLDEN / "tran_5t_nom.deck").read_text()
    got_ss = _render_tran(spec, tgrid, corner="ss", inputs=inputs)
    assert _norm(got_ss) == (GOLDEN / "tran_5t_ss.deck").read_text()

    fd = load_circuit_json(ROOT / "examples" / "freepdk45_fd_ota.json")
    got_fd = _render_tran(fd, np.linspace(0.0, 5e-9, 6), corner="ff")
    assert _norm(got_fd) == (GOLDEN / "tran_fdota_ff.deck").read_text()

    cfg = json.load(open(ROOT / "examples" / "freepdk45_5t_ota.json"))
    cfg["transient_inputs"] = {"M1": "vip", "M2": "vin"}
    cfg["models"]["M3"]["vb"] = 0.7
    sp = circuit_from_dict(cfg)
    got_g = _render_tran(sp, tgrid, corner="nom", mismatch={"M1": 0.01},
                         inputs={"vip": np.linspace(0.55, 0.56, 11),
                                 "vin": np.linspace(0.55, 0.54, 11)})
    assert _norm(got_g) == (GOLDEN / "tran_5t_gatepwl_bulk_mismatch.deck").read_text()


# ── 3. ac_ngspice ──────────────────────────────────────────────────────────────
def _rc_topo(R=1e3, C=1e-12):
    from circuitopt.topology import Topology
    return Topology(solved=["IN", "OUT"], devices=[], rails={"GND": 0.0}, outputs=("OUT",),
                    vsources=[("VIN", "IN", "GND", 0.0)],
                    resistors=[("R1", "IN", "OUT", R)],
                    capacitors=[("C1", "OUT", "GND", C)])


def test_ac_ngspice_rc_lowpass_matches_analytic():
    from circuitopt.ngspice_ac import ac_ngspice, ac_response
    R, C = 1e3, 1e-12
    topo = _rc_topo(R, C)
    res = ac_ngspice({}, {}, topo=topo, acmag={"VIN": (1.0, 0.0)},
                     fstart=1e5, fstop=1e10, points=20, out_nodes=["OUT"])
    f = res["freq"]
    H = ac_response(res, "OUT", vin=1.0)
    Ha = 1.0 / (1.0 + 1j * 2 * np.pi * f * R * C)
    assert np.max(np.abs(np.abs(H) - np.abs(Ha)) / np.abs(Ha)) < 0.01
    assert np.max(np.abs(np.angle(H, deg=True) - np.angle(Ha, deg=True))) < 1.0  # <1 deg
    # f3dB = 1/(2 pi RC) = 159.2 MHz within 1%.
    i3 = int(np.argmin(np.abs(20 * np.log10(np.abs(H)) + 3.0103)))
    assert f[i3] == pytest.approx(1.0 / (2 * np.pi * R * C), rel=0.01)


def test_ac_ngspice_fd_ota_gain_ugbw_pm():
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.ngspice_ac import ac_ngspice, ac_response, peak_gain_db, \
        unity_gain_freq, phase_margin
    spec = load_circuit_json(ROOT / "examples" / "freepdk45_fd_ota.json")
    seed = spec.topology.dc_guesses[0]
    b = spec.binding()
    res = ac_ngspice(spec.sizes, spec.bias, topo=spec.topology,
                     acmag={"VINP": (0.5, 0.0), "VINN": (0.5, 180.0)},
                     fstart=1e3, fstop=1e11, points=15, out_nodes=["OUTP", "OUTN"],
                     nf=spec.nf, model_types=b.model_types, device_kwargs=b.device_kwargs,
                     corner="nom", x0_guess=seed)
    f = res["freq"]
    H = ac_response(res, "OUTP", "OUTN", vin=1.0)
    gain = peak_gain_db(f, H)
    ugbw = unity_gain_freq(f, H)
    pm = phase_margin(f, H)
    assert np.isfinite(gain) and np.isfinite(ugbw) and np.isfinite(pm)
    assert gain == pytest.approx(58.9, abs=3.0)     # docs §4.5 passband gain
    assert 80e6 < ugbw < 160e6                       # docs ~119.9 MHz
    assert 60.0 < pm < 100.0                          # docs ~84 deg


# ── 4. noise_ngspice ─────────────────────────────────────────────────────────
def test_noise_ngspice_bare_resistor_is_4kTR():
    from circuitopt.topology import Topology
    from circuitopt.ngspice_ac import noise_ngspice
    R = 1000.0
    topo = Topology(solved=["IN", "OUT"], devices=[], rails={"GND": 0.0}, outputs=("OUT",),
                    vsources=[("VIN", "IN", "GND", 0.0)],
                    resistors=[("R1", "IN", "OUT", R), ("R2", "OUT", "GND", 1e9)])
    res = noise_ngspice({}, {}, topo=topo, out="OUT", src="VIN",
                        fstart=1.0, fstop=1e6, points=10, band=(10.0, 1e5))
    kT = 1.380649e-23 * 300.15
    assert np.mean(res["onoise_psd"]) == pytest.approx(4 * kT * R, rel=0.02)
    assert res["band"] == (10.0, 1e5)
    assert res["onoise_rms"] > 0.0


# ── 5. op_ngspice ────────────────────────────────────────────────────────────
def test_op_ngspice_saturation_and_triode_region_check():
    from circuitopt.circuit_loader import circuit_from_dict
    from circuitopt.ngspice_ac import op_ngspice
    spec = circuit_from_dict({
        "name": "optest", "solved": ["D", "DT"],
        "rails": {"VDD": "VDD", "GND": 0.0, "VG": "VG", "VDLOW": "VDLOW"},
        "bias": {"VDD": 1.0, "VG": 1.0, "VDLOW": 0.05},
        "devices": [
            {"name": "MDIODE", "drain": "D", "gate": "D", "source": "GND", "W": 1.0, "L": 0.05},
            {"name": "MTRIODE", "drain": "DT", "gate": "VG", "source": "GND", "W": 1.0, "L": 0.05}],
        "models": {"MDIODE": {"type": "freepdk45.nmos"}, "MTRIODE": {"type": "freepdk45.nmos"}},
        "resistors": [["RD", "VDD", "D", 5000.0], ["RT", "VDLOW", "DT", 1.0]],
        "outputs": ["D"]})
    op = op_ngspice(spec.sizes, spec.bias, topo=spec.topology,
                    model_types=spec.model_types, device_kwargs=spec.device_kwargs)
    # diode-connected: |vds| = |vgs| >= |vdsat|  → saturation.
    assert op["MDIODE"]["region_ok"] is True
    assert abs(op["MDIODE"]["vds"]) >= abs(op["MDIODE"]["vdsat"])
    # gate high, drain pinned ~50 mV  → deep triode.
    assert op["MTRIODE"]["region_ok"] is False
    assert abs(op["MTRIODE"]["vds"]) < abs(op["MTRIODE"]["vdsat"])


# ── 6. loop_gain_ngspice on an analytic single-pole loop ──────────────────────
def test_loop_gain_ngspice_single_pole_matches_analytic():
    from circuitopt.topology import Topology
    from circuitopt.ngspice_ac import loop_gain_ngspice
    gm, R, C = 1e-3, 1e5, 1e-12
    # gm block (VCCS) into an R||C load; unity feedback (VCVS mu=+1) closes a negative
    # loop given ngspice's VCCS current sense; Vinj breaks the loop at the high-Z gm
    # control node for Middlebrook single voltage injection.
    topo = Topology(solved=["OUT", "CTRLA", "CTRLB"], devices=[], rails={"GND": 0.0},
                    outputs=("OUT",),
                    resistors=[("RL", "OUT", "GND", R)], capacitors=[("CL", "OUT", "GND", C)],
                    vccs=[("G1", "OUT", "GND", "CTRLB", "GND", gm)],
                    vcvs=[("Efb", "CTRLA", "GND", "OUT", "GND", 1.0)],
                    vsources=[("Vinj", "CTRLB", "CTRLA", 0.0)])
    res = loop_gain_ngspice({}, {}, topo=topo, inject="Vinj",
                            fstart=1e3, fstop=1e10, points=30)
    f, T = res["freq"], res["loop_gain"]
    A0, fp = gm * R, 1.0 / (2 * np.pi * R * C)
    Ta = A0 / (1.0 + 1j * f / fp)
    assert np.max(np.abs(T - Ta) / np.abs(Ta)) < 0.02   # matches the analytic loop gain
    ugf_a = fp * np.sqrt(A0 ** 2 - 1.0)
    pm_a = 180.0 - np.degrees(np.arctan(ugf_a / fp))
    assert res["ugf"] == pytest.approx(ugf_a, rel=0.03)
    assert res["pm"] == pytest.approx(pm_a, abs=3.0)     # analytic ~90.6 deg


# ── 7. temperature trends (grid + ac_ngspice) ─────────────────────────────────
def test_temperature_trends_monotonic():
    """At these overdrive/gain biases FreePDK45 is mobility-dominated, so both the
    device current and the OTA gain fall MONOTONICALLY as temperature rises
    (-40 -> 27 -> 125 C)."""
    from circuitopt.device_model import create_device
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.ngspice_ac import ac_ngspice, ac_response, peak_gain_db
    temps_k = [233.15, 300.15, 398.15]

    ids = [create_device("freepdk45.nmos", W=1.0, L=0.05, corner="nom",
                         temperature=tk).get_Idc(0.0, 0.4, 0.4) for tk in temps_k]
    assert ids[0] > ids[1] > ids[2]

    spec = load_circuit_json(ROOT / "examples" / "freepdk45_fd_ota.json")
    seed = spec.topology.dc_guesses[0]
    b = spec.binding()
    gains = []
    for tk in temps_k:
        r = ac_ngspice(spec.sizes, spec.bias, topo=spec.topology,
                       acmag={"VINP": (0.5, 0.0), "VINN": (0.5, 180.0)},
                       fstart=1e4, fstop=1e10, points=8, out_nodes=["OUTP", "OUTN"],
                       nf=spec.nf, model_types=b.model_types, device_kwargs=b.device_kwargs,
                       corner="nom", temperature=tk, x0_guess=seed)
        gains.append(peak_gain_db(r["freq"], ac_response(r, "OUTP", "OUTN")))
    assert gains[0] > gains[1] > gains[2]
