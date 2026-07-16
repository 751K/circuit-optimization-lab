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
pytestmark = [
    pytest.mark.ngspice_oracle,
    pytest.mark.skipif(not _HAVE, reason="FreePDK45 cards / ngspice not present"),
]


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


def test_noise_ngspice_can_disable_testbench_dc_helper_noise():
    from circuitopt.topology import Topology
    from circuitopt.ngspice_ac import noise_ngspice
    topo = Topology(solved=["IN", "OUT"], devices=[], rails={"GND": 0.0},
                    outputs=("OUT",), vsources=[("VIN", "IN", "GND", 0.0)],
                    resistors=[("RHELP", "IN", "OUT", 1e3),
                               ("RLEAK", "OUT", "GND", 1e9)])
    noisy = noise_ngspice({}, {}, topo=topo, out="OUT", src="VIN",
                          fstart=1.0, fstop=1e6, points=10)
    quiet = noise_ngspice({}, {}, topo=topo, out="OUT", src="VIN",
                          fstart=1.0, fstop=1e6, points=10,
                          noiseless_resistors={"RHELP"})
    assert np.mean(quiet["onoise_psd"]) < np.mean(noisy["onoise_psd"]) * 1e-5


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


def _tian_two_pole_loop(gate_shunt_farad):
    """Two-pole analytic loop for the Tian probe tests.

    Inverting VCVS (x1000) -> 1k/15.9nF pole (10 kHz) -> unity buffer -> 9k/1k
    divider (beta=0.1, source resistance 900 ohm) -> Vinj break -> gate node X.
    ``gate_shunt_farad`` loads the GATE side of the break: nonzero values break the
    single-injection high-Z premise right around loop crossover; 0.0 keeps the gate
    ideally high-Z (VCVS control input only). Analytic loop gain:
    T(s) = 100 / ((1 + s/w1) * (1 + s*900*Cg)),  w1 = 2*pi*10 kHz."""
    from circuitopt.topology import Topology
    caps = [("C1", "P1", "GND", 15.9155e-9)]
    if gate_shunt_farad:
        caps.append(("CG", "X", "GND", gate_shunt_farad))
    return Topology(
        solved=["X", "FB", "Y", "P1", "OUT"], devices=[], rails={"GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "Y", "P1", 1e3), ("RA", "OUT", "FB", 9e3),
                   ("RB", "FB", "GND", 1e3)],
        capacitors=caps,
        vsources=[("Vinj", "X", "FB", 0.0)],
        vcvs=[("E1", "Y", "GND", "GND", "X", 1000.0),
              ("E2", "OUT", "GND", "P1", "GND", 1.0)])


def _tian_analytic_T(freq, gate_shunt_farad):
    s = 2j * np.pi * np.asarray(freq, float)
    T = 100.0 / (1.0 + s / (2 * np.pi * 1e4))
    if gate_shunt_farad:
        T = T / (1.0 + s * 900.0 * gate_shunt_farad)
    return T


def test_loop_gain_tian_matches_analytic_where_single_injection_fails():
    """A 1.77 nF gate-side shunt drops the break's input impedance to ~90 ohm at
    1 MHz (vs the 900 ohm feedback side) — the exact regime where the MOS-gate
    Cgg kills single voltage injection. Tian must still track the closed form;
    the legacy probe must NOT (that deviation is what makes this test able to
    discriminate at all)."""
    from circuitopt.ngspice_ac import loop_gain_ngspice, loop_gain_tian_ngspice
    cg = 1.7684e-9
    topo = _tian_two_pole_loop(cg)
    tian = loop_gain_tian_ngspice({}, {}, topo=topo, inject="Vinj",
                                  fstart=1e2, fstop=1e7, points=10)
    legacy = loop_gain_ngspice({}, {}, topo=topo, inject="Vinj",
                               fstart=1e2, fstop=1e7, points=10)
    Ta = _tian_analytic_T(tian["freq"], cg)
    assert np.max(np.abs(tian["loop_gain"] - Ta) / np.abs(Ta)) < 0.02
    assert np.max(np.abs(legacy["loop_gain"] - Ta) / np.abs(Ta)) > 0.20

    fgrid = np.logspace(2, 7, 4001)
    Tg = _tian_analytic_T(fgrid, cg)
    iu = int(np.argmin(np.abs(np.abs(Tg) - 1.0)))
    pm_a = 180.0 + np.degrees(np.angle(Tg[iu]))
    assert tian["pm"] == pytest.approx(pm_a, abs=2.0)    # analytic ~19.8 deg


def test_loop_gain_tian_differential_mirror_probe_matches_analytic():
    """The MDAC dm-loop testbench breaks BOTH input gates: Vinj on the P side and a
    unity VCVS (Emir) that copies -e onto the N side. A single-ended run-2 current
    injection would excite the orthogonal mode; the oracle must auto-detect the
    mirror and inject an anti-phase counter current so the P-side measurements stay
    the differential half-circuit's Tian quantities. Both halves here are identical
    two-pole loops with the gate shunt, so the analytic T is the same closed form."""
    from circuitopt.topology import Topology
    from circuitopt.ngspice_ac import loop_gain_tian_ngspice
    cg = 1.7684e-9
    topo = Topology(
        solved=["X", "FB", "Y", "P1", "OUT", "XN", "FBN", "YN", "P1N", "OUTN"],
        devices=[], rails={"GND": 0.0}, outputs=("OUT",),
        resistors=[("R1", "Y", "P1", 1e3), ("RA", "OUT", "FB", 9e3),
                   ("RB", "FB", "GND", 1e3),
                   ("R1N", "YN", "P1N", 1e3), ("RAN", "OUTN", "FBN", 9e3),
                   ("RBN", "FBN", "GND", 1e3)],
        capacitors=[("C1", "P1", "GND", 15.9155e-9), ("CG", "X", "GND", cg),
                    ("C1N", "P1N", "GND", 15.9155e-9), ("CGN", "XN", "GND", cg)],
        vsources=[("Vinj", "X", "FB", 0.0)],
        vcvs=[("E1", "Y", "GND", "GND", "X", 1000.0),
              ("E2", "OUT", "GND", "P1", "GND", 1.0),
              ("E1N", "YN", "GND", "GND", "XN", 1000.0),
              ("E2N", "OUTN", "GND", "P1N", "GND", 1.0),
              ("Emir", "XN", "FBN", "FB", "X", 1.0)])
    tian = loop_gain_tian_ngspice({}, {}, topo=topo, inject="Vinj",
                                  fstart=1e2, fstop=1e7, points=10)
    Ta = _tian_analytic_T(tian["freq"], cg)
    assert np.max(np.abs(tian["loop_gain"] - Ta) / np.abs(Ta)) < 0.02


def test_loop_gain_tian_agrees_with_single_injection_at_high_z_break():
    """With a truly high-impedance gate side (no shunt) the legacy premise holds:
    both probes must reproduce the closed form."""
    from circuitopt.ngspice_ac import loop_gain_ngspice, loop_gain_tian_ngspice
    topo = _tian_two_pole_loop(0.0)
    tian = loop_gain_tian_ngspice({}, {}, topo=topo, inject="Vinj",
                                  fstart=1e2, fstop=1e7, points=10)
    legacy = loop_gain_ngspice({}, {}, topo=topo, inject="Vinj",
                               fstart=1e2, fstop=1e7, points=10)
    Ta = _tian_analytic_T(tian["freq"], 0.0)
    assert np.max(np.abs(tian["loop_gain"] - Ta) / np.abs(Ta)) < 0.02
    assert np.max(np.abs(legacy["loop_gain"] - Ta) / np.abs(Ta)) < 0.02
    assert tian["pm"] == pytest.approx(legacy["pm"], abs=1.0)


# ── 6b. chained same-process analyses (S4 speed lever) ─────────────────────────
def _count_runs(monkeypatch, module):
    """Count ngspice subprocess launches through *module*'s ``_run_ngspice``."""
    counter = {"n": 0}
    real = module._run_ngspice

    def counting(*args, **kwargs):
        counter["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "_run_ngspice", counting)
    return counter


def test_loop_gain_tian_chained_one_process_matches_two_process(monkeypatch):
    """chain=True must fold the Tian v-/i-injection pair into ONE ngspice process
    (second sweep via ``alter @src[acmag]``) and reproduce the two-process result
    on the analytic two-pole loop — including the differential-mirror probe."""
    import circuitopt.ngspice_ac as ngac
    from circuitopt.ngspice_ac import loop_gain_tian_ngspice
    counter = _count_runs(monkeypatch, ngac)
    cg = 1.7684e-9
    topo = _tian_two_pole_loop(cg)
    two = loop_gain_tian_ngspice({}, {}, topo=topo, inject="Vinj",
                                 fstart=1e2, fstop=1e7, points=10, chain=False)
    n_two = counter["n"]
    one = loop_gain_tian_ngspice({}, {}, topo=topo, inject="Vinj",
                                 fstart=1e2, fstop=1e7, points=10, chain=True)
    n_one = counter["n"] - n_two
    assert (n_two, n_one) == (2, 1)
    np.testing.assert_allclose(one["loop_gain"], two["loop_gain"], rtol=1e-9, atol=0.0)
    assert one["pm"] == pytest.approx(two["pm"], abs=1e-6)
    assert one["ugf"] == pytest.approx(two["ugf"], rel=1e-9)
    assert one["gm_db"] == pytest.approx(two["gm_db"], abs=1e-6, nan_ok=True)
    Ta = _tian_analytic_T(one["freq"], cg)
    assert np.max(np.abs(one["loop_gain"] - Ta) / np.abs(Ta)) < 0.02

    # Differential double-break probe: the pre-set "ac 0 <phase>" mirror source
    # must stay inert in run 1 and carry the anti-phase injection in run 2.
    from circuitopt.topology import Topology
    mirror = Topology(
        solved=["X", "FB", "Y", "P1", "OUT", "XN", "FBN", "YN", "P1N", "OUTN"],
        devices=[], rails={"GND": 0.0}, outputs=("OUT",),
        resistors=[("R1", "Y", "P1", 1e3), ("RA", "OUT", "FB", 9e3),
                   ("RB", "FB", "GND", 1e3),
                   ("R1N", "YN", "P1N", 1e3), ("RAN", "OUTN", "FBN", 9e3),
                   ("RBN", "FBN", "GND", 1e3)],
        capacitors=[("C1", "P1", "GND", 15.9155e-9), ("CG", "X", "GND", cg),
                    ("C1N", "P1N", "GND", 15.9155e-9), ("CGN", "XN", "GND", cg)],
        vsources=[("Vinj", "X", "FB", 0.0)],
        vcvs=[("E1", "Y", "GND", "GND", "X", 1000.0),
              ("E2", "OUT", "GND", "P1", "GND", 1.0),
              ("E1N", "YN", "GND", "GND", "XN", 1000.0),
              ("E2N", "OUTN", "GND", "P1N", "GND", 1.0),
              ("Emir", "XN", "FBN", "FB", "X", 1.0)])
    two_m = loop_gain_tian_ngspice({}, {}, topo=mirror, inject="Vinj",
                                   fstart=1e2, fstop=1e7, points=10, chain=False)
    one_m = loop_gain_tian_ngspice({}, {}, topo=mirror, inject="Vinj",
                                   fstart=1e2, fstop=1e7, points=10, chain=True)
    np.testing.assert_allclose(one_m["loop_gain"], two_m["loop_gain"],
                               rtol=1e-9, atol=0.0)
    Tm = _tian_analytic_T(one_m["freq"], cg)
    assert np.max(np.abs(one_m["loop_gain"] - Tm) / np.abs(Tm)) < 0.02


def test_loop_gain_tian_env_toggle_read_at_call_time(monkeypatch):
    """CIRCUITOPT_NGSPICE_CHAIN: unset/"1" -> one process, "0" -> two; the
    ``chain`` kwarg overrides the env in both directions."""
    import circuitopt.ngspice_ac as ngac
    from circuitopt.ngspice_ac import loop_gain_tian_ngspice
    counter = _count_runs(monkeypatch, ngac)
    topo = _tian_two_pole_loop(0.0)
    kw = dict(topo=topo, inject="Vinj", fstart=1e3, fstop=1e6, points=5)

    monkeypatch.delenv("CIRCUITOPT_NGSPICE_CHAIN", raising=False)
    loop_gain_tian_ngspice({}, {}, **kw)
    assert counter["n"] == 1                      # unset -> chained
    monkeypatch.setenv("CIRCUITOPT_NGSPICE_CHAIN", "0")
    loop_gain_tian_ngspice({}, {}, **kw)
    assert counter["n"] == 3                      # "0" -> two processes
    loop_gain_tian_ngspice({}, {}, **kw, chain=True)
    assert counter["n"] == 4                      # kwarg beats env
    monkeypatch.setenv("CIRCUITOPT_NGSPICE_CHAIN", "1")
    loop_gain_tian_ngspice({}, {}, **kw)
    assert counter["n"] == 5                      # "1" -> chained
    loop_gain_tian_ngspice({}, {}, **kw, chain=False)
    assert counter["n"] == 7                      # kwarg beats env


# ── 6c. chained multi-case transient (alter @v[pwl] between tran runs) ─────────
def _pwl_rc_topo():
    """RC testbench with a PWL-driven input source plus a constant 'hold' source
    (stand-in for the MDAC hold clocks, identical across cases -> never altered)."""
    from circuitopt.topology import Topology
    return Topology(
        solved=["IN", "OUT", "H"], devices=[], rails={"GND": 0.0}, outputs=("OUT",),
        vsources=[("VIN", "IN", "GND", "vin"), ("VH", "H", "GND", "hold")],
        resistors=[("R1", "IN", "OUT", 1e3), ("R2", "H", "OUT", 5e3)],
        capacitors=[("C1", "OUT", "GND", 1e-12)])


def test_transient_chain_matches_per_process_runs(monkeypatch):
    """One chained process (tran + alter @v[pwl] + tran ...) must reproduce one
    process per case: same waveforms, branch currents and result shape.  Case 3
    has a DIFFERENT t=0 input value, so the chained process must redo the
    ``.nodeset``-seeded DC solve for that tran, not reuse the previous state."""
    import circuitopt.ngspice_transient as ngtr
    from circuitopt.ngspice_transient import transient_ngspice, transient_ngspice_chain
    topo = _pwl_rc_topo()
    n = 101
    tg = np.linspace(0.0, 5e-9, n)
    hold = np.full(n, 0.2)

    def case(v0, v1):
        v = np.full(n, v1); v[0] = v0
        return {"inputs": {"vin": v, "hold": hold}}

    cases = [case(0.45, 0.45 - 0.9 / 32), case(0.45, 0.45),
             case(0.45, 0.45 + 0.9 / 32), case(0.10, 0.60)]
    V0 = np.array([0.45, 0.40, 0.20])

    counter = _count_runs(monkeypatch, ngtr)
    separate = [transient_ngspice({}, {}, tg, topo=topo, V0=V0, **c) for c in cases]
    n_separate = counter["n"]
    chained = transient_ngspice_chain({}, {}, tg, topo=topo, V0=V0, cases=cases)
    n_chained = counter["n"] - n_separate
    assert (n_separate, n_chained) == (len(cases), 1)

    assert len(chained) == len(cases)
    for ref, got in zip(separate, chained):
        assert sorted(got) == sorted(ref)
        for node in ("IN", "OUT", "H"):
            np.testing.assert_allclose(got["nodes"][node], ref["nodes"][node],
                                       rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(got["output"], ref["output"], rtol=0.0, atol=1e-9)
        for name in ref["branch_currents"]:
            np.testing.assert_allclose(got["branch_currents"][name],
                                       ref["branch_currents"][name],
                                       rtol=0.0, atol=1e-12)
        assert got["process"] == ref["process"]


def test_transient_chain_rejects_non_input_variation():
    from circuitopt.ngspice_transient import transient_ngspice_chain
    topo = _pwl_rc_topo()
    n = 11
    tg = np.linspace(0.0, 5e-9, n)
    wave = {"vin": np.full(n, 0.45), "hold": np.full(n, 0.2)}
    with pytest.raises(ValueError, match="non-chainable"):
        transient_ngspice_chain({}, {}, tg, topo=topo,
                                cases=[{"inputs": wave},
                                       {"inputs": wave, "max_step": 1e-12}])
    with pytest.raises(ValueError, match="at least one case"):
        transient_ngspice_chain({}, {}, tg, topo=topo, cases=[])


def test_chained_pwl_compression_is_exact_and_guarded():
    """The constant-run PWL compressor must reproduce the identical piecewise-
    linear function (endpoints + ramp edge preserved bit-exactly) and the alter
    builder must never emit a line past ngspice's silent ~1000-word command
    cliff — compress when possible, refuse loudly otherwise."""
    from circuitopt.ngspice_transient import (
        _NGSPICE_CMD_MAX_WORDS, _alter_pwl_line, _compress_pwl_tokens)
    t = np.linspace(0.0, 5e-9, 501)
    v = np.full(501, 0.478125); v[0] = 0.45          # bp1-style step, h=0.45
    tokens = tuple(f"{x:.17g}" for pair in zip(t, v) for x in pair)
    comp = _compress_pwl_tokens(tokens)
    assert len(comp) == 6                             # 3-point PWL
    assert comp[:4] == tokens[:4]                     # t0/v0 + ramp edge exact
    assert comp[-2:] == tokens[-2:]                   # final endpoint exact
    ct = np.array([float(x) for x in comp[0::2]])
    cv = np.array([float(x) for x in comp[1::2]])
    np.testing.assert_array_equal(np.interp(t, ct, cv), v)   # identical function
    line = _alter_pwl_line("vbp1", tokens)            # 1006 words dense -> compressed
    assert len(line.split()) <= _NGSPICE_CMD_MAX_WORDS
    ramp = tuple(f"{x:.17g}" for pair in zip(t, np.linspace(0.0, 1.0, 501))
                 for x in pair)
    with pytest.raises(RuntimeError, match="command"):
        _alter_pwl_line("vbp1", ramp)                 # incompressible -> loud refusal


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
