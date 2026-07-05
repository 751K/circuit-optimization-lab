"""FreePDK45 PDK — ngspice-C as the device evaluator.

FreePDK45's BSIM4 ``version = 4.0`` cards diverge ~30 % from our BSIM4.8 OSDI VA
(version-independently), so FreePDK45 binds to ngspice-C via a cached
characterisation grid (:mod:`core.ngspice_device`) rather than the OSDI host. The
oracle is therefore ngspice itself: these tests pin that (1) a device's Id/gm/gds
reproduce a direct ngspice ``.op`` at the grid nodes, and (2) a 5T OTA through the
project's ``ac_solve`` matches ngspice's own ``.ac`` on the equivalent netlist.

Needs the external ngspice + FreePDK45 cards; skips cleanly without them.
"""
import json
import os
import subprocess
import tempfile

import numpy as np
import pytest

from core.ngspice_char import ngspice_binary

PDK_ROOT = os.environ.get("PDK_ROOT", "/Volumes/MacoutDsik/pdk")
_FP45 = os.path.join(PDK_ROOT, "freepdk45", "models_nom", "NMOS_VTG.inc")
_RUN = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "run-ngspice.sh")
_HAVE = os.path.exists(_FP45) and ngspice_binary() is not None

pytestmark = pytest.mark.skipif(not _HAVE, reason="FreePDK45 cards / ngspice not present")

_CFG = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    "examples", "freepdk45_5t_ota.json")


def _ngspice_op(model, card, W, L, vd, vg, vs, vb):
    """Direct ngspice ``.op`` (built-in C-BSIM4) → (Id, gm, gds) — the oracle."""
    deck = (f"* op\n.include \"{card}\"\n"
            f"mx d g s b {model} w={W}u l={L}u\n"
            f"vd d 0 {vd}\nvg g 0 {vg}\nvs s 0 {vs}\nvb b 0 {vb}\n"
            f".control\nop\nprint @mx[id] @mx[gm] @mx[gds]\n.endc\n.end\n")
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as fh:
        fh.write(deck)
        cir = fh.name
    try:
        out = subprocess.run([_RUN, "-b", cir], capture_output=True, text=True).stdout
    finally:
        os.unlink(cir)
    v = {}
    for line in out.splitlines():
        for k in ("@mx[id]", "@mx[gm]", "@mx[gds]"):
            if line.strip().startswith(k) and "=" in line:
                v[k[4:-1]] = float(line.split("=")[1])
    return v["id"], v["gm"], v["gds"]


def test_pdk_registered():
    from core.device_model import create_transistor, list_pdks
    assert "freepdk45" in list_pdks()
    n = create_transistor("nmos", pdk="freepdk45", W=0.09, L=0.05, corner="nom")
    p = create_transistor("pmos", pdk="freepdk45", W=0.09, L=0.05, corner="nom", vb=1.0)
    assert type(n).__name__ == "Fp45Nfet" and type(p).__name__ == "Fp45Pfet"


def test_nmos_matches_ngspice_op():
    """Grid-node Id/gm/gds are exact ngspice-C (this is the model==oracle anchor)."""
    from core.device_model import create_transistor
    card = os.path.join(PDK_ROOT, "freepdk45", "models_nom", "NMOS_VTG.inc")
    n = create_transistor("nmos", pdk="freepdk45", W=0.09, L=0.05, corner="nom")
    # (Vs,Vd,Vg)=(0,0.5,0.7) is on the 25 mV grid → no interpolation error
    Id = n.get_Idc(0.0, 0.5, 0.7)
    ss = n.get_ss_params(0.0, 0.5, 0.7)
    id_ng, gm_ng, gds_ng = _ngspice_op("NMOS_VTG", card, 0.09, 0.05, 0.5, 0.7, 0.0, 0.0)
    assert Id == pytest.approx(abs(id_ng), rel=1e-3)
    assert ss["gm"] == pytest.approx(abs(gm_ng), rel=1e-3)
    assert ss["gds"] == pytest.approx(abs(gds_ng), rel=1e-3)


def test_pmos_matches_ngspice_op():
    from core.device_model import create_transistor
    card = os.path.join(PDK_ROOT, "freepdk45", "models_nom", "PMOS_VTG.inc")
    p = create_transistor("pmos", pdk="freepdk45", W=0.09, L=0.05, corner="nom", vb=1.0)
    # |Vgs|=0.7,|Vds|=0.5,Vsb=0: Vs=1.0, Vd=0.5, Vg=0.3, bulk=1.0
    Id = abs(p.get_Idc(1.0, 0.5, 0.3))
    ss = p.get_ss_params(1.0, 0.5, 0.3)
    id_ng, gm_ng, gds_ng = _ngspice_op("PMOS_VTG", card, 0.09, 0.05, 0.5, 0.3, 1.0, 1.0)
    assert Id == pytest.approx(abs(id_ng), rel=1e-3)
    assert ss["gm"] == pytest.approx(abs(gm_ng), rel=1e-3)
    assert ss["gds"] == pytest.approx(abs(gds_ng), rel=1e-3)


def test_corners_shift_threshold():
    """ss/ff corners select different cards → higher/lower drive than nom."""
    from core.device_model import create_transistor
    kw = dict(pdk="freepdk45", W=0.09, L=0.05)
    i_nom = create_transistor("nmos", corner="nom", **kw).get_Idc(0.0, 0.5, 0.7)
    i_ss = create_transistor("nmos", corner="ss", **kw).get_Idc(0.0, 0.5, 0.7)
    i_ff = create_transistor("nmos", corner="ff", **kw).get_Idc(0.0, 0.5, 0.7)
    assert i_ss < i_nom < i_ff          # ss weaker, ff stronger (Vth shift)


def test_ota_ac_matches_ngspice(tmp_path):
    """The 5T OTA through ac_solve matches ngspice .ac on the equivalent netlist."""
    from core.ac_solver import ac_solve
    from core.circuit_loader import circuit_from_dict
    cfg = json.load(open(_CFG))
    spec = circuit_from_dict(cfg)
    freqs = np.logspace(3, 11, 121)
    ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf,
                  x0_guess=dict(cfg["dc_guesses"][0]),
                  model_types=spec.model_types, device_kwargs=spec.device_kwargs)
    assert ac is not None
    H = np.asarray(ac["gains"], float)
    a0 = 20 * np.log10(H.max())

    ncard = os.path.join(PDK_ROOT, "freepdk45", "models_nom", "NMOS_VTG.inc")
    pcard = os.path.join(PDK_ROOT, "freepdk45", "models_nom", "PMOS_VTG.inc")
    nm = {"VDD": "vdd", "GND": "0", "vinp": "vip", "vinn": "vin", "vbias": "vb"}
    lines = ["* fp45 5t ota", f'.include "{ncard}"', f'.include "{pcard}"',
             "vdd vdd 0 1.0", "vb vb 0 0.55",
             "vip vip 0 dc 0.55 ac 0.5", "vin vin 0 dc 0.55 ac -0.5"]
    for d in cfg["devices"]:
        mdl = "PMOS_VTG" if cfg["models"][d["name"]]["type"].endswith("pmos") else "NMOS_VTG"
        b = "vdd" if mdl == "PMOS_VTG" else "0"
        lines.append(f'm{d["name"].lower()} {nm.get(d["drain"], d["drain"])} '
                     f'{nm.get(d["gate"], d["gate"])} {nm.get(d["source"], d["source"])} '
                     f'{b} {mdl} w={d["W"]}u l={d["L"]}u')
    out_txt = str(tmp_path / "out.txt")
    lines.append("cl vout 0 0.2p")
    lines.append(f".control\nac dec 20 1k 100g\nwrdata {out_txt} vdb(vout)\n.endc\n.end")
    cir = str(tmp_path / "deck.cir")
    with open(cir, "w") as fh:
        fh.write("\n".join(lines))
    subprocess.run([_RUN, "-b", cir], capture_output=True, text=True)
    data = np.loadtxt(out_txt)
    a0_ng = data[:, 1].max()
    assert a0 == pytest.approx(a0_ng, abs=0.5)      # <0.5 dB vs ngspice's own .ac


def _ngspice_noise(card, model, W, L, vgs, vds, vb):
    """Direct ngspice ``.noise`` drain-current PSD → (S_thermal, S_flicker@1Hz) — the
    same CCVS transimpedance + A+B/f fit the characteriser uses, run standalone."""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out.txt")
        cir = os.path.join(td, "deck.cir")
        deck = (f"* n\n.include \"{card}\"\n"
                f"mn d g s b {model} w={W}u l={L}u\n"
                f"vd d 0 {vds} \nvg g 0 {vgs} ac 1\nvs s 0 0\nvb b 0 {vb}\n"
                f"hn out 0 vd 1\nrout out 0 1e12\n"
                f".control\nset filetype=ascii\nnoise v(out) vg dec 4 1 1e11\n"
                f"setplot noise1\nwrdata {out} onoise_spectrum\n.endc\n.end\n")
        with open(cir, "w") as fh:
            fh.write(deck)
        subprocess.run([_RUN, "-b", cir], capture_output=True, text=True)
        raw = np.loadtxt(out)
    f, sid = raw[:, 0], raw[:, 1] ** 2
    A, B = np.linalg.lstsq(np.column_stack([np.ones_like(f), 1.0 / f]), sid, rcond=None)[0]
    return float(A), float(B)


def test_noise_matches_ngspice():
    """get_noise_psd (grid-interpolated) tracks a direct ngspice .noise fit — the
    thermal (BSIM4 tnoimod, 45 nm velocity-sat excess) and 1/f coefficient are the
    real ngspice-C values, not an 8/3·kT·gm estimate."""
    from core.device_model import create_transistor
    card = os.path.join(PDK_ROOT, "freepdk45", "models_nom", "NMOS_VTG.inc")
    n = create_transistor("nmos", pdk="freepdk45", W=0.5, L=0.1, corner="nom")
    s_th, s_fl = n.get_noise_psd(0.0, 0.5, 0.6, frequency=1.0)
    th_ng, fl_ng = _ngspice_noise(card, "NMOS_VTG", 0.5, 0.1, 0.6, 0.5, 0.0)
    assert s_th > 0 and s_fl > 0
    assert s_th == pytest.approx(th_ng, rel=0.15)       # thermal: near-exact
    assert s_fl == pytest.approx(fl_ng, rel=0.25)       # flicker: coarse-grid slack
    # sanity: 45 nm velocity-sat excess makes thermal exceed the long-channel 8/3 kTgm
    kb = 1.380649e-23
    gm = n.get_ss_params(0.0, 0.5, 0.6)["gm"]
    assert s_th > (8.0 / 3.0) * kb * 300.15 * gm


def test_grid_cache_roundtrips(tmp_path):
    from core.ngspice_char import characterize
    card = os.path.join(PDK_ROOT, "freepdk45", "models_nom", "NMOS_VTG.inc")
    g1 = characterize(card, "NMOS_VTG", "nmos", 0.09, 0.05, "nom", vdd=1.0)
    g2 = characterize(card, "NMOS_VTG", "nmos", 0.09, 0.05, "nom", vdd=1.0)   # cache hit
    assert np.array_equal(g1.data["id"], g2.data["id"])
    assert g1.data["id"].shape == (len(g1.vsb), len(g1.vds), len(g1.vgs))


def test_extract_w_matches_true_w():
    """extract_w characterises one reference-W grid and linearly scales the actual W;
    for a wide device it reproduces the true per-W card to <2 % (BSIM4 W-linearity),
    which is what makes the dataset/optimizer W sweeps cheap."""
    from core.device_model import create_transistor
    vs, vd, vg = 0.0, 0.5, 0.55
    true = create_transistor("nmos", pdk="freepdk45", W=4.0, L=0.1, corner="nom")
    scal = create_transistor("nmos", pdk="freepdk45", W=4.0, L=0.1, corner="nom",
                             extract_w=1.0)                       # char @ W=1, scale ×4
    assert abs(scal.get_Idc(vs, vd, vg)) == pytest.approx(abs(true.get_Idc(vs, vd, vg)),
                                                          rel=0.02)
    assert scal.get_ss_params(vs, vd, vg)["gm"] == pytest.approx(
        true.get_ss_params(vs, vd, vg)["gm"], rel=0.02)


def test_temperature_shifts_current():
    """The temperature kwarg re-characterises the card at that °C (BSIM4 temp eqns).
    Above threshold, mobility roll-off dominates → Id falls with temperature."""
    from core.device_model import create_transistor
    kw = dict(pdk="freepdk45", W=1.0, L=0.1, corner="nom")
    i27 = abs(create_transistor("nmos", temperature=300.15, **kw).get_Idc(0.0, 0.5, 0.7))
    i90 = abs(create_transistor("nmos", temperature=363.15, **kw).get_Idc(0.0, 0.5, 0.7))
    assert i90 < i27                              # above-threshold: mobility wins
    assert i90 == pytest.approx(i27, rel=0.5)     # a physical shift, not a blow-up


def _fd_ota_diff_ugbw(freqs, H):
    """UGBW (last |H|=1 crossing) + passband-referenced PM from a differential H."""
    mag = np.abs(H)
    lf = np.log10(freqs)
    ph = np.unwrap(np.angle(H))
    above = np.where(mag >= 1.0)[0]
    idx = above[-1] + 1
    x0, x1 = np.log10(mag[idx - 1]), np.log10(mag[idx])
    fu = 10 ** (lf[idx - 1] - x0 * (lf[idx] - lf[idx - 1]) / (x1 - x0))
    pm = 180.0 + np.degrees(np.interp(np.log10(fu), lf, ph) - ph[int(np.argmax(mag))])
    return fu, pm


def test_fd_ota_ac_matches_ngspice(tmp_path):
    """The WHOLE FD-OTA — CMFB loop, AC-coupled input, Rs/CL — through ac_solve vs
    ngspice's own .ac on the equivalent netlist. Passband gain and PM match tightly;
    UGBW reads ~8 % high because the grid AC model omits drain/source junction caps
    (Cdb/Csb) that ngspice includes, so the crossing is pinned only to <12 %."""
    from core.ac_solver import ac_solve
    from core.circuit_loader import circuit_from_dict
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      "examples", "freepdk45_fd_ota.json")))
    spec = circuit_from_dict(cfg)
    freqs = np.logspace(3, 11, 141)
    ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf,
                  x0_guess=dict(cfg["dc_guesses"][0]),
                  model_types=spec.model_types, device_kwargs=spec.device_kwargs)
    H = ac["response"]
    our_gain = 20 * np.log10(np.abs(H).max())
    our_fu, our_pm = _fd_ota_diff_ugbw(freqs, H)

    ncard = os.path.join(PDK_ROOT, "freepdk45", "models_nom", "NMOS_VTG.inc")
    pcard = os.path.join(PDK_ROOT, "freepdk45", "models_nom", "PMOS_VTG.inc")
    nm = {"GND": "0", "VDD": "vdd", "VINP": "vip", "VINN": "vin", "VCMI": "vcmi",
          "VCM_REF": "vcmref", "VB_N": "vbn", "VB_CN": "vbcn", "VB_CP": "vbcp"}

    def net(n):
        return nm.get(n, n.lower())
    b = cfg["bias"]
    lines = ["* fp45 fd-ota ac", f'.include "{ncard}"', f'.include "{pcard}"',
             f"vdd vdd 0 {b['VDD']}", f"vcmi vcmi 0 {b['VCMI']}",
             f"vcmref vcmref 0 {b['VCM_REF']}", f"vbn vbn 0 {b['VB_N']}",
             f"vbcn vbcn 0 {b['VB_CN']}", f"vbcp vbcp 0 {b['VB_CP']}",
             f"vip vip 0 dc {b['VINP']} ac 0.5", f"vin vin 0 dc {b['VINN']} ac -0.5"]
    for d in cfg["devices"]:
        mdl = "PMOS_VTG" if cfg["models"][d["name"]]["type"].endswith("pmos") else "NMOS_VTG"
        bulk = "vdd" if mdl == "PMOS_VTG" else "0"
        lines.append(f"m{d['name'].lower()} {net(d['drain'])} {net(d['gate'])} "
                     f"{net(d['source'])} {bulk} {mdl} w={d['W']}u l={d['L']}u")
    for r in cfg.get("resistors", []):
        lines.append(f"r{r['name'].lower()} {net(r['a'])} {net(r['b'])} {r['R']}")
    for c in cfg.get("capacitors", []):
        lines.append(f"c{c['name'].lower()} {net(c['a'])} {net(c['b'])} {c['C']}")
    ns = " ".join(f"v({net(k)})={v}" for k, v in cfg["dc_guesses"][0].items())
    out_txt = str(tmp_path / "out.txt")
    lines.append(f".nodeset {ns}")
    lines.append(f".control\nset filetype=ascii\nac dec 30 1k 1e11\n"
                 f"wrdata {out_txt} vr(outp) vi(outp) vr(outn) vi(outn)\n.endc\n.end")
    cir = str(tmp_path / "deck.cir")
    with open(cir, "w") as fh:
        fh.write("\n".join(lines))
    subprocess.run([_RUN, "-b", cir], capture_output=True, text=True)
    raw = np.loadtxt(out_txt)
    f = raw[:, 0]
    Hng = (raw[:, 1] + 1j * raw[:, 3]) - (raw[:, 5] + 1j * raw[:, 7])
    ng_gain = 20 * np.log10(np.abs(Hng).max())
    ng_fu, ng_pm = _fd_ota_diff_ugbw(f, Hng)

    assert our_gain == pytest.approx(ng_gain, abs=1.0)       # passband gain: <1 dB
    assert our_pm == pytest.approx(ng_pm, abs=10.0)          # phase margin: <10 deg
    assert our_fu == pytest.approx(ng_fu, rel=0.12)          # UGBW: <12 % (junction caps)
    assert ng_fu > 1.0e8                                     # oracle clears the 0.1 GHz spec


def test_fd_ota_meets_spec():
    """The optimized FreePDK45 FD-OTA testbench meets its headline specs through the
    project's ac_solve: passband gain > 40 dB and UGBW > 100 MHz (single-pole rolloff)."""
    from core.ac_solver import ac_solve
    from core.circuit_loader import circuit_from_dict
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      "examples", "freepdk45_fd_ota.json")))
    spec = circuit_from_dict(cfg)
    freqs = np.logspace(3, 11, 141)
    ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf,
                  x0_guess=dict(cfg["dc_guesses"][0]),
                  model_types=spec.model_types, device_kwargs=spec.device_kwargs)
    assert ac is not None
    mag = np.abs(ac["response"])
    peak_dB = 20 * np.log10(mag.max())
    assert peak_dB > 40.0                                    # passband gain spec
    # UGBW = last downward crossing of |H| = 1 (AC-coupled band-pass response)
    above = np.where(mag >= 1.0)[0]
    idx = above[-1] + 1
    x0, x1 = np.log10(mag[idx - 1]), np.log10(mag[idx])
    lf = np.log10(freqs)
    fu = 10 ** (lf[idx - 1] - x0 * (lf[idx] - lf[idx - 1]) / (x1 - x0))
    assert fu > 1.0e8                                        # UGBW > 0.1 GHz spec


def test_run_ngspice_surfaces_failure(tmp_path):
    """A broken deck must raise RuntimeError carrying ngspice's own diagnostics,
    not fall through to a bare FileNotFoundError from np.loadtxt downstream.

    The deck ``.include``s a path that does not exist; ngspice aborts, so
    ``_run_ngspice`` re-raises with the deck purpose, return code, and the tail
    of ngspice's stderr/stdout so the root cause is visible at the call site."""
    from core.ngspice_char import _run_ngspice
    cir = str(tmp_path / "bad.cir")
    out_txt = str(tmp_path / "out.txt")
    missing = str(tmp_path / "does_not_exist.inc")
    deck = (f'* deliberately broken deck\n.include "{missing}"\n'
            f"mn d g s b NMOS_VTG w=1u l=0.1u\n"
            f"vd d 0 0.5\nvg g 0 0.6\nvs s 0 0\nvb b 0 0\n"
            f".control\nop\nwrdata {out_txt} @mn[id]\n.endc\n.end\n")
    with open(cir, "w") as fh:
        fh.write(deck)
    with pytest.raises(RuntimeError, match="ngspice"):
        _run_ngspice(cir, out_txt, timeout=30, what="dc-sweep")
