"""Silicon (SKY130) chopper amplifier — PSS orchestration on the OSDI engine.

The testbench (``examples/sky130_chopper.json``) is an NMOS input chopper →
NMOS diff pair with diode-connected PMOS loads → NMOS output chopper → hold
caps, clocked at 250 kHz by square-wave vsources. The demodulated differential
output is DC at ``gain·(VINP−VINN)`` with gain = −gm1/gm3 ≈ −1.77.

Validates the silicon periodic-analysis chain three ways: (1) PSS converges
and its orbit matches the settled long transient; (2) the chopper conversion
gain matches the static (phase-frozen) small-signal gain; (3) ngspice running
the same BSIM4 cards + the same ``.osdi`` agrees on the gain (model == oracle).

Needs the external toolchain; skips cleanly without it.
"""
import json
import os
import subprocess

import numpy as np
import pytest

from core.ngspice_char import ngspice_binary
from core.osdi_device import openvaf_binary

PDK_ROOT = os.environ.get("PDK_ROOT", "/Volumes/MacoutDsik/pdk")
_NGSPICE_LIB = os.path.join(PDK_ROOT, "sky130A/libs.tech/ngspice/sky130.lib.spice")
VAF_ROOT = os.environ.get("OPENVAF_ROOT", "/Volumes/MacoutDsik/Code/VAF/OpenVAF-Reloaded")
_TOOLS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")
_VACOMPILE = os.path.join(_TOOLS, "vacompile.sh")
RUN_NGSPICE = os.path.join(_TOOLS, "run-ngspice.sh")
_HAVE = os.path.exists(_NGSPICE_LIB) and openvaf_binary() is not None

pytestmark = pytest.mark.skipif(
    not _HAVE, reason="SKY130 PDK / OpenVAF toolchain not present")

_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "examples", "sky130_chopper.json")
DELTA = 0.01                       # VINP - VINN differential input [V]
PERIOD = 1.0 / 250e3
NPTS = 401


@pytest.fixture(scope="module")
def spec():
    from core.circuit_loader import circuit_from_dict
    with open(_CFG_PATH) as fh:
        return circuit_from_dict(json.load(fh))


@pytest.fixture(scope="module")
def suite(spec):
    from core.analysis_dispatch import run_analysis_suite
    return run_analysis_suite(spec, selected=["transient", "pss", "pac", "pnoise"])


def _gain(nodes, n_last):
    vod = nodes["OUTP"] - nodes["OUTN"]
    return float(vod[-n_last:].mean() / DELTA), vod


def test_pss_converges(suite):
    ps = suite["pss"]
    assert ps["converged"] is True
    assert ps["nfail"] == 0
    assert ps["residual_norm"] < 2e-5


def test_transient_and_pss_gain_agree(suite):
    n_per = NPTS - 1
    g_tr, vod_tr = _gain(suite["transient"]["nodes"], n_per)
    g_ps, vod_ps = _gain(suite["pss"]["nodes"], NPTS)
    # PSS orbit == the settled last period of the 10-period transient
    # (vod_ps[0] is t=0 ≡ the period boundary; the transient slice starts one
    #  step after it, so compare vod_ps[1:] point-for-point). The clock-edge
    #  samples slew hard, so residual-tol-level state differences amplify
    #  there — 1 mV pointwise; away from edges agreement is ~µV.
    assert abs(g_tr - g_ps) < 0.01
    assert np.max(np.abs(vod_ps[1:] - vod_tr[-n_per:])) < 1e-3
    # inverting diode-load gain, ~ -1.73 at this bias
    assert -2.2 < g_tr < -1.3
    # demodulated output is DC apart from the commutation glitches (both
    # output switches partially conduct during the 100 ns clock edges, briefly
    # pulling OUTP/OUTN together) — away from edges the ripple is ~0.1 mV
    assert np.ptp(vod_ps) < 0.3 * abs(vod_ps.mean())


def test_pac_baseband_matches_large_signal(suite):
    """Time-domain PAC(fm→0) == the large-signal PSS conversion gain."""
    g_ps, _ = _gain(suite["pss"]["nodes"], NPTS)
    gains = np.asarray(suite["pac"]["gains"])
    freqs = np.asarray(suite["pac"]["freqs"])
    assert freqs[0] <= 1e3
    assert gains[0] == pytest.approx(abs(g_ps), rel=0.01)   # <1% vs large-signal
    # demodulated baseband is flat well below f_chop
    in_band = gains[freqs <= 1e4]
    assert np.ptp(in_band) < 0.01 * gains[0]


def test_pnoise_runs_and_is_physical(suite):
    pn = suite["pnoise"]
    asd = np.asarray(pn["out_asd"])
    assert np.all(np.isfinite(asd)) and np.all(asd > 0)
    # flat demodulated output noise across the band (the tiny W=2/L=0.15
    # switches are strongly flicker-dominated with an above-band 1/f corner)
    assert 1e-5 < asd[0] < 1e-2
    assert np.ptp(asd) < 0.2 * asd[0]


def test_dataset_chopper_labels(tmp_path, suite):
    """The chopper's periodic figures of merit flow into surrogate-dataset labels.

    Pins the single LHS candidate to the validated nominal design (min==max
    ranges), so the pss/pac/pnoise labels must reproduce the suite fixture's
    numbers — same solvers, same validated ``analyses`` settings, threaded
    through ``build_dataset``'s suite runner."""
    import core.dataset as ds
    with open(_CFG_PATH) as fh:
        cfg = json.load(fh)
    for var in cfg["explore"]["variables"].values():
        var["min"] = var["max"] = 0.5 * (var["min"] + var["max"])   # = nominal
    p = tmp_path / "chopper_ds.json"
    p.write_text(json.dumps(cfg))
    dataset = ds.run_from_config(str(p), n=1, seed=0,
                                 label_groups=("pss", "pac", "pnoise"))
    assert dataset["manifest"]["labels"] == list(
        ds.PSS_LABELS + ds.PAC_LABELS + ds.PNOISE_LABELS)
    row = dataset["rows"][0]
    m = row["metrics"]
    assert row["status"]["dc_converged"]
    assert m["pss_converged"] == 1.0
    g_ps, _ = _gain(suite["pss"]["nodes"], NPTS)
    assert m["pac_gain"] == pytest.approx(abs(g_ps), rel=0.02)      # conversion gain
    assert m["pac_gain_dB"] == pytest.approx(20 * np.log10(m["pac_gain"]), abs=1e-9)
    assert m["pnoise_irn_uV"] is not None and m["pnoise_irn_uV"] > 0.0
    # flat in-band gain ⇒ output noise ≈ gain × input-referred noise
    assert m["pnoise_out_uV"] == pytest.approx(m["pnoise_irn_uV"] * m["pac_gain"],
                                               rel=0.05)


def test_frozen_clock_lti_oracles(spec):
    """With constant clocks the PAC/PNoise HB folds must reduce EXACTLY to the
    stationary AC / noise analyses — validates the silicon linearization, HB
    conversion, adjoint, and noise fold end to end."""
    import copy
    from core.analysis_dispatch import run_analysis_suite
    from core.circuit_loader import circuit_from_dict
    from core.pac_solver import pac_solve
    from core.pnoise_solver import pnoise_solve
    with open(_CFG_PATH) as fh:
        cfg = copy.deepcopy(json.load(fh))
    cfg["periodic"]["inputs"]["clk"] = {"type": "constant", "value": 1.8}
    cfg["periodic"]["inputs"]["clkb"] = {"type": "constant", "value": 0.0}
    fspec = circuit_from_dict(cfg)
    ps = run_analysis_suite(fspec, selected=["pss"])["pss"]
    assert ps["converged"]
    freqs = np.logspace(2, 5, 5)
    drive = {"vinp": 0.5, "vinn": -0.5}
    pac_hb = pac_solve(fspec.sizes, fspec.bias, freqs, pss_result=ps,
                       input_drive=drive, nf=fspec.nf, lti_fast_path=False)
    pac_ac = pac_solve(fspec.sizes, fspec.bias, freqs, pss_result=ps,
                       input_drive=drive, nf=fspec.nf, lti_fast_path=True)
    assert np.allclose(pac_hb["gains"], pac_ac["gains"], rtol=1e-3)
    pn_hb = pnoise_solve(fspec.sizes, fspec.bias, freqs, pss_result=ps,
                         input_drive=drive, nf=fspec.nf, lti_fast_path=False,
                         max_sideband=6, band=(100.0, 1e4))
    pn_ac = pnoise_solve(fspec.sizes, fspec.bias, freqs, pss_result=ps,
                         input_drive=drive, nf=fspec.nf, lti_fast_path=True,
                         band=(100.0, 1e4))
    assert np.allclose(pn_hb["out_asd"], pn_ac["out_asd"], rtol=1e-3)


@pytest.mark.skipif(ngspice_binary() is None,
                    reason="OSDI-enabled ngspice not present")
def test_chopper_gain_matches_ngspice(spec, suite, tmp_path):
    from core.device_factory import build_devices
    from core.osdi_device import compile_va
    from core.sky130_model import _BSIM4_VA
    with open(_CFG_PATH) as fh:
        cfg = json.load(fh)
    wrappers = build_devices(spec.sizes, nf=spec.nf, topo=spec.topology,
                             model_types=spec.model_types,
                             device_kwargs=spec.device_kwargs)
    cards, dev_model = {}, {}
    for name, w in wrappers.items():
        key = tuple(sorted(w._osdi_card.items()))
        if key not in cards:
            cards[key] = (f"m{len(cards)}", w._osdi_card)
        dev_model[name] = cards[key][0]

    def card_lines(card):
        lines, cur = [], "+"
        for k, v in card.items():
            tok = f" {k}={v:g}"
            if len(cur) + len(tok) > 110:
                lines.append(cur)
                cur = "+"
            cur += tok
        lines.append(cur)
        return "\n".join(lines)

    node_map = {"GND": "0", "VDD": "vdd", "VINP": "vinp", "VINN": "vinn",
                "VBIAS": "vbias"}

    def nm(node):
        return node_map.get(node, node.lower())

    vdd, vb = cfg["bias"]["VDD"], cfg["bias"]["VB"]
    tstop = 10 * PERIOD
    out_csv = str(tmp_path / "out.csv")
    lines = [f"* sky130 chopper (osdi)\n.control\npre_osdi {compile_va(_BSIM4_VA)}\n.endc",
             f"vdd vdd 0 dc {vdd}",
             f"vinp vinp 0 dc {cfg['bias']['VINP']:g}",
             f"vinn vinn 0 dc {cfg['bias']['VINN']:g}",
             f"vb vbias 0 dc {vb}",
             f"vck clk 0 pulse({vdd} 0 {PERIOD/2:g} 100n 100n {PERIOD/2:g} {PERIOD:g})",
             f"vckb clkb 0 pulse(0 {vdd} {PERIOD/2:g} 100n 100n {PERIOD/2:g} {PERIOD:g})"]
    for d in cfg["devices"]:
        b = "vdd" if cfg["models"][d["name"]]["type"].endswith("pmos") else "0"
        lines.append(f"N{d['name'].lower()} {nm(d['drain'])} {nm(d['gate'])} "
                     f"{nm(d['source'])} {b} {dev_model[d['name']]}")
    for c in cfg["capacitors"]:
        lines.append(f"c{c['name'].lower()} {nm(c['a'])} {nm(c['b'])} {c['C']:g}")
    for _, (mname, card) in cards.items():
        lines.append(f".model {mname} bsim4va\n{card_lines(card)}")
    lines.append(f".control\ntran {PERIOD/400:g} {tstop:g}\n"
                 f"wrdata {out_csv} v(outp) v(outn)\n.endc\n.end")
    cir = str(tmp_path / "deck.cir")
    with open(cir, "w") as fh:
        fh.write("\n".join(lines))
    subprocess.run([RUN_NGSPICE, "-b", cir], capture_output=True,
                   text=True, timeout=600)
    data = np.loadtxt(out_csv)
    t_ng, vod_ng = data[:, 0], data[:, 1] - data[:, 3]
    sel = t_ng >= (tstop - PERIOD)
    gain_ng = float(vod_ng[sel].mean() / DELTA)
    vod_ps = suite["pss"]["nodes"]["OUTP"] - suite["pss"]["nodes"]["OUTN"]
    gain_ps = float(vod_ps.mean() / DELTA)
    assert gain_ps == pytest.approx(gain_ng, rel=0.01)    # model == oracle
