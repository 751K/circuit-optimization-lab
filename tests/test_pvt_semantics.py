"""Adversarial semantic tests for the PVT machinery work package.

Reviewer-side verification, fifth round. Attacks the contracts the 45-point PVT
campaign will lean on: corner-name strictness (a typo must not silently run
nom), PMOS polarity in the .op saturation check, the op margin knob, input-
referred noise correctness, integration self-consistency, and loop-gain phase
unwrapping on a two-pole loop. Skip-guarded on cards + ngspice.
"""
from pathlib import Path

import numpy as np
import pytest

from circuitopt.ngspice_char import ngspice_binary
from circuitopt.toolchain import pdk_root


ROOT = Path(__file__).resolve().parents[1]
_HAVE = (Path(pdk_root()) / "freepdk45" / "models_nom" / "NMOS_VTG.inc").is_file()
_HAVE = _HAVE and ngspice_binary() is not None
pytestmark = [
    pytest.mark.ngspice_oracle,
    pytest.mark.skipif(
        not _HAVE, reason="FreePDK45 cards / ngspice oracle not present"),
]


# ── corner-name strictness (campaign safety) ──────────────────────────────────
def test_grid_corner_names_are_case_insensitive_or_rejected():
    """'SF' must behave as 'sf' (or raise) — silently falling back to nom would
    poison a 45-point campaign with wrong-corner data."""
    from circuitopt.device_model import create_device
    lo = create_device("freepdk45.nmos", W=1.0, L=0.05, corner="sf").get_Idc(0.0, 0.6, 0.6)
    nom = create_device("freepdk45.nmos", W=1.0, L=0.05, corner="nom").get_Idc(0.0, 0.6, 0.6)
    assert lo != nom                        # sanity: sf really differs from nom
    try:
        hi = create_device("freepdk45.nmos", W=1.0, L=0.05, corner="SF").get_Idc(0.0, 0.6, 0.6)
    except ValueError:
        return                              # rejecting mixed case is acceptable
    assert hi == lo, "'SF' silently produced nom data instead of sf"


def test_grid_unknown_corner_rejected_not_nommed():
    """A typo like 'sx' must raise on the FreePDK45 grid path, matching the
    ngspice renderer's hard error — not silently produce nominal data."""
    from circuitopt.device_model import create_device
    with pytest.raises(ValueError):
        create_device("freepdk45.nmos", W=1.0, L=0.05, corner="sx")


def test_tt_alias_deck_matches_nom():
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.sar import _sar_config, sar_input_waveforms, sar_time_grid
    from circuitopt.device_factory import resolve_binding
    from circuitopt.ngspice_transient import render_freepdk45_transient_netlist
    spec = load_circuit_json(ROOT / "examples" / "freepdk45_sar3.json")
    cfg = _sar_config(spec)
    tgrid = sar_time_grid(spec, cfg)
    waveforms = sar_input_waveforms(spec, 0.5, [None] * 3, 0, config=cfg, tgrid=tgrid)
    def render(corner):
        topo, nf, c, mt, dk, _ = resolve_binding(spec.binding().at_corner(corner))
        return render_freepdk45_transient_netlist(
            spec.sizes, spec.bias, tgrid, topo=topo, output_path="/tmp/x.dat", nf=nf,
            inputs=waveforms, corner=c, model_types=mt, device_kwargs=dk).netlist
    assert render("tt") == render("nom")


# ── op_ngspice: PMOS polarity + margin knob ───────────────────────────────────
def _pmos_op_spec():
    from circuitopt.circuit_loader import circuit_from_dict
    return circuit_from_dict({
        "name": "pmos_regions", "solved": ["D", "DT"],
        "rails": {"VDD": "VDD", "GND": 0.0, "VG": "VG", "VDHIGH": "VDHIGH"},
        "bias": {"VDD": 1.0, "VG": 0.0, "VDHIGH": 0.95},
        "devices": [
            {"name": "MPDIODE", "drain": "D", "gate": "D", "source": "VDD",
             "W": 2.0, "L": 0.05},
            {"name": "MPTRIODE", "drain": "DT", "gate": "VG", "source": "VDD",
             "W": 2.0, "L": 0.05}],
        "models": {"MPDIODE": {"type": "freepdk45.pmos", "vb": 1.0},
                   "MPTRIODE": {"type": "freepdk45.pmos", "vb": 1.0}},
        "resistors": [["RD", "D", "GND", 5000.0], ["RT", "VDHIGH", "DT", 1.0]],
        "outputs": ["D"]})


def test_op_ngspice_pmos_polarity():
    """PMOS vds/vdsat are negative — sign handling must not flip the verdict."""
    from circuitopt.ngspice_ac import op_ngspice
    spec = _pmos_op_spec()
    op = op_ngspice(spec.sizes, spec.bias, topo=spec.topology,
                    model_types=spec.model_types, device_kwargs=spec.device_kwargs)
    assert op["MPDIODE"]["region_ok"] is True     # diode-connected PMOS: saturated
    assert op["MPTRIODE"]["region_ok"] is False   # |vds| = 50 mV: deep triode
    assert abs(op["MPDIODE"]["vds"]) >= abs(op["MPDIODE"]["vdsat"])


def test_op_ngspice_margin_flips_marginal_device():
    """margin is the saturation guard band: a saturated device must fail
    region_ok once margin exceeds its |vds|-|vdsat| headroom."""
    from circuitopt.ngspice_ac import op_ngspice
    spec = _pmos_op_spec()
    base = op_ngspice(spec.sizes, spec.bias, topo=spec.topology,
                      model_types=spec.model_types, device_kwargs=spec.device_kwargs)
    headroom = abs(base["MPDIODE"]["vds"]) - abs(base["MPDIODE"]["vdsat"])
    strict = op_ngspice(spec.sizes, spec.bias, topo=spec.topology,
                        model_types=spec.model_types, device_kwargs=spec.device_kwargs,
                        margin=headroom + 0.05)
    assert strict["MPDIODE"]["region_ok"] is False
    assert base["MPDIODE"]["region_ok"] is True


# ── noise_ngspice: input-referred + integration consistency ───────────────────
def test_noise_inoise_matches_transfer_referral():
    """Resistive divider (R1=R2): onoise = 4kT(R1||R2), |H|=1/2, so the
    input-referred PSD must be 4x the output PSD."""
    from circuitopt.topology import Topology
    from circuitopt.ngspice_ac import noise_ngspice
    R = 10e3
    topo = Topology(solved=["IN", "OUT"], devices=[], rails={"GND": 0.0}, outputs=("OUT",),
                    vsources=[("VIN", "IN", "GND", 0.0)],
                    resistors=[("R1", "IN", "OUT", R), ("R2", "OUT", "GND", R)])
    res = noise_ngspice({}, {}, topo=topo, out="OUT", src="VIN",
                        fstart=1.0, fstop=1e6, points=10, band=(10.0, 1e5))
    kT = 1.380649e-23 * 300.15
    assert np.mean(res["onoise_psd"]) == pytest.approx(4 * kT * (R / 2), rel=0.02)
    ratio = np.mean(np.asarray(res["inoise_psd"]) / np.asarray(res["onoise_psd"]))
    assert ratio == pytest.approx(4.0, rel=0.02)
    # integration self-consistency: rms^2 == trapz(psd) over the reported band
    f = np.asarray(res["freq"])
    mask = (f >= res["band"][0]) & (f <= res["band"][1])
    integral = np.trapezoid(np.asarray(res["onoise_psd"])[mask], f[mask])
    assert res["onoise_rms"] ** 2 == pytest.approx(integral, rel=0.05)


# ── loop_gain_ngspice: two-pole phase unwrap ──────────────────────────────────
def test_loop_gain_two_pole_pm_matches_analytic():
    """Two well-separated poles: PM ~ 45-60 deg region; catches unwrap and
    interpolation errors that a single-pole ~90 deg loop cannot see."""
    from circuitopt.topology import Topology
    from circuitopt.ngspice_ac import loop_gain_ngspice
    gm, R1, C1 = 1e-3, 1e5, 1e-12          # pole1 = 1.59 MHz, A0 = 100
    R2, C2 = 1e3, 1e-12                    # pole2 = 159 MHz (buffered by E-source)
    topo = Topology(solved=["N1", "N1B", "OUT", "CTRLA", "CTRLB"], devices=[],
                    rails={"GND": 0.0}, outputs=("OUT",),
                    resistors=[("RL", "N1", "GND", R1), ("R2", "N1B", "OUT", R2)],
                    capacitors=[("CL", "N1", "GND", C1), ("C2", "OUT", "GND", C2)],
                    vccs=[("G1", "N1", "GND", "CTRLB", "GND", gm)],
                    vcvs=[("Ebuf", "N1B", "GND", "N1", "GND", 1.0),
                          ("Efb", "CTRLA", "GND", "OUT", "GND", 1.0)],
                    vsources=[("Vinj", "CTRLB", "CTRLA", 0.0)])
    res = loop_gain_ngspice({}, {}, topo=topo, inject="Vinj",
                            fstart=1e3, fstop=1e10, points=40)
    A0 = gm * R1
    p1, p2 = 1.0 / (2 * np.pi * R1 * C1), 1.0 / (2 * np.pi * R2 * C2)
    f = np.geomspace(1e3, 1e10, 4001)
    T = A0 / ((1 + 1j * f / p1) * (1 + 1j * f / p2))
    ugf_a = f[np.argmin(np.abs(np.abs(T) - 1.0))]
    pm_a = 180.0 + np.degrees(np.angle(T[np.argmin(np.abs(np.abs(T) - 1.0))]))
    assert res["ugf"] == pytest.approx(ugf_a, rel=0.05)
    assert res["pm"] == pytest.approx(pm_a, abs=3.0)
    assert 30.0 < res["pm"] < 75.0          # far from the single-pole 90 deg regime


# ── ac_ngspice honors supply ──────────────────────────────────────────────────
def test_ac_ngspice_gain_tracks_supply():
    """FD-OTA gain at VDD=0.85 V must differ from 1.0 V (headroom compression);
    guards against the oracle ignoring the bias dict."""
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.ngspice_ac import ac_ngspice, ac_response, peak_gain_db
    spec = load_circuit_json(ROOT / "examples" / "freepdk45_fd_ota.json")
    seed = spec.topology.dc_guesses[0]
    b = spec.binding()

    def gain(vdd):
        bias = dict(spec.bias)
        scale = vdd / bias["VDD"]
        bias = {k: v * scale for k, v in bias.items()}
        res = ac_ngspice(spec.sizes, bias, topo=spec.topology,
                         acmag={"VINP": (0.5, 0.0), "VINN": (0.5, 180.0)},
                         fstart=1e3, fstop=1e9, points=10,
                         out_nodes=["OUTP", "OUTN"], nf=spec.nf,
                         model_types=b.model_types, device_kwargs=b.device_kwargs,
                         corner="nom", x0_guess=seed)
        return peak_gain_db(res["freq"], ac_response(res, "OUTP", "OUTN", vin=1.0))

    g10, g085 = gain(1.0), gain(0.85)
    assert np.isfinite(g10) and np.isfinite(g085)
    assert abs(g10 - g085) > 0.5            # supply must visibly move the gain
