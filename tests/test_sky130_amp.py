"""Silicon (SKY130) device through the actual ac_solve / noise_analysis engine.

Phase A end-to-end: a common-source SKY130 PMOS amp (source at VDD → matches the
solver's +Id-at-drain convention; bulk at VDD via ``vb``) solves DC, its AC gain
equals the analytic ``gm*(RL||ro)``, and its output noise equals
``device_S_id*Zout^2 + resistor``. Needs the SKY130 PDK + OpenVAF + ngspice
(external drive), so it skips cleanly in CI.
"""
import os

import numpy as np
import pytest

import core
from core.ac_solver import ac_solve
from core.noise_solver import noise_analysis
from core.osdi_device import openvaf_binary
from core.topology import Topology

_PDK_ROOT = os.environ.get("PDK_ROOT", "/Volumes/MacoutDsik/pdk")
_HAVE = os.path.exists(os.path.join(_PDK_ROOT, "sky130A/libs.tech/ngspice/sky130.lib.spice")) \
    and openvaf_binary() is not None
pytestmark = pytest.mark.skipif(not _HAVE, reason="SKY130 PDK / OpenVAF toolchain not present")

_RL = 5e3
_SIZES = {"M1": (10.0, 0.15)}
_BIAS = {"VDD": 1.8, "VIN": 0.5}
_MT = {"M1": "sky130.pmos"}
_DK = {"M1": {"vb": 1.8}}


def _amp():
    return Topology(
        solved=["vout"], devices=[("M1", "vout", "vin", "VDD")],
        rails={"VDD": "VDD", "GND": 0.0, "vin": "VIN"},
        resistors=[("RL", "vout", "GND", _RL)],
        input_drives={"M1": 1.0}, outputs=("vout",))


def test_silicon_ac_gain_matches_analytic():
    res = ac_solve(_SIZES, _BIAS, np.logspace(0, 8, 60), topo=_amp(),
                   model_types=_MT, device_kwargs=_DK)
    assert res is not None                               # DC converged on silicon
    gm = res["ss"]["M1"]["gm"]
    ro = 1.0 / res["ss"]["M1"]["gds"]
    av_analytic = gm * (_RL * ro / (_RL + ro))
    assert res["gains"][0] == pytest.approx(av_analytic, rel=1e-6)   # solver AC == device
    assert 2.0 < 20 * np.log10(res["gains"][0]) < 20.0              # a physical gain


def test_silicon_noise_matches_device_psd():
    freqs = np.logspace(1, 7, 60)
    r = noise_analysis(_SIZES, _BIAS, freqs, topo=_amp(), model_types=_MT, device_kwargs=_DK)
    assert r is not None
    vout = r["dc"]["vout"]
    pf = core.create_transistor("pmos", pdk="sky130", W=10.0, L=0.15, vb=1.8)
    ro = 1.0 / pf.get_ss_params(1.8, vout, _BIAS["VIN"])["gds"]
    zout = _RL * ro / (_RL + ro)
    s_th, s_fl1 = pf.get_noise_psd(1.8, vout, _BIAS["VIN"], 1.0)
    kT = 1.380649e-23 * 300.15
    expected = (s_th + s_fl1 / freqs + 4 * kT / _RL) * zout ** 2   # device + resistor
    assert np.allclose(r["out_psd"], expected, rtol=2e-3)
    assert np.all(np.isfinite(r["irn_psd"])) and np.all(r["irn_psd"] > 0)


def test_nmos_cs_gain_matches_analytic():
    """NMOS (source at GND) converges after the kcl_sign fix, gain == gm*(RL||ro)."""
    csn = Topology(solved=["vout"], devices=[("M1", "vout", "vin", "GND")],
                   rails={"VDD": "VDD", "GND": 0.0, "vin": "VIN"},
                   resistors=[("RL", "VDD", "vout", 3e3)],
                   input_drives={"M1": 1.0}, outputs=("vout",))
    r = ac_solve({"M1": (10.0, 0.5)}, {"VDD": 1.8, "VIN": 0.8}, np.logspace(0, 8, 60),
                 topo=csn, model_types={"M1": "sky130.nmos"})
    assert r is not None                                     # NMOS DC now converges
    ro = 1.0 / r["ss"]["M1"]["gds"]
    assert r["gains"][0] == pytest.approx(r["ss"]["M1"]["gm"] * (3e3 * ro / (3e3 + ro)),
                                          rel=1e-5)


def test_complementary_5t_ota_differential_gain():
    """Complementary silicon OTA (NMOS diff pair + PMOS mirror + NMOS tail): the
    differential gain equals gm1*(ro2||ro4) — nmos+pmos with correct signs."""
    ota = Topology(
        solved=["tail", "n1", "vout"],
        devices=[("M1", "n1", "vinp", "tail"), ("M2", "vout", "vinn", "tail"),
                 ("M3", "n1", "n1", "VDD"), ("M4", "vout", "n1", "VDD"),
                 ("M5", "tail", "vbias", "GND")],
        rails={"VDD": "VDD", "GND": 0.0, "vinp": "VCM", "vinn": "VCM", "vbias": "VB"},
        input_drives={"M1": 1.0, "M2": -1.0}, outputs=("vout",),
        dc_guesses=[{"tail": 0.30, "n1": 0.95, "vout": 0.90}])
    mt = {"M1": "sky130.nmos", "M2": "sky130.nmos", "M5": "sky130.nmos",
          "M3": "sky130.pmos", "M4": "sky130.pmos"}
    dk = {"M3": {"vb": 1.8}, "M4": {"vb": 1.8}}
    szs = {"M1": (20.0, 0.5), "M2": (20.0, 0.5), "M3": (10.0, 0.5),
           "M4": (10.0, 0.5), "M5": (40.0, 0.5)}
    r = ac_solve(szs, {"VDD": 1.8, "VCM": 0.9, "VB": 0.75}, np.logspace(0, 8, 60),
                 topo=ota, model_types=mt, device_kwargs=dk)
    assert r is not None                                     # multi-node OTA DC converges
    rout = 1.0 / (r["ss"]["M2"]["gds"] + r["ss"]["M4"]["gds"])
    assert r["gains"][0] == pytest.approx(r["ss"]["M1"]["gm"] * rout, rel=0.05)
    assert 20 * np.log10(r["gains"][0]) > 25.0              # a real OTA gain (~35 dB)


def test_silicon_transient_settles_to_dc_with_rc_tau():
    """Phase B: pure-Python backward-Euler OSDI transient settles to the DC op with
    the right RC time constant."""
    from core.osdi_transient import cs_transient
    pmos = core.create_transistor("pmos", pdk="sky130", W=10.0, L=0.15, vb=1.8)
    CL = 1e-12
    tg = np.linspace(0, 20e-9, 160)
    vout = cs_transient(pmos, 1.8, _RL, CL, lambda t: 0.5 if t <= 0 else 0.51, tg)
    # settles to the DC operating point at the stepped input
    dc = ac_solve(_SIZES, {"VDD": 1.8, "VIN": 0.51}, np.array([1.0]),
                  topo=_amp(), model_types=_MT, device_kwargs=_DK)
    assert vout[-1] == pytest.approx(dc["dc_op"]["vout"], rel=2e-3)
    # time constant ~ (RL||ro)*CL (BE 1st-order → few-% tolerance)
    ro = 1.0 / pmos.get_ss_params(1.8, vout[-1], 0.51)["gds"]
    reff = _RL * ro / (_RL + ro)
    frac = (vout - vout[0]) / (vout[-1] - vout[0])
    tau_meas = np.interp(1 - np.exp(-1), frac, tg)
    assert tau_meas == pytest.approx(reff * CL, rel=0.1)
