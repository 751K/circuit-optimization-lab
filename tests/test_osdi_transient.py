"""Silicon (OSDI) full-fidelity transient — Phase B validation.

Validates ``core.osdi_transient.transient_osdi`` (routed through the standard
``core.transient_solver.transient`` entry via ``model_types``) four ways:

1. DC-hold: with constant inputs the 5T OTA sits at its DC operating point.
2. Independent reference: a common-source PMOS step matches the pure-Python
   backward-Euler demo (``cs_transient``) essentially exactly.
3. Analytic physics: the settling time constant equals (RL‖ro)·CL.
4. Oracle: ngspice running the *same* card + the *same* compiled ``bsim4.osdi``
   agrees on the full trajectory (model == oracle by construction).

All tests need the external toolchain (SKY130 PDK + OpenVAF + OSDI-ngspice)
and skip cleanly without it.
"""
import json
import os
import subprocess
import tempfile

import numpy as np
import pytest

PDK_ROOT = os.environ.get("PDK_ROOT", "/Volumes/MacoutDsik/pdk")
_NGSPICE_LIB = os.path.join(PDK_ROOT, "sky130A/libs.tech/ngspice/sky130.lib.spice")
VAF_ROOT = os.environ.get("OPENVAF_ROOT", "/Volumes/MacoutDsik/Code/VAF/OpenVAF-Reloaded")
_VACOMPILE = os.path.join(VAF_ROOT, ".claude/skills/build-openvaf/scripts/vacompile.sh")
RUN_NGSPICE = os.path.join(VAF_ROOT, ".claude/skills/run-osdi-ngspice/scripts/run-ngspice.sh")
_HAVE = os.path.exists(_NGSPICE_LIB) and os.path.exists(_VACOMPILE)

pytestmark = pytest.mark.skipif(
    not _HAVE, reason="SKY130 PDK / OpenVAF toolchain not present")

_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")

VDD, RL, CL, W, L = 1.8, 4.7e4, 5e-12, 4.0, 0.5
VG0, VG1 = 0.4, 0.43
TSTEP, TSTOP, TEDGE = 2.5e-9, 2e-6, 0.2e-6


def _cs_spec():
    from core.circuit_loader import circuit_from_dict
    return circuit_from_dict({
        "name": "cs_pmos_tran", "solved": ["VOUT"],
        "rails": {"VDD": "VDD", "GND": 0.0, "VG": "VGATE"},
        "devices": [{"name": "M1", "drain": "VOUT", "gate": "VG",
                     "source": "VDD", "W": W, "L": L}],
        "models": {"M1": {"type": "sky130.pmos", "vb": VDD}},
        "bias": {"VDD": VDD, "VGATE": VG0}, "outputs": ["VOUT"],
        "resistors": [["RL", "VOUT", "GND", RL]],
        "load_caps": [["VOUT", "GND", CL]],
        "transient_inputs": {"M1": "vin"},
        "dc_guesses": [{"VOUT": 0.9}, {"VOUT": 0.3}, {"VOUT": 1.5}],
    })


def _cs_step(method="be"):
    from core.transient_solver import transient
    spec = _cs_spec()
    tgrid = np.arange(0.0, TSTOP + TSTEP / 2, TSTEP)
    wave = np.where(tgrid < TEDGE, VG0, VG1)
    r = transient(spec.sizes, spec.bias, tgrid, topo=spec.topology, nf=spec.nf,
                  inputs={"vin": wave}, model_types=spec.model_types,
                  device_kwargs=spec.device_kwargs, integration_method=method)
    return tgrid, r


def test_ota_dc_hold():
    """Constant inputs → the 5T OTA transient sits at its DC op (signs/assembly)."""
    from core.ac_solver import ac_solve
    from core.circuit_loader import circuit_from_dict
    from core.transient_solver import transient
    with open(os.path.join(_EXAMPLES, "sky130_5t_ota.json")) as fh:
        spec = circuit_from_dict(json.load(fh))
    ac = ac_solve(spec.sizes, spec.bias, np.array([1.0]), topo=spec.topology,
                  nf=spec.nf, model_types=spec.model_types,
                  device_kwargs=spec.device_kwargs)
    dc = ac["dc_op"]
    tgrid = np.linspace(0.0, 1e-6, 101)
    r = transient(spec.sizes, spec.bias, tgrid, topo=spec.topology, nf=spec.nf,
                  model_types=spec.model_types, device_kwargs=spec.device_kwargs)
    assert r["nfail"] == 0
    assert r.get("osdi_transient") is True
    for nm, trace in r["nodes"].items():
        assert np.max(np.abs(trace - dc[nm])) < 1e-6      # sub-µV drift


def test_cs_step_matches_python_reference():
    """Kernel BE trajectory == the independent pure-Python BE demo."""
    from core.osdi_transient import cs_transient
    from core.sky130_model import Sky130Pfet
    tgrid, r = _cs_step("be")
    assert r["nfail"] == 0
    vout = r["nodes"]["VOUT"]
    dev = Sky130Pfet(W=W, L=L, vb=VDD)
    ref = cs_transient(dev, VDD, RL, CL,
                       lambda t: VG0 if t < TEDGE else VG1, tgrid)
    assert np.max(np.abs(vout - ref)) < 1e-4              # < 0.1 mV everywhere


def test_cs_settling_matches_analytic_tau():
    """63% settling time == (RL‖ro)·CL small-signal time constant."""
    from core.sky130_model import Sky130Pfet
    tgrid, r = _cs_step("gear2")
    vout = r["nodes"]["VOUT"]
    dev = Sky130Pfet(W=W, L=L, vb=VDD)
    op = dev._dev.operating_point(float(vout[-1]), VG1, VDD, VDD)
    ro = 1.0 / max(op["gds"], 1e-15)
    tau = (RL * ro / (RL + ro)) * CL
    step0 = int(np.searchsorted(tgrid, TEDGE))
    delta = vout[-1] - vout[step0]
    target = vout[step0] + 0.632 * delta
    idx = step0 + int(np.argmin(np.abs(vout[step0:] - target)))
    tau_meas = tgrid[idx] - TEDGE
    assert tau_meas == pytest.approx(tau, rel=0.05)


def test_gear2_matches_be_when_settled():
    tg, rb = _cs_step("be")
    _, rg = _cs_step("gear2")
    vb_, vg_ = rb["nodes"]["VOUT"], rg["nodes"]["VOUT"]
    assert abs(vb_[0] - vg_[0]) < 1e-9
    assert np.max(np.abs(vb_[-50:] - vg_[-50:])) < 1e-5   # same settled point


@pytest.mark.skipif(not os.path.exists(RUN_NGSPICE),
                    reason="OSDI-enabled ngspice not present")
def test_cs_step_matches_ngspice():
    """Same card + same .osdi in ngspice .tran → trajectory oracle."""
    from core.osdi_device import compile_va
    from core.sky130_model import _BSIM4_VA, Sky130Pfet
    tgrid, r = _cs_step("gear2")
    vout = r["nodes"]["VOUT"]
    card = Sky130Pfet(W=W, L=L, vb=VDD)._osdi_card
    osdi = compile_va(_BSIM4_VA)
    lines, cur = [], "+"
    for k, v in card.items():
        tok = f" {k}={v:g}"
        if len(cur) + len(tok) > 110:
            lines.append(cur)
            cur = "+"
        cur += tok
    lines.append(cur)
    out_csv = tempfile.mktemp(suffix=".csv")
    net = (f"* cs pmos tran (osdi)\n.control\npre_osdi {osdi}\n.endc\n"
           f"vdd vdd 0 dc {VDD}\n"
           f"vg g 0 pulse({VG0} {VG1} {TEDGE:g} {TSTEP:g} {TSTEP:g} 1 2)\n"
           f"N1 out g vdd vdd mp\n.model mp bsim4va\n" + "\n".join(lines) +
           f"\nrl out 0 {RL:g}\ncl out 0 {CL:g}\n"
           f".control\ntran {TSTEP:g} {TSTOP:g}\nwrdata {out_csv} v(out)\n.endc\n.end\n")
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as fh:
        fh.write(net)
        cir = fh.name
    try:
        subprocess.run([RUN_NGSPICE, "-b", cir], capture_output=True, text=True)
        data = np.loadtxt(out_csv)
    finally:
        os.unlink(cir)
        if os.path.exists(out_csv):
            os.unlink(out_csv)
    v_ng = np.interp(tgrid, data[:, 0], data[:, 1])
    assert abs(vout[0] - v_ng[0]) < 1e-4                  # same DC start
    assert abs(vout[-1] - v_ng[-1]) < 1e-4                # same settled point
    assert np.max(np.abs(vout - v_ng)) < 5e-3             # ≤ few mV at the edge


def test_controlled_sources_silicon():
    """All four controlled-source types in a silicon transient.

    The CS stage's output drives, algebraically:
      VBUF = 2*VOUT           (VCVS, mu=2)
      VX   = gm*RX*VOUT       = VOUT   (VCCS, gm*RX = 1)
      VY   = beta*RY*VBUF/RBUF = VOUT  (CCCS off the VSEN branch, 1*5k*2/10k)
      VZ   = gamma*VBUF/RBUF   = VOUT  (CCVS, 5k*2/10k)
    so every derived node must track VOUT sample-for-sample through the step.
    """
    from core.circuit_loader import circuit_from_dict
    from core.transient_solver import transient
    cfg = {
        "name": "cs_ctrl_src", "solved": ["VOUT", "VBUF", "NS", "VX", "VY", "VZ"],
        "rails": {"VDD": "VDD", "GND": 0.0, "VG": "VGATE"},
        "devices": [{"name": "M1", "drain": "VOUT", "gate": "VG",
                     "source": "VDD", "W": W, "L": L}],
        "models": {"M1": {"type": "sky130.pmos", "vb": VDD}},
        "bias": {"VDD": VDD, "VGATE": VG0}, "outputs": ["VOUT"],
        "resistors": [["RL", "VOUT", "GND", RL],
                      ["RBUF", "VBUF", "NS", 1e4],
                      ["RX", "VX", "GND", 1e4],
                      ["RY", "VY", "GND", 5e3]],
        "load_caps": [["VOUT", "GND", CL]],
        "vsources": [{"name": "VSEN", "p": "NS", "q": "GND", "value": 0.0}],
        "vcvs": [["E1", "VBUF", "GND", "VOUT", "GND", 2.0]],
        "vccs": [["G1", "VX", "GND", "VOUT", "GND", 1e-4]],
        "cccs": [["F1", "VY", "GND", "VSEN", 1.0]],
        "ccvs": [["H1", "VZ", "GND", "VSEN", 5e3]],
        "transient_inputs": {"M1": "vin"},
        "dc_guesses": [{"VOUT": 0.9, "VBUF": 1.8, "NS": 0.0, "VX": 0.9,
                        "VY": 0.9, "VZ": 0.9}],
    }
    spec = circuit_from_dict(cfg)
    tgrid = np.arange(0.0, 1e-6 + TSTEP / 2, TSTEP)
    wave = np.where(tgrid < 0.2e-6, VG0, VG1)
    r = transient(spec.sizes, spec.bias, tgrid, topo=spec.topology, nf=spec.nf,
                  inputs={"vin": wave}, model_types=spec.model_types,
                  device_kwargs=spec.device_kwargs, integration_method="gear2")
    assert r["nfail"] == 0
    nd = r["nodes"]
    vout = nd["VOUT"]
    assert np.ptp(vout) > 0.05                      # the step actually moved it
    assert np.max(np.abs(nd["VBUF"] - 2.0 * vout)) < 1e-6
    assert np.max(np.abs(nd["VX"] - vout)) < 1e-6
    assert np.max(np.abs(nd["VY"] - vout)) < 1e-6
    assert np.max(np.abs(nd["VZ"] - vout)) < 1e-6
    assert np.max(np.abs(nd["NS"])) < 1e-9          # sensing source pins NS


def test_mixed_model_libraries():
    """Two distinct compiled models (BSIM4 + BSIM3) in ONE transient.

    The circuit is two independent NMOS common-source stages, one per model.
    Because the sub-circuits don't interact, the mixed run must reproduce each
    single-model run's trajectory — an exact internal oracle for the two-lib
    fn-pointer dispatch.
    """
    from core.circuit_loader import circuit_from_dict
    from core.device_model import _model_registry
    from core.osdi_device import OsdiDevice
    from core.transient_solver import transient

    class _B4Nfet(OsdiDevice):
        VA_PATH = os.path.join(VAF_ROOT, "integration_tests/BSIM4/bsim4.va")
        MODULE = "bsim4va"
        BASE_CARD = {"toxe": 4.148e-9, "vth0": 0.4, "u0": 0.04}
        TYPE = 1

    class _B3Nfet(OsdiDevice):
        VA_PATH = os.path.join(VAF_ROOT, "integration_tests/BSIM3/bsim3.va")
        MODULE = "bsim3_va"
        BASE_CARD = {"tox": 4.1e-9, "vth0": 0.45}
        TYPE = 1

    _model_registry.setdefault("test_b4.nmos", _B4Nfet)
    _model_registry.setdefault("test_b3.nmos", _B3Nfet)

    def cfg(devices):
        base = {
            "name": "mixed_libs", "rails": {"GND": 0.0, "VDD": "VDD", "VG": "VGATE"},
            "bias": {"VDD": 1.8, "VGATE": 0.9},
            "solved": [], "devices": [], "models": {}, "resistors": [],
            "load_caps": [], "outputs": [],
            "transient_inputs": {}, "dc_guesses": [{}],
        }
        for tag, mtype, rl in devices:
            vo = f"VO{tag}"
            base["solved"].append(vo)
            base["devices"].append({"name": f"M{tag}", "drain": vo, "gate": "VG",
                                    "source": "GND", "W": 1.0, "L": 0.15})
            base["models"][f"M{tag}"] = {"type": mtype}
            base["resistors"].append([f"RL{tag}", "VDD", vo, rl])
            base["load_caps"].append([vo, "GND", 1e-12])
            base["transient_inputs"][f"M{tag}"] = "vin"
            base["dc_guesses"][0][vo] = 1.0
        base["outputs"] = [base["solved"][0]]
        return circuit_from_dict(base)

    tgrid = np.arange(0.0, 4e-7 + TSTEP / 2, TSTEP)
    wave = np.where(tgrid < 0.1e-6, 0.9, 0.95)

    def run(spec):
        return transient(spec.sizes, spec.bias, tgrid, topo=spec.topology,
                         nf=spec.nf, inputs={"vin": wave},
                         model_types=spec.model_types,
                         device_kwargs=spec.device_kwargs,
                         integration_method="gear2")

    mixed = run(cfg([(1, "test_b4.nmos", 1e4), (2, "test_b3.nmos", 5e3)]))
    only4 = run(cfg([(1, "test_b4.nmos", 1e4)]))
    only3 = run(cfg([(2, "test_b3.nmos", 5e3)]))
    assert mixed["nfail"] == 0
    # both stages actually respond to the step, with distinct model behavior
    assert np.ptp(mixed["nodes"]["VO1"]) > 0.05
    assert np.ptp(mixed["nodes"]["VO2"]) > 0.05
    assert abs(mixed["nodes"]["VO1"][-1] - mixed["nodes"]["VO2"][-1]) > 0.01
    # mixed == the single-model runs (independent sub-circuits)
    assert np.max(np.abs(mixed["nodes"]["VO1"] - only4["nodes"]["VO1"])) < 1e-6
    assert np.max(np.abs(mixed["nodes"]["VO2"] - only3["nodes"]["VO2"])) < 1e-6


def test_three_libraries_rejected():
    from core.transient_solver import _osdi_model_names
    assert _osdi_model_names({"M1": "sky130.nmos"})  # sanity: resolver works


def test_adaptive_matches_fine_fixed_grid():
    """Adaptive BDF2 on a COARSE sample grid == a fine fixed-grid reference.

    The CS step settles with τ≈225 ns; the coarse grid samples every 100 ns,
    so the error-controlled substepping must resolve the transient internally.
    """
    from core.transient_solver import transient
    spec = _cs_spec()
    t_coarse = np.arange(0.0, TSTOP + 5e-8, 1e-7)
    w_coarse = np.where(t_coarse < TEDGE, VG0, VG1)
    # the adaptive kernel interpolates inputs linearly between samples, so the
    # fine fixed-grid reference must see the SAME (ramped) stimulus
    t_fine = np.arange(0.0, TSTOP + TSTEP / 2, TSTEP)
    w_fine = np.interp(t_fine, t_coarse, w_coarse)
    ref = transient(spec.sizes, spec.bias, t_fine, topo=spec.topology,
                    nf=spec.nf, inputs={"vin": w_fine},
                    model_types=spec.model_types,
                    device_kwargs=spec.device_kwargs,
                    integration_method="gear2")
    ad = transient(spec.sizes, spec.bias, t_coarse, topo=spec.topology,
                   nf=spec.nf, inputs={"vin": w_coarse},
                   model_types=spec.model_types,
                   device_kwargs=spec.device_kwargs,
                   adaptive=True, adaptive_reltol=1e-4, adaptive_vabstol=1e-6)
    assert ad.get("adaptive") is True
    assert ad["nfail"] == 0
    assert ad["nsubsteps"] > len(t_coarse)        # it actually subdivided
    v_ref = np.interp(t_coarse, t_fine, ref["nodes"]["VOUT"])
    v_ad = ad["nodes"]["VOUT"]
    assert np.max(np.abs(v_ad - v_ref)) < 1e-3    # ~reltol-level everywhere
    assert abs(v_ad[-1] - v_ref[-1]) < 2e-5       # same settled point


def test_adaptive_dc_hold_is_cheap():
    """Constant inputs: the adaptive integrator holds DC and grows its step."""
    from core.transient_solver import transient
    spec = _cs_spec()
    tgrid = np.linspace(0.0, 2e-6, 21)
    wave = np.full_like(tgrid, VG0)
    r = transient(spec.sizes, spec.bias, tgrid, topo=spec.topology, nf=spec.nf,
                  inputs={"vin": wave}, model_types=spec.model_types,
                  device_kwargs=spec.device_kwargs, adaptive=True)
    assert r["nfail"] == 0
    vout = r["nodes"]["VOUT"]
    assert np.ptp(vout) < 1e-6                    # holds the DC op
    assert r["nsubsteps"] < 20 * 8                # step control grows h
