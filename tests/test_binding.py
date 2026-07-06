"""CircuitBinding — Phase A of the solver binding refactor.

The binding bundles the six per-circuit inputs each solver used to thread by hand
(topo / model_types / device_kwargs / nf / corner / x0_guess) so a caller passes
``binding=`` instead, closing the "dropped model_types silently reverts to the
default OTFT PDK" bug class.

These tests pin two things:
  1. the binding is constructed correctly from a CircuitSpec, and ``at_corner``
     routes silicon vs OTFT corners the way ``apply_silicon_corner`` does;
  2. the byte-gate — every solver's ``binding=`` path reproduces the equivalent
     explicit-kwargs path exactly (``np.array_equal`` on the key output arrays),
     and explicit kwargs still override the binding.

The FreePDK45 cases reuse the ngspice availability gate from the FreePDK45 device
tests: they skip cleanly when the cards / ngspice runner are absent.
"""
import dataclasses
import os

import numpy as np
import pytest

from circuitopt import CircuitBinding
from circuitopt.ac_solver import ac_solve
from circuitopt.analysis_dispatch import run_analysis_suite
from circuitopt.circuit_loader import load_circuit_json
from circuitopt.device_factory import CORNERS, SILICON_CORNERS
from circuitopt.ngspice_char import ngspice_binary
from circuitopt.noise_solver import noise_analysis
from circuitopt.pac_solver import pac_solve
from circuitopt.pnoise_solver import pnoise_solve
from circuitopt.pss_solver import pss_solve
from circuitopt.topology import AFE_TOPO, Topology
from circuitopt.transient_solver import transient


# ── FreePDK45 availability gate (mirrors tests/test_freepdk45.py) ────────────
PDK_ROOT = os.environ.get("PDK_ROOT", "/Volumes/MacoutDsik/pdk")
_FP45 = os.path.join(PDK_ROOT, "freepdk45", "models_nom", "NMOS_VTG.inc")
_HAVE_FP45 = os.path.exists(_FP45) and ngspice_binary() is not None
_requires_fp45 = pytest.mark.skipif(
    not _HAVE_FP45, reason="FreePDK45 cards / ngspice not present")

_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


# ── OTFT AFE fixture (validated smoke-test design) ──────────────────────────
AFE_SIZES = {
    "M6": (2264, 78), "M7": (61365, 61), "M8": (61365, 61),
    "M9": (3175, 468), "M10": (3175, 468), "M11": (465, 66),
    "M12": (894, 85), "M13": (894, 85), "M14": (5224, 46), "M15": (5224, 46),
}
AFE_BIAS = {"VDD": 40.0, "VCM": 30.65, "VB": 9.84, "VC": 16.0}


def _fp45_spec():
    return load_circuit_json(os.path.join(_EXAMPLES, "freepdk45_fd_ota.json"))


def _rc_lowpass_topology(R=1e5, C=1e-9):
    return Topology(
        solved=["OUT"],
        devices=[],
        rails={"VIN": "VIN", "GND": 0.0},
        outputs=("OUT",),
        resistors=[("R1", "VIN", "OUT", R)],
        capacitors=[("C1", "OUT", "GND", C)],
    )


# ── T1 ───────────────────────────────────────────────────────────────────────
@_requires_fp45
def test_binding_from_spec_carries_models():
    spec = _fp45_spec()
    b = spec.binding()
    assert b.model_types, "binding lost the per-device model_types"
    assert all(str(v).startswith("freepdk45") for v in b.model_types.values())
    assert isinstance(b.dc_seed, dict) and b.dc_seed
    # dc_seed is the first dict dc_guess (matches the inline analysis seed)
    assert b.dc_seed == spec.topology.dc_guesses[0]
    assert b.topo is spec.topology
    assert b.device_kwargs is spec.device_kwargs
    assert b.corner is None


# ── T2 ───────────────────────────────────────────────────────────────────────
@_requires_fp45
def test_at_corner_silicon_vs_otft():
    # Silicon binding: a discrete corner bakes onto each freepdk45 device's kwargs
    # and clears the solver corner.
    b = _fp45_spec().binding()
    assert "ss" in SILICON_CORNERS
    b_ss = b.at_corner("ss")
    sil = [n for n, m in b.model_types.items() if str(m).startswith("freepdk45")]
    assert sil
    for name in sil:
        assert b_ss.device_kwargs[name]["corner"] == "ss"
    assert b_ss.corner is None
    # original binding is untouched (frozen + copy-on-write in apply_silicon_corner)
    assert b.corner is None
    if b.device_kwargs:
        for name in sil:
            assert "corner" not in (b.device_kwargs.get(name) or {})

    # OTFT binding: "slow" is not a silicon corner, so apply_silicon_corner passes
    # it through to the solver corner and leaves device kwargs alone.
    ob = CircuitBinding(topo=AFE_TOPO)
    assert "slow" not in SILICON_CORNERS and "slow" in CORNERS
    ob_slow = ob.at_corner("slow")
    assert ob_slow.corner == "slow"
    assert ob_slow.device_kwargs is None

    # None returns self unchanged.
    assert ob.at_corner(None) is ob


# ── T3 ───────────────────────────────────────────────────────────────────────
def test_ac_noise_parity():
    # OTFT AFE: kwargs path vs binding path, byte-for-byte on the key arrays.
    freqs = np.logspace(0, 4, 21)
    ob = CircuitBinding(topo=AFE_TOPO)
    ac_kw = ac_solve(AFE_SIZES, AFE_BIAS, freqs)
    ac_bd = ac_solve(AFE_SIZES, AFE_BIAS, freqs, binding=ob)
    assert ac_kw is not None and ac_bd is not None
    assert np.array_equal(ac_kw["gains"], ac_bd["gains"])
    assert np.array_equal(ac_kw["response"], ac_bd["response"])

    nz_kw = noise_analysis(AFE_SIZES, AFE_BIAS, freqs)
    nz_bd = noise_analysis(AFE_SIZES, AFE_BIAS, freqs, binding=ob)
    assert nz_kw is not None and nz_bd is not None
    assert np.array_equal(nz_kw["out_psd"], nz_bd["out_psd"])
    assert np.array_equal(nz_kw["irn_psd"], nz_bd["irn_psd"])


@_requires_fp45
def test_ac_noise_parity_freepdk45():
    spec = _fp45_spec()
    b = spec.binding()
    freqs = np.logspace(3, 8, 8)
    # kwargs path == the exact cluster the dispatcher threads inline.
    ac_kw = ac_solve(
        spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf,
        x0_guess=b.dc_seed, model_types=spec.model_types,
        device_kwargs=spec.device_kwargs)
    ac_bd = ac_solve(spec.sizes, spec.bias, freqs, binding=b)
    assert ac_kw is not None and ac_bd is not None
    assert np.array_equal(ac_kw["gains"], ac_bd["gains"])
    assert np.array_equal(ac_kw["response"], ac_bd["response"])

    nz_kw = noise_analysis(
        spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf,
        x0_guess=b.dc_seed, model_types=spec.model_types,
        device_kwargs=spec.device_kwargs)
    nz_bd = noise_analysis(spec.sizes, spec.bias, freqs, binding=b)
    assert nz_kw is not None and nz_bd is not None
    assert np.array_equal(nz_kw["out_psd"], nz_bd["out_psd"])
    assert np.array_equal(nz_kw["irn_psd"], nz_bd["irn_psd"])


# ── T4 ───────────────────────────────────────────────────────────────────────
def test_transient_parity():
    # Small OTFT single-stage circuit: kwargs vs binding, byte-for-byte.
    spec = load_circuit_json(os.path.join(_EXAMPLES, "single_stage.json"))
    b = spec.binding()
    n = 60
    t = np.linspace(0.0, 1e-3, n)
    vin = np.full(n, spec.bias["VIN"]) + np.where(t >= 0.5e-3, 1e-3, 0.0)
    kw = dict(inputs={"vin": vin})
    tr_kw = transient(spec.sizes, spec.bias, t, topo=spec.topology, nf=spec.nf,
                      model_types=spec.model_types,
                      device_kwargs=spec.device_kwargs, **kw)
    tr_bd = transient(spec.sizes, spec.bias, t, binding=b, **kw)
    assert np.array_equal(tr_kw["output"], tr_bd["output"])
    assert np.array_equal(tr_kw["t"], tr_bd["t"])
    assert tr_kw["nfail"] == tr_bd["nfail"]


# ── T5 ───────────────────────────────────────────────────────────────────────
def test_periodic_parity():
    # Generic RC low-pass (the periodic-solver test topology): the binding carries
    # topo, so pss/pac/pnoise reproduce the kwargs path byte-for-byte.
    topo = _rc_lowpass_topology()
    b = CircuitBinding(topo=topo)
    period = 1e-3
    t = np.linspace(0.0, period, 101)
    common = dict(inputs={"vin": np.zeros_like(t)}, node_inputs={"VIN": "vin"},
                  V0=np.array([0.0]), residual_tol=1e-12, max_shooting_iters=2)

    pss_kw = pss_solve({}, {"VIN": 0.0}, period, topo=topo, tgrid=t, **common)
    pss_bd = pss_solve({}, {"VIN": 0.0}, period, binding=b, tgrid=t, **common)
    assert np.array_equal(np.asarray(pss_kw["output"]), np.asarray(pss_bd["output"]))
    assert np.array_equal(np.asarray(pss_kw["x0"]), np.asarray(pss_bd["x0"]))

    freqs = np.array([100.0, 500.0])
    pac_kw = pac_solve({}, {"VIN": 0.0}, freqs, pss_result=pss_kw,
                       input_drive={"vin": 1.0})
    pac_bd = pac_solve({}, {"VIN": 0.0}, freqs, pss_result=pss_bd,
                       input_drive={"vin": 1.0}, binding=b)
    assert np.array_equal(pac_kw["response"], pac_bd["response"])

    pn_kw = pnoise_solve({}, {"VIN": 0.0}, freqs, pss_result=pss_kw,
                         input_drive={"vin": 1.0}, band=(50.0, 500.0))
    pn_bd = pnoise_solve({}, {"VIN": 0.0}, freqs, pss_result=pss_bd,
                         input_drive={"vin": 1.0}, band=(50.0, 500.0), binding=b)
    assert np.array_equal(np.asarray(pn_kw["out_psd"]),
                          np.asarray(pn_bd["out_psd"]))


# ── T6 ───────────────────────────────────────────────────────────────────────
def test_binding_override():
    freqs = np.logspace(0, 4, 15)

    # (a) explicit corner overrides binding.corner: binding("slow") + corner="fast"
    #     must equal the direct kwargs corner="fast".
    b_slow = CircuitBinding(topo=AFE_TOPO, corner="slow")
    ac_override = ac_solve(AFE_SIZES, AFE_BIAS, freqs, corner="fast", binding=b_slow)
    ac_direct = ac_solve(AFE_SIZES, AFE_BIAS, freqs, corner="fast")
    assert ac_override is not None and ac_direct is not None
    assert np.array_equal(ac_override["gains"], ac_direct["gains"])
    # and it is *not* the "slow" result (guards against the override being a no-op)
    ac_slow = ac_solve(AFE_SIZES, AFE_BIAS, freqs, corner="slow")
    assert not np.array_equal(ac_override["gains"], ac_slow["gains"])

    # (b) explicit x0_guess overrides binding.dc_seed. Use a dc_op from a prior
    #     solve as the seed; the binding carries a different (still valid) seed.
    ac_ref = ac_solve(AFE_SIZES, AFE_BIAS, freqs)
    seed = ac_ref["dc_op"]
    b_seeded = CircuitBinding(topo=AFE_TOPO, dc_seed={"VOP": 5.0, "VON": 5.0})
    ac_x0 = ac_solve(AFE_SIZES, AFE_BIAS, freqs, x0_guess=seed, binding=b_seeded)
    ac_x0_direct = ac_solve(AFE_SIZES, AFE_BIAS, freqs, x0_guess=seed)
    assert np.array_equal(ac_x0["gains"], ac_x0_direct["gains"])

    # (c) explicit model_types overrides the binding's. A binding carrying a bogus
    #     model map, overridden by an explicit empty-ish OTFT map ({}), must match
    #     the plain default-PDK solve.
    b_models = CircuitBinding(topo=AFE_TOPO,
                              model_types={"M7": "does.not.exist"})
    ac_mt = ac_solve(AFE_SIZES, AFE_BIAS, freqs, model_types={}, binding=b_models)
    assert ac_mt is not None
    assert np.array_equal(ac_mt["gains"], ac_ref["gains"])


# ── T7: dispatch never silently reverts to OTFT ──────────────────────────────
@_requires_fp45
def test_dispatch_no_silent_otft_fallback():
    """Phase B guard: ``run_analysis_suite`` threads ``spec.binding()`` to every
    branch, so a silicon config's AC gain matches the direct ``ac_solve(binding=)``
    byte-for-byte — and a binding that has *lost* its models (the exact regression
    the refactor closes) diverges by ~100 dB, not silently by a fraction. Pins the
    "dropped model_types silently reverts to the default OTFT PDK" bug class."""
    spec = _fp45_spec()
    freqs = [1e3, 1e4, 1e5, 1e6]            # short grid: fast, exercises the AC branch
    analyses = {"ac": {"freqs": freqs}}

    # (a) the dispatcher's AC gain == the direct binding-path solve, byte-for-byte.
    suite = run_analysis_suite(spec, analyses, selected=["ac"])
    direct = ac_solve(spec.sizes, spec.bias, freqs, binding=spec.binding())
    assert suite["ac"] is not None and direct is not None
    assert np.array_equal(suite["ac"]["gains"], direct["gains"])

    # (b) strip the per-device models (model_types/device_kwargs → None): the circuit
    #     silently falls back to the default OTFT PDK, and the gain is off by a mile.
    stripped = dataclasses.replace(spec, model_types=None, device_kwargs=None)
    otft = run_analysis_suite(stripped, analyses, selected=["ac"])
    assert otft["ac"] is not None
    good_dB = 20.0 * np.log10(suite["ac"]["gains"][0])
    otft_dB = 20.0 * np.log10(otft["ac"]["gains"][0])
    assert abs(good_dB - otft_dB) > 3.0    # in fact ~100 dB — a catastrophic revert
