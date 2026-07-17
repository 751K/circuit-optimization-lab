"""SKY130 chopper amplifier on the native C BSIM4 engine.

The testbench (``examples/sky130_chopper.json``) is an NMOS input chopper →
NMOS diff pair with diode-connected PMOS loads → NMOS output chopper → hold
caps, clocked at 250 kHz by square-wave vsources. The demodulated differential
output is DC at ``gain·(VINP−VINN)`` with gain = −gm1/gm3 ≈ −1.77.

Validates the silicon periodic-analysis chain through PSS/transient agreement,
PAC conversion gain, PNoise, and frozen-clock LTI reductions.
"""
import json
import os

import numpy as np
import pytest

_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "examples", "sky130_chopper.json")
DELTA = 0.01                       # VINP - VINN differential input [V]
PERIOD = 1.0 / 250e3
NPTS = 401


@pytest.fixture(scope="module")
def spec():
    from circuitopt.circuit_loader import circuit_from_dict
    with open(_CFG_PATH) as fh:
        return circuit_from_dict(json.load(fh))


@pytest.fixture(scope="module")
def suite(spec):
    from circuitopt.analysis_dispatch import run_analysis_suite
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
    # Native BSIM4 produces a finite sub-uV/sqrt(Hz demodulated floor.
    assert 1e-8 < asd[0] < 1e-4
    assert np.ptp(asd) < 0.35 * asd[0]


def test_dataset_chopper_labels(tmp_path, suite):
    """The chopper's periodic figures of merit flow into surrogate-dataset labels.

    Pins the single LHS candidate to the validated nominal design (min==max
    ranges), so the pss/pac/pnoise labels must reproduce the suite fixture's
    numbers — same solvers, same validated ``analyses`` settings, threaded
    through ``build_dataset``'s suite runner."""
    import circuitopt.dataset as ds
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
    from circuitopt.analysis_dispatch import run_analysis_suite
    from circuitopt.circuit_loader import circuit_from_dict
    from circuitopt.pac_solver import pac_solve
    from circuitopt.pnoise_solver import pnoise_solve
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
