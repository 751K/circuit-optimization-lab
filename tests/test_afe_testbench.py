"""Smoke + sanity tests for the AFE testbench (electrode + AC coupling + amp)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from core.ac_solver import ac_solve
from core.noise_solver import band_rms, noise_analysis
from core.transient_solver import transient

# examples/ is not a package; load the testbench module by path.
_SPEC = importlib.util.spec_from_file_location(
    "afe_testbench", Path(__file__).resolve().parents[1] / "examples" / "afe_testbench.py")
tb_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tb_mod)


@pytest.fixture(scope="module")
def built():
    topo, sizes, bias = tb_mod.build_afe_testbench()
    seed = tb_mod.dc_seed(sizes, bias)
    return topo, sizes, bias, seed


def test_dc_inputs_bias_to_vcm(built):
    topo, sizes, bias, seed = built
    ac = ac_solve(sizes, bias, np.array([1.0]), topo=topo, x0_guess=seed)
    assert ac is not None
    # C_AC blocks DC; R_AC pulls both gates to VCM.
    assert ac["dc_op"]["INP"] == pytest.approx(bias["VCM"], abs=0.1)
    assert ac["dc_op"]["INN"] == pytest.approx(bias["VCM"], abs=0.1)
    # symmetric output operating point
    assert ac["dc_op"]["VOP"] == pytest.approx(ac["dc_op"]["VON"], abs=1e-3)


def test_ac_is_a_bandpass(built):
    topo, sizes, bias, seed = built
    freqs = np.logspace(-3, 4, 161)
    ac = ac_solve(sizes, bias, freqs, topo=topo, x0_guess=seed)
    gains = ac["gains"]
    lo, fpk, hi, peak = tb_mod._bw_edges(freqs, gains)

    # passband gain close to the bare amplifier (~22.9 dB), minus small front-end loss
    assert 21.0 < 20 * np.log10(peak) < 23.5
    # low -3 dB corner set by the 0.05 Hz coupling high-pass
    assert 0.02 < lo < 0.1
    # high -3 dB corner set by the amplifier bandwidth (hundreds of Hz)
    assert hi > 100.0
    # genuinely band-limited: rolled off far below and far above the passband
    assert gains[0] < peak / 10.0                       # 1 mHz heavily attenuated
    assert gains[-1] < peak / 10.0                       # 10 kHz heavily attenuated


def test_noise_includes_frontend_resistors(built):
    topo, sizes, bias, seed = built
    fn = np.logspace(-2, 3, 81)
    nz = noise_analysis(sizes, bias, fn, topo=topo, x0_guess=seed)
    assert nz is not None
    for r in ("REL_P", "REL_N", "RAC_P", "RAC_N"):
        assert r in nz["dev_psd"]
        assert np.all(np.isfinite(nz["dev_psd"][r]))
    irn = band_rms(fn, nz["irn_psd"], 0.05, 100.0) * 1e6
    assert 20.0 < irn < 80.0                              # near the bare AFE's ~37 uVrms


def test_transient_matches_ac_gain_in_band(built):
    topo, sizes, bias, seed = built
    f0, amp = 10.0, 0.5e-3
    t = np.linspace(0, 6 / f0, 900)
    vip = amp * np.sin(2 * np.pi * f0 * t)
    tr = transient(sizes, bias, t, topo=topo,
                   V0=np.array([seed[n] for n in topo.solved]),
                   inputs={"vip": vip, "vin": -vip},
                   node_inputs={"VINP": "vip", "VINN": "vin"})
    assert tr["nfail"] == 0
    half = tr["output"][len(t) // 2:]
    g_tr = (half.max() - half.min()) / 2 / (2 * amp)     # out / in, both differential zero-to-peak
    g_ac = ac_solve(sizes, bias, np.array([f0]), topo=topo, x0_guess=seed)["gains"][0]
    assert g_tr == pytest.approx(g_ac, rel=0.05)
