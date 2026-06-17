import numpy as np

from core.ac_solver import ac_solve
from core.noise_solver import band_rms, noise_analysis
from core.transient_solver import transient


SIZES = {
    "M6": (2264, 78), "M7": (61365, 61), "M8": (61365, 61),
    "M9": (3175, 468), "M10": (3175, 468), "M11": (465, 66),
    "M12": (894, 85), "M13": (894, 85), "M14": (5224, 46), "M15": (5224, 46),
}

BIAS = {"VDD": 40.0, "VCM": 30.65, "VB": 9.84, "VC": 16.0}


def test_afe_ac_noise_smoke():
    freqs = np.logspace(0, 4, 41)
    ac = ac_solve(SIZES, BIAS, freqs)
    assert ac is not None
    gain_db = 20 * np.log10(ac["gains"].max())
    assert 22.0 < gain_db < 24.0
    assert ac["bw_Hz"] > 0.0
    assert abs(ac["dc_op"]["VOP"] - ac["dc_op"]["VON"]) < 1e-6

    noise = noise_analysis(SIZES, BIAS, freqs)
    assert noise is not None
    irn = band_rms(freqs, noise["irn_psd"], 1.0, 100.0)
    assert 1e-6 < irn < 100e-6


def test_afe_transient_step_smoke():
    n = 80
    t = np.linspace(0, 2e-3, n)
    vcm = np.full(n, BIAS["VCM"])
    vp = vcm + np.where(t >= 0.5e-3, +0.5e-3, 0.0)
    vn = vcm - np.where(t >= 0.5e-3, +0.5e-3, 0.0)

    tr = transient(SIZES, BIAS, t, vp, vn)
    assert tr["nfail"] == 0
    assert np.isfinite(tr["vout"]).all()
    assert tr["vout"][-1] < -1e-3
