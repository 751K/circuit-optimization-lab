import numpy as np

from circuitopt.ac_solver import ac_solve
from circuitopt.noise_solver import band_rms, noise_analysis
from circuitopt.transient_solver import transient


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


def test_afe_dc_bounded_fallback_recovers_extreme_nominal_point():
    sizes = {
        "M6": (197183.70917148597, 121.82588865916728),
        "M7": (4293.704945469774, 477.87876884562235),
        "M8": (4293.704945469774, 477.87876884562235),
        "M9": (1157.128720613522, 25.730656744649842),
        "M10": (1157.128720613522, 25.730656744649842),
        "M11": (5507.2464489571, 446.850988354204),
        "M12": (3304.842189679181, 369.5436340915099),
        "M13": (3304.842189679181, 369.5436340915099),
        "M14": (1942.124648968475, 30.952434394142728),
        "M15": (1942.124648968475, 30.952434394142728),
    }
    bias = {"VDD": 40.0, "VCM": 22.417140125884185,
            "VB": 30.373926588097426, "VC": 18.902189620759245}
    freqs = np.array([1.0])

    ac = ac_solve(sizes, bias, freqs)

    # The RESULT is the platform-independent contract: this extreme nominal point
    # resolves to a bounded, finite DC operating point -- whether the production main
    # path found it directly (e.g. glibc, where the whole libm trajectory converges) or
    # the reference-oracle fallback recovered it (e.g. macOS, where the main path fails
    # and the 1-ULP libm-``pow`` Vt square tips onto the convergent branch).
    assert ac is not None
    assert np.isfinite(ac["gains"]).all()
    assert all(-0.5 <= ac["dc_op"][node] <= 40.5
               for node in ["VOP", "VON", "VFBP", "VFBN", "NET20", "NET2"])

    from circuitopt._engine import current_engine
    if current_engine() == "rust":
        # Whether the reference-oracle fallback was NEEDED is platform-dependent: it
        # fires only where the production main path fails. Probe this platform's main
        # path directly (retry disabled) and require the recovery flag iff it failed.
        main_path = ac_solve(sizes, bias, freqs, _rust_reference_retry=True)
        fired = ac.get("rust_otft_reference_fallback", False)
        if main_path is None:
            assert fired is True        # main path failed here -> the oracle recovered it
        else:
            assert fired is False       # main path converged here -> no fallback needed
            assert all(-0.5 <= main_path["dc_op"][node] <= 40.5
                       for node in ["VOP", "VON", "VFBP", "VFBN", "NET20", "NET2"])
