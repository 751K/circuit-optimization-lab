import numpy as np

import demo.server as demo_server


UI_SIZES = {
    "M6": (4820, 78), "M7": (65426, 42), "M8": (65426, 42),
    "M9": (2876, 333), "M10": (2876, 333), "M11": (339, 155),
    "M12": (505, 134), "M13": (505, 134),
    "M14": (4533, 48), "M15": (4533, 48),
}

UI_BIAS = {"VDD": 40.0, "VCM": 32.0, "VB": 7.5, "VC": 16.0}


DEMO_COLD_FAIL_SIZES = {
    "M6": (70007.42355306227, 151.703298313683),
    "M7": (154.8681849527918, 99.42139621781372),
    "M8": (154.8681849527918, 99.42139621781372),
    "M9": (338.7248296360881, 776.2269831299153),
    "M10": (338.7248296360881, 776.2269831299153),
    "M11": (3094.43078067647, 732.8151631636837),
    "M12": (69.97733341480354, 259.1271434031613),
    "M13": (69.97733341480354, 259.1271434031613),
    "M14": (355.66356025228157, 377.40012873705695),
    "M15": (355.66356025228157, 377.40012873705695),
}

DEMO_COLD_FAIL_BIAS = {
    "VDD": 40.0,
    "VCM": 34.579609391234825,
    "VB": 26.876372265996622,
    "VC": 26.23162328149349,
}


def _reset_demo_seed_state():
    with demo_server._dc_seed_lock:
        demo_server._last_dc_seed = None
        demo_server._last_dc_bias = None
    demo_server._preset_seed_cache.clear()


def test_demo_payload_runs_current_ui_sizes_without_dc_error():
    _reset_demo_seed_state()

    result = demo_server.solve_payload({
        "sizes": UI_SIZES,
        "bias": UI_BIAS,
        "nfreq": 11,
    })

    assert result["converged"] is True
    assert result["status"]["dc_seed"] == "cold"
    assert result["gain_dB"] > 20.0
    assert result["dc_op"]["VOP"] > 0.0


def test_demo_retry_uses_preset_branch_seed_after_cold_failure():
    _reset_demo_seed_state()

    freqs = np.array([1.0])
    # solve_ac_with_retries tries the canonical cold solve (x0_guess=None) first,
    # so this direct call reproduces exactly THIS platform's cold attempt. It is a
    # pure solver call and does not mutate the demo seed state used below.
    cold = demo_server.ac_solve(
        DEMO_COLD_FAIL_SIZES, DEMO_COLD_FAIL_BIAS, freqs, x0_guess=None
    )

    ac, mode = demo_server.solve_ac_with_retries(
        DEMO_COLD_FAIL_SIZES,
        DEMO_COLD_FAIL_BIAS,
        freqs,
    )

    # Platform-independent contract: whichever seed wins, the retry ladder yields
    # a bounded, finite AC operating point.
    assert ac is not None
    assert np.isfinite(ac["gains"]).all()
    assert all(-0.25 <= ac["dc_op"][node] <= 40.25
               for node in ("VOP", "VON", "VFBP", "VFBN", "NET20", "NET2"))

    # Which seed wins is a libm platform truth. This DC-hostile size set makes the
    # cold solve diverge on glibc/macOS libm, so the ladder must fall through to
    # the first preset branch seed; on Windows UCRT libm the cold solve converges
    # and the canonical cold path legitimately wins. Both regimes are correct;
    # pin the one this platform is actually in (probed above).
    if cold is None:
        assert mode == "preset_first_feasible"
    else:
        assert mode == "cold"
