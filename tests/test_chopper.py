import copy
import os

import numpy as np
import pytest

import core.chopper as chopper_mod
from core.circuit_loader import load_circuit_json
from core.chopper import (
    build_afe_pmos_chopper,
    chopper_analysis,
    finite_edge_chopper_harmonics,
    finite_edge_clock_pair,
    pmos_chopper_analysis,
    pmos_chopper_lptv_analysis,
    pmos_chopper_pac,
    pmos_chopper_pnoise,
    pmos_chopper_pss,
    pmos_chopper_phase_bias,
    pmos_chopper_transient,
    refine_chopper_tgrid,
    square_chopper_harmonics,
)
from core.topology import AFE_TOPO, Topology


SIZES = {
    "M6": (2264, 78), "M7": (61365, 61), "M8": (61365, 61),
    "M9": (3175, 468), "M10": (3175, 468), "M11": (465, 66),
    "M12": (894, 85), "M13": (894, 85), "M14": (5224, 46), "M15": (5224, 46),
}

BIAS = {"VDD": 40.0, "VCM": 30.65, "VB": 9.84, "VC": 16.0}

CHOPPER_UI_SIZES = {
    "M6": (4820, 78), "M7": (65426, 42), "M8": (65426, 42),
    "M9": (2876, 333), "M10": (2876, 333), "M11": (339, 155),
    "M12": (505, 134), "M13": (505, 134),
    "M14": (4533, 48), "M15": (4533, 48),
}

CHOPPER_UI_BIAS = {"VDD": 40.0, "VCM": 32.0, "VB": 7.5, "VC": 16.0}


def _spectre_dec_grid(start, stop, points_per_dec=20):
    step = 10.0 ** (1.0 / points_per_dec)
    vals = []
    x = float(start)
    while x <= float(stop) * (1.0 + 1e-12):
        vals.append(x)
        x *= step
    return np.array(vals)


def test_square_chopper_harmonic_weights_are_truncated_square_power():
    harmonics, weights = square_chopper_harmonics(9)

    assert tuple(harmonics) == (-9, -7, -5, -3, -1, 1, 3, 5, 7, 9)
    np.testing.assert_allclose(weights, 4.0 / (np.pi ** 2 * harmonics ** 2))
    assert 0.9 < weights.sum() < 1.0


def test_finite_edge_harmonics_reduce_clock_power():
    harmonics, square_weights = square_chopper_harmonics(9)
    h2, coeffs, edge_weights = finite_edge_chopper_harmonics(
        9, edge_fraction=0.03, dead_fraction=0.01, samples=2048)

    assert tuple(h2) == tuple(harmonics)
    assert np.iscomplexobj(coeffs)
    assert edge_weights.sum() < square_weights.sum()

    t = np.linspace(0.0, 0.02, 101)
    clk_a, clk_b, a_on, b_on = finite_edge_clock_pair(
        t, 100.0, v_low=0.0, v_high=40.0, edge_time=2e-4, dead_time=1e-4)
    assert clk_a.shape == t.shape
    assert clk_b.shape == t.shape
    assert np.all((0.0 <= a_on) & (a_on <= 1.0))
    assert np.all((0.0 <= b_on) & (b_on <= 1.0))


def test_refine_chopper_tgrid_adds_edge_points():
    t = np.linspace(0.0, 0.005, 31)
    refined = refine_chopper_tgrid(
        t, 100.0, edge_time=1e-3, dead_time=2e-4,
        phase_offset=0.25, edge_points=7)

    assert refined[0] == t[0]
    assert refined[-1] == t[-1]
    assert len(refined) > len(t)


def test_chopper_gain_matches_flat_resistor_divider_weight_sum():
    topo = Topology(
        solved=["OUT"],
        devices=[],
        rails={"VIN": "VIN", "GND": 0.0},
        outputs=("OUT",),
        ac_drives={"VIN": 1.0},
        resistors=[("R1", "VIN", "OUT", 1e3), ("R2", "OUT", "GND", 1e3)],
    )
    freqs = np.array([1.0, 10.0])
    result = chopper_analysis({}, {"VIN": 0.0}, freqs, f_chop=100.0, topo=topo,
                              max_harmonic=9, band=(1.0, 10.0))

    assert result is not None
    # The divider is frequency-flat with H=0.5, so finite-harmonic chopping gives
    # H_chop = 0.5 * sum(|c_k|^2).
    np.testing.assert_allclose(
        result["gains"],
        0.5 * result["harmonic_weight_sum"],
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.all(np.isfinite(result["irn_psd"]))


def test_chopper_analysis_runs_on_default_afe():
    freqs = np.logspace(0, 2, 11)
    result = chopper_analysis(SIZES, BIAS, freqs, f_chop=100.0, topo=AFE_TOPO,
                              max_harmonic=5, band=(1.0, 100.0))

    assert result is not None
    assert np.isfinite(result["gains"]).all()
    assert np.isfinite(result["irn_psd"]).all()
    assert result["bw_Hz"] > 0.0
    assert result["harmonic_weight_sum"] < 1.0
    assert result["irn_uV_band"] > 0.0


def test_build_afe_pmos_chopper_adds_eight_switch_devices():
    build = build_afe_pmos_chopper(switch_size=(12000.0, 80.0), switch_nf=3)
    topo = build.topology
    dev_names = {name for name, *_ in topo.devices}

    assert len(build.switch_names) == 8
    assert set(build.switch_names) <= dev_names
    assert topo.outputs == build.output_nodes
    assert build.input_nodes == ("CH_VIP", "CH_VIN")
    assert build.clock_nodes == ("CH_CLK_A", "CH_CLK_B")
    assert topo.ac_drives["CH_VIP"] == +0.5
    assert topo.ac_drives["CH_VIN"] == -0.5
    assert topo.rails["CH_CLK_A"] == "CLK_A"
    assert topo.rails["CH_CLK_B"] == "CLK_B"
    assert build.switch_sizes["CH_SW_INP_A"] == (12000.0, 80.0)
    assert build.switch_nf["CH_SW_INP_A"] == 3

    phase_b = pmos_chopper_phase_bias(BIAS, "B")
    assert phase_b["VIP"] == BIAS["VCM"]
    assert phase_b["VIN"] == BIAS["VCM"]
    assert phase_b["CLK_A"] == BIAS["VDD"]
    assert phase_b["CLK_B"] == 0.0


def test_build_afe_pmos_chopper_accepts_json_afe_topology():
    spec = load_circuit_json("examples/afe_explore.json")
    build = build_afe_pmos_chopper(base_topo=spec.topology, switch_size=(5000.0, 30.0))

    assert len(build.switch_names) == 8
    assert build.topology.outputs == build.output_nodes
    assert build.input_nodes == ("CH_VIP", "CH_VIN")


def test_pmos_chopper_analysis_runs_on_default_afe_static_phases():
    freqs = np.logspace(0, 2, 7)
    result = pmos_chopper_analysis(
        SIZES,
        BIAS,
        freqs,
        switch_size=(20000.0, 80.0),
        switch_nf=1,
        band=(1.0, 100.0),
    )

    assert result is not None
    assert set(result["phases"]) == {"A", "B"}
    assert len(result["switch_names"]) == 8
    assert np.isfinite(result["gains"]).all()
    assert np.isfinite(result["irn_psd"]).all()
    assert result["gains"][0] > 1.0
    assert result["bw_Hz"] > 0.0
    assert result["irn_uV_band"] > 0.0
    assert "Static phase" in result["analysis_note"]


def test_pmos_chopper_auto_seed_handles_changed_ui_sizes():
    freqs = np.array([1.0, 10.0, 100.0])
    result = pmos_chopper_analysis(
        CHOPPER_UI_SIZES,
        CHOPPER_UI_BIAS,
        freqs,
        switch_size=(5000.0, 30.0),
        band=(1.0, 100.0),
    )

    assert result is not None
    assert set(result["phases"]) == {"A", "B"}
    assert np.isfinite(result["gains"]).all()
    assert result["gains"][0] > 1.0
    for phase in ("A", "B"):
        dc = result["phases"][phase]["dc"]
        assert abs(dc["CH_AMP_OP"] - dc["CH_VOP"]) < 1e-3
        assert abs(dc["CH_AMP_ON"] - dc["CH_VON"]) < 1e-3


def test_pmos_chopper_auto_seed_reuses_bare_dc_cache(monkeypatch):
    build = build_afe_pmos_chopper(switch_size=(5000.0, 30.0))
    chopper_mod._PMOS_CHOPPER_BARE_DC_SEED_CACHE.clear()
    calls = {"n": 0}
    dc_op = {
        "VOP": 10.0,
        "VON": 11.0,
        "VFBP": 12.0,
        "VFBN": 13.0,
        "NET20": 14.0,
        "NET2": 15.0,
    }

    def fake_ac_solve(*_args, **_kwargs):
        calls["n"] += 1
        return {"dc_op": dict(dc_op)}

    monkeypatch.setattr(chopper_mod, "ac_solve", fake_ac_solve)

    seed_a = chopper_mod._pmos_chopper_auto_seed(
        CHOPPER_UI_SIZES, CHOPPER_UI_BIAS, "A", build, nf=None,
        base_topo=AFE_TOPO)
    seed_b = chopper_mod._pmos_chopper_auto_seed(
        CHOPPER_UI_SIZES, CHOPPER_UI_BIAS, "B", build, nf=None,
        base_topo=AFE_TOPO)

    assert calls["n"] == 1
    assert seed_a["CH_VOP"] == dc_op["VOP"]
    assert seed_b["CH_VOP"] == dc_op["VON"]


def test_pmos_chopper_lptv_analysis_runs_with_finite_edges():
    freqs = np.array([1.0, 10.0])
    result = pmos_chopper_lptv_analysis(
        SIZES,
        BIAS,
        freqs,
        f_chop=100.0,
        max_harmonic=3,
        edge_time=1e-4,
        dead_time=5e-5,
        harmonic_samples=512,
        band=(1.0, 10.0),
    )

    assert result is not None
    assert np.isfinite(result["gains"]).all()
    assert np.isfinite(result["irn_psd"]).all()
    assert result["harmonic_weight_sum"] < 1.0
    assert result["irn_uV_band"] > 0.0
    assert "quasi-static" in result["analysis_note"].lower()


def test_pmos_chopper_lptv_auto_seed_handles_changed_ui_sizes():
    result = pmos_chopper_lptv_analysis(
        CHOPPER_UI_SIZES,
        CHOPPER_UI_BIAS,
        np.array([1.0, 10.0, 100.0]),
        f_chop=225.0,
        switch_size=(5000.0, 30.0),
        max_harmonic=3,
        band=(1.0, 100.0),
    )

    assert result is not None
    assert np.isfinite(result["gains"]).all()
    assert result["gains"][0] > 1.0
    assert result["pmos_sideband"]["phases"]["A"]["dc"]["CH_VOP"] > 0.0


def test_pmos_chopper_lptv_is_first_order_underestimate():
    # The empirical Cadence-fit constants (conversion phase 24.93 deg, noise scale
    # 1.0355) were retired 2026-06-22: lptv_analysis is now an honest FIRST-ORDER
    # quasi-static estimate. It underestimates the Spectre PSS/PAC gain (21.369 dB)
    # by ~1 dB because it omits the higher-order LPTV conversion. Cadence-grade
    # gain/noise come from the constant-free HB path (pmos_chopper_pac/pnoise),
    # asserted by test_pmos_chopper_pac_matches_cadence_baseband_gain + calibration.
    freqs = _spectre_dec_grid(0.05, 10000.0, points_per_dec=20)
    result = pmos_chopper_lptv_analysis(
        CHOPPER_UI_SIZES, CHOPPER_UI_BIAS, freqs, f_chop=225.0,
        switch_size=(5000.0, 30.0), max_harmonic=31, edge_time=20e-6,
        band=(0.05, 100.0))
    assert "cadence_calibrated" not in result          # constant retired
    assert np.isfinite(result["gains"]).all()
    assert result["gains"][0] > 1.0
    # First-order: a bit BELOW Spectre's 21.369 dB (no phase fudge), but in the
    # right ballpark — not the old constant-matched value.
    assert 19.5 < result["Av_dc_dB"] < 21.369
    assert result["bw_Hz"] > 100.0
    assert result["irn_uV_band"] > 0.0
    # The raw response equals the returned one now (no built-in correction).
    np.testing.assert_allclose(result["raw_quasi_gains"], result["gains"], rtol=1e-12)


def test_pmos_chopper_transient_refines_edges_and_converges():
    t = np.linspace(0.0, 0.005, 31)
    result = pmos_chopper_transient(
        SIZES,
        BIAS,
        t,
        f_chop=100.0,
        input_diff=0.0,
        edge_time=1e-3,
        dead_time=2e-4,
        clock_style="phase",
        charge_injection=True,
        charge_scale=0.05,
        switch_size=(10000.0, 80.0),
        edge_points=7,
    )

    assert result["nfail"] <= 2
    assert result["refined_point_count"] > len(t)
    assert len(result["charge_injection_sources"]) > 0
    assert np.isfinite(result["output"]).all()
    assert np.isfinite(result["requested_output"]).all()


def test_pmos_chopper_transient_ui_sizes_do_not_run_away():
    t = np.linspace(0.0, 1.0 / 225.0, 31)
    result = pmos_chopper_transient(
        CHOPPER_UI_SIZES,
        CHOPPER_UI_BIAS,
        t,
        f_chop=225.0,
        input_diff=0.0,
        charge_injection=False,
        switch_size=(5000.0, 30.0),
        edge_points=5,
    )
    nodes = result["requested_nodes"]
    input_dm = nodes["CH_INP"] - nodes["CH_INN"]
    core_out_dm = nodes["CH_AMP_OP"] - nodes["CH_AMP_ON"]

    assert result["nfail"] <= 1
    assert np.ptp(input_dm) < 1e-3
    assert np.ptp(core_out_dm) < 1e-3
    assert np.ptp(result["requested_output"]) < 1e-3
    assert np.isfinite(result["requested_output"]).all()


def test_pmos_chopper_transient_ui_finite_edge_matches_cadence_scale():
    period = 1.0 / 225.0
    t = np.linspace(0.0, 2.0 * period, 161)
    result = pmos_chopper_transient(
        CHOPPER_UI_SIZES,
        CHOPPER_UI_BIAS,
        t,
        f_chop=225.0,
        input_diff=1e-3,
        edge_time=20e-6,
        charge_injection=False,
        switch_size=(5000.0, 30.0),
        edge_points=5,
    )
    rt = result["t"]
    mask = rt >= rt[-1] - period - 1e-15
    nodes = result["nodes"]
    input_cm = 0.5 * (nodes["CH_INP"] + nodes["CH_INN"]) - CHOPPER_UI_BIAS["VCM"]
    core_cm = 0.5 * (nodes["CH_AMP_OP"] + nodes["CH_AMP_ON"]) - CHOPPER_UI_BIAS["VCM"]
    out = result["output"]

    assert result["nfail"] <= 1
    assert 0.015 < np.ptp(out[mask]) < 0.03
    assert -0.02 < np.mean(out[mask]) < -0.005
    assert 4.5 < np.ptp(input_cm[mask]) < 6.0
    assert -3.5 < np.mean(core_cm[mask]) < -0.5


def test_pmos_chopper_transient_gear2_retry_handles_stiff_edges():
    # gear2 on stiff chopper edges should stay in the numba grid path even when
    # raw transient maxstep/retry subdivision is requested.  The waveform should
    # still stay close to the BE reference because both paths resolve the same
    # finite-edge transient.
    period = 1.0 / 225.0
    t = np.linspace(0.0, 2.0 * period, 161)
    common = dict(
        f_chop=225.0,
        input_diff=1e-3,
        edge_time=20e-6,
        charge_injection=False,
        switch_size=(5000.0, 30.0),
        edge_points=5,
        profile=True,
    )
    be = pmos_chopper_transient(
        CHOPPER_UI_SIZES, CHOPPER_UI_BIAS, t, integration_method="be", **common)
    g2 = pmos_chopper_transient(
        CHOPPER_UI_SIZES, CHOPPER_UI_BIAS, t, integration_method="gear2", **common)

    assert not g2.get("gear2_be_fallback_used", False)
    assert not g2.get("gear2_python_retry_solver", False)
    assert g2.get("numba_grid_solver", False)
    assert g2.get("transient_profile", {}).get("numba_grid_solver", False)
    assert g2.get("transient_profile", {}).get("failed_intervals", 0) == 0
    assert g2["nfail"] <= be["nfail"]
    np.testing.assert_allclose(g2["output"], be["output"], rtol=1e-3, atol=3e-3)


def test_pmos_chopper_pss_shooting_smoke_converges():
    result = pmos_chopper_pss(
        CHOPPER_UI_SIZES,
        CHOPPER_UI_BIAS,
        225.0,
        switch_size=(5000.0, 30.0),
        edge_time=20e-6,
        n_points=17,
        refine_edges=False,
        charge_injection=False,
        tstab_periods=0,
        max_shooting_iters=2,
        residual_tol=1e-5,
    )

    assert result["converged"]
    assert result["nfail"] == 0
    assert result["residual_norm"] < 1e-5
    assert len(result["t"]) == 17
    assert result["shooting_period_runs"] <= 3


def test_pmos_chopper_transient_flat_step_profile_reduces_work():
    period = 1.0 / 225.0
    t = np.linspace(0.0, 2.0 * period, 161)
    common = dict(
        sizes=CHOPPER_UI_SIZES,
        bias=CHOPPER_UI_BIAS,
        tgrid=t,
        f_chop=225.0,
        input_diff=1e-3,
        edge_time=20e-6,
        charge_injection=False,
        switch_size=(5000.0, 30.0),
        edge_points=5,
        profile=True,
    )
    strict = pmos_chopper_transient(**common, transient_flat_max_step=0.0)
    fast = pmos_chopper_transient(**common)

    assert strict["nfail"] <= 1
    assert fast["nfail"] == 0
    assert fast["numba_grid_solver"] is True
    assert fast["transient_profile"]["numba_grid_partial"] is False
    assert fast["transient_profile"]["failed_substeps"] == 0
    assert fast["transient_profile"]["failed_intervals"] == 0
    assert fast["nsubsteps"] < strict["nsubsteps"]
    assert fast["transient_profile"]["flat_substeps"] < strict["transient_profile"]["flat_substeps"]
    assert fast["transient_profile"]["internal_fd_jac_fallbacks"] == 0
    assert fast["transient_profile"]["terminal_fd_jac_fallbacks"] == 0

    rt = strict["t"]
    mask = rt >= rt[-1] - period - 1e-15
    strict_metrics = np.array([
        np.mean(strict["output"][mask]),
        np.ptp(strict["output"][mask]),
        np.ptp(strict["nodes"]["CH_AMP_OP"][mask] -
               strict["nodes"]["CH_AMP_ON"][mask]),
    ])
    fast_metrics = np.array([
        np.mean(fast["output"][mask]),
        np.ptp(fast["output"][mask]),
        np.ptp(fast["nodes"]["CH_AMP_OP"][mask] -
               fast["nodes"]["CH_AMP_ON"][mask]),
    ])
    np.testing.assert_allclose(fast_metrics, strict_metrics, rtol=8e-4, atol=2e-6)


def test_pmos_chopper_output_filter_adds_filtered_sense_nodes():
    build = build_afe_pmos_chopper(
        switch_size=(5000.0, 30.0),
        output_filter=(1e6, 680e-12),
    )

    assert "CH_VOP_F" in build.topology.solved
    assert "CH_VON_F" in build.topology.solved
    assert build.output_nodes == ("CH_VOP", "CH_VON")
    assert build.sense_output_nodes == ("CH_VOP_F", "CH_VON_F")
    assert build.topology.outputs == ("CH_VOP_F", "CH_VON_F")
    assert build.topology.aliases["vop_raw"] == "CH_VOP"
    assert build.topology.aliases["vop_f"] == "CH_VOP_F"


def test_pmos_chopper_pac_pss_finite_difference_smoke():
    freqs = np.array([1.0, 50.0])
    result = pmos_chopper_pac(
        CHOPPER_UI_SIZES,
        CHOPPER_UI_BIAS,
        freqs,
        225.0,
        analytic=False,
        pss_kwargs=dict(
            switch_size=(5000.0, 30.0),
            edge_time=20e-6,
            n_points=17,
            refine_edges=False,
            charge_injection=False,
            tstab_periods=0,
            max_shooting_iters=0,
        ),
        transient_kwargs=dict(
            max_retry_subdivisions=0,
            fallback_least_squares=False,
        ),
    )

    assert result["method"] == "pss_finite_difference_shooting"
    assert result["response"].shape == freqs.shape
    assert np.isfinite(result["response"]).all()
    assert np.all(result["gains"] > 0.0)
    assert np.all(result["pac_residual"] < 1e-8)


# design #3 (Cadence afe_chop reference: chopped gain 22.8 dB, IRN ~8.65 uVrms)
_CHOP_D3_SIZES = {
    "M6": (4819, 63), "M7": (65426, 42), "M8": (65426, 42),
    "M9": (2876, 333), "M10": (2876, 333), "M11": (739, 50),
    "M12": (505, 134), "M13": (505, 134), "M14": (4553, 48), "M15": (4553, 48),
}
_CHOP_D3_NF = {"M6": 4, "M7": 128, "M8": 128, "M9": 6, "M10": 6,
               "M11": 1, "M12": 2, "M13": 2, "M14": 10, "M15": 10}
_CHOP_D3_BIAS = {"VDD": 40.0, "VCM": 31.38, "VB": 10.6, "VC": 16.47}


@pytest.mark.skipif(not os.environ.get("RUN_SLOW_CHOPPER"),
                    reason="slow PSS+pnoise verification; set RUN_SLOW_CHOPPER=1 to run")
def test_pmos_chopper_pnoise_matches_cadence_band():
    # PSS-based LPTV PNoise vs official chop_tb_d3 slow-corner Cadence reference
    # (IRN=12.4886 uVrms over 0.05-100 Hz; re-run at maxsideband=40 converges to
    # 12.498).  The local HB noise conversion converges more slowly in sidebands
    # than Spectre's shooting PNoise, so the chopper wrapper defaults to
    # max_sideband=32 (msb=10 lands typical/fast IRN -6%; msb=32 is within ~1.4%
    # across all corners).  gains default to the chopper PAC (K=64) so the
    # input-referral is consistent.  The converged local slow IRN sits ~+1.4%
    # above Spectre (a small corner-dependent noise residual).
    freqs = _spectre_dec_grid(0.05, 200.0, points_per_dec=10)
    pss = pmos_chopper_pss(
        _CHOP_D3_SIZES, _CHOP_D3_BIAS, 200.0, switch_size=(5000.0, 30.0),
        switch_nf=1, nf=_CHOP_D3_NF, edge_time=20e-6, input_diff=0.0,
        input_common_mode=31.38, charge_injection=False, tstab_periods=2,
        fallback_least_squares=False, n_points=161,
        output_filter=(1e6, 680e-12), corner="slow")
    r = pmos_chopper_pnoise(
        _CHOP_D3_SIZES, _CHOP_D3_BIAS, freqs, 200.0, pss_result=pss,
        nf=_CHOP_D3_NF, corner="slow", band=(0.05, 100.0))
    assert r["method"] == "pss_harmonic_balance_conversion_matrix"
    assert np.all(np.isfinite(r["out_psd"])) and np.all(r["out_psd"] > 0.0)
    assert r["max_sideband"] == 32
    assert r["pnoise_internal_gate1_states"] == len(pss["topology"].devices)
    assert r["pnoise_state_size"] == pss["topology"].n + len(pss["topology"].devices)
    # This older design-#3 / f_chop=200 reference is kept as a convergence guard.
    # The authoritative chopper calibration remains the fresh f_chop=225
    # 3-corner Spectre sweep; this test mainly prevents regressions in the PNoise
    # HB fold and now also guards that the PMOS gate1 states stay in the matrix.
    assert abs(r["irn_uV_band"] - 12.4886) / 12.4886 < 0.05


@pytest.mark.skipif(not os.environ.get("RUN_SLOW_CHOPPER"),
                    reason="slow PSS+pnoise; set RUN_SLOW_CHOPPER=1 to run")
def test_pmos_chopper_pnoise_time_domain_is_truncation_free():
    # The time-domain Floquet-adjoint PNoise (sparse BVP solve of F^H zeta = c)
    # computes the adjoint transfer EXACTLY in the sideband index, so its output is
    # ~independent of max_sideband -- unlike the HB fold, whose K-truncation needs
    # K~96 to converge (K=32 is +1.8% on the slow chopper). Guards that the TD path
    # is taken and is truncation-free.
    pss = pmos_chopper_pss(
        _CHOP_D3_SIZES, _CHOP_D3_BIAS, 200.0, switch_size=(5000.0, 30.0),
        switch_nf=1, nf=_CHOP_D3_NF, edge_time=20e-6, input_diff=0.0,
        input_common_mode=31.38, charge_injection=False, tstab_periods=2,
        fallback_least_squares=False, n_points=161,
        output_filter=(1e6, 680e-12), corner="slow")
    freqs = np.array([0.05, 1.0, 100.0])

    def irn(td, K):
        r = pmos_chopper_pnoise(
            _CHOP_D3_SIZES, _CHOP_D3_BIAS, freqs, 200.0, pss_result=copy.deepcopy(pss),
            nf=_CHOP_D3_NF, corner="slow", band=(0.05, 100.0),
            time_domain=td, max_sideband=K)
        return r["out_uV_band"], bool(r.get("pnoise_time_domain_used"))

    td_lo, used_lo = irn(True, 16)
    td_hi, used_hi = irn(True, 64)
    hb_lo, _ = irn(False, 16)
    hb_hi, _ = irn(False, 64)
    assert used_lo and used_hi                      # the TD path is taken
    # TD is truncation-free: K=16 vs K=64 agree tightly (no sideband truncation in
    # the adjoint), and the TD K-spread is far smaller than the HB's K-drift.
    td_spread = abs(td_hi - td_lo) / td_hi
    hb_spread = abs(hb_hi - hb_lo) / hb_hi
    assert td_spread < 2e-3
    assert td_spread < 0.5 * hb_spread


@pytest.mark.skipif(not os.environ.get("RUN_SLOW_CHOPPER"),
                    reason="slow PSS+PAC verification; set RUN_SLOW_CHOPPER=1 to run")
def test_pmos_chopper_pac_matches_cadence_baseband_gain():
    # PSS+PAC vs Cadence design-#3 PSS/PAC reference at f_chop=200 Hz.
    # The official ADE netlist is slow corner. The default chopper PAC path is
    # the time-domain shooting solver with PMOS gate1 internal states retained;
    # this avoids both HB sideband truncation and the old terminal-cap collapse.
    freqs = np.array([0.05, 1.0, 200.0])
    pss = pmos_chopper_pss(
        _CHOP_D3_SIZES, _CHOP_D3_BIAS, 200.0, switch_size=(5000.0, 30.0),
        switch_nf=1, nf=_CHOP_D3_NF, edge_time=20e-6, input_diff=0.0,
        input_common_mode=31.38, charge_injection=False, tstab_periods=2,
        fallback_least_squares=False, n_points=161,
        output_filter=(1e6, 680e-12), corner="slow")
    pac = pmos_chopper_pac(
        _CHOP_D3_SIZES, _CHOP_D3_BIAS, freqs, 200.0, pss_result=pss,
        nf=_CHOP_D3_NF, corner="slow")
    assert pac["method"] == "pss_time_domain"
    assert pac["pac_internal_gate1_states"] == len(pss["topology"].devices)
    assert pac["pac_state_size"] == pss["topology"].n + len(pss["topology"].devices)
    g = pac["gains"]
    assert abs(g[0] - 10.3975) / 10.3975 < 0.01
    assert abs(g[1] - 10.3975) / 10.3975 < 0.01
    assert abs(g[2] - 2.7305) / 2.7305 < 0.01


@pytest.mark.skipif(not os.environ.get("RUN_SLOW_CHOPPER"),
                    reason="slow gear2 PSS+PAC verification; set RUN_SLOW_CHOPPER=1")
def test_pmos_chopper_pac_gear2_matches_cadence_within_1pct():
    # gear2/BDF2 transient closes the backward-Euler switch-edge error: chopper
    # PAC baseband lands within 1% of Cadence (typical corner 13.921 V/V), vs
    # ~-2.6% with backward-Euler. Uses the Python gear2 path + FD shooting
    # Jacobian (the analytic monodromy is BE-specific).
    freqs = np.array([0.05, 200.0])
    pss = pmos_chopper_pss(
        _CHOP_D3_SIZES, _CHOP_D3_BIAS, 200.0, switch_size=(5000.0, 30.0),
        switch_nf=1, nf=_CHOP_D3_NF, edge_time=20e-6, input_diff=0.0,
        input_common_mode=31.38, charge_injection=False, tstab_periods=2,
        fallback_least_squares=False, n_points=321, max_shooting_iters=5,
        output_filter=(1e6, 680e-12), corner="typical",
        integration_method="gear2", analytic_jacobian=False)
    pac = pmos_chopper_pac(
        _CHOP_D3_SIZES, _CHOP_D3_BIAS, freqs, 200.0, pss_result=pss,
        nf=_CHOP_D3_NF, corner="typical")
    assert abs(pac["gains"][0] - 13.921) / 13.921 < 0.01


@pytest.mark.skipif(not os.environ.get("RUN_SLOW_CHOPPER"),
                    reason="slow time-domain PAC verification; set RUN_SLOW_CHOPPER=1")
def test_pmos_chopper_pac_time_domain_matches_cadence():
    # The default time-domain (shooting) PAC integrates the linearized orbit with
    # the Floquet BC x(T)=e^{jwT}x(0): a frequency-independent monodromy + a small
    # boundary solve per frequency, truncation-free. It must stay within 1% of
    # Cadence (13.921 V/V) while retaining the hidden PMOS gate1 states that
    # Spectre keeps during PAC conversion.
    freqs = np.array([0.05, 200.0])
    pss = pmos_chopper_pss(
        _CHOP_D3_SIZES, _CHOP_D3_BIAS, 200.0, switch_size=(5000.0, 30.0),
        switch_nf=1, nf=_CHOP_D3_NF, edge_time=20e-6, input_diff=0.0,
        input_common_mode=31.38, charge_injection=False, tstab_periods=2,
        fallback_least_squares=False, n_points=321, max_shooting_iters=5,
        output_filter=(1e6, 680e-12), corner="typical",
        integration_method="gear2", analytic_jacobian=False)
    td = pmos_chopper_pac(
        _CHOP_D3_SIZES, _CHOP_D3_BIAS, freqs, 200.0, pss_result=pss,
        nf=_CHOP_D3_NF, corner="typical")
    assert td["method"] == "pss_time_domain"
    assert td["pac_td_integration"] == "gear2"
    assert td["pac_internal_gate1_states"] == len(pss["topology"].devices)
    assert abs(td["gains"][0] - 13.921) / 13.921 < 0.01


def test_pmos_chopper_pac_gate1_numba_matches_python():
    # The gate1-retained Verilog-A (C(V)*ddt(V)) linearization has a numba kernel
    # (_pac_linearize_orbit_gate1_impl). It must reproduce the pure-Python assembly
    # to numerical noise (the only diff is the nested cap-derivative FD's internal
    # solve warm-start). Guards future kernel edits.
    import core.pac_solver as psp
    if psp.pac_linearize_orbit_gate1_numba is None:
        pytest.skip("numba unavailable")
    freqs = np.array([0.05, 1.0, 200.0])
    pss = pmos_chopper_pss(
        _CHOP_D3_SIZES, _CHOP_D3_BIAS, 200.0, switch_size=(5000.0, 30.0),
        switch_nf=1, nf=_CHOP_D3_NF, edge_time=20e-6, input_diff=0.0,
        input_common_mode=31.38, charge_injection=False, tstab_periods=2,
        fallback_least_squares=False, n_points=161, max_shooting_iters=4,
        output_filter=(1e6, 680e-12), corner="slow", cap_mode="average")

    def run():
        return pmos_chopper_pac(
            _CHOP_D3_SIZES, _CHOP_D3_BIAS, freqs, 200.0,
            pss_result=copy.deepcopy(pss), nf=_CHOP_D3_NF, corner="slow")["gains"]

    g_numba = np.abs(run())
    orig = psp._try_numba_pac_linearization_gate1
    psp._try_numba_pac_linearization_gate1 = lambda *a, **k: None
    try:
        g_python = np.abs(run())
    finally:
        psp._try_numba_pac_linearization_gate1 = orig
    # both retain gate1 states; values agree to FD noise (<0.05%).
    np.testing.assert_allclose(g_numba, g_python, rtol=5e-4)
