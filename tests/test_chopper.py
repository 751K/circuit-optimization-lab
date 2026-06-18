import numpy as np

from core.chopper import (
    build_afe_pmos_chopper,
    chopper_analysis,
    finite_edge_chopper_harmonics,
    finite_edge_clock_pair,
    pmos_chopper_analysis,
    pmos_chopper_lptv_analysis,
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
    assert "Quasi-static" in result["analysis_note"]


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
        charge_injection=True,
        charge_scale=0.05,
        switch_size=(10000.0, 80.0),
        edge_points=7,
    )

    assert result["nfail"] == 0
    assert result["refined_point_count"] > len(t)
    assert len(result["charge_injection_sources"]) > 0
    assert np.isfinite(result["output"]).all()
    assert np.isfinite(result["requested_output"]).all()
