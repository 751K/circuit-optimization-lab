"""Pure-Python logic tests for the MDAC PVT-campaign layer.

These exercise the campaign *machinery* — settle-time aggregation, the noise
sign-off gate, the errors CSV, risk-first grid ordering, ``--smoke`` / ``--points``
handling — WITHOUT any foundry model payload or a real ngspice run.  Every
simulation entry point is monkeypatched; ``main`` is driven with a fake
``run_point`` and a temporary output path.

Importing ``tsmc28_mdac_pvt_campaign`` runs module-level code that imports the
circuitopt oracles and the TSMC28 generator (no sims execute at import), so both
modules load here as long as ``examples`` and ``experiments`` are on ``sys.path``.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "examples", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mdac_ota_pvt_campaign as base  # noqa: E402
import tsmc28_mdac_pvt_campaign as tsmc  # noqa: E402

campaign = tsmc.campaign  # the same module object as ``base`` after monkeypatching
FS = base.FS


# ── fixtures / helpers ──────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset_module_state():
    """Keep module-level campaign flags/hooks from leaking across tests."""
    saved = {
        "GRID_PRIORITY": list(campaign.GRID_PRIORITY),
        "SKIP_NOISE": campaign.SKIP_NOISE,
        "SKIP_CODE_TRANSITION": campaign.SKIP_CODE_TRANSITION,
        "run_point": campaign.run_point,
    }
    yield
    campaign.GRID_PRIORITY = saved["GRID_PRIORITY"]
    campaign.SKIP_NOISE = saved["SKIP_NOISE"]
    campaign.SKIP_CODE_TRANSITION = saved["SKIP_CODE_TRANSITION"]
    campaign.run_point = saved["run_point"]


def _trace(residue, *, settles):
    """A synthetic residue transient.  ``settles`` True => the late samples sit on
    the ideal ``vod = -8*residue`` (inside the 0.1% band); False => they stay a
    volt away so the level never enters the band (settle time -> nan)."""
    t = np.linspace(0.0, 5e-9, 11)
    ideal = -8.0 * residue
    vod = np.full_like(t, ideal if settles else ideal + 1.0)
    # OUTP - OUTN == vod with a benign CM of 0 relative to vdd/2 handled by caller.
    outp = ideal * 0.0 + vod / 2.0
    outn = -vod / 2.0
    return {"t": t, "nodes": {"OUTP": outp, "OUTN": outn}}


def _fake_base_row(**overrides):
    """A base run_point row with every base pass flag present and passing."""
    row = {
        "corner": "tt", "temp_c": 27.0, "vdd": 0.9,
        "gain_db": 90.0, "ac_ugbw_hz": 1e9, "ac_pm_deg": 80.0,
        "dm_ugf_hz": 1e8, "dm_pm_deg": 90.0, "dm_gm_db": 20.0,
        "cmfb1_ugf_hz": 1e7, "cmfb1_pm_deg": 80.0,
        "cmfb2_ugf_hz": 1e7, "cmfb2_pm_deg": 80.0,
        "settle_n2_pct": 0.01, "settle_n1_pct": 0.01, "settle_z_pct": 0.01,
        "settle_p1_pct": 0.01, "settle_p2_pct": 0.01, "settle_worst_pct": 0.01,
        "overshoot_worst_pct": 1.0, "cm5_worst_mv": 2.0,
        "cm_static_mv": 1.0,
        "sat_static_ok": True, "sat_settled_ok": True, "sat_bad": "",
        "noise_onoise_uv": 50.0, "noise_adc_uv": 6.0,
        "isupply_ma": 1.0, "power_mw": 1.0,
        "pass_gain": True, "pass_dmpm": True, "pass_cmfb1pm": True,
        "pass_cmfb2pm": True, "pass_settle": True, "pass_cm": True,
        "pass_sat": True, "pass_noise": True, "pass_all": True,
    }
    row.update(overrides)
    return row


def _install_fake_base(monkeypatch, *, settle_flags, wideband_uv, base_overrides=None):
    """Replace tsmc._base_run_point with a fake that fills _tls like the real base
    would (five residue traces + wideband noise) and returns a synthetic base row."""
    def fake_base(corner, temp_c, vdd):
        tsmc._tls.transients = [
            _trace(residue, settles=flag)
            for residue, flag in zip(base.RESIDUE_LEVELS, settle_flags, strict=True)
        ]
        tsmc._tls.noise_wideband = float("nan") if wideband_uv is None else wideband_uv / 1e6
        return _fake_base_row(**(base_overrides or {}))
    monkeypatch.setattr(tsmc, "_base_run_point", fake_base)


def _fake_code(**overrides):
    code = {
        "code_transition_pct": 0.02, "code_settle_ns": 3.0,
        "code_peak_glitch_pct": 5.0, "code_cm5_mv": 2.0,
        "code_cm5_signed_mv": 2.0, "code_sat_bad": [], "code_runtime_s": 0.0,
    }
    code.update(overrides)
    return code


# ── settle-time aggregation: any nan level => worst is nan ───────────────────────
@pytest.mark.parametrize("bad_index", [0, 2, 4])
def test_settle_time_worst_is_nan_when_any_level_never_settles(monkeypatch, bad_index):
    settle_flags = [True] * 5
    settle_flags[bad_index] = False       # nan first / middle / last position
    _install_fake_base(monkeypatch, settle_flags=settle_flags, wideband_uv=100.0)
    monkeypatch.setattr(tsmc, "_code_transition", lambda *a, **k: _fake_code())
    row = tsmc.run_point("tt", 27.0, 0.9)
    assert np.isnan(row["settle_time_worst_ns"]), (
        f"a never-settled level (index {bad_index}) must force worst=nan")


def test_settle_time_worst_is_finite_when_all_settle(monkeypatch):
    _install_fake_base(monkeypatch, settle_flags=[True] * 5, wideband_uv=100.0)
    monkeypatch.setattr(tsmc, "_code_transition", lambda *a, **k: _fake_code())
    row = tsmc.run_point("tt", 27.0, 0.9)
    assert np.isfinite(row["settle_time_worst_ns"])
    assert row["settle_time_worst_ns"] >= 0.0


# ── noise sign-off gate: wideband decides, narrowband is inert ───────────────────
def test_pass_noise_uses_wideband_460_fails(monkeypatch):
    # Narrowband report value is comfortably under spec; only the wideband quantity
    # (460 > 452) must decide, so pass_noise is False and pass_all is False.
    _install_fake_base(monkeypatch, settle_flags=[True] * 5, wideband_uv=460.0,
                       base_overrides={"noise_onoise_uv": 50.0, "pass_noise": True})
    monkeypatch.setattr(tsmc, "_code_transition", lambda *a, **k: _fake_code())
    row = tsmc.run_point("tt", 27.0, 0.9)
    assert row["noise_wideband_onoise_uv"] == pytest.approx(460.0)
    assert row["pass_noise"] is False
    assert row["pass_all"] is False


def test_pass_noise_uses_wideband_440_passes(monkeypatch):
    _install_fake_base(monkeypatch, settle_flags=[True] * 5, wideband_uv=440.0,
                       base_overrides={"noise_onoise_uv": 50.0})
    monkeypatch.setattr(tsmc, "_code_transition", lambda *a, **k: _fake_code())
    row = tsmc.run_point("tt", 27.0, 0.9)
    assert row["noise_wideband_onoise_uv"] == pytest.approx(440.0)
    assert row["pass_noise"] is True
    assert row["pass_all"] is True


def test_narrowband_value_plays_no_role_in_gate(monkeypatch):
    # Narrowband huge (would fail the OLD gate), wideband fine -> pass_noise True.
    _install_fake_base(monkeypatch, settle_flags=[True] * 5, wideband_uv=100.0,
                       base_overrides={"noise_onoise_uv": 9999.0, "pass_noise": False})
    monkeypatch.setattr(tsmc, "_code_transition", lambda *a, **k: _fake_code())
    row = tsmc.run_point("tt", 27.0, 0.9)
    assert row["pass_noise"] is True, "narrowband must not gate sign-off"
    assert row["pass_all"] is True


def test_pass_noise_spec_boundary_is_le(monkeypatch):
    # exactly at spec (452) passes (<=); just above (452.001) fails.
    for value, expected in ((campaign.SPEC_NOISE_UV, True),
                            (campaign.SPEC_NOISE_UV + 0.001, False)):
        _install_fake_base(monkeypatch, settle_flags=[True] * 5, wideband_uv=value)
        monkeypatch.setattr(tsmc, "_code_transition", lambda *a, **k: _fake_code())
        row = tsmc.run_point("tt", 27.0, 0.9)
        assert row["pass_noise"] is expected, f"wideband {value} -> {expected}"


# ── errors.csv on a failing point; main CSV untouched by the failure ────────────
def test_errors_csv_written_for_failing_point(monkeypatch, tmp_path):
    out = tmp_path / "run.csv"

    def flaky_run_point(corner, temp_c, vdd):
        if (corner, temp_c, vdd) == ("ss", 125.0, 0.85):
            raise RuntimeError("ngspice blew up\nwith a two line message")
        row = _fake_base_row(corner=corner, temp_c=temp_c, vdd=vdd)
        # add the tsmc extra columns so the writer (tsmc CSV_FIELDS) is satisfied
        for field in tsmc.EXTRA_FIELDS:
            row.setdefault(field, float("nan"))
        row["smoke"] = 0
        row["pass_code_transition"] = True
        return row

    monkeypatch.setattr(campaign, "run_point", flaky_run_point)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--out", str(out), "--workers", "1", "--force",
        "--points", "ss/125/0.85,tt/27/0.9",
    ])
    campaign.main()

    errors = out.with_suffix(".errors.csv")
    assert errors.is_file(), "a failing point must append to the errors CSV"
    rows = list(csv.DictReader(errors.open()))
    assert len(rows) == 1
    r = rows[0]
    assert (r["corner"], r["temp_c"], r["vdd"]) == ("ss", "125", "0.85")
    assert "\n" not in r["error"] and "ngspice blew up" in r["error"]
    assert "two line message" in r["error"]        # both lines folded onto one
    assert float(r["elapsed_s"]) >= 0.0

    # main CSV holds only the successful point.
    main_rows = list(csv.DictReader(out.open()))
    assert len(main_rows) == 1
    assert (main_rows[0]["corner"], main_rows[0]["vdd"]) == ("tt", "0.9")


def test_errors_csv_absent_when_no_failure(monkeypatch, tmp_path):
    out = tmp_path / "run.csv"

    def ok_run_point(corner, temp_c, vdd):
        row = _fake_base_row(corner=corner, temp_c=temp_c, vdd=vdd)
        for field in tsmc.EXTRA_FIELDS:
            row.setdefault(field, float("nan"))
        row["smoke"] = 0
        row["pass_code_transition"] = True
        return row

    monkeypatch.setattr(campaign, "run_point", ok_run_point)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--out", str(out), "--workers", "1", "--force",
        "--points", "tt/27/0.9",
    ])
    campaign.main()
    assert not out.with_suffix(".errors.csv").exists()


# ── risk-first grid ordering ─────────────────────────────────────────────────────
def test_grid_priority_runs_first_in_order():
    grid = [(c, t, v) for c in campaign.CORNERS
            for t in campaign.TEMPS_C for v in campaign.SUPPLIES]
    priority = [
        ("ss", 125.0, 0.85), ("sf", -40.0, 0.85), ("ss", -40.0, 0.85),
        ("ff", 125.0, 0.95), ("fs", -40.0, 0.95), ("tt", 27.0, 0.90),
    ]
    campaign.GRID_PRIORITY = priority
    todo = campaign._order_todo(grid, {})
    assert todo[:len(priority)] == priority
    # every grid point is present exactly once, priority points not duplicated later.
    assert sorted(campaign._key(*p) for p in todo) == sorted(
        campaign._key(*p) for p in grid)
    assert len(todo) == len(grid)


def test_grid_priority_empty_preserves_natural_order():
    grid = [(c, t, v) for c in campaign.CORNERS
            for t in campaign.TEMPS_C for v in campaign.SUPPLIES]
    campaign.GRID_PRIORITY = []
    assert campaign._order_todo(grid, {}) == grid


def test_grid_priority_skips_already_done():
    grid = [(c, t, v) for c in campaign.CORNERS
            for t in campaign.TEMPS_C for v in campaign.SUPPLIES]
    campaign.GRID_PRIORITY = [("ss", 125.0, 0.85), ("tt", 27.0, 0.90)]
    done = {campaign._key("ss", 125.0, 0.85): {}}
    todo = campaign._order_todo(grid, done)
    assert campaign._key("ss", 125.0, 0.85) not in {campaign._key(*p) for p in todo}
    assert todo[0] == ("tt", 27.0, 0.90)


# ── --smoke: 6-point subset + skip flags + smoke column ─────────────────────────
def test_smoke_runs_priority_subset_and_sets_skip_flags(monkeypatch, tmp_path):
    out = tmp_path / "smoke.csv"
    seen = []

    def rec_run_point(corner, temp_c, vdd):
        seen.append((corner, temp_c, vdd))
        # smoke flags must already be set when run_point executes
        assert campaign.SKIP_NOISE and campaign.SKIP_CODE_TRANSITION
        row = _fake_base_row(corner=corner, temp_c=temp_c, vdd=vdd)
        for field in tsmc.EXTRA_FIELDS:
            row.setdefault(field, float("nan"))
        row["smoke"] = 1
        row["pass_code_transition"] = False
        return row

    campaign.GRID_PRIORITY = [
        ("ss", 125.0, 0.85), ("sf", -40.0, 0.85), ("ss", -40.0, 0.85),
        ("ff", 125.0, 0.95), ("fs", -40.0, 0.95), ("tt", 27.0, 0.90),
    ]
    monkeypatch.setattr(campaign, "run_point", rec_run_point)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--out", str(out), "--workers", "1", "--force", "--smoke",
    ])
    campaign.main()

    assert sorted(campaign._key(*p) for p in seen) == sorted(
        campaign._key(*p) for p in campaign.GRID_PRIORITY)
    assert len(seen) == 6
    rows = list(csv.DictReader(out.open()))
    assert all(r["smoke"] == "1" for r in rows)
    assert "smoke" in campaign.CSV_FIELDS


def test_smoke_skip_flags_change_run_point_columns(monkeypatch):
    # With skip flags set, tsmc.run_point must NOT call _code_transition and must
    # null the code columns / not sign off noise or code.
    campaign.SKIP_NOISE = True
    campaign.SKIP_CODE_TRANSITION = True
    _install_fake_base(monkeypatch, settle_flags=[True] * 5, wideband_uv=None,
                       base_overrides={"pass_noise": False})

    def _boom(*a, **k):
        raise AssertionError("_code_transition must be skipped in smoke mode")
    monkeypatch.setattr(tsmc, "_code_transition", _boom)

    row = tsmc.run_point("tt", 27.0, 0.9)
    assert row["smoke"] == 1
    assert np.isnan(row["code_transition_pct"])
    assert np.isnan(row["code_settle_ns"])
    assert row["pass_code_transition"] is False
    assert row["pass_noise"] is False
    # pass_all is computed over the measured specs only; all of those pass here.
    assert row["pass_all"] is True


# ── --points / --smoke argument validation ──────────────────────────────────────
def test_points_parsing_rejects_malformed_and_unknown():
    grid = [(c, t, v) for c in campaign.CORNERS
            for t in campaign.TEMPS_C for v in campaign.SUPPLIES]
    # well-formed, known
    assert campaign._parse_points("tt/27/0.9", grid) == [("tt", 27.0, 0.9)]
    assert campaign._parse_points("ss/125/0.85,tt/27/0.9", grid) == [
        ("ss", 125.0, 0.85), ("tt", 27.0, 0.9)]
    # malformed token (wrong field count)
    with pytest.raises(ValueError):
        campaign._parse_points("tt/27", grid)
    # non-numeric temp/vdd
    with pytest.raises(ValueError):
        campaign._parse_points("tt/warm/0.9", grid)
    # unknown corner / temp / vdd (each off-grid)
    with pytest.raises(ValueError):
        campaign._parse_points("zz/27/0.9", grid)
    with pytest.raises(ValueError):
        campaign._parse_points("tt/999/0.9", grid)
    with pytest.raises(ValueError):
        campaign._parse_points("tt/27/9.9", grid)
    # empty
    with pytest.raises(ValueError):
        campaign._parse_points("   ", grid)


def test_points_parsing_dedupes_preserving_order():
    grid = [(c, t, v) for c in campaign.CORNERS
            for t in campaign.TEMPS_C for v in campaign.SUPPLIES]
    got = campaign._parse_points("tt/27/0.9,tt/27/0.9,ss/125/0.85", grid)
    assert got == [("tt", 27.0, 0.9), ("ss", 125.0, 0.85)]


def test_smoke_and_points_mutually_exclusive(monkeypatch, tmp_path):
    out = tmp_path / "x.csv"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--out", str(out), "--smoke", "--points", "tt/27/0.9",
    ])
    with pytest.raises(SystemExit):        # argparse .error() raises SystemExit
        campaign.main()


# ── S4 chained-ngspice toggle + batch-hook wiring (no sims execute) ──────────────
def _boom(*_a, **_k):
    raise AssertionError("this oracle must not be called on this code path")


def test_ngspice_chain_enabled_env_and_override(monkeypatch):
    from circuitopt.ngspice_char import ngspice_chain_enabled
    monkeypatch.delenv("CIRCUITOPT_NGSPICE_CHAIN", raising=False)
    assert ngspice_chain_enabled() is True            # unset -> chained
    monkeypatch.setenv("CIRCUITOPT_NGSPICE_CHAIN", "1")
    assert ngspice_chain_enabled() is True
    monkeypatch.setenv("CIRCUITOPT_NGSPICE_CHAIN", "0")
    assert ngspice_chain_enabled() is False           # "0" -> every lever off
    assert ngspice_chain_enabled(True) is True        # kwarg beats env
    monkeypatch.setenv("CIRCUITOPT_NGSPICE_CHAIN", "1")
    assert ngspice_chain_enabled(False) is False      # ... in both directions


def test_base_transient_batch_env_off_loops_per_case(monkeypatch):
    """env=0: one campaign.transient_ngspice call per case (wrapper slot), the
    chain oracle never runs, per-case inputs and results keep case order.
    ``tsmc._base_transient_batch`` is the base module's default hook, saved by
    the TSMC override exactly like ``_base_run_point``."""
    monkeypatch.setenv("CIRCUITOPT_NGSPICE_CHAIN", "0")
    seen = []

    def recorder(sizes, bias, tgrid, **kwargs):
        seen.append(kwargs["inputs"])
        return {"case": len(seen) - 1, "corner": kwargs["corner"]}

    monkeypatch.setattr(campaign, "transient_ngspice", recorder)
    monkeypatch.setattr(campaign, "transient_ngspice_chain", _boom)
    cases = [{"inputs": {"bp1": pos}} for pos in range(3)]
    out = tsmc._base_transient_batch(("S", "B", "T"), cases, topo="TOPO", corner="tt")
    assert out == [{"case": 0, "corner": "tt"}, {"case": 1, "corner": "tt"},
                   {"case": 2, "corner": "tt"}]
    assert seen == [{"bp1": 0}, {"bp1": 1}, {"bp1": 2}]


def test_base_transient_batch_env_on_single_chain_call(monkeypatch):
    monkeypatch.setenv("CIRCUITOPT_NGSPICE_CHAIN", "1")
    calls = []

    def rec_chain(sizes, bias, tgrid, *, cases, **kwargs):
        calls.append((sizes, bias, tgrid, cases, kwargs))
        return [{"case": pos} for pos in range(len(cases))]

    monkeypatch.setattr(campaign, "transient_ngspice_chain", rec_chain)
    monkeypatch.setattr(campaign, "transient_ngspice", _boom)
    cases = [{"inputs": {"bp1": pos}} for pos in range(3)]
    out = tsmc._base_transient_batch(("S", "B", "T"), cases, topo="TOPO", corner="ss")
    assert out == [{"case": 0}, {"case": 1}, {"case": 2}]
    assert len(calls) == 1
    sizes, bias, tgrid, chain_cases, kwargs = calls[0]
    assert (sizes, bias, tgrid) == ("S", "B", "T")
    assert chain_cases == cases
    assert kwargs == {"topo": "TOPO", "corner": "ss"}


def test_tsmc_transient_batch_env_on_contracts(monkeypatch):
    """env=1: ONE chain call for all levels; hold clocks merged into every case,
    op_devices=CORE_DEVS, whole-chain timeout, _tls.transients filled in
    RESIDUE_LEVELS order, caller's case dicts untouched."""
    monkeypatch.setenv("CIRCUITOPT_NGSPICE_CHAIN", "1")
    tgrid = np.linspace(0.0, 5e-9, 11)
    spec_args = ({}, {"VDD": 0.9}, tgrid)
    cases = []
    for residue in base.RESIDUE_LEVELS:
        bp1 = np.full(11, 0.45 + residue / 2); bp1[0] = 0.45
        cases.append({"inputs": {"bp1": bp1}})
    fake_results = [{"level": pos} for pos in range(len(cases))]
    calls = []

    def rec_chain(sizes, bias, tg, *, cases, **kwargs):
        calls.append((cases, kwargs))
        return list(fake_results)

    monkeypatch.setattr(tsmc, "_transient_ngspice_chain", rec_chain)
    monkeypatch.setattr(campaign, "transient_ngspice", _boom)
    tsmc._tls.transients = []
    out = campaign.transient_batch(spec_args, cases, topo="TOPO")
    assert out == fake_results
    assert tsmc._tls.transients == fake_results        # order preserved
    (chain_cases, kwargs), = calls
    assert kwargs["op_devices"] == campaign.CORE_DEVS
    assert kwargs["timeout"] == pytest.approx(1200.0 * len(cases))
    hold = tsmc.G.hold_clock_inputs(tgrid, 0.9)
    for pos, chained in enumerate(chain_cases):
        assert set(chained["inputs"]) == {"bp1", "DCH", "DCHB"}
        np.testing.assert_array_equal(chained["inputs"]["DCH"], hold["DCH"])
        np.testing.assert_array_equal(chained["inputs"]["DCHB"], hold["DCHB"])
        np.testing.assert_array_equal(chained["inputs"]["bp1"],
                                      cases[pos]["inputs"]["bp1"])
    assert all(set(case["inputs"]) == {"bp1"} for case in cases)   # no mutation


def test_tsmc_transient_batch_env_off_routes_through_wrapper_slot(monkeypatch):
    """env=0: the TSMC batch hook must fall back to per-case calls through the
    campaign.transient_ngspice slot (today's per-process wrappers), never the
    chain oracle."""
    monkeypatch.setenv("CIRCUITOPT_NGSPICE_CHAIN", "0")
    tgrid = np.linspace(0.0, 5e-9, 11)
    seen = []

    def recorder(sizes, bias, tg, **kwargs):
        seen.append(kwargs["inputs"])
        return {"case": len(seen) - 1}

    monkeypatch.setattr(campaign, "transient_ngspice", recorder)
    monkeypatch.setattr(tsmc, "_transient_ngspice_chain", _boom)
    cases = [{"inputs": {"bp1": pos}} for pos in range(5)]
    out = campaign.transient_batch(({}, {"VDD": 0.9}, tgrid), cases, topo="TOPO")
    assert out == [{"case": pos} for pos in range(5)]
    assert seen == [{"bp1": pos} for pos in range(5)]


class _FakeBinding:
    model_types = {}
    device_kwargs = {}


class _FakeSpec:
    sizes = {}
    bias = {}
    nf = None
    topology = "TOPO-SENTINEL"

    def binding(self):
        return _FakeBinding()


def test_ac_power_static_env_off_two_calls_in_todays_order(monkeypatch):
    monkeypatch.setenv("CIRCUITOPT_NGSPICE_CHAIN", "0")
    order = []

    def rec_ac(sizes, bias, **kwargs):
        order.append(("ac", kwargs["acmag"], kwargs["out_nodes"]))
        return {"freq": "F"}

    def rec_supply(spec, dk, corner, seed, core_devs=None):
        order.append(("op", core_devs))
        return 1e-3, {"M0": True}

    monkeypatch.setattr(campaign, "ac_ngspice", rec_ac)
    monkeypatch.setattr(campaign, "_supply_current", rec_supply)
    monkeypatch.setattr(campaign, "_ac_power_static_merged", _boom)
    ac, isup, regions = campaign._ac_power_static(
        _FakeSpec(), {}, "tt", {"OUTP": 0.45},
        acmag={"VACP": (0.5, 0.0)}, fstart=1e4, fstop=5e10, points=25,
        out_nodes=["OUTP", "OUTN"], core_devs=["M0"])
    assert ac == {"freq": "F"} and isup == 1e-3 and regions == {"M0": True}
    assert order == [("ac", {"VACP": (0.5, 0.0)}, ["OUTP", "OUTN"]),
                     ("op", ["M0"])]                    # today's call order


def test_ac_power_static_env_on_single_merged_call(monkeypatch):
    monkeypatch.setenv("CIRCUITOPT_NGSPICE_CHAIN", "1")
    calls = []
    sentinel = ({"freq": "F"}, 2e-3, {"M0": False})

    def rec_merged(spec, dk, corner, seed, **kwargs):
        calls.append((spec, dk, corner, seed, kwargs))
        return sentinel

    monkeypatch.setattr(campaign, "_ac_power_static_merged", rec_merged)
    monkeypatch.setattr(campaign, "ac_ngspice", _boom)
    monkeypatch.setattr(campaign, "_supply_current", _boom)
    spec = _FakeSpec()
    result = campaign._ac_power_static(
        spec, {"M0": {}}, "ss", {"OUTP": 0.45},
        acmag={"VACP": (0.5, 0.0)}, fstart=1e4, fstop=5e10, points=25,
        out_nodes=["OUTP", "OUTN"], core_devs=["M0"])
    assert result == sentinel
    (got_spec, dk, corner, seed, kwargs), = calls
    assert got_spec is spec and corner == "ss" and dk == {"M0": {}}
    assert kwargs == {"acmag": {"VACP": (0.5, 0.0)}, "fstart": 1e4, "fstop": 5e10,
                      "points": 25, "out_nodes": ["OUTP", "OUTN"],
                      "core_devs": ["M0"]}
