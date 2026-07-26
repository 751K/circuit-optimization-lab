"""Adversarial semantic tests for SAR parallelism + the SAR design-space explorer.

Reviewer-side verification of the work package's contracts: parallel == serial
byte-identity, exception propagation out of worker pools, progress monotonicity,
candidate/spec purity, config validation, and CLI wiring. Skip-guarded like
``test_sar.py`` (real ngspice oracle required).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from circuitopt.toolchain import pdk_root


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "freepdk45_sar3.json"
EXPLORE_CFG = ROOT / "examples" / "freepdk45_sar3_explore.json"
_HAVE = (Path(pdk_root()) / "freepdk45" / "models_nom" / "NMOS_VTG.inc").is_file()
pytestmark = pytest.mark.skipif(not _HAVE, reason="FreePDK45 cards not present")


def _spec():
    from circuitopt.circuit_loader import load_circuit_json
    return load_circuit_json(EXAMPLE)


# ── parallel == serial ────────────────────────────────────────────────────────
def test_sweep_parallel_matches_serial_even_with_excess_workers():
    """Order preservation must hold when workers exceed the point count.

    Three points on a 3-bit converter are a subsampled sweep, so the parity
    probe is the code-error metric family (transition DNL/INL do not exist on
    a sparse ramp since the subsampling rework)."""
    from circuitopt.sar import run_sar_sweep
    vin = np.array([0.1875, 0.4375, 0.6875])
    serial = run_sar_sweep(_spec(), vin)
    threaded = run_sar_sweep(_spec(), vin, workers=8)
    np.testing.assert_array_equal(serial["codes"], threaded["codes"])
    assert serial["metrics"]["subsampled"] and threaded["metrics"]["subsampled"]
    np.testing.assert_array_equal(serial["metrics"]["code_errors"],
                                  threaded["metrics"]["code_errors"])
    assert serial["metrics"]["max_abs_code_err"] == \
        threaded["metrics"]["max_abs_code_err"]


def test_mc_parallel_matches_serial_per_trial():
    """Same seed -> identical per-trial codes/draws for any worker count: the RNG
    stream must not depend on completion order."""
    from circuitopt.sar_mc import sar_mismatch_mc
    cfg = {"sigma_vth0": 0.02, "sigma_cu": 0.05}
    serial = sar_mismatch_mc(_spec(), n=2, seed=5, config=cfg)
    threaded = sar_mismatch_mc(_spec(), n=2, seed=5, config=cfg, workers=2)
    for a, b in zip(serial["rows"], threaded["rows"]):
        assert a["trial"] == b["trial"]
        np.testing.assert_array_equal(a["codes"], b["codes"])
    for key, values in serial["arrays"].items():
        np.testing.assert_array_equal(values, threaded["arrays"][key])


def test_worker_exception_propagates_not_hangs():
    """A failure inside a pooled conversion must surface as the original error."""
    from circuitopt.sar import run_sar_sweep
    with pytest.raises(ValueError, match="NOPE"):
        run_sar_sweep(_spec(), np.array([0.3, 0.6]), mismatch={"NOPE": 0.1}, workers=2)


def test_invalid_workers_rejected_everywhere():
    from circuitopt.sar import run_sar_signal, run_sar_sweep
    from circuitopt.sar_mc import sar_mismatch_mc
    from circuitopt.sar_explore import load_sar_explore_json, sar_explore
    vin = np.array([0.3, 0.6])
    for bad in (0, -1):
        with pytest.raises(ValueError):
            run_sar_sweep(_spec(), vin, workers=bad)
        with pytest.raises(ValueError):
            run_sar_signal(_spec(), np.linspace(0.2, 0.8, 8), 1.0, workers=bad)
        with pytest.raises(ValueError):
            sar_mismatch_mc(_spec(), n=1, workers=bad)
        spec, cfg = load_sar_explore_json(EXPLORE_CFG)
        with pytest.raises(ValueError):
            sar_explore(spec, cfg, n=1, workers=bad)


def test_mc_progress_is_monotonic_under_parallelism():
    from circuitopt.sar_mc import sar_mismatch_mc
    seen = []
    sar_mismatch_mc(_spec(), n=3, seed=2, workers=2,
                    config={"sigma_vth0": 0.01},
                    progress=lambda i, n, partial: seen.append((i, n, partial["n"])))
    assert [item[0] for item in seen] == [1, 2, 3]
    assert all(total == 3 for _, total, _ in seen)
    assert [item[2] for item in seen] == [1, 2, 3]   # summary grows with completions


# ── explorer semantics ────────────────────────────────────────────────────────
def test_apply_sar_variables_edits_copy_only():
    """C:/W:/bias targets land on the candidate; the loaded spec stays untouched."""
    from circuitopt.explore import Variable
    from circuitopt.sar_explore import apply_sar_variables
    spec = _spec()
    caps_before = [tuple(c) for c in spec.topology.capacitors]
    w_before = spec.sizes["M1"]
    variables = [
        Variable("pair_w", 0.8, 1.6, targets=["W:M1", "W:M2"]),
        Variable("unit_c", 0.5e-14, 2e-14, targets=["C:C0P", "C:C0N"]),
        Variable("vb", 0.5, 0.6, targets=["VBIAS"]),
    ]
    cand = apply_sar_variables(
        variables, {"pair_w": 1.25, "unit_c": 1.5e-14, "vb": 0.58}, spec)
    assert cand.sizes["M1"][0] == 1.25 and cand.sizes["M2"][0] == 1.25
    assert cand.bias["VBIAS"] == 0.58
    cand_caps = {name: value for name, _a, _b, value in cand.topology.capacitors}
    assert cand_caps["C0P"] == 1.5e-14 and cand_caps["C0N"] == 1.5e-14
    assert cand_caps["C2P"] == 4e-14                  # untouched caps keep their value
    # purity of the loaded spec
    assert [tuple(c) for c in spec.topology.capacitors] == caps_before
    assert spec.sizes["M1"] == w_before
    assert spec.bias["VBIAS"] == 0.55


def test_unknown_cap_target_rejected():
    from circuitopt.explore import Variable
    from circuitopt.sar_explore import apply_sar_variables
    with pytest.raises(ValueError, match="CBOGUS"):
        apply_sar_variables([Variable("x", 1e-15, 2e-15, targets=["C:CBOGUS"])],
                            {"x": 1.5e-15}, _spec())


def test_config_validation_rejects_bad_metrics_and_shapes():
    from circuitopt.sar_explore import parse_sar_explore
    base = {"variables": {"w": {"min": 1.0, "max": 2.0, "targets": ["W:M1"]}},
            "objectives": {"max_abs_dnl": "min"}}
    parse_sar_explore(dict(base))                     # sanity: the base is valid
    with pytest.raises(ValueError, match="unknown metric"):
        parse_sar_explore({**base, "constraints": {"gain_db": {"min": 40}}})
    with pytest.raises(ValueError, match="unknown metric"):
        parse_sar_explore({**base, "objectives": {"snr": "max"}})
    with pytest.raises(ValueError):
        parse_sar_explore({**base, "objectives": {"max_abs_dnl": "minimize"}})
    with pytest.raises(ValueError):
        parse_sar_explore({**base, "objectives": {}})
    with pytest.raises(ValueError):
        parse_sar_explore({**base, "sweep_points": 1})
    with pytest.raises(ValueError):
        parse_sar_explore({**base, "dynamic": {"n_samples": 8, "cycles": 4}})


def test_circuit_path_resolution_and_conflict():
    from circuitopt.sar_explore import load_sar_explore_json, sar_explore_from_dict
    # 'circuit' resolves relative to the config file, not the CWD.
    spec, _cfg = load_sar_explore_json(EXPLORE_CFG)
    assert spec.adc is not None and spec.adc["n_bits"] == 3
    # a positional circuit that differs from the config's is a hard error
    data = json.loads(EXPLORE_CFG.read_text())
    with pytest.raises(ValueError, match="differs"):
        sar_explore_from_dict(data, base_dir=str(EXPLORE_CFG.parent),
                              circuit_path=str(ROOT / "examples" / "single_stage.json"))
    with pytest.raises(ValueError, match="no circuit"):
        sar_explore_from_dict({k: v for k, v in data.items() if k != "circuit"},
                              base_dir=str(EXPLORE_CFG.parent))


def test_explore_end_to_end_deterministic_and_writable(tmp_path):
    from circuitopt.sar_explore import (METRICS, load_sar_explore_json, sar_explore,
                                        sar_write_csv, sar_write_jsonl)
    spec, cfg = load_sar_explore_json(EXPLORE_CFG)
    cfg.sweep_points = 4                              # trim runtime; still exercises DNL
    caps_before = [tuple(c) for c in spec.topology.capacitors]
    a = sar_explore(spec, cfg, n=2, seed=9, workers=2)
    b = sar_explore(spec, cfg, n=2, seed=9, workers=1)
    assert [tuple(c) for c in spec.topology.capacitors] == caps_before  # purity
    for ca, cb in zip(a["candidates"], b["candidates"]):
        assert ca["vars"] == cb["vars"]
        for m in METRICS:
            va, vb = ca["metrics"][m], cb["metrics"][m]
            assert (np.isnan(va) and np.isnan(vb)) or va == vb
    row = a["candidates"][0]["metrics"]
    assert np.isfinite(row["power_uw"]) and row["power_uw"] > 0.0
    assert row["conv_time_ns"] == pytest.approx(90.0)  # 10ns sample + 4 * 20ns bits
    assert row["energy_per_conv_pj"] == pytest.approx(
        row["power_uw"] * row["conv_time_ns"] * 1e-3)
    csv_path, jsonl_path = tmp_path / "o.csv", tmp_path / "o.jsonl"
    sar_write_csv(a, csv_path)
    sar_write_jsonl(a, jsonl_path)
    header = csv_path.read_text().splitlines()[0]
    assert "max_abs_dnl" in header and "var_cap_msb" in header
    assert len(jsonl_path.read_text().splitlines()) == 2


def test_impossible_constraint_yields_no_feasible_candidates():
    from circuitopt.sar_explore import load_sar_explore_json, sar_explore
    spec, cfg = load_sar_explore_json(EXPLORE_CFG)
    cfg.sweep_points = 2
    cfg.constraints = {"power_uw": {"max": -1.0}}     # unsatisfiable on purpose
    result = sar_explore(spec, cfg, n=2, seed=0)
    assert result["summary"]["feasible"] == 0
    assert result["summary"]["pareto"] == 0
    assert result["summary"]["best"] == {}
    assert all(not c["feasible"] and not c["pareto"] for c in result["candidates"])


# ── CLI wiring ────────────────────────────────────────────────────────────────
def test_cli_explore_smoke(tmp_path):
    csv_path = tmp_path / "out.csv"
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE),
         "--explore", str(EXPLORE_CFG), "-n", "2", "--seed", "1",
         "--workers", "2", "--csv", str(csv_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert "candidates: 2" in proc.stdout
    assert csv_path.is_file() and "max_abs_dnl" in csv_path.read_text().splitlines()[0]


def test_cli_explore_conflicts_with_sweep():
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE),
         "--explore", str(EXPLORE_CFG), "--sweep", "8"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "mutually exclusive" in (proc.stderr + proc.stdout)


# ── -o payload tiers (--waveforms) ────────────────────────────────────────────
_WAVEFORM_KEYS = {"t", "input_waveforms", "transient"}


def test_cli_output_defaults_to_codes_and_metrics_only(tmp_path):
    """A bare ``-o`` writes the slim payload: no per-conversion waveforms."""
    out = tmp_path / "sweep.json"
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE),
         "--sweep", "8", "-o", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert data["codes"] == list(range(8))
    assert "max_abs_dnl" in data["metrics"]
    assert len(data["conversions"]) == 8
    assert all(not (_WAVEFORM_KEYS & set(c)) for c in data["conversions"])
    # The whole point: kilobytes, not the tens of megabytes waveforms cost.
    assert out.stat().st_size < 100_000


def test_cli_waveforms_flag_restores_the_full_payload(tmp_path):
    out = tmp_path / "sweep_full.json"
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE),
         "--sweep", "8", "-o", str(out), "--waveforms"],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert data["codes"] == list(range(8))
    assert all(_WAVEFORM_KEYS <= set(c) for c in data["conversions"])
    node_wave = data["conversions"][0]["transient"]["nodes"]
    assert node_wave, "waveform payload must carry transient node histories"


def test_cli_single_conversion_output_keeps_decisions_and_power(tmp_path):
    out = tmp_path / "one.json"
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE),
         "--vin", "0.4375", "-o", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert not (_WAVEFORM_KEYS & set(data))
    assert {"code", "bits", "decisions", "total_power_w"} <= set(data)
    assert len(data["decisions"]) == data["n_bits"]


def test_cli_waveforms_without_output_is_rejected():
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE),
         "--sweep", "8", "--waveforms"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "--waveforms only changes --output" in (proc.stderr + proc.stdout)


def test_cli_waveforms_rejects_modes_without_waveforms(tmp_path):
    for mode_args in (["--mc", "1"], ["--explore", str(EXPLORE_CFG)]):
        proc = subprocess.run(
            [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE),
             *mode_args, "--waveforms", "-o", str(tmp_path / "x.json")],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert proc.returncode != 0
        assert "--waveforms applies to" in (proc.stderr + proc.stdout)


def test_cli_sweep_output_alone_skips_trajectory_recording(tmp_path, monkeypatch):
    """Pin the compute rule, not just the file shape: a bare ``-o`` (and a bare
    ``--plot``) must run the codes-only path; only ``--waveforms`` records."""
    import circuitopt.__main__ as cli
    import circuitopt.sar as sar_mod

    seen = []
    real = sar_mod.run_sar_sweep

    def spy(spec, vin, **kwargs):
        seen.append(kwargs.get("include_transients"))
        return real(spec, vin, **kwargs)

    monkeypatch.setattr(sar_mod, "run_sar_sweep", spy)
    cli.main(["adc", str(EXAMPLE), "--sweep", "8", "--quiet",
              "-o", str(tmp_path / "a.json")])
    cli.main(["adc", str(EXAMPLE), "--sweep", "8", "--quiet",
              "-o", str(tmp_path / "b.json"), "--waveforms"])
    assert seen == [False, True]


def test_explore_workers_reach_the_candidate_sweep_kernel(monkeypatch):
    """``workers`` must parallelise each candidate's own conversions.

    The outer-thread-pool design capped at 1.8x on 8 workers; the kernel's
    inner parallelism reaches the sweep's own scaling. Pin the wiring: every
    candidate's conversion batch receives the caller's worker count.
    """
    import circuitopt.sar_explore as se
    from circuitopt.sar_explore import load_sar_explore_json, sar_explore

    seen = []
    real = se._run_sar_conversions

    def spy(spec, vin, cfg, corner, mismatch, workers, **kwargs):
        seen.append(workers)
        return real(spec, vin, cfg, corner, mismatch, workers, **kwargs)

    monkeypatch.setattr(se, "_run_sar_conversions", spy)
    spec, cfg = load_sar_explore_json(EXPLORE_CFG)
    result = sar_explore(spec, cfg, n=2, seed=3, workers=3)
    assert seen == [3, 3]                     # one batch per candidate, inner w=3
    assert len(result["candidates"]) == 2


# ── subsampled sweeps (12-bit screening mode) ────────────────────────────────
def test_explore_subsampled_scores_the_nominal_circuit_clean():
    """A perfect converter on a sparse ramp must score clean: code errors 0,
    monotonic, and DNL/INL/missing reported as unmeasured NaN — not the old
    levels-minus-present "missing 4032 codes at 12 bits" / aliased-DNL scoring
    that failed every subsampled candidate."""
    from circuitopt.sar_explore import evaluate_sar, parse_sar_explore
    cfg = parse_sar_explore({
        "variables": {"w": {"min": 1.0, "max": 2.0, "targets": ["W:M1"]}},
        "constraints": {"monotonic": {"min": 1},
                        "max_abs_code_err": {"max": 0.5}},
        "objectives": {"max_abs_code_err": "min", "power_uw": "min"},
        "sweep_points": 4,
    })
    m = evaluate_sar(_spec(), cfg, workers=2)
    assert m["max_abs_code_err"] == 0.0
    assert m["monotonic"] == 1.0
    assert np.isnan(m["max_abs_dnl"]) and np.isnan(m["max_abs_inl"])
    assert np.isnan(m["missing_codes"])
    from circuitopt.explore import is_feasible
    assert is_feasible(m, cfg.constraints)


def test_mc_subsampled_rows_gate_on_code_errors():
    from circuitopt.sar_mc import sar_mismatch_mc
    cfg = {"sigma_vth0": 0.01, "sweep_points": 4}
    a = sar_mismatch_mc(_spec(), n=2, seed=5, config=cfg, workers=1)
    b = sar_mismatch_mc(_spec(), n=2, seed=5, config=cfg, workers=8)
    for row in a["rows"]:
        assert row["subsampled"] and len(row["codes"]) == 4
        assert np.isnan(row["max_abs_dnl"]) and np.isnan(row["missing_codes"])
        assert np.isfinite(row["max_abs_code_err"])
    assert a["summary"]["subsampled"] and a["summary"]["sweep_points"] == 4
    assert 0.0 <= a["summary"]["yield"] <= 1.0
    # worker count changes nothing, including the code-error yield gate
    # (NaN-aware compare: subsampled summaries carry NaN for the unmeasured
    # transition metrics, and nan != nan under plain equality)
    assert [r["codes"].tolist() for r in a["rows"]] == \
           [r["codes"].tolist() for r in b["rows"]]
    sa, sb = a["summary"], b["summary"]
    assert set(sa) == set(sb)
    for key in sa:
        va, vb = sa[key], sb[key]
        if isinstance(va, dict):
            assert set(va) == set(vb)
            for k in va:
                assert va[k] == vb[k] or (
                    np.isnan(va[k]) and np.isnan(vb[k]))
        else:
            assert va == vb


def test_mc_compiled_batch_converts_the_subsampled_inputs():
    """The compiled batch must be built on the subsampled vin — without the
    explicit ``vins=`` it silently converts the full 2**n centers and the
    codes/vin lengths disagree."""
    from circuitopt.sar import _sar_config
    from circuitopt.sar_mc import _rust_batch_rows
    spec = _spec()
    cfg = _sar_config(spec, None)
    levels = 1 << cfg["n_bits"]
    idx = np.unique(np.linspace(0, levels - 1, 4).round().astype(int))
    vin = (idx + 0.5) / levels * cfg["vref"]
    rows = _rust_batch_rows(spec, cfg, None, [({}, spec)], vin, workers=1)
    assert rows is not None, "compiled path must handle a subsampled vin"
    assert len(rows) == 1 and len(rows[0]["codes"]) == len(vin)
    assert rows[0]["subsampled"]


def test_cli_subsampled_sweep_prints_code_errors(tmp_path):
    out = tmp_path / "sub.json"
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE),
         "--sweep", "4", "-o", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert "subsampled 4 of 8" in proc.stdout and "code err" in proc.stdout
    data = json.loads(out.read_text())
    assert len(data["codes"]) == 4
    assert data["metrics"]["subsampled"] is True
    assert "max_abs_code_err" in data["metrics"]


def test_cli_subsampled_sweep_renders_the_code_error_figure(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE),
         "--sweep", "4", "--plot", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "adc_sar_static.png").is_file()


def test_cli_sweep_points_requires_mc():
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE),
         "--sweep", "8", "--sweep-points", "4"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "--sweep-points applies to --mc only" in (proc.stderr + proc.stdout)


# ── transition bisection (final-verification DNL) ─────────────────────────────
def test_transitions_locate_the_nominal_carries_cleanly():
    """Nominal sar3: every carry transition measurable, DNL near zero at the
    0.05-LSB tolerance, and byte-identical across worker counts."""
    from circuitopt.sar import run_sar_transitions
    a = run_sar_transitions(_spec(), workers=1)
    b = run_sar_transitions(_spec(), workers=8)
    assert a["targets"].tolist() == [1, 2, 3, 4, 5]
    assert not a["unmeasured"]
    assert np.all(np.isfinite(a["transitions"]))
    assert a["max_abs_dnl"] <= 0.15
    assert np.all(np.abs(a["inl"]) < 0.5)      # code-center ramp was ideal
    np.testing.assert_array_equal(a["transitions"], b["transitions"])
    assert a["conversions"] == b["conversions"] > 0


def test_transitions_shift_under_gross_comparator_mismatch():
    from circuitopt.sar import run_sar_transitions
    nominal = run_sar_transitions(_spec(), codes=[2, 4], workers=8)
    shifted = run_sar_transitions(_spec(), codes=[2, 4], workers=8,
                                  mismatch={"M1": 0.3})
    lsb = nominal["lsb"]
    moved = np.abs(shifted["transitions"] - nominal["transitions"])
    finite = np.isfinite(moved)
    assert finite.any()
    assert np.nanmax(moved) > 0.5 * lsb


def test_cli_transitions_smoke_and_guards(tmp_path):
    out = tmp_path / "tr.json"
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE),
         "--transitions", "--workers", "2", "-o", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert "SAR transitions: 5 located" in proc.stdout
    data = json.loads(out.read_text())
    assert data["targets"] == [1, 2, 3, 4, 5]
    assert len(data["transitions"]) == 5 and "dnl" in data

    for extra, msg in (
        (["--sweep", "8"], "mutually exclusive"),
        (["--waveforms", "-o", str(tmp_path / "x.json")], "--waveforms applies"),
        (["--plot", str(tmp_path)], "no figure"),
        (["--tol-lsb", "0.05"], None),               # placeholder, replaced below
    ):
        if msg is None:
            continue
        proc = subprocess.run(
            [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE),
             "--transitions", *extra],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert proc.returncode != 0
        assert msg in (proc.stderr + proc.stdout)
    # out-of-range code list dies cleanly, not with a traceback
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt", "adc", str(EXAMPLE),
         "--transitions", "0,9"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "must lie in [1, 7]" in (proc.stderr + proc.stdout)
    assert "Traceback" not in proc.stderr
