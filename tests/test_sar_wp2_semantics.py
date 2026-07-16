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
    """Order preservation must hold when workers exceed the point count."""
    from circuitopt.sar import run_sar_sweep
    vin = np.array([0.1875, 0.4375, 0.6875])
    serial = run_sar_sweep(_spec(), vin)
    threaded = run_sar_sweep(_spec(), vin, workers=8)
    np.testing.assert_array_equal(serial["codes"], threaded["codes"])
    np.testing.assert_array_equal(serial["metrics"]["dnl"], threaded["metrics"]["dnl"])
    np.testing.assert_array_equal(serial["metrics"]["inl"], threaded["metrics"]["inl"])


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
