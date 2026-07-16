"""ADC design-space exploration over the native transistor-level SAR workflow.

Scale is kept tiny: two candidates over a four-point subsampled code-center
sweep, so the end-to-end pass stays below the larger SAR regressions.
"""
from pathlib import Path

import numpy as np
import pytest

from circuitopt.toolchain import pdk_root


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "freepdk45_sar3.json"
CONFIG = ROOT / "examples" / "freepdk45_sar3_explore.json"
_HAVE = (Path(pdk_root()) / "freepdk45" / "models_nom" / "NMOS_VTG.inc").is_file()
pytestmark = pytest.mark.skipif(not _HAVE, reason="FreePDK45 cards not present")


def _spec():
    from circuitopt.circuit_loader import load_circuit_json
    return load_circuit_json(EXAMPLE)


def _small_cfg():
    from circuitopt.sar_explore import parse_sar_explore
    return parse_sar_explore({
        "sweep_points": 4,
        "variables": {
            "in_pair_W": {"min": 0.9, "max": 1.3, "round": 2,
                          "targets": ["W:M1", "W:M2"]},
            "unit_cap": {"min": 9e-15, "max": 1.1e-14, "targets": ["C:C0P", "C:C0N"]},
        },
        "constraints": {"monotonic": {"min": 1}},
        "objectives": {"max_abs_dnl": "min", "power_uw": "min"},
    })


def test_c_target_changes_evaluated_capacitor():
    """A ``C:`` target rebinds exactly the named caps on a copy; the spec is untouched."""
    from circuitopt.sar_explore import apply_sar_variables
    spec = _spec()
    cfg = _small_cfg()
    cand = apply_sar_variables(cfg.variables, {"in_pair_W": 1.2, "unit_cap": 9.5e-15}, spec)
    caps = {c[0]: c[3] for c in cand.topology.capacitors}
    assert caps["C0P"] == pytest.approx(9.5e-15)
    assert caps["C0N"] == pytest.approx(9.5e-15)
    assert caps["C2P"] == pytest.approx(4e-14)          # untargeted cap unchanged
    assert cand.sizes["M1"] == (1.2, 0.1) and cand.sizes["M2"] == (1.2, 0.1)
    # The loaded spec is never mutated.
    orig = {c[0]: c[3] for c in spec.topology.capacitors}
    assert orig["C0P"] == 1e-14 and spec.sizes["M1"] == (1.0, 0.1)


def test_unknown_cap_target_rejected():
    from circuitopt.sar_explore import Variable, apply_sar_variables
    spec = _spec()
    var = Variable("bad", 1e-14, 2e-14, targets=["C:NOPE"])
    with pytest.raises(ValueError):
        apply_sar_variables([var], {"bad": 1.5e-14}, spec)


def test_sar_explore_end_to_end(tmp_path):
    """Two candidates run through sample->evaluate->constrain->pareto with finite metrics."""
    from circuitopt.sar_explore import (METRICS, sar_explore, sar_write_csv,
                                        sar_write_jsonl)
    spec = _spec()
    cfg = _small_cfg()
    res = sar_explore(spec, cfg, n=2, seed=0, workers=2)
    assert res["summary"]["n"] == 2
    assert len(res["candidates"]) == 2
    for c in res["candidates"]:
        assert c["converged"]
        m = c["metrics"]
        for key in ("power_uw", "conv_time_ns", "energy_per_conv_pj",
                    "max_abs_dnl", "missing_codes", "monotonic"):
            assert np.isfinite(m[key]), key
        assert m["conv_time_ns"] > 0.0 and m["power_uw"] > 0.0
    # Pareto points are a subset of the feasible set.
    assert res["summary"]["pareto"] <= res["summary"]["feasible"]

    csv_path = tmp_path / "out.csv"
    jsonl_path = tmp_path / "out.jsonl"
    sar_write_csv(res, csv_path)
    sar_write_jsonl(res, jsonl_path)
    header = csv_path.read_text().splitlines()[0]
    assert header.startswith("idx,")
    for key in METRICS:
        assert key in header
    assert len(jsonl_path.read_text().splitlines()) == 2


def test_sar_explore_workers_match_serial():
    """The candidate set is order-preserving and identical across worker counts."""
    from circuitopt.sar_explore import sar_explore
    spec = _spec()
    cfg = _small_cfg()
    serial = sar_explore(spec, cfg, n=2, seed=1, workers=1)
    parallel = sar_explore(spec, cfg, n=2, seed=1, workers=2)
    for a, b in zip(serial["candidates"], parallel["candidates"]):
        assert a["idx"] == b["idx"]
        assert a["metrics"]["max_abs_dnl"] == b["metrics"]["max_abs_dnl"]
        assert a["metrics"]["power_uw"] == pytest.approx(
            b["metrics"]["power_uw"], rel=1e-12, abs=1e-12)


def test_example_config_loads_and_matches_positional():
    """The shipped example config loads; a mismatched positional circuit is rejected."""
    from circuitopt.sar_explore import load_sar_explore_json
    spec, cfg = load_sar_explore_json(CONFIG, circuit_path=str(EXAMPLE))
    assert spec.adc["n_bits"] == 3
    assert len(cfg.variables) == 5
    assert cfg.sweep_points == 8
    with pytest.raises(ValueError):
        load_sar_explore_json(CONFIG, circuit_path="examples/afe_explore.json")
