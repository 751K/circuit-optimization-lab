import numpy as np

from core.explore import (
    Variable,
    apply_variables,
    explore,
    is_feasible,
    load_explore_json,
    pareto_front,
    sample,
    write_csv,
    write_jsonl,
)


def test_pareto_front_unit():
    # minimize both: (1,4) and (4,1) are non-dominated; (2,5) is dominated by (1,4).
    rows = [{"a": 1.0, "b": 4.0}, {"a": 4.0, "b": 1.0}, {"a": 2.0, "b": 5.0},
            {"a": 2.0, "b": 2.0}]
    front = set(pareto_front(rows, {"a": "min", "b": "min"}))
    assert front == {0, 1, 3}


def test_pareto_respects_max_sense():
    rows = [{"g": 25.0, "p": 100.0}, {"g": 20.0, "p": 200.0}, {"g": 15.0, "p": 50.0}]
    # maximize gain, minimize power: row1 is beaten by row0 on both axes (more gain,
    # less power) -> dominated. row0 (best gain) and row2 (cheapest power) survive.
    front = set(pareto_front(rows, {"g": "max", "p": "min"}))
    assert front == {0, 2}


def test_is_feasible_bounds():
    m = {"gain_dB": 22.0, "bw_Hz": 500.0, "irn_uV": 40.0, "power_uW": 1.0, "area": 1.0}
    assert is_feasible(m, {"gain_dB": {"min": 20}, "irn_uV": {"max": 44.5}})
    assert not is_feasible(m, {"gain_dB": {"min": 25}})
    assert not is_feasible({**m, "irn_uV": float("nan")}, {"irn_uV": {"max": 44.5}})


def test_sample_is_deterministic_and_in_range():
    v = [Variable("W", 100, 200, is_int=True), Variable("VCM", 1.0, 2.0, round_to=2)]
    s1 = sample(v, 16, seed=7, method="lhs")
    s2 = sample(v, 16, seed=7, method="lhs")
    assert s1 == s2
    assert len(s1) == 16
    for row in s1:
        assert 100 <= row["W"] <= 200 and float(row["W"]).is_integer()
        assert 1.0 <= row["VCM"] <= 2.0


def test_apply_variables_targets_and_bias():
    variables = [Variable("in_pair_W", 1, 2, targets=["M7.W", "M8.W"]),
                 Variable("in_pair_NF", 1, 2, targets=["M7.NF", "M8.NF"]),
                 Variable("VCM", 1, 2)]
    sizes, bias, nf = apply_variables(
        variables, {"in_pair_W": 50000.0, "in_pair_NF": 120.0, "VCM": 31.0},
        base_sizes={"M7": (1.0, 60.0), "M8": (1.0, 60.0)}, base_bias={"VCM": 30.0})
    assert sizes["M7"] == (50000.0, 60.0)
    assert sizes["M8"] == (50000.0, 60.0)
    assert bias["VCM"] == 31.0
    assert nf == {"M7": 120, "M8": 120}        # NF variable -> integer finger count


def test_single_stage_explore_end_to_end(tmp_path):
    topo, sizes, bias, nf, cfg = load_explore_json("examples/single_stage.json")
    results = explore(topo, sizes, bias, nf, cfg, n=12, seed=0, method="lhs")

    cands = results["candidates"]
    assert len(cands) == 12
    assert results["summary"]["converged"] >= 1

    for c in cands:
        if c["converged"]:
            for m in ("gain_dB", "bw_Hz", "irn_uV", "power_uW", "area"):
                assert np.isfinite(c["metrics"][m])

    feasible = [c for c in cands if c["feasible"]]
    pareto = [c for c in cands if c["pareto"]]
    # every Pareto point must itself be feasible, and counts must agree.
    assert all(c["feasible"] for c in pareto)
    assert len(pareto) == results["summary"]["pareto"]
    assert len(feasible) == results["summary"]["feasible"]
    assert len(pareto) <= len(feasible)

    csv_path = tmp_path / "out.csv"
    jsonl_path = tmp_path / "out.jsonl"
    write_csv(results, csv_path)
    write_jsonl(results, jsonl_path)
    header = csv_path.read_text().splitlines()[0]
    assert header.startswith("idx,")
    assert "gain_dB" in header and "feasible" in header and "pareto" in header
    assert len(jsonl_path.read_text().splitlines()) == 12


def test_afe_explore_finds_feasible():
    # The baseline ("final locked") design sits inside the search box and meets
    # spec, so a small LHS sweep must turn up at least one feasible candidate.
    topo, sizes, bias, nf, cfg = load_explore_json("examples/afe_explore.json")
    results = explore(topo, sizes, bias, nf, cfg, n=8, seed=0, method="lhs")
    assert results["summary"]["converged"] == 8
    assert results["summary"]["feasible"] >= 1
