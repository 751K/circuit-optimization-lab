"""Tests for the surrogate dataset builder (``core.dataset``).

These pin the dataset *contract* a downstream surrogate depends on: a stable,
versioned schema, provenance in the manifest, failure-retaining rows (every sample
kept — a DC failure is a label, not a dropped point), JSON-valid output (no NaN
tokens), and determinism for a fixed ``(config, seed)``. The underlying physics is
:mod:`core.explore`'s, already tested; here we test only the dataset layer.
"""
import json

import numpy as np
import pytest

import core.dataset as ds
from core.dataset import (_finite_or_none, _resolve_corner, _row, _topology_hash,
                          build_dataset, load_dataset_config, to_arrays)

CONFIG = "examples/single_stage.json"


def _build(n=6, seed=0):
    data, topo, sizes, bias, nf, cfg = load_dataset_config(CONFIG)
    return build_dataset(topo, sizes, bias, nf, cfg, n=n, seed=seed,
                         config_dict=data, config_path=CONFIG)


def test_row_schema_and_all_samples_kept():
    dataset = _build(n=6)
    rows = dataset["rows"]
    assert len(rows) == 6                          # every sample retained, no filtering
    var_names = set(dataset["manifest"]["variables"])
    for i, r in enumerate(rows):
        assert r["idx"] == i
        assert set(r["design"]) == var_names
        assert set(r["metrics"]) == set(ds.LABELS)
        assert set(r["status"]) == {"dc_converged", "noise_evaluated", "metrics_finite"}
        json.dumps(r)                              # valid JSON (NaN/inf coerced to null)


def test_failed_candidate_kept_with_null_labels():
    r = _row(3, {"MPU.W": 1500.0, "MLD.W": 900.0, "VIN": 25.0}, None)  # DC failed
    assert r["status"]["dc_converged"] is False
    assert r["status"]["metrics_finite"] is False
    assert r["status"]["noise_evaluated"] is False
    assert all(r["metrics"][k] is None for k in ds.LABELS)
    assert r["design"]["MPU.W"] == 1500.0          # design inputs still recorded


def test_manifest_provenance():
    m = _build(n=4)["manifest"]
    assert m["schema_version"] == ds.SCHEMA_VERSION
    assert m["labels"] == list(ds.LABELS)
    assert m["corner"] == "typical"
    assert m["pdk"] == "at4000tg.pmos"
    assert set(m["solver"]) == {"commit", "dirty"}
    assert m["topology_hash"].startswith("sha1:")
    assert set(m["variables"]) == {"MPU.W", "MLD.W", "VIN"}
    assert m["sampling"] == {"n": 4, "seed": 0, "method": "lhs"}
    assert m["counts"]["total"] == 4
    assert 0 <= m["counts"]["metrics_finite"] <= m["counts"]["dc_converged"] <= 4


def test_deterministic_for_fixed_seed():
    a = _build(n=6, seed=5)["rows"]
    b = _build(n=6, seed=5)["rows"]
    assert a == b                                  # same (config, seed) ⇒ identical
    c = _build(n=6, seed=6)["rows"]
    assert a != c                                  # different seed ⇒ different samples


def test_to_arrays_shapes_and_masks():
    dataset = _build(n=8)
    X, Y, var_names, label_names, dc, fin = to_arrays(dataset)
    assert X.shape == (8, 3) and Y.shape == (8, len(ds.LABELS))
    assert var_names == list(dataset["manifest"]["variables"])
    assert label_names == list(ds.LABELS)
    assert np.isfinite(X).all()                    # design inputs always populated
    assert np.isfinite(Y[fin]).all()               # labeled rows carry finite labels
    assert dc.dtype == bool and fin.dtype == bool


def test_topology_hash_ignores_explore_block():
    data, *_ = load_dataset_config(CONFIG)
    h0 = _topology_hash(data)
    changed = json.loads(json.dumps(data))
    changed["explore"]["variables"]["MPU.W"]["max"] = 9999    # search range only
    assert _topology_hash(changed) == h0                      # same circuit ⇒ same hash
    changed2 = json.loads(json.dumps(data))
    changed2["devices"][0]["W"] = 12345                       # structural change
    assert _topology_hash(changed2) != h0


def test_finite_or_none():
    assert _finite_or_none(1.5) == 1.5
    assert _finite_or_none(None) is None
    assert _finite_or_none(float("nan")) is None
    assert _finite_or_none(float("inf")) is None


def test_resolve_corner():
    assert _resolve_corner(None) == (None, "typical")
    assert _resolve_corner("typical") == (None, "typical")
    shift, name = _resolve_corner("slow")
    assert name == "slow" and isinstance(shift, dict)
    with pytest.raises(ValueError):
        _resolve_corner("bogus")


def test_write_dataset_round_trip(tmp_path):
    dataset = _build(n=5)
    paths = ds.write_dataset(dataset, str(tmp_path / "ds"), npz=True)
    lines = (tmp_path / "ds.jsonl").read_text().splitlines()
    assert len(lines) == 5 and all(json.loads(ln)["idx"] == i for i, ln in enumerate(lines))
    manifest = json.loads((tmp_path / "ds.manifest.json").read_text())
    assert manifest["schema_version"] == ds.SCHEMA_VERSION
    npz = np.load(paths["npz"], allow_pickle=True)
    assert npz["X"].shape == (5, 3) and npz["Y"].shape == (5, len(ds.LABELS))
    assert json.loads(str(npz["manifest"]))["topology_hash"] == manifest["topology_hash"]


def test_parquet_round_trip_or_clear_error(tmp_path):
    dataset = _build(n=4)
    path = str(tmp_path / "d.parquet")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        with pytest.raises(ImportError, match="pyarrow"):
            ds.write_parquet(dataset, path)
        return
    ds.write_parquet(dataset, path)
    table = pq.read_table(path)
    assert table.num_rows == 4
    cols = table.column_names
    assert "gain_dB" in cols and "design_MPU.W" in cols and "dc_converged" in cols


# ── transient label group (schema 1.1) ──────────────────────────────────────

# A vsource-driven single stage with a periodic square drive: exercises the
# transient group. (Its AC labels are degenerate — a transient-driven input has no
# AC excitation — which is fine here: we only assert the transient features.)
_TRAN_CONFIG = {
    "name": "tran_test",
    "solved": ["IN", "OUT"],
    "rails": {"VDD": 40.0, "GND": 0.0},
    "devices": [
        {"name": "MPU", "drain": "OUT", "gate": "IN", "source": "VDD", "W": 2000, "L": 80},
        {"name": "MLD", "drain": "GND", "gate": "GND", "source": "OUT", "W": 1500, "L": 80},
    ],
    "vsources": [{"name": "V_IN", "p": "IN", "q": "GND", "value": "vin"}],
    "bias": {"VDD": 40.0},
    "outputs": ["OUT"],
    "load_caps": [["OUT", "GND", 2e-12]],
    "dc_guesses": [{"IN": 25.0, "OUT": 20.0}, {"IN": 25.0, "OUT": 5.0}],
    "periodic": {"frequency": 1000.0, "n_points": 101,
                 "inputs": {"vin": {"type": "square", "low": 24.5, "high": 25.5, "duty": 0.5}}},
    "explore": {"variables": {"MPU.W": {"min": 1500, "max": 4000, "int": True},
                              "MLD.W": {"min": 800, "max": 2000, "int": True}},
                "objectives": {"area": "min"}, "band": [0.05, 100.0],
                "freqs": {"start": -2, "stop": 3, "num": 21}},
}


def _write_tran_config(tmp_path):
    p = tmp_path / "tran.json"
    p.write_text(json.dumps(_TRAN_CONFIG))
    return str(p)


def test_transient_features_math():
    # ramp 0→3 over 3 s then hold: pp=3, slew=1 V/s, mean=1.8, final=3
    tr = {"t": np.array([0., 1., 2., 3., 4.]), "output": np.array([0., 1., 2., 3., 3.])}
    f = ds._transient_features(tr)
    assert f["out_pp"] == 3.0 and f["final_value"] == 3.0
    assert abs(f["slew_rate"] - 1.0) < 1e-12
    assert abs(f["out_mean"] - 1.8) < 1e-12
    assert abs(f["out_rms"] - np.sqrt((0 + 1 + 4 + 9 + 9) / 5)) < 1e-12
    bad = ds._transient_features({"t": np.array([0., 1.]), "output": np.array([np.nan, 1.])})
    assert set(bad) == set(ds.TRANSIENT_LABELS) and all(np.isnan(v) for v in bad.values())


def test_labels_for_groups():
    assert ds._labels_for(("ac_noise",)) == ds.AC_NOISE_LABELS
    assert ds._labels_for(("ac_noise", "transient")) == ds.AC_NOISE_LABELS + ds.TRANSIENT_LABELS
    assert ds._labels_for(("pss",)) == ds.PSS_LABELS
    assert (ds._labels_for(("ac_noise", "transient", "pss"))
            == ds.AC_NOISE_LABELS + ds.TRANSIENT_LABELS + ds.PSS_LABELS)
    with pytest.raises(ValueError):
        ds._labels_for(("bogus",))


def test_transient_group_end_to_end(tmp_path):
    config = _write_tran_config(tmp_path)
    dataset = ds.run_from_config(config, n=3, seed=0, label_groups=("ac_noise", "transient"))
    m = dataset["manifest"]
    assert m["schema_version"] == "1.1"
    assert m["label_groups"] == ["ac_noise", "transient"]
    assert set(ds.TRANSIENT_LABELS) <= set(m["labels"])
    for r in dataset["rows"]:
        assert set(r["metrics"]) == set(m["labels"])
        if r["status"]["dc_converged"]:                # transient ran ⇒ features present
            assert all(r["metrics"][k] is not None for k in ds.TRANSIENT_LABELS)


def test_transient_group_requires_periodic():
    data, topo, sizes, bias, nf, cfg = load_dataset_config(CONFIG)   # single_stage: no periodic
    with pytest.raises(ValueError, match="periodic"):
        build_dataset(topo, sizes, bias, nf, cfg, n=2, label_groups=("ac_noise", "transient"),
                      config_dict=data, config_path=CONFIG)


def test_pss_features_math():
    res = {"converged": True, "residual_norm": 1e-9, "shooting_iters": 3,
           "output": np.array([1.0, 3.0, 2.0, 1.0])}
    f = ds._pss_features(res)
    assert f["pss_converged"] == 1.0 and f["pss_residual"] == 1e-9 and f["pss_iters"] == 3.0
    assert f["pss_out_pp"] == 2.0 and abs(f["pss_out_mean"] - 1.75) < 1e-12
    diverged = ds._pss_features({"converged": False, "output": []})
    assert diverged["pss_converged"] == 0.0                  # a label, not a null
    assert np.isnan(diverged["pss_out_pp"]) and np.isnan(diverged["pss_residual"])


def test_pss_group_end_to_end(tmp_path):
    config = _write_tran_config(tmp_path)                    # periodic amp
    dataset = ds.run_from_config(config, n=3, seed=0, label_groups=("ac_noise", "pss"))
    m = dataset["manifest"]
    assert m["label_groups"] == ["ac_noise", "pss"]
    assert set(ds.PSS_LABELS) <= set(m["labels"])
    for r in dataset["rows"]:
        assert set(r["metrics"]) == set(m["labels"])
        if r["status"]["dc_converged"]:
            assert r["metrics"]["pss_converged"] in (0.0, 1.0)   # trust flag always set
            if r["metrics"]["pss_converged"] == 1.0:
                assert all(r["metrics"][k] is not None for k in ds.PSS_LABELS)


def test_pss_group_requires_periodic():
    data, topo, sizes, bias, nf, cfg = load_dataset_config(CONFIG)   # no periodic block
    with pytest.raises(ValueError, match="periodic"):
        build_dataset(topo, sizes, bias, nf, cfg, n=2, label_groups=("pss",),
                      config_dict=data, config_path=CONFIG)


# ── structural / stimulus design axes (Cap.C, periodic.frequency) ────────────

def _struct_config():
    c = json.loads(json.dumps(_TRAN_CONFIG))            # deep copy of the periodic amp
    c["capacitors"] = [{"name": "CL", "a": "OUT", "b": "GND", "C": 2e-12}]
    c["explore"]["variables"] = {
        "MPU.W": {"min": 1500, "max": 4000, "int": True},
        "CL.C": {"min": 1e-12, "max": 5e-10},
        "periodic.frequency": {"min": 500.0, "max": 2000.0},
    }
    return c


def test_target_kind_classifies_axes():
    assert ds._target_kind("MPU.W") == "size_bias"
    assert ds._target_kind("VCM") == "size_bias"
    assert ds._target_kind("CL.C") == "cap"
    assert ds._target_kind("periodic.frequency") == "clock"


def test_patch_structural_applies_and_isolates():
    from core.explore import Variable
    cfg = _struct_config()
    struct = [Variable("CL.C", 1e-12, 5e-10), Variable("periodic.frequency", 500, 2000)]
    patched = ds._patch_structural(cfg, struct,
                                   {"CL.C": 7.77e-9, "periodic.frequency": 1234.0})
    assert patched["capacitors"][0]["C"] == 7.77e-9
    assert patched["periodic"]["frequency"] == 1234.0
    assert cfg["capacitors"][0]["C"] == 2e-12            # original untouched (deep copy)
    with pytest.raises(ValueError, match="no named capacitor"):
        ds._patch_structural(cfg, [Variable("NOPE.C", 0, 1)], {"NOPE.C": 1e-12})


def test_cap_axis_flows_into_rebuilt_topology():
    from core.circuit_loader import circuit_from_dict
    from core.explore import Variable
    patched = ds._patch_structural(_struct_config(), [Variable("CL.C", 0, 1)],
                                   {"CL.C": 3.33e-9})
    caps = circuit_from_dict(patched).topology.capacitors
    assert any(abs(c[-1] - 3.33e-9) < 1e-18 for c in caps)   # value reached the topology


def test_build_dataset_structural_axes(tmp_path):
    config = tmp_path / "struct.json"
    config.write_text(json.dumps(_struct_config()))
    dataset = ds.run_from_config(str(config), n=4, seed=1,
                                 label_groups=("ac_noise", "transient"))
    m = dataset["manifest"]
    kinds = {k: v["kind"] for k, v in m["variables"].items()}
    assert kinds == {"MPU.W": "size_bias", "CL.C": "structural",
                     "periodic.frequency": "structural"}
    for r in dataset["rows"]:
        assert 1e-12 <= r["design"]["CL.C"] <= 5e-10
        assert 500.0 <= r["design"]["periodic.frequency"] <= 2000.0
        if r["status"]["dc_converged"]:
            assert all(r["metrics"][k] is not None for k in ds.TRANSIENT_LABELS)
    # deterministic under the structural (rebuild) path too
    again = ds.run_from_config(str(config), n=4, seed=1,
                               label_groups=("ac_noise", "transient"))
    assert dataset["rows"] == again["rows"]


def test_structural_axis_needs_config_dict():
    from core.explore import ExploreConfig, Variable
    cfg = ExploreConfig([Variable("CL.C", 1e-12, 5e-10)], {}, {"area": "min"},
                        (0.05, 100.0), np.logspace(-2, 3, 11))
    with pytest.raises(ValueError, match="config"):
        build_dataset(None, {}, {}, None, cfg, n=2, config_dict=None)


def test_dataset_cli_end_to_end(tmp_path):
    import argparse

    from core.__main__ import _add_dataset_parser, _cmd_dataset

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command")
    _add_dataset_parser(sub)
    out = tmp_path / "cli_ds"
    args = ap.parse_args(["dataset", CONFIG, "-n", "5", "--seed", "3",
                          "--out", str(out), "--quiet"])
    assert args.n == 5 and args.seed == 3 and args.corner == "typical"
    dataset = _cmd_dataset(args)
    assert dataset["manifest"]["counts"]["total"] == 5
    assert out.with_suffix(".jsonl").exists()
    assert (tmp_path / "cli_ds.manifest.json").exists()
    assert out.with_suffix(".npz").exists()
