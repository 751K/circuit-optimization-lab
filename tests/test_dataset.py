"""Tests for the surrogate dataset builder (``circuitopt.dataset``).

These pin the dataset *contract* a downstream surrogate depends on: a stable,
versioned schema, provenance in the manifest, failure-retaining rows (every sample
kept — a DC failure is a label, not a dropped point), JSON-valid output (no NaN
tokens), and determinism for a fixed ``(config, seed)``. The underlying physics is
:mod:`circuitopt.explore`'s, already tested; here we test only the dataset layer.
"""
import json
import os

import numpy as np
import pytest

import circuitopt.dataset as ds
from circuitopt.dataset import (_finite_or_none, _resolve_corner, _row, _topology_hash,
                          build_dataset, load_dataset_config, to_arrays)
from circuitopt.ngspice_char import ngspice_binary

CONFIG = "examples/single_stage.json"

# ── FreePDK45 availability gate (mirrors tests/test_freepdk45.py) ────────────
_PDK_ROOT = os.environ.get("PDK_ROOT", "/Volumes/MacoutDsik/pdk")
_FP45_INC = os.path.join(_PDK_ROOT, "freepdk45", "models_nom", "NMOS_VTG.inc")
_HAVE_FP45 = os.path.exists(_FP45_INC) and ngspice_binary() is not None
_requires_fp45 = pytest.mark.skipif(
    not _HAVE_FP45, reason="FreePDK45 cards / ngspice not present")


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
    assert ds._labels_for(("pac",)) == ds.PAC_LABELS
    assert ds._labels_for(("pnoise",)) == ds.PNOISE_LABELS
    assert (ds._labels_for(("ac_noise", "transient", "pss"))
            == ds.AC_NOISE_LABELS + ds.TRANSIENT_LABELS + ds.PSS_LABELS)
    assert (ds._labels_for(("pss", "pac", "pnoise"))
            == ds.PSS_LABELS + ds.PAC_LABELS + ds.PNOISE_LABELS)
    with pytest.raises(ValueError):
        ds._labels_for(("bogus",))


def test_transient_group_end_to_end(tmp_path):
    config = _write_tran_config(tmp_path)
    dataset = ds.run_from_config(config, n=3, seed=0, label_groups=("ac_noise", "transient"))
    m = dataset["manifest"]
    assert m["schema_version"] == ds.SCHEMA_VERSION
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


# ── pac / pnoise label groups (schema 1.2) ───────────────────────────────────

def test_pac_features_math():
    res = {"gains": np.array([2.0, 1.9, 0.2]), "Av_dc_dB": 6.02, "bw_Hz": 55.0}
    f = ds._pac_features(res)
    assert f["pac_gain"] == 2.0                       # |H| at the lowest analysis freq
    assert f["pac_gain_dB"] == 6.02 and f["pac_bw_Hz"] == 55.0
    empty = ds._pac_features({})                      # missing keys -> NaN, never raises
    assert set(empty) == set(ds.PAC_LABELS) and all(np.isnan(v) for v in empty.values())


def test_pnoise_features_math():
    f = ds._pnoise_features({"out_uV_band": 12.5, "irn_uV_band": 4.5})
    assert f["pnoise_out_uV"] == 12.5 and f["pnoise_irn_uV"] == 4.5
    empty = ds._pnoise_features({})
    assert set(empty) == set(ds.PNOISE_LABELS) and all(np.isnan(v) for v in empty.values())


def test_pac_group_requires_analyses_block(tmp_path):
    config = _write_tran_config(tmp_path)             # periodic, but no analyses.pac
    with pytest.raises(ValueError, match="analyses.pac"):
        ds.run_from_config(config, n=1, label_groups=("pac",))
    with pytest.raises(ValueError, match="analyses.pnoise"):
        ds.run_from_config(config, n=1, label_groups=("pnoise",))


def test_pac_group_requires_drive_for_multi_input(tmp_path):
    c = json.loads(json.dumps(_TRAN_CONFIG))
    c["periodic"]["inputs"]["vaux"] = {"type": "constant", "value": 0.0}
    c["analyses"] = {"pac": {"freqs": {"start": 1, "stop": 100, "num": 3}}}
    p = tmp_path / "multi.json"
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="input_drive"):
        ds.run_from_config(str(p), n=1, label_groups=("pac",))


def test_pac_pnoise_groups_end_to_end(tmp_path):
    # single periodic input -> input_drive defaults to {vin: 1.0}; the analyses
    # blocks carry the (HB) PAC/PNoise settings exactly as `run -a pss,pac,pnoise`.
    c = json.loads(json.dumps(_TRAN_CONFIG))
    c["analyses"] = {
        "pac": {"freqs": {"start": 1.0, "stop": 100.0, "num": 5, "scale": "log"}},
        "pnoise": {"freqs": {"start": 1.0, "stop": 100.0, "num": 5, "scale": "log"},
                   "band": [1.0, 100.0]},
    }
    p = tmp_path / "pp.json"
    p.write_text(json.dumps(c))
    dataset = ds.run_from_config(str(p), n=2, seed=0,
                                 label_groups=("pss", "pac", "pnoise"))
    m = dataset["manifest"]
    assert m["label_groups"] == ["pss", "pac", "pnoise"]
    assert m["labels"] == list(ds.PSS_LABELS + ds.PAC_LABELS + ds.PNOISE_LABELS)
    labeled = 0
    for r in dataset["rows"]:
        assert set(r["metrics"]) == set(m["labels"])
        if r["status"]["dc_converged"] and r["metrics"]["pss_converged"] == 1.0:
            assert r["metrics"]["pac_gain"] is not None and r["metrics"]["pac_gain"] > 0.0
            assert (r["metrics"]["pnoise_irn_uV"] is not None
                    and r["metrics"]["pnoise_irn_uV"] > 0.0)
            labeled += 1
    assert labeled > 0                                # guards against silent all-fail


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
    assert ds._target_kind("pvt0") == "corner"
    assert ds._target_kind("pbeta0") == "corner"


def test_corner_shift_from_sampled_vars():
    from circuitopt.explore import Variable
    cvars = [Variable("pvt0", -1, 1), Variable("pbeta0", -1, 1)]
    assert ds._corner_shift(cvars, {"pvt0": 0.1, "pbeta0": -0.2}, None) == \
        {"pvt0": 0.1, "pbeta0": -0.2}


def test_corner_axis_samples_process_shift():
    from circuitopt.explore import Variable
    data, topo, sizes, bias, nf, cfg = load_dataset_config(CONFIG)   # single_stage (PMOS)
    cfg.variables += [Variable("pvt0", -0.2, 0.2), Variable("pbeta0", -0.5, 0.5)]
    d = build_dataset(topo, sizes, bias, nf, cfg, n=5, seed=0,
                      config_dict=data, config_path=CONFIG)
    m = d["manifest"]
    assert m["corner"] == "sampled"                     # dataset spans PVT, not one point
    assert m["variables"]["pvt0"]["kind"] == "corner"
    for r in d["rows"]:
        assert -0.2 <= r["design"]["pvt0"] <= 0.2       # sampled shift recorded in X
        assert -0.5 <= r["design"]["pbeta0"] <= 0.5


def test_patch_structural_applies_and_isolates():
    from circuitopt.explore import Variable
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
    from circuitopt.circuit_loader import circuit_from_dict
    from circuitopt.explore import Variable
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
    from circuitopt.explore import ExploreConfig, Variable
    cfg = ExploreConfig([Variable("CL.C", 1e-12, 5e-10)], {}, {"area": "min"},
                        (0.05, 100.0), np.logspace(-2, 3, 11))
    with pytest.raises(ValueError, match="config"):
        build_dataset(None, {}, {}, None, cfg, n=2, config_dict=None)


def test_freqs_override():
    # overriding the AC grid pushes bw's ceiling up (avoids bw_Hz clipping)
    d = ds.run_from_config(CONFIG, n=3, seed=0,
                           freqs=np.logspace(-2, 4, 51))          # 0.01 Hz – 10 kHz
    assert d["manifest"]["freqs"] == {"n_points": 51, "f_min": 0.01, "f_max": 10000.0}


def test_dataset_cli_end_to_end(tmp_path):
    import argparse

    from circuitopt.__main__ import _add_dataset_parser, _cmd_dataset

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


# ── silicon model pass-through into the ac_noise label group ─────────────────
@_requires_fp45
def test_ac_noise_group_carries_silicon_models():
    """The ac_noise label group threads the FreePDK45 per-device model map through
    the binding, so its labels are silicon-magnitude — not the OTFT PDK a dropped
    ``model_types`` would silently fall back to (the Phase-B bug class, at the
    dataset layer). A FreePDK45 45nm OTA gives tens of dB of gain and a µm²-scale
    area; the OTFT default would give a near-zero gain and a padded g_area in the
    millions."""
    dataset = ds.run_from_config("examples/freepdk45_fd_ota.json", n=2, seed=0)
    m = dataset["manifest"]
    assert all(str(v).startswith("freepdk45") for v in m["models"].values())
    _, Y, _, ln, dc, _ = to_arrays(dataset)
    peak = Y[dc, ln.index("gain_peak_dB")]
    area = Y[dc, ln.index("area")]
    assert peak.size                                    # at least one DC-converged row
    assert np.all(peak > 20.0)                          # real silicon gain, not ~0 (OTFT revert)
    assert np.all(area < 1e3)                           # µm²-scale silicon area, not the
    #                                                     OTFT padded g_area (~1e6)
