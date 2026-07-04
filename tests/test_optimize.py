"""Smoke test for the surrogate-accelerated optimization loop (``core.optimize``).

Gated on scikit-learn. Trains a tiny surrogate on a small single_stage dataset, then
runs the screen → Pareto → verify loop and checks the report structure (the loop
wiring: surrogate screen, constrained Pareto select, solver verification of the
shortlist). Accuracy is the surrogate's concern, tested elsewhere.
"""
import numpy as np
import pytest

pytest.importorskip("sklearn")

import core.dataset as ds
import core.optimize as opt
import core.surrogate as sg

CONFIG = "examples/single_stage.json"          # has explore constraints + objectives


def _train_tiny(tmp_path, n=80):
    ds.run_from_config(CONFIG, n=n, seed=0, out=str(tmp_path / "ds"))
    X, Y, var_names, label_names, _ = sg.load_xy(str(tmp_path / "ds.npz"))
    path = str(tmp_path / "m.pkl")
    sg.save(sg.train(X, Y, var_names, label_names, max_iter=40), path)
    return path, label_names


def test_optimize_screen_pareto_verify(tmp_path):
    model_path, label_names = _train_tiny(tmp_path)
    rep = opt.optimize(CONFIG, model_path, n_screen=300, top_k=3, seed=1,
                       freqs=np.logspace(-2, 3, 21))
    assert rep["n_screen"] == 300
    assert 0 <= rep["surrogate_pareto"] <= rep["surrogate_feasible"] <= 300
    assert rep["screen_seconds"] >= 0.0
    assert len(rep["verified"]) <= 3
    for e in rep["verified"]:
        assert set(e["surrogate"]) == set(label_names)          # every label predicted
        assert e["solver"] is None or set(e["solver"]) == set(label_names)
        assert e["solver_feasible"] in (True, False, None)
    assert isinstance(opt._format_report(rep), str)             # renders without error


def test_optimize_no_verify(tmp_path):
    model_path, _ = _train_tiny(tmp_path, n=60)
    rep = opt.optimize(CONFIG, model_path, n_screen=200, top_k=5, verify=False,
                       freqs=np.logspace(-2, 3, 21))
    assert rep["verified"] == [] and "picks" in rep and len(rep["picks"]) <= 5
