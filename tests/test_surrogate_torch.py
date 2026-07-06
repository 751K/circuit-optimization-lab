"""Smoke test for the differentiable PyTorch surrogate (``circuitopt.surrogate_torch``).

Gated on a *working* torch: skips where torch is absent or its numpy ABI is broken
(e.g. the daily env's numpy-2 vs torch-built-for-numpy-1 mismatch); runs in the mps
env. Trains a tiny MLP on a synthetic monotone map, checks it learns + predicts,
gradient-optimizes a design under a box + objective, and round-trips save/load.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:                                    # daily's torch imports but can't touch numpy → skip
    torch.tensor(np.zeros(1, dtype=np.float32))
except Exception:                       # pragma: no cover
    pytest.skip("torch/numpy interop broken in this env", allow_module_level=True)

import circuitopt.surrogate_torch as st


def _synthetic(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 3))
    y_lin = 3 * X[:, 0] + 2.0 - X[:, 1]           # smooth, monotone
    y_wide = np.exp(3 * X[:, 2] + 1.0)            # strictly positive, >1 decade → log
    return X, np.column_stack([y_lin, y_wide]), ["a", "b", "c"], ["lin", "wide"]


def test_train_predict_learns():
    X, Y, var_names, label_names = _synthetic()
    m = st.train(X, Y, var_names, label_names, epochs=150, hidden=(32, 32))
    assert "wide" in m.log_labels                 # wide-range label auto-logged
    Yp = m.predict(X)
    assert Yp.shape == (len(X), 2)
    err = np.median(np.abs(Yp - Y) / np.abs(Y), axis=0)
    assert err[0] < 0.05 and err[1] < 0.05        # actually fits the map


def test_optimize_design_in_box_and_improves():
    X, Y, var_names, label_names = _synthetic()
    m = st.train(X, Y, var_names, label_names, epochs=150, hidden=(32, 32))
    lo, hi = np.zeros(3), np.ones(3)
    obj = st.penalty_objective({}, {"lin": "min"}, {"lin": 1.0})   # minimize 'lin'
    x, metrics, hist = st.optimize_design(m, 0.5 * np.ones(3), obj, (lo, hi), steps=200)
    assert np.all(x >= lo - 1e-3) and np.all(x <= hi + 1e-3)       # stayed in the box
    center_lin = float(m.predict(0.5 * np.ones(3))[0][0])
    assert metrics["lin"] < center_lin            # gradient actually reduced the objective


def test_save_load_round_trip(tmp_path):
    X, Y, var_names, label_names = _synthetic()
    m = st.train(X, Y, var_names, label_names, epochs=60, hidden=(16, 16))
    path = str(tmp_path / "m.pt")
    st.save(m, path)
    m2 = st.load(path)
    assert m2.label_names == label_names and m2.log_labels == m.log_labels
    assert np.allclose(m2.predict(X), m.predict(X), atol=1e-4)     # same weights
