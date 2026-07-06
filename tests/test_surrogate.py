"""Smoke test for the baseline metric surrogate (``circuitopt.surrogate``).

Gated on the optional scikit-learn dependency. Trains a tiny surrogate on a
synthetic monotone relationship, checks it learns (high R²) and predicts the right
shape, auto-logs a wide-range label, and round-trips through save/load — the
persisted-as-dict path that must load without the ``__main__.Surrogate`` pickling
hazard.
"""
import numpy as np
import pytest

pytest.importorskip("sklearn")
import circuitopt.surrogate as sg


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 3))
    y_lin = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]      # smooth, linear-ish
    y_wide = np.exp(4 * X[:, 0] + 1.0)                 # strictly positive, >1 decade
    return X, np.column_stack([y_lin, y_wide]), ["a", "b", "c"], ["lin", "wide"]


def test_auto_log_labels_picks_wide_range():
    Y = np.array([[1.0, 10.0], [2.0, 1000.0], [1.5, 50.0]])   # col b spans 100× (>1 decade)
    assert sg.auto_log_labels(Y, ["a", "b"]) == ("b",)


def test_train_predict_and_score():
    X, Y, var_names, label_names = _synthetic()
    model = sg.train(X, Y, var_names, label_names, max_iter=60)
    assert model.var_names == var_names and model.label_names == label_names
    assert "wide" in model.log_labels                 # wide-range label fitted in log-space
    Yp = model.predict(X)
    assert Yp.shape == (len(X), 2)
    s = sg.score(Y, Yp, label_names)
    assert s["lin"]["r2"] > 0.9 and s["wide"]["r2"] > 0.9   # actually learns the mapping


def test_predict_single_design_vector():
    X, Y, var_names, label_names = _synthetic()
    model = sg.train(X, Y, var_names, label_names, max_iter=40)
    one = model.predict(X[0])                          # a 1-D vector, not a batch
    assert one.shape == (1, 2)


def test_filter_rows_region_of_interest():
    X = np.arange(12, dtype=float).reshape(6, 2)
    Y = np.array([[10.0, 1.0], [25.0, 5.0], [8.0, 90.0],
                  [30.0, 2.0], [15.0, 50.0], [22.0, 200.0]])
    labels = ["gain", "irn"]
    Xf, Yf = sg.filter_rows(X, Y, labels, {"gain": (15.0, None), "irn": (None, 80.0)})
    # keep rows with gain>=15 AND irn<=80: rows 1 (25,5), 3 (30,2), 4 (15,50)
    assert Yf.shape == (3, 2)
    assert set(Yf[:, 0]) == {25.0, 30.0, 15.0}
    assert Xf.shape == (3, 2)


def test_load_multi_corner_appends_shift_features(tmp_path):
    import json

    from circuitopt.corners import CORNERS

    def _write(path, corner, n=3):
        X = np.arange(n * 2, dtype=float).reshape(n, 2)
        np.savez(path, X=X, Y=np.ones((n, 1)),
                 var_names=np.array(["a", "b"], dtype=object),
                 label_names=np.array(["g"], dtype=object),
                 dc_converged=np.ones(n, bool), metrics_finite=np.ones(n, bool),
                 manifest=json.dumps({"corner": corner}))

    p_typ, p_slow = str(tmp_path / "typ.npz"), str(tmp_path / "slow.npz")
    _write(p_typ, "typical"); _write(p_slow, "slow")
    X, Y, var_names, label_names = sg.load_multi_corner([p_typ, p_slow])
    assert var_names == ["a", "b", "pvt0", "pbeta0"]      # process shift appended
    assert X.shape == (6, 4) and Y.shape == (6, 1)
    assert np.allclose(X[:3, 2], 0.0)                     # typical pvt0 = 0
    assert np.allclose(X[3:, 2], CORNERS["slow"]["pvt0"])  # slow pvt0 shift


def test_save_load_round_trip(tmp_path):
    X, Y, var_names, label_names = _synthetic()
    model = sg.train(X, Y, var_names, label_names, max_iter=40)
    path = str(tmp_path / "m.pkl")
    sg.save(model, path)
    loaded = sg.load(path)                             # reconstructed from the plain dict
    assert loaded.label_names == label_names and loaded.log_labels == model.log_labels
    assert np.allclose(loaded.predict(X), model.predict(X))
