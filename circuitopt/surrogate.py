"""Baseline metric surrogate — a fast ML approximation of a labeled dataset.

Fits one gradient-boosted-tree regressor per label (scikit-learn
``HistGradientBoostingRegressor``) on a dataset built by :mod:`circuitopt.dataset`, and
scores held-out accuracy against the teacher solver. This is the first ML layer of
the surrogate roadmap: the calibrated solvers stay the
source of truth; the surrogate only *screens* candidates orders of magnitude faster
during design refinement, with anything promising handed back to the solver.

scikit-learn is an **optional** dependency (like numba / matplotlib / pyarrow),
imported lazily with a clear message if missing::

    pip install -r requirements-ml.txt

Usage::

    python -m circuitopt.surrogate train results/datasets/afe/afe_typical_train.npz \\
        --test results/datasets/afe/afe_typical_test.npz --out results/models/afe.pkl
    python -m circuitopt.surrogate predict results/models/afe.pkl --x 65000,70,3500,30.5,10.0
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:                          # optional dependency
        raise ImportError("the surrogate needs scikit-learn; "
                          "pip install -r requirements-ml.txt") from exc


@dataclass
class Surrogate:
    """One fitted regressor per label + the input/output column names."""
    regressors: dict          # label -> fitted estimator
    var_names: list
    label_names: list
    metadata: dict            # provenance: solver commit, corner, n_train, model, ...
    log_labels: tuple = ()    # labels fitted in log-space (predict exponentiates them)

    def predict(self, X):
        X = np.asarray(X, float).reshape(-1, len(self.var_names))
        cols = []
        for lab in self.label_names:
            y = self.regressors[lab].predict(X)
            cols.append(np.exp(y) if lab in self.log_labels else y)
        return np.column_stack(cols)


# ── data ────────────────────────────────────────────────────────────────────
def load_xy(npz_path, *, finite_only=True):
    """Load ``(X, Y, var_names, label_names, manifest)`` from a dataset ``.npz``.

    ``finite_only`` keeps only rows with all labels present (``metrics_finite``) —
    the trainable set. Failure / partial rows stay in the dataset but are not fed
    to the regressor."""
    import json
    d = np.load(npz_path, allow_pickle=True)
    X, Y = np.asarray(d["X"], float), np.asarray(d["Y"], float)
    var_names = [str(v) for v in d["var_names"]]
    label_names = [str(v) for v in d["label_names"]]
    manifest = json.loads(str(d["manifest"])) if "manifest" in d else {}
    if finite_only:
        mask = np.asarray(d["metrics_finite"], bool)
        X, Y = X[mask], Y[mask]
    return X, Y, var_names, label_names, manifest


def load_multi_corner(npz_paths, *, finite_only=True):
    """Stack several per-corner datasets into one training set for a PVT-spanning model.

    Each corner's **physical process shift** ``(pvt0, pbeta0)`` (from
    :data:`circuitopt.corners.CORNERS`, keyed by the dataset manifest's ``corner``) is
    appended to ``X`` as two design columns — a continuous, physically-meaningful
    encoding (not a one-hot label), so the model can *interpolate* across corners.
    Returns ``(X_aug, Y, var_names + ['pvt0','pbeta0'], label_names)``."""
    from circuitopt.device_factory import CORNERS
    Xs, Ys, var_names, label_names = [], [], None, None
    for path in npz_paths:
        X, Y, var_names, label_names, manifest = load_xy(path, finite_only=finite_only)
        shift = CORNERS.get(manifest.get("corner", "typical"), {"pvt0": 0.0, "pbeta0": 0.0})
        feat = np.tile([shift["pvt0"], shift["pbeta0"]], (len(X), 1))
        Xs.append(np.column_stack([X, feat]))
        Ys.append(Y)
    return (np.vstack(Xs), np.vstack(Ys),
            list(var_names) + ["pvt0", "pbeta0"], list(label_names))


def filter_rows(X, Y, label_names, bounds):
    """Keep only rows whose labels fall within ``{label: (lo, hi)}`` bounds (``None``
    = open). Use to train/score the metric surrogate on a *region of interest*
    (near-spec designs): wildly-out-of-spec outliers (collapsed gain, milli-volt
    noise) dominate the error tail and get rejected by a cheap feasibility screen
    anyway, so precise regression there is wasted capacity."""
    Y = np.asarray(Y, float)
    idx = {lab: j for j, lab in enumerate(label_names)}
    keep = np.ones(len(Y), bool)
    for lab, (lo, hi) in bounds.items():
        col = Y[:, idx[lab]]
        if lo is not None:
            keep &= col >= lo
        if hi is not None:
            keep &= col <= hi
    return np.asarray(X, float)[keep], Y[keep]


# ── train / score ────────────────────────────────────────────────────────────
def auto_log_labels(Y, label_names, *, ratio=10.0):
    """Labels worth fitting in log-space: strictly positive and spanning >1 decade.

    Input-referred noise runs from ~30 µV (feasible) to >1 mV (infeasible designs);
    a squared-error fit on that raw range is dominated by the huge-noise tail and
    fits the feasible region poorly. Log-space makes the loss weigh *relative*
    error uniformly across the range."""
    Y = np.asarray(Y, float)
    picks = []
    for j, lab in enumerate(label_names):
        col = Y[:, j]
        if np.all(col > 0.0) and float(col.max() / col.min()) > ratio:
            picks.append(lab)
    return tuple(picks)


def train(X, Y, var_names, label_names, *, max_iter=400, learning_rate=0.1,
          log_labels=None, metadata=None, **params):
    """Fit one ``HistGradientBoostingRegressor`` per label. Returns a :class:`Surrogate`.

    Wide-range positive labels are fit in log-space (``log_labels``; auto-detected
    when ``None``) — the model predicts ``log(y)`` and :meth:`Surrogate.predict`
    exponentiates, giving uniform *relative* accuracy across skewed targets."""
    _require_sklearn()
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if log_labels is None:
        log_labels = auto_log_labels(Y, label_names)
    log_set = set(log_labels)
    regressors = {}
    for j, lab in enumerate(label_names):
        target = np.log(Y[:, j]) if lab in log_set else Y[:, j]
        reg = HistGradientBoostingRegressor(max_iter=max_iter,
                                            learning_rate=learning_rate, **params)
        reg.fit(X, target)
        regressors[lab] = reg
    meta = {"model": "HistGradientBoostingRegressor",
            "max_iter": max_iter, "learning_rate": learning_rate,
            "n_train": int(X.shape[0]), "log_labels": list(log_labels)}
    meta.update(metadata or {})
    return Surrogate(regressors, list(var_names), list(label_names), meta, tuple(log_labels))


def score(Ytrue, Ypred, label_names):
    """Per-label held-out accuracy: median / P95 relative error (%), R², MAE."""
    from sklearn.metrics import r2_score
    Ytrue, Ypred = np.asarray(Ytrue, float), np.asarray(Ypred, float)
    out = {}
    for j, lab in enumerate(label_names):
        t, p = Ytrue[:, j], Ypred[:, j]
        rel = np.abs(p - t) / np.maximum(np.abs(t), 1e-30)
        out[lab] = {
            "median_rel_pct": float(np.median(rel) * 100.0),
            "p95_rel_pct": float(np.percentile(rel, 95) * 100.0),
            "r2": float(r2_score(t, p)),
            "mae": float(np.mean(np.abs(p - t))),
        }
    return out


# ── persistence ──────────────────────────────────────────────────────────────
def save(surrogate, path):
    # Persist a plain dict (sklearn estimators + lists) rather than the Surrogate
    # instance: the wrapper class would pickle as ``__main__.Surrogate`` when trained
    # via ``python -m circuitopt.surrogate`` and then fail to unpickle from any other entry
    # point. The estimators' own classes live in sklearn, so they load anywhere.
    _require_sklearn()
    import os

    import joblib
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    joblib.dump({"regressors": surrogate.regressors, "var_names": surrogate.var_names,
                 "label_names": surrogate.label_names, "metadata": surrogate.metadata,
                 "log_labels": tuple(surrogate.log_labels)}, path)


def load(path):
    _require_sklearn()
    import joblib
    d = joblib.load(path)
    return Surrogate(d["regressors"], d["var_names"], d["label_names"],
                     d["metadata"], tuple(d.get("log_labels", ())))


# ── CLI ─────────────────────────────────────────────────────────────────────
def _format_scores(scores):
    lines = [f"{'label':16s} {'median%':>9s} {'p95%':>8s} {'R2':>8s}"
             "   (target: median<1%, p95<5%)"]
    for lab, s in scores.items():
        ok = "ok" if (s["median_rel_pct"] < 1.0 and s["p95_rel_pct"] < 5.0) else "  "
        lines.append(f"{lab:16s} {s['median_rel_pct']:9.3f} {s['p95_rel_pct']:8.2f} "
                     f"{s['r2']:8.4f}  {ok}")
    return "\n".join(lines)


def _parse_filter(spec, label_names):
    """``"gain_dB:0:60,irn_uV::100"`` → ``{"gain_dB": (0, 60), "irn_uV": (None, 100)}``.

    Each clause is ``label:lo:hi``; an empty bound is open. Restricts training to a
    region of interest (see :func:`filter_rows`) — e.g. drop railed/collapsed designs
    whose extreme labels would dominate the fit yet get screened out anyway."""
    bounds = {}
    for clause in spec.split(","):
        clause = clause.strip()
        if not clause:
            continue
        parts = clause.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--filter clause {clause!r} must be label:lo:hi (bounds may be empty)")
        lab, lo, hi = parts
        if lab not in label_names:
            raise SystemExit(f"--filter label {lab!r} not in dataset labels {label_names}")
        bounds[lab] = (float(lo) if lo.strip() else None, float(hi) if hi.strip() else None)
    return bounds


def _cmd_train(args):
    Xtr, Ytr, var_names, label_names, manifest = load_xy(args.train_npz)
    n_all = Xtr.shape[0]
    if args.filter:
        Xtr, Ytr = filter_rows(Xtr, Ytr, label_names, _parse_filter(args.filter, label_names))
        print(f"filtered to region of interest [{args.filter}]: {Xtr.shape[0]}/{n_all} samples")
    meta = {"train_npz": args.train_npz, "filter": args.filter,
            "solver_commit": (manifest.get("solver") or {}).get("commit"),
            "corner": manifest.get("corner"), "topology_hash": manifest.get("topology_hash")}
    model = train(Xtr, Ytr, var_names, label_names, max_iter=args.max_iter, metadata=meta)
    print(f"trained {len(label_names)} label regressors "
          f"(HGBR max_iter={args.max_iter}) on {Xtr.shape[0]} samples"
          + (f"; log-space: {list(model.log_labels)}" if model.log_labels else ""))
    if args.test:
        Xte, Yte, _, _, _ = load_xy(args.test)
        if args.filter:                         # score the same region of interest
            Xte, Yte = filter_rows(Xte, Yte, label_names, _parse_filter(args.filter, label_names))
        print(f"held-out test: {Xte.shape[0]} samples from {args.test}")
        print(_format_scores(score(Yte, model.predict(Xte), label_names)))
    if args.out:
        save(model, args.out)
        print(f"saved surrogate -> {args.out}")
    return model


def _cmd_predict(args):
    model = load(args.model)
    x = np.array([float(v) for v in args.x.split(",")])
    yhat = model.predict(x)[0]
    for lab, val in zip(model.label_names, yhat):
        print(f"  {lab:16s} {val:.6g}")
    return yhat


def main(argv=None):
    p = argparse.ArgumentParser(description="Baseline metric surrogate (train / predict).")
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="Fit a surrogate on a dataset .npz and score a test set")
    tr.add_argument("train_npz", help="training dataset .npz (from `circuitopt dataset`)")
    tr.add_argument("--test", default=None, help="held-out test dataset .npz")
    tr.add_argument("--out", default=None, help="save the fitted surrogate here (joblib)")
    tr.add_argument("--max-iter", type=int, default=400, help="HGBR boosting iterations")
    tr.add_argument("--filter", default=None,
                    help="train/score on a region of interest: 'label:lo:hi[,...]' "
                         "(empty bound = open), e.g. 'gain_dB:0:60' to drop railed designs")
    pr = sub.add_parser("predict", help="Predict labels for one design vector")
    pr.add_argument("model", help="saved surrogate (joblib)")
    pr.add_argument("--x", required=True, help="comma-separated design vector (in var order)")
    args = p.parse_args(argv)
    try:
        return _cmd_train(args) if args.cmd == "train" else _cmd_predict(args)
    except ImportError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
