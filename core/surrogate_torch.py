"""Differentiable metric surrogate (PyTorch MLP) — gradient-based design optimization.

The GBT surrogate (:mod:`core.surrogate`) screens by brute force; a *differentiable*
surrogate lets you optimize the design vector **directly** by following gradients of
an objective through the model — a few hundred steps instead of 100k random samples.
This is the "differentiable optimization loop" of the roadmap (``docs/futureplan.md``
§7): a small MLP over standardized inputs/outputs (wide-range labels in log-space),
trained on a ``core.dataset`` ``.npz``, then :func:`optimize_design` does projected
gradient descent on the design under soft constraint penalties.

PyTorch is an **optional** dependency (lazy-imported, like sklearn). On Apple Silicon
the MPS device is picked automatically. The solvers stay the source of truth — a
gradient-optimized design is still verified on the real solver (:mod:`core.optimize`).

Env note: torch and the scipy-based solvers can live in different conda envs; this
module only needs torch + the dataset ``.npz`` (no solver), so it trains wherever
torch works.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from .surrogate import auto_log_labels, load_xy


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:                          # optional dependency
        raise ImportError("the differentiable surrogate needs pytorch; "
                          "pip install torch (use the MPS build on Apple Silicon)") from exc


def _device(pref=None):
    import torch
    if pref:
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _standardize_fit(A):
    """Column mean / std (std floored so constant columns don't divide by zero)."""
    A = np.asarray(A, float)
    mean = A.mean(axis=0)
    std = A.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def _build_net(d_in, d_out, hidden):
    import torch.nn as nn
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.SiLU()]
        prev = h
    layers += [nn.Linear(prev, d_out)]
    return nn.Sequential(*layers)


@dataclass
class TorchSurrogate:
    """An MLP over standardized design → (log-)metrics, plus its normalization.

    ``predict`` is numpy-in/out; ``forward_real`` is the differentiable path
    (real-unit design tensor → real-unit metric tensor) that :func:`optimize_design`
    backprops through. Wide-range labels (``log_labels``) are learned in log-space
    and exponentiated on output."""
    var_names: list
    label_names: list
    log_labels: tuple
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray            # in transformed (log for log_labels) space
    y_std: np.ndarray
    hidden: tuple
    state: dict                   # torch state_dict (cpu tensors)
    metadata: dict = field(default_factory=dict)
    _net: object = field(default=None, repr=False, compare=False)

    def _module(self, device):
        if self._net is None:
            net = _build_net(len(self.var_names), len(self.label_names), self.hidden)
            net.load_state_dict(self.state)
            net.eval()
            self._net = net
        return self._net.to(device)

    def _log_mask(self, device):
        import torch
        return torch.tensor([lab in self.log_labels for lab in self.label_names],
                            device=device)

    def forward_real(self, x, device=None):
        """Differentiable design(real) → metrics(real). ``x`` is a ``(n, d)`` tensor."""
        import torch
        device = device or x.device
        xm = torch.as_tensor(self.x_mean, dtype=torch.float32, device=device)
        xs = torch.as_tensor(self.x_std, dtype=torch.float32, device=device)
        ym = torch.as_tensor(self.y_mean, dtype=torch.float32, device=device)
        ys = torch.as_tensor(self.y_std, dtype=torch.float32, device=device)
        out = self._module(device)((x - xm) / xs)
        y_t = out * ys + ym                              # transformed units
        # exp only for log labels; clamp the exponent so out-of-domain designs can't
        # overflow to inf/NaN (which would poison the optimizer's gradients).
        return torch.where(self._log_mask(device),
                           torch.exp(torch.clamp(y_t, max=30.0)), y_t)

    def predict(self, X):
        import torch
        X = np.asarray(X, float).reshape(-1, len(self.var_names))
        with torch.no_grad():
            y = self.forward_real(torch.tensor(X, dtype=torch.float32, device="cpu"),
                                  device="cpu")
        return y.cpu().numpy()


def train(X, Y, var_names, label_names, *, log_labels=None, hidden=(128, 128),
          epochs=400, lr=1e-3, batch=256, val_frac=0.1, device=None, seed=0,
          verbose=False, metadata=None):
    """Train the MLP surrogate. Returns a :class:`TorchSurrogate` (best-val weights)."""
    _require_torch()
    import torch
    dev = _device(device)
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if log_labels is None:
        log_labels = auto_log_labels(Y, label_names)
    logm = np.array([lab in set(log_labels) for lab in label_names])
    Yt = Y.copy()
    Yt[:, logm] = np.log(Yt[:, logm])

    x_mean, x_std = _standardize_fit(X)
    y_mean, y_std = _standardize_fit(Yt)
    Xs = ((X - x_mean) / x_std).astype(np.float32)
    Ys = ((Yt - y_mean) / y_std).astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    nval = max(1, int(len(X) * val_frac))
    vi, ti = idx[:nval], idx[nval:]
    to = lambda a: torch.tensor(a, device=dev)          # noqa: E731
    Xtr, Ytr, Xva, Yva = to(Xs[ti]), to(Ys[ti]), to(Xs[vi]), to(Ys[vi])

    net = _build_net(X.shape[1], Y.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for b in range(0, len(Xtr), batch):
            j = perm[b:b + batch]
            opt.zero_grad()
            loss_fn(net(Xtr[j]), Ytr[j]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(Xva), Yva).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if verbose and ep % 50 == 0:
            print(f"  epoch {ep:4d}  val_mse={vl:.5f}")
    meta = {"model": "torch-mlp", "hidden": list(hidden), "epochs": epochs,
            "device": str(dev), "n_train": int(len(ti)), "best_val_mse": best_val}
    meta.update(metadata or {})
    return TorchSurrogate(list(var_names), list(label_names), tuple(log_labels),
                          x_mean, x_std, y_mean, y_std, tuple(hidden),
                          best_state or net.state_dict(), meta)


def optimize_design(surrogate, x0, objective, bounds, *, steps=400, lr=0.02,
                    device=None):
    """Projected-gradient-descend a design vector through the surrogate.

    ``objective(metrics)`` maps ``{label: tensor}`` (real units) → a scalar loss
    (lower is better); ``bounds=(lo, hi)`` box-constrains the design (clamped each
    step). Returns ``(best_x, best_metrics, history)`` — the design at the lowest
    loss and its surrogate metrics. Differentiable ⇒ this is a handful of steps, not
    a 100k-sample screen."""
    _require_torch()
    import torch
    dev = _device(device)
    lo = torch.as_tensor(bounds[0], dtype=torch.float32, device=dev)
    hi = torch.as_tensor(bounds[1], dtype=torch.float32, device=dev)
    span = (hi - lo).clamp_min(1e-12)
    # Optimize in normalized [0,1] design space: uniform scale across dims (W~1e4 vs
    # VCM~30) so one lr works, and clamping to [0,1] keeps the design in-domain.
    x0t = torch.as_tensor(np.asarray(x0, float), dtype=torch.float32, device=dev)
    u = (((x0t - lo) / span).clamp(0.0, 1.0)).requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    best_loss, best_x, history = float("inf"), None, []
    for _ in range(steps):
        opt.zero_grad()
        x = lo + u.clamp(0.0, 1.0) * span                # real-unit design (in box)
        metrics = surrogate.forward_real(x.unsqueeze(0), device=dev)[0]
        loss = objective({lab: metrics[j] for j, lab in enumerate(surrogate.label_names)})
        loss.backward()
        opt.step()
        with torch.no_grad():
            u.clamp_(0.0, 1.0)
        lv = float(loss.item())
        history.append(lv)
        if np.isfinite(lv) and lv < best_loss:
            best_loss = lv
            best_x = (lo + u.detach().clamp(0.0, 1.0) * span).cpu().numpy().copy()
    if best_x is None:                                    # never improved → return the endpoint
        best_x = (lo + u.detach().clamp(0.0, 1.0) * span).cpu().numpy()
    metrics = {lab: float(v) for lab, v in zip(surrogate.label_names,
                                               surrogate.predict(best_x)[0])}
    return best_x, metrics, history


def penalty_objective(constraints, objectives, scales, *, weight=50.0):
    """Build ``objective(metrics)`` = (scaled objectives) + soft constraint penalties.

    Minimizes each ``objectives`` metric (``max`` negated) divided by a **fixed**
    per-label ``scales`` reference (so the term keeps a real gradient — normalizing
    by the current value would cancel the magnitude), and adds ``weight × relu``
    hinge penalties for each ``constraints`` ``min``/``max`` violation (relative), so
    the optimizer descends the objectives while being pushed into the feasible region."""
    import torch

    def obj(metrics):
        loss = metrics[next(iter(metrics))].new_zeros(())     # scalar on the right device
        for m, sense in objectives.items():
            v = metrics[m]
            loss = loss + (v if sense == "min" else -v) / (abs(scales.get(m, 1.0)) + 1e-9)
        for m, bound in constraints.items():
            v = metrics[m]
            if "min" in bound:
                loss = loss + weight * torch.relu((bound["min"] - v) / abs(bound["min"] + 1e-9))
            if "max" in bound:
                loss = loss + weight * torch.relu((v - bound["max"]) / abs(bound["max"] + 1e-9))
        return loss
    return obj


def save(surrogate, path):
    _require_torch()
    import os

    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save({"var_names": surrogate.var_names, "label_names": surrogate.label_names,
                "log_labels": list(surrogate.log_labels), "x_mean": surrogate.x_mean,
                "x_std": surrogate.x_std, "y_mean": surrogate.y_mean,
                "y_std": surrogate.y_std, "hidden": list(surrogate.hidden),
                "state": surrogate.state, "metadata": surrogate.metadata}, path)


def load(path):
    _require_torch()
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    return TorchSurrogate(d["var_names"], d["label_names"], tuple(d["log_labels"]),
                          d["x_mean"], d["x_std"], d["y_mean"], d["y_std"],
                          tuple(d["hidden"]), d["state"], d.get("metadata", {}))


# ── CLI ─────────────────────────────────────────────────────────────────────
def _cmd_train(args):
    from .surrogate import score
    X, Y, var_names, label_names, _ = load_xy(args.train_npz)
    model = train(X, Y, var_names, label_names, epochs=args.epochs, hidden=(128, 128),
                  verbose=not args.quiet)
    print(f"trained torch-mlp on {X.shape[0]} samples "
          f"(device={model.metadata['device']}, hidden={model.hidden}, "
          f"log={list(model.log_labels)})")
    if args.test:
        Xte, Yte, _, _, _ = load_xy(args.test)
        s = score(Yte, model.predict(Xte), label_names)
        for lab in label_names:
            print(f"  {lab:14s} median={s[lab]['median_rel_pct']:.3f}%  "
                  f"p95={s[lab]['p95_rel_pct']:.2f}%  R2={s[lab]['r2']:.4f}")
    if args.out:
        save(model, args.out)
        print(f"saved -> {args.out}")
    return model


def _cmd_optimize(args):
    from .dataset import load_dataset_config
    from .explore import apply_variables, evaluate
    _, topo, base_sizes, base_bias, nf, cfg = load_dataset_config(args.config)
    model = load(args.model)
    lo = np.array([v.lo for v in cfg.variables], float)
    hi = np.array([v.hi for v in cfg.variables], float)
    x0 = 0.5 * (lo + hi)                                   # start at the box centre
    ref = {lab: float(v) for lab, v in zip(model.label_names, model.predict(x0)[0])}
    obj = penalty_objective(cfg.constraints, cfg.objectives, ref)
    x, metrics, history = optimize_design(model, x0, obj, (lo, hi), steps=args.steps)
    print(f"gradient-optimized a design in {args.steps} steps "
          f"(loss {history[0]:.3f} → {min(history):.3f})")
    for m in cfg.objectives:                               # did the objectives actually improve?
        print(f"  objective {m}: start {ref[m]:.4g} → optimized {metrics[m]:.4g}")
    for name, val in zip(model.var_names, x):
        print(f"  {name:14s} = {val:.4g}")
    print("  surrogate metrics: " + "  ".join(f"{m}={metrics[m]:.4g}" for m in model.label_names))
    if args.verify:                                        # confirm on the real solver
        var_values = {v.name: float(x[i]) for i, v in enumerate(cfg.variables)}
        sizes, bias, cand_nf = apply_variables(cfg.variables, var_values,
                                               base_sizes, base_bias, base_nf=nf)
        cfg.freqs = np.logspace(-2, 4, 101)
        true = evaluate(topo, sizes, bias, cand_nf, cfg.freqs, cfg.band, require_noise=True)
        if true:
            print("  solver check:     " + "  ".join(
                f"{m}={true.get(m):.4g}" for m in model.label_names if true.get(m) is not None))
    return x, metrics


def main(argv=None):
    p = argparse.ArgumentParser(description="Differentiable (PyTorch) metric surrogate.")
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="Train an MLP surrogate on a dataset .npz")
    tr.add_argument("train_npz")
    tr.add_argument("--test", default=None)
    tr.add_argument("--out", default=None)
    tr.add_argument("--epochs", type=int, default=400)
    tr.add_argument("--quiet", action="store_true")
    op = sub.add_parser("optimize", help="Gradient-optimize a design through the surrogate")
    op.add_argument("config", help="circuit JSON with an 'explore' block")
    op.add_argument("model", help="trained torch surrogate (.pt)")
    op.add_argument("--steps", type=int, default=400)
    op.add_argument("--verify", action="store_true", help="check the result on the solver")
    args = p.parse_args(argv)
    try:
        return _cmd_train(args) if args.cmd == "train" else _cmd_optimize(args)
    except ImportError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
