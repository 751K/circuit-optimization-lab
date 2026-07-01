"""Periodic steady-state solver based on transient shooting.

The solver treats the existing transient engine as the source of truth for the
nonlinear device model and time discretization.  It solves the shooting equation

    Phi_T(x0) - x0 = 0

where Phi_T is one-period backward-Euler integration over the supplied periodic
input waveforms.  The first implementation uses a finite-difference shooting
Jacobian, which is practical for the current small AFE/chopper topologies and
keeps all device/capacitance behavior identical to transient analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ac_mna import _stamp_adm, _stamp_mos_lti, _branch_incidence
from .ac_solver import ac_solve, _dev_nf, build_devices, get_ss_params
from .adaptive_config import resolve_adaptive_config
from .topology import AFE_TOPO
from .transient_solver import transient
from . import diagnostics


def _finite_component_max(a):
    arr = np.asarray(a)
    if arr.size == 0:
        return 0.0
    if np.iscomplexobj(arr):
        parts = (np.abs(arr.real), np.abs(arr.imag))
    else:
        parts = (np.abs(arr),)
    out = 0.0
    for part in parts:
        finite = part[np.isfinite(part)]
        if finite.size:
            out = max(out, float(np.max(finite)))
    return out


def _shooting_monodromy(tr, topo, sizes, nf, bias, inputs, node_inputs,
                        dev_inst, gmin=1e-12, integration_method="be"):
    """Analytic one-period monodromy d(x_end)/d(x0) from the orbit small signal.

    Replicates the transient's step linearization on the converged PSS
    trajectory.  For backward-Euler the per-step map is A_m = (G_m + C_m/h_m)^{-1}
    (C_m/h_m) and Phi = prod_m A_m.  For gear2/BDF2 the step is a 2-history
    recurrence, so the monodromy is built on the augmented state [x_m; x_{m-1}]:
    M_m = [[-B1, -B2], [I, 0]] with B1 = J^{-1}(a1 C/h), B2 = J^{-1}(a2 C/h),
    J = G + a0 C/h (step 1 is backward-Euler self-start); d(x_end)/d(x0) is the
    top n rows of (prod M_m)[A_1; I].  Either way the shooting Jacobian is Phi-I,
    built in one orbit pass instead of n_state finite-difference period runs.
    """
    t = np.asarray(tr["t"], float)
    nodes = tr["nodes"]
    inputs = tr.get("inputs", inputs)
    N = len(t)
    n = topo.n
    nbr = topo.n_branches                       # ideal voltage-source branch unknowns
    idx = topo.idx
    rails = topo.rail_values(bias)
    all_branch_sources = list(topo.vsources) + list(topo.vcvs) + list(topo.ccvs)
    Binc = _branch_incidence(all_branch_sources, idx, n) if nbr else None

    def term(node):
        return ("n", idx[node]) if node in idx else ("v", 0.0)

    def term_value(node, m):
        if node in idx:
            return nodes[node][m]
        if node in node_inputs:
            return inputs[node_inputs[node]][m]
        return rails[node]

    G0 = np.zeros((n, n)); C0 = np.zeros((n, n))
    rg = np.zeros(n); rc = np.zeros(n)
    for a, b, cap in topo.cap_list():
        _stamp_adm(C0, rc, term(a), term(b), cap)
    for _, a, b, R in topo.resistors:
        _stamp_adm(G0, rg, term(a), term(b), 1.0 / R)
    for k in range(n):
        G0[k, k] += gmin

    def _GC(m):
        G = G0.copy(); C = C0.copy()
        for name, d, g, s in topo.devices:
            Vs = term_value(s, m); Vd = term_value(d, m); Vg = term_value(g, m)
            p = get_ss_params(sizes[name][0], sizes[name][1], Vs, Vd, Vg,
                              nf=_dev_nf(nf, name), dev_inst=dev_inst[name])
            _stamp_mos_lti(G, C, rg, rc, term(d), term(g), term(s),
                           p["gm"], p["gds"], p["Cgs"], p["Cgd"])
        return G, C

    def _solve(J, B):
        try:
            return np.linalg.solve(J, B)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(J, B, rcond=None)[0]

    def _bsolve(Jnode, B):
        """Node-space step operator ``inv([[Jnode, Binc],[Binc^T, 0]])[:n, :n] @ B``.

        Ideal voltage sources add one branch-current unknown each. The branch current is
        algebraic (no capacitor -> zero row/col in C, so it never enters the C/h
        propagation), hence the exact node-voltage monodromy is the node block of the
        bordered per-step solve. With no sources (Binc is None) this is just _solve."""
        if Binc is None:
            return _solve(Jnode, B)
        Jaug = np.zeros((n + nbr, n + nbr))
        Jaug[:n, :n] = Jnode
        Jaug[:n, n:] = Binc
        Jaug[n:, :n] = Binc.T
        rhs = np.zeros((n + nbr, n))
        rhs[:n] = B
        return _solve(Jaug, rhs)[:n]

    gear2 = str(integration_method).lower() in ("gear2", "bdf2")
    if not gear2:
        phi = np.eye(n)
        for m in range(1, N):
            h = float(t[m] - t[m - 1])
            if h <= 0.0:
                continue
            G, C = _GC(m)
            Ch = C / h
            phi = _bsolve(G + Ch, Ch) @ phi
        return phi

    # gear2/BDF2: augmented 2n-state monodromy on [x_m; x_{m-1}]
    eye = np.eye(n)
    P = None                                    # 2n x n  (maps dx0 -> [dx_m; dx_{m-1}])
    h_prev = None
    for m in range(1, N):
        h = float(t[m] - t[m - 1])
        if h <= 0.0:
            continue
        G, C = _GC(m)
        Ch = C / h
        rho = (h / h_prev) if h_prev is not None else 0.0
        if P is None or rho > 2.0:              # BE self-start / large-ratio step
            A1 = _bsolve(G + Ch, Ch)            # a0=1, a1=-1, a2=0
            if P is None:
                P = np.vstack([A1, eye])
            else:
                P = np.vstack([A1 @ P[:n], P[:n]])
        else:
            a0 = (1.0 + 2.0 * rho) / (1.0 + rho)
            a1 = -(1.0 + rho)
            a2 = (rho * rho) / (1.0 + rho)
            J = G + a0 * Ch
            B1 = _bsolve(J, a1 * Ch)
            B2 = _bsolve(J, a2 * Ch)
            top = -(B1 @ P[:n]) - (B2 @ P[n:])
            P = np.vstack([top, P[:n]])
        h_prev = h
    return P[:n] if P is not None else np.eye(n)


def _make_period_grid(period, tgrid, n_points):
    period = float(period)
    if period <= 0.0:
        raise ValueError("period must be positive")
    if tgrid is None:
        n_points = max(2, int(n_points))
        return np.linspace(0.0, period, n_points)
    out = np.asarray(tgrid, float)
    if out.ndim != 1 or len(out) < 2:
        raise ValueError("tgrid must be a one-dimensional array with at least two points")
    if not np.all(np.diff(out) > 0.0):
        raise ValueError("tgrid must be strictly increasing")
    if not np.isclose(out[0], 0.0, rtol=0.0, atol=max(1e-18, period * 1e-14)):
        raise ValueError("PSS tgrid must start at 0")
    if not np.isclose(out[-1], period, rtol=1e-12, atol=max(1e-18, period * 1e-12)):
        raise ValueError("PSS tgrid must end at period")
    return out.copy()


def _prepare_inputs(inputs, tgrid, *, check_periodic, periodic_tol):
    if inputs is None:
        return {}
    n = len(tgrid)
    out = {}
    for key, value in inputs.items():
        arr = value(tgrid) if callable(value) else value
        arr = np.asarray(arr, float)
        if arr.ndim == 0:
            arr = np.full(n, float(arr))
        if len(arr) != n:
            raise ValueError(f"Input waveform {key!r} length {len(arr)} != len(tgrid) {n}")
        if check_periodic:
            scale = max(1.0, abs(float(arr[0])), abs(float(arr[-1])))
            if not np.isclose(arr[0], arr[-1], rtol=periodic_tol,
                              atol=periodic_tol * scale):
                raise ValueError(
                    f"Input waveform {key!r} is not periodic at the PSS boundary: "
                    f"{arr[0]} != {arr[-1]}"
                )
        out[key] = arr
    return out


def _resample_inputs(inputs, t_src, t_dst, period):
    out = {}
    t_src = np.asarray(t_src, float)
    t_dst = np.asarray(t_dst, float)
    for key, val in (inputs or {}).items():
        out[key] = np.interp(t_dst, t_src, np.asarray(val, float), period=float(period))
    return out


def _initial_vector(sizes, bias, topo, nf, V0, corner=None):
    if V0 is not None:
        if isinstance(V0, dict):
            default = topo.default_guess_value(bias)
            # guess_vector pads ideal-source branch currents (n_aug); the shooting state
            # is node voltages only -> slice to n.
            return np.asarray(topo.guess_vector(V0, default=default), float)[:topo.n]
        arr = np.asarray(V0, float)
        if arr.shape != (topo.n,):
            raise ValueError(f"V0 shape {arr.shape} does not match topology size {topo.n}")
        return arr.copy()

    try:
        ac = ac_solve(sizes, bias, np.array([1.0]), topo=topo, nf=nf,
                      corner=corner)
        if ac is not None and "dc_op" in ac:
            return np.asarray([ac["dc_op"][node] for node in topo.solved], float)
    except Exception as exc:
        diagnostics.note("pss.dc_seed_fail", exc)

    guesses = topo.dc_guess_vectors(bias)
    if not guesses:
        return np.full(topo.n, topo.default_guess_value(bias), dtype=float)
    return np.asarray(guesses[0], float)[:topo.n]   # node voltages only (drop branch pads)


def _end_vector(tran_result, topo):
    return np.asarray([tran_result["nodes"][node][-1] for node in topo.solved], float)


def _rail_clip(vector, topo, bias, margin):
    if margin is None:
        return vector
    rails = [v for v in topo.rail_values(bias).values() if isinstance(v, (int, float))]
    if not rails:
        return vector
    lo = min(rails) - float(margin)
    hi = max(rails) + float(margin)
    return np.clip(vector, lo, hi)


def _residual_score(norm, nfail):
    return float(norm) * (1.0 + 100.0 * max(0, int(nfail)))


def _physical_span(topo, bias, factor):
    """Return (lo, hi) physical bounds for node voltages: the rail range expanded
    by ``factor`` x its span on each side. A converged PSS orbit must stay inside;
    an orbit that leaves it is a numerical runaway, not a steady state. Returns
    None when the topology has no constant rails to anchor against."""
    rails = [v for v in topo.rail_values(bias).values()
             if isinstance(v, (int, float))]
    if not rails:
        return None
    lo, hi = min(rails), max(rails)
    span = max(hi - lo, 1.0)
    return lo - factor * span, hi + factor * span


def _within(vector, bounds, topo):
    """True if all node voltages (the first ``topo.n`` entries — branch currents
    are unbounded) lie within ``bounds``."""
    if bounds is None:
        return True
    lo, hi = bounds
    nodes = np.asarray(vector, float)[:topo.n]
    return bool(np.all(np.isfinite(nodes)) and np.all(nodes >= lo)
                and np.all(nodes <= hi))


def _dominant_multiplier(phi):
    """Largest |Floquet multiplier| of the node-space monodromy ``phi`` — the
    stiffness/stability diagnostic. |lambda|>~1 means the implemented one-period
    map is (near-)unstable at this orbit; tau>>T circuits sit near 1."""
    try:
        return float(np.max(np.abs(np.linalg.eigvals(np.asarray(phi, float)))))
    except Exception as exc:
        diagnostics.note("pss.dominant_multiplier_fail", exc)
        return float("nan")


@dataclass(frozen=True, slots=True)
class _Orbit:
    """A single shooting orbit / one-period run: ``x_end = Phi(x0)`` with
    ``residual = x_end - x0``.

    *The* representation of a period result everywhere in the solver — there is
    no competing tuple/dict layout. ``run_period``/``rerun_frozen`` build live
    orbits (array refs); ``_best_orbit`` builds retained snapshots (copied
    arrays) for best-so-far and stabilization-converged tracking.
    """
    tr: dict
    x0: np.ndarray
    x_end: np.ndarray
    residual: np.ndarray
    norm: float
    nfail: int

    @property
    def score(self) -> float:
        return _residual_score(self.norm, self.nfail)


class _PSSPeriodRunner:
    """Own one-period transient runs plus adaptive-grid freeze state.

    This isolates the interaction between PSS shooting/stabilization and the
    transient driver. Shooting logic can request period runs without knowing
    whether the grid is still adaptive or has been frozen.
    """

    def __init__(self, *, sizes, bias, period, topo, tgrid, inputs,
                 transient_kwargs, residual_tol, adaptive, adaptive_config):
        self.sizes = sizes
        self.bias = bias
        self.period = float(period)
        self.topo = topo
        self.tgrid = tgrid
        self.inputs = inputs
        self.transient_kwargs = dict(transient_kwargs)
        self.residual_tol = float(residual_tol)
        self.adaptive = bool(adaptive)
        self.adaptive_config = adaptive_config
        self.period_runs = 0
        self.adaptive_grid_frozen = False
        self.frozen_tgrid = None
        self.frozen_inputs = None

    def period_kwargs(self):
        if self.adaptive_grid_frozen:
            kw = dict(self.transient_kwargs)
            kw["adaptive"] = False
            kw["edge_mask"] = None
            return self.frozen_tgrid, self.frozen_inputs, kw
        return self.tgrid, self.inputs, self.transient_kwargs

    def maybe_freeze_grid(self, tr, norm, nfail):
        if (not self.adaptive or self.adaptive_grid_frozen or nfail != 0 or
                norm > float(self.adaptive_config.freeze_factor) * self.residual_tol):
            return
        tr_t = np.asarray(tr["t"], float)
        if tr_t.ndim != 1 or len(tr_t) < 2:
            return
        self.adaptive_grid_frozen = True
        self.frozen_tgrid = tr_t.copy()
        self.frozen_inputs = _resample_inputs(
            self.inputs, self.tgrid, self.frozen_tgrid, self.period)

    def run_period(self, x0, *, allow_freeze=True, profile=False):
        self.period_runs += 1
        run_tgrid, run_inputs, run_kwargs = self.period_kwargs()
        tr = transient(self.sizes, self.bias, run_tgrid, V0=x0,
                       profile=bool(profile), inputs=run_inputs, **run_kwargs)
        if "inputs" not in tr:
            tr["inputs"] = {
                key: np.asarray(val, float).copy()
                for key, val in run_inputs.items()
            }
        x_end = _end_vector(tr, self.topo)
        residual = x_end - x0
        norm = float(np.linalg.norm(residual, ord=np.inf))
        nfail = int(tr.get("nfail", 0))
        if allow_freeze:
            self.maybe_freeze_grid(tr, norm, nfail)
        return _Orbit(tr, x0, x_end, residual, norm, nfail)

    def stabilize(self, x0, max_periods, phys_bounds):
        """Pseudo-transient stabilization with best-physical orbit tracking."""
        x = x0.copy()
        best_phys = None       # _Orbit | None (best physical orbit so far)
        for period_idx in range(max(0, int(max_periods))):
            x_start = x.copy()
            run = self.run_period(x_start, allow_freeze=True, profile=False)
            bounded = (_within(x_start, phys_bounds, self.topo)
                       and _within(run.x_end, phys_bounds, self.topo))
            if run.nfail == 0 and bounded and (
                    best_phys is None or run.norm < best_phys.norm):
                best_phys = _best_orbit(x_start, run.x_end, run.residual,
                                        run.norm, run.nfail, run.tr)
            if run.nfail == 0 and run.norm <= self.residual_tol and bounded:
                conv = _best_orbit(x_start, run.x_end, run.residual,
                                   run.norm, run.nfail, run.tr)
                return conv, best_phys, x_start, False, period_idx + 1
            diverging = (
                best_phys is not None and best_phys.norm < 0.5 and
                run.norm > max(3.0 * best_phys.norm, 5.0 * self.residual_tol)
            )
            if (not bounded) or diverging:
                return None, best_phys, x_start, True, period_idx + 1
            # Do not clip here: clipping can hide a runaway as a false fixed point.
            x = run.x_end
        return None, best_phys, x, False, max(0, int(max_periods))

    def rerun_frozen(self, x, *, profile=False):
        if not self.adaptive_grid_frozen:
            return None
        kw = dict(self.transient_kwargs)
        kw["adaptive"] = False
        kw["edge_mask"] = None
        self.period_runs += 1
        tr = transient(self.sizes, self.bias, self.frozen_tgrid, V0=x,
                       profile=bool(profile), inputs=self.frozen_inputs, **kw)
        if "inputs" not in tr:
            tr["inputs"] = {
                key: np.asarray(val, float).copy()
                for key, val in self.frozen_inputs.items()
            }
        x_end = _end_vector(tr, self.topo)
        residual = x_end - x
        norm = float(np.linalg.norm(residual, ord=np.inf))
        nfail = int(tr.get("nfail", 0))
        return _Orbit(tr, x, x_end, residual, norm, nfail)


def _build_shooting_jacobian_fd(x_base, residual_base, *, topo, bias,
                                rail_margin, fd_step, run_period):
    out = np.empty((topo.n, topo.n), dtype=float)
    for col in range(topo.n):
        h = float(fd_step) * max(1.0, abs(float(x_base[col])))
        if h == 0.0:
            h = float(fd_step)
        xp = x_base.copy()
        xp[col] += h
        xp = _rail_clip(xp, topo, bias, rail_margin)
        delta = float(xp[col] - x_base[col])
        if abs(delta) < 1e-30:
            xp = x_base.copy()
            xp[col] -= h
            xp = _rail_clip(xp, topo, bias, rail_margin)
            delta = float(xp[col] - x_base[col])
        if abs(delta) < 1e-30:
            out[:, col] = 0.0
            continue
        run = run_period(xp, allow_freeze=False)
        out[:, col] = (run.residual - residual_base) / delta
    return out


def _update_broyden_jacobian(jacobian, step, residual_delta):
    denom = float(np.dot(step, step))
    if denom <= 1e-30 or not np.isfinite(denom):
        return None
    correction = residual_delta - jacobian @ step
    if not np.all(np.isfinite(correction)):
        return None
    return jacobian + np.outer(correction, step) / denom


def _best_orbit(x0, x_end, residual, norm, nfail, tr):
    """Snapshot a run into a retained :class:`_Orbit`, copying the arrays so a
    later in-place reuse of the source can't corrupt the best-so-far."""
    return _Orbit(
        tr=tr,
        x0=np.asarray(x0, float).copy(),
        x_end=np.asarray(x_end, float).copy(),
        residual=np.asarray(residual, float).copy(),
        norm=float(norm),
        nfail=int(nfail),
    )


def _pss_status(converged, converged_stabilization, x, phys_bounds, topo):
    diverged = not _within(x, phys_bounds, topo)
    if diverged:
        return False, True, "diverged"
    if converged and converged_stabilization is not None:
        return True, False, "converged_stabilization"
    if converged:
        return True, False, "converged_shooting"
    return False, False, "best_physical"


def _finalize_pss_result(cur, runner, stepper, *, converged,
                         converged_stabilization, stab_runaway, phys_bounds,
                         iterations, history):
    """Assemble the public PSS result dict from the final orbit (``cur``) plus
    the two solver objects — ``runner`` (period grid, prepared inputs, the
    transient config, period-run count) and ``stepper`` (Jacobian counters,
    dominant multiplier). Only genuine solve-outcome flags are passed
    explicitly; everything else is read off those objects."""
    converged, diverged, pss_status = _pss_status(
        converged, converged_stabilization, cur.x0, phys_bounds, runner.topo)
    tr = cur.tr
    tk = runner.transient_kwargs
    final_inputs = tr.get("inputs")
    if final_inputs is None:
        final_inputs = _resample_inputs(
            runner.inputs, runner.tgrid, np.asarray(tr["t"], float), runner.period)
    result = dict(tr)
    result.update({
        "converged": bool(converged),
        "pss_status": pss_status,
        "diverged": bool(diverged),
        "dominant_multiplier": float(stepper.dominant_multiplier),
        "stabilization_runaway": bool(stab_runaway),
        "period": float(runner.period),
        "x0": np.asarray(cur.x0, float),
        "x_end": np.asarray(cur.x_end, float),
        "residual": np.asarray(cur.residual, float),
        "residual_norm": float(cur.norm),
        "residual_tol": float(runner.residual_tol),
        "shooting_iters": int(iterations),
        "shooting_history": history,
        "shooting_period_runs": int(runner.period_runs),
        "shooting_jacobian_evals": int(stepper.shooting_jacobian_evals),
        "shooting_jacobian_reuses": int(stepper.shooting_jacobian_reuses),
        "shooting_jacobian_reuse_enabled": bool(stepper.jacobian_reuse),
        "shooting_jacobian_rebuild_interval": int(stepper.jacobian_rebuild_interval),
        "nfail": int(cur.nfail),
        "topology": runner.topo,
        "inputs": {key: np.asarray(val, float).copy()
                   for key, val in final_inputs.items()},
        "node_inputs": dict(tk["node_inputs"] or {}),
        "current_inputs": tuple(tk["current_inputs"] or ()),
        "signed_devices": tuple(tk["signed_devices"] or ()),
        "transient_max_step": tk["max_step"],
        "transient_flat_max_step": tk["flat_max_step"],
        "rail_margin": tk["rail_margin"],
        "corner": tk["corner"],
        "adaptive": bool(runner.adaptive),
        "adaptive_grid_frozen": bool(runner.adaptive_grid_frozen),
    })
    return result


@dataclass(frozen=True, slots=True)
class _ShootingStepResult:
    accepted: bool
    best: _Orbit
    orbit: _Orbit = None          # the accepted orbit; None when no step was taken
    accepted_alpha: float = 0.0
    jacobian_kind: str = None
    rebuilt_after_reuse_failure: bool = False


class _PSSShootingStepper:
    """Own one shooting Newton/LM update and its Jacobian cache.

    ``pss_solve`` still owns the high-level convergence loop, while this class
    owns the fragile details: analytic-vs-FD Jacobian selection, Broyden reuse,
    LM/traditional damping, and trial-period acceptance.
    """

    def __init__(self, *, sizes, nf, bias, topo, inputs, node_inputs, runner,
                 adaptive, analytic_jacobian, integration_method, fd_step,
                 rail_margin, phys_bounds, residual_tol, min_damping,
                 jacobian_reuse, jacobian_rebuild_interval,
                 levenberg_marquardt):
        self.sizes = sizes
        self.nf = nf
        self.bias = bias
        self.topo = topo
        self.inputs = inputs
        self.node_inputs = node_inputs or {}
        self.runner = runner
        self.adaptive = bool(adaptive)
        self.analytic_jacobian = bool(analytic_jacobian)
        self.integration_method = integration_method
        self.fd_step = float(fd_step)
        self.rail_margin = rail_margin
        self.phys_bounds = phys_bounds
        self.residual_tol = float(residual_tol)
        self.min_damping = float(min_damping)
        self.jacobian_reuse = bool(jacobian_reuse)
        self.jacobian_rebuild_interval = max(0, int(jacobian_rebuild_interval))
        self.levenberg_marquardt = bool(levenberg_marquardt)

        self.jac = None
        self.jac_age = 0
        self.mono_dev_inst = None
        self.shooting_jacobian_evals = 0
        self.shooting_jacobian_reuses = 0
        self.dominant_multiplier = float("nan")
        # lm_mu == 0 reproduces the plain Newton step exactly; it grows only
        # after rejected trials and is carried across shooting iterations.
        self.lm_mu = 0.0
        self._lm_mu0 = 1e-3
        self._lm_up = 8.0
        self._lm_down = 1.0 / 3.0
        self._lm_mu_max = 1e8
        self._lm_max_tries = 15

    def _build_or_reuse_jacobian(self, tr, x, residual):
        rebuild = (
            self.jac is None or
            not self.jacobian_reuse or
            (self.jacobian_rebuild_interval > 0 and
             self.jac_age >= self.jacobian_rebuild_interval)
        )
        if not rebuild:
            self.shooting_jacobian_reuses += 1
            return self.jac, "broyden", True

        jac = None
        if self.analytic_jacobian and (
                not self.adaptive or self.runner.adaptive_grid_frozen):
            try:
                if self.mono_dev_inst is None:
                    self.mono_dev_inst = build_devices(
                        self.sizes, nf=self.nf, corner=self.runner.transient_kwargs.get("corner"),
                        topo=self.topo)
                phi = _shooting_monodromy(
                    tr, self.topo, self.sizes, self.nf, self.bias, self.inputs,
                    self.node_inputs, self.mono_dev_inst,
                    integration_method=self.integration_method)
                jac = phi - np.eye(self.topo.n)
                self.dominant_multiplier = _dominant_multiplier(phi)
                self.shooting_jacobian_evals += 1
                self.jac_age = 0
                self.jac = jac
                return jac, "analytic_monodromy", False
            except Exception as exc:
                diagnostics.note("pss.analytic_monodromy_fail", exc)
                jac = None

        if jac is None:
            self.shooting_jacobian_evals += 1
            self.jac_age = 0
            jac = _build_shooting_jacobian_fd(
                x, residual, topo=self.topo, bias=self.bias,
                rail_margin=self.rail_margin, fd_step=self.fd_step,
                run_period=self.runner.run_period)
            self.jac = jac
            return jac, "finite_difference", False

    def _try_lm_step(self, x, residual, norm, nfail, jac, best):
        current_score = _residual_score(norm, nfail)
        H = grad = diagH = None
        mu = self.lm_mu
        for _ in range(self._lm_max_tries):
            if mu == 0.0:
                try:
                    dx = np.linalg.solve(jac, -residual)
                except np.linalg.LinAlgError:
                    dx = np.linalg.lstsq(jac, -residual, rcond=None)[0]
            else:
                if H is None:
                    lm_scale = max(
                        1.0,
                        _finite_component_max(jac),
                        _finite_component_max(residual),
                    )
                    if not np.isfinite(lm_scale) or lm_scale <= 0.0:
                        break
                    jac_lm = np.nan_to_num(
                        jac / lm_scale, nan=0.0, posinf=0.0, neginf=0.0)
                    residual_lm = np.nan_to_num(
                        residual / lm_scale, nan=0.0, posinf=0.0, neginf=0.0)
                    jac_lm = np.asarray(jac_lm, dtype=float)
                    residual_lm = np.asarray(residual_lm, dtype=float)
                    post_scale = _finite_component_max(jac_lm)
                    if (not np.isfinite(post_scale)) or post_scale <= 0.0:
                        break
                    if post_scale > 1.0:
                        jac_lm = jac_lm / post_scale
                        residual_lm = residual_lm / post_scale
                    try:
                        with np.errstate(over="raise", invalid="raise",
                                         divide="raise"):
                            H = jac_lm.T @ jac_lm
                            grad = jac_lm.T @ residual_lm
                    except FloatingPointError:
                        H = grad = None
                        break
                    if (not np.all(np.isfinite(H)) or
                            not np.all(np.isfinite(grad))):
                        H = grad = None
                        break
                    diagH = np.maximum(np.abs(np.diag(H)), 1e-30)
                A = H + mu * np.diag(diagH)
                try:
                    dx = np.linalg.solve(A, -grad)
                except np.linalg.LinAlgError:
                    dx = np.linalg.lstsq(A, -grad, rcond=None)[0]
            xt = x + dx
            if not _within(xt, self.phys_bounds, self.topo):
                mu = mu * self._lm_up if mu > 0.0 else self._lm_mu0
                if mu > self._lm_mu_max:
                    break
                continue
            xt = _rail_clip(xt, self.topo, self.bias, self.rail_margin)
            trial = self.runner.run_period(xt, allow_freeze=False)
            score_t = _residual_score(trial.norm, trial.nfail)
            bounded_t = _within(trial.x_end, self.phys_bounds, self.topo)
            if bounded_t and (score_t < current_score or
                              trial.norm <= self.residual_tol):
                self.lm_mu = mu * self._lm_down
                return (mu, trial), best
            if bounded_t and trial.nfail == 0 and score_t < best.score:
                best = _best_orbit(
                    xt, trial.x_end, trial.residual, trial.norm,
                    trial.nfail, trial.tr)
            mu = mu * self._lm_up if mu > 0.0 else self._lm_mu0
            if mu > self._lm_mu_max:
                break
        return None, best

    def _try_damped_step(self, x, residual, norm, nfail, jac, best):
        current_score = _residual_score(norm, nfail)
        try:
            dx = np.linalg.solve(jac, -residual)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(jac, -residual, rcond=None)[0]
        alpha = 1.0
        while alpha >= self.min_damping:
            xt = _rail_clip(x + alpha * dx, self.topo, self.bias, self.rail_margin)
            trial = self.runner.run_period(xt, allow_freeze=False)
            score_t = _residual_score(trial.norm, trial.nfail)
            if score_t < current_score or trial.norm <= self.residual_tol:
                return (alpha, trial), best
            if score_t < best.score:
                best = _best_orbit(
                    xt, trial.x_end, trial.residual, trial.norm,
                    trial.nfail, trial.tr)
            alpha *= 0.5
        return None, best

    def step(self, cur, best):
        accepted = None
        jacobian_kind = None
        rebuilt_after_reuse_failure = False
        old_x = cur.x0.copy()
        old_residual = cur.residual.copy()

        for jac_attempt in range(2):
            jac, jacobian_kind, used_reused_jac = self._build_or_reuse_jacobian(
                cur.tr, cur.x0, cur.residual)

            if self.levenberg_marquardt and not np.all(np.isfinite(jac)):
                self.jac = None
                self.jac_age = 0
                if used_reused_jac and jac_attempt == 0:
                    rebuilt_after_reuse_failure = True
                    continue
                break

            if self.levenberg_marquardt:
                accepted, best = self._try_lm_step(
                    cur.x0, cur.residual, cur.norm, cur.nfail, jac, best)
            else:
                accepted, best = self._try_damped_step(
                    cur.x0, cur.residual, cur.norm, cur.nfail, jac, best)

            if accepted is not None:
                break
            if used_reused_jac and jac_attempt == 0:
                self.jac = None
                self.jac_age = 0
                rebuilt_after_reuse_failure = True
                continue
            break

        if accepted is None:
            return _ShootingStepResult(
                accepted=False,
                best=best,
                jacobian_kind=jacobian_kind,
                rebuilt_after_reuse_failure=rebuilt_after_reuse_failure,
            )

        alpha, orbit = accepted
        self.runner.maybe_freeze_grid(orbit.tr, orbit.norm, orbit.nfail)
        if orbit.score < best.score:
            best = _best_orbit(orbit.x0, orbit.x_end, orbit.residual,
                               orbit.norm, orbit.nfail, orbit.tr)
        if self.jacobian_reuse and (
                not self.adaptive or self.runner.adaptive_grid_frozen):
            self.jac = _update_broyden_jacobian(
                self.jac, orbit.x0 - old_x, orbit.residual - old_residual)
            if self.jac is None:
                self.jac_age = 0
            else:
                self.jac_age += 1
        else:
            self.jac = None
            self.jac_age = 0

        return _ShootingStepResult(
            accepted=True,
            best=best,
            orbit=orbit,
            accepted_alpha=float(alpha),
            jacobian_kind=jacobian_kind,
            rebuilt_after_reuse_failure=rebuilt_after_reuse_failure,
        )


def pss_solve(sizes, bias, period, *, topo=AFE_TOPO, nf=None, tgrid=None,
              n_points=161, inputs=None, node_inputs=None, current_inputs=None,
              corner=None,
              V0=None, tstab_periods=0, max_step=None, flat_max_step=None,
              max_retry_subdivisions=0, newton_maxit=30,
              newton_step_limit=5.0, newton_vtol=1e-8,
              fallback_full_jacobian=False, fallback_least_squares=False,
              fallback_tol=1e-9, signed_devices=None, residual_tol=1e-7,
              max_shooting_iters=8, fd_step=1e-5, min_damping=1.0 / 64.0,
              jacobian_reuse=True, jacobian_rebuild_interval=0,
              analytic_jacobian=True,
              rail_margin=0.5, check_periodic_inputs=True,
              input_periodic_tol=1e-9, profile=False, edge_mask=None,
              integration_method="gear2",
              physical_factor=2.0, max_stabilization_periods=200,
              levenberg_marquardt=True, cap_mode=None, cap_mode_id=None,
              adaptive=False, adaptive_reltol=1e-4, adaptive_vabstol=1e-6,
              adaptive_iabstol=1e-12, adaptive_max_steps=200000,
              adaptive_h0=None, adaptive_freeze_factor=10.0,
              adaptive_config=None):
    """Solve periodic steady state with transient shooting.

    Parameters are intentionally close to :func:`transient` so the same topology,
    waveform, current-source, and switch-current metadata can be reused.

    Returns a dictionary containing the final one-period trajectory, the PSS
    initial state ``x0``, ``residual = x(T)-x0``, residual norm, convergence flag,
    and shooting iteration history.  Non-convergence is reported in the result
    instead of raising, so callers can inspect the best trajectory.
    """
    adaptive_config = resolve_adaptive_config(
        adaptive_config,
        adaptive_reltol=adaptive_reltol,
        adaptive_vabstol=adaptive_vabstol,
        adaptive_iabstol=adaptive_iabstol,
        adaptive_max_steps=adaptive_max_steps,
        adaptive_h0=adaptive_h0,
        adaptive_freeze_factor=adaptive_freeze_factor,
    )
    tgrid = _make_period_grid(period, tgrid, n_points)
    period = float(period)
    inputs = _prepare_inputs(
        inputs, tgrid,
        check_periodic=bool(check_periodic_inputs),
        periodic_tol=float(input_periodic_tol),
    )
    if edge_mask is not None:
        edge_mask = np.asarray(edge_mask, dtype=bool)
        if len(edge_mask) != len(tgrid):
            raise ValueError("edge_mask length must match tgrid")

    step_fallback_tol = min(float(fallback_tol), 0.1 * float(residual_tol))
    transient_kwargs = dict(
        topo=topo,
        node_inputs=node_inputs,
        current_inputs=current_inputs,
        nf=nf,
        corner=corner,
        max_step=max_step,
        flat_max_step=flat_max_step,
        max_retry_subdivisions=max_retry_subdivisions,
        newton_maxit=newton_maxit,
        newton_step_limit=newton_step_limit,
        newton_vtol=newton_vtol,
        fallback_full_jacobian=fallback_full_jacobian,
        fallback_least_squares=fallback_least_squares,
        fallback_tol=step_fallback_tol,
        signed_devices=signed_devices,
        rail_margin=rail_margin,
        edge_mask=edge_mask,
        integration_method=integration_method,
        adaptive=bool(adaptive),
        adaptive_config=adaptive_config,
        cap_mode=cap_mode,
        cap_mode_id=cap_mode_id,
        # Shooting manages its own convergence per period; never let a single
        # period silently fall back to a BE orbit mid-iteration.
        gear2_be_fallback=False,
    )

    x = _initial_vector(sizes, bias, topo, nf, V0, corner=corner)
    x = _rail_clip(x, topo, bias, rail_margin)

    phys_bounds = _physical_span(topo, bias, float(physical_factor))
    stab_runaway = False
    runner = _PSSPeriodRunner(
        sizes=sizes,
        bias=bias,
        period=period,
        topo=topo,
        tgrid=tgrid,
        inputs=inputs,
        transient_kwargs=transient_kwargs,
        residual_tol=residual_tol,
        adaptive=adaptive,
        adaptive_config=adaptive_config,
    )

    converged_stabilization, stab_best, x, stab_runaway, _ = runner.stabilize(
        x, tstab_periods, phys_bounds)
    # If the chase drifted out of bounds, roll back to the best physical orbit
    # instead of carrying the runaway state into shooting.
    if converged_stabilization is None and stab_runaway and stab_best is not None:
        x = stab_best.x0.copy()

    history = []

    if converged_stabilization is None:
        cur = runner.run_period(x)
    else:
        cur = converged_stabilization
    best = _best_orbit(cur.x0, cur.x_end, cur.residual, cur.norm, cur.nfail, cur.tr)
    history.append({
        "iter": 0,
        "residual_norm": cur.norm,
        "nfail": cur.nfail,
        "accepted_alpha": 0.0,
    })

    converged = (cur.nfail == 0 and cur.norm <= float(residual_tol))
    iterations = 0
    jacobian_reuse = bool(jacobian_reuse)
    jacobian_rebuild_interval = max(0, int(jacobian_rebuild_interval))
    stepper = _PSSShootingStepper(
        sizes=sizes,
        nf=nf,
        bias=bias,
        topo=topo,
        inputs=inputs,
        node_inputs=node_inputs,
        runner=runner,
        adaptive=adaptive,
        analytic_jacobian=analytic_jacobian,
        integration_method=integration_method,
        fd_step=fd_step,
        rail_margin=rail_margin,
        phys_bounds=phys_bounds,
        residual_tol=residual_tol,
        min_damping=min_damping,
        jacobian_reuse=jacobian_reuse,
        jacobian_rebuild_interval=jacobian_rebuild_interval,
        levenberg_marquardt=levenberg_marquardt,
    )

    for iteration in range(1, int(max_shooting_iters) + 1):
        if converged:
            break
        iterations = iteration

        step = stepper.step(cur, best)
        best = step.best
        if not step.accepted:
            history.append({
                "iter": iteration,
                "residual_norm": cur.norm,
                "nfail": cur.nfail,
                "accepted_alpha": 0.0,
                "jacobian": step.jacobian_kind,
                "stalled": True,
            })
            break

        cur = step.orbit
        history.append({
            "iter": iteration,
            "residual_norm": cur.norm,
            "nfail": cur.nfail,
            "accepted_alpha": step.accepted_alpha,
            "jacobian": step.jacobian_kind,
            "rebuilt_after_reuse_failure": bool(step.rebuilt_after_reuse_failure),
        })
        converged = (cur.nfail == 0 and cur.norm <= float(residual_tol))

    # A2: adaptive-stabilization fallback. If shooting did not converge but the
    # best orbit is physical (no runaway), extend pseudo-transient stabilization
    # from it up to the budget. Well-conditioned circuits (e.g. the chopper)
    # converge during shooting and never reach here, so their path is unchanged.
    if (not converged and int(max_stabilization_periods) > 0 and not stab_runaway
            and _within(best.x0, phys_bounds, topo)):
        extra = int(max_stabilization_periods) - runner.period_runs
        if extra > 0:
            conv2, best2, _, runaway2, _ = runner.stabilize(
                best.x0, extra, phys_bounds)
            if conv2 is not None:
                cur = conv2
                converged = True
                converged_stabilization = conv2
            elif best2 is not None and best2.norm < best.norm:
                best = best2
            stab_runaway = stab_runaway or runaway2

    if not converged:
        cur = best
        converged = (cur.nfail == 0 and cur.norm <= float(residual_tol))

    if runner.adaptive_grid_frozen and bool(cur.tr.get("adaptive", False)):
        cur = runner.rerun_frozen(cur.x0)
        converged = (cur.nfail == 0 and cur.norm <= float(residual_tol))

    if profile:
        cur = runner.run_period(cur.x0, allow_freeze=False, profile=True)
        converged = (cur.nfail == 0 and cur.norm <= float(residual_tol))

    return _finalize_pss_result(
        cur, runner, stepper,
        converged=converged,
        converged_stabilization=converged_stabilization,
        stab_runaway=stab_runaway, phys_bounds=phys_bounds,
        iterations=iterations, history=history)
