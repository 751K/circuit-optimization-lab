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

import numpy as np

try:
    from .ac_solver import ac_solve
    from .topology import AFE_TOPO
    from .transient_solver import transient
except ImportError:  # pragma: no cover - legacy direct module import
    from ac_solver import ac_solve
    from topology import AFE_TOPO
    from transient_solver import transient


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


def _initial_vector(sizes, bias, topo, nf, V0):
    if V0 is not None:
        if isinstance(V0, dict):
            default = topo.default_guess_value(bias)
            return np.asarray(topo.guess_vector(V0, default=default), float)
        arr = np.asarray(V0, float)
        if arr.shape != (topo.n,):
            raise ValueError(f"V0 shape {arr.shape} does not match topology size {topo.n}")
        return arr.copy()

    try:
        ac = ac_solve(sizes, bias, np.array([1.0]), topo=topo, nf=nf)
        if ac is not None and "dc_op" in ac:
            return np.asarray([ac["dc_op"][node] for node in topo.solved], float)
    except Exception:
        pass

    guesses = topo.dc_guess_vectors(bias)
    if not guesses:
        return np.full(topo.n, topo.default_guess_value(bias), dtype=float)
    return np.asarray(guesses[0], float)


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


def pss_solve(sizes, bias, period, *, topo=AFE_TOPO, nf=None, tgrid=None,
              n_points=161, inputs=None, node_inputs=None, current_inputs=None,
              V0=None, tstab_periods=0, max_step=None, flat_max_step=None,
              max_retry_subdivisions=0, newton_maxit=30,
              newton_step_limit=5.0, newton_vtol=1e-8,
              fallback_full_jacobian=False, fallback_least_squares=False,
              fallback_tol=1e-9, signed_devices=None, residual_tol=1e-7,
              max_shooting_iters=8, fd_step=1e-5, min_damping=1.0 / 64.0,
              jacobian_reuse=True, jacobian_rebuild_interval=0,
              rail_margin=0.5, check_periodic_inputs=True,
              input_periodic_tol=1e-9, profile=False, edge_mask=None):
    """Solve periodic steady state with transient shooting.

    Parameters are intentionally close to :func:`transient` so the same topology,
    waveform, current-source, and switch-current metadata can be reused.

    Returns a dictionary containing the final one-period trajectory, the PSS
    initial state ``x0``, ``residual = x(T)-x0``, residual norm, convergence flag,
    and shooting iteration history.  Non-convergence is reported in the result
    instead of raising, so callers can inspect the best trajectory.
    """
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

    transient_kwargs = dict(
        topo=topo,
        inputs=inputs,
        node_inputs=node_inputs,
        current_inputs=current_inputs,
        nf=nf,
        max_step=max_step,
        flat_max_step=flat_max_step,
        max_retry_subdivisions=max_retry_subdivisions,
        newton_maxit=newton_maxit,
        newton_step_limit=newton_step_limit,
        newton_vtol=newton_vtol,
        fallback_full_jacobian=fallback_full_jacobian,
        fallback_least_squares=fallback_least_squares,
        fallback_tol=fallback_tol,
        signed_devices=signed_devices,
        rail_margin=rail_margin,
        edge_mask=edge_mask,
    )

    x = _initial_vector(sizes, bias, topo, nf, V0)
    x = _rail_clip(x, topo, bias, rail_margin)

    period_runs = 0
    shooting_jacobian_evals = 0
    shooting_jacobian_reuses = 0
    for _ in range(max(0, int(tstab_periods))):
        period_runs += 1
        tr_stab = transient(sizes, bias, tgrid, V0=x, profile=False,
                            **transient_kwargs)
        x = _rail_clip(_end_vector(tr_stab, topo), topo, bias, rail_margin)

    history = []

    def run_period(x0):
        nonlocal period_runs
        period_runs += 1
        tr = transient(sizes, bias, tgrid, V0=x0, profile=False,
                       **transient_kwargs)
        x_end = _end_vector(tr, topo)
        residual = x_end - x0
        norm = float(np.linalg.norm(residual, ord=np.inf))
        nfail = int(tr.get("nfail", 0))
        return tr, x_end, residual, norm, nfail

    tr, x_end, residual, norm, nfail = run_period(x)
    best = {
        "x0": x.copy(),
        "x_end": x_end.copy(),
        "residual": residual.copy(),
        "residual_norm": norm,
        "nfail": nfail,
        "transient": tr,
        "score": _residual_score(norm, nfail),
    }
    history.append({
        "iter": 0,
        "residual_norm": norm,
        "nfail": nfail,
        "accepted_alpha": 0.0,
    })

    converged = (nfail == 0 and norm <= float(residual_tol))
    iterations = 0
    jac = None
    jac_age = 0
    jacobian_reuse = bool(jacobian_reuse)
    jacobian_rebuild_interval = max(0, int(jacobian_rebuild_interval))

    def build_shooting_jacobian(x_base, residual_base):
        nonlocal shooting_jacobian_evals, jac_age
        shooting_jacobian_evals += 1
        jac_age = 0
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
            _, _, rp, _, _ = run_period(xp)
            out[:, col] = (rp - residual_base) / delta
        return out

    def update_broyden(jacobian, step, residual_delta):
        nonlocal jac_age
        denom = float(np.dot(step, step))
        if denom <= 1e-30 or not np.isfinite(denom):
            jac_age = 0
            return None
        correction = residual_delta - jacobian @ step
        if not np.all(np.isfinite(correction)):
            jac_age = 0
            return None
        jac_age += 1
        return jacobian + np.outer(correction, step) / denom

    for iteration in range(1, int(max_shooting_iters) + 1):
        if converged:
            break
        iterations = iteration

        accepted = None
        jacobian_kind = None
        rebuilt_after_reuse_failure = False
        old_x = x.copy()
        old_residual = residual.copy()
        for jac_attempt in range(2):
            rebuild = (
                jac is None or
                not jacobian_reuse or
                (jacobian_rebuild_interval > 0 and
                 jac_age >= jacobian_rebuild_interval)
            )
            used_reused_jac = False
            if rebuild:
                jac = build_shooting_jacobian(x, residual)
                jacobian_kind = "finite_difference"
            else:
                shooting_jacobian_reuses += 1
                used_reused_jac = True
                jacobian_kind = "broyden"

            try:
                dx = np.linalg.solve(jac, -residual)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(jac, -residual, rcond=None)[0]

            alpha = 1.0
            current_score = _residual_score(norm, nfail)
            while alpha >= float(min_damping):
                xt = _rail_clip(x + alpha * dx, topo, bias, rail_margin)
                tr_t, x_end_t, residual_t, norm_t, nfail_t = run_period(xt)
                score_t = _residual_score(norm_t, nfail_t)
                if score_t < current_score or norm_t <= float(residual_tol):
                    accepted = (alpha, xt, tr_t, x_end_t, residual_t, norm_t,
                                nfail_t, score_t)
                    break
                if score_t < best["score"]:
                    best = {
                        "x0": xt.copy(),
                        "x_end": x_end_t.copy(),
                        "residual": residual_t.copy(),
                        "residual_norm": norm_t,
                        "nfail": nfail_t,
                        "transient": tr_t,
                        "score": score_t,
                    }
                alpha *= 0.5

            if accepted is not None:
                break
            if used_reused_jac and jac_attempt == 0:
                jac = None
                jac_age = 0
                rebuilt_after_reuse_failure = True
                continue
            break

        if accepted is None:
            history.append({
                "iter": iteration,
                "residual_norm": norm,
                "nfail": nfail,
                "accepted_alpha": 0.0,
                "jacobian": jacobian_kind,
                "stalled": True,
            })
            break

        alpha, x, tr, x_end, residual, norm, nfail, score = accepted
        if score < best["score"]:
            best = {
                "x0": x.copy(),
                "x_end": x_end.copy(),
                "residual": residual.copy(),
                "residual_norm": norm,
                "nfail": nfail,
                "transient": tr,
                "score": score,
            }
        if jacobian_reuse:
            jac = update_broyden(jac, x - old_x, residual - old_residual)
        else:
            jac = None
            jac_age = 0
        history.append({
            "iter": iteration,
            "residual_norm": norm,
            "nfail": nfail,
            "accepted_alpha": float(alpha),
            "jacobian": jacobian_kind,
            "rebuilt_after_reuse_failure": bool(rebuilt_after_reuse_failure),
        })
        converged = (nfail == 0 and norm <= float(residual_tol))

    if not converged:
        x = best["x0"]
        tr = best["transient"]
        x_end = best["x_end"]
        residual = best["residual"]
        norm = best["residual_norm"]
        nfail = best["nfail"]
        converged = (nfail == 0 and norm <= float(residual_tol))

    if profile:
        period_runs += 1
        tr = transient(sizes, bias, tgrid, V0=x, profile=True, **transient_kwargs)
        x_end = _end_vector(tr, topo)
        residual = x_end - x
        norm = float(np.linalg.norm(residual, ord=np.inf))
        nfail = int(tr.get("nfail", 0))
        converged = (nfail == 0 and norm <= float(residual_tol))

    result = dict(tr)
    result.update({
        "converged": bool(converged),
        "period": period,
        "x0": np.asarray(x, float),
        "x_end": np.asarray(x_end, float),
        "residual": np.asarray(residual, float),
        "residual_norm": float(norm),
        "residual_tol": float(residual_tol),
        "shooting_iters": int(iterations),
        "shooting_history": history,
        "shooting_period_runs": int(period_runs),
        "shooting_jacobian_evals": int(shooting_jacobian_evals),
        "shooting_jacobian_reuses": int(shooting_jacobian_reuses),
        "shooting_jacobian_reuse_enabled": bool(jacobian_reuse),
        "shooting_jacobian_rebuild_interval": int(jacobian_rebuild_interval),
        "nfail": int(nfail),
        "topology": topo,
        "inputs": {key: val.copy() for key, val in inputs.items()},
        "node_inputs": dict(node_inputs or {}),
        "current_inputs": tuple(current_inputs or ()),
        "signed_devices": tuple(signed_devices or ()),
        "transient_max_step": max_step,
        "transient_flat_max_step": flat_max_step,
        "rail_margin": rail_margin,
    })
    return result
