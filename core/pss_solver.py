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
    from .ac_mna import _stamp_adm, _stamp_mos_lti, _branch_incidence
    from .ac_solver import ac_solve, _dev_corner, get_ss_params
    from .device_model import create_device, get_default_model_type
    from .topology import AFE_TOPO
    from .transient_solver import transient
except ImportError:  # pragma: no cover - legacy direct module import
    from ac_mna import _stamp_adm, _stamp_mos_lti, _branch_incidence
    from ac_solver import ac_solve, _dev_corner, get_ss_params
    from device_model import create_device, get_default_model_type
    from topology import AFE_TOPO
    from transient_solver import transient


def _nfval(nf, name):
    if isinstance(nf, dict):
        return int(nf.get(name, 1))
    return int(nf) if nf else 1


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
                              nf=_nfval(nf, name), dev_inst=dev_inst[name])
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
    except Exception:
        pass

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
    except Exception:
        return float("nan")


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
              levenberg_marquardt=True):
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

    step_fallback_tol = min(float(fallback_tol), 0.1 * float(residual_tol))
    transient_kwargs = dict(
        topo=topo,
        inputs=inputs,
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
        # Shooting manages its own convergence per period; never let a single
        # period silently fall back to a BE orbit mid-iteration.
        gear2_be_fallback=False,
    )

    x = _initial_vector(sizes, bias, topo, nf, V0, corner=corner)
    x = _rail_clip(x, topo, bias, rail_margin)

    period_runs = 0
    shooting_jacobian_evals = 0
    shooting_jacobian_reuses = 0
    phys_bounds = _physical_span(topo, bias, float(physical_factor))
    stab_runaway = False

    def _stabilize(x0, max_periods):
        """Pseudo-transient stabilization: advance period-by-period, tracking the
        best *physically bounded* min-residual orbit, and bail on a runaway (a step
        that leaves the physical box). Returns (converged_tuple_or_None,
        best_physical_or_None, last_x, hit_runaway, periods_run)."""
        nonlocal period_runs
        x = x0.copy()
        best_phys = None       # (norm, x0, x_end, residual, nfail, tr)
        for _ in range(max(0, int(max_periods))):
            x_start = x.copy()
            period_runs += 1
            tr_s = transient(sizes, bias, tgrid, V0=x, profile=False,
                             **transient_kwargs)
            x_end_s = _end_vector(tr_s, topo)
            residual_s = x_end_s - x_start
            norm_s = float(np.linalg.norm(residual_s, ord=np.inf))
            nfail_s = int(tr_s.get("nfail", 0))
            bounded = (_within(x_start, phys_bounds, topo)
                       and _within(x_end_s, phys_bounds, topo))
            if nfail_s == 0 and bounded and (best_phys is None or norm_s < best_phys[0]):
                best_phys = (norm_s, x_start.copy(), x_end_s.copy(),
                             residual_s.copy(), nfail_s, tr_s)
            if nfail_s == 0 and norm_s <= float(residual_tol) and bounded:
                conv = (tr_s, x_start.copy(), x_end_s.copy(),
                        residual_s.copy(), norm_s, nfail_s)
                return conv, best_phys, x_start, False, _ + 1
            # Runaway detection: a step out of the physical box, or the period
            # residual turning back UP past a good minimum (the orbit is drifting
            # off a thin basin toward a spurious fixed point). Bail to best_phys.
            diverging = (best_phys is not None and best_phys[0] < 0.5
                         and norm_s > max(3.0 * best_phys[0], 5.0 * float(residual_tol)))
            if (not bounded) or diverging:
                return None, best_phys, x_start, True, _ + 1
            # Advance WITHOUT rail-clipping: clipping a runaway to the rail forges a
            # deceptive in-bounds zero-residual fixed point that fools best_phys;
            # letting the true trajectory run makes the runaway detectable instead.
            x = x_end_s
        return None, best_phys, x, False, max(0, int(max_periods))

    converged_stabilization, stab_best, x, stab_runaway, _ = _stabilize(
        x, tstab_periods)
    # If the chase drifted out of bounds, roll back to the best physical orbit
    # instead of carrying the runaway state into shooting.
    if converged_stabilization is None and stab_runaway and stab_best is not None:
        x = stab_best[1].copy()

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

    if converged_stabilization is None:
        tr, x_end, residual, norm, nfail = run_period(x)
    else:
        tr, x, x_end, residual, norm, nfail = converged_stabilization
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
    mono_dev_inst = None
    jacobian_reuse = bool(jacobian_reuse)
    jacobian_rebuild_interval = max(0, int(jacobian_rebuild_interval))
    # Levenberg–Marquardt damping state, carried across shooting iterations.
    # lm_mu == 0 reproduces the plain Newton step exactly (well-conditioned
    # circuits unchanged); it grows only when a step is rejected.
    lm_mu = 0.0
    lm_mu0, lm_up, lm_down, lm_mu_max, lm_max_tries = 1e-3, 8.0, 1.0 / 3.0, 1e8, 15
    dominant_multiplier = float("nan")   # max |Floquet multiplier| (stiffness)

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
                jac = None
                if analytic_jacobian:
                    try:
                        if mono_dev_inst is None:
                            mono_dev_inst = {
                                name: create_device(get_default_model_type(),
                                    W=sizes[name][0], L=sizes[name][1],
                                    NF=_nfval(nf, name),
                                    **_dev_corner(corner, name))
                                for name, *_ in topo.devices
                            }
                        phi = _shooting_monodromy(tr, topo, sizes, nf, bias, inputs,
                                                  node_inputs or {}, mono_dev_inst,
                                                  integration_method=integration_method)
                        jac = phi - np.eye(topo.n)
                        jacobian_kind = "analytic_monodromy"
                        dominant_multiplier = _dominant_multiplier(phi)
                        shooting_jacobian_evals += 1
                        jac_age = 0
                    except Exception:
                        jac = None
                if jac is None:
                    jac = build_shooting_jacobian(x, residual)
                    jacobian_kind = "finite_difference"
            else:
                shooting_jacobian_reuses += 1
                used_reused_jac = True
                jacobian_kind = "broyden"

            current_score = _residual_score(norm, nfail)
            # A non-finite Jacobian (a diverged trial orbit polluted a Broyden update,
            # or the monodromy of a failed period) can't yield a usable step — force a
            # rebuild next attempt instead of forming a NaN J^T J.
            if levenberg_marquardt and not np.all(np.isfinite(jac)):
                jac = None
                jac_age = 0
                if used_reused_jac and jac_attempt == 0:
                    rebuilt_after_reuse_failure = True
                    continue
                break
            if levenberg_marquardt:
                # LM trust region: mu=0 -> the exact Newton step (well-conditioned
                # circuits, e.g. the chopper, are byte-identical); mu grows only on
                # rejection, regularizing near-singular (I-M) so a stiff (tau>>T)
                # orbit's step cannot overshoot the basin into a runaway.
                H = jac.T @ jac
                grad = jac.T @ residual
                diagH = np.maximum(np.abs(np.diag(H)), 1e-30)
                mu = lm_mu
                for _lm in range(lm_max_tries):
                    if mu == 0.0:
                        try:
                            dx = np.linalg.solve(jac, -residual)
                        except np.linalg.LinAlgError:
                            dx = np.linalg.lstsq(jac, -residual, rcond=None)[0]
                    else:
                        A = H + mu * np.diag(diagH)
                        try:
                            dx = np.linalg.solve(A, -grad)
                        except np.linalg.LinAlgError:
                            dx = np.linalg.lstsq(A, -grad, rcond=None)[0]
                    xt = x + dx
                    if not _within(xt, phys_bounds, topo):
                        mu = mu * lm_up if mu > 0.0 else lm_mu0
                        if mu > lm_mu_max:
                            break
                        continue
                    xt = _rail_clip(xt, topo, bias, rail_margin)
                    tr_t, x_end_t, residual_t, norm_t, nfail_t = run_period(xt)
                    score_t = _residual_score(norm_t, nfail_t)
                    bounded_t = _within(x_end_t, phys_bounds, topo)
                    if bounded_t and (score_t < current_score or
                                      norm_t <= float(residual_tol)):
                        accepted = (mu, xt, tr_t, x_end_t, residual_t, norm_t,
                                    nfail_t, score_t)
                        lm_mu = mu * lm_down
                        break
                    if bounded_t and nfail_t == 0 and score_t < best["score"]:
                        best = {
                            "x0": xt.copy(), "x_end": x_end_t.copy(),
                            "residual": residual_t.copy(), "residual_norm": norm_t,
                            "nfail": nfail_t, "transient": tr_t, "score": score_t,
                        }
                    mu = mu * lm_up if mu > 0.0 else lm_mu0
                    if mu > lm_mu_max:
                        break
            else:
                try:
                    dx = np.linalg.solve(jac, -residual)
                except np.linalg.LinAlgError:
                    dx = np.linalg.lstsq(jac, -residual, rcond=None)[0]
                alpha = 1.0
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
                            "x0": xt.copy(), "x_end": x_end_t.copy(),
                            "residual": residual_t.copy(), "residual_norm": norm_t,
                            "nfail": nfail_t, "transient": tr_t, "score": score_t,
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

    # A2: adaptive-stabilization fallback. If shooting did not converge but the
    # best orbit is physical (no runaway), extend pseudo-transient stabilization
    # from it up to the budget. Well-conditioned circuits (e.g. the chopper)
    # converge during shooting and never reach here, so their path is unchanged.
    if (not converged and int(max_stabilization_periods) > 0 and not stab_runaway
            and _within(best["x0"], phys_bounds, topo)):
        extra = int(max_stabilization_periods) - period_runs
        if extra > 0:
            conv2, best2, _, runaway2, _ = _stabilize(best["x0"], extra)
            if conv2 is not None:
                tr, x, x_end, residual, norm, nfail = conv2
                converged = True
                converged_stabilization = conv2
            elif best2 is not None and best2[0] < best["residual_norm"]:
                norm2, x0_2, xend2, res2, nfail2, tr2 = best2
                best = {"x0": x0_2, "x_end": xend2, "residual": res2,
                        "residual_norm": norm2, "nfail": nfail2, "transient": tr2,
                        "score": _residual_score(norm2, nfail2)}
            stab_runaway = stab_runaway or runaway2

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

    # A4: honest status. A physically-out-of-bounds final orbit is a numerical
    # runaway, never a steady state — report it as diverged, not converged.
    diverged = not _within(x, phys_bounds, topo)
    if diverged:
        converged = False
        pss_status = "diverged"
    elif converged and converged_stabilization is not None:
        pss_status = "converged_stabilization"
    elif converged:
        pss_status = "converged_shooting"
    else:
        pss_status = "best_physical"

    result = dict(tr)
    result.update({
        "converged": bool(converged),
        "pss_status": pss_status,
        "diverged": bool(diverged),
        "dominant_multiplier": float(dominant_multiplier),
        "stabilization_runaway": bool(stab_runaway),
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
        "corner": corner,
    })
    return result
