"""
DC operating-point solving — bounded least-squares fallback + the AFE
symmetric-seeding/continuation heuristics.

The bounded least-squares fallback (`bounded_least_squares_dc`) is a generic
last-resort DC solve. Everything AFE-flavoured here — `_AFE_SYMMETRIC_PAIRS`,
`is_afe_topology`, `is_pairwise_symmetric_afe`, `symmetric_seed`,
`symmetric_continuation` — is a CIRCUIT-SPECIFIC seeding strategy for the AFE
topology, NOT general solver logic. It selects the physical (Spectre-matching)
symmetric power-up branch for that one circuit and lives out here so the generic
solver stays free of per-circuit branches.
"""
import numpy as np
from .device_factory import dev_nf
from .topology import AFE_TOPO
from . import diagnostics


_AFE_SYMMETRIC_PAIRS = (("M7", "M8"), ("M9", "M10"), ("M12", "M13"), ("M14", "M15"))
DC_FALLBACK_TOL = 1e-10


def is_afe_topology(topo):
    """Structural AFE check.

    JSON-loaded AFE topologies are not the same Python object as `AFE_TOPO`, but
    they should still use the AFE-specific symmetric DC continuation and guards.
    Keep this strict: those helpers assume the canonical AFE node/device names.
    """
    return (
        tuple(getattr(topo, "solved", ())) == tuple(AFE_TOPO.solved) and
        tuple(getattr(topo, "devices", ())) == tuple(AFE_TOPO.devices) and
        dict(getattr(topo, "rails", {})) == dict(AFE_TOPO.rails)
    )


def is_pairwise_symmetric_afe(sizes, nf, topo):
    """True only when the default AFE can be reduced to the 4-node symmetric DC solve."""
    if not is_afe_topology(topo):
        return False
    for left, right in _AFE_SYMMETRIC_PAIRS:
        if sizes.get(left) != sizes.get(right):
            return False
        if dev_nf(nf, left) != dev_nf(nf, right):
            return False
    return True


def dc_residual_ok(residuals, x, tol=1e-9):
    try:
        return np.linalg.norm(residuals(x), ord=np.inf) < tol
    except Exception as exc:
        diagnostics.note("dc.residual_eval_fail", exc)
        return False


def bounded_least_squares_dc(residuals, guesses, topo, bias, tol=DC_FALLBACK_TOL):
    """Last-resort bounded DC solve.

    fsolve is fast near a good root but can run to absurd voltages on bad AFE
    slider combinations. This fallback keeps nodes inside the rail box and only
    accepts a solution when KCL is still tight.
    """
    from scipy.optimize import least_squares
    rails = [v for v in topo.rail_values(bias).values() if isinstance(v, (int, float))]
    if not rails:
        return None
    lo = min(rails) - 0.5
    hi = max(rails) + 0.5
    # Ideal voltage-source branch currents are NOT node voltages: keep node rows in the
    # rail box but leave branch rows unbounded, else least_squares would clamp the source
    # current to the voltage box. (m=0 -> scalar bounds, unchanged.)
    m = getattr(topo, "n_branches", 0)
    if m:
        lo = np.concatenate([np.full(topo.n, lo), np.full(m, -np.inf)])
        hi = np.concatenate([np.full(topo.n, hi), np.full(m, np.inf)])
    best_x = None
    best_norm = np.inf
    for x0 in guesses[:6]:
        try:
            x0 = np.clip(np.asarray(x0, float), lo, hi)
            sol = least_squares(residuals, x0, bounds=(lo, hi), x_scale="jac",
                                xtol=1e-13, ftol=1e-13, gtol=1e-13,
                                max_nfev=1200)
            norm = np.linalg.norm(residuals(sol.x), ord=np.inf)
            if norm < best_norm:
                best_norm = norm
                best_x = sol.x
        except Exception as exc:
            diagnostics.note("dc.bounded_lsq_fail", exc)
    if best_x is not None and best_norm < tol:
        return best_x
    return None


def symmetric_seed(sizes, bias, Id, gmin, seeds=None):
    """AFE-specific symmetric DC solve (matched halves: VON=VOP, VFBN=VFBP). 4 unknowns
    [net2, vop, vfb, net20]. Used as a POST-PROCESS guard: when the full 6-node solve
    latches to a symmetry-broken root (cross-coupled positive feedback) in a no-mismatch
    case, re-solving this symmetric system — seeded from the symmetrized average of the
    latched solution — recovers the physical symmetric branch that Spectre finds.
    Returns a {node:V} dict, or None."""
    from scipy.optimize import fsolve
    VDD, VCM, VB, VC = bias["VDD"], bias["VCM"], bias["VB"], bias["VC"]

    def f(u):
        net2, vop, vfb, net20 = u
        return [
            Id("M6", VDD, net2, VB) - 2 * Id("M7", net2, vop, VCM) - net2 * gmin,
            Id("M7", net2, vop, VCM) - Id("M9", vop, 0.0, vfb) - vop * gmin,
            Id("M13", net20, vfb, vop) - Id("M15", vfb, 0.0, 0.0) - vfb * gmin,
            Id("M11", VDD, net20, VC) - 2 * Id("M12", net20, vfb, vop) - net20 * gmin,
        ]
    trials = list(seeds or []) + [[VCM + 6, VCM - 1, 6.0, bias["VDD"] - 2],
                                  [VCM + 7, VCM - 4, max(VCM - 25, 4.0), VCM + 8],
                                  [VCM + 7, VCM - 4, VCM - 8, VCM + 15],
                                  [VCM + 9, VCM - 2, VCM - 10, VCM + 12],
                                  [VCM + 5, VCM - 6, VCM - 6, VCM + 18]]
    for u0 in trials:
        try:
            sol, _, ier, _ = fsolve(f, u0, full_output=True, xtol=1e-12, maxfev=4000)
            residual_norm = np.linalg.norm(f(sol), ord=np.inf)
            in_box = all(-0.5 <= v <= VDD + 0.5 for v in sol)
            if in_box and residual_norm < DC_FALLBACK_TOL:
                n2, vop, vfb, n20 = sol
                return {"VOP": vop, "VON": vop, "VFBP": vfb, "VFBN": vfb,
                        "NET20": n20, "NET2": n2}
        except Exception as exc:
            diagnostics.note("dc.symmetric_seed_fail", exc)
    return None


def symmetric_continuation(sizes, bias, Id, gmin):
    """AFE symmetric DC via SOURCE-RAMP continuation (power-up homotopy): scale all
    rails 0->1 and track the solution from the powered-down state. This follows the
    same physical branch Spectre's pseudo-transient/gmin-stepping converges to, so it
    selects the correct equilibrium even when the symmetric circuit is multistable
    (a 'normal-on' branch vs a degenerate near-off branch). 4 unknowns: net2,vop,vfb,net20.
    Returns a symmetric {node:V} dict (used as the PRIMARY DC seed), or None."""
    from scipy.optimize import fsolve
    VDD, VCM, VB, VC = bias["VDD"], bias["VCM"], bias["VB"], bias["VC"]

    def f(u, sc):
        net2, vop, vfb, net20 = u
        Vdd, Vcm, Vb, Vc = VDD * sc, VCM * sc, VB * sc, VC * sc
        return [
            Id("M6", Vdd, net2, Vb) - 2 * Id("M7", net2, vop, Vcm) - net2 * gmin,
            Id("M7", net2, vop, Vcm) - Id("M9", vop, 0.0, vfb) - vop * gmin,
            Id("M13", net20, vfb, vop) - Id("M15", vfb, 0.0, 0.0) - vfb * gmin,
            Id("M11", Vdd, net20, Vc) - 2 * Id("M12", net20, vfb, vop) - net20 * gmin,
        ]
    def track(seed_sets):
        u = np.array([VCM, VCM, VCM, VCM]) * 0.1      # near powered-down
        for sc in np.linspace(0.1, 1.0, 19):
            ok = False
            for seed in seed_sets(u, sc):
                try:
                    s, _, ier, _ = fsolve(lambda z: f(z, sc), seed,
                                          full_output=True, xtol=1e-12, maxfev=4000)
                    if dc_residual_ok(lambda z: f(z, sc), s, tol=DC_FALLBACK_TOL):
                        u = s; ok = True; break
                except Exception as exc:
                    diagnostics.note("dc.continuation_step_fail", exc)
            if not ok:
                return None
        n2, vop, vfb, n20 = u
        return {"VOP": vop, "VON": vop, "VFBP": vfb, "VFBN": vfb,
                "NET20": n20, "NET2": n2}

    original = track(lambda u, sc: (u, np.array([VCM+7, VCM-4, VCM-8, VCM+15]) * sc))
    if original is not None:
        return original

    def low_vfb_seeds(u, sc):
        return (
            u,
            np.array([VCM + 6, VCM - 1, 6.0, VDD - 2]) * sc,
            np.array([VCM + 7, VCM - 4, max(VCM - 25, 4.0), VCM + 8]) * sc,
        )
    return track(low_vfb_seeds)
