"""Process corners, mismatch Monte-Carlo, and latch detection — local-solver side.

Single source of truth for the things that were re-derived over and over during
corner / robustness work, so search, verification and MC all agree:

  * CORNERS        — global process shifts (pvt0 = ±3σ, pbeta0 = ±15σ, from the
                     PDK monte.scs sections).
  * mismatch_corner — per-device random mvt0/mbeta0 on top of a process corner.
  * latch_kick_corner — a deterministic ±kσ DIFFERENTIAL mismatch that pushes each
                     symmetric pair apart; a cheap screen for the cross-coupled
                     positive-feedback latch-up (one solve instead of a full MC).
  * metrics        — evaluate one design at one corner -> gain/BW/IRN + latch_dV.
  * corner_table   — metrics across typ/slow/fast.
  * mismatch_mc    — per-device mismatch MC at one corner, seeded from the nominal op.

This module drives the local Python solvers; Cadence/Spectre comparison should
live in dedicated verification scripts instead of the core solver package.
"""
import itertools
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import wraps

import numpy as np

from .ac_solver import ac_solve
from .circuit_loader import circuit_from_dict
from .device_factory import CORNERS
from .noise_solver import band_rms, noise_analysis
from .topology import AFE_TOPO
from ._engine import current_engine
from . import diagnostics

# Per-device mismatch sigmas: Vth (area-scaled inside the model) and beta (flat).
SIGMA_MVT0 = 1.27e-5
SIGMA_MBETA0 = 0.019
# AFE differential pairs — used to drive the latch screen.
AFE_PAIRS = (("M7", "M8"), ("M9", "M10"), ("M12", "M13"), ("M14", "M15"))

_DEFAULT_FREQS = np.logspace(-2, 4, 121)


def _root_sensitive_otft_reference_context(function):
    """Preserve the calibrated root choice for bifurcation-edge OTFT screens."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if current_engine() != "rust":
            return function(*args, **kwargs)
        from .pmos_tft_model import rust_otft_reference_mode

        with rust_otft_reference_mode():
            return function(*args, **kwargs)

    return wrapped


def _base(corner):
    return CORNERS[corner] if isinstance(corner, str) else dict(corner)


def mismatch_corner(rng, devices, base="typical"):
    """Per-device corner map: process `base` + random mvt0/mbeta0 on each device."""
    b = _base(base)
    return {d: {**b, "mvt0": float(rng.normal(0, SIGMA_MVT0)),
                "mbeta0": float(rng.normal(0, SIGMA_MBETA0))} for d in devices}


def latch_kick_corner(base="slow", pairs=AFE_PAIRS, k=3.0, signs=None):
    """A DIFFERENTIAL ±kσ mismatch on each symmetric pair. `signs` is a ±1 per pair
    (default all +1). NOTE: which sign pattern triggers the cross-coupled latch
    varies by design — screen with `latch_screen`, which scans all patterns, not a
    single kick (a single direction has false negatives)."""
    b = _base(base)
    signs = signs if signs is not None else (1,) * len(pairs)
    c = {d: {**b, "mvt0": 0.0, "mbeta0": 0.0} for p in pairs for d in p}
    for (hi, lo), sg in zip(pairs, signs):
        c[hi] = {**b, "mvt0": +sg * k * SIGMA_MVT0, "mbeta0": +sg * k * SIGMA_MBETA0}
        c[lo] = {**b, "mvt0": -sg * k * SIGMA_MVT0, "mbeta0": -sg * k * SIGMA_MBETA0}
    return c


@_root_sensitive_otft_reference_context
def latch_screen(sizes, bias, nf=None, base="slow", topo=AFE_TOPO, k=3.0,
                 pairs=AFE_PAIRS, x0_guess=None, freqs=None):
    """Worst-case differential-mismatch latch screen. Pushes each symmetric pair
    ±kσ apart over ALL sign patterns and returns the largest output imbalance
    |out+ - out-|. Small => robust against the regenerative latch; large =>
    latch-prone. Deterministic — a cheap, reliable screen to use inside a search
    instead of a full per-candidate mismatch MC.

    Each sign pattern is solved twice more from a *split seed* (the neutral op
    with the two outputs pulled apart by half a rail): if a latched solution
    exists, the seeded Newton lands in it regardless of which basin the neutral
    solve happens to hit, so detection does not depend on floating-point luck
    (x86 vs arm64 rounding was observed to flip the neutral solve's basin);
    a monostable design returns to the symmetric op from any seed."""
    worst = 0.0
    split = 0.5 * max((abs(float(v)) for v in bias.values()), default=1.0)
    outs = topo.outputs
    for combo in itertools.product((1, -1), repeat=len(pairs) - 1):
        kick = latch_kick_corner(base, pairs, k, (1,) + combo)
        m = metrics(sizes, bias, nf=nf, corner=kick,
                    topo=topo, x0_guess=x0_guess, freqs=freqs,
                    include_noise=False)
        if m is None:
            continue
        worst = max(worst, m["latch_dV"])
        if len(outs) == 2:
            op = m["dc_op"]
            for sgn in (1.0, -1.0):
                seeded = {**op, outs[0]: op[outs[0]] + sgn * split,
                          outs[1]: op[outs[1]] - sgn * split}
                m2 = metrics(sizes, bias, nf=nf, corner=kick,
                             topo=topo, x0_guess=seeded, freqs=freqs,
                             include_noise=False)
                if m2 is not None:
                    worst = max(worst, m2["latch_dV"])
    return worst


def metrics(sizes, bias, nf=None, corner=None, topo=AFE_TOPO, x0_guess=None,
            freqs=None, band=(0.05, 100.0), include_noise=True,
            noise_gate=None):
    """Evaluate one design at one corner. Returns a dict with:
        gain_peak_dB, bw_Hz, irn_uV, latch_dV (|out+ - out-| at the DC op;
        large => regenerative latch), and dc_op. None if the DC solve fails.

    Noise is optional because latch/gain/BW screens only need the AC/DC result.
    `noise_gate(out)` can defer IRN until after AC/latch checks, e.g. mismatch MC
    skips IRN for latched samples that are excluded from final stats."""
    if freqs is None:
        freqs = _DEFAULT_FREQS
    ac = ac_solve(sizes, bias, freqs, corner=corner, nf=nf, topo=topo, x0_guess=x0_guess)
    if ac is None:
        return None
    out = {"gain_peak_dB": float(ac["peak_dB"]), "bw_Hz": float(ac["bw_Hz"]),
           "dc_op": ac["dc_op"]}
    outs = topo.outputs
    out["latch_dV"] = (abs(ac["dc_op"][outs[0]] - ac["dc_op"][outs[1]])
                       if len(outs) == 2 else 0.0)
    out["irn_uV"] = float("nan")
    out["_noise_evaluated"] = False
    if include_noise and (noise_gate is None or noise_gate(out)):
        try:
            nz = noise_analysis(sizes, bias, freqs, corner=corner, nf=nf, topo=topo,
                                x0_guess=ac["dc_op"])
            out["irn_uV"] = band_rms(freqs, nz["irn_psd"], *band) * 1e6 if nz else float("nan")
            out["_noise_evaluated"] = True
        except Exception as exc:
            diagnostics.note("corners.irn_eval_fail", exc)
            out["irn_uV"] = float("nan")
    return out


def corner_table(sizes, bias, nf=None, topo=AFE_TOPO,
                 corners=("typical", "slow", "fast"), freqs=None, band=(0.05, 100.0),
                 include_noise=True, workers=1):
    """Evaluate a design across process corners -> {corner: metrics-or-None}."""
    if workers is None or workers < 1:
        raise ValueError("workers must be a positive integer")
    corner_names = tuple(corners)
    corner_values = tuple(CORNERS[c] for c in corner_names)

    def evaluate_corner(corner):
        return metrics(sizes, bias, nf=nf, corner=corner, topo=topo,
                       freqs=freqs, band=band, include_noise=include_noise)

    if workers == 1:
        values = map(evaluate_corner, corner_values)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            values = list(executor.map(evaluate_corner, corner_values))
    return dict(zip(corner_names, values))


def _mc_summary(rows, latch_dV, noise_evaluated, *, stopped_early=False):
    """Build the MC summary/arrays payload from accumulated per-sample ``rows``.

    Shared by the completed and the early-stopped return paths so both emit the
    identical structure (arrays / latched / summary); ``stopped_early`` only adds
    the extra flag on the summary. Kept separate so cooperative cancellation can
    reuse the exact same stats over however many samples have finished."""
    arr = {k: np.asarray(v, float) for k, v in rows.items()}
    latched = arr["latch_dV"] > latch_dV
    good = ~latched
    summary = {"n": int(arr["gain_peak_dB"].size), "latched": int(latched.sum()),
               "latch_rate": float(latched.mean()) if latched.size else 0.0,
               "noise_evaluated": int(noise_evaluated)}
    for k in ("gain_peak_dB", "bw_Hz", "irn_uV"):
        col = arr[k][good]
        if col.size:
            summary[k] = {"mean": float(col.mean()), "std": float(col.std()),
                          "p5": float(np.percentile(col, 5)),
                          "p95": float(np.percentile(col, 95))}
    if stopped_early:
        summary["stopped_early"] = True
    out = {"arrays": arr, "latched": latched, "summary": summary}
    if stopped_early:
        out["stopped_early"] = True
    return out


def mismatch_mc(sizes, bias, nf=None, topo=AFE_TOPO, base="slow", n=300, seed=0,
                latch_dV=5.0, freqs=None, band=(0.05, 100.0), include_noise=True,
                progress=None, should_stop=None, workers=1):
    """Per-device mismatch MC at one process corner, seeded from the nominal op.

    Returns {"arrays": {metric: ndarray}, "latched": bool ndarray, "summary": ...}.
    A run is "latched" when latch_dV exceeds the threshold; summary stats are over
    the non-latched runs (mean/std/P5/P95) plus the latch_rate.

    ``progress(i, n, partial)`` — optional callback fired after each of the ``n``
    samples finishes: ``i`` is the 1-based sample index just completed, ``n`` the
    total requested, and ``partial`` a lightweight running summary dict (the same
    shape as the final ``summary``, over the samples done so far) so a UI can show
    live stats. Default ``None`` disables the callback.
    ``should_stop()`` — optional zero-arg predicate checked *before* each sample;
    returning ``True`` finishes early and returns the stats over the samples already
    completed, with ``"stopped_early": True`` added at both the top level and in the
    summary. Cancellation is cooperative: a sample already in flight runs to
    completion before the next check. Default ``None`` never stops.

    ``workers`` evaluates independent samples concurrently. Random mismatch maps
    are drawn up front in sample order and final rows are reduced in that same
    order, so a fixed seed produces identical final results for every worker count.
    Progress callbacks run on the caller thread with a monotonic completed count.

    With ``workers=1``, ``progress=None`` and ``should_stop=None`` the result is
    byte-for-byte identical to the pre-hook behaviour."""
    if workers is None or workers < 1:
        raise ValueError("workers must be a positive integer")
    if freqs is None:
        freqs = _DEFAULT_FREQS
    devices = [d for d, *_ in topo.devices]
    rng = np.random.default_rng(seed)
    nom = ac_solve(sizes, bias, freqs, corner=_base(base), nf=nf, topo=topo)
    if nom is None:
        raise RuntimeError(f"nominal {base!r} DC solve failed; cannot seed MC")
    x0 = nom["dc_op"]
    keys = ("gain_peak_dB", "bw_Hz", "irn_uV", "latch_dV")
    rows = {k: [] for k in keys}
    noise_evaluated = 0
    def evaluate_sample(cm):
        return metrics(sizes, bias, nf=nf, corner=cm, topo=topo, x0_guess=x0,
                       freqs=freqs, band=band, include_noise=include_noise,
                       noise_gate=lambda out: out["latch_dV"] <= latch_dV)

    if workers == 1:
        for i in range(n):
            if should_stop is not None and should_stop():
                return _mc_summary(rows, latch_dV, noise_evaluated, stopped_early=True)
            m = evaluate_sample(mismatch_corner(rng, devices, base=base))
            if m is not None:
                noise_evaluated += int(m.get("_noise_evaluated", False))
                for k in keys:
                    rows[k].append(m[k])
            if progress is not None:
                partial = _mc_summary(rows, latch_dV, noise_evaluated)["summary"]
                progress(i + 1, n, partial)
        return _mc_summary(rows, latch_dV, noise_evaluated)

    # Freeze the RNG stream before scheduling. Worker completion order therefore
    # cannot perturb later draws or the final sample ordering.
    draws = [mismatch_corner(rng, devices, base=base) for _ in range(n)]
    results = [None] * n
    completed = 0
    next_index = 0
    stopped_early = False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}

        def submit_available():
            nonlocal next_index, stopped_early
            while next_index < n and len(pending) < workers:
                if should_stop is not None and should_stop():
                    stopped_early = True
                    return
                future = executor.submit(evaluate_sample, draws[next_index])
                pending[future] = next_index
                next_index += 1

        submit_available()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(finished, key=lambda item: pending[item]):
                index = pending.pop(future)
                results[index] = future.result()
                completed += 1
                finished_metrics = [item for item in results if item is not None]
                if progress is not None:
                    partial_rows = {k: [item[k] for item in finished_metrics] for k in keys}
                    partial_noise = sum(int(item.get("_noise_evaluated", False))
                                        for item in finished_metrics)
                    partial = _mc_summary(partial_rows, latch_dV,
                                          partial_noise)["summary"]
                    progress(completed, n, partial)
            if not stopped_early:
                submit_available()

    for m in results:
        if m is not None:
            noise_evaluated += int(m.get("_noise_evaluated", False))
            for k in keys:
                rows[k].append(m[k])
    stopped_early = stopped_early or next_index < n
    return _mc_summary(rows, latch_dV, noise_evaluated,
                       stopped_early=stopped_early)


def mismatch_mc_from_dict(data, n=300, seed=0, corner="typical", freqs=None,
                          band=(0.05, 100.0), progress=None, should_stop=None,
                          workers=1):
    """Run a mismatch MC from a parsed circuit-JSON *dict*. Returns the
    :func:`mismatch_mc` result dict.

    The shared entry point for `circuit-opt mc` (via :meth:`__main__._cmd_mc`) and
    the service's ``POST /api/v1/jobs/mc`` — both parse the circuit and call
    :func:`mismatch_mc` through here, so the two surfaces can't drift. ``corner``
    is the base process corner (typical/slow/fast). ``progress``/``should_stop``
    are threaded straight through for the background-job hooks."""
    spec = circuit_from_dict(data)
    return mismatch_mc(spec.sizes, spec.bias, nf=spec.nf, topo=spec.topology,
                       base=corner, n=n, seed=seed, freqs=freqs, band=band,
                       progress=progress, should_stop=should_stop, workers=workers)
