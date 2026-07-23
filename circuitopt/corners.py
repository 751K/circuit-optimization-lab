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
import dataclasses
import itertools
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import wraps

import numpy as np

from .ac_solver import ac_solve
from ._campaign_sweep import silicon_campaign_for
from .circuit_loader import circuit_from_dict
from .device_factory import CORNERS
from .noise_solver import band_rms, noise_analysis
from .topology import AFE_TOPO
from . import diagnostics

# Per-device mismatch sigmas: Vth (area-scaled inside the model) and beta (flat).
SIGMA_MVT0 = 1.27e-5
SIGMA_MBETA0 = 0.019
# Silicon (BSIM4) per-device threshold-offset (``delvto`` volts) sigma. The silicon
# mismatch draw is the structural mirror of the OTFT ``mvt0`` draw — the same fixed
# threshold-offset sigma and per-device i.i.d. normal structure — applied to the
# BSIM4 ``delvto`` instance knob instead of the OTFT ``mvt0`` model param. The OTFT
# ``mbeta0`` beta-mismatch knob has no single-``delvto`` analog and is omitted. (A
# physical Pelgrom area-law Vth model, for SAR static-linearity yield, lives in
# ``sar_mc.draw_device_mismatch``; this corners-module draw stays aligned with the
# frozen OTFT rule per the R9 brief — no new physics.)
SIGMA_DELVTO = SIGMA_MVT0
# AFE differential pairs — used to drive the latch screen.
AFE_PAIRS = (("M7", "M8"), ("M9", "M10"), ("M12", "M13"), ("M14", "M15"))

_DEFAULT_FREQS = np.logspace(-2, 4, 121)


def _root_sensitive_otft_reference_context(function):
    """Preserve the calibrated root choice for bifurcation-edge OTFT screens."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        from .pmos_tft_model import otft_reference_mode

        with otft_reference_mode():
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
            noise_gate=None, *, binding=None):
    """Evaluate one design at one corner. Returns a dict with:
        gain_peak_dB, bw_Hz, irn_uV, latch_dV (|out+ - out-| at the DC op;
        large => regenerative latch), and dc_op. None if the DC solve fails.

    Noise is optional because latch/gain/BW screens only need the AC/DC result.
    `noise_gate(out)` can defer IRN until after AC/latch checks, e.g. mismatch MC
    skips IRN for latched samples that are excluded from final stats.

    ``binding`` (a :class:`CircuitBinding`) supplies the per-device model map so a
    silicon circuit keeps its BSIM4 devices instead of reverting to the default
    OTFT PDK; ``binding=None`` reproduces the legacy path byte-for-byte. It is the
    frozen scalar reference the silicon compiled-campaign arm is validated against
    and falls back to."""
    if freqs is None:
        freqs = _DEFAULT_FREQS
    ac = ac_solve(sizes, bias, freqs, corner=corner, nf=nf, topo=topo,
                  x0_guess=x0_guess, binding=binding)
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
                                x0_guess=ac["dc_op"], binding=binding)
            out["irn_uV"] = band_rms(freqs, nz["irn_psd"], *band) * 1e6 if nz else float("nan")
            out["_noise_evaluated"] = True
        except Exception as exc:
            diagnostics.note("corners.irn_eval_fail", exc)
            out["irn_uV"] = float("nan")
    return out


def _metrics_from_campaign_row(row, solved, noise_evaluated):
    """A compiled-campaign result row -> the :func:`metrics` dict shape, or ``None``.

    ``None`` when the candidate did not converge (``ok`` False) — the same signal
    :func:`metrics` gives on a failed DC solve, so callers treat a non-converged
    campaign corner/sample exactly like a failed scalar one. ``dc_op`` is rebuilt as
    a ``{node: V}`` map from the solved-order vector; ``latch_dV`` / ``irn_uV`` /
    gains come straight from the row (the campaign computes them 1:1 with the frozen
    reductions). ``noise_evaluated`` mirrors whether the ``"noise"`` analysis ran."""
    if not row.get("ok"):
        return None
    dc_vec = row.get("dc_op") or []
    return {
        "gain_peak_dB": float(row["gain_peak_dB"]),
        "bw_Hz": float(row["bw_Hz"]),
        "dc_op": {node: float(v) for node, v in zip(solved, dc_vec)},
        "latch_dV": float(row["latch_dV"]),
        "irn_uV": float(row["irn_uV"]),
        "_noise_evaluated": bool(noise_evaluated),
    }


def silicon_corner_names(model_types):
    """Default card-corner sweep for a silicon circuit's model family.

    The OTFT ``typical/slow/fast`` names have no silicon card; this maps a silicon
    circuit to the process corners its cards carry:

      * freepdk45 → ``nom/ss/ff/sf/fs`` (5). ``sf``/``fs`` are the cross corners
        (NMOS-slow/PMOS-fast and vice-versa) that reuse the per-polarity ``ss``/``ff``
        model directories, so they always resolve.
      * tsmc28 → ``tt/ss/ff/sf/fs`` (5). All five are sections of the core ``.l``
        delivery; a geometry that selects zero bins in one of them is skipped per
        corner by the :func:`corner_table` pre-probe (None + counted), never crashing
        the sweep — see the 0-bin note in rust/crates/co-pdk/PARITY.md.
      * sky130 → ``tt/ss`` (2). The *bundled* card set only ships tt/ss (+ ff for a
        few widths); this stays the documented data boundary for the in-repo cards.

    A caller that wants a specific corner set passes ``corners=`` to
    :func:`corner_table` directly (as the parity gates do)."""
    from ._rust_campaign import _silicon_pdk_of

    fam = _silicon_pdk_of(model_types)
    if fam == "freepdk45":
        return ("nom", "ss", "ff", "sf", "fs")
    if fam == "tsmc28":
        return ("tt", "ss", "ff", "sf", "fs")
    return ("tt", "ss")   # sky130: bundled-card data boundary


def _is_silicon_binding(binding) -> bool:
    """True iff ``binding`` binds an all/any-silicon circuit (non-empty model_types).

    The gate for the compiled-campaign / silicon-scalar arm. ``binding=None`` or an
    empty ``model_types`` (the AFE OTFT / default-PDK family) stays on the legacy
    scalar path, threaded **without** a binding so it is byte-for-byte unchanged
    (a binding would inject its default DC seed and perturb the cold OTFT solve)."""
    return binding is not None and bool(binding.model_types or {})


def corner_table(sizes, bias, nf=None, topo=AFE_TOPO,
                 corners=("typical", "slow", "fast"), freqs=None, band=(0.05, 100.0),
                 include_noise=True, workers=1, *, binding=None, temps=None):
    """Evaluate a design across process corners -> {corner: metrics-or-None}.

    ``binding`` (a :class:`CircuitBinding`): when it binds a silicon circuit the whole
    corner batch runs through the compiled campaign (one Rayon pool, per-candidate
    corner, ``workers`` scaled), with the frozen scalar :func:`metrics` path as the
    per-corner fallback (any corner the campaign fails to converge rolls back to
    scalar and is flagged). AFE / default-PDK (``binding=None`` or empty model_types)
    keeps the legacy scalar path byte-for-byte — ``corners`` are then OTFT process
    names (``typical/slow/fast``); for silicon they are card corners (``tt/ss/ff/...``).

    ``temps`` (silicon only) adds a **temperature axis** in °C. ``temps=None`` is the
    frozen behaviour — a flat ``{corner: metrics}`` at the device-default 300.15 K,
    byte-for-byte identical to the pre-R10 path. A sequence (e.g. ``(-40, 27, 125)``)
    nests the result as ``{corner: {temp_c: metrics}}``; the temperature rides on the
    frozen silicon-device ``temperature`` ctor kwarg (Kelvin), so both the compiled
    campaign and the scalar reference see it. An OTFT / default-PDK circuit rejects
    ``temps`` (its model has no defined temperature axis)."""
    if workers is None or workers < 1:
        raise ValueError("workers must be a positive integer")
    corner_names = tuple(corners)

    if temps is not None:
        # The temperature axis is silicon-only: the OTFT / default-PDK model has no
        # defined temperature semantics, so it rejects the axis rather than silently
        # ignoring it (and injecting a binding would perturb the cold OTFT solve).
        if not _is_silicon_binding(binding):
            raise ValueError(
                "temps requires an all-silicon binding (BSIM4); the OTFT / default-PDK "
                "family has no defined temperature axis")
        if len(tuple(temps)) == 0:
            raise ValueError("temps must be a non-empty sequence of °C values, or None")
        freqs_eff = _DEFAULT_FREQS if freqs is None else freqs
        return _corner_table_pvt(sizes, bias, nf, topo, corner_names, freqs_eff, band,
                                 include_noise, workers, binding, temps)

    if _is_silicon_binding(binding):
        freqs_eff = _DEFAULT_FREQS if freqs is None else freqs
        camp = silicon_campaign_for(topo, sizes, bias, nf, binding, freqs_eff, band)
        return _corner_table_silicon(camp, sizes, bias, nf, topo, corner_names,
                                     freqs_eff, band, include_noise, workers, binding)

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


def _corner_table_silicon(camp, sizes, bias, nf, topo, corner_names, freqs, band,
                          include_noise, workers, binding):
    """Silicon corner sweep: compiled campaign (``camp``) or the scalar reference.

    ``camp is None`` (campaign unavailable) evaluates every corner through the frozen
    scalar :func:`metrics` path under ``binding`` — the reference the campaign is
    validated against. With a campaign, the whole corner matrix runs in one batch;
    any corner the campaign fails to converge rolls back to that same scalar path and
    is flagged (no silent root substitution).

    **Per-(corner, geometry) 0-bin skip.** A corner whose cards select zero bins for
    this geometry (the tsmc28 ``ff/sf/fs`` sections on some geometries, or an
    out-of-grid width) is recorded as ``None`` and counted rather than crashing the
    whole sweep: both arms reject it identically (the campaign candidate as
    ``{ok: False}``, the scalar path as a PDK ``*ModelError`` — a ``ValueError``
    subclass; see the 0-bin note in rust/crates/co-pdk/PARITY.md). Corners that
    resolve keep the exact prior evaluation path, so a sweep whose corners all resolve
    is byte-for-byte unchanged."""
    def scalar_corner(name):
        # 0-bin corner -> PDK *ModelError (ValueError): skip (None) + count so one
        # unresolvable (corner, geometry) never sinks the sweep. Resolving corners
        # never raise here, so this leaves their result byte-for-byte identical.
        try:
            return metrics(sizes, bias, nf=nf, corner=name, topo=topo, freqs=freqs,
                           band=band, include_noise=include_noise, binding=binding)
        except ValueError as exc:
            diagnostics.note("corners.corner_zero_bin_skip", exc, detail=str(name))
            return None

    if camp is None:
        if workers == 1:
            values = [scalar_corner(c) for c in corner_names]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                values = list(executor.map(scalar_corner, corner_names))
        return dict(zip(corner_names, values))

    analyses = ("dc", "ac", "noise") if include_noise else ("dc", "ac")
    cands = [camp.candidate(sizes, corner=name) for name in corner_names]
    rows = camp.evaluate_batch(cands, workers=workers, analyses=analyses)
    solved = camp.solved
    out = {}
    for name, row in zip(corner_names, rows):
        m = _metrics_from_campaign_row(row, solved, include_noise)
        if m is None:                       # campaign did not converge -> scalar + flag
            diagnostics.note("corners.corner_table_rollback", name)
            m = scalar_corner(name)         # 0-bin here returns None (+ counted)
        out[name] = m
    return out


def _temperature_binding(binding, temp_c, devices):
    """A binding with a uniform device temperature (°C) baked onto every device.

    Reuses the frozen temperature primitive — the silicon device ctor's
    ``temperature`` kwarg (Kelvin) rides on ``device_kwargs`` exactly as
    :func:`_silicon_sample_binding` rides ``delvto`` on it. Both arms then pick it up
    identically: the compiled campaign (``dev.temperature`` -> the template device
    record -> ``CompiledPdk::numeric_card`` card selection for tsmc28 +
    ``co_bsim4::create`` handle temperature) and the scalar ``metrics`` reference (the
    ctor ``temperature`` kwarg -> ``Bsim4Bias.temperature_k``). ``temp_c is None``
    returns ``binding`` unchanged (the frozen 300.15 K device default)."""
    if temp_c is None:
        return binding
    kelvin = float(temp_c) + 273.15
    base_dk = binding.device_kwargs or {}
    dk = {d: {**base_dk.get(d, {}), "temperature": kelvin} for d in devices}
    return dataclasses.replace(binding, device_kwargs=dk)


def _corner_table_pvt(sizes, bias, nf, topo, corner_names, freqs, band,
                      include_noise, workers, binding, temps, vdd_scale=None):
    """PVT grid: nest the silicon corner sweep over the temperature (°C) axis.

    Result shape (documented on :func:`corner_table`): each active axis nests under
    the corner in the fixed order ``[temp_c, vdd_scale]``. With only ``temps`` active
    the shape is ``{corner: {temp_c: metrics}}``.

    Each ``(temp, vdd)`` slice is one compiled campaign over the (scaled) bias and the
    baked device temperature — the R9 dataset-layering precedent: temperature and bias
    are template-baked (only the corner is per-candidate), so a distinct
    temperature/bias means a fresh :func:`silicon_campaign_for`. Every slice runs all
    corners through :func:`_corner_table_silicon`, inheriting its 0-bin per-corner skip
    and non-convergence rollback unchanged."""
    devices = [d for d, *_ in topo.devices]
    temp_axis = tuple(temps) if temps is not None else (None,)
    vdd_axis = tuple(vdd_scale) if vdd_scale is not None else (None,)

    out = {c: {} for c in corner_names}
    for tc in temp_axis:
        tbind = _temperature_binding(binding, tc, devices)
        for vs in vdd_axis:
            sbias = (bias if vs is None
                     else {k: v * float(vs) for k, v in bias.items()})
            camp = silicon_campaign_for(topo, sizes, sbias, nf, tbind, freqs, band)
            tbl = _corner_table_silicon(camp, sizes, sbias, nf, topo, corner_names,
                                        freqs, band, include_noise, workers, tbind)
            for c in corner_names:
                _pvt_place(out[c], tc, vs, tbl[c])
    return out


def _pvt_place(node, tc, vs, value):
    """Place ``value`` in a per-corner nest at the active-axis depth ``[temp, vdd]``.

    ``tc``/``vs`` are the temperature (°C) / supply-scale key, or ``None`` when that
    axis is inactive. The inactive axis is collapsed so a single-axis grid stays a
    one-level nest (``{corner: {temp: m}}`` or ``{corner: {vdd: m}}``) and a two-axis
    grid nests temp-outer, vdd-inner (``{corner: {temp: {vdd: m}}}``)."""
    if tc is not None and vs is not None:
        node.setdefault(tc, {})[vs] = value
    elif tc is not None:
        node[tc] = value
    else:                       # vs is not None (the PVT path always has >=1 axis)
        node[vs] = value


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
                progress=None, should_stop=None, workers=1, *, binding=None):
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
    byte-for-byte identical to the pre-hook behaviour.

    ``binding`` (a :class:`CircuitBinding`): a silicon circuit routes through the
    compiled-campaign silicon arm (:func:`_mismatch_mc_silicon`) — per-device
    ``delvto`` mismatch drawn up front, the whole sample batch in one Rayon pool
    seeded from the shared nominal op, with the frozen scalar ``metrics`` path as
    the reference/fallback. AFE / default-PDK (``binding=None`` or empty
    model_types) keeps this OTFT ``mvt0``/``mbeta0`` path byte-for-byte."""
    if workers is None or workers < 1:
        raise ValueError("workers must be a positive integer")
    if freqs is None:
        freqs = _DEFAULT_FREQS
    if _is_silicon_binding(binding):
        return _mismatch_mc_silicon(sizes, bias, nf, topo, binding, base, n, seed,
                                    latch_dV, freqs, band, include_noise, progress,
                                    should_stop, workers)
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


# ── silicon (BSIM4) mismatch MC — compiled campaign arm + scalar reference ────
_MC_KEYS = ("gain_peak_dB", "bw_Hz", "irn_uV", "latch_dV")


def _silicon_mismatch(rng, devices):
    """Per-device ``delvto`` [V] draw for one sample (silicon mirror of
    :func:`mismatch_corner`): each device gets an i.i.d. ``N(0, SIGMA_DELVTO)``
    threshold offset — the OTFT ``mvt0`` draw's structure on the BSIM4 knob."""
    return {d: float(rng.normal(0.0, SIGMA_DELVTO)) for d in devices}


def _silicon_sample_binding(binding, base, delvto, devices):
    """A per-sample binding: the base card corner baked on + per-device ``delvto``.

    Composes frozen primitives only — ``delvto`` rides on ``device_kwargs`` (the
    sky130/freepdk45/tsmc28 device ``delvto`` ctor knob) and ``at_corner`` bakes the
    card corner exactly as every silicon path does — so the scalar reference applies
    the identical offset the campaign candidate carries."""
    base_dk = binding.device_kwargs or {}
    dk = {d: {**base_dk.get(d, {}), "delvto": float(delvto.get(d, 0.0))}
          for d in devices}
    return dataclasses.replace(binding, device_kwargs=dk).at_corner(base)


def _silicon_mc_row(m, latch_dV):
    """One metrics dict -> (row, noise_counted) with the MC noise gate applied.

    Mirrors the scalar ``noise_gate`` (no IRN for a latched sample) so the campaign
    arrays match the reference sample-for-sample; ``noise_counted`` is 1 only for a
    non-latched sample whose noise was actually evaluated."""
    latched = m["latch_dV"] > latch_dV
    counted = int((not latched) and bool(m.get("_noise_evaluated", False)))
    row = {"gain_peak_dB": m["gain_peak_dB"], "bw_Hz": m["bw_Hz"],
           "irn_uV": float("nan") if latched else m["irn_uV"],
           "latch_dV": m["latch_dV"]}
    return row, counted


def _accumulate_mc(items, latch_dV):
    """Reduce metrics dicts -> ``({key: [values]}, noise_evaluated)`` in order."""
    rows = {k: [] for k in _MC_KEYS}
    noise_evaluated = 0
    for m in items:
        if m is None:
            continue
        row, counted = _silicon_mc_row(m, latch_dV)
        noise_evaluated += counted
        for k in _MC_KEYS:
            rows[k].append(row[k])
    return rows, noise_evaluated


def _mismatch_mc_silicon(sizes, bias, nf, topo, binding, base, n, seed, latch_dV,
                         freqs, band, include_noise, progress, should_stop, workers):
    """Silicon per-device mismatch MC — compiled campaign arm + scalar reference.

    Draws all per-device ``delvto`` up front in sample order (seed-deterministic,
    worker-count independent), seeds every sample from the shared nominal ``base``
    op, and evaluates the batch through the compiled campaign (one Rayon pool, no
    per-candidate Python callback) or — when the campaign is unavailable — the frozen
    scalar ``metrics`` reference. Same summary/arrays shape as :func:`mismatch_mc`;
    ``progress``/``should_stop`` mirror it (the campaign fires ``progress`` post-batch
    in sample order, since no per-candidate Python frame runs during the detached
    batch)."""
    devices = [d for d, *_ in topo.devices]
    rng = np.random.default_rng(seed)
    draws = [_silicon_mismatch(rng, devices) for _ in range(n)]

    nom = ac_solve(sizes, bias, freqs, corner=base, nf=nf, binding=binding)
    if nom is None:
        raise RuntimeError(f"nominal {base!r} DC solve failed; cannot seed silicon MC")
    x0 = nom["dc_op"]
    analyses = ("dc", "ac", "noise") if include_noise else ("dc", "ac")
    camp = silicon_campaign_for(topo, sizes, bias, nf, binding, freqs, band)

    if camp is not None:
        if should_stop is not None and should_stop():
            return _mc_summary({k: [] for k in _MC_KEYS}, latch_dV, 0,
                               stopped_early=True)
        seed_vec = camp.seed_vector(x0)
        cands = [camp.candidate(sizes, corner=base, mismatch=mm, seed=seed_vec,
                                trust_seed_as_op=False) for mm in draws]
        results = camp.evaluate_batch(cands, workers=workers, analyses=analyses)
        metrics_list = [_metrics_from_campaign_row(r, camp.solved, include_noise)
                        for r in results]
        if progress is not None:                # post-batch replay, sample order
            for i in range(n):
                rows, ne = _accumulate_mc(metrics_list[:i + 1], latch_dV)
                progress(i + 1, n, _mc_summary(rows, latch_dV, ne)["summary"])
        rows, noise_evaluated = _accumulate_mc(metrics_list, latch_dV)
        return _mc_summary(rows, latch_dV, noise_evaluated)

    # ── scalar silicon reference (campaign unavailable) ──────────────────────
    def evaluate_sample(mm):
        sb = _silicon_sample_binding(binding, base, mm, devices)
        return metrics(sizes, bias, nf=nf, corner=None, topo=topo, x0_guess=x0,
                       freqs=freqs, band=band, include_noise=include_noise,
                       binding=sb, noise_gate=lambda out: out["latch_dV"] <= latch_dV)

    if workers == 1:
        collected = []
        for i in range(n):
            if should_stop is not None and should_stop():
                rows, ne = _accumulate_mc(collected, latch_dV)
                return _mc_summary(rows, latch_dV, ne, stopped_early=True)
            collected.append(evaluate_sample(draws[i]))
            if progress is not None:
                rows, ne = _accumulate_mc(collected, latch_dV)
                progress(i + 1, n, _mc_summary(rows, latch_dV, ne)["summary"])
        rows, noise_evaluated = _accumulate_mc(collected, latch_dV)
        return _mc_summary(rows, latch_dV, noise_evaluated)

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
                fut = executor.submit(evaluate_sample, draws[next_index])
                pending[fut] = next_index
                next_index += 1

        submit_available()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in sorted(finished, key=lambda it: pending[it]):
                index = pending.pop(fut)
                results[index] = fut.result()
                completed += 1
                if progress is not None:
                    done = [it for it in results if it is not None]
                    prows, pnoise = _accumulate_mc(done, latch_dV)
                    progress(completed, n, _mc_summary(prows, latch_dV, pnoise)["summary"])
            if not stopped_early:
                submit_available()
    rows, noise_evaluated = _accumulate_mc(results, latch_dV)
    stopped_early = stopped_early or next_index < n
    return _mc_summary(rows, latch_dV, noise_evaluated, stopped_early=stopped_early)


def _silicon_base_corner(model_types, name):
    """Map a corner name to a silicon card corner for a silicon circuit.

    The OTFT ``typical/slow/fast`` names have no silicon card; they map to the
    family's ``nominal/ss/ff`` (freepdk45 nominal is ``nom``, sky130/tsmc28 ``tt``).
    A name that is already a silicon corner (``tt/ss/ff/sf/fs``) passes through, so a
    caller can request a specific card corner directly."""
    from ._rust_campaign import _silicon_pdk_of

    nominal = "nom" if _silicon_pdk_of(model_types) == "freepdk45" else "tt"
    key = name.lower() if isinstance(name, str) else name
    return {"typical": nominal, "slow": "ss", "fast": "ff", None: nominal}.get(key, key)


def mismatch_mc_from_dict(data, n=300, seed=0, corner="typical", freqs=None,
                          band=(0.05, 100.0), progress=None, should_stop=None,
                          workers=1):
    """Run a mismatch MC from a parsed circuit-JSON *dict*. Returns the
    :func:`mismatch_mc` result dict.

    The shared entry point for `circuit-opt mc` (via :meth:`__main__._cmd_mc`) and
    the service's ``POST /api/v1/jobs/mc`` — both parse the circuit and call
    :func:`mismatch_mc` through here, so the two surfaces can't drift. ``corner``
    is the base process corner (OTFT typical/slow/fast, or a silicon card corner).
    ``progress``/``should_stop`` are threaded straight through for the background-job
    hooks.

    The circuit's model binding is threaded through, so an all-silicon circuit
    auto-routes to the compiled-campaign silicon arm (its ``corner`` is mapped to the
    matching card corner) while AFE circuits keep the OTFT ``mvt0``/``mbeta0`` path —
    the result contract is unchanged."""
    spec = circuit_from_dict(data)
    binding = spec.binding()
    base = (_silicon_base_corner(binding.model_types, corner)
            if binding.model_types else corner)
    return mismatch_mc(spec.sizes, spec.bias, nf=spec.nf, topo=spec.topology,
                       base=base, n=n, seed=seed, freqs=freqs, band=band,
                       progress=progress, should_stop=should_stop, workers=workers,
                       binding=binding)
