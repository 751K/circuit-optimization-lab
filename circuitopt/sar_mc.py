"""Per-instance mismatch Monte-Carlo for the FreePDK45 SAR ADC workflow.

The local-solver mismatch MC in :mod:`circuitopt.corners` perturbs OTFT model
params (``mvt0``/``mbeta0``) inside circuitopt's own device model; it never
reaches the silicon path, where the transistors are BSIM4 card devices. This
module fills that gap for the SAR ADC: it draws two families of
per-instance variation, feeds them into the closed-loop conversion, and reports
static-linearity yield.

Two mismatch sources, matching how a real CDAC SAR fails to hit its codes:

  * **Transistor Vth** — a per-device threshold offset injected as the BSIM4
    instance parameter ``delvto`` (a positive offset raises Vth and cuts drain
    current). Comparator input-pair Vth
    mismatch is the dominant SAR offset/first-transition error, so this is the
    knob that moves DNL/INL. Sigma follows Pelgrom's area law
    ``sigma_vth0 / sqrt(W*L / (w0*l0))`` — bigger devices average out local
    fluctuations — with an optional per-polarity override (nMOS/pMOS A_Vth differ).

  * **CDAC unit capacitors** — a per-capacitor relative perturbation
    ``sigma_cu / sqrt(C / c_unit)``: a binary-weighted cap is physically N unit
    caps in parallel, so its matching improves as ``sqrt(N) = sqrt(C/c_unit)``.
    This is applied by perturbing a *copy* of the spec's topology, never the
    loaded spec, so trials stay independent and the caller's spec is untouched.

Each trial runs the code-center input sweep
``(arange(2**n_bits) + 0.5) / 2**n_bits * vref`` — the input that lands one sample
squarely in every code bin — and records ``max_abs_dnl``, ``max_abs_inl``,
``missing_codes`` and the first-transition offset. The summary adds
mean/std/worst and a yield fraction against configurable DNL/INL limits.
"""
from __future__ import annotations

import copy
import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Mapping

import numpy as np

from .adc import static_ramp_metrics
from .circuit_loader import CircuitSpec
from .sar import _sar_config, run_sar_conversion


def _mismatch_config(spec: CircuitSpec, override: Mapping[str, Any] | None = None) -> dict:
    """Resolve the ``adc.mismatch`` block + function-arg override into a flat dict.

    Mirrors :func:`circuitopt.sar._sar_config`'s override precedence (JSON block is
    the base, the ``config`` argument wins) so the CLI/service and a direct call
    share one config surface. All sigmas default to ``0.0`` — an all-zero config
    reproduces the nominal conversion, which is what the zero-sigma regression test
    relies on.

    Keys
    ----
    sigma_vth0 : A_Vth-style Vth sigma [V] at the reference area ``w0*l0``.
    sigma_vth0_nmos / sigma_vth0_pmos : optional per-polarity overrides of the above.
    w0, l0 : reference W/L [um] for the Pelgrom area scaling (default 1.0/0.05,
             the example's switch geometry). Only the product matters.
    sigma_cu : unit-cap relative sigma (dimensionless) at capacitance ``c_unit``.
    c_unit : reference unit capacitance [F] for the cap area scaling.
    dnl_threshold / inl_threshold : |DNL|/|INL| yield limits in LSB (default 0.5).
    """
    cfg = dict((spec.adc or {}).get("mismatch") or {})
    cfg.update(override or {})
    out = {
        "sigma_vth0": float(cfg.get("sigma_vth0", 0.0)),
        "w0": float(cfg.get("w0", 1.0)),
        "l0": float(cfg.get("l0", 0.05)),
        "sigma_cu": float(cfg.get("sigma_cu", 0.0)),
        "c_unit": float(cfg.get("c_unit", 1e-14)),
        "dnl_threshold": float(cfg.get("dnl_threshold", 0.5)),
        "inl_threshold": float(cfg.get("inl_threshold", 0.5)),
    }
    out["sigma_vth0_nmos"] = float(cfg.get("sigma_vth0_nmos", out["sigma_vth0"]))
    out["sigma_vth0_pmos"] = float(cfg.get("sigma_vth0_pmos", out["sigma_vth0"]))
    if min(out["w0"], out["l0"], out["c_unit"]) <= 0.0:
        raise ValueError("mismatch w0/l0/c_unit must be positive")
    if min(out["sigma_vth0"], out["sigma_vth0_nmos"], out["sigma_vth0_pmos"],
           out["sigma_cu"]) < 0.0:
        raise ValueError("mismatch sigmas must be non-negative")
    return out


def _device_polarity(spec: CircuitSpec, name: str) -> str | None:
    """``'nmos'``/``'pmos'`` for a FreePDK45 transistor, else ``None`` (skip it)."""
    mt = str((spec.model_types or {}).get(name, ""))
    if not mt.startswith("freepdk45."):
        return None
    return mt.rsplit(".", 1)[-1]


def draw_device_mismatch(spec: CircuitSpec, rng: np.random.Generator,
                         mcfg: Mapping[str, Any]) -> dict[str, float]:
    """Per-transistor ``delvto`` [V] draw for one trial (area-scaled Pelgrom Vth).

    Only FreePDK45-bound transistors get an offset; anything else is left nominal.
    sigma = ``sigma_vth0[_pol] / sqrt(W*L / (w0*l0))``.
    """
    ref_area = mcfg["w0"] * mcfg["l0"]
    offsets: dict[str, float] = {}
    for name, *_ in spec.topology.devices:
        polarity = _device_polarity(spec, name)
        if polarity is None:
            continue
        sigma0 = mcfg[f"sigma_vth0_{polarity}"]
        if sigma0 <= 0.0:
            continue
        W, L = spec.sizes[name]
        sigma = sigma0 / np.sqrt(max(float(W) * float(L), 1e-30) / ref_area)
        offsets[name] = float(rng.normal(0.0, sigma))
    return offsets


def _copy_with_capacitors(spec: CircuitSpec, new_caps) -> CircuitSpec:
    """Return a spec whose topology's ``capacitors`` list is rebound to *new_caps*.

    The topology is shallow-copied and only its ``capacitors`` field is replaced, so
    the caller's spec — and any other trial/candidate sharing it — is unaffected.
    Shared by mismatch MC (:func:`perturb_capacitors`) and the SAR design-space
    explorer's ``C:`` targets so the "never mutate the loaded spec" rule has one
    implementation."""
    topo = copy.copy(spec.topology)
    topo.capacitors = list(new_caps)
    return dataclasses.replace(spec, topology=topo)


def perturb_capacitors(spec: CircuitSpec, rng: np.random.Generator,
                       mcfg: Mapping[str, Any]) -> CircuitSpec:
    """Return a spec copy whose CDAC capacitors carry a per-instance perturbation.

    Relative sigma per cap is ``sigma_cu / sqrt(C / c_unit)``. The topology is
    shallow-copied and only its ``capacitors`` list is rebound, so the caller's
    spec — and every other trial — is unaffected. With ``sigma_cu == 0`` the copy
    is value-identical to the input.
    """
    sigma_cu = mcfg["sigma_cu"]
    if sigma_cu <= 0.0:
        return spec
    c_unit = mcfg["c_unit"]
    new_caps = []
    for name, a, b, value in spec.topology.capacitors:
        rel = sigma_cu / np.sqrt(max(float(value), 1e-30) / c_unit)
        # Clamp at a floor so a deep negative tail can't flip the cap sign; a
        # SAR cap physically cannot go non-positive.
        factor = max(1.0 + float(rng.normal(0.0, rel)), 1e-3)
        new_caps.append((name, a, b, float(value) * factor))
    return _copy_with_capacitors(spec, new_caps)


def _trial_metrics(spec: CircuitSpec, vin: np.ndarray, cfg: dict,
                   corner: str | None, delvto: Mapping[str, float]) -> dict:
    """One code-center sweep -> per-trial linearity row.

    Reuses :func:`run_sar_conversion` (the same machinery as ``run_sar_sweep``) but
    guards :func:`static_ramp_metrics`, which requires monotonic codes: under heavy
    mismatch a SAR can go non-monotonic, and that trial must be scored as a failure
    rather than crash the sweep.
    """
    n_bits = cfg["n_bits"]
    levels = 1 << n_bits
    codes = np.array(
        [run_sar_conversion(spec, float(v), config=cfg, corner=corner,
                            mismatch=delvto)["code"] for v in vin],
        dtype=np.int64)
    present = np.unique(codes)
    missing = int(levels - present.size)
    monotonic = bool(np.all(np.diff(codes) >= 0))
    row = {"codes": codes, "missing_codes": missing, "monotonic": monotonic}
    if monotonic:
        m = static_ramp_metrics(vin, codes, n_bits, vmin=0.0, vmax=cfg["vref"])
        row["max_abs_dnl"] = float(m["max_abs_dnl"])
        row["max_abs_inl"] = float(m["max_abs_inl"])
        # First-transition INL == offset in LSB (NaN if that transition never fired).
        inl = m["inl"]
        row["offset_lsb"] = float(inl[0]) if inl.size else np.nan
    else:
        # Non-monotonic: linearity is undefined; mark worst so the trial fails yield.
        row["max_abs_dnl"] = np.inf
        row["max_abs_inl"] = np.inf
        row["offset_lsb"] = np.nan
    return row


def _summ(values: np.ndarray) -> dict:
    """mean/std/worst (max |.|) over the finite entries of ``values``."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"mean": float("nan"), "std": float("nan"), "worst": float("nan")}
    return {"mean": float(finite.mean()), "std": float(finite.std()),
            "worst": float(np.max(np.abs(finite)))}


def sar_mismatch_mc(spec: CircuitSpec, *, n: int = 50, seed: int = 0,
                    corner: str | None = None, config: Mapping[str, Any] | None = None,
                    progress: Callable[[int, int, dict], None] | None = None,
                    workers: int = 1) -> dict:
    """Per-instance mismatch MC of a FreePDK45 SAR ADC's static linearity.

    Each of ``n`` trials draws transistor ``delvto`` offsets and CDAC capacitor
    perturbations from the resolved ``adc.mismatch`` config (``config`` overrides
    the JSON block), runs the code-center sweep, and records ``max_abs_dnl``,
    ``max_abs_inl``, ``missing_codes`` and the first-transition ``offset_lsb``.

    Returns ``{"rows": [...], "arrays": {...}, "summary": {...}, "config": {...}}``.
    ``summary`` carries per-metric mean/std/worst plus ``yield`` — the fraction of
    trials meeting |DNL| and |INL| limits with no missing codes. Trials that go
    non-monotonic count against the yield (their DNL/INL are recorded as ``inf``).

    ``progress(i, n, partial)`` — optional callback fired after each trial with the
    1-based completed count, the total, and the running summary (same shape as the
    final ``summary``), mirroring :func:`circuitopt.corners.mismatch_mc` for service
    integration. ``None`` disables it.

    ``workers`` runs the independent trials across a thread pool (the work is
    ngspice-subprocess-bound; ``run_sar_conversion`` has no shared mutable state).
    To keep results seed-deterministic *regardless of worker count*, ALL trials'
    random draws are taken up front, in trial order, from the single seeded RNG —
    so the RNG stream never depends on completion order. ``workers=1`` keeps the
    exact serial order; for ``workers>1`` the trials still land in ``rows`` by trial
    index (deterministic final result), but the ``progress`` callback fires in
    *completion* order — its running summary aggregates whichever trials have
    finished, so per-trial completion order is not deterministic while the final
    result is. Same ``seed`` -> identical draws/results for any worker count.
    """
    if n < 1:
        raise ValueError("n must be at least one")
    if workers is None or workers < 1:
        raise ValueError("workers must be a positive integer")
    cfg = _sar_config(spec, config)          # validates the SAR block, resolves timing
    mcfg = _mismatch_config(spec, config)
    n_bits = cfg["n_bits"]
    levels = 1 << n_bits
    vin = (np.arange(levels) + 0.5) / levels * cfg["vref"]
    rng = np.random.default_rng(seed)

    # Draw both families for every trial up front, in trial order, so the RNG stream
    # is fixed by the seed alone — identical to the old interleaved serial loop, and
    # independent of the order in which parallel workers later complete.
    draws = []
    for _ in range(n):
        delvto = draw_device_mismatch(spec, rng, mcfg)
        trial_spec = perturb_capacitors(spec, rng, mcfg)
        draws.append((delvto, trial_spec))

    def _run_trial(i: int) -> dict:
        delvto, trial_spec = draws[i]
        row = _trial_metrics(trial_spec, vin, cfg, corner, delvto)
        row["trial"] = i
        return row

    rows: list[dict | None] = [None] * n
    if workers == 1:
        # Exact serial path: evaluate in trial order, fire progress with the running
        # summary over the trials completed so far (byte-identical to the old loop).
        ordered: list[dict] = []
        for i in range(n):
            row = _run_trial(i)
            rows[i] = row
            ordered.append(row)
            if progress is not None:
                progress(i + 1, n, _summarize(ordered, mcfg))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_run_trial, i) for i in range(n)]
            done: list[dict] = []
            for completed, fut in enumerate(as_completed(futures), start=1):
                row = fut.result()
                rows[row["trial"]] = row          # final result stays in trial order
                done.append(row)
                if progress is not None:
                    # Monotonic completed-count; summary aggregates finished trials.
                    progress(completed, n, _summarize(done, mcfg))
    rows = [row for row in rows if row is not None]
    return {"rows": rows, "arrays": _arrays(rows), "summary": _summarize(rows, mcfg),
            "config": mcfg}


def _arrays(rows: list[dict]) -> dict:
    return {k: np.array([r[k] for r in rows], dtype=float)
            for k in ("max_abs_dnl", "max_abs_inl", "offset_lsb", "missing_codes")}


def _summarize(rows: list[dict], mcfg: Mapping[str, Any]) -> dict:
    """Aggregate accumulated ``rows`` into the summary payload (reused for progress)."""
    arr = _arrays(rows)
    passed = ((arr["max_abs_dnl"] <= mcfg["dnl_threshold"]) &
              (arr["max_abs_inl"] <= mcfg["inl_threshold"]) &
              (arr["missing_codes"] == 0))
    return {
        "n": len(rows),
        "max_abs_dnl": _summ(arr["max_abs_dnl"]),
        "max_abs_inl": _summ(arr["max_abs_inl"]),
        "offset_lsb": _summ(arr["offset_lsb"]),
        "missing_codes": {"mean": float(arr["missing_codes"].mean()),
                          "worst": float(arr["missing_codes"].max())},
        "monotonic_rate": float(np.mean([r["monotonic"] for r in rows])),
        "yield": float(passed.mean()),
        "dnl_threshold": mcfg["dnl_threshold"],
        "inl_threshold": mcfg["inl_threshold"],
    }
