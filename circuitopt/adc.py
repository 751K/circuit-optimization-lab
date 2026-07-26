"""ADC waveform decoding and static/dynamic performance metrics."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _as_1d(value, name: str) -> np.ndarray:
    out = np.asarray(value, float)
    if out.ndim != 1 or len(out) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return out


def decode_bit_waveforms(
    t,
    nodes: Mapping[str, Sequence[float]],
    bit_nodes: Sequence[str],
    sample_times,
    *,
    threshold: float | Mapping[str, float],
    msb_first: bool = True,
) -> dict:
    """Sample digital output nodes and assemble unsigned ADC codes.

    ``bit_nodes`` is MSB-to-LSB by default. A scalar threshold applies to every
    bit; a mapping permits per-node thresholds for asymmetric output stages.
    """
    t = _as_1d(t, "t")
    sample_times = _as_1d(sample_times, "sample_times")
    if len(t) < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be strictly increasing with at least two points")
    if sample_times[0] < t[0] or sample_times[-1] > t[-1]:
        raise ValueError("sample_times must lie inside the transient time range")
    names = tuple(str(name) for name in bit_nodes)
    if not names:
        raise ValueError("bit_nodes must contain at least one output node")
    sampled = np.empty((len(sample_times), len(names)), dtype=np.int8)
    voltages = np.empty_like(sampled, dtype=float)
    for pos, name in enumerate(names):
        if name not in nodes:
            raise ValueError(f"ADC bit node {name!r} is missing from transient results")
        wave = _as_1d(nodes[name], f"nodes[{name!r}]")
        if len(wave) != len(t):
            raise ValueError(f"nodes[{name!r}] length differs from t")
        level = float(threshold[name] if isinstance(threshold, Mapping) else threshold)
        voltages[:, pos] = np.interp(sample_times, t, wave)
        sampled[:, pos] = voltages[:, pos] >= level
    ordered = sampled if msb_first else sampled[:, ::-1]
    weights = 1 << np.arange(len(names) - 1, -1, -1, dtype=np.int64)
    codes = ordered.astype(np.int64) @ weights
    return {
        "sample_times": sample_times,
        "bit_nodes": names,
        "bit_voltages": voltages,
        "bits": sampled,
        "codes": codes,
        "n_bits": len(names),
    }


def static_ramp_metrics(vin, codes, n_bits: int, *, vmin=None, vmax=None) -> dict:
    """Transition-level DNL/INL from a monotonic ramp conversion.

    Missing transitions remain ``NaN`` and are reported explicitly. DNL contains
    one value per output code; INL contains one value per interior transition.
    """
    vin = _as_1d(vin, "vin")
    codes = np.asarray(codes, np.int64)
    if codes.ndim != 1 or len(codes) != len(vin):
        raise ValueError("codes must be one-dimensional and match vin length")
    if np.any(np.diff(vin) <= 0.0):
        raise ValueError("vin must be strictly increasing")
    levels = 1 << int(n_bits)
    if n_bits < 1 or np.any(codes < 0) or np.any(codes >= levels):
        raise ValueError(f"codes must lie in [0, {levels - 1}]")
    if np.any(np.diff(codes) < 0):
        raise ValueError("codes must be monotonic for ramp-based linearity")
    lo = float(vin[0] if vmin is None else vmin)
    hi = float(vin[-1] if vmax is None else vmax)
    if hi <= lo:
        raise ValueError("vmax must be greater than vmin")
    lsb = (hi - lo) / levels
    transitions = np.full(levels - 1, np.nan)
    for code in range(1, levels):
        upper = int(np.searchsorted(codes, code, side="left"))
        if 0 < upper < len(codes):
            transitions[code - 1] = 0.5 * (vin[upper - 1] + vin[upper])
    boundaries = np.concatenate(([lo], transitions, [hi]))
    widths = np.diff(boundaries)
    dnl = widths / lsb - 1.0
    ideal_transitions = lo + lsb * np.arange(1, levels)
    inl = (transitions - ideal_transitions) / lsb
    missing = np.flatnonzero(~np.isfinite(transitions)) + 1
    return {
        "n_bits": int(n_bits),
        "lsb": lsb,
        "transitions": transitions,
        "widths": widths,
        "dnl": dnl,
        "inl": inl,
        "missing_transitions": missing,
        "missing_codes": np.unique(np.concatenate((
            np.flatnonzero(~np.isfinite(widths)),
            np.setdiff1d(np.arange(levels), np.unique(codes)),
        ))).astype(np.int64),
        "max_abs_dnl": float(np.nanmax(np.abs(dnl))) if np.any(np.isfinite(dnl)) else np.nan,
        "max_abs_inl": float(np.nanmax(np.abs(inl))) if np.any(np.isfinite(inl)) else np.nan,
    }


def sampled_transfer_metrics(vin, codes, n_bits: int, *, vmin=None, vmax=None) -> dict:
    """Code-error transfer metrics that stay valid at any sampling density.

    A subsampled ramp cannot resolve individual transitions: a gap of ``g``
    codes between adjacent samples aliases ``g`` boundaries onto one midpoint,
    so :func:`static_ramp_metrics` reads *wrong* DNL/INL there (a perfect 6-bit
    converter sampled at every fourth center scores a multi-LSB ``max_abs_dnl``),
    not merely incomplete ones. What a sparse ramp does measure exactly is the
    signed **code error** at each sample — the produced code minus the code of
    the bin the input lies in — and ``|INL|`` at that sample lies within half
    an LSB of ``|code error|``. This is the screening metric for resolutions
    where the full ``2**n`` ramp is unaffordable.

    Unlike the transition metrics, non-monotonic codes are accepted (the error
    at each sample is defined regardless) and reported through ``monotonic``.
    Missing codes are deliberately NOT reported: a sparse ramp cannot measure
    them — any code can hide between two samples, and any "expected vs
    produced" bookkeeping misfires on a plain offset (a +1-code offset on a
    stride-2 subsample produces codes disjoint from every sampled ideal).
    Callers report missing codes as unmeasured (NaN) below full density.
    """
    vin = _as_1d(vin, "vin")
    codes = np.asarray(codes, np.int64)
    if codes.ndim != 1 or len(codes) != len(vin):
        raise ValueError("codes must be one-dimensional and match vin length")
    if np.any(np.diff(vin) <= 0.0):
        raise ValueError("vin must be strictly increasing")
    levels = 1 << int(n_bits)
    if n_bits < 1 or np.any(codes < 0) or np.any(codes >= levels):
        raise ValueError(f"codes must lie in [0, {levels - 1}]")
    lo = float(vin[0] if vmin is None else vmin)
    hi = float(vin[-1] if vmax is None else vmax)
    if hi <= lo:
        raise ValueError("vmax must be greater than vmin")
    lsb = (hi - lo) / levels
    ideal = np.clip(np.floor((vin - lo) / lsb), 0, levels - 1).astype(np.int64)
    errors = codes - ideal
    return {
        "n_bits": int(n_bits),
        "lsb": lsb,
        "ideal_codes": ideal,
        "code_errors": errors,
        "max_abs_code_err": float(np.max(np.abs(errors))),
        "monotonic": bool(np.all(np.diff(codes) >= 0)),
    }


def carry_transition_codes(n_bits: int) -> list[int]:
    """Transition indices around every binary major carry, plus the offset.

    A binary-weighted CDAC stresses the bins on both sides of each carry
    ``2**j`` hardest — the transition into ``2**j`` swaps the largest cap
    group against the sum of all smaller ones, and any weight error lands in
    the widths of codes ``2**j - 1`` and ``2**j``. Measuring ``T(m-1)``,
    ``T(m)`` and ``T(m+1)`` around every carry therefore yields the two
    critical DNL bins per carry; ``T(1)`` anchors the offset.
    """
    if n_bits < 1:
        raise ValueError("n_bits must be at least 1")
    levels = 1 << n_bits
    targets = {1}
    for j in range(1, n_bits):
        m = 1 << j
        targets.update((m - 1, m, m + 1))
    return sorted(t for t in targets if 1 <= t <= levels - 1)


def bisect_code_transitions(probe, targets, n_bits: int, *, vmin: float,
                            vmax: float, tol_lsb: float = 0.05,
                            bracket_lsb: float = 2.0,
                            max_rounds: int = 64) -> dict:
    """Locate transition voltages ``T(k)`` by lockstep bisection.

    ``T(k)`` is the smallest input whose code reaches ``k`` (monotone
    converter assumed). Every pending transition contributes one probe per
    round, and ``probe(vins) -> codes`` receives them as one array — so a
    compiled batch converts all of them in a single parallel call and the
    round count, not the transition count, sets the serial depth.

    Each target starts from the bracket ``k*lsb ± bracket_lsb*lsb``; a
    bracket whose ends do not straddle the transition (screening said codes
    are close to ideal, but it may lie) is widened to the full range once.
    A transition outside even the full range — the converter never reaches
    ``k``, or starts at or above it — is reported as NaN in ``transitions``
    and listed in ``unmeasured``. Bisection resolves a bracket to
    ``tol_lsb * lsb``; a full-range recovery needs ``log2(range/tol)``
    rounds (~16 at 12 bits), a surviving bracket ~``log2(2*bracket/tol)``.

    This is both cheaper and *finer* than a full code-center ramp: the ramp
    quantizes every transition to half its sample spacing (±0.5 LSB at best),
    while bisection reaches ``tol_lsb`` directly.
    """
    targets = np.asarray(sorted(set(int(t) for t in targets)), dtype=np.int64)
    levels = 1 << int(n_bits)
    if targets.size == 0 or targets[0] < 1 or targets[-1] > levels - 1:
        raise ValueError(f"transition targets must lie in [1, {levels - 1}]")
    if tol_lsb <= 0.0 or bracket_lsb <= 0.0:
        raise ValueError("tol_lsb and bracket_lsb must be positive")
    lo_v, hi_v = float(vmin), float(vmax)
    if hi_v <= lo_v:
        raise ValueError("vmax must be greater than vmin")
    lsb = (hi_v - lo_v) / levels
    tol = tol_lsb * lsb

    ideal = lo_v + targets * lsb
    lo = np.maximum(ideal - bracket_lsb * lsb, lo_v)
    hi = np.minimum(ideal + bracket_lsb * lsb, hi_v)
    conversions = 0

    def codes_at(vins):
        nonlocal conversions
        vins = np.asarray(vins, dtype=float)
        conversions += len(vins)
        out = np.asarray(probe(vins), dtype=np.int64)
        if out.shape != vins.shape:
            raise ValueError("probe must return one code per input")
        return out

    # Round 0: validate every bracket end in one call; a failed bracket is
    # widened to the full range and its ends re-checked in one more call.
    # The invariant maintained from here on: code(lo) < k <= code(hi).
    ends = codes_at(np.concatenate([lo, hi]))
    lo_ok = ends[: len(targets)] < targets
    hi_ok = ends[len(targets):] >= targets
    bad = ~(lo_ok & hi_ok)
    unmeasured = np.zeros(len(targets), dtype=bool)
    if np.any(bad):
        lo[bad] = lo_v
        hi[bad] = hi_v
        ends = codes_at(np.concatenate([lo[bad], hi[bad]]))
        n_bad = int(bad.sum())
        unmeasured[np.flatnonzero(bad)[
            (ends[:n_bad] >= targets[bad]) | (ends[n_bad:] < targets[bad])
        ]] = True

    rounds = 0
    while True:
        pending = ~unmeasured & ((hi - lo) > tol)
        if not np.any(pending):
            break
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError(
                f"transition bisection did not converge in {max_rounds} rounds")
        mid = 0.5 * (lo[pending] + hi[pending])
        reached = codes_at(mid) >= targets[pending]
        idx = np.flatnonzero(pending)
        hi[idx[reached]] = mid[reached]
        lo[idx[~reached]] = mid[~reached]

    transitions = 0.5 * (lo + hi)
    transitions[unmeasured] = np.nan
    return {
        "targets": targets,
        "transitions": transitions,
        "unmeasured": targets[unmeasured].tolist(),
        "conversions": conversions,
        "rounds": rounds,
        "lsb": lsb,
        "tol_v": tol,
    }


def transition_dnl_inl(targets, transitions, n_bits: int, *, vmin: float,
                       vmax: float) -> dict:
    """DNL/INL from measured transition voltages.

    ``inl[i]`` is per measured transition: ``(T(k) - k*lsb) / lsb``.
    ``dnl`` covers exactly the codes whose *both* boundaries were measured:
    ``dnl(c) = (T(c+1) - T(c))/lsb - 1``. Codes with an unmeasured boundary
    are simply absent — never guessed.
    """
    targets = np.asarray(targets, dtype=np.int64)
    transitions = np.asarray(transitions, dtype=float)
    levels = 1 << int(n_bits)
    lo_v, hi_v = float(vmin), float(vmax)
    lsb = (hi_v - lo_v) / levels
    inl = (transitions - (lo_v + targets * lsb)) / lsb
    by_code = {int(k): float(t) for k, t in zip(targets, transitions)
               if np.isfinite(t)}
    dnl_codes = sorted(k for k in by_code if k + 1 in by_code)
    dnl = np.array([(by_code[k + 1] - by_code[k]) / lsb - 1.0
                    for k in dnl_codes])
    finite_inl = inl[np.isfinite(inl)]
    return {
        "inl": inl,
        "dnl_codes": np.asarray(dnl_codes, dtype=np.int64),
        "dnl": dnl,
        "max_abs_dnl": float(np.max(np.abs(dnl))) if dnl.size else float("nan"),
        "max_abs_inl": (float(np.max(np.abs(finite_inl)))
                        if finite_inl.size else float("nan")),
        "lsb": lsb,
    }


def code_density_metrics(codes, n_bits: int) -> dict:
    """Histogram DNL/INL for a uniformly distributed ADC input."""
    codes = np.asarray(codes, np.int64)
    levels = 1 << int(n_bits)
    if codes.ndim != 1 or len(codes) == 0:
        raise ValueError("codes must be a non-empty one-dimensional array")
    if n_bits < 1 or np.any(codes < 0) or np.any(codes >= levels):
        raise ValueError(f"codes must lie in [0, {levels - 1}]")
    counts = np.bincount(codes, minlength=levels)
    ideal = len(codes) / levels
    dnl = counts / ideal - 1.0
    inl = np.cumsum(dnl)[:-1]
    return {
        "n_bits": int(n_bits),
        "counts": counts,
        "dnl": dnl,
        "inl": inl,
        "missing_codes": np.flatnonzero(counts == 0),
        "max_abs_dnl": float(np.max(np.abs(dnl))),
        "max_abs_inl": float(np.max(np.abs(inl))) if len(inl) else 0.0,
    }


def _aliased_bin(index: int, length: int) -> int:
    value = index % length
    return length - value if value > length // 2 else value


def dynamic_metrics(
    codes,
    sample_rate: float,
    *,
    fundamental_bin: int | None = None,
    harmonics: int = 5,
    window: str = "none",
) -> dict:
    """Coherent-record SNDR, SNR, SFDR and ENOB from output codes.

    ``window='none'`` is the accurate choice for coherent sampling. ``'hann'``
    is available for exploratory non-coherent records and integrates the main
    lobe (fundamental +/- 1 bin).
    """
    codes = _as_1d(codes, "codes")
    n = len(codes)
    if n < 8:
        raise ValueError("dynamic metrics require at least eight samples")
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    mode = str(window).lower()
    if mode in {"none", "rect", "rectangular"}:
        win = np.ones(n)
        half_width = 0
    elif mode in {"hann", "hanning"}:
        win = np.hanning(n)
        half_width = 1
    else:
        raise ValueError("window must be 'none' or 'hann'")
    centered = codes - np.mean(codes)
    spectrum = np.fft.rfft(centered * win)
    power = np.abs(spectrum) ** 2
    power[0] = 0.0
    if fundamental_bin is None:
        fundamental_bin = int(np.argmax(power[1:]) + 1)
    fundamental_bin = int(fundamental_bin)
    if not 0 < fundamental_bin < len(power):
        raise ValueError("fundamental_bin must select a positive-frequency FFT bin")

    signal_bins = {k for k in range(fundamental_bin - half_width,
                                    fundamental_bin + half_width + 1)
                   if 0 < k < len(power)}
    signal_power = float(sum(power[k] for k in signal_bins))
    if signal_power <= 0.0:
        raise ValueError("fundamental has zero power")
    distortion_bins = set()
    harmonic_rows = []
    for order in range(2, int(harmonics) + 1):
        center = _aliased_bin(order * fundamental_bin, n)
        bins = {k for k in range(center - half_width, center + half_width + 1)
                if 0 < k < len(power)} - signal_bins
        distortion_bins.update(bins)
        harmonic_rows.append((order, center, float(sum(power[k] for k in bins))))
    occupied = signal_bins | distortion_bins | {0}
    total_other = float(sum(value for k, value in enumerate(power) if k not in signal_bins))
    noise_power = float(sum(value for k, value in enumerate(power) if k not in occupied))
    spur_power = max((float(value) for k, value in enumerate(power)
                      if k not in signal_bins and k != 0), default=0.0)
    tiny = np.finfo(float).tiny
    sndr = 10.0 * np.log10(signal_power / max(total_other, tiny))
    snr = 10.0 * np.log10(signal_power / max(noise_power, tiny))
    sfdr = 10.0 * np.log10(signal_power / max(spur_power, tiny))
    return {
        "sample_rate": float(sample_rate),
        "n_samples": n,
        "fundamental_bin": fundamental_bin,
        "fundamental_frequency": fundamental_bin * float(sample_rate) / n,
        "sndr_db": float(sndr),
        "snr_db": float(snr),
        "sfdr_db": float(sfdr),
        "enob": float((sndr - 1.76) / 6.02),
        "harmonics": harmonic_rows,
        "spectrum_power": power,
        "frequencies": np.fft.rfftfreq(n, 1.0 / float(sample_rate)),
    }


def average_supply_power(t, branch_currents: Mapping[str, Sequence[float]],
                         rail_voltages: Mapping[str, float], *, start=None) -> dict:
    """Average power delivered by ideal rail sources over a transient interval."""
    t = _as_1d(t, "t")
    if len(t) < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be strictly increasing with at least two points")
    begin = t[0] if start is None else float(start)
    if not t[0] <= begin < t[-1]:
        raise ValueError("start must lie before the end of the transient")
    mask = t >= begin
    tm = t[mask]
    per_rail = {}
    for rail, voltage in rail_voltages.items():
        key = rail if rail in branch_currents else f"rail:{rail}"
        if key not in branch_currents:
            raise ValueError(f"missing branch current for supply rail {rail!r}")
        current = _as_1d(branch_currents[key], f"branch_currents[{key!r}]")
        if len(current) != len(t):
            raise ValueError(f"branch current {key!r} length differs from t")
        instantaneous = -float(voltage) * current[mask]
        per_rail[str(rail)] = float(np.trapezoid(instantaneous, tm) / (tm[-1] - tm[0]))
    return {
        "per_rail_w": per_rail,
        "total_w": float(sum(per_rail.values())),
        "start": begin,
        "stop": float(t[-1]),
    }


def average_waveform_source_power(t, branch_currents: Mapping[str, Sequence[float]],
                                  source_waveforms: Mapping[str, Sequence[float]],
                                  *, start=None) -> dict:
    """Average power delivered by time-varying voltage sources.

    Keys must match the transient result's ``branch_currents`` names. Positive
    results mean net energy delivered into the circuit; negative values mean the
    ideal driver recovered energy from capacitive switching.
    """
    t = _as_1d(t, "t")
    begin = t[0] if start is None else float(start)
    if len(t) < 2 or np.any(np.diff(t) <= 0.0) or not t[0] <= begin < t[-1]:
        raise ValueError("invalid transient time range or power start")
    mask = t >= begin
    tm = t[mask]
    per_source = {}
    for name, waveform in source_waveforms.items():
        if name not in branch_currents:
            raise ValueError(f"missing branch current for waveform source {name!r}")
        voltage = _as_1d(waveform, f"source_waveforms[{name!r}]")
        current = _as_1d(branch_currents[name], f"branch_currents[{name!r}]")
        if len(voltage) != len(t) or len(current) != len(t):
            raise ValueError(f"waveform source {name!r} length differs from t")
        power = -voltage[mask] * current[mask]
        per_source[str(name)] = float(np.trapezoid(power, tm) / (tm[-1] - tm[0]))
    return {
        "per_source_w": per_source,
        "total_w": float(sum(per_source.values())),
        "start": begin,
        "stop": float(t[-1]),
    }
