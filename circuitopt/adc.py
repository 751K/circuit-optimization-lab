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
