"""Backend-independent frequency-response measurements."""
from __future__ import annotations

import numpy as np


def unity_gain_freq(freq, response):
    """First descending unity-gain crossing above the response peak, in hertz."""
    freq = np.asarray(freq, float)
    magnitude = np.abs(np.asarray(response, complex))
    order = np.argsort(freq)
    freq, magnitude = freq[order], magnitude[order]
    peak = int(np.argmax(magnitude))
    if magnitude[peak] < 1.0:
        return float("nan")
    for index in range(peak + 1, len(magnitude)):
        if magnitude[index] <= 1.0:
            f0, f1 = freq[index - 1], freq[index]
            g0 = np.log10(max(magnitude[index - 1], 1e-300))
            g1 = np.log10(max(magnitude[index], 1e-300))
            if f0 <= 0.0 or f1 <= 0.0 or g1 == g0:
                return float(f1)
            x0, x1 = np.log10(f0), np.log10(f1)
            crossing = x0 - g0 * (x1 - x0) / (g1 - g0)
            return float(10.0 ** np.clip(crossing, min(x0, x1), max(x0, x1)))
    return float("nan")


def phase_margin(freq, response):
    """Unity-feedback phase margin in degrees, or NaN without a unity crossing."""
    crossing = unity_gain_freq(freq, response)
    if not np.isfinite(crossing):
        return float("nan")
    freq = np.asarray(freq, float)
    response = np.asarray(response, complex)
    order = np.argsort(freq)
    sorted_freq = freq[order]
    phase = np.degrees(np.unwrap(np.angle(response[order])))
    magnitude = np.abs(response[order])
    reference = phase[int(np.argmax(magnitude))]
    crossing_phase = float(np.interp(
        np.log10(crossing), np.log10(sorted_freq), phase))
    return 180.0 + crossing_phase - reference
