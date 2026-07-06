"""Shared adaptive transient/PSS timestep configuration."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


_DEFAULT_RELTOL = 1e-4
_DEFAULT_VABSTOL = 1e-6
_DEFAULT_IABSTOL = 1e-12
_DEFAULT_MAX_STEPS = 200000
_DEFAULT_H0 = None
_DEFAULT_FREEZE_FACTOR = 10.0

ADAPTIVE_ACCEPT_WRMS = 1.0
ADAPTIVE_DONE_ABS = 1e-18
ADAPTIVE_DONE_REL = 1e-13
ADAPTIVE_ERR_FLOOR = 1e-12
ADAPTIVE_GROWTH_MAX = 2.0
ADAPTIVE_GROWTH_MIN = 0.2
ADAPTIVE_INITIAL_MIN_DENOM = 16
ADAPTIVE_INPUT_SLOPE_BREAK_FRACTION = 0.1
ADAPTIVE_LTE_DIVISOR = 3.0
ADAPTIVE_MIN_H_ABS = 1e-18
ADAPTIVE_MIN_H_REL = 1e-15
ADAPTIVE_SAFETY = 0.9
ADAPTIVE_SCALE_FLOOR = 1e-30
ADAPTIVE_STEP_ORDER = 3.0


@dataclass(frozen=True)
class AdaptiveConfig:
    """LTE adaptive gear2 options shared by transient, PSS, and wrappers.

    The short field names are used internally. ``to_transient_kwargs()`` and
    ``to_pss_kwargs()`` keep the legacy public keyword names available at API
    boundaries.
    """

    reltol: float = _DEFAULT_RELTOL
    vabstol: float = _DEFAULT_VABSTOL
    iabstol: float = _DEFAULT_IABSTOL
    max_steps: int = _DEFAULT_MAX_STEPS
    h0: float | None = _DEFAULT_H0
    freeze_factor: float = _DEFAULT_FREEZE_FACTOR

    def __post_init__(self):
        object.__setattr__(self, "reltol", float(self.reltol))
        object.__setattr__(self, "vabstol", float(self.vabstol))
        object.__setattr__(self, "iabstol", float(self.iabstol))
        object.__setattr__(self, "max_steps", int(self.max_steps))
        object.__setattr__(
            self, "h0", None if self.h0 is None else float(self.h0))
        object.__setattr__(self, "freeze_factor", float(self.freeze_factor))
        if self.reltol <= 0.0:
            raise ValueError("adaptive reltol must be positive")
        if self.vabstol <= 0.0:
            raise ValueError("adaptive vabstol must be positive")
        if self.iabstol <= 0.0:
            raise ValueError("adaptive iabstol must be positive")
        if self.max_steps < 1:
            raise ValueError("adaptive max_steps must be >= 1")
        if self.h0 is not None and self.h0 <= 0.0:
            raise ValueError("adaptive h0 must be positive when provided")
        if self.freeze_factor <= 0.0:
            raise ValueError("adaptive freeze_factor must be positive")

    @classmethod
    def coerce(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**_normalize_adaptive_mapping(value))
        raise TypeError("adaptive_config must be an AdaptiveConfig, dict, or None")

    def with_updates(self, **updates):
        clean = {key: val for key, val in updates.items() if val is not None}
        return replace(self, **clean) if clean else self

    def to_transient_kwargs(self):
        return {
            "adaptive_reltol": self.reltol,
            "adaptive_vabstol": self.vabstol,
            "adaptive_iabstol": self.iabstol,
            "adaptive_max_steps": self.max_steps,
            "adaptive_h0": self.h0,
        }

    def to_pss_kwargs(self):
        out = self.to_transient_kwargs()
        out["adaptive_freeze_factor"] = self.freeze_factor
        return out


def _normalize_adaptive_mapping(mapping):
    aliases = {
        "adaptive_reltol": "reltol",
        "adaptive_vabstol": "vabstol",
        "adaptive_iabstol": "iabstol",
        "adaptive_max_steps": "max_steps",
        "adaptive_h0": "h0",
        "adaptive_freeze_factor": "freeze_factor",
    }
    valid = {"reltol", "vabstol", "iabstol", "max_steps", "h0", "freeze_factor"}
    out = {}
    for key, value in mapping.items():
        name = aliases.get(key, key)
        if name in valid:
            out[name] = value
    return out


def resolve_adaptive_config(
    adaptive_config=None, *,
    adaptive_reltol=_DEFAULT_RELTOL,
    adaptive_vabstol=_DEFAULT_VABSTOL,
    adaptive_iabstol=_DEFAULT_IABSTOL,
    adaptive_max_steps=_DEFAULT_MAX_STEPS,
    adaptive_h0=_DEFAULT_H0,
    adaptive_freeze_factor=_DEFAULT_FREEZE_FACTOR,
):
    """Merge optional config with legacy ``adaptive_*`` keyword arguments.

    Legacy keyword values keep their historical defaults. When both a config and
    old-style keyword are supplied, a keyword only overrides the config if it
    differs from that historical default. This preserves backwards compatibility
    without forcing every caller to pass six fields through every layer.
    """
    if adaptive_config is None:
        return AdaptiveConfig(
            reltol=adaptive_reltol,
            vabstol=adaptive_vabstol,
            iabstol=adaptive_iabstol,
            max_steps=adaptive_max_steps,
            h0=adaptive_h0,
            freeze_factor=adaptive_freeze_factor,
        )

    cfg = AdaptiveConfig.coerce(adaptive_config)
    updates = {}
    if adaptive_reltol != _DEFAULT_RELTOL:
        updates["reltol"] = adaptive_reltol
    if adaptive_vabstol != _DEFAULT_VABSTOL:
        updates["vabstol"] = adaptive_vabstol
    if adaptive_iabstol != _DEFAULT_IABSTOL:
        updates["iabstol"] = adaptive_iabstol
    if adaptive_max_steps != _DEFAULT_MAX_STEPS:
        updates["max_steps"] = adaptive_max_steps
    if adaptive_h0 != _DEFAULT_H0:
        updates["h0"] = adaptive_h0
    if adaptive_freeze_factor != _DEFAULT_FREEZE_FACTOR:
        updates["freeze_factor"] = adaptive_freeze_factor
    return cfg.with_updates(**updates)


def adaptive_done_tol(span):
    return max(ADAPTIVE_DONE_ABS, ADAPTIVE_DONE_REL * float(span))


def adaptive_min_h(span):
    return max(ADAPTIVE_MIN_H_ABS, ADAPTIVE_MIN_H_REL * float(span))


def adaptive_initial_h(tgrid, max_step_eff, h0=None):
    if h0 is not None:
        h = float(h0)
    else:
        tgrid = np.asarray(tgrid, float)
        span = float(tgrid[-1] - tgrid[0])
        denom = max(ADAPTIVE_INITIAL_MIN_DENOM, max(1, len(tgrid) - 1))
        h = max(span / denom, ADAPTIVE_MIN_H_ABS)
        if len(tgrid) > 1:
            h = min(h, float(np.min(np.diff(tgrid))))
    if h <= 0.0 or not math.isfinite(h):
        span = float(np.asarray(tgrid, float)[-1] - np.asarray(tgrid, float)[0])
        h = min(float(max_step_eff), span / 100.0)
    return min(h, float(max_step_eff))


def adaptive_next_h(h, err):
    if err <= 0.0:
        fac = ADAPTIVE_GROWTH_MAX
    elif not math.isfinite(float(err)):
        fac = ADAPTIVE_GROWTH_MIN
    else:
        fac = ADAPTIVE_SAFETY * float(err) ** (-1.0 / ADAPTIVE_STEP_ORDER)
        fac = min(ADAPTIVE_GROWTH_MAX, max(ADAPTIVE_GROWTH_MIN, fac))
    return float(h) * fac


def adaptive_lte_wrms(v_half, v_full, n_nodes, reltol, vabstol, iabstol):
    v_half = np.asarray(v_half, float)
    v_full = np.asarray(v_full, float)
    scale = float(reltol) * np.maximum(np.abs(v_half), np.abs(v_full))
    n_nodes = int(n_nodes)
    if n_nodes > 0:
        scale[:n_nodes] += float(vabstol)
    if len(scale) > n_nodes:
        scale[n_nodes:] += float(iabstol)
    scale = np.maximum(scale, ADAPTIVE_SCALE_FLOOR)
    lte = (v_half - v_full) / ADAPTIVE_LTE_DIVISOR
    return float(np.sqrt(np.mean((lte / scale) ** 2)))


def adaptive_critical_times(tgrid, input_values):
    tgrid = np.asarray(tgrid, float)
    input_values = np.asarray(input_values, float)
    if input_values.size == 0 or input_values.shape[0] == 0 or len(tgrid) < 3:
        return np.empty(0, float)
    dt = np.diff(tgrid)
    slopes = np.diff(input_values, axis=1) / dt[None, :]
    global_slope = max(1.0, float(np.max(np.abs(slopes))))
    out = []
    for kk in range(1, len(tgrid) - 1):
        jump = np.max(np.abs(slopes[:, kk] - slopes[:, kk - 1]))
        if jump > ADAPTIVE_INPUT_SLOPE_BREAK_FRACTION * global_slope:
            out.append(float(tgrid[kk]))
    return np.asarray(out, float)
