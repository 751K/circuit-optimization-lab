"""Run analyses described by a circuit JSON file.

The loader keeps JSON parsing separate from solver execution.  This module is
the thin dispatch layer that turns optional ``periodic`` and ``analyses`` blocks
into calls to AC/noise/transient/PSS/PAC/PNoise solvers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .ac_solver import ac_solve
from .adaptive_config import resolve_adaptive_config
from .analysis_options import ADAPTIVE_OPTION_NAMES, solver_kwargs
from .circuit_loader import CircuitSpec, load_circuit_json
from .noise_solver import band_rms, noise_analysis
from .pac_solver import pac_solve
from .pnoise_solver import pnoise_solve
from .pss_solver import pss_solve
from .transient_solver import transient


_ANALYSIS_ORDER = ("ac", "noise", "transient", "pss", "pac", "pnoise")


def _pack_adaptive_config(kwargs):
    legacy = {key: kwargs.pop(key) for key in tuple(kwargs) if key in ADAPTIVE_OPTION_NAMES}
    if "adaptive_config" in kwargs:
        kwargs["adaptive_config"] = resolve_adaptive_config(
            kwargs["adaptive_config"], **legacy)
    elif legacy:
        kwargs["adaptive_config"] = resolve_adaptive_config(legacy)
    return kwargs


def _as_spec(spec_or_path):
    if isinstance(spec_or_path, CircuitSpec):
        return spec_or_path
    if isinstance(spec_or_path, (str, Path)):
        return load_circuit_json(spec_or_path)
    raise TypeError("spec_or_path must be a CircuitSpec or JSON path")


def _num(value, bias, field):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value not in bias:
            raise ValueError(f"{field} references unknown bias key {value!r}")
        return float(bias[value])
    raise ValueError(f"{field} must be numeric or a bias key")


def _frequency_grid(cfg, *, default=None):
    if cfg is None:
        if default is None:
            raise ValueError("frequency grid is required")
        return np.asarray(default, float)
    if isinstance(cfg, (list, tuple)):
        out = np.asarray(cfg, float)
    elif isinstance(cfg, dict):
        start = float(cfg["start"])
        stop = float(cfg["stop"])
        num = int(cfg["num"])
        scale = str(cfg.get("scale", "log")).lower()
        if scale == "linear":
            out = np.linspace(start, stop, num)
        elif scale == "log":
            out = np.logspace(np.log10(start), np.log10(stop), num)
        else:
            raise ValueError("frequency grid scale must be 'log' or 'linear'")
    else:
        raise ValueError("frequency grid must be a list or object")
    if out.ndim != 1 or len(out) == 0 or np.any(out <= 0.0):
        raise ValueError("frequency grid must contain positive frequencies")
    return out


def _period_from(periodic):
    if "period" in periodic:
        period = float(periodic["period"])
    elif "frequency" in periodic:
        period = 1.0 / float(periodic["frequency"])
    elif "fundamental" in periodic:
        period = 1.0 / float(periodic["fundamental"])
    else:
        raise ValueError("periodic requires period, frequency, or fundamental")
    if period <= 0.0:
        raise ValueError("period must be positive")
    return period


def _fundamental_from(periodic):
    return 1.0 / _period_from(periodic)


def _time_grid(cfg, *, default_stop=None, default_points=101):
    cfg = dict(cfg or {})
    if "points" in cfg:
        out = np.asarray(cfg["points"], float)
    else:
        start = float(cfg.get("start", 0.0))
        if "stop" in cfg:
            stop = float(cfg["stop"])
        elif "duration" in cfg:
            stop = start + float(cfg["duration"])
        elif default_stop is not None:
            stop = float(default_stop)
        else:
            raise ValueError("time grid requires stop, duration, or a default stop")
        n_points = int(cfg.get("n_points", default_points))
        out = np.linspace(start, stop, n_points)
    if out.ndim != 1 or len(out) < 2 or not np.all(np.diff(out) > 0.0):
        raise ValueError("time grid must be strictly increasing with at least two points")
    return out


def _with_adaptive_waveform_breakpoints(periodic, tgrid, period):
    """Add pulse/square discontinuity and edge boundary times for adaptive runs."""
    extras = []
    for spec in (periodic or {}).get("inputs", {}).values():
        if not isinstance(spec, dict):
            continue
        kind = str(spec.get("type", "constant")).lower()
        if kind not in {"square", "pulse"}:
            continue
        delay = float(spec.get("delay", 0.0))
        duty = float(spec.get("duty", 0.5))
        width = duty * float(period)
        offsets = [0.0, width]
        if kind == "pulse":
            rise = max(0.0, float(spec.get("rise", 0.0)))
            fall = max(0.0, float(spec.get("fall", 0.0)))
            offsets.extend([rise, width + fall])
        for off in offsets:
            tm = np.mod(delay + off, period)
            extras.append(tm)
            if tm == 0.0:
                extras.append(period)
    if not extras:
        return tgrid
    out = np.unique(np.concatenate([np.asarray(tgrid, float), np.asarray(extras, float)]))
    out = out[(out >= -1e-15) & (out <= period + 1e-15)]
    out[0] = 0.0
    if not np.isclose(out[-1], period, rtol=1e-12, atol=max(1e-18, period * 1e-12)):
        out = np.append(out, period)
    return out


def _merge_periodic(base, override):
    merged = dict(base or {})
    override = dict(override or {})
    for key in ("inputs", "node_inputs"):
        if key in override:
            inner = dict(merged.get(key, {}))
            inner.update(override[key] or {})
            merged[key] = inner
    if "current_inputs" in override:
        merged["current_inputs"] = list(override["current_inputs"] or [])
    if "signed_devices" in override:
        merged["signed_devices"] = list(override["signed_devices"] or [])
    for key, value in override.items():
        if key not in {"inputs", "node_inputs", "current_inputs", "signed_devices"}:
            merged[key] = value
    return merged


def _waveform(spec, tgrid, period, bias, fundamental, key):
    if isinstance(spec, (int, float, str)):
        return np.full_like(tgrid, _num(spec, bias, f"periodic.inputs.{key}"), dtype=float)
    if not isinstance(spec, dict):
        raise ValueError(f"periodic.inputs.{key} must be a number, bias key, or object")

    kind = str(spec.get("type", "constant")).lower()
    if kind in {"constant", "dc"}:
        value = spec.get("value", spec.get("dc", spec.get("offset", 0.0)))
        return np.full_like(tgrid, _num(value, bias, f"periodic.inputs.{key}.value"), dtype=float)

    if kind in {"sine", "sin", "cosine", "cos"}:
        offset = _num(spec.get("dc", spec.get("offset", 0.0)), bias,
                      f"periodic.inputs.{key}.dc")
        amp = float(spec.get("amplitude", 1.0))
        phase = float(spec.get("phase", 0.0))
        freq = float(spec.get("frequency", spec.get("harmonic", 1.0) * fundamental))
        angle = 2.0 * np.pi * freq * tgrid + phase
        trig = np.cos(angle) if kind in {"cosine", "cos"} else np.sin(angle)
        return offset + amp * trig

    if kind in {"square", "pulse"}:
        low = _num(spec.get("low", 0.0), bias, f"periodic.inputs.{key}.low")
        high = _num(spec.get("high", 1.0), bias, f"periodic.inputs.{key}.high")
        duty = float(spec.get("duty", 0.5))
        if not 0.0 < duty < 1.0:
            raise ValueError(f"periodic.inputs.{key}.duty must be between 0 and 1")
        delay = float(spec.get("delay", 0.0))
        phase_t = np.mod(tgrid - delay, period)
        width = duty * period
        if kind == "square":
            return np.where(phase_t < width, high, low)
        rise = max(0.0, float(spec.get("rise", 0.0)))
        fall = max(0.0, float(spec.get("fall", 0.0)))
        out = np.full_like(tgrid, low, dtype=float)
        if rise > 0.0:
            mask = phase_t < rise
            out[mask] = low + (high - low) * phase_t[mask] / rise
        high_start = rise if rise > 0.0 else 0.0
        high_end = max(high_start, width)
        mask = (phase_t >= high_start) & (phase_t < high_end)
        out[mask] = high
        if fall > 0.0:
            mask = (phase_t >= high_end) & (phase_t < high_end + fall)
            out[mask] = high + (low - high) * (phase_t[mask] - high_end) / fall
        return out

    if kind == "pwl":
        times = np.asarray(spec["times"], float)
        values = np.asarray(spec["values"], float)
        if len(times) != len(values):
            raise ValueError(f"periodic.inputs.{key}.times/values length mismatch")
        if len(times) < 2 or not np.all(np.diff(times) > 0.0):
            raise ValueError(f"periodic.inputs.{key}.times must be strictly increasing")
        return np.interp(np.mod(tgrid, period), times, values, period=period)

    raise ValueError(f"Unsupported waveform type {kind!r} for input {key!r}")


def build_periodic_context(spec, periodic_cfg, *, tgrid=None):
    """Build transient/PSS input arguments from a JSON ``periodic`` block."""
    periodic = dict(periodic_cfg or {})
    period = _period_from(periodic)
    fundamental = 1.0 / period
    if tgrid is None:
        grid_cfg = dict(periodic.get("tgrid", {}))
        if "n_points" not in grid_cfg and "n_points" in periodic:
            grid_cfg["n_points"] = periodic["n_points"]
        tgrid = _time_grid(grid_cfg, default_stop=period,
                           default_points=int(periodic.get("n_points", 101)))
    inputs = {
        str(key): _waveform(value, tgrid, period, spec.bias, fundamental, str(key))
        for key, value in periodic.get("inputs", {}).items()
    }
    current_inputs = []
    for item in periodic.get("current_inputs", []) or []:
        if isinstance(item, dict):
            current_inputs.append({
                "p": str(item["p"]), "q": str(item["q"]),
                "input": str(item["input"]),
            })
        else:
            p, q, key = item
            current_inputs.append((str(p), str(q), str(key)))
    return {
        "period": period,
        "fundamental": fundamental,
        "tgrid": tgrid,
        "inputs": inputs,
        "node_inputs": {str(k): str(v) for k, v in periodic.get("node_inputs", {}).items()},
        "current_inputs": tuple(current_inputs),
        "signed_devices": tuple(str(x) for x in periodic.get("signed_devices", ()) or ()),
    }


def _complex_value(value, field):
    if isinstance(value, (int, float)):
        return complex(float(value), 0.0)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    if isinstance(value, dict):
        return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    raise ValueError(f"{field} must be a number, [real, imag], or object")


def _resolve_corner(corner):
    if isinstance(corner, str):
        from .corners import CORNERS
        if corner not in CORNERS:
            raise ValueError(f"Unknown process corner {corner!r}; expected one of {sorted(CORNERS)}")
        return CORNERS[corner]
    return corner


def _corner_from_cfg(cfg, *, default=None):
    corner = (cfg or {}).get("corner", default)
    return _resolve_corner(corner)


def _input_drive(cfg, periodic):
    raw = cfg.get("input_drive")
    if raw is None:
        keys = list((periodic or {}).get("inputs", {}).keys())
        if len(keys) == 1:
            return {str(keys[0]): 1.0}
        raise ValueError("PAC/PNoise requires input_drive when periodic has multiple inputs")
    return {str(k): _complex_value(v, f"input_drive.{k}") for k, v in raw.items()}


def _propagate_shared_pss_corner(analysis_cfg):
    pss_cfg = analysis_cfg.get("pss")
    if not isinstance(pss_cfg, dict) or "corner" in pss_cfg:
        return
    dep_corners = []
    for name in ("pac", "pnoise"):
        cfg = analysis_cfg.get(name)
        if isinstance(cfg, dict) and "corner" in cfg:
            dep_corners.append(cfg["corner"])
    if not dep_corners:
        return
    first = _resolve_corner(dep_corners[0])
    for raw in dep_corners[1:]:
        if _resolve_corner(raw) != first:
            raise ValueError("PAC and PNoise request different process corners for shared PSS")
    pss_cfg["corner"] = dep_corners[0]


def _pss_config(spec, analyses, owner_cfg):
    cfg = dict((analyses or {}).get("pss", {}) or {})
    cfg.update(owner_cfg.get("pss", {}) or {})
    if "corner" not in cfg and "corner" in owner_cfg:
        cfg["corner"] = owner_cfg["corner"]
    periodic = _merge_periodic(spec.periodic or {}, cfg.pop("periodic", None))
    return cfg, periodic


def _analysis_periodic_grid(periodic, cfg):
    periodic = dict(periodic or {})
    if "tgrid" in cfg:
        periodic["tgrid"] = cfg["tgrid"]
    elif "n_points" in cfg:
        periodic["n_points"] = cfg["n_points"]
    return periodic


def _run_pss(spec, pss_cfg, periodic):
    kwargs = _pack_adaptive_config(solver_kwargs("pss", pss_cfg))
    periodic = _analysis_periodic_grid(periodic, pss_cfg)
    context = build_periodic_context(spec, periodic)
    if bool(kwargs.get("adaptive", False)):
        tgrid = _with_adaptive_waveform_breakpoints(periodic, context["tgrid"], context["period"])
        if len(tgrid) != len(context["tgrid"]):
            context = build_periodic_context(spec, periodic, tgrid=tgrid)
    if "corner" in kwargs:
        kwargs["corner"] = _corner_from_cfg(pss_cfg)
    return pss_solve(
        spec.sizes, spec.bias, context["period"], topo=spec.topology, nf=spec.nf,
        tgrid=context["tgrid"], inputs=context["inputs"],
        node_inputs=context["node_inputs"], current_inputs=context["current_inputs"],
        signed_devices=context["signed_devices"],
        model_types=spec.model_types, device_kwargs=spec.device_kwargs, **kwargs,
    )


def _run_transient(spec, cfg):
    periodic = _merge_periodic(spec.periodic or {}, cfg.get("periodic"))
    if periodic:
        period = _period_from(periodic)
        grid_cfg = dict(cfg.get("tgrid", periodic.get("tgrid", {})) or {})
        if "duration" not in grid_cfg and "stop" not in grid_cfg and "tstop" in cfg:
            grid_cfg["stop"] = cfg["tstop"]
        if "duration" not in grid_cfg and "stop" not in grid_cfg and "duration" in cfg:
            grid_cfg["duration"] = cfg["duration"]
        if "n_points" not in grid_cfg and "n_points" in cfg:
            grid_cfg["n_points"] = cfg["n_points"]
        default_stop = float(cfg.get("tstop", cfg.get("duration", period)))
        tgrid = _time_grid(grid_cfg, default_stop=default_stop,
                           default_points=int(cfg.get("n_points", 101)))
        context = build_periodic_context(spec, periodic, tgrid=tgrid)
        kwargs = _pack_adaptive_config(solver_kwargs("transient", cfg))
        if bool(kwargs.get("adaptive", False)):
            tgrid = _with_adaptive_waveform_breakpoints(periodic, context["tgrid"], period)
            if len(tgrid) != len(context["tgrid"]):
                context = build_periodic_context(spec, periodic, tgrid=tgrid)
        if "corner" in kwargs:
            kwargs["corner"] = _corner_from_cfg(cfg)
        return transient(
            spec.sizes, spec.bias, tgrid, topo=spec.topology, nf=spec.nf,
            inputs=context["inputs"], node_inputs=context["node_inputs"],
            current_inputs=context["current_inputs"],
            signed_devices=context["signed_devices"],
            model_types=spec.model_types, device_kwargs=spec.device_kwargs, **kwargs,
        )
    tgrid = _time_grid(cfg.get("tgrid", cfg), default_stop=cfg.get("tstop", cfg.get("duration")),
                       default_points=int(cfg.get("n_points", 101)))
    kwargs = _pack_adaptive_config(solver_kwargs("transient", cfg))
    if "corner" in kwargs:
        kwargs["corner"] = _corner_from_cfg(cfg)
    return transient(
        spec.sizes, spec.bias, tgrid, topo=spec.topology, nf=spec.nf,
        signed_devices=tuple(cfg.get("signed_devices", ()) or ()),
        model_types=spec.model_types, device_kwargs=spec.device_kwargs, **kwargs,
    )


def run_analysis_suite(spec_or_path, analyses=None, *, selected=None):
    """Run the analyses configured in a ``CircuitSpec`` or JSON file.

    ``selected`` can restrict execution to a subset such as ``["pss", "pac"]``.
    Required dependencies are run automatically: PAC/PNoise will compute PSS if
    it is not already present.
    """
    spec = _as_spec(spec_or_path)
    raw_analysis_cfg = analyses if analyses is not None else (spec.analyses or {})
    analysis_cfg = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in dict(raw_analysis_cfg).items()
    }
    if not analysis_cfg:
        raise ValueError("No analyses configured")
    _propagate_shared_pss_corner(analysis_cfg)
    selected_set = set(selected) if selected is not None else None
    results = {}

    def want(name):
        return selected_set is None or name in selected_set

    def ensure_pss(owner_cfg):
        if "pss" not in results:
            pss_cfg, periodic = _pss_config(spec, analysis_cfg, owner_cfg)
            results["pss"] = _run_pss(spec, pss_cfg, periodic)
        elif "corner" in owner_cfg:
            requested = _corner_from_cfg(owner_cfg)
            existing = results["pss"].get("corner")
            if existing != requested:
                raise ValueError(
                    "PSS/PAC/PNoise corner mismatch: existing PSS was run with "
                    f"{existing!r}, but dependent analysis requested {requested!r}"
                )
        return results["pss"]

    for name in _ANALYSIS_ORDER:
        if name not in analysis_cfg or not want(name):
            continue
        cfg = dict(analysis_cfg.get(name) or {})
        # Seed the (possibly multistable) DC from the config's first dc_guess, and
        # bind any non-default per-device models (e.g. silicon sky130/freepdk45) —
        # both default to the old behaviour (None) for OTFT configs without them.
        dc_seed = next((g for g in spec.topology.dc_guesses if isinstance(g, dict)), None)
        if name == "ac":
            freqs = _frequency_grid(cfg.get("freqs"))
            corner = _corner_from_cfg(cfg)
            results[name] = ac_solve(
                spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf,
                corner=corner, x0_guess=dc_seed,
                model_types=spec.model_types, device_kwargs=spec.device_kwargs,
            )
        elif name == "noise":
            freqs = _frequency_grid(cfg.get("freqs"))
            corner_default = (analysis_cfg.get("ac") or {}).get("corner")
            corner = _corner_from_cfg(cfg, default=corner_default)
            x0_guess = dc_seed
            ac_result = None
            if "ac" in results and results["ac"] is not None:
                same_corner = results["ac"].get("corner") == corner
                if same_corner:
                    x0_guess = results["ac"].get("dc_op")
                    ac_freqs = np.asarray(results["ac"].get("freqs", ()), float)
                    if ac_freqs.shape == freqs.shape and np.allclose(
                            ac_freqs, freqs, rtol=0.0, atol=0.0):
                        ac_result = results["ac"]
            noise = noise_analysis(
                spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf,
                corner=corner, x0_guess=x0_guess, ac_result=ac_result,
                model_types=spec.model_types, device_kwargs=spec.device_kwargs,
            )
            if noise is not None and "band" in cfg:
                lo, hi = map(float, cfg["band"])
                noise["out_uV_band"] = band_rms(freqs, noise["out_psd"], lo, hi) * 1e6
                noise["irn_uV_band"] = band_rms(freqs, noise["irn_psd"], lo, hi) * 1e6
            results[name] = noise
        elif name == "transient":
            results[name] = _run_transient(spec, cfg)
        elif name == "pss":
            ensure_pss(cfg)
        elif name == "pac":
            pss = ensure_pss(cfg)
            freqs = _frequency_grid(cfg.get("freqs"))
            _, periodic = _pss_config(spec, analysis_cfg, cfg)
            corner = _corner_from_cfg(cfg, default=pss.get("corner"))
            kwargs = solver_kwargs("pac", cfg, include_defaults=True)
            results[name] = pac_solve(
                spec.sizes, spec.bias, freqs, pss_result=pss,
                input_drive=_input_drive(cfg, periodic), nf=spec.nf,
                corner=corner, **kwargs,
            )
        elif name == "pnoise":
            pss = ensure_pss(cfg)
            freqs = _frequency_grid(cfg.get("freqs"))
            _, periodic = _pss_config(spec, analysis_cfg, cfg)
            input_drive = _input_drive(cfg, periodic)
            pac_result = results.get("pac")
            corner = _corner_from_cfg(cfg, default=pss.get("corner"))
            kwargs = solver_kwargs("pnoise", cfg, include_defaults=True)
            band = kwargs.pop("band")
            results[name] = pnoise_solve(
                spec.sizes, spec.bias, freqs, pss_result=pss,
                fundamental=_fundamental_from(periodic), nf=spec.nf,
                corner=corner, band=band,
                pac_result=pac_result, input_drive=input_drive,
                **kwargs,
            )
    return results


run_json_analyses = run_analysis_suite
