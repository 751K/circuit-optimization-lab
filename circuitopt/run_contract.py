"""Strict simulation-result contract and unit-bearing design measurements."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class SimulationInvalid(RuntimeError):
    """A run completed no trustworthy physical result."""

    def __init__(self, code: str, message: str, *, analysis: str | None = None):
        super().__init__(message)
        self.code = str(code)
        self.analysis = analysis

    def as_dict(self) -> dict[str, str]:
        out = {"code": self.code, "message": str(self)}
        if self.analysis is not None:
            out["analysis"] = self.analysis
        return out


class ModelEvaluationError(SimulationInvalid):
    """A compact-model call failed and no physical substitute is permitted."""

    def __init__(self, device: str, operation: str, exc: Exception):
        super().__init__(
            "model_evaluation_failed",
            f"{device}: compact-model {operation} failed: {type(exc).__name__}: {exc}",
        )
        self.device = str(device)
        self.operation = str(operation)


class ModelBindingError(SimulationInvalid, ValueError):
    """A candidate cannot instantiate its explicitly selected model/bin."""

    def __init__(self, message: str, *, device: str | None = None):
        prefix = f"{device}: " if device else ""
        super().__init__(
            "model_binding_failed",
            prefix + str(message),
            analysis="model_binding",
        )
        self.device = device


def reraise_invalid(exc: Exception) -> None:
    """Keep broad numerical-recovery handlers from swallowing strict failures."""
    # A compact-model domain failure at an intermediate Newton trial point is a
    # rejected numerical step, not a fabricated physical result. Recovery may try
    # another seed/bounded solve; the run is invalid only if no solve converges.
    if isinstance(exc, SimulationInvalid) and not isinstance(exc, ModelEvaluationError):
        raise exc


def _require_finite(value: Any, path: str, analysis: str) -> None:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise SimulationInvalid(
            "invalid_result_type", f"{path} is not numeric: {exc}", analysis=analysis
        ) from exc
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise SimulationInvalid(
            "non_finite_result", f"{path} contains an empty or non-finite result",
            analysis=analysis,
        )


_REQUIRED_NUMERIC = {
    "ac": ("gains", "dc_op"),
    "noise": ("out_psd", "irn_psd"),
    "transient": ("output",),
    "pss": (),
    "pac": ("response",),
    "pnoise": ("out_psd", "irn_psd"),
}


def ensure_analysis_valid(analysis: str, result: Mapping[str, Any] | None) -> None:
    """Raise when an analysis did not produce a trustworthy finite result."""
    name = str(analysis)
    if result is None:
        raise SimulationInvalid(
            "not_converged", f"{name} analysis did not converge", analysis=name
        )
    if name in {"transient", "pss"} and int(result.get("nfail", 0)) != 0:
        raise SimulationInvalid(
            "not_converged",
            f"{name} analysis has {int(result.get('nfail', 0))} failed time points",
            analysis=name,
        )
    if name == "pac":
        failures = np.asarray(result.get("nfail", ()), dtype=int)
        if failures.size and np.any(failures != 0):
            raise SimulationInvalid(
                "not_converged",
                f"pac analysis has {int(np.count_nonzero(failures))} failed "
                "frequency points",
                analysis=name,
            )
    if name == "pss" and result.get("converged") is not True:
        raise SimulationInvalid(
            "not_converged", "PSS shooting did not converge", analysis=name
        )
    if name == "pss" and "output" in result:
        _require_finite(result["output"], "pss.output", name)
    for key in _REQUIRED_NUMERIC.get(name, ()):
        if key not in result:
            raise SimulationInvalid(
                "missing_result", f"{name} result is missing {key!r}", analysis=name
            )
        value = result[key]
        if isinstance(value, Mapping):
            for child, child_value in value.items():
                _require_finite(child_value, f"{name}.{key}.{child}", name)
        else:
            _require_finite(value, f"{name}.{key}", name)
    if name == "ac":
        power = result.get("source_power")
        required_power = {
            "total_w", "per_source_w", "source_currents_a", "source_voltages_v",
        }
        if not isinstance(power, Mapping) or not required_power <= set(power):
            raise SimulationInvalid(
                "missing_result",
                "ac result is missing complete source-power branch data",
                analysis=name,
            )
        _require_finite(power["total_w"], "ac.source_power.total_w", name)
        branch_names = set(power["per_source_w"])
        if (
            branch_names != set(power["source_currents_a"])
            or branch_names != set(power["source_voltages_v"])
        ):
            raise SimulationInvalid(
                "inconsistent_result",
                "ac source-power branch names are inconsistent",
                analysis=name,
            )
        for field in ("per_source_w", "source_currents_a", "source_voltages_v"):
            for branch, value in power[field].items():
                _require_finite(value, f"ac.source_power.{field}.{branch}", name)


def metric(value: Any, unit: str, *, status: str = "valid", **metadata) -> dict:
    """One JSON-safe, unit-bearing scalar or structured measurement."""
    out = {"value": value, "unit": str(unit), "status": str(status)}
    out.update(metadata)
    return out


def source_power_metric(power: Mapping[str, Any]) -> dict:
    """Unit-bearing total and per-source branch power measurement."""
    per_source = power["per_source_w"]
    currents = power["source_currents_a"]
    voltages = power["source_voltages_v"]
    branches = {
        name: {
            "voltage": metric(float(voltages[name]), "V"),
            "current": metric(float(currents[name]), "A"),
            "power": metric(float(per_source[name]), "W"),
        }
        for name in per_source
    }
    return metric(float(power["total_w"]), "W", branches=branches)


def saturation_metric(regions: Mapping[str, Mapping[str, Any]]) -> dict:
    """Unit-bearing aggregate and per-MOS saturation result."""
    devices: dict[str, dict] = {}
    known: list[Mapping[str, Any]] = []
    for name, row in regions.items():
        if row.get("status") == "unsupported":
            devices[name] = metric(None, "boolean", status="unsupported")
            continue
        known.append(row)
        devices[name] = metric(
            bool(row["saturated"]),
            "boolean",
            vds=metric(float(row["vds_v"]), "V"),
            vdsat=metric(float(row["vdsat_v"]), "V"),
            headroom=metric(float(row["headroom_v"]), "V"),
        )
    return metric(
        bool(known) and all(row.get("saturated") is True for row in known),
        "boolean",
        devices=devices,
        status="valid" if known else "unsupported",
    )


def _settling_measurement(transient: Mapping[str, Any], tolerance: float = 1e-3) -> dict:
    t = np.asarray(transient["t"], float)
    y = np.asarray(transient["output"], float)
    initial = float(y[0])
    final = float(y[-1])
    span = max(abs(final - initial), abs(final), np.finfo(float).eps)
    limit = float(tolerance) * span
    outside = np.flatnonzero(np.abs(y - final) > limit)
    if outside.size and int(outside[-1]) == len(y) - 1:
        return metric(None, "s", status="not_settled", tolerance=float(tolerance))
    index = int(outside[-1] + 1) if outside.size else 0
    return metric(
        float(t[index] - t[0]), "s", tolerance=float(tolerance),
        final_value=final, final_value_unit="V",
    )


def summarize_design_metrics(
    spec,
    results: Mapping[str, Mapping[str, Any]],
    *,
    noise_band: tuple[float, float] | None = None,
) -> dict:
    """Build the stable LLM-facing measurement surface from analysis results."""
    out: dict[str, dict] = {}
    ac = results.get("ac")
    if ac is not None:
        from .frequency_metrics import phase_margin, unity_gain_freq

        out["gain"] = metric(float(ac["Av_dc_dB"]), "dB")
        ugf = float(unity_gain_freq(ac["freqs"], ac["response"]))
        pm = float(phase_margin(ac["freqs"], ac["response"]))
        out["unity_gain_frequency"] = metric(
            ugf if np.isfinite(ugf) else None, "Hz",
            status="valid" if np.isfinite(ugf) else "no_crossing",
        )
        out["phase_margin"] = metric(
            pm if np.isfinite(pm) else None, "deg",
            status="valid" if np.isfinite(pm) else "no_crossing",
        )
        out["dc_source_power"] = source_power_metric(ac["source_power"])
        out["saturation"] = saturation_metric(ac.get("operating_regions", {}))

    noise = results.get("noise")
    if noise is not None:
        cfg = (spec.analyses or {}).get("noise", {})
        band = (
            noise_band
            if noise_band is not None
            else cfg.get(
                "band",
                [float(noise["freqs"][0]), float(noise["freqs"][-1])],
            )
        )
        from .noise_solver import band_rms

        lo, hi = map(float, band)
        out["integrated_output_noise"] = metric(
            band_rms(noise["freqs"], noise["out_psd"], lo, hi), "V_rms",
            integration_band_hz=[lo, hi],
        )
        out["integrated_input_noise"] = metric(
            band_rms(noise["freqs"], noise["irn_psd"], lo, hi), "V_rms",
            integration_band_hz=[lo, hi],
        )

    transient = results.get("transient")
    if transient is not None:
        cfg = (spec.analyses or {}).get("transient", {})
        out["settling_time"] = _settling_measurement(
            transient, float(cfg.get("settling_tolerance", 1e-3))
        )
    return out
