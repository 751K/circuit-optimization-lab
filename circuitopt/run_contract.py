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


class SignoffConfigurationError(SimulationInvalid, ValueError):
    """A signoff request is ambiguous or cannot be evaluated from this DUT."""

    def __init__(self, message: str):
        super().__init__(
            "signoff_configuration",
            str(message),
            analysis="signoff",
        )


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


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SignoffConfigurationError(f"{path} must be an object")
    return value


def _weighted_signal(
    values: Mapping[str, Any],
    weights: Mapping[str, Any],
    *,
    path: str,
    dtype=float,
):
    if not weights:
        raise SignoffConfigurationError(f"{path} must contain at least one node")
    total = None
    for node, raw_weight in weights.items():
        if node not in values:
            raise SignoffConfigurationError(
                f"{path} references unavailable node {node!r}")
        weight = float(raw_weight)
        contribution = weight * np.asarray(values[node], dtype=dtype)
        total = contribution if total is None else total + contribution
    _require_finite(total, path, "signoff")
    return total


def _require_analysis(cfg: Mapping[str, Any], expected: str, path: str) -> None:
    actual = cfg.get("analysis")
    if actual != expected:
        raise SignoffConfigurationError(
            f"{path}.analysis must explicitly be {expected!r}")


def _reject_unknown(cfg: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise SignoffConfigurationError(
            f"{path} has unknown key(s): {', '.join(unknown)}")


def validate_signoff_config(
    spec,
    analyses: Mapping[str, Any] | None = None,
) -> None:
    """Validate signoff structure and DUT references before any solver work."""
    if spec.signoff is None:
        return
    config = _require_mapping(spec.signoff, "signoff")
    _reject_unknown(config, {"measurements", "constraints"}, "signoff")
    measurements = _require_mapping(config.get("measurements"), "signoff.measurements")
    constraints = _require_mapping(config.get("constraints"), "signoff.constraints")
    if not constraints:
        raise SignoffConfigurationError(
            "signoff.constraints must contain at least one constraint")
    _reject_unknown(
        measurements,
        {"phase_margin", "settling_time", "checkpoint_error", "noise",
         "saturation"},
        "signoff.measurements",
    )

    analyses = analyses if analyses is not None else (spec.analyses or {})
    topo = spec.topology
    produced = set()
    if "ac" in analyses:
        produced.update({"gain", "unity_gain_frequency", "dc_source_power"})

    if "phase_margin" in measurements:
        path = "signoff.measurements.phase_margin"
        cfg = _require_mapping(measurements["phase_margin"], path)
        _reject_unknown(
            cfg,
            {"analysis", "injection_source", "return_signal", "polarity",
             "return_scale"},
            path,
        )
        _require_analysis(cfg, "ac", path)
        if "ac" not in analyses:
            raise SignoffConfigurationError(f"{path} requires analyses.ac")
        source = str(cfg.get("injection_source", "")).strip()
        if source not in topo.vsource_index:
            raise SignoffConfigurationError(
                f"{path}.injection_source {source!r} is not a DUT voltage source")
        probe = next(
            (item for item in topo.vsources if item[0] == source),
            None,
        )
        if probe is None:
            raise SignoffConfigurationError(
                f"{path}.injection_source {source!r} has no voltage-source element")
        _, probe_p, probe_q, probe_dc = probe
        if probe_p not in topo.solved or probe_q not in topo.solved:
            raise SignoffConfigurationError(
                f"{path}.injection_source must break between two solved nodes")
        if isinstance(probe_dc, str) or float(probe_dc) != 0.0:
            raise SignoffConfigurationError(
                f"{path}.injection_source must be a constant 0 V loop break")
        active = sorted(
            name for name, value in topo.ac_drives.items()
            if float(value) != 0.0
        )
        if active != [source]:
            raise SignoffConfigurationError(
                f"{path} requires exactly injection source {source!r} to be "
                f"non-zero in ac_drives; found {active}")
        signal = _require_mapping(cfg.get("return_signal"), f"{path}.return_signal")
        if not signal:
            raise SignoffConfigurationError(f"{path}.return_signal must not be empty")
        unknown_nodes = sorted(set(signal) - set(topo.solved))
        if unknown_nodes:
            raise SignoffConfigurationError(
                f"{path}.return_signal references unsolved node(s): "
                + ", ".join(unknown_nodes))
        polarity = cfg.get("polarity")
        if polarity not in {-1, 1}:
            raise SignoffConfigurationError(f"{path}.polarity must be -1 or 1")
        if "return_scale" in cfg and (
            not np.isfinite(float(cfg["return_scale"]))
            or float(cfg["return_scale"]) <= 0.0
        ):
            raise SignoffConfigurationError(
                f"{path}.return_scale must be a positive finite number")
        produced.update({
            "phase_margin", "loop_unity_gain_frequency", "loop_gain_dc",
        })

    for settle_name in ("settling_time",):
        if settle_name not in measurements:
            continue
        path = f"signoff.measurements.{settle_name}"
        cfg = _require_mapping(measurements[settle_name], path)
        _reject_unknown(
            cfg,
            {"analysis", "signal", "target", "start_time", "end_time", "tolerance"},
            path,
        )
        _require_analysis(cfg, "transient", path)
        if "transient" not in analyses:
            raise SignoffConfigurationError(f"{path} requires analyses.transient")
        for required in ("signal", "target", "start_time", "tolerance"):
            if required not in cfg:
                raise SignoffConfigurationError(f"{path}.{required} is required")
        signal = _require_mapping(cfg["signal"], f"{path}.signal")
        if not signal:
            raise SignoffConfigurationError(f"{path}.signal must not be empty")
        unknown_nodes = sorted(set(signal) - set(topo.solved))
        if unknown_nodes:
            raise SignoffConfigurationError(
                f"{path}.signal references unsolved node(s): "
                + ", ".join(unknown_nodes))
        if not np.isfinite(float(cfg["target"])):
            raise SignoffConfigurationError(f"{path}.target must be finite")
        if float(cfg["start_time"]) < 0.0:
            raise SignoffConfigurationError(f"{path}.start_time must be non-negative")
        _settling_tolerance(cfg)
        produced.add(settle_name)

    # ``checkpoint_error`` gates |weighted signal - target| at explicit instants
    # rather than over a window.  A sampled system's spec lives at the instants
    # its successor observes: the MDAC brief's "output common mode within 20 mV
    # of VDD/2" is a statement about quiescence and about the end of the hold
    # phase, not about the middle of a 450 mV slew (where a class-A stage's CM
    # excursion is physical, unavoidable, and unobserved).
    if "checkpoint_error" in measurements:
        path = "signoff.measurements.checkpoint_error"
        cfg = _require_mapping(measurements["checkpoint_error"], path)
        _reject_unknown(
            cfg, {"analysis", "signal", "target", "checkpoints"}, path)
        _require_analysis(cfg, "transient", path)
        if "transient" not in analyses:
            raise SignoffConfigurationError(f"{path} requires analyses.transient")
        for required in ("signal", "target", "checkpoints"):
            if required not in cfg:
                raise SignoffConfigurationError(f"{path}.{required} is required")
        signal = _require_mapping(cfg["signal"], f"{path}.signal")
        if not signal:
            raise SignoffConfigurationError(f"{path}.signal must not be empty")
        unknown_nodes = sorted(set(signal) - set(topo.solved))
        if unknown_nodes:
            raise SignoffConfigurationError(
                f"{path}.signal references unsolved node(s): "
                + ", ".join(unknown_nodes))
        if not np.isfinite(float(cfg["target"])):
            raise SignoffConfigurationError(f"{path}.target must be finite")
        checkpoints = list(cfg["checkpoints"])
        if not checkpoints:
            raise SignoffConfigurationError(
                f"{path}.checkpoints must not be empty")
        for checkpoint in checkpoints:
            entry = _require_mapping(checkpoint, f"{path}.checkpoints[]")
            _reject_unknown(entry, {"name", "time"}, f"{path}.checkpoints[]")
            for required in ("name", "time"):
                if required not in entry:
                    raise SignoffConfigurationError(
                        f"{path}.checkpoints[].{required} is required")
            if not np.isfinite(float(entry["time"])) or float(entry["time"]) < 0.0:
                raise SignoffConfigurationError(
                    f"{path}.checkpoints[].time must be finite and non-negative")
        produced.add("checkpoint_error")

    if "noise" in measurements:
        path = "signoff.measurements.noise"
        cfg = _require_mapping(measurements["noise"], path)
        _reject_unknown(cfg, {"analysis", "band", "references"}, path)
        _require_analysis(cfg, "noise", path)
        if "noise" not in analyses:
            raise SignoffConfigurationError(f"{path} requires analyses.noise")
        if "band" not in cfg:
            raise SignoffConfigurationError(f"{path}.band is required")
        band = list(cfg["band"])
        if len(band) != 2 or float(band[0]) < 0.0 or float(band[1]) <= float(band[0]):
            raise SignoffConfigurationError(
                f"{path}.band must be [low, high] with 0 <= low < high")
        references = list(cfg.get("references", ()))
        if not references or len(set(references)) != len(references):
            raise SignoffConfigurationError(
                f"{path}.references must contain unique input and/or output values")
        unknown = sorted(set(references) - {"input", "output"})
        if unknown:
            raise SignoffConfigurationError(
                f"{path}.references has unknown value(s): {', '.join(unknown)}")
        if "input" in references:
            produced.add("integrated_input_noise")
        if "output" in references:
            produced.add("integrated_output_noise")

    if "saturation" in measurements:
        path = "signoff.measurements.saturation"
        cfg = _require_mapping(measurements["saturation"], path)
        _reject_unknown(
            cfg, {"analysis", "devices", "minimum_headroom", "checkpoints"}, path)
        analysis = cfg.get("analysis")
        if analysis not in {"ac", "transient"}:
            raise SignoffConfigurationError(
                f"{path}.analysis must explicitly be 'ac' or 'transient'")
        if analysis not in analyses:
            raise SignoffConfigurationError(
                f"{path} requires analyses.{analysis}")
        devices = [str(name) for name in cfg.get("devices", ())]
        if not devices or len(set(devices)) != len(devices):
            raise SignoffConfigurationError(
                f"{path}.devices must contain unique MOS device names")
        known_devices = {name for name, *_ in topo.devices}
        unknown = sorted(set(devices) - known_devices)
        if unknown:
            raise SignoffConfigurationError(
                f"{path}.devices references unknown MOS device(s): "
                + ", ".join(unknown))
        if "minimum_headroom" not in cfg or not np.isfinite(
                float(cfg["minimum_headroom"])):
            raise SignoffConfigurationError(
                f"{path}.minimum_headroom must be finite")
        checkpoints = cfg.get("checkpoints")
        if analysis == "transient":
            if not isinstance(checkpoints, list) or not checkpoints:
                raise SignoffConfigurationError(
                    f"{path}.checkpoints must be a non-empty array for transient")
            names = set()
            for index, checkpoint in enumerate(checkpoints):
                cpath = f"{path}.checkpoints[{index}]"
                checkpoint = _require_mapping(checkpoint, cpath)
                _reject_unknown(checkpoint, {"name", "time"}, cpath)
                name = str(checkpoint.get("name", "")).strip()
                if not name or name in names:
                    raise SignoffConfigurationError(
                        f"{cpath}.name must be non-empty and unique")
                names.add(name)
                if "time" not in checkpoint or not np.isfinite(
                        float(checkpoint["time"])):
                    raise SignoffConfigurationError(
                        f"{cpath}.time must be finite")
        elif checkpoints is not None:
            raise SignoffConfigurationError(
                f"{path}.checkpoints is only valid for transient saturation")
        produced.add("saturation")

    for name, raw_limits in constraints.items():
        if name not in produced:
            raise SignoffConfigurationError(
                f"signoff constraint {name!r} has no configured measurement")
        limits = _require_mapping(raw_limits, f"signoff.constraints.{name}")
        _reject_unknown(limits, {"min", "max", "equals"},
                        f"signoff.constraints.{name}")
        if not limits:
            raise SignoffConfigurationError(
                f"signoff.constraints.{name} must not be empty")
        for key in ("min", "max"):
            if key in limits and not np.isfinite(float(limits[key])):
                raise SignoffConfigurationError(
                    f"signoff.constraints.{name}.{key} must be finite")


def _phase_margin_measurement(spec, ac: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict:
    """Measure PM only from a declared voltage-source loop break and return signal."""
    cfg = _require_mapping(cfg, "signoff.measurements.phase_margin")
    _require_analysis(cfg, "ac", "signoff.measurements.phase_margin")
    source = str(cfg.get("injection_source", "")).strip()
    if not source:
        raise SignoffConfigurationError(
            "signoff.measurements.phase_margin.injection_source is required")
    topo = spec.topology
    if source not in topo.vsource_index:
        raise SignoffConfigurationError(
            f"phase_margin injection_source {source!r} is not a DUT voltage source")
    probe = next((item for item in topo.vsources if item[0] == source), None)
    if probe is None:
        raise SignoffConfigurationError(
            f"phase_margin injection_source {source!r} has no source element")
    _, reference_node, return_node, probe_dc = probe
    if reference_node not in topo.solved or return_node not in topo.solved:
        raise SignoffConfigurationError(
            "phase-margin loop source must break between two solved nodes")
    if isinstance(probe_dc, str) or float(probe_dc) != 0.0:
        raise SignoffConfigurationError(
            "phase-margin loop source must be a constant 0 V break")

    stimulus = _require_mapping(ac.get("ac_stimulus"), "ac.ac_stimulus")
    drives = _require_mapping(stimulus.get("drives"), "ac.ac_stimulus.drives")
    drive = complex(drives.get(source, 0.0))
    if source not in drives or abs(drive) == 0.0:
        raise SignoffConfigurationError(
            f"phase_margin injection_source {source!r} has no non-zero AC drive")
    active = sorted(name for name, value in drives.items()
                    if abs(complex(value)) != 0.0)
    if active != [source]:
        raise SignoffConfigurationError(
            "phase_margin requires exactly one active AC loop injection source; "
            f"found {active}")

    return_signal = _require_mapping(
        cfg.get("return_signal"),
        "signoff.measurements.phase_margin.return_signal",
    )
    node_voltages = _require_mapping(ac.get("node_voltages"), "ac.node_voltages")
    returned = _weighted_signal(
        node_voltages,
        return_signal,
        path="signoff.measurements.phase_margin.return_signal",
        dtype=complex,
    )
    reference = np.asarray(node_voltages[reference_node], dtype=complex)
    _require_finite(reference, "phase_margin.reference_signal", "signoff")
    if np.any(np.abs(reference) <= np.finfo(float).tiny):
        raise SignoffConfigurationError(
            "phase-margin injection-side reference signal contains zero")
    polarity = float(cfg.get("polarity", -1.0))
    if polarity not in {-1.0, 1.0}:
        raise SignoffConfigurationError("phase_margin.polarity must be -1 or 1")
    return_scale = float(cfg.get("return_scale", 1.0))
    if not np.isfinite(return_scale) or return_scale <= 0.0:
        raise SignoffConfigurationError(
            "phase_margin.return_scale must be a positive finite number")
    loop_gain = polarity * return_scale * returned / reference
    _require_finite(loop_gain, "phase_margin.loop_gain", "signoff")

    from .frequency_metrics import phase_margin, unity_gain_freq

    freqs = np.asarray(ac["freqs"], float)
    ugf = float(unity_gain_freq(freqs, loop_gain))
    pm = float(phase_margin(freqs, loop_gain))
    metadata = {
        "analysis": "ac",
        "response_kind": "loop_gain",
        "injection_source": source,
        "return_signal": {str(k): float(v) for k, v in return_signal.items()},
        "reference_signal": {reference_node: 1.0},
        "polarity": int(polarity),
        "return_scale": return_scale,
    }
    return {
        "phase_margin": metric(
            pm if np.isfinite(pm) else None,
            "deg",
            status="valid" if np.isfinite(pm) else "no_crossing",
            **metadata,
        ),
        "loop_unity_gain_frequency": metric(
            ugf if np.isfinite(ugf) else None,
            "Hz",
            status="valid" if np.isfinite(ugf) else "no_crossing",
            **metadata,
        ),
        "loop_gain_dc": metric(
            float(20.0 * np.log10(max(abs(loop_gain[0]), 1e-300))),
            "dB",
            **metadata,
        ),
    }


def _settling_tolerance(cfg: Mapping[str, Any]) -> tuple[float, dict]:
    tolerance = _require_mapping(
        cfg.get("tolerance"),
        "signoff.measurements.settling_time.tolerance",
    )
    if "absolute" in tolerance:
        if set(tolerance) != {"absolute"}:
            raise SignoffConfigurationError(
                "settling tolerance with absolute must not also define relative/reference")
        limit = float(tolerance["absolute"])
        metadata = {"mode": "absolute", "value": limit, "unit": "V"}
    else:
        if set(tolerance) != {"relative", "reference"}:
            raise SignoffConfigurationError(
                "relative settling tolerance requires exactly relative and reference")
        relative = float(tolerance["relative"])
        reference = float(tolerance["reference"])
        limit = relative * reference
        metadata = {
            "mode": "relative",
            "value": relative,
            "unit": "ratio",
            "reference": metric(reference, "V"),
            "absolute_limit": metric(limit, "V"),
        }
    if not np.isfinite(limit) or limit <= 0.0:
        raise SignoffConfigurationError(
            "settling tolerance must resolve to a positive finite voltage")
    return limit, metadata


def _settling_measurement(
    transient: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict:
    cfg = _require_mapping(cfg, "signoff.measurements.settling_time")
    _require_analysis(cfg, "transient", "signoff.measurements.settling_time")
    for required in ("target", "start_time"):
        if required not in cfg:
            raise SignoffConfigurationError(
                f"signoff.measurements.settling_time.{required} is required")
    t = np.asarray(transient["t"], float)
    nodes = _require_mapping(transient.get("nodes"), "transient.nodes")
    signal_cfg = _require_mapping(
        cfg.get("signal"),
        "signoff.measurements.settling_time.signal",
    )
    y = _weighted_signal(
        nodes,
        signal_cfg,
        path="signoff.measurements.settling_time.signal",
    )
    target = float(cfg["target"])
    start_time = float(cfg["start_time"])
    end_time = float(cfg.get("end_time", t[-1]))
    if not np.isfinite(target):
        raise SignoffConfigurationError("settling target must be finite")
    if start_time < t[0] or start_time > t[-1]:
        raise SignoffConfigurationError(
            "settling start_time is outside the transient time grid")
    if end_time <= start_time or end_time > t[-1]:
        raise SignoffConfigurationError(
            "settling end_time must be after start_time and inside the time grid")
    limit, tolerance = _settling_tolerance(cfg)
    window = np.flatnonzero((t >= start_time) & (t <= end_time))
    if window.size == 0:
        raise SignoffConfigurationError(
            "settling measurement window contains no transient samples")
    errors = np.abs(y[window] - target)
    outside = np.flatnonzero(errors > limit)
    metadata = {
        "analysis": "transient",
        "signal": {str(k): float(v) for k, v in signal_cfg.items()},
        "target": metric(target, "V"),
        "tolerance": tolerance,
        "window": {
            "start": metric(start_time, "s"),
            "end": metric(end_time, "s"),
        },
        "final_error": metric(float(errors[-1]), "V"),
    }
    if outside.size and int(outside[-1]) == len(window) - 1:
        return metric(None, "s", status="not_settled", **metadata)
    if not outside.size:
        # Already inside the band when the window opened, so nothing had to
        # settle. Reporting `t[window[0]] - start_time` instead measured the
        # gap between the declared start and the first sample at or after it,
        # which is float noise: a 2e-11 start against a grid sample one ULP
        # above it produced a 3.2e-27 s "settling time".
        return metric(0.0, "s", **metadata)
    index = int(window[int(outside[-1] + 1)])
    return metric(
        float(t[index] - start_time),
        "s",
        **metadata,
    )


def _noise_measurements(noise: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict:
    cfg = _require_mapping(cfg, "signoff.measurements.noise")
    _require_analysis(cfg, "noise", "signoff.measurements.noise")
    if "band" not in cfg:
        raise SignoffConfigurationError(
            "signoff.measurements.noise.band is required")
    lo, hi = map(float, cfg["band"])
    freqs = np.asarray(noise["freqs"], float)
    if lo < float(freqs[0]) or hi > float(freqs[-1]):
        raise SignoffConfigurationError(
            f"noise band [{lo}, {hi}] Hz is outside simulated range "
            f"[{float(freqs[0])}, {float(freqs[-1])}] Hz")
    references = list(cfg.get("references", ()))
    if not references:
        raise SignoffConfigurationError(
            "signoff.measurements.noise.references must select input and/or output")
    out = {}
    metadata = {"analysis": "noise", "integration_band_hz": [lo, hi]}
    for reference in references:
        if reference == "input":
            out["integrated_input_noise"] = metric(
                _integrated_noise_rms(freqs, noise["irn_psd"], lo, hi),
                "V_rms",
                reference="input",
                **metadata,
            )
        elif reference == "output":
            out["integrated_output_noise"] = metric(
                _integrated_noise_rms(freqs, noise["out_psd"], lo, hi),
                "V_rms",
                reference="output",
                **metadata,
            )
        else:
            raise SignoffConfigurationError(
                f"unknown noise reference {reference!r}; expected input or output")
    return out


def _integrated_noise_rms(freqs, psd, lo: float, hi: float) -> float:
    """Integrate PSD over exact declared endpoints, including interpolated edges."""
    freqs = np.asarray(freqs, float)
    psd = np.asarray(psd, float)
    order = np.argsort(freqs)
    freqs, psd = freqs[order], psd[order]
    interior = (freqs > lo) & (freqs < hi)
    grid = np.concatenate(([lo], freqs[interior], [hi]))
    values = np.interp(grid, freqs, psd)
    integral = float(np.trapezoid(values, grid))
    if not np.isfinite(integral) or integral < 0.0:
        raise SimulationInvalid(
            "non_finite_result",
            "noise integration produced an invalid power",
            analysis="signoff",
        )
    return float(np.sqrt(integral))


def _saturation_from_regions(
    regions: Mapping[str, Any],
    requested: list[str],
    minimum: float,
) -> dict:
    """Build one aggregate saturation metric from precomputed operating regions."""
    missing = [name for name in requested if name not in regions]
    if missing:
        raise SignoffConfigurationError(
            "saturation references unavailable MOS device(s): " + ", ".join(missing))
    unsupported = [
        name for name in requested
        if regions[name].get("status") == "unsupported"
    ]
    if unsupported:
        raise SignoffConfigurationError(
            "saturation is unsupported for device(s): " + ", ".join(unsupported))

    devices = {}
    passed = True
    for name in requested:
        row = regions[name]
        headroom = float(row["headroom_v"])
        saturated = bool(row["saturated"]) and headroom >= minimum
        passed = passed and saturated
        devices[name] = metric(
            saturated,
            "boolean",
            vds=metric(float(row["vds_v"]), "V"),
            vdsat=metric(float(row["vdsat_v"]), "V"),
            headroom=metric(headroom, "V"),
        )
    return metric(passed, "boolean", devices=devices)


def _checkpoint_error_measurement(
    transient: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict:
    """Worst |weighted signal - target| over the configured instants."""
    path = "signoff.measurements.checkpoint_error"
    cfg = _require_mapping(cfg, path)
    t = np.asarray(transient["t"], float)
    nodes = _require_mapping(transient.get("nodes"), "transient.nodes")
    y = _weighted_signal(
        nodes, _require_mapping(cfg.get("signal"), f"{path}.signal"),
        path=f"{path}.signal")
    target = float(cfg["target"])
    checkpoint_metrics = {}
    worst = 0.0
    for checkpoint in cfg["checkpoints"]:
        name = str(checkpoint["name"])
        time = float(checkpoint["time"])
        if time < float(t[0]) or time > float(t[-1]):
            raise SignoffConfigurationError(
                f"{path} checkpoint {name!r} at {time:g} s is outside transient "
                f"range [{float(t[0]):g}, {float(t[-1]):g}] s")
        value = float(np.interp(time, t, y))
        error = abs(value - target)
        checkpoint_metrics[name] = metric(
            error, "V", time=metric(time, "s"), signal_value=metric(value, "V"))
        worst = max(worst, error)
    return metric(
        worst, "V", analysis="transient",
        target=metric(target, "V"), checkpoints=checkpoint_metrics)


def _saturation_measurement(
    spec,
    results: Mapping[str, Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> dict:
    cfg = _require_mapping(cfg, "signoff.measurements.saturation")
    if "minimum_headroom" not in cfg:
        raise SignoffConfigurationError(
            "signoff.measurements.saturation.minimum_headroom is required")
    requested = [str(name) for name in cfg.get("devices", ())]
    if not requested:
        raise SignoffConfigurationError(
            "signoff.measurements.saturation.devices must not be empty")
    minimum = float(cfg.get("minimum_headroom", 0.0))
    if not np.isfinite(minimum):
        raise SignoffConfigurationError(
            "saturation minimum_headroom must be finite")
    analysis = cfg.get("analysis")
    if analysis == "ac":
        ac = results.get("ac")
        if ac is None:
            raise SignoffConfigurationError(
                "AC saturation requires the ac operating-point result")
        regions = _require_mapping(
            ac.get("operating_regions"), "ac.operating_regions")
        result = _saturation_from_regions(regions, requested, minimum)
        result.update({
            "analysis": "ac",
            "minimum_headroom": metric(minimum, "V"),
        })
        return result
    if analysis != "transient":
        raise SignoffConfigurationError(
            "saturation.analysis must explicitly be 'ac' or 'transient'")

    transient = results.get("transient")
    if transient is None:
        raise SignoffConfigurationError(
            "transient saturation requires the transient analysis result")
    t = np.asarray(transient["t"], float)
    nodes = _require_mapping(transient.get("nodes"), "transient.nodes")
    for checkpoint in cfg.get("checkpoints", ()):
        name = str(checkpoint["name"])
        time = float(checkpoint["time"])
        if time < float(t[0]) or time > float(t[-1]):
            raise SignoffConfigurationError(
                f"saturation checkpoint {name!r} at {time:g} s is outside "
                f"transient range [{float(t[0]):g}, {float(t[-1]):g}] s")
    devices = spec.binding().build(spec.sizes)
    from .dc_measurements import operating_regions

    checkpoint_metrics = {}
    passed = True
    for checkpoint in cfg.get("checkpoints", ()):
        name = str(checkpoint["name"])
        time = float(checkpoint["time"])
        node_values = {
            node: float(np.interp(time, t, np.asarray(values, float)))
            for node, values in nodes.items()
        }
        regions = operating_regions(
            spec.topology, spec.bias, node_values, devices)
        checkpoint_metric = _saturation_from_regions(
            regions, requested, minimum)
        checkpoint_metric["time"] = metric(time, "s")
        checkpoint_metrics[name] = checkpoint_metric
        passed = passed and bool(checkpoint_metric["value"])
    return metric(
        passed,
        "boolean",
        analysis="transient",
        minimum_headroom=metric(minimum, "V"),
        checkpoints=checkpoint_metrics,
    )


def summarize_design_metrics(
    spec,
    results: Mapping[str, Mapping[str, Any]],
) -> dict:
    """Build unit-bearing measurements, gating signoff-only metrics explicitly."""
    out: dict[str, dict] = {}
    ac = results.get("ac")
    if ac is not None:
        from .frequency_metrics import unity_gain_freq

        out["gain"] = metric(float(ac["Av_dc_dB"]), "dB")
        ugf = float(unity_gain_freq(ac["freqs"], ac["response"]))
        out["unity_gain_frequency"] = metric(
            ugf if np.isfinite(ugf) else None, "Hz",
            status="valid" if np.isfinite(ugf) else "no_crossing",
        )
        out["dc_source_power"] = source_power_metric(ac["source_power"])

    configured = dict((spec.signoff or {}).get("measurements", {}))
    if "phase_margin" in configured:
        if ac is None:
            raise SignoffConfigurationError(
                "phase_margin requires the ac analysis result")
        out.update(_phase_margin_measurement(spec, ac, configured["phase_margin"]))
    if "settling_time" in configured:
        transient = results.get("transient")
        if transient is None:
            raise SignoffConfigurationError(
                "settling_time requires the transient analysis result")
        out["settling_time"] = _settling_measurement(
            transient, configured["settling_time"])
    if "checkpoint_error" in configured:
        transient = results.get("transient")
        if transient is None:
            raise SignoffConfigurationError(
                "checkpoint_error requires the transient analysis result")
        out["checkpoint_error"] = _checkpoint_error_measurement(
            transient, configured["checkpoint_error"])
    if "noise" in configured:
        noise = results.get("noise")
        if noise is None:
            raise SignoffConfigurationError(
                "noise measurement requires the noise analysis result")
        out.update(_noise_measurements(noise, configured["noise"]))
    if "saturation" in configured:
        out["saturation"] = _saturation_measurement(
            spec, results, configured["saturation"])
    return out


def _constraint_result(name: str, observed: Mapping[str, Any], limits: Mapping[str, Any]):
    value = observed.get("value")
    unit = str(observed.get("unit", ""))
    status = str(observed.get("status", "invalid"))
    valid = status == "valid" and value is not None
    passed = valid
    checks = {}
    normalized_margins = []
    if "min" in limits:
        limit = float(limits["min"])
        margin_value = float(value) - limit if valid else None
        check_passed = valid and float(value) >= limit
        checks["min"] = {
            "limit": metric(limit, unit),
            "passed": bool(check_passed),
            "margin": metric(
                margin_value, unit, status="valid" if margin_value is not None else status),
        }
        passed = passed and bool(check_passed)
        if margin_value is not None:
            normalized_margins.append(
                margin_value / max(abs(limit), np.finfo(float).eps))
    if "max" in limits:
        limit = float(limits["max"])
        margin_value = limit - float(value) if valid else None
        check_passed = valid and float(value) <= limit
        checks["max"] = {
            "limit": metric(limit, unit),
            "passed": bool(check_passed),
            "margin": metric(
                margin_value, unit, status="valid" if margin_value is not None else status),
        }
        passed = passed and bool(check_passed)
        if margin_value is not None:
            normalized_margins.append(
                margin_value / max(abs(limit), np.finfo(float).eps))
    if "equals" in limits:
        expected = limits["equals"]
        check_passed = valid and value == expected
        checks["equals"] = {
            "expected": metric(expected, unit),
            "passed": bool(check_passed),
        }
        passed = passed and bool(check_passed)
        normalized_margins.append(1.0 if check_passed else -1.0)
    if not normalized_margins and status == "not_settled":
        final_error = observed.get("final_error", {}).get("value")
        tolerance = observed.get("tolerance", {})
        if tolerance.get("mode") == "relative":
            tolerance_limit = tolerance.get("absolute_limit", {}).get("value")
        else:
            tolerance_limit = tolerance.get("value")
        if (
            final_error is not None
            and tolerance_limit is not None
            and np.isfinite(float(final_error))
            and np.isfinite(float(tolerance_limit))
            and float(tolerance_limit) > 0.0
        ):
            # A missing settling time is still rankable: zero margin occurs at
            # the declared error-band edge and more negative means farther out.
            normalized_margins.append(
                1.0 - float(final_error) / float(tolerance_limit))
    if not checks:
        raise SignoffConfigurationError(
            f"signoff constraint {name!r} requires min, max, and/or equals")
    normalized_margin = min(normalized_margins) if normalized_margins else -np.inf
    return {
        "observed": dict(observed),
        "checks": checks,
        "passed": bool(passed),
        "normalized_margin": (
            float(normalized_margin) if np.isfinite(normalized_margin) else None
        ),
    }


def evaluate_signoff(
    spec,
    results: Mapping[str, Mapping[str, Any]],
) -> dict:
    """Return the single stable signoff envelope used by CLI and service APIs."""
    # Validate against analyses actually present in this result envelope. This keeps
    # explicit ``run_analysis_suite(..., analyses=...)`` overrides correct and makes
    # a selected subset fail clearly when it omits a required signoff analysis.
    validate_signoff_config(spec, results)
    measurements = summarize_design_metrics(spec, results)
    config = spec.signoff or {}
    constraints_cfg = dict(config.get("constraints", {}))
    if not config:
        return {
            "status": "not_configured",
            "measurements": measurements,
            "constraints": {},
            "passed": None,
            "worst_case": None,
        }
    if not constraints_cfg:
        raise SignoffConfigurationError(
            "signoff.constraints must contain at least one constraint")

    constraints = {}
    for name, limits in constraints_cfg.items():
        if name not in measurements:
            raise SignoffConfigurationError(
                f"signoff constraint {name!r} has no configured measurement")
        constraints[name] = _constraint_result(
            str(name),
            measurements[name],
            _require_mapping(limits, f"signoff.constraints.{name}"),
        )
    passed = all(item["passed"] for item in constraints.values())
    worst_name, worst = min(
        constraints.items(),
        key=lambda item: (
            item[1]["normalized_margin"]
            if item[1]["normalized_margin"] is not None
            else -np.inf
        ),
    )
    return {
        "status": "pass" if passed else "fail",
        "measurements": measurements,
        "constraints": constraints,
        "passed": bool(passed),
        "worst_case": {
            "measurement": worst_name,
            "passed": worst["passed"],
            "normalized_margin": worst["normalized_margin"],
        },
    }
