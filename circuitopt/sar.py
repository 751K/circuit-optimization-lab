"""Closed-loop SAR conversion driven by full-charge transient simulations."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .adc import (average_supply_power, average_waveform_source_power,
                  dynamic_metrics, static_ramp_metrics)
from .circuit_loader import CircuitSpec
from .transient_solver import transient


def _required(cfg: Mapping, name: str):
    if name not in cfg:
        raise ValueError(f"adc.{name} is required")
    return cfg[name]


def _sar_config(spec: CircuitSpec, override=None) -> dict:
    cfg = dict(spec.adc or {})
    cfg.update(override or {})
    if str(cfg.get("type", "sar")).lower() != "sar":
        raise ValueError("this workflow requires adc.type='sar'")
    n_bits = int(_required(cfg, "n_bits"))
    bit_inputs = tuple(str(v) for v in _required(cfg, "bit_inputs"))
    if n_bits < 1 or len(bit_inputs) != n_bits:
        raise ValueError("adc.bit_inputs length must equal adc.n_bits")
    cfg.update({
        "n_bits": n_bits,
        "bit_inputs": bit_inputs,
        "vref": float(_required(cfg, "vref")),
        "sample_input": str(_required(cfg, "sample_input")),
        "sample_bar_input": str(_required(cfg, "sample_bar_input")),
        "comparator_node": str(_required(cfg, "comparator_node")),
        "comparator_threshold": float(_required(cfg, "comparator_threshold")),
        "sample_end": float(_required(cfg, "sample_end")),
        "bit_period": float(_required(cfg, "bit_period")),
        "edge_time": float(_required(cfg, "edge_time")),
        "high_means_clear": bool(cfg.get("high_means_clear", True)),
        "points_per_period": int(cfg.get("points_per_period", 100)),
        "input_common_mode": float(cfg.get("input_common_mode", cfg["vref"] / 2.0)),
    })
    if cfg["vref"] <= 0.0 or min(cfg["sample_end"], cfg["bit_period"], cfg["edge_time"]) <= 0.0:
        raise ValueError("ADC reference and timing values must be positive")
    if cfg["edge_time"] * 4 >= cfg["bit_period"]:
        raise ValueError("adc.edge_time must be less than one quarter bit_period")
    if cfg["points_per_period"] < 8:
        raise ValueError("adc.points_per_period must be at least 8")
    dummy = cfg.get("dummy_input")
    cfg["dummy_input"] = None if dummy is None else str(dummy)
    bit_inputs_bar = cfg.get("bit_inputs_bar")
    if bit_inputs_bar is not None:
        bit_inputs_bar = tuple(str(v) for v in bit_inputs_bar)
        if len(bit_inputs_bar) != n_bits:
            raise ValueError("adc.bit_inputs_bar length must equal adc.n_bits")
    cfg["bit_inputs_bar"] = bit_inputs_bar
    dummy_bar = cfg.get("dummy_input_bar")
    cfg["dummy_input_bar"] = None if dummy_bar is None else str(dummy_bar)
    if bit_inputs_bar is None and cfg["dummy_input_bar"] is not None:
        raise ValueError("dummy_input_bar requires differential bit_inputs_bar")
    if bit_inputs_bar is not None and ((cfg["dummy_input"] is None) !=
                                       (cfg["dummy_input_bar"] is None)):
        raise ValueError("differential dummy inputs must be provided as a pair")
    return cfg


def sar_time_grid(spec: CircuitSpec, config=None) -> np.ndarray:
    """Uniform grid containing every sample/trial/decision edge."""
    cfg = _sar_config(spec, config)
    tstop = cfg["sample_end"] + (cfg["n_bits"] + 1) * cfg["bit_period"]
    step = min(cfg["edge_time"], cfg["bit_period"] / cfg["points_per_period"])
    count = int(np.ceil(tstop / step))
    return np.linspace(0.0, tstop, count + 1)


def _wave(tgrid, events) -> np.ndarray:
    ordered = sorted((float(t), float(v)) for t, v in events)
    compact = []
    for item in ordered:
        if compact and item[0] == compact[-1][0]:
            compact[-1] = item
        else:
            compact.append(item)
    return np.interp(tgrid, [x for x, _ in compact], [y for _, y in compact])


def sar_input_waveforms(spec: CircuitSpec, vin: float, decisions: Sequence[int | None],
                        trial_index: int, *, config=None, tgrid=None) -> dict:
    """PWL sampling/CDAC controls for one replayed SAR decision."""
    cfg = _sar_config(spec, config)
    if not 0 <= trial_index < cfg["n_bits"]:
        raise ValueError("trial_index is outside the SAR bit range")
    if len(decisions) != cfg["n_bits"]:
        raise ValueError("decisions length must equal adc.n_bits")
    tgrid = sar_time_grid(spec, cfg) if tgrid is None else np.asarray(tgrid, float)
    sample_end = cfg["sample_end"]
    period = cfg["bit_period"]
    edge = cfg["edge_time"]
    tstop = float(tgrid[-1])
    sample = _wave(tgrid, [
        (0.0, cfg["vref"]),
        (sample_end - edge, cfg["vref"]),
        (sample_end, 0.0),
        (tstop, 0.0),
    ])
    out = {
        cfg["sample_input"]: sample,
        cfg["sample_bar_input"]: cfg["vref"] - sample,
    }
    hold_start = sample_end + edge
    hold_done = sample_end + 2.0 * edge
    differential = cfg["bit_inputs_bar"] is not None
    common_mode = cfg["input_common_mode"]
    sampled_p = common_mode + 0.5 * vin if differential else vin
    sampled_n = common_mode - 0.5 * vin
    for bit, key in enumerate(cfg["bit_inputs"]):
        baseline = common_mode if differential else 0.0
        events = [(0.0, sampled_p), (hold_start, sampled_p), (hold_done, baseline)]
        if bit <= trial_index:
            trial_start = sample_end + (bit + 0.5) * period
            decision_time = sample_end + (bit + 1.0) * period
            events.extend([(trial_start, baseline), (trial_start + edge, cfg["vref"])])
            if decisions[bit] == 0:
                events.extend([(decision_time, cfg["vref"]),
                               (decision_time + edge, baseline)])
        events.append((tstop, events[-1][1]))
        out[key] = _wave(tgrid, events)
        if differential:
            bar_key = cfg["bit_inputs_bar"][bit]
            bar_events = [(0.0, sampled_n), (hold_start, sampled_n),
                          (hold_done, common_mode)]
            if bit <= trial_index:
                trial_start = sample_end + (bit + 0.5) * period
                decision_time = sample_end + (bit + 1.0) * period
                bar_events.extend([(trial_start, common_mode),
                                   (trial_start + edge, 0.0)])
                if decisions[bit] == 0:
                    bar_events.extend([(decision_time, 0.0),
                                       (decision_time + edge, common_mode)])
            bar_events.append((tstop, bar_events[-1][1]))
            out[bar_key] = _wave(tgrid, bar_events)
    if cfg["dummy_input"] is not None:
        out[cfg["dummy_input"]] = _wave(tgrid, [
            (0.0, sampled_p), (hold_start, sampled_p), (hold_done, common_mode),
            (tstop, common_mode)])
    if cfg["dummy_input_bar"] is not None:
        out[cfg["dummy_input_bar"]] = _wave(tgrid, [
            (0.0, sampled_n), (hold_start, sampled_n), (hold_done, common_mode),
            (tstop, common_mode)])
    return out


def run_sar_conversion(spec: CircuitSpec, vin: float, *, config=None,
                       corner: str | None = None) -> dict:
    """Run one closed-loop SAR conversion using physical comparator decisions.

    Each bit replays the conversion from sampling through that decision. Replaying
    preserves ngspice's device and capacitor state exactly while allowing Python
    to update future CDAC controls from the comparator result.
    """
    cfg = _sar_config(spec, config)
    if not 0.0 <= vin <= cfg["vref"]:
        raise ValueError("vin must lie between 0 and adc.vref")
    tgrid = sar_time_grid(spec, cfg)
    binding = spec.binding().at_corner(corner)
    decisions: list[int | None] = [None] * cfg["n_bits"]
    trace = []
    for bit in range(cfg["n_bits"]):
        waveforms = sar_input_waveforms(
            spec, vin, decisions, bit, config=cfg, tgrid=tgrid)
        result = transient(
            spec.sizes, spec.bias, tgrid, binding=binding, inputs=waveforms,
            integration_method="gear2", max_step=cfg["edge_time"],
        )
        node = cfg["comparator_node"]
        if node not in result["nodes"]:
            raise ValueError(f"comparator node {node!r} is absent from transient result")
        decision_time = cfg["sample_end"] + (bit + 1.0) * cfg["bit_period"]
        comparator_v = float(np.interp(decision_time, tgrid, result["nodes"][node]))
        high = comparator_v >= cfg["comparator_threshold"]
        decisions[bit] = int(not high) if cfg["high_means_clear"] else int(high)
        trace.append({
            "bit": bit,
            "weight": 1 << (cfg["n_bits"] - 1 - bit),
            "decision_time": decision_time,
            "comparator_v": comparator_v,
            "kept": bool(decisions[bit]),
        })
    waveforms = sar_input_waveforms(
        spec, vin, decisions, cfg["n_bits"] - 1, config=cfg, tgrid=tgrid)
    result = transient(
        spec.sizes, spec.bias, tgrid, binding=binding, inputs=waveforms,
        integration_method="gear2", max_step=cfg["edge_time"],
    )
    bits = np.asarray(decisions, np.int8)
    weights = 1 << np.arange(cfg["n_bits"] - 1, -1, -1, dtype=np.int64)
    rail_values = spec.topology.rail_values(spec.bias)
    supply_rails = cfg.get("power_rails", ("VDD",))
    supply_power = average_supply_power(
        tgrid, result["branch_currents"],
        {rail: rail_values[rail] for rail in supply_rails},
    )
    driver_waves = {}
    for name, _p, _q, value in spec.topology.vsources:
        if isinstance(value, str) and value in waveforms:
            driver_waves[name] = waveforms[value]
    for name, key in spec.topology.transient_inputs.items():
        if key in waveforms:
            driver_waves[f"gate:{name}"] = waveforms[key]
    driver_power = average_waveform_source_power(
        tgrid, result["branch_currents"], driver_waves)
    return {
        "vin": float(vin),
        "code": int(bits @ weights),
        "bits": bits,
        "decisions": trace,
        "t": tgrid,
        "input_waveforms": waveforms,
        "transient": result,
        "n_bits": cfg["n_bits"],
        "vref": cfg["vref"],
        "supply_power": supply_power,
        "driver_power": driver_power,
        "total_power_w": supply_power["total_w"] + driver_power["total_w"],
    }


def run_sar_sweep(spec: CircuitSpec, vin_values, *, config=None,
                  corner: str | None = None) -> dict:
    """Convert a monotonic input sweep and calculate SAR static linearity."""
    cfg = _sar_config(spec, config)
    vin = np.asarray(vin_values, float)
    if vin.ndim != 1 or len(vin) < 2 or np.any(np.diff(vin) <= 0.0):
        raise ValueError("vin_values must be a strictly increasing one-dimensional array")
    conversions = [run_sar_conversion(spec, value, config=cfg, corner=corner)
                   for value in vin]
    codes = np.array([item["code"] for item in conversions], np.int64)
    metrics = static_ramp_metrics(
        vin, codes, cfg["n_bits"], vmin=0.0, vmax=cfg["vref"])
    return {
        "vin": vin,
        "codes": codes,
        "metrics": metrics,
        "conversions": conversions,
        "n_bits": cfg["n_bits"],
        "vref": cfg["vref"],
    }


def run_sar_signal(spec: CircuitSpec, vin_values, sample_rate: float, *, config=None,
                   corner: str | None = None, fundamental_bin: int | None = None) -> dict:
    """Convert an arbitrary sampled signal and calculate dynamic ADC metrics."""
    cfg = _sar_config(spec, config)
    vin = np.asarray(vin_values, float)
    if vin.ndim != 1 or len(vin) < 8:
        raise ValueError("vin_values must contain at least eight samples")
    conversions = [run_sar_conversion(spec, value, config=cfg, corner=corner)
                   for value in vin]
    codes = np.array([item["code"] for item in conversions], np.int64)
    return {
        "vin": vin,
        "codes": codes,
        "metrics": dynamic_metrics(
            codes, sample_rate, fundamental_bin=fundamental_bin),
        "average_power_w": float(np.mean([item["total_power_w"] for item in conversions])),
        "conversions": conversions,
        "n_bits": cfg["n_bits"],
        "vref": cfg["vref"],
    }
