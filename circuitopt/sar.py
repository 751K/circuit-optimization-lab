"""Closed-loop SAR conversion driven by full-charge transient simulations."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Sequence

import numpy as np

from .adc import (average_supply_power, average_waveform_source_power,
                  dynamic_metrics, static_ramp_metrics)
from ._parallel import worker_device_eval
from .circuit_loader import CircuitSpec
from .transient_solver import transient


_SAR_NEWTON_VTOL = 1e-8


def _normalize_sar_mismatch(
    spec: CircuitSpec,
    mismatch: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate per-device offsets before either SAR implementation runs."""
    if not mismatch:
        return {}
    offsets = {str(name): float(value) for name, value in mismatch.items()}
    unknown = sorted(set(offsets) - {item[0] for item in spec.topology.devices})
    if unknown:
        raise ValueError(
            "mismatch references unknown devices: " + ", ".join(unknown))
    non_finite = sorted(
        name for name, value in offsets.items() if not np.isfinite(value))
    if non_finite:
        raise ValueError(
            "mismatch offsets must be finite for devices: "
            + ", ".join(non_finite))
    return offsets


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
    cfg["clock"] = _clock_config(cfg)
    return cfg


def _clock_config(cfg: Mapping) -> dict | None:
    """Resolve the optional ``adc.clock`` strobe block for a clocked comparator.

    Backward compatible: absent ``clock`` -> ``None`` and
    :func:`sar_input_waveforms` emits no clock key, so a static-comparator SAR
    (e.g. ``freepdk45_sar3``) renders a byte-identical netlist. When present, a
    single strobe waveform (key ``clock.input``) is generated that rests at
    ``low`` and pulses to ``high`` around every bit's ``decision_time`` so a
    dynamic latch (StrongARM) precharges during CDAC settling and evaluates at the
    decision instant. The strobe schedule is independent of the trial index and
    decisions, so the same waveform serves both the compiled continuation loop
    and the frozen replay fallback.
    """
    clock = cfg.get("clock")
    if clock is None:
        return None
    clock = dict(clock)
    period = cfg["bit_period"]
    edge = cfg["edge_time"]
    bar_input = clock.get("bar_input")
    ck = {
        "input": str(_required(clock, "input")),
        "bar_input": None if bar_input is None else str(bar_input),
        "high": float(clock.get("high", cfg["vref"])),
        "low": float(clock.get("low", 0.0)),
        "eval_before": float(clock.get("eval_before", 0.3 * period)),
        "reset_hold": float(clock.get("reset_hold", 0.1 * period)),
    }
    if ck["high"] <= ck["low"]:
        raise ValueError("adc.clock.high must exceed adc.clock.low")
    # rise = decision_time - eval_before must fall after the trial cap has switched
    # (trial_start + edge = decision_time - 0.5*period + edge) so the latch samples a
    # settled differential; keep eval_before < half a period (minus one edge).
    if not 0.0 < ck["eval_before"] < 0.5 * period - edge:
        raise ValueError("adc.clock.eval_before must be in (0, bit_period/2 - edge_time)")
    if ck["reset_hold"] < 0.0 or ck["eval_before"] + ck["reset_hold"] >= period:
        raise ValueError("adc.clock.eval_before + reset_hold must be below one bit_period")
    return ck


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
    if cfg["clock"] is not None:
        ck = cfg["clock"]
        events = [(0.0, ck["low"])]
        for bit in range(cfg["n_bits"]):
            decision_time = sample_end + (bit + 1.0) * period
            rise = decision_time - ck["eval_before"]
            fall = decision_time + ck["reset_hold"]
            events.extend([
                (rise - edge, ck["low"]), (rise, ck["high"]),
                (fall, ck["high"]), (fall + edge, ck["low"]),
            ])
        events.append((tstop, ck["low"]))
        strobe = _wave(tgrid, events)
        out[ck["input"]] = strobe
        if ck["bar_input"] is not None:
            # The complemented strobe for latches whose reset devices need the
            # opposite phase (double-tail second stages and PMOS-tail latches).
            out[ck["bar_input"]] = ck["high"] + ck["low"] - strobe
    return out


def _sar_initial_state(binding) -> np.ndarray | None:
    if not isinstance(binding.dc_seed, Mapping):
        return None
    try:
        return np.asarray(
            [binding.dc_seed[name] for name in binding.topo.solved],
            dtype=float,
        )
    except KeyError:
        return None


def _run_sar_decisions_replayed(spec, vin, cfg, corner, mismatch):
    """Frozen Python reference: replay one complete transient per decision."""
    tgrid = sar_time_grid(spec, cfg)
    binding = spec.binding().at_corner(corner)
    initial = _sar_initial_state(binding)
    decisions: list[int | None] = [None] * cfg["n_bits"]
    trace = []
    for bit in range(cfg["n_bits"]):
        waveforms = sar_input_waveforms(
            spec, vin, decisions, bit, config=cfg, tgrid=tgrid)
        result = transient(
            spec.sizes, spec.bias, tgrid, binding=binding, inputs=waveforms,
            V0=initial, integration_method="gear2",
            max_step=cfg["edge_time"], newton_vtol=_SAR_NEWTON_VTOL,
            mismatch=mismatch,
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
    return np.asarray(decisions, np.int8), trace


def _finalize_sar_conversion(
    spec,
    vin,
    cfg,
    corner,
    mismatch,
    bits,
    *,
    trace=None,
    decision_backend,
    native_trajectory=None,
):
    """Build the public conversion result from a decided physical trajectory."""
    tgrid = sar_time_grid(spec, cfg)
    decisions = [int(value) for value in np.asarray(bits, np.int8)]
    if native_trajectory is None:
        binding = spec.binding().at_corner(corner)
        initial = _sar_initial_state(binding)
        waveforms = sar_input_waveforms(
            spec, vin, decisions, cfg["n_bits"] - 1, config=cfg, tgrid=tgrid)
        result = transient(
            spec.sizes, spec.bias, tgrid, binding=binding, inputs=waveforms,
            V0=initial, integration_method="gear2",
            max_step=cfg["edge_time"], newton_vtol=_SAR_NEWTON_VTOL,
            mismatch=mismatch,
        )
    else:
        waveforms = native_trajectory["input_waveforms"]
        result = native_trajectory["transient"]
    bits = np.asarray(decisions, np.int8)
    weights = 1 << np.arange(cfg["n_bits"] - 1, -1, -1, dtype=np.int64)
    if trace is None:
        node = cfg["comparator_node"]
        if node not in result["nodes"]:
            raise ValueError(
                f"comparator node {node!r} is absent from transient result")
        trace = []
        for bit, kept in enumerate(bits):
            decision_time = cfg["sample_end"] + (bit + 1.0) * cfg["bit_period"]
            comparator_v = float(np.interp(
                decision_time,
                result["t"],
                result["nodes"][node],
            ))
            trace.append({
                "bit": bit,
                "weight": int(weights[bit]),
                "decision_time": decision_time,
                "comparator_v": comparator_v,
                "kept": bool(kept),
            })
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
        "decision_backend": decision_backend,
    }


def _run_sar_conversion_reference(spec: CircuitSpec, vin: float, *, config=None,
                                  corner: str | None = None,
                                  mismatch: Mapping[str, float] | None = None) -> dict:
    """Run the frozen full-replay SAR path for parity and unsupported circuits."""
    cfg = _sar_config(spec, config)
    mismatch = _normalize_sar_mismatch(spec, mismatch)
    if not 0.0 <= vin <= cfg["vref"]:
        raise ValueError("vin must lie between 0 and adc.vref")
    bits, trace = _run_sar_decisions_replayed(
        spec, float(vin), cfg, corner, mismatch)
    return _finalize_sar_conversion(
        spec,
        float(vin),
        cfg,
        corner,
        mismatch,
        bits,
        trace=trace,
        decision_backend="python_replay",
    )


def _compiled_sar_codes(spec, cfg, vin, corner, mismatch):
    from .sar_rust import build_sar_batch

    batch = build_sar_batch(spec, cfg, corner=corner, vins=vin)
    cap_values = [float(capacitor[3]) for capacitor in spec.topology.capacitors]
    return batch.run([(mismatch or {}, cap_values)], workers=1)[0]


def _run_sar_conversions(
    spec,
    vin,
    cfg,
    corner,
    mismatch,
    workers,
    *,
    include_transients=True,
):
    """Run an ordered input batch through complete Rust continuation trajectories."""
    from .sar_rust import SarRustUnavailable, build_sar_batch

    if workers is None or workers < 1:
        raise ValueError("workers must be a positive integer")
    mismatch = _normalize_sar_mismatch(spec, mismatch)
    values = np.asarray(vin, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > cfg["vref"])
    ):
        raise ValueError("SAR inputs must be a non-empty finite 1D array inside [0, vref]")
    try:
        batch = build_sar_batch(spec, cfg, corner=corner, vins=values)
        cap_values = [
            float(capacitor[3]) for capacitor in spec.topology.capacitors
        ]
        trial = (mismatch or {}, cap_values)
        if include_transients:
            trajectories = batch.run_trajectories(trial, workers=workers)
        else:
            codes = batch.run_codes(trial, workers=workers)
    except SarRustUnavailable:
        if not include_transients:
            weights = 1 << np.arange(
                cfg["n_bits"] - 1, -1, -1, dtype=np.int64)

            def run_reference_code(value):
                bits, trace = _run_sar_decisions_replayed(
                    spec, float(value), cfg, corner, mismatch)
                return {
                    "vin": float(value),
                    "code": int(bits @ weights),
                    "bits": bits,
                    "decisions": trace,
                    "n_bits": cfg["n_bits"],
                    "vref": cfg["vref"],
                    "decision_backend": "python_replay",
                }

            return _run_independent_conversions(
                run_reference_code, values, workers)
        return _run_independent_conversions(
            lambda value: _run_sar_conversion_reference(
                spec,
                float(value),
                config=cfg,
                corner=corner,
                mismatch=mismatch,
            ),
            values,
            workers,
        )

    if not include_transients:
        weights = 1 << np.arange(
            cfg["n_bits"] - 1, -1, -1, dtype=np.int64)
        return [
            {
                "vin": float(value),
                "code": int(code),
                "bits": ((int(code) & weights) != 0).astype(np.int8),
                "n_bits": cfg["n_bits"],
                "vref": cfg["vref"],
                "decision_backend": "rust_continuation",
            }
            for value, code in zip(values, codes, strict=True)
        ]

    def finalize(item):
        value, trajectory = item
        return _finalize_sar_conversion(
            spec,
            float(value),
            cfg,
            corner,
            mismatch,
            trajectory["bits"],
            decision_backend="rust_continuation",
            native_trajectory=trajectory,
        )

    return [
        finalize(item)
        for item in zip(values, trajectories, strict=True)
    ]


def run_sar_conversion(spec: CircuitSpec, vin: float, *, config=None,
                       corner: str | None = None,
                       mismatch: Mapping[str, float] | None = None) -> dict:
    """Run one physical SAR conversion using the compiled continuation loop.

    Rust advances one trajectory across all bit decisions and finishes the
    decided tail while recording terminal history. Python reconstructs the
    public transient, source-power, and comparator-trace payload from that same
    trajectory. Unsupported topologies use the frozen replay path explicitly.
    """
    cfg = _sar_config(spec, config)
    return _run_sar_conversions(
        spec,
        np.asarray([vin], dtype=float),
        cfg,
        corner,
        mismatch,
        workers=1,
    )[0]


def _run_independent_conversions(run, values, workers: int) -> list:
    """Evaluate an ordered batch of *independent* conversions, optionally threaded.

    Every final SAR transient here is independent of the others and owns its
    native handles. A :class:`ThreadPoolExecutor` therefore parallelises the
    result/power reconstruction without sharing mutable solver state.

    ``ex.map`` preserves input order, so any worker count returns an ordered
    result. Bit decisions within one conversion stay in the Rust continuation
    kernel; only independent final transients are parallelised here.
    """
    if workers is None or workers < 1:
        raise ValueError("workers must be a positive integer")
    values = list(values)
    if workers == 1:
        return [run(value) for value in values]

    def run_worker(value):
        # Conversions are the parallel level, so each one evaluates its device
        # batches inline instead of contending for the shared pool.
        with worker_device_eval(workers, len(values)):
            return run(value)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_worker, values))


def run_sar_sweep(spec: CircuitSpec, vin_values, *, config=None,
                  corner: str | None = None,
                  mismatch: Mapping[str, float] | None = None,
                  workers: int = 1,
                  include_transients: bool = True) -> dict:
    """Convert a monotonic input sweep and calculate SAR static linearity.

    ``mismatch`` (per-instance ``{device: delvto[V]}``) is forwarded to every
    conversion; ``None`` reproduces the nominal sweep.

    ``include_transients=False`` keeps only code/bits metadata per input and
    avoids recording terminal history. This is the fast path for DNL/INL-only
    sweeps; the default preserves the complete public waveform payload.
    """
    cfg = _sar_config(spec, config)
    vin = np.asarray(vin_values, float)
    if vin.ndim != 1 or len(vin) < 2 or np.any(np.diff(vin) <= 0.0):
        raise ValueError("vin_values must be a strictly increasing one-dimensional array")
    conversions = _run_sar_conversions(
        spec, vin, cfg, corner, mismatch, workers,
        include_transients=include_transients)
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
                   corner: str | None = None, fundamental_bin: int | None = None,
                   workers: int = 1) -> dict:
    """Convert an arbitrary sampled signal and calculate dynamic ADC metrics.

    ``workers`` parallelises the independent per-sample conversions across a thread
    pool; ``workers=1`` (default) is the serial path and any worker count is
    order-preserving and byte-identical to it.
    """
    cfg = _sar_config(spec, config)
    vin = np.asarray(vin_values, float)
    if vin.ndim != 1 or len(vin) < 8:
        raise ValueError("vin_values must contain at least eight samples")
    conversions = _run_sar_conversions(
        spec, vin, cfg, corner, None, workers)
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
