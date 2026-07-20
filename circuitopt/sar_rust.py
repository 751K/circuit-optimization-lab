"""Marshalling for the compiled SAR conversion batch (rewrite step R8).

Builds a :class:`circuitopt_core.CompiledSarConversion` template once from a SAR
spec and drives a whole mismatch Monte-Carlo trial sweep through it under one
``py.detach`` with a single Rayon pool. The frozen Python loops in
:mod:`circuitopt.sar`/:mod:`circuitopt.sar_mc` remain the reference and the
fallback: :func:`build_sar_batch` raises :class:`SarRustUnavailable` for any spec
the compiled path does not reproduce bit-for-bit (non-native devices, an
incomplete DC seed, an unmapped waveform), and the caller drops back to the
reference loop.

The compiled path reproduces the reference codes exactly: the device cards are
the frozen :func:`circuitopt.device_factory.build_devices` model/instance
parameters (only ``delvto`` varies per trial), the circuit is the same compiled
topology (only capacitor values vary per trial), the initial state is the fixed
``dc_guesses`` seed, and the per-bit waveform/grid/comparator arithmetic mirrors
``run_sar_conversion`` (see :mod:`co_core::sar`).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ._rust_transient import _optional_index, _term_record, passive_problem_spec
from .compiled_topology import TERM_RAIL, CompiledTopology
from .device_factory import (apply_silicon_corner, build_devices,
                             resolve_binding)
from .sar import _sar_config, sar_input_waveforms, sar_time_grid
from .transient_solver import native_bsim4_model_names

# Role kind codes shared with ``co_core::sar::Role`` (order = enum discriminants).
_ROLE_SAMPLE = 0
_ROLE_SAMPLE_BAR = 1
_ROLE_BIT = 2
_ROLE_BIT_BAR = 3
_ROLE_DUMMY = 4
_ROLE_DUMMY_BAR = 5
_ROLE_CLOCK = 6


class SarRustUnavailable(RuntimeError):
    """The compiled SAR batch cannot reproduce this spec; use the frozen path."""


def _core():
    try:
        import circuitopt_core
    except ImportError as exc:  # pragma: no cover - extension always present in-tree
        raise SarRustUnavailable(f"circuitopt_core import failed: {exc}") from exc
    if not hasattr(circuitopt_core, "CompiledSarConversion"):
        raise SarRustUnavailable("circuitopt_core lacks CompiledSarConversion")
    return circuitopt_core


def _roles(cfg: Mapping, input_keys: Sequence[str]) -> list[tuple[int, int]]:
    """Classify each canonical input key into a ``(kind, bit)`` role record."""
    differential = cfg["bit_inputs_bar"] is not None
    bit_index = {key: i for i, key in enumerate(cfg["bit_inputs"])}
    bar_index = ({key: i for i, key in enumerate(cfg["bit_inputs_bar"])}
                 if differential else {})
    clock_key = cfg["clock"]["input"] if cfg["clock"] is not None else None
    roles: list[tuple[int, int]] = []
    for key in input_keys:
        if key == cfg["sample_input"]:
            roles.append((_ROLE_SAMPLE, 0))
        elif key == cfg["sample_bar_input"]:
            roles.append((_ROLE_SAMPLE_BAR, 0))
        elif key in bit_index:
            roles.append((_ROLE_BIT, bit_index[key]))
        elif key in bar_index:
            roles.append((_ROLE_BIT_BAR, bar_index[key]))
        elif key == cfg["dummy_input"]:
            roles.append((_ROLE_DUMMY, 0))
        elif key == cfg["dummy_input_bar"]:
            roles.append((_ROLE_DUMMY_BAR, 0))
        elif clock_key is not None and key == clock_key:
            roles.append((_ROLE_CLOCK, 0))
        else:  # pragma: no cover - guarded by build_sar_batch device checks
            raise SarRustUnavailable(f"waveform key {key!r} has no SAR role")
    return roles


class CompiledSarBatch:
    """A compiled SAR conversion template + trial driver.

    ``device_names`` orders the per-trial ``delvto`` vector to the template's
    device slots; ``cap_count`` is the CDAC capacitor count each trial supplies.
    """

    def __init__(self, rust_obj, device_names: Sequence[str], cap_count: int):
        self._rust = rust_obj
        self._device_names = tuple(device_names)
        self._cap_count = int(cap_count)

    @property
    def levels(self) -> int:
        return int(self._rust.levels())

    def run(self, trials: Sequence[tuple[Mapping[str, float], Sequence[float]]],
            workers: int) -> list[np.ndarray]:
        """Convert every trial's code-center sweep, returning per-trial codes.

        ``trials`` is a sequence of ``(delvto_map, cap_values)``; the result is a
        list of ``int64`` code arrays in trial order (byte-identical for any
        worker count).
        """
        records = []
        for delvto_map, cap_values in trials:
            cap_values = list(cap_values)
            if len(cap_values) != self._cap_count:
                raise SarRustUnavailable(
                    "trial cap count does not match template")
            records.append({
                "delvto": [float(delvto_map.get(name, 0.0))
                           for name in self._device_names],
                "cap_values": [float(v) for v in cap_values],
            })
        codes = self._rust.evaluate_batch(records, int(workers))
        return [np.asarray(row, dtype=np.int64) for row in codes]


def build_sar_batch(spec, cfg: Mapping | None = None, *,
                    corner: str | None = None) -> CompiledSarBatch:
    """Compile a SAR conversion template for ``spec``.

    Raises :class:`SarRustUnavailable` when the compiled path would not reproduce
    the frozen :func:`run_sar_conversion` bit-for-bit, so the caller can fall
    back to the reference loop.
    """
    core = _core()
    cfg = _sar_config(spec, cfg)
    tgrid = sar_time_grid(spec, cfg)

    binding = spec.binding().at_corner(corner)
    topo, nf, resolved_corner, model_types, device_kwargs, _ = resolve_binding(
        binding)

    # The compiled loop is the native BSIM4 transient; anything else stays on the
    # frozen path (ngspice / OTFT device backends, mixed topologies).
    native_names = set(native_bsim4_model_names(model_types))
    device_names = [item[0] for item in topo.devices]
    if not device_names or set(device_names) != native_names:
        raise SarRustUnavailable(
            "compiled SAR batch requires every device on the native BSIM4 backend")

    # Fixed DC seed (delvto-independent, matches run_sar_conversion's `initial`).
    if not isinstance(binding.dc_seed, Mapping):
        raise SarRustUnavailable("compiled SAR batch requires a dc_guesses seed")
    try:
        initial = np.asarray(
            [binding.dc_seed[name] for name in binding.topo.solved], dtype=float)
    except KeyError as exc:
        raise SarRustUnavailable(
            f"dc_guesses seed missing solved node {exc}") from exc

    # Canonical input order = sar_input_waveforms dict insertion order.
    probe = sar_input_waveforms(
        spec, 0.0, [None] * cfg["n_bits"], 0, config=cfg, tgrid=tgrid)
    input_keys = tuple(probe)
    roles = _roles(cfg, input_keys)

    plan = CompiledTopology(topo, spec.bias, input_keys=input_keys,
                            node_inputs={}, transient_inputs=True)
    if cfg["comparator_node"] not in plan.idx:
        raise SarRustUnavailable("comparator node is not a solved node")

    # Nominal device cards (native BSIM path device build), matching transient().
    native_kwargs = {name: dict(values)
                     for name, values in (device_kwargs or {}).items()}
    native_kwargs, native_corner = apply_silicon_corner(
        model_types, native_kwargs, resolved_corner)
    devices = build_devices(spec.sizes, nf=nf, corner=native_corner, topo=topo,
                            model_types=model_types, device_kwargs=native_kwargs)

    bsim_devices = []
    device_cards = []
    plan_device_names = []
    for item in plan.devices:
        wrapper = devices[item.name]
        model_card = getattr(wrapper, "model_card", None)
        instance_card = getattr(wrapper, "instance_card", None)
        if model_card is None or instance_card is None:
            raise SarRustUnavailable(
                f"device {item.name!r} has no native BSIM4 card")
        terms = [_term_record(item.d), _term_record(item.g), _term_record(item.s),
                 _term_record((TERM_RAIL, wrapper.vb))]
        rows = [_optional_index(item.di), _optional_index(item.gi),
                _optional_index(item.si), -1]
        bsim_devices.append((terms, rows))
        device_cards.append((
            int(model_card.polarity),
            float(wrapper.temperature),
            [(str(name), float(value))
             for name, value in model_card.parameters.items()],
            [(str(name), float(value))
             for name, value in instance_card.parameters.items()],
        ))
        plan_device_names.append(item.name)

    v0 = np.asarray(initial, dtype=float)
    n_aug = plan.n_aug
    if v0.shape[0] < n_aug:
        v0 = np.concatenate([v0, np.zeros(n_aug - v0.shape[0])])
    elif v0.shape[0] > n_aug:
        v0 = v0[:n_aug]

    levels = 1 << cfg["n_bits"]
    vins = (np.arange(levels) + 0.5) / levels * cfg["vref"]

    clock = None
    if cfg["clock"] is not None:
        ck = cfg["clock"]
        clock = [float(ck["high"]), float(ck["low"]),
                 float(ck["eval_before"]), float(ck["reset_hold"])]

    # Native-path Newton controls (transient() -> transient_native_bsim4):
    # gear2, maxit=max(30,40), vtol=1e-8, step_limit=min(5.0,0.25), gmin=1e-12.
    newton = [1.0, 40.0, 1e-8, 0.25, 1e-12]

    circuit = core.OtftTransientProblem(passive_problem_spec(plan))
    template = core.CompiledSarConversion({
        "circuit": circuit,
        "bsim_devices": bsim_devices,
        "device_cards": device_cards,
        "roles": roles,
        "n_bits": int(cfg["n_bits"]),
        "vref": float(cfg["vref"]),
        "sample_end": float(cfg["sample_end"]),
        "bit_period": float(cfg["bit_period"]),
        "edge_time": float(cfg["edge_time"]),
        "input_common_mode": float(cfg["input_common_mode"]),
        "comparator_index": int(plan.idx[cfg["comparator_node"]]),
        "comparator_threshold": float(cfg["comparator_threshold"]),
        "high_means_clear": bool(cfg["high_means_clear"]),
        "differential": bool(cfg["bit_inputs_bar"] is not None),
        "clock": clock,
        "newton": newton,
        "v0": [float(x) for x in v0],
        "tgrid": [float(x) for x in tgrid],
        "vins": [float(x) for x in vins],
    })
    return CompiledSarBatch(template, plan_device_names, len(plan.capacitors))
