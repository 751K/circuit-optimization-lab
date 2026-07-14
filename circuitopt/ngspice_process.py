"""Process adapters for model-card-backed ngspice simulation.

The circuit solvers describe transistors uniformly as ``(name, d, g, s)`` plus
geometry and a model-registry key.  Foundry decks do not: one process may expose
plain ``M`` devices, another may require a four-terminal ``X`` wrapper, custom
library sections, hierarchical operating-point vectors, or simulator
compatibility flags.  :class:`NgspiceProcessAdapter` is the narrow boundary
between those two worlds.

Adapters contain no model parameters and no proprietary files.  They only resolve
an externally installed model root and describe how to reference it from a deck.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from .device_model import get_model_class


class NgspiceProcessAdapter(ABC):
    """Netlist and characterisation conventions for one ngspice-backed process."""

    name: str
    model_prefix: str
    corners: tuple[str, ...]
    default_corner: str
    vdd: float
    cache_namespace: str
    command_args: tuple[str, ...] = ()

    def normalize_corner(self, corner: Any) -> str:
        if corner is None or corner == "":
            return self.default_corner
        key = corner.lower() if isinstance(corner, str) else corner
        if key not in self.corners:
            raise ValueError(
                f"unknown {self.name} corner {corner!r}; expected one of "
                f"{sorted(self.corners)}")
        return str(key)

    def common_corner(self, device_kwargs, device_names) -> str:
        kwargs = {k: dict(v) for k, v in (device_kwargs or {}).items()}
        corners = {
            self.normalize_corner(kwargs.get(name, {}).get("corner"))
            for name in device_names
        }
        if not corners:
            return self.default_corner
        if len(corners) != 1:
            raise ValueError(
                f"one common {self.name} corner from {sorted(self.corners)} is required, "
                f"got {sorted(corners)}")
        return next(iter(corners))

    def validate_model_types(self, model_types, device_names) -> None:
        model_types = dict(model_types or {})
        missing = sorted(set(device_names) - set(model_types))
        if missing:
            raise NotImplementedError(
                f"{self.name} full-circuit ngspice analysis requires explicit model "
                f"bindings; missing: {', '.join(missing)}")
        bad = {
            name: model_types[name]
            for name in device_names
            if not str(model_types[name]).startswith(self.model_prefix + ".")
        }
        if bad:
            raise NotImplementedError(
                f"mixed {self.name}/other-process ngspice analysis is not supported: "
                + ", ".join(f"{k}={v}" for k, v in sorted(bad.items())))

    @abstractmethod
    def deck_preamble(self, model_types, device_kwargs, device_names) -> tuple[str, list[str]]:
        """Return ``(corner, model-card lines)`` for a complete circuit deck."""

    @abstractmethod
    def render_instance(self, *, name: str, d: str, g: str, s: str, b: str,
                        model_type: str, width_um: float, length_um: float,
                        nf: int, mismatch: float = 0.0, mult: int = 1) -> str:
        """Render one four-terminal transistor instance.

        ``mult`` is the parallel-instance multiplicity (default 1 -> omitted, so an
        existing deck stays byte-identical); ``mult > 1`` appends ``m=<int>`` to fold
        that many identical parallel instances into one line."""

    @abstractmethod
    def characterization_preamble(self, corner: str, polarity: str,
                                  card_path: str) -> list[str]:
        """Model-card lines for a one-device DC/noise characterisation deck."""

    @abstractmethod
    def characterization_instance(self, *, name: str, polarity: str,
                                  width_um: float, length_um: float, nf: int = 1) -> str:
        """One-device instance line using nodes ``d g s b``."""

    def op_vector(self, instance_name: str, variable: str) -> str:
        """ngspice vector for an instance operating-point variable."""
        return f"@{instance_name}[{variable}]"

    def normalize_op_data(self, variable: str, values):
        """Map process-specific op-vector conventions to circuitopt's grid convention."""
        return values


def adapter_for_model_types(model_types: Mapping[str, str] | None,
                            device_names=None) -> NgspiceProcessAdapter | None:
    """Resolve one shared adapter from registered model classes.

    ``None`` means none of the selected classes advertises a process adapter.  A
    mixture of adapter-backed and other models is rejected because one circuit
    deck cannot safely combine unrelated foundry setup semantics.
    """
    if not model_types:
        return None
    names = set(device_names) if device_names is not None else set(model_types)
    adapters = []
    without = []
    for name in names:
        model_type = model_types.get(name)
        if model_type is None:
            without.append(name)
            continue
        cls = get_model_class(str(model_type))
        adapter = getattr(cls, "NGSPICE_ADAPTER", None) if cls is not None else None
        if adapter is None:
            without.append(name)
        else:
            adapters.append(adapter)
    if not adapters:
        return None
    unique = {id(adapter): adapter for adapter in adapters}
    if len(unique) != 1 or without:
        labels = sorted({adapter.name for adapter in adapters})
        raise NotImplementedError(
            "one ngspice process adapter is required for the full circuit; "
            f"adapters={labels}, unadapted={sorted(without)}")
    return next(iter(unique.values()))
