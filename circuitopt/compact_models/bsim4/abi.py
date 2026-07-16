"""Simulator-neutral data contract for a BSIM4 numerical backend.

Terminal order is always ``(drain, gate, source, bulk)``. Currents are positive
leaving each terminal. ``conductance[i, j]`` is ``d I_i / d V_j``. Charges use
the same terminal sign convention and ``capacitance[i, j]`` is
``d Q_i / d V_j``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


class Bsim4ValidationError(ValueError):
    """A compact-model input/output violates the native backend contract."""


def _numeric_map(values: Mapping[str, float], label: str) -> dict[str, float]:
    result = {}
    for name, value in values.items():
        key = str(name).lower()
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise Bsim4ValidationError(
                f"{label} parameter {name!r} is not numeric") from exc
        if not np.isfinite(numeric):
            raise Bsim4ValidationError(
                f"{label} parameter {name!r} is non-finite: {numeric}")
        result[key] = numeric
    return result


@dataclass(frozen=True)
class Bsim4ModelCard:
    polarity: int
    parameters: Mapping[str, float]
    version: float = 4.5

    def __post_init__(self):
        if self.polarity not in {-1, 1}:
            raise Bsim4ValidationError("BSIM4 polarity must be +1 (NMOS) or -1 (PMOS)")
        version = float(self.version)
        if version not in {4.0, 4.5}:
            raise Bsim4ValidationError(
                "native backend accepts BSIM4 4.0-compatible and 4.5 cards, "
                f"got {self.version}")
        normalized = _numeric_map(self.parameters, "model")
        normalized.pop("level", None)
        normalized.pop("version", None)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "parameters", normalized)


@dataclass(frozen=True)
class Bsim4InstanceCard:
    parameters: Mapping[str, float]

    def __post_init__(self):
        normalized = _numeric_map(self.parameters, "instance")
        for required in ("w", "l"):
            if normalized.get(required, 0.0) <= 0:
                raise Bsim4ValidationError(
                    f"instance parameter {required!r} must be positive")
        for integer_name in ("nf", "m"):
            value = normalized.get(integer_name, 1.0)
            if value < 1 or value != int(value):
                raise Bsim4ValidationError(
                    f"instance parameter {integer_name!r} must be a positive integer")
        object.__setattr__(self, "parameters", normalized)


@dataclass(frozen=True)
class Bsim4Bias:
    drain: float
    gate: float
    source: float
    bulk: float
    temperature_k: float = 300.15

    def __post_init__(self):
        values = (self.drain, self.gate, self.source, self.bulk, self.temperature_k)
        if not all(np.isfinite(float(value)) for value in values):
            raise Bsim4ValidationError("bias values must be finite")
        if self.temperature_k <= 0:
            raise Bsim4ValidationError("temperature must be positive kelvin")

    @property
    def terminals(self) -> np.ndarray:
        return np.asarray(
            (self.drain, self.gate, self.source, self.bulk),
            dtype=float,
        )


@dataclass(frozen=True)
class Bsim4Noise:
    """Terminal-current cross-spectral density matrices [A^2/Hz]."""

    spectral_density: np.ndarray
    components: Mapping[str, np.ndarray] | None = None

    def __post_init__(self):
        matrix = np.asarray(self.spectral_density, dtype=np.complex128)
        if matrix.shape != (4, 4):
            raise Bsim4ValidationError(
                f"noise matrix must be 4x4, got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise Bsim4ValidationError("noise matrix contains non-finite values")
        if not np.allclose(matrix, matrix.conj().T, rtol=1e-8, atol=1e-30):
            raise Bsim4ValidationError("noise matrix must be Hermitian")
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.conj().T) * 0.5)
        if float(np.min(eigenvalues)) < -1e-8 * max(
            float(np.max(np.abs(eigenvalues))), 1e-30
        ):
            raise Bsim4ValidationError("noise matrix must be positive semidefinite")
        normalized_components = {}
        for name, component in (self.components or {}).items():
            value = np.asarray(component, dtype=np.complex128)
            if value.shape != (4, 4) or not np.all(np.isfinite(value)):
                raise Bsim4ValidationError(
                    f"noise component {name!r} must be a finite 4x4 matrix")
            if not np.allclose(value, value.conj().T, rtol=1e-8, atol=1e-30):
                raise Bsim4ValidationError(
                    f"noise component {name!r} must be Hermitian")
            normalized_components[str(name).lower()] = value
        object.__setattr__(self, "spectral_density", matrix)
        object.__setattr__(self, "components", normalized_components)


@dataclass(frozen=True)
class Bsim4Evaluation:
    terminal_currents: np.ndarray
    conductance: np.ndarray
    terminal_charges: np.ndarray
    capacitance: np.ndarray
    operating_point: Mapping[str, float]
    noise: Bsim4Noise | None = None

    def __post_init__(self):
        currents = np.asarray(self.terminal_currents, dtype=float)
        conductance = np.asarray(self.conductance, dtype=float)
        charges = np.asarray(self.terminal_charges, dtype=float)
        capacitance = np.asarray(self.capacitance, dtype=float)
        if currents.shape != (4,) or charges.shape != (4,):
            raise Bsim4ValidationError("current and charge vectors must have length 4")
        if conductance.shape != (4, 4) or capacitance.shape != (4, 4):
            raise Bsim4ValidationError("conductance and capacitance matrices must be 4x4")
        arrays = (currents, conductance, charges, capacitance)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise Bsim4ValidationError("BSIM4 evaluation contains non-finite values")
        scale_i = max(float(np.max(np.abs(currents))), 1e-18)
        scale_q = max(float(np.max(np.abs(charges))), 1e-24)
        if abs(float(np.sum(currents))) > 1e-8 * scale_i + 1e-18:
            raise Bsim4ValidationError("terminal currents do not satisfy KCL")
        if abs(float(np.sum(charges))) > 1e-8 * scale_q + 1e-24:
            raise Bsim4ValidationError("terminal charges are not conserved")
        if not np.allclose(
            np.sum(conductance, axis=0), 0.0, rtol=1e-7, atol=1e-15
        ):
            raise Bsim4ValidationError("conductance columns violate KCL")
        if not np.allclose(
            np.sum(capacitance, axis=0), 0.0, rtol=1e-7, atol=1e-21
        ):
            raise Bsim4ValidationError("capacitance columns violate charge conservation")
        object.__setattr__(self, "terminal_currents", currents)
        object.__setattr__(self, "conductance", conductance)
        object.__setattr__(self, "terminal_charges", charges)
        object.__setattr__(self, "capacitance", capacitance)
        object.__setattr__(
            self,
            "operating_point",
            _numeric_map(self.operating_point, "operating-point"),
        )


class Bsim4Backend(Protocol):
    """Numerical backend consumed by the future ``TransistorModel`` adapter."""

    name: str
    version: str

    def evaluate(
        self,
        model: Bsim4ModelCard,
        instance: Bsim4InstanceCard,
        bias: Bsim4Bias,
        *,
        frequency_hz: float | None = None,
    ) -> Bsim4Evaluation:
        ...
