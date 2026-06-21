"""Device model interface — abstract base, parameter bundle, and factory.

This module defines :class:`TransistorModel`, the abstract base class that
every transistor compact model must implement.  Solvers depend on this ABC
instead of concrete model classes, so adding a new transistor type only
requires a new subclass + one ``register_model`` call — no solver edits.

The :class:`NumbaParams` dataclass is the canonical bundle of scalar model
parameters consumed by :file:`numba_kernels.py`.  Models expose it via
:meth:`TransistorModel.get_numba_params` so the transient solver can extract
every parameter in one pass and never touch the model object inside the
timestepping loop.

Usage::

    from .device_model import create_device

    dev = create_device("pmos_tft", W=1000, L=20, NF=1)
    dev.get_Idc(Vs=40, Vd=0, Vg=20)
    dev.get_numba_params()
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, Type

# ──────────────────────────────────────────────────────────────────────
# 1.  Numba kernel parameter bundle
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NumbaParams:
    """Frozen bundle of scalar model parameters for :file:`numba_kernels.py`.

    The transient solver extracts one of these per device at construction
    time, then copies the fields into per‑device numpy arrays for the
    numba‑accelerated inner loop.  The dataclass is frozen so extraction is
    a single atomic snapshot of the model instance.

    The current 16‑field schema matches :class:`~pmos_tft_model.PMOS_TFT`;
    other transistor models provide the same bundle with their own physical
    parameters.
    """
    Vfb: float
    Vss: float
    Lc: float
    lambda_: float
    contact_scale: float
    channel_exponent: float
    current_scale: float
    inv_Rleak: float
    two_over_pi: float
    cap_cgs1: float
    cap_cgd1: float
    cap_half_wl_ci: float
    cap_cgs3_base: float
    cap_cgd3_base: float
    k1: float
    gate_leak_g: float          # = 1 / R_cap2


# ──────────────────────────────────────────────────────────────────────
# 2.  Abstract base class
# ──────────────────────────────────────────────────────────────────────

class TransistorModel(ABC):
    """Abstract interface for a transistor compact model.

    Every solver in the stack works against this interface.  Concrete
    models (e.g. :class:`~pmos_tft_model.PMOS_TFT`) inherit and implement
    the abstract methods; solvers never import concrete model classes
    directly.

    **Core DC** — every solver depends on these two methods:

    .. method:: get_Idc(Vs, Vd, Vg) -> float
        Drain‑source DC current at the given terminal biases [A].

    .. method:: get_op(Vs, Vd, Vg) -> Tuple[float, float]
        Solve the internal operating point.  Returns a model‑specific
        tuple of internal‑node voltages (for PMOS_TFT: ``(Vs1, Vd1)``).
        The result is reused by capacitance‑charge and noise methods to
        avoid redundant OP solves inside the timestepping loop.

    **Small‑signal** — used by AC / PSS / PAC / PNoise:

    .. method:: get_ss_params(Vs, Vd, Vg) -> Dict[str, float]
        Terminal gm, gds, Cgs, Cgd, Ich at the given bias.  Default
        implementation uses central finite‑differences of
        :meth:`get_Idc` and :meth:`get_capacitances`.  Concrete models
        may override with an optimised analytic or numba path.

    **Capacitance** — used by transient / AC / PAC:

    .. method:: get_capacitances(Vs, Vd, Vg) -> Tuple[float, float]
        Small‑signal parasitic capacitances ``(Cgss, Cgdd)`` [F].

    .. method:: get_capacitance_charges_from_op(Vs, Vd, Vg, Vs1, Vd1) -> Tuple
        Branch charges from a previously‑solved operating point.
        Used by the transient solver for charge‑based companion models.

    .. method:: get_capacitance_branch_terms_from_op(Vs, Vd, Vg, Vs1, Vd1) -> Tuple
        Self‑charge branch terms for step‑integrated C(V)*dV transient
        experiments.

    **Noise** — used by noise / PNoise:

    .. method:: get_noise_psd(Vs, Vd, Vg, frequency) -> Tuple[float, float]
        Drain‑current noise PSD ``(S_thermal, S_flicker)`` [A²/Hz].

    **Numba bridge** — used by transient:

    .. method:: get_numba_params() -> NumbaParams
        Return the scalar parameter bundle consumed by numba kernels.

    **Auxiliary** (optional, with default no‑op implementations):

    .. attribute:: g_area
        Geometric area [µm²] — for design‑space exploration.

    .. method:: estimate_channel_charge(Vs, Vd, Vg, mobile_only=True) -> float
        Estimate turn‑off channel charge [C] — for chopper charge‑injection
        modelling.
    """

    # ── Core DC ───────────────────────────────────────────────────────

    @abstractmethod
    def get_Idc(self, Vs: float, Vd: float, Vg: float) -> float:
        """Drain‑source DC current [A]."""
        ...

    @abstractmethod
    def get_op(self, Vs: float, Vd: float, Vg: float) -> Tuple[float, float]:
        """Solve internal operating point; return model‑specific voltages."""
        ...

    # ── Small‑signal (default finite‑difference; override for speed) ──

    def get_ss_params(self, Vs: float, Vd: float, Vg: float) -> Dict[str, float]:
        """Terminal gm, gds, Cgs, Cgd, Ich at the given bias.

        Default: central finite‑differences of :meth:`get_Idc` plus
        :meth:`get_capacitances`.  Concrete models with an analytic or
        numba‑accelerated path override this method.
        """
        h = 1e-3
        gm = (self.get_Idc(Vs, Vd, Vg + h) - self.get_Idc(Vs, Vd, Vg - h)) / (2 * h)
        gds = (self.get_Idc(Vs, Vd + h, Vg) - self.get_Idc(Vs, Vd - h, Vg)) / (2 * h)
        Cgss, Cgdd = self.get_capacitances(Vs, Vd, Vg)
        return {"gm": max(gm, 0.0), "gds": max(gds, 1e-12),
                "Cgs": Cgss, "Cgd": Cgdd, "Ich": 0.0}

    # ── Capacitance ───────────────────────────────────────────────────

    @abstractmethod
    def get_capacitances(self, Vs: float, Vd: float, Vg: float) -> Tuple[float, float]:
        """Small‑signal parasitic capacitances ``(Cgss, Cgdd)`` [F]."""
        ...

    @abstractmethod
    def get_capacitance_charges_from_op(self, Vs: float, Vd: float, Vg: float,
                                        Vs1: float, Vd1: float):
        """Branch charges from a pre‑solved operating point.

        Returns ``(qgs, qgd, Cgss, Cgdd)``.
        """
        ...

    @abstractmethod
    def get_capacitance_branch_terms_from_op(self, Vs: float, Vd: float, Vg: float,
                                             Vs1: float, Vd1: float):
        """Self‑charge branch terms from a pre‑solved operating point.

        Returns ``(qgs_self, qgd_self, cgs_cross, cgd_cross, Cgss, Cgdd)``.
        """
        ...

    # ── Noise ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_noise_psd(self, Vs: float, Vd: float, Vg: float,
                      frequency: float) -> Tuple[float, float]:
        """Drain‑current noise PSD ``(S_thermal, S_flicker)`` [A²/Hz]."""
        ...

    # ── Numba bridge ──────────────────────────────────────────────────

    @abstractmethod
    def get_numba_params(self) -> NumbaParams:
        """Scalar parameter bundle for numba‑accelerated transient inner loop."""
        ...

    # ── Auxiliary (optional; subclasses may override or set as attributes) ─

    g_area: float = 0.0
    """Geometric area [µm²].  Models that precompute this can assign directly
    in ``_precompute_constants``."""

    def estimate_channel_charge(self, Vs: float, Vd: float, Vg: float,
                                mobile_only: bool = True) -> float:
        """Estimate turn‑off channel charge [C].

        The default returns 0; switch models that need charge‑injection
        modelling override this.
        """
        return 0.0


# ──────────────────────────────────────────────────────────────────────
# 3.  Factory / registry
# ──────────────────────────────────────────────────────────────────────

_model_registry: Dict[str, Type[TransistorModel]] = {}


def register_model(model_type: str, cls: Type[TransistorModel]) -> None:
    """Register a concrete model class under a short string name.

    Called once per model at module import time (e.g. in
    :file:`pmos_tft_model.py`)::

        register_model("pmos_tft", PMOS_TFT)
    """
    if not isinstance(model_type, str) or not model_type:
        raise ValueError(f"model_type must be a non‑empty string, got {model_type!r}")
    _model_registry[model_type] = cls


def create_device(model_type: str, **kwargs) -> TransistorModel:
    """Create a transistor model instance by name.

    Args:
        model_type: Short name registered via :func:`register_model`
            (e.g. ``"pmos_tft"``).
        **kwargs: Forwarded to the concrete model's constructor
            (geometry, process shifts, …).

    Returns:
        A :class:`TransistorModel` instance.

    Raises:
        ValueError: If *model_type* is not registered.
    """
    cls = _model_registry.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model type {model_type!r}; "
            f"known models: {sorted(_model_registry)}"
        )
    return cls(**kwargs)


def get_default_model_type() -> str:
    """Return the model type used when none is specified (``"pmos_tft"``)."""
    return "pmos_tft"
