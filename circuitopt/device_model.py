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

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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

    # ── Backend-capability flags ──────────────────────────────────────
    # Generic solvers dispatch on *capabilities*, never on a concrete backend
    # type.  A model advertises what it can do via these class attributes and
    # solvers read them (e.g. ``dev.HAS_TERMINAL_LINEARIZATION``), instead of
    # ``isinstance(dev, OsdiDevice)``.  Base defaults describe the plain OTFT
    # analytic model; only backends that add a capability override them.

    HAS_TERMINAL_LINEARIZATION: bool = False
    """True if the model exposes :meth:`get_terminal_linearization` — the full
    quasi-static 4×4 terminal (G, C) stamp used by the periodic PAC/PNoise
    linearizer.  Backends without it fall back to the terminal-gm/gds path."""

    TRANSIENT_BACKEND: str | None = None
    """Which specialised transient integrator this model routes to, or ``None``
    to use the generic (OTFT numba) transient path.  ``"osdi"`` routes the
    circuit to :func:`circuitopt.osdi_transient.transient_osdi`."""

    # ── Auxiliary (optional; subclasses may override or set as attributes) ─

    g_area: float = 0.0
    """Geometric area [µm²].  Models that precompute this can assign directly
    in ``_precompute_constants``."""

    kcl_sign: float = 1.0
    """Sign of the current the device sources INTO its drain node for the DC KCL.
    ``+1`` for a source-high (PMOS-like) device — the current flows source→drain, so
    it enters the drain; this is the OTFT convention.  ``-1`` for a source-low
    (NMOS-like) device, whose drain current flows drain→source (out of the drain).
    Solvers apply ``kcl_sign * abs(get_Idc(...))`` so both polarities share one KCL."""

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

    Registration deliberately *replaces* any existing entry — the registry has
    always allowed intentional substitution (e.g. a test swapping in a stub).
    But an **unintentional** name clash — two different PDK modules registering
    a different class under the same key/alias, where the last import silently
    wins — is a robustness hazard.  So a genuine collision (a *different* class,
    identified by ``__module__ + __qualname__``, taking over an occupied name)
    emits a :class:`RuntimeWarning` and still performs the override; callers who
    want the override stay unaffected, while accidental clashes become visible.

    Re-registering the *same* class — a repeat ``import`` or an
    :func:`importlib.reload` (which rebinds the class to a fresh object under
    the same qualified name) — is silent: same fully-qualified name means the
    ``is not`` identity check would over-report, so we compare on qualname too.
    """
    if not isinstance(model_type, str) or not model_type:
        raise ValueError(f"model_type must be a non‑empty string, got {model_type!r}")
    prev = _model_registry.get(model_type)
    if prev is not None and prev is not cls:
        prev_id = f"{prev.__module__}.{prev.__qualname__}"
        new_id = f"{cls.__module__}.{cls.__qualname__}"
        if prev_id != new_id:
            warnings.warn(
                f"model registry: {model_type!r} already registered to "
                f"{prev_id}; overwriting with {new_id}. "
                f"Two PDKs (or an alias) may be claiming the same name.",
                RuntimeWarning,
                stacklevel=2,
            )
    _model_registry[model_type] = cls


def get_model_class(model_type: str) -> Type[TransistorModel] | None:
    """Return the model class registered under *model_type*, or ``None``.

    Public read-only accessor over the registry so solvers can inspect a
    model's capability flags (class attributes) without importing a concrete
    backend class or reaching into the private ``_model_registry`` dict.
    """
    return _model_registry.get(model_type)


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


# ──────────────────────────────────────────────────────────────────────
# 4.  PDK / polarity layer  (over the flat model registry)
# ──────────────────────────────────────────────────────────────────────
#
# A *PDK* (process design kit) groups the transistor polarities that share a
# fabrication process — e.g. the AT4000TG process provides a ``pmos`` device
# (``pmos_TFT`` in ``PDK/veriloga.va``) and may later add an ``nmos``.  Each
# (pdk, polarity) pair is registered into the flat ``_model_registry`` above
# under a structured key ``"<pdk>.<polarity>"`` so :func:`create_device`
# resolves it for free; the PDK registry here just records the grouping and
# which PDK is the default.
#
# Generic elements — resistors, capacitors, ideal V/I sources, controlled
# sources — are process-independent and live in the topology / MNA layer, NOT
# here.  A PDK owns transistors only, so a new process reuses every source
# primitive unchanged.

@dataclass(frozen=True)
class PDK:
    """A named process: maps a device *polarity* to its compact-model class.

    Args:
        name: Process identifier (e.g. ``"at4000tg"``).
        devices: ``{polarity: TransistorModel subclass}`` — e.g.
            ``{"pmos": PMOS_TFT}``.  A future process adds ``"nmos"`` here.
        corners: Optional process-shift presets ``{name: {param: value}}``.
            The corner authority currently lives in :mod:`circuitopt.corners`, so
            this stays empty unless a PDK ships its own.
    """
    name: str
    devices: Dict[str, Type[TransistorModel]]
    corners: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def model_type(self, polarity: str) -> str:
        """Structured registry key for one polarity, e.g. ``"at4000tg.pmos"``."""
        if polarity not in self.devices:
            raise ValueError(
                f"PDK {self.name!r} has no {polarity!r} device; "
                f"available: {sorted(self.devices)}"
            )
        return f"{self.name}.{polarity}"


_pdk_registry: Dict[str, PDK] = {}
_default_pdk: str = ""        # name of the default PDK; "" until first register_pdk


def register_pdk(name: str, devices: Dict[str, Type[TransistorModel]], *,
                 corners: Dict[str, Dict[str, float]] | None = None,
                 default: bool = False,
                 aliases: Dict[str, str] | None = None) -> PDK:
    """Register a PDK and its polarities.

    Each ``polarity -> cls`` is also registered into the flat model registry
    under the structured key ``"<name>.<polarity>"`` (so
    :func:`create_device` resolves it), plus any back-compat *aliases*
    (``{alias: polarity}``, e.g. ``{"pmos_tft": "pmos"}``).  The first PDK
    registered — or any registered with ``default=True`` — becomes the default
    consulted by :func:`get_default_model_type` / :func:`transistor_type`.
    """
    global _default_pdk
    if not isinstance(name, str) or not name:
        raise ValueError(f"PDK name must be a non-empty string, got {name!r}")
    pdk = PDK(name, dict(devices), dict(corners or {}))
    _pdk_registry[name] = pdk
    for polarity, cls in pdk.devices.items():
        register_model(pdk.model_type(polarity), cls)
    for alias, polarity in (aliases or {}).items():
        register_model(alias, pdk.devices[polarity])
    if default or not _default_pdk:
        _default_pdk = name
    return pdk


def get_default_pdk() -> str:
    """Name of the default PDK (the one consulted for unannotated devices)."""
    if not _default_pdk:
        raise RuntimeError("no PDK registered")
    return _default_pdk


def list_pdks() -> list[str]:
    """Sorted names of all registered PDKs."""
    return sorted(_pdk_registry)


def get_pdk(name: str | None = None) -> PDK:
    """Return a PDK by name, or the default PDK when *name* is None."""
    key = name if name is not None else get_default_pdk()
    pdk = _pdk_registry.get(key)
    if pdk is None:
        raise ValueError(f"Unknown PDK {key!r}; registered: {list_pdks()}")
    return pdk


def transistor_type(polarity: str = "pmos", pdk: str | None = None) -> str:
    """Resolve a ``(pdk, polarity)`` pair to a model-registry key.

    Defaults to the default PDK's ``pmos`` — the single switch point every
    solver consults via :func:`get_default_model_type` instead of hardcoding a
    model name.  Pass ``pdk=`` / ``polarity=`` to target another process or an
    ``nmos`` once registered.
    """
    return get_pdk(pdk).model_type(polarity)


def create_transistor(polarity: str = "pmos", pdk: str | None = None,
                      **kwargs) -> TransistorModel:
    """Create a transistor for a ``(pdk, polarity)`` pair (default PDK's pmos)."""
    return create_device(transistor_type(polarity, pdk), **kwargs)


def get_default_model_type() -> str:
    """Model-registry key used when a device declares no model.

    Resolves to the default PDK's ``pmos`` (``"at4000tg.pmos"`` once that PDK
    is registered).  Solvers call this instead of naming a model literally, so
    adding or flipping the default PDK reroutes every unannotated device.
    Falls back to the legacy ``"pmos_tft"`` alias before any PDK loads.
    """
    if _default_pdk and "pmos" in _pdk_registry[_default_pdk].devices:
        return transistor_type("pmos")
    return "pmos_tft"
