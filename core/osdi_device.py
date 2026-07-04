"""``TransistorModel`` adapter over an OpenVAF-compiled OSDI model (e.g. BSIM4).

Wraps :class:`core.osdi_host.Device` so a compiled Verilog-A compact model plugs
into the solver stack through the standard :class:`~core.device_model.TransistorModel`
interface. This is the bridge that lets a *silicon* PDK (SKY130) run inside the
same AC / noise engine used for the OTFT — see the ``silicon-pdk-openvaf`` memory.

**Scope (Phase A):** DC + small-signal (gm/gds) + capacitances + noise, i.e. the
methods AC / noise / DC solvers call on the model object. The transient-only hooks
(charge companion, numba params) raise :class:`NotImplementedError` — silicon
transient/chopper is deferred Phase B (the `.osdi` can't live inside the numba loop).

The OpenVAF toolchain lives on the external drive; compilation is lazy (only when a
device is first built) so importing this module never needs the toolchain.
"""
from __future__ import annotations

import os
import subprocess
import threading
from typing import Dict, Optional, Tuple

from .device_model import TransistorModel

_VAF_ROOT = os.environ.get("OPENVAF_ROOT", "/Volumes/MacoutDsik/Code/VAF/OpenVAF-Reloaded")
_VACOMPILE = os.path.join(_VAF_ROOT, ".claude/skills/build-openvaf/scripts/vacompile.sh")
_OSDI_CACHE_DIR = os.environ.get(
    "OSDI_CACHE_DIR", os.path.join(_VAF_ROOT, ".osdi_cache"))

_lib_cache: Dict[str, object] = {}       # va_path -> OsdiLibrary
_lib_lock = threading.Lock()


def compile_va(va_path: str, *, cache_dir: Optional[str] = None) -> str:
    """Compile a ``.va`` to ``.osdi`` via the OpenVAF wrapper (cached by mtime)."""
    if not os.path.exists(_VACOMPILE):
        raise RuntimeError(
            f"OpenVAF compiler wrapper not found at {_VACOMPILE}; set OPENVAF_ROOT")
    cache_dir = cache_dir or _OSDI_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(va_path))[0]
    osdi = os.path.join(cache_dir, stem + ".osdi")
    if not (os.path.exists(osdi) and os.path.getmtime(osdi) >= os.path.getmtime(va_path)):
        subprocess.run([_VACOMPILE, va_path, "-o", osdi], check=True, capture_output=True)
    return osdi


def load_model(va_path: str):
    """Compile (if needed) + load a ``.va`` model, returning a cached OsdiLibrary."""
    from .osdi_host import load_osdi
    with _lib_lock:
        lib = _lib_cache.get(va_path)
        if lib is None:
            lib = load_osdi(compile_va(va_path))
            _lib_cache[va_path] = lib
        return lib


class OsdiDevice(TransistorModel):
    """A transistor backed by an OSDI (compiled Verilog-A) compact model.

    Subclasses bind the model: set :attr:`VA_PATH` (Verilog-A source), :attr:`MODULE`
    (Verilog-A module name, or ``None`` for the first descriptor), :attr:`BASE_CARD`
    (process ``.model`` params), and :attr:`TYPE` (BSIM4 polarity, +1 NMOS / -1 PMOS).

    Geometry ``W``/``L`` are in µm (matching the rest of the stack); the bulk bias
    ``vb`` defaults to 0 V (tie to the reference) and can be overridden per instance.
    """
    VA_PATH: str = ""
    MODULE: Optional[str] = None
    BASE_CARD: Dict[str, float] = {}
    TYPE: int = 1                # +1 NMOS, -1 PMOS (BSIM4 `type`)

    def __init__(self, W: float = 1.0, L: float = 0.15, NF: int = 1, *,
                 vb: float = 0.0, temperature: float = 300.15, **corner):
        from .osdi_host import Device
        card = dict(self.BASE_CARD)
        card["type"] = self.TYPE             # polarity is authoritative, not the card
        card["l"] = float(L) * 1e-6          # µm -> m
        card["w"] = float(W) * 1e-6
        card["nf"] = int(NF)
        card.update({k: v for k, v in corner.items() if not k.startswith("_")})
        self.W, self.L, self.NF = float(W), float(L), int(NF)
        self.vb = float(vb)
        self.g_area = float(W) * float(L)
        # NMOS (TYPE=+1): source-low, drain current leaves the drain → kcl_sign=-1.
        # PMOS (TYPE=-1): source-high, sources current into the drain → kcl_sign=+1.
        self.kcl_sign = -1.0 if self.TYPE > 0 else 1.0
        self._dev = Device(load_model(self.VA_PATH), card,
                           model_name=self.MODULE, temperature=temperature)
        self._op_cache: Dict[Tuple[float, float, float], Dict[str, float]] = {}

    # ── operating point (cached per bias) ────────────────────────────────
    def _op(self, Vs: float, Vd: float, Vg: float) -> Dict[str, float]:
        key = (Vs, Vd, Vg)
        op = self._op_cache.get(key)
        if op is None:
            op = self._dev.operating_point(Vd, Vg, Vs, self.vb)
            self._op_cache[key] = op
        return op

    # ── core DC ──────────────────────────────────────────────────────────
    def get_Idc(self, Vs: float, Vd: float, Vg: float) -> float:
        return self._op(Vs, Vd, Vg)["Id"]

    def get_op(self, Vs: float, Vd: float, Vg: float) -> Tuple:
        return ()   # internal nodes are solved inside Device; nothing to expose

    def id_and_drain_charge(self, Vs: float, Vd: float, Vg: float):
        """(drain current [A], drain reactive charge [C]) at a bias — for transient
        backward-Euler (dQ/dt). Solves internals, then reads the reactive residual."""
        op = self._dev.operating_point(Vd, Vg, Vs, self.vb)
        Q = self._dev.charges(self._dev._last_redv)
        return op["Id"], float(Q[0])          # reduced node 0 = drain

    # ── small-signal (AC / PSS / PAC / PNoise) ───────────────────────────
    def get_ss_params(self, Vs: float, Vd: float, Vg: float) -> Dict[str, float]:
        op = self._op(Vs, Vd, Vg)
        return {"gm": max(op["gm"], 0.0), "gds": max(op["gds"], 1e-12),
                "Cgs": op["Cgs"], "Cgd": op["Cgd"], "Ich": abs(op["Id"])}

    def get_capacitances(self, Vs: float, Vd: float, Vg: float) -> Tuple[float, float]:
        op = self._op(Vs, Vd, Vg)
        return op["Cgs"], op["Cgd"]

    # ── noise ────────────────────────────────────────────────────────────
    def get_noise_psd(self, Vs: float, Vd: float, Vg: float,
                      frequency: float) -> Tuple[float, float]:
        # re-solve at this bias so Device's noise re-eval uses the right op point
        self._dev.operating_point(Vd, Vg, Vs, self.vb)
        return self._dev.noise_by_type(frequency)   # (S_thermal, S_flicker@f)

    # ── transient-only hooks: deferred Phase B ───────────────────────────
    _PHASE_B = "silicon transient/chopper is Phase B (OSDI can't run in the numba loop)"

    def get_capacitance_charges_from_op(self, *a):
        raise NotImplementedError(self._PHASE_B)

    def get_capacitance_branch_terms_from_op(self, *a):
        raise NotImplementedError(self._PHASE_B)

    def get_numba_params(self):
        raise NotImplementedError(self._PHASE_B)
