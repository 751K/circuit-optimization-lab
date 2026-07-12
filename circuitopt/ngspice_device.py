"""``TransistorModel`` backed by ngspice-C via a cached characterisation grid.

For FreePDK45 the oracle is ngspice's built-in C-BSIM4 (its ``version = 4.0`` cards
diverge ~30 % from our BSIM4.8 OSDI VA — see :mod:`circuitopt.ngspice_char`). This adapter
makes ngspice the evaluator: a device characterises its ``(model, W, L, corner)`` once
into a (Vsb, Vds, Vgs) grid of Id / gm / gds / Cgs / Cgd (:func:`circuitopt.ngspice_char.characterize`,
exact ngspice-C at the nodes), then answers the ABC's Phase-A methods by interpolating
that grid — µs / eval, so the DC Newton and AC/noise sweeps run at solver speed.

**Grid scope:** DC + small-signal (gm/gds) + capacitances + noise. The grid-level
transient hooks raise :class:`NotImplementedError` because it carries no charge
companion; complete FreePDK45 circuits are instead routed by ``transient()`` to
the direct-ngspice full-charge backend in :mod:`circuitopt.ngspice_transient`.
gm / gds / Cgs / Cgd are read straight from
ngspice op-vars (Cgs = −dQg/dVs, Cgd = −dQg/dVd — the same definition the OSDI host
uses), i.e. true ngspice-C quantities, not differentiated interpolants.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .device_model import TransistorModel
from .ngspice_char import CharGrid, characterize, characterize_noise


def _clamp(v: float, bound) -> float:
    """Clip a query coordinate to its own axis range (reuse the edge value just past
    a rail rather than extrapolate)."""
    lo, hi = bound
    return min(max(v, lo), hi)


def _interpolator(grid: CharGrid):
    """Build one (Vsb, Vds, Vgs) -> value interpolant per op-var.

    Axes are stored device-natural (PMOS negative for vgs/vds); we interpolate in
    ascending-|V| space so ``RegularGridInterpolator`` gets monotone axes, and clip
    queries to the grid (a solver probing just past the rail reuses the edge value
    rather than extrapolating to NaN)."""
    from scipy.interpolate import RegularGridInterpolator
    ax = (np.abs(grid.vsb), np.abs(grid.vds), np.abs(grid.vgs))
    # Linear (not cubic): scipy's 3-D cubic silently returns 0 on the ~1e-17 cap
    # arrays. gm/gds/caps are characterised as direct op-vars (never differentiated),
    # so linear interpolation is exact at nodes and piecewise-linear between — smooth
    # enough for AC and fine for the DC Newton (which uses the separate gm op-var).
    interps = {name: RegularGridInterpolator(ax, arr, method="linear",
                                             bounds_error=False, fill_value=None)
               for name, arr in grid.data.items()}
    bounds = tuple((float(a[0]), float(a[-1])) for a in ax)
    return interps, bounds


class NgspiceDevice(TransistorModel):
    """A transistor evaluated by ngspice-C through a cached characterisation grid.

    Subclasses bind the process: :attr:`CARD_PATH` (FreePDK45 ``.inc``),
    :attr:`MODEL_NAME` (the card's ``.model`` name, e.g. ``NMOS_VTG``),
    :attr:`POLARITY` (``"nmos"``/``"pmos"``), :attr:`VDD` (sweep ceiling). Geometry
    ``W``/``L`` are in µm; ``vb`` is the bulk node voltage (0 for NMOS, VDD for PMOS)."""
    CARD_PATH: str = ""
    MODEL_NAME: str = ""
    POLARITY: str = "nmos"
    VDD: float = 1.0
    EXTRACT_W: float = None      # if set (µm), characterise the grid at this fixed W and
    #                              linearly scale the actual W — fast + smooth for W sweeps
    #                              (dataset/optimization); default characterises per W.

    def __init__(self, W: float = 0.09, L: float = 0.05, NF: int = 1, *,
                 vb: float = 0.0, corner: str = "nom", temperature: float = 300.15,
                 extract_w: float = None, **_ignored):
        self.W, self.L, self.NF = float(W), float(L), int(NF)
        self.vb = float(vb)
        self.corner = corner
        self.temp_c = float(temperature) - 273.15    # ngspice wants °C
        self.g_area = float(W) * float(L)
        self._sign = 1.0 if self.POLARITY == "nmos" else -1.0
        # NMOS: drain current leaves the drain (kcl_sign=-1); PMOS sources it (+1).
        self.kcl_sign = -1.0 if self.POLARITY == "nmos" else 1.0
        # extract_w (per instance) overrides EXTRACT_W: characterise one reference-W
        # grid and scale the extensive op-vars (Id/gm/gds/caps + noise PSD) by W/W_ref.
        # BSIM4 current is ~linear in W for the wide devices used here, so a W sweep
        # reuses one characterisation instead of re-running ngspice per W (the final
        # design is re-verified at its true per-W card). Absent → characterise at W.
        ref = extract_w if extract_w is not None else self.EXTRACT_W
        self._w_char = float(ref) if ref is not None else self.W
        self._w_scale = self.W / self._w_char
        grid = characterize(self.CARD_PATH, self.MODEL_NAME, self.POLARITY,
                            self._w_char, self.L, corner, vdd=self.VDD, temp_c=self.temp_c)
        self._interp, self._bounds = _interpolator(grid)
        self._op_cache: Dict[Tuple[float, float, float], Dict[str, float]] = {}
        self._noise = None          # (interps, bounds) built lazily on first noise call

    # ── grid lookup (reduced bias, cached) ───────────────────────────────
    def _lookup(self, Vs: float, Vd: float, Vg: float) -> Dict[str, float]:
        """Interpolate all op-vars at (Vgs, Vds, Vsb) reduced from terminal voltages.

        NF fingers scale current/caps linearly (BSIM4 is characterised at NF=1).
        Currents/caps returned carry ngspice's device-natural sign folded to the
        solver convention (Id>0 for a conducting device of either polarity)."""
        key = (Vs, Vd, Vg)
        op = self._op_cache.get(key)
        if op is not None:
            return op
        vgs = abs(Vg - Vs)                    # |Vgs|; grid axes are |V|
        vds = abs(Vd - Vs)
        vsb = abs(Vs - self.vb)
        pt = np.array([[_clamp(vsb, self._bounds[0]), _clamp(vds, self._bounds[1]),
                        _clamp(vgs, self._bounds[2])]])   # axis order (Vsb, Vds, Vgs)
        # NF fingers in parallel and the W/W_ref reference-W scale both multiply every
        # extensive op-var linearly (=1.0 when characterised at the instance W).
        nf = float(self.NF) * self._w_scale
        op = {name: float(self._interp[name](pt)[0]) * nf
              for name in ("id", "gm", "gds", "cgs", "cgd")}
        self._op_cache[key] = op
        return op

    # ── core DC ──────────────────────────────────────────────────────────
    def get_Idc(self, Vs: float, Vd: float, Vg: float) -> float:
        # grid Id is the device-natural magnitude for the swept |Vgs|,|Vds|; fold to
        # the solver's signed drain current (NMOS sinks, PMOS sources at the drain).
        return self._sign * self._lookup(Vs, Vd, Vg)["id"]

    def get_op(self, Vs: float, Vd: float, Vg: float) -> Tuple:
        return ()

    # ── small-signal (AC / noise) ────────────────────────────────────────
    def get_ss_params(self, Vs: float, Vd: float, Vg: float) -> Dict[str, float]:
        op = self._lookup(Vs, Vd, Vg)
        return {"gm": max(op["gm"], 0.0), "gds": max(op["gds"], 1e-12),
                "Cgs": max(-op["cgs"], 0.0), "Cgd": max(-op["cgd"], 0.0),
                "Ich": abs(op["id"])}

    def get_capacitances(self, Vs: float, Vd: float, Vg: float) -> Tuple[float, float]:
        op = self._lookup(Vs, Vd, Vg)
        return max(-op["cgs"], 0.0), max(-op["cgd"], 0.0)

    # ── noise (exact ngspice-C, lazily characterised) ────────────────────
    def _ensure_noise(self):
        """Build the (S_thermal, S_flicker@1Hz) interpolators on first noise call.

        Lazy because the ngspice ``.noise`` characterisation (one run per coarse-grid
        bias) is the slow part and not every use needs IRN. Cached to disk, so only
        the very first noise call on a new geometry pays it."""
        if self._noise is not None:
            return
        from scipy.interpolate import RegularGridInterpolator
        ng = characterize_noise(self.CARD_PATH, self.MODEL_NAME, self.POLARITY,
                                self._w_char, self.L, self.corner, vdd=self.VDD,
                                temp_c=self.temp_c)
        ax = (np.abs(ng.vsb), np.abs(ng.vds), np.abs(ng.vgs))
        # Interpolate in LOG space: S_thermal ∝ gm and S_flicker span decades across
        # bias, so linear interpolation of a convex positive quantity over-estimates
        # (flicker by ~40 % on a coarse grid). log10-interp + 10** is exact at nodes
        # and tracks the multiplicative variation between them.
        it = {k: RegularGridInterpolator(ax, np.log10(np.maximum(arr, 1e-30)),
                                         method="linear", bounds_error=False, fill_value=None)
              for k, arr in (("th", ng.s_thermal), ("fl", ng.s_flicker_1hz))}
        bounds = tuple((float(a[0]), float(a[-1])) for a in ax)
        self._noise = (it, bounds)

    def get_noise_psd(self, Vs: float, Vd: float, Vg: float,
                      frequency: float) -> Tuple[float, float]:
        """(S_thermal, S_flicker@1Hz) drain-current noise PSD [A²/Hz] — exact ngspice-C.

        Both terms come from a cached ngspice ``.noise`` characterisation: S_id(f) was
        fit as S_thermal + S_flicker@1Hz / f at each grid bias (BSIM4 tnoimod white
        channel noise + fnoimod 1/f — the 45 nm velocity-saturation excess is included,
        unlike an 8/3·kT·gm estimate). NF fingers add uncorrelated → PSD scales ×NF."""
        self._ensure_noise()
        it, bounds = self._noise
        vgs = abs(Vg - Vs)
        vds = abs(Vd - Vs)
        vsb = abs(Vs - self.vb)
        pt = np.array([[_clamp(vsb, bounds[0]), _clamp(vds, bounds[1]),
                        _clamp(vgs, bounds[2])]])          # axis order (Vsb, Vds, Vgs)
        nf = float(self.NF) * self._w_scale                # interps are log10(PSD)
        return 10.0 ** float(it["th"](pt)[0]) * nf, 10.0 ** float(it["fl"](pt)[0]) * nf

    # ── transient-only hooks: not supported on the grid path ─────────────
    _NO_TRAN = "FreePDK45/ngspice-grid path is DC+AC+noise (no transient charge companion)"

    def get_capacitance_charges_from_op(self, *a):
        raise NotImplementedError(self._NO_TRAN)

    def get_capacitance_branch_terms_from_op(self, *a):
        raise NotImplementedError(self._NO_TRAN)

    def get_numba_params(self):
        raise NotImplementedError(self._NO_TRAN)
