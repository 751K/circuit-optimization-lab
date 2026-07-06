"""ngspice-C as the device evaluator — batch DC characterization + cache.

FreePDK45 declares BSIM4 ``version = 4.0``; our OpenVAF ``bsim4va`` is BSIM4.8 and
carries no version switch, so it computes ~30 % different I-V from ngspice's built-in
C-BSIM4 on these aggressive 45 nm cards (confirmed version-independent: ngspice gives
the same Id for a card marked 4.0 … 4.8). The OSDI host therefore cannot reproduce
"real FreePDK45" behaviour. For FreePDK45 the *oracle is ngspice-C itself*, so we make
ngspice the evaluator.

Per-bias ngspice subprocess calls are far too slow for the DC Newton inner loop, and
no ``libngspice`` shared build is present. Instead we exploit that a batch ``.dc``
sweep evaluates thousands of bias points in one process (~0.03 s / 1000 points): one
sweep per ``(model, W, L, corner)`` characterises the whole (Vgs, Vds, Vsb) space into
a grid of Id / gm / gds / Cgs / Cgd, cached to ``data/pdk/freepdk45/*.npz``. Downstream
:class:`circuitopt.ngspice_device.NgspiceDevice` interpolates that grid (µs / eval), so every
value the solver sees is exact ngspice-C at the grid nodes. gm / gds / caps are read
straight from ngspice op-vars (not differentiated), i.e. true ngspice-C quantities.

The ngspice binary + FreePDK45 cards live on the external drive; characterisation is
lazy (only on first use of a geometry) so importing this module needs no toolchain.
Override locations with ``PDK_ROOT`` / ``NGSPICE_BIN``.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

_PDK_ROOT = os.environ.get("PDK_ROOT", "/Volumes/MacoutDsik/pdk")
_FP45_DIR = os.path.join(_PDK_ROOT, "freepdk45")
# The run-ngspice wrapper is vendored in-repo (self-contained); it resolves the
# ngspice binary via NGSPICE_BIN at call time. RUN_NGSPICE can override the
# wrapper path itself (escape hatch for wheel installs).
_RUN_NGSPICE = os.environ.get(
    "RUN_NGSPICE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools", "run-ngspice.sh")))
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/pdk/freepdk45")

# op-vars pulled from ngspice at every grid node (order fixed for wrdata parsing)
_OPVARS = ("id", "gm", "gds", "cgs", "cgd")


def _ngspice() -> str:
    if os.path.exists(_RUN_NGSPICE):
        return _RUN_NGSPICE
    return os.environ.get("NGSPICE_BIN", "ngspice")


def ngspice_binary() -> "str | None":
    """The ngspice executable the wrapper would run, or ``None`` if unreachable.

    Mirrors ``tools/run-ngspice.sh``: ``$NGSPICE_BIN`` if set (default is the
    external-drive build), else ``ngspice`` on ``PATH``. Returns ``None`` when
    no runnable binary exists, so tests can gate on the *real* ngspice
    dependency rather than on the presence of the (always-vendored) wrapper.
    """
    import shutil
    env_bin = os.environ.get("NGSPICE_BIN", "/Volumes/MacoutDsik/ngspice/install/bin/ngspice")
    if os.access(env_bin, os.X_OK):
        return env_bin
    return shutil.which("ngspice")


def _run_ngspice(cir: str, out_txt: str, timeout: float, what: str) -> None:
    """Run one ngspice batch deck, surfacing failures instead of swallowing them.

    ngspice's stderr carries harmless warnings even on a good run, so we key
    failure on the exit code and the output file's existence — never on stderr
    being non-empty. On a non-zero exit, or a silent failure (exit 0 but no
    ``out_txt``), raise :class:`RuntimeError` carrying ``what`` (deck purpose),
    the return code, and the tail of stderr (falling back to stdout) so the
    caller sees what ngspice actually said instead of a downstream
    ``FileNotFoundError`` from :func:`numpy.loadtxt`."""
    proc = subprocess.run([_ngspice(), "-b", cir], capture_output=True,
                          text=True, timeout=timeout)
    tail = (proc.stderr or "").strip() or (proc.stdout or "").strip()
    tail = tail[-800:]
    if proc.returncode != 0:
        raise RuntimeError(
            f"ngspice {what} failed (returncode {proc.returncode}); "
            f"deck={cir}\n--- ngspice output tail ---\n{tail}")
    if not os.path.exists(out_txt):
        raise RuntimeError(
            f"ngspice {what} produced no output {out_txt} (returncode 0, "
            f"silent failure); deck={cir}\n--- ngspice output tail ---\n{tail}")


@dataclass(frozen=True)
class CharGrid:
    """A characterised MOSFET: op-vars on a regular (Vsb, Vds, Vgs) grid.

    ``vgs``/``vds``/``vsb`` are the 1-D axis vectors (device-natural sign: NMOS
    positive, PMOS negative for vgs/vds). ``data`` maps each op-var name to a
    ``(n_vsb, n_vds, n_vgs)`` array. Terminal voltages the solver passes are
    reduced to (Vgs, Vds, Vsb) before lookup by :class:`NgspiceDevice`."""
    vgs: np.ndarray
    vds: np.ndarray
    vsb: np.ndarray
    data: Dict[str, np.ndarray]


# ── grid spec ─────────────────────────────────────────────────────────────────
def _default_grid(polarity: str, vdd: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Vgs, Vds, Vsb) axes for a polarity. FreePDK45 Vdd≈1.0 V; sweep to 1.1×Vdd
    with a small over-range so a solver probing just past the rail stays in-grid."""
    hi = 1.15 * vdd
    vgs = np.round(np.arange(0.0, hi + 1e-9, 0.025), 6)   # 25 mV: gm/gds resolution
    vds = np.round(np.arange(0.0, hi + 1e-9, 0.025), 6)
    vsb = np.round(np.arange(0.0, 0.8 + 1e-9, 0.1), 6)    # 100 mV: body effect
    if polarity == "pmos":                                # natural PMOS sign
        return -vgs, -vds, -vsb
    return vgs, vds, vsb


def _grid_key(model_name, W, L, corner, vgs, vds, vsb, temp_c=27.0) -> str:
    axes = np.concatenate([vgs, vds, vsb]).tobytes()
    h = hashlib.sha1(axes).hexdigest()[:8]
    tag = "" if abs(temp_c - 27.0) < 1e-9 else f"_T{temp_c:g}"
    return f"{model_name}_{corner}_W{W:g}_L{L:g}_{h}{tag}"


# ── ngspice batch characterisation ─────────────────────────────────────────────
def _sweep_one_vsb(card_path, model_name, W, L, vb, vgs, vds, temp_c=27.0):
    """Run one 2-D ``dc`` sweep (Vgs inner, Vds outer) at a fixed bulk bias → op-vars.

    ``vgs``/``vds`` are the *signed* device-natural axes (NMOS 0→+hi, PMOS 0→−hi);
    ``vb`` is the signed bulk voltage that sets Vsb (NMOS bulk ≤ 0, PMOS bulk ≥ 0,
    source at 0). ``temp_c`` sets the ngspice circuit temperature (°C) so BSIM4's
    temperature equations run at the PVT point. ngspice accepts a negative sweep step
    for the PMOS descent. wrdata flattens the sweep as Vds-blocks of Vgs-rows,
    reshaped to ``(n_vds, n_vgs)``."""
    saves = " ".join(f"@mn[{v}]" for v in _OPVARS)
    g0, g1, gd = vgs[0], vgs[-1], (vgs[1] - vgs[0])
    d0, d1, dd = vds[0], vds[-1], (vds[1] - vds[0])
    with tempfile.TemporaryDirectory() as td:
        out_txt = os.path.join(td, "out.txt")
        cir = os.path.join(td, "deck.cir")
        deck = (f"* freepdk45 char {model_name} W={W:g} L={L:g} vb={vb:g} T={temp_c:g}\n"
                f'.include "{card_path}"\n'
                f".options temp={temp_c:g}\n"
                f"mn d g s b {model_name} w={W:g}u l={L:g}u\n"
                f"vd d 0 {d0:g}\nvg g 0 {g0:g}\nvs s 0 0\nvb b 0 {vb:g}\n"
                f".control\nset filetype=ascii\nsave {saves}\n"
                f"dc vg {g0:g} {g1:g} {gd:g} vd {d0:g} {d1:g} {dd:g}\n"
                f"wrdata {out_txt} {saves}\n.endc\n.end\n")
        with open(cir, "w") as fh:
            fh.write(deck)
        _run_ngspice(cir, out_txt, timeout=120, what="dc-sweep")
        raw = np.loadtxt(out_txt)
    # wrdata writes (scale, value) column-pairs; values are the even columns.
    vals = raw[:, 1::2]                        # (n_rows, n_opvars)
    n_vgs, n_vds = len(vgs), len(vds)
    if vals.shape[0] != n_vgs * n_vds:
        raise RuntimeError(
            f"freepdk45 char {model_name}: dc sweep returned {vals.shape[0]} rows, "
            f"expected {n_vgs * n_vds} ({n_vds} Vds × {n_vgs} Vgs)")
    out = {}
    for j, name in enumerate(_OPVARS):
        out[name] = vals[:, j].reshape(n_vds, n_vgs)   # (Vds, Vgs)
    return out


def characterize(card_path, model_name, polarity, W, L, corner, vdd=1.0,
                 temp_c=27.0) -> CharGrid:
    """Characterise one FreePDK45 FET into a cached :class:`CharGrid`.

    One ngspice ``dc`` sweep per Vsb slice; the slices stack into
    ``(n_vsb, n_vds, n_vgs)`` op-var arrays. Cached to ``data/pdk/freepdk45/*.npz``
    (so reuse needs no ngspice) and returned. ``vdd`` sets the sweep ceiling;
    ``temp_c`` the device temperature (°C, keyed into the cache for PVT)."""
    vgs, vds, vsb = _default_grid(polarity, vdd)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache = os.path.join(_CACHE_DIR,
                         _grid_key(model_name, W, L, corner, vgs, vds, vsb, temp_c) + ".npz")
    if os.path.exists(cache):
        d = np.load(cache)
        data = {n: d[f"op_{n}"] for n in _OPVARS}
        return CharGrid(d["vgs"], d["vds"], d["vsb"], data)
    if not os.path.exists(card_path):
        raise RuntimeError(f"FreePDK45 card not found at {card_path}; set PDK_ROOT")
    # Sweep in the device-natural polarity: NMOS vg/vd 0→+hi with bulk ≤ 0; PMOS
    # vg/vd 0→−hi with bulk ≥ 0 (source at 0). Vsb = Vs − Vb, so bulk = −Vsb(nmos) /
    # +|Vsb|(pmos); the signed axis vectors carry the polarity into the sweep.
    stack = {n: [] for n in _OPVARS}
    for sb in np.abs(vsb):
        vb = sb if polarity == "pmos" else -sb
        slc = _sweep_one_vsb(card_path, model_name, W, L, vb, vgs, vds, temp_c)
        for n in _OPVARS:
            stack[n].append(slc[n])
    data = {n: np.stack(stack[n], axis=0) for n in _OPVARS}   # (Vsb, Vds, Vgs)
    np.savez(cache, vgs=vgs, vds=vds, vsb=vsb, **{f"op_{n}": data[n] for n in _OPVARS})
    return CharGrid(vgs, vds, vsb, data)


# ── noise characterisation (exact ngspice-C via .noise) ─────────────────────────
# Noise varies smoothly with bias, so a coarser grid than the DC sweep suffices;
# each point is one ngspice `.noise` (the loop-in-one-process idiom mis-fires on
# per-bias re-solve, so we spawn per point — still cached, first-use only).
def _noise_grid(polarity: str, vdd: float):
    hi = 1.1 * vdd
    # Fine in Vgs: S_flicker varies steeply (orders of magnitude) with Vgs, so a
    # coarse grid + linear interp systematically over-estimates flicker; NgspiceDevice
    # interpolates these in log-space, but a fine axis keeps the residual small too.
    vgs = np.round(np.linspace(0.15 * vdd, hi, 14), 6)    # skip deep subthreshold
    vds = np.round(np.linspace(0.2 * vdd, hi, 5), 6)
    vsb = np.array([0.0, 0.4 * vdd])
    if polarity == "pmos":
        return -vgs, -vds, -vsb
    return vgs, vds, vsb


def _noise_one(card_path, model_name, W, L, vgs, vds, vb, temp_c=27.0):
    """One ngspice ``.noise`` at a bias → (S_thermal, S_flicker@1Hz) [A²/Hz].

    An ideal drain source holds Vds so the device's drain-noise current flows
    through it; a 1 Ω CCVS mirrors that current to a node whose output-noise ASD
    equals √S_id. Sweeping 1 Hz→100·Vdd·GHz and least-squares fitting
    S_id(f)=A+B/f splits white channel noise (A) from the 1/f coefficient (B) —
    all exact ngspice-C BSIM4 (tnoimod/fnoimod), ~0.2 % fit residual. ``temp_c``
    sets the device temperature (°C)."""
    with tempfile.TemporaryDirectory() as td:
        out_txt = os.path.join(td, "out.txt")
        cir = os.path.join(td, "deck.cir")
        deck = (f"* fp45 noise {model_name}\n"
                f'.include "{card_path}"\n'
                f".options temp={temp_c:g}\n"
                f"mn d g s b {model_name} w={W:g}u l={L:g}u\n"
                f"vd d 0 {vds:g}\nvg g 0 {vgs:g} ac 1\nvs s 0 0\nvb b 0 {vb:g}\n"
                f"hn out 0 vd 1\nrout out 0 1e12\n"
                f".control\nset filetype=ascii\nnoise v(out) vg dec 4 1 1e11\n"
                f"setplot noise1\nwrdata {out_txt} onoise_spectrum\n.endc\n.end\n")
        with open(cir, "w") as fh:
            fh.write(deck)
        _run_ngspice(cir, out_txt, timeout=60, what="noise")
        raw = np.loadtxt(out_txt)
    f, asd = raw[:, 0], raw[:, 1]            # onoise_spectrum is ASD [V/√Hz] = √S_id
    s_id = asd ** 2
    A, B = np.linalg.lstsq(np.column_stack([np.ones_like(f), 1.0 / f]), s_id,
                           rcond=None)[0]
    return max(float(A), 0.0), max(float(B), 0.0)


@dataclass(frozen=True)
class NoiseGrid:
    """(S_thermal, S_flicker@1Hz) on a coarse (Vsb, Vds, Vgs) grid [A²/Hz]."""
    vgs: np.ndarray
    vds: np.ndarray
    vsb: np.ndarray
    s_thermal: np.ndarray
    s_flicker_1hz: np.ndarray


def characterize_noise(card_path, model_name, polarity, W, L, corner, vdd=1.0,
                       temp_c=27.0) -> NoiseGrid:
    """Characterise drain-noise (S_thermal, S_flicker@1Hz) over a coarse grid, cached.

    Exact ngspice-C BSIM4 noise per bias (:func:`_noise_one`); the grid is coarser
    than the DC sweep because noise is smooth, and cached to ``*_noise.npz`` so IRN
    reuse needs no ngspice. ``temp_c`` sets the device temperature (°C, keyed)."""
    vgs, vds, vsb = _noise_grid(polarity, vdd)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache = os.path.join(
        _CACHE_DIR,
        _grid_key(model_name, W, L, corner, vgs, vds, vsb, temp_c) + "_noise.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return NoiseGrid(d["vgs"], d["vds"], d["vsb"], d["s_thermal"], d["s_flicker_1hz"])
    if not os.path.exists(card_path):
        raise RuntimeError(f"FreePDK45 card not found at {card_path}; set PDK_ROOT")
    sh = (len(vsb), len(vds), len(vgs))
    s_th = np.zeros(sh)
    s_fl = np.zeros(sh)
    for i, sb in enumerate(vsb):
        vb = -sb                                  # bulk = −Vsb (source at 0); sb signed
        for j, vd in enumerate(vds):
            for k, vg in enumerate(vgs):
                s_th[i, j, k], s_fl[i, j, k] = _noise_one(
                    card_path, model_name, W, L, vg, vd, vb, temp_c)
    np.savez(cache, vgs=vgs, vds=vds, vsb=vsb, s_thermal=s_th, s_flicker_1hz=s_fl)
    return NoiseGrid(vgs, vds, vsb, s_th, s_fl)
