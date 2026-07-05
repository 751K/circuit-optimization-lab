"""Generic PSFASCII parser for Cadence/Spectre reference data.

Cadence writes ASCII PSF (``spectre -format psfascii``) in three sections:

    HEADER   provenance key/value pairs ("PSFversion", "version", "date", ...)
    TYPE     signal type declarations (struct column order for noise, etc.)
    VALUE    the data block, terminated by END

This module turns each analysis output into plain numpy arrays / dicts for the
calibration engine (:mod:`core.calibration`), and pulls provenance straight out
of the HEADER so reference files are self-describing. It is solver-independent.

Supported analyses:
    DC op point   parse_dc(path)        -> {signal: float}
    DC sweep      parse_dc_sweep(path)  -> (sweep, {signal: array})
    AC / PAC      parse_ac(path)        -> (freqs, {signal: complex array})
    noise/pnoise  parse_noise(path)     -> (freqs, out_asd, {device: (Nf,3) flick/therm/total})
    tran / PSS    parse_tran(path)      -> (time, {signal: real array})

The ad-hoc parsers in ``tools/calibrate_switch.py`` are the historical seed for
this module (DC sweep + AC); the noise/pnoise/pss handling matches the verified
Cadence cross-check workflow.
"""
from __future__ import annotations

import re

import numpy as np

# Spectre prints doubles like -7.007000e+00; accept ints, decimals, exponents.
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
# Sweep variables that delimit one data point in a VALUE block.
_SWEEP_VARS = ("freq", "hertz", "time", "sweep", "dc")


def _read_lines(path) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\n") for ln in f]


def _value_lines(path) -> list[str]:
    """The lines between the ``VALUE`` and ``END`` markers."""
    lines = _read_lines(path)
    try:
        i0 = next(i for i, l in enumerate(lines) if l.strip() == "VALUE")
    except StopIteration:
        raise ValueError(f"{path}: no VALUE section (not a PSFASCII file?)")
    i1 = next((i for i in range(i0 + 1, len(lines)) if lines[i].strip() == "END"),
              len(lines))
    return lines[i0 + 1:i1]


# ── HEADER / provenance ──────────────────────────────────────────────────────

def parse_header(path) -> dict:
    """HEADER key -> value (strings unquoted, numerics coerced to float)."""
    out = {}
    for ln in _read_lines(path):
        s = ln.strip()
        if s in ("TYPE", "VALUE"):
            break
        if s == "HEADER":
            continue
        m = re.match(r'^"([^"]+)"\s+(.*)$', s)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        if raw.startswith('"') and raw.endswith('"'):
            out[key] = raw[1:-1]
        else:
            try:
                out[key] = float(raw)
            except ValueError:
                out[key] = raw
    return out


def provenance(path) -> dict:
    """Compact provenance dict pulled from the PSF HEADER — Spectre version, run
    date, analysis type, and (for periodic analyses) the fundamental."""
    h = parse_header(path)
    return {
        "psf_version": h.get("PSFversion"),
        "simulator": h.get("simulator"),
        "spectre_version": h.get("version"),
        "date": h.get("date"),
        "analysis_type": h.get("analysis type"),
        "analysis_name": h.get("analysis name"),
        # PSF spells the periodic drive frequency "fundamental frequency"
        # (e.g. pac.0.pac / pnoise HEADER); fall back to the bare "fundamental"
        # spelling only if a non-standard writer ever emits it. Non-periodic
        # analyses (dc/ac/noise/tran) carry neither key, so this stays None there.
        "fundamental": h.get("fundamental frequency", h.get("fundamental")),
    }


# ── DC ───────────────────────────────────────────────────────────────────────

_DC_OP = re.compile(r'^"([^"]+)"\s+(?:"[^"]*"\s+)?(' + _NUM + r")\s*$")


def parse_dc(path) -> dict:
    """DC operating point -> {signal: value}. Accepts ``"v" "V" 1.0`` and ``"v" 1.0``.
    For a swept DC analysis this returns only the last sweep point — use
    :func:`parse_dc_sweep` for sweeps."""
    out = {}
    for ln in _value_lines(path):
        m = _DC_OP.match(ln.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def parse_dc_sweep(path) -> tuple:
    """Swept DC -> (sweep_array, {signal: array}). The sweep variable is whichever
    of ``sweep``/``dc`` appears; every other scalar is a traced signal."""
    sweep, data = [], {}
    for ln in _value_lines(path):
        m = _DC_OP.match(ln.strip())
        if not m:
            continue
        name, val = m.group(1), float(m.group(2))
        if name in ("sweep", "dc"):
            sweep.append(val)
        else:
            data.setdefault(name, []).append(val)
    return np.array(sweep), {k: np.array(v) for k, v in data.items()}


# ── AC / PAC ─────────────────────────────────────────────────────────────────

_FREQ = re.compile(r'^"(?:freq|hertz)"\s+(' + _NUM + r")")
_CPLX = re.compile(r'^"([^"]+)"\s+\(\s*(' + _NUM + r")\s+(" + _NUM + r")\s*\)")


def parse_ac(path) -> tuple:
    """AC / PAC -> (freqs, {signal: complex array}). Values are ``(real imag)``."""
    freqs, data = [], {}
    for ln in _value_lines(path):
        s = ln.strip()
        m = _FREQ.match(s)
        if m:
            freqs.append(float(m.group(1)))
            continue
        m = _CPLX.match(s)
        if m:
            data.setdefault(m.group(1), []).append(
                complex(float(m.group(2)), float(m.group(3))))
    return np.array(freqs), {k: np.array(v) for k, v in data.items()}


parse_pac = parse_ac


# ── noise / pnoise ───────────────────────────────────────────────────────────

_OUT = re.compile(r'^"out"\s+(?:"[^"]*"\s+)?(' + _NUM + r")\s*$")
_TUPLE_OPEN = re.compile(r'^"([^"]+)"\s+\(\s*$')


def parse_noise(path) -> tuple:
    """noise / pnoise -> (freqs, out_asd, {device: (Nfreq, W) array}).

    ``out_asd`` is the total output noise ASD (the ``"out"`` signal, V/√Hz). Each
    per-device contribution is the multi-line struct declared for that device's
    master in the TYPE section, parsed in declared column order.

    **The per-device width W is NOT fixed — it follows the TYPE struct, so a
    mixed-device fixture is ragged.** A MOSFET (``pmos_TFT_behavioral``) struct
    declares ``(flicker, thermal, total)`` → W=3; a resistor (``resistor``)
    struct declares only ``(rn, total)`` → W=2; other masters may declare other
    field counts. Callers MUST NOT assume W=3: to read the total contribution,
    index the LAST column (``[:, -1]``), not ``[:, 2]``, and check ``.shape[1]``
    before slicing a specific field. (``freqs`` and ``out_asd`` are separate
    ``"freq"``/``"out"`` records, not part of any per-device struct.)"""
    body = _value_lines(path)
    freqs, out, dev = [], [], {}
    i = 0
    while i < len(body):
        s = body[i].strip()
        m = _FREQ.match(s)
        if m:
            freqs.append(float(m.group(1)))
            i += 1
            continue
        m = _OUT.match(s)
        if m:
            out.append(float(m.group(1)))
            i += 1
            continue
        m = _TUPLE_OPEN.match(s)
        if m:
            name, vals = m.group(1), []
            i += 1
            while i < len(body) and not body[i].strip().startswith(")"):
                tok = body[i].strip()
                if re.match(r"^" + _NUM + r"$", tok):
                    vals.append(float(tok))
                i += 1
            i += 1                                    # skip the ")"
            dev.setdefault(name, []).append(vals)
            continue
        i += 1
    return (np.array(freqs), np.array(out),
            {k: np.array(v) for k, v in dev.items()})


parse_pnoise = parse_noise


# ── transient / PSS time-domain ──────────────────────────────────────────────

_TIME = re.compile(r'^"time"\s+(' + _NUM + r")")
_REAL = re.compile(r'^"([^"]+)"\s+(' + _NUM + r")\s*$")


def parse_tran(path, signals=None) -> tuple:
    """transient / PSS td -> (time, {signal: real array}). ``signals`` optionally
    restricts which traces are kept."""
    keep = set(signals) if signals is not None else None
    time, data = [], {}
    for ln in _value_lines(path):
        s = ln.strip()
        m = _TIME.match(s)
        if m:
            time.append(float(m.group(1)))
            continue
        m = _REAL.match(s)
        if m and m.group(1) != "time":
            name = m.group(1)
            if keep is None or name in keep:
                data.setdefault(name, []).append(float(m.group(2)))
    return np.array(time), {k: np.array(v) for k, v in data.items()}


parse_pss = parse_tran
