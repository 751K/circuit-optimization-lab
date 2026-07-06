"""JSON-safe serialization of solver result payloads.

Solver results are dicts of numpy scalars, ndarrays, Python complex numbers,
and nested dicts/lists. HTTP responses must be strict JSON, so this module
walks a result recursively and rewrites every value into a JSON-safe form.

Conversion conventions (single source of truth for the service layer):

* numpy scalar (``np.float64``, ``np.int64``, ``np.bool_``, …) → the
  corresponding native Python scalar (via ``.item()``).
* ``numpy.ndarray`` → nested Python ``list`` (element-wise through the same
  conventions, so a complex/NaN-bearing array is handled too).
* ``complex`` (Python or numpy) → ``{"re": <float>, "im": <float>}``.
* **Non-finite floats — ``NaN``, ``+Inf``, ``-Inf`` → ``null``.** Strict JSON
  (RFC 8259) has no literal for these; emitting them would produce output that
  many parsers reject (and ``json.dumps`` only emits via the non-standard
  ``NaN``/``Infinity`` tokens). We map every non-finite real to ``None`` so the
  response is always valid JSON. This applies inside complex parts and array
  elements as well — a complex value with a NaN imaginary part serializes to
  ``{"re": <float>, "im": null}``.
* ``dict`` → dict with string keys and recursively-serialized values.
* ``list`` / ``tuple`` / ``set`` → list of recursively-serialized values.
* Callables and any key starting with ``"_"`` are **dropped** from dicts —
  results occasionally carry private caches / closures that are neither JSON
  nor useful to a client.
* ``bytes`` → decoded UTF-8 string (best effort; ``errors="replace"``).
* Anything else already JSON-native (``str``, ``bool``, ``int``, ``None``, and
  finite ``float``) passes through unchanged; a truly unknown type falls back
  to ``str(value)`` rather than raising, so one odd field never sinks a
  response.

The functions here are pure (no I/O, no solver imports) so they can be unit
tested directly and reused by any transport (WebSocket frames in S2, etc.).
"""
from __future__ import annotations

import math

import numpy as np


def _finite_or_none(x: float):
    """Real float → itself if finite, else ``None`` (NaN/±Inf are not JSON)."""
    return x if math.isfinite(x) else None


def to_jsonable(value):
    """Recursively convert *value* into a strict-JSON-safe structure.

    See the module docstring for the full conversion table. The result contains
    only ``dict`` / ``list`` / ``str`` / ``bool`` / ``int`` / finite ``float`` /
    ``None`` and is therefore safe for ``json.dumps`` with default settings and
    for FastAPI's default (strict) JSON encoder.
    """
    # ── None / bool (bool before int: bool is a subclass of int) ──
    if value is None or isinstance(value, bool):
        return value

    # ── numpy scalar → native python scalar, then re-dispatch ──
    if isinstance(value, np.generic):
        return to_jsonable(value.item())

    # ── complex (python or numpy already unwrapped above) → {re, im} ──
    if isinstance(value, complex):
        return {"re": _finite_or_none(value.real), "im": _finite_or_none(value.imag)}

    # ── real numbers ──
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _finite_or_none(value)

    # ── strings / bytes ──
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")

    # ── ndarray → nested list (element-wise, so complex/NaN handled) ──
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()] if value.dtype == object \
            else _array_to_jsonable(value)

    # ── mappings ──
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = str(k)
            if key.startswith("_") or callable(v):
                continue
            out[key] = to_jsonable(v)
        return out

    # ── sequences ──
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]

    # ── unknown: don't raise, stringify (keeps one odd field from 500-ing) ──
    return str(value)


def _array_to_jsonable(arr: np.ndarray):
    """Convert a (possibly complex, possibly non-finite) ndarray to nested lists.

    ``arr.tolist()`` yields native Python scalars (``float``/``complex``/``int``/
    ``bool``); recursing through :func:`to_jsonable` applies the NaN→null and
    complex→{re,im} conventions uniformly at every depth.
    """
    return [to_jsonable(v) for v in arr.tolist()]


def serialize_results(results) -> dict:
    """Serialize a ``run_analysis_suite`` result mapping for an HTTP response.

    Thin wrapper over :func:`to_jsonable` that also skips ``None`` analysis
    entries (an analysis that produced no result) so the ``results`` object only
    carries analyses that actually ran.
    """
    return {name: to_jsonable(payload)
            for name, payload in (results or {}).items()
            if payload is not None}
