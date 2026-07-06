"""Observability for the solvers' deliberate exception-fallback paths.

The numerical solvers here catch broad exceptions on purpose: a diverged Newton
re-seeds, a singular Jacobian drops to least-squares, a failed device evaluation
substitutes zero current / gm so a design sweep keeps going instead of aborting
on one bad bias point. That resilience is intentional and must stay -- recovering
from a bad operating point is why a hundred-point sweep completes at all.

The hazard is a *silent* recovery. When a model evaluation fails and the code
returns gm=0 / Id=0, the circuit still "solves" -- to a plausible-looking but
physically wrong answer -- with nothing recorded. This module makes every such
fallback observable **without changing any numerical result**::

    from . import diagnostics
    ...
    except Exception as exc:
        diagnostics.note_critical(
            "model.ss_params_zeroed", exc, detail="gm/gds -> 0/1e-12")
        return {"gm": 0.0, "gds": 1e-12, ...}

- ``note()`` / ``note_critical()`` bump a process-wide counter keyed by
  ``category``. They only run on the exceptional branch, never raise, and return
  nothing the caller uses, so behaviour -- and the Cadence byte-gate -- is
  untouched. ``note_critical`` is for fallbacks that *fabricate a physical value*
  (a zeroed device); its first sighting logs at WARNING so it surfaces on stderr
  under Python's default logging while repeats stay counter-only.
- ``snapshot()`` / ``last_details()`` / ``summary()`` read the tallies;
  ``reset()`` clears them. A driver can fold ``snapshot()`` into its result to
  report "this run leaned on N fallbacks", and tests assert specific categories
  fired.
- Set ``CIRCUIT_DIAG=1`` to log *every* event (>= INFO) for verbose triage.

Category convention: ``"<area>.<reason>"`` -- ``model.*`` for device
evaluations, ``dc.*`` for the DC continuation ladder, and ``transient.*`` /
``pss.*`` / ``pac.*`` / ``pnoise.*`` for the periodic solvers. Suffix ``_zeroed``
marks a fabricated-value fallback (the ones routed through ``note_critical``).
"""
from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger("circuitopt.diagnostics")

_lock = threading.Lock()
_counts: dict[str, int] = {}
_last_detail: dict[str, str] = {}
_seen: set[str] = set()

# CIRCUIT_DIAG=1 -> log every event at >= INFO (verbose triage runs).
_VERBOSE = os.environ.get("CIRCUIT_DIAG", "").strip().lower() not in (
    "", "0", "false", "no", "off")


def note(category, exc=None, *, detail="", level=logging.DEBUG):
    """Record one solver-fallback event under ``category``.

    Cheap (a dict increment), silent by default, and guaranteed never to raise --
    it runs only on an already-exceptional branch, so it must not add a second
    failure mode. The first sighting of a category is logged at ``level``;
    repeats stay counter-only (``CIRCUIT_DIAG=1`` logs every event at >= INFO).
    """
    try:
        with _lock:
            count = _counts[category] = _counts.get(category, 0) + 1
            msg = detail or (
                f"{type(exc).__name__}: {exc}" if exc is not None else "")
            if msg:
                _last_detail[category] = msg
            first = category not in _seen
            if first:
                _seen.add(category)
        if _VERBOSE:
            log_level = max(level, logging.INFO)
        elif first:
            log_level = level
        else:
            log_level = logging.DEBUG
        if logger.isEnabledFor(log_level):
            logger.log(log_level, "solver fallback: %s (#%d)%s",
                       category, count, f" [{msg}]" if msg else "")
    except Exception:       # diagnostics must never perturb the solve
        pass


def note_critical(category, exc=None, *, detail=""):
    """Record a fallback that *fabricates a physical value* (zeroed device/current).

    Same as :func:`note`, but the first sighting logs at WARNING so a silently
    zeroed model surfaces on stderr under Python's default logging. Use this for
    the paths that substitute ``gm=0`` / ``Id=0`` / zero noise when a device
    evaluation fails -- the ones that turn a diverged model into a
    plausible-but-wrong solve.
    """
    note(category, exc, detail=detail, level=logging.WARNING)


def snapshot():
    """Return a ``{category: count}`` copy of the events recorded so far."""
    with _lock:
        return dict(_counts)


def last_details():
    """Return a ``{category: last message}`` copy (exception repr or detail)."""
    with _lock:
        return dict(_last_detail)


def total():
    """Total number of fallback events across all categories."""
    with _lock:
        return sum(_counts.values())


def reset():
    """Clear all counters -- call at the start of a run or test."""
    with _lock:
        _counts.clear()
        _last_detail.clear()
        _seen.clear()


def summary():
    """Human-readable multi-line tally, most frequent first."""
    with _lock:
        if not _counts:
            return "diagnostics: no solver-fallback events recorded"
        rows = sorted(_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        lines = ["diagnostics: solver-fallback events (count  category)"]
        for cat, cnt in rows:
            d = _last_detail.get(cat, "")
            lines.append(f"  {cnt:6d}  {cat}" + (f"  ({d})" if d else ""))
        return "\n".join(lines)
