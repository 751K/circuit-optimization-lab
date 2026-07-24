"""Transport-neutral application operations for HTTP and MCP adapters.

The numerical stack has one public control surface here.  Protocol adapters
translate their own request/error conventions, but circuit parsing, option
validation, solver dispatch, signoff evaluation, and JSON serialization stay
identical across transports.
"""
from __future__ import annotations

import time
from typing import Any, Iterable

from .. import __version__
from ..analysis_dispatch import ANALYSIS_ORDER, run_analysis_suite
from ..analysis_options import known_keys, validate_analysis_cfg
from ..circuit_loader import circuit_from_dict
from ..device_factory import CORNERS, SKY130_CORNERS
from ..device_model import registered_models
from ..freepdk45_model import FREEPDK45_CORNERS
from ..run_contract import (
    SimulationInvalid,
    evaluate_signoff,
    validate_signoff_config,
)
from .jobs import JOB_KINDS
from .serialize import serialize_results, to_jsonable


class OperationError(RuntimeError):
    """A stable parse/solve failure independent of HTTP or MCP."""

    def __init__(self, stage: str, message: str, **details: Any):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            **to_jsonable(self.details),
        }


def build_capabilities(*, jobs: Iterable[str] = JOB_KINDS) -> dict[str, Any]:
    """Return capabilities assembled from the authoritative registries."""
    analyses = {name: sorted(known_keys(name)) for name in ANALYSIS_ORDER}
    return {
        "version": __version__,
        "api": "v1",
        "models": registered_models(),
        "analyses": analyses,
        "corners": {
            "otft": sorted(CORNERS),
            "sky130": sorted(SKY130_CORNERS),
            "freepdk45": sorted(FREEPDK45_CORNERS),
        },
        "jobs": list(jobs),
    }


def validate_circuit(circuit: dict[str, Any]) -> dict[str, Any]:
    """Parse and validate a circuit without running numerical analyses."""
    errors: list[str] = []
    try:
        spec = circuit_from_dict(circuit)
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)]}

    for name, cfg in (spec.analyses or {}).items():
        if isinstance(cfg, dict):
            try:
                validate_analysis_cfg(name, cfg)
            except Exception as exc:
                errors.append(str(exc))
    try:
        validate_signoff_config(spec)
    except Exception as exc:
        errors.append(str(exc))
    return {"valid": not errors, **({"errors": errors} if errors else {})}


def solve_circuit(
    circuit: dict[str, Any],
    *,
    selected: list[str] | None = None,
    corner: str | None = None,
) -> dict[str, Any]:
    """Run one analysis suite and return the stable serialized envelope."""
    try:
        spec = circuit_from_dict(circuit)
    except Exception as exc:
        raise OperationError("parse", str(exc)) from exc

    started = time.perf_counter()
    try:
        results = run_analysis_suite(spec, selected=selected, corner=corner)
        signoff = evaluate_signoff(spec, results)
    except SimulationInvalid as exc:
        payload = exc.as_dict()
        message = str(payload.pop("message", str(exc)))
        raise OperationError(
            "solve", message, status="invalid", **payload
        ) from exc
    except Exception as exc:
        raise OperationError("solve", str(exc)) from exc

    return {
        "status": "valid",
        "results": serialize_results(results),
        "signoff": to_jsonable(signoff),
        "elapsed_s": to_jsonable(time.perf_counter() - started),
    }
