"""Multi-testbench PVT signoff campaigns.

The campaign layer is deliberately a control-plane module: circuit JSON files
remain the source of topology and analysis truth, while a small manifest binds
those testbenches to an explicit process/voltage/temperature grid.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from ._parallel import worker_device_eval
from .analysis_dispatch import run_analysis_suite
from .compact_models.bsim4 import isolated_native_device_cache
from .circuit_loader import circuit_from_dict
from .run_contract import (
    SimulationInvalid,
    evaluate_signoff,
    validate_signoff_config,
)


SCHEMA_VERSION = "1.0"
_ROOT_KEYS = {"name", "pvt", "cases"}
_PVT_KEYS = {
    "corners", "temperatures_c", "supplies_v",
    "nominal_supply_v", "supply_bias_key",
}
_CASE_KEYS = {"name", "circuit", "overrides"}
_PVT_EXPR_KEYS = {"vdd", "temperature_c", "constant"}


class CampaignConfigurationError(ValueError):
    """A campaign manifest is ambiguous, unsafe, or structurally invalid."""


def _strict_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise CampaignConfigurationError(
            f"{path} has unknown field(s): {', '.join(unknown)}")


def _finite_float(value: Any, path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CampaignConfigurationError(f"{path} must be numeric") from exc
    if not math.isfinite(number):
        raise CampaignConfigurationError(f"{path} must be finite")
    return number


def _unique_strings(values: Any, path: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise CampaignConfigurationError(f"{path} must be a non-empty array")
    out = [str(value).strip().lower() for value in values]
    if any(not value for value in out) or len(set(out)) != len(out):
        raise CampaignConfigurationError(
            f"{path} must contain unique non-empty strings")
    return out


def _unique_numbers(values: Any, path: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise CampaignConfigurationError(f"{path} must be a non-empty array")
    out = [_finite_float(value, f"{path}[{index}]")
           for index, value in enumerate(values)]
    if len(set(out)) != len(out):
        raise CampaignConfigurationError(f"{path} must contain unique values")
    return out


def validate_campaign_config(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one parsed campaign manifest."""
    if not isinstance(data, Mapping):
        raise CampaignConfigurationError("campaign root must be an object")
    _strict_keys(data, _ROOT_KEYS, "campaign")
    name = str(data.get("name", "")).strip()
    if not name:
        raise CampaignConfigurationError("campaign.name is required")

    pvt = data.get("pvt")
    if not isinstance(pvt, Mapping):
        raise CampaignConfigurationError("campaign.pvt must be an object")
    _strict_keys(pvt, _PVT_KEYS, "campaign.pvt")
    corners = _unique_strings(pvt.get("corners"), "campaign.pvt.corners")
    temperatures = _unique_numbers(
        pvt.get("temperatures_c"), "campaign.pvt.temperatures_c")
    supplies = _unique_numbers(
        pvt.get("supplies_v"), "campaign.pvt.supplies_v")
    if any(supply <= 0.0 for supply in supplies):
        raise CampaignConfigurationError(
            "campaign.pvt.supplies_v values must be positive")
    nominal = _finite_float(
        pvt.get("nominal_supply_v"), "campaign.pvt.nominal_supply_v")
    if nominal <= 0.0:
        raise CampaignConfigurationError(
            "campaign.pvt.nominal_supply_v must be positive")
    supply_key = str(pvt.get("supply_bias_key", "")).strip()
    if not supply_key:
        raise CampaignConfigurationError(
            "campaign.pvt.supply_bias_key is required")

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise CampaignConfigurationError(
            "campaign.cases must be a non-empty array")
    cases = []
    names = set()
    for index, case in enumerate(raw_cases):
        path = f"campaign.cases[{index}]"
        if not isinstance(case, Mapping):
            raise CampaignConfigurationError(f"{path} must be an object")
        _strict_keys(case, _CASE_KEYS, path)
        case_name = str(case.get("name", "")).strip()
        if not case_name or case_name in names:
            raise CampaignConfigurationError(
                f"{path}.name must be non-empty and unique")
        names.add(case_name)
        circuit = str(case.get("circuit", "")).strip()
        if not circuit:
            raise CampaignConfigurationError(f"{path}.circuit is required")
        circuit_path = Path(circuit)
        if circuit_path.is_absolute():
            raise CampaignConfigurationError(
                f"{path}.circuit must be relative to the campaign file")
        overrides = case.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise CampaignConfigurationError(f"{path}.overrides must be an object")
        cases.append({
            "name": case_name,
            "circuit": circuit,
            "overrides": deepcopy(dict(overrides)),
        })

    return {
        "name": name,
        "pvt": {
            "corners": corners,
            "temperatures_c": temperatures,
            "supplies_v": supplies,
            "nominal_supply_v": nominal,
            "supply_bias_key": supply_key,
        },
        "cases": cases,
    }


def load_campaign_json(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a campaign and return ``(normalized_config, manifest_path)``."""
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return validate_campaign_config(data), manifest_path


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged = deepcopy(dict(base))
        for key, value in override.items():
            merged[key] = (
                _deep_merge(merged[key], value)
                if key in merged else deepcopy(value)
            )
        return merged
    return deepcopy(override)


def _resolve_pvt_expressions(value: Any, *, vdd: float, temperature_c: float) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"$pvt"}:
            expression = value["$pvt"]
            if not isinstance(expression, Mapping) or not expression:
                raise CampaignConfigurationError(
                    "$pvt must contain at least one coefficient")
            _strict_keys(expression, _PVT_EXPR_KEYS, "$pvt")
            total = 0.0
            for key, coefficient in expression.items():
                coefficient = _finite_float(coefficient, f"$pvt.{key}")
                if key == "vdd":
                    total += coefficient * vdd
                elif key == "temperature_c":
                    total += coefficient * temperature_c
                else:
                    total += coefficient
            if not math.isfinite(total):
                raise CampaignConfigurationError("$pvt expression is non-finite")
            return total
        if "$pvt" in value:
            raise CampaignConfigurationError(
                "$pvt expression objects cannot contain sibling fields")
        return {
            key: _resolve_pvt_expressions(
                child, vdd=vdd, temperature_c=temperature_c)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_pvt_expressions(
                child, vdd=vdd, temperature_c=temperature_c)
            for child in value
        ]
    return deepcopy(value)


def _scale_numeric_vsources(deck: dict[str, Any], ratio: float) -> None:
    for source in deck.get("vsources", ()):
        if isinstance(source, list) and len(source) == 4:
            if isinstance(source[3], (int, float)) and not isinstance(source[3], bool):
                source[3] = float(source[3]) * ratio
        elif isinstance(source, dict):
            value = source.get("V")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                source["V"] = float(value) * ratio


def prepare_case_dict(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    corner: str,
    temperature_c: float,
    supply_v: float,
    nominal_supply_v: float,
    supply_bias_key: str,
) -> dict[str, Any]:
    """Return a circuit dictionary with one exact PVT point baked into it."""
    resolved_overrides = _resolve_pvt_expressions(
        overrides, vdd=supply_v, temperature_c=temperature_c)
    deck = _deep_merge(base, resolved_overrides)
    ratio = supply_v / nominal_supply_v

    bias = deck.setdefault("bias", {})
    if not isinstance(bias, dict):
        raise CampaignConfigurationError("circuit bias must be an object")
    bias[supply_bias_key] = supply_v

    for guess in deck.get("dc_guesses", ()):
        if not isinstance(guess, dict):
            raise CampaignConfigurationError(
                "circuit dc_guesses entries must be objects")
        for node, value in list(guess.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                guess[node] = float(value) * ratio

    models = deck.get("models", {})
    if not isinstance(models, dict) or (deck.get("devices") and "models" not in deck):
        raise CampaignConfigurationError(
            "campaign circuits must explicitly contain a models object")
    temperature_k = temperature_c + 273.15
    if temperature_k <= 0.0:
        raise CampaignConfigurationError(
            f"temperature {temperature_c:g} degC is below absolute zero")
    for name, model in models.items():
        if not isinstance(model, dict):
            raise CampaignConfigurationError(
                f"circuit models.{name} must be an object")
        section = str(model.get("section", "")).strip().lower()
        if section == "inherit":
            model["section"] = corner
        elif section != corner:
            raise CampaignConfigurationError(
                f"models.{name}.section {section!r} conflicts with "
                f"campaign corner {corner!r}")
        model["temperature"] = temperature_k
        if "vb" in model:
            model["vb"] = float(model["vb"]) * ratio

    _scale_numeric_vsources(deck, ratio)
    return deck


def _load_case_bases(
    config: Mapping[str, Any],
    manifest_path: Path,
) -> list[dict[str, Any]]:
    root = manifest_path.parent.resolve()
    loaded = []
    for case in config["cases"]:
        path = (root / case["circuit"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise CampaignConfigurationError(
                f"case {case['name']!r} circuit path escapes campaign directory: "
                f"{case['circuit']}") from exc
        if not path.is_file():
            raise CampaignConfigurationError(
                f"case {case['name']!r} circuit file not found: {case['circuit']}")
        with path.open("r", encoding="utf-8") as handle:
            base = json.load(handle)
        loaded.append({**case, "base": base})
    return loaded


def _case_invalid(name: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, SimulationInvalid):
        error = exc.as_dict()
    else:
        error = {
            "code": "campaign_case_failed",
            "message": f"{type(exc).__name__}: {exc}",
        }
    return {
        "name": name,
        "status": "invalid",
        "passed": False,
        "error": error,
        "signoff": None,
    }


def _case_worst(case: Mapping[str, Any]) -> dict[str, Any] | None:
    if case["status"] == "invalid":
        return {
            "case": case["name"],
            "measurement": None,
            "passed": False,
            "normalized_margin": None,
            "status": "invalid",
        }
    worst = case["signoff"].get("worst_case")
    if worst is None:
        return None
    return {
        "case": case["name"],
        "measurement": worst["measurement"],
        "passed": bool(worst["passed"]),
        "normalized_margin": worst["normalized_margin"],
        "status": case["status"],
    }


def _select_worst(cases: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    candidates = [candidate for case in cases
                  if (candidate := _case_worst(case)) is not None]
    if not candidates:
        return None
    invalid = [candidate for candidate in candidates
               if candidate["status"] == "invalid"]
    if invalid:
        return invalid[0]
    return min(
        candidates,
        key=lambda item: (
            item["normalized_margin"]
            if item["normalized_margin"] is not None else -math.inf
        ),
    )


def _run_point(
    point_index: int,
    point: tuple[str, float, float],
    cases: list[dict[str, Any]],
    pvt: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Evaluate one PVT point, isolated from every other point.

    Native BSIM handles carry state between calls, so points that share a card
    -- same corner and temperature, different supply -- would otherwise lease
    the same handle and see each other's warm start. That made the campaign
    reproduce only at one worker. Leasing into a per-point namespace makes a
    point's result a function of that point alone at any worker count.
    """
    with isolated_native_device_cache():
        return _run_point_isolated(point_index, point, cases, pvt)


def _run_point_isolated(
    point_index: int,
    point: tuple[str, float, float],
    cases: list[dict[str, Any]],
    pvt: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    corner, temperature_c, supply_v = point
    case_results = []
    for case in cases:
        try:
            deck = prepare_case_dict(
                case["base"],
                case["overrides"],
                corner=corner,
                temperature_c=temperature_c,
                supply_v=supply_v,
                nominal_supply_v=pvt["nominal_supply_v"],
                supply_bias_key=pvt["supply_bias_key"],
            )
            spec = circuit_from_dict(deck)
            if not spec.signoff:
                raise CampaignConfigurationError(
                    f"case {case['name']!r} has no signoff configuration")
            results = run_analysis_suite(spec)
            signoff = evaluate_signoff(spec, results)
            if signoff["status"] not in {"pass", "fail"}:
                raise CampaignConfigurationError(
                    f"case {case['name']!r} signoff status is "
                    f"{signoff['status']!r}")
            case_results.append({
                "name": case["name"],
                "status": signoff["status"],
                "passed": bool(signoff["passed"]),
                "error": None,
                "signoff": signoff,
            })
        except Exception as exc:
            case_results.append(_case_invalid(case["name"], exc))

    if any(case["status"] == "invalid" for case in case_results):
        status = "invalid"
    elif all(case["passed"] for case in case_results):
        status = "pass"
    else:
        status = "fail"
    worst = _select_worst(case_results)
    if worst is not None:
        worst = {
            **worst,
            "corner": corner,
            "temperature_c": temperature_c,
            "supply_v": supply_v,
        }
    return point_index, {
        "pvt": {
            "corner": corner,
            "temperature_c": temperature_c,
            "supply_v": supply_v,
        },
        "status": status,
        "passed": status == "pass",
        "cases": {case["name"]: {
            key: value for key, value in case.items() if key != "name"
        } for case in case_results},
        "worst_case": worst,
    }


def _run_point_worker(
    point_index: int,
    point: tuple[str, float, float],
    cases: list[dict[str, Any]],
    pvt: Mapping[str, Any],
    workers: int,
) -> tuple[int, dict[str, Any]]:
    """:func:`_run_point` on a worker thread of a parallel campaign.

    Points are the parallel level here, so each one evaluates its device
    batches inline rather than contending for the shared pool.
    """
    with worker_device_eval(workers):
        return _run_point(point_index, point, cases, pvt)


def _global_worst(points: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    candidates = [point["worst_case"] for point in points
                  if point["worst_case"] is not None]
    if not candidates:
        return None
    invalid = [candidate for candidate in candidates
               if candidate["status"] == "invalid"]
    if invalid:
        return invalid[0]
    return min(
        candidates,
        key=lambda item: (
            item["normalized_margin"]
            if item["normalized_margin"] is not None else -math.inf
        ),
    )


def run_signoff_campaign(
    config_or_path: Mapping[str, Any] | str | Path,
    *,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run every case over the Cartesian PVT grid and aggregate strict signoff.

    ``should_stop`` is checked between PVT points.  It provides cooperative
    cancellation for service transports; a point already executing is allowed
    to finish so solver state is never interrupted mid-analysis.
    """
    if workers < 1:
        raise CampaignConfigurationError("workers must be at least 1")
    if isinstance(config_or_path, Mapping):
        config = validate_campaign_config(config_or_path)
        manifest_path = Path.cwd() / "signoff_campaign.json"
    else:
        config, manifest_path = load_campaign_json(config_or_path)
    cases = _load_case_bases(config, manifest_path)
    pvt = config["pvt"]
    points = [
        (corner, temperature, supply)
        for corner in pvt["corners"]
        for temperature in pvt["temperatures_c"]
        for supply in pvt["supplies_v"]
    ]

    # Parse every corner before launching workers. Structural and fixed-section
    # conflicts are configuration failures, not repeated simulation-invalid rows.
    first_temperature = pvt["temperatures_c"][0]
    first_supply = pvt["supplies_v"][0]
    for corner in pvt["corners"]:
        for case in cases:
            deck = prepare_case_dict(
                case["base"],
                case["overrides"],
                corner=corner,
                temperature_c=first_temperature,
                supply_v=first_supply,
                nominal_supply_v=pvt["nominal_supply_v"],
                supply_bias_key=pvt["supply_bias_key"],
            )
            spec = circuit_from_dict(deck)
            if not spec.signoff:
                raise CampaignConfigurationError(
                    f"case {case['name']!r} has no signoff configuration")
            validate_signoff_config(spec)

    completed = 0
    ordered: list[dict[str, Any] | None] = [None] * len(points)
    if workers == 1:
        for index, point in enumerate(points):
            if should_stop is not None and should_stop():
                break
            result_index, result = _run_point(index, point, cases, pvt)
            ordered[result_index] = result
            completed += 1
            if progress is not None:
                progress(completed, len(points))
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {
                executor.submit(_run_point_worker, index, point, cases, pvt,
                                workers): index
                for index, point in enumerate(points)
            }
            for future in as_completed(futures):
                if should_stop is not None and should_stop():
                    for pending in futures:
                        pending.cancel()
                    break
                result_index, result = future.result()
                ordered[result_index] = result
                completed += 1
                if progress is not None:
                    progress(completed, len(points))
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    point_results = [point for point in ordered if point is not None]
    stopped_early = len(point_results) != len(points)

    counts = {
        status: sum(point["status"] == status for point in point_results)
        for status in ("pass", "fail", "invalid")
    }
    case_counts = {}
    for case in config["cases"]:
        name = case["name"]
        case_counts[name] = {
            status: sum(point["cases"][name]["status"] == status
                        for point in point_results)
            for status in ("pass", "fail", "invalid")
        }
    if counts["invalid"]:
        status = "invalid"
    elif counts["fail"]:
        status = "fail"
    else:
        status = "pass"
    result = {
        "schema_version": SCHEMA_VERSION,
        "name": config["name"],
        "status": status,
        "passed": status == "pass",
        "grid": {
            **deepcopy(pvt),
            "total_points": len(points),
        },
        "cases": [
            {"name": case["name"], "circuit": case["circuit"]}
            for case in config["cases"]
        ],
        "summary": {
            "points": {**counts, "total": len(point_results)},
            "total_case_runs": len(point_results) * len(config["cases"]),
            "cases": case_counts,
        },
        "points": point_results,
        "worst_case": _global_worst(point_results),
    }
    if stopped_early:
        result["stopped_early"] = True
        result["status"] = "cancelled"
        result["passed"] = False
    return result
