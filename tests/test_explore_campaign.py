"""CompiledCampaign wiring gates for the generic analog ``explore()`` driver."""
from __future__ import annotations

import copy
import json
import os

os.environ.setdefault("CIRCUIT_ENGINE", "rust")

import numpy as np
import pytest

pytest.importorskip("circuitopt_core")

import circuitopt._campaign_sweep as campaign_sweep
import circuitopt.explore as explore_module
from circuitopt.explore import explore_from_dict


def _sky130():
    with open("examples/sky130_5t_ota.json", encoding="utf-8") as handle:
        return json.load(handle)


def _scalar(monkeypatch, data, *, n, seed):
    monkeypatch.setattr(campaign_sweep, "campaign_for", lambda *_a, **_k: None)
    return explore_from_dict(data, n=n, seed=seed, workers=1)


def _assert_metric_parity(native, scalar, keys):
    assert native["status"] == scalar["status"]
    assert native["converged"] == scalar["converged"]
    assert native["feasible"] == scalar["feasible"]
    assert native["pareto"] == scalar["pareto"]
    assert native["noise_evaluated"] == scalar["noise_evaluated"]
    if native["metrics"] is None:
        assert scalar["metrics"] is None
        return
    for key in keys:
        left = float(native["metrics"][key])
        right = float(scalar["metrics"][key])
        if np.isnan(left) and np.isnan(right):
            continue
        assert left == pytest.approx(right, rel=1e-12, abs=1e-15), key


def test_sky130_explore_campaign_matches_scalar_with_candidate_bias(monkeypatch):
    data = _sky130()
    data["explore"]["variables"]["VCM"] = {
        "min": 0.84, "max": 0.94, "round": 3
    }
    native = explore_from_dict(copy.deepcopy(data), n=8, seed=3, workers=4)
    scalar = _scalar(monkeypatch, copy.deepcopy(data), n=8, seed=3)

    for left, right in zip(native["candidates"], scalar["candidates"]):
        _assert_metric_parity(
            left,
            right,
            ("gain_dB", "gain_peak_dB", "bw_Hz", "power_uW", "area"),
        )
    for key in ("converged", "feasible", "pareto", "noise_evaluated", "best"):
        assert native["summary"][key] == scalar["summary"][key]


def test_sky130_explore_campaign_preserves_lazy_noise(monkeypatch):
    data = _sky130()
    data["explore"]["constraints"] = {
        "gain_dB": {"min": 34.0},
        "irn_uV": {"max": 1e9},
    }
    data["explore"]["objectives"] = {
        "irn_uV": "min",
        "power_uW": "min",
    }
    native = explore_from_dict(copy.deepcopy(data), n=6, seed=4, workers=4)
    scalar = _scalar(monkeypatch, copy.deepcopy(data), n=6, seed=4)

    assert native["summary"]["noise_evaluated"] == scalar["summary"]["noise_evaluated"]
    assert 0 < native["summary"]["noise_evaluated"] < 6
    for left, right in zip(native["candidates"], scalar["candidates"]):
        _assert_metric_parity(
            left,
            right,
            ("gain_dB", "gain_peak_dB", "bw_Hz", "irn_uV", "power_uW", "area"),
        )
        if left["noise_evaluated"]:
            left_out = left["metrics"]["measurements"]["integrated_output_noise"]["value"]
            right_out = right["metrics"]["measurements"]["integrated_output_noise"]["value"]
            assert left_out == pytest.approx(right_out, rel=1e-12)


def test_compiled_explore_has_no_python_solver_or_device_callback(monkeypatch):
    from circuitopt.compact_models.bsim4 import NativeBsim4Backend

    calls = {"count": 0}

    def boom(*_args, **_kwargs):
        calls["count"] += 1
        raise AssertionError("Python solver/device callback entered compiled explore")

    monkeypatch.setattr(explore_module, "evaluate", boom)
    monkeypatch.setattr(explore_module, "ac_solve", boom)
    monkeypatch.setattr(explore_module, "noise_analysis", boom)
    monkeypatch.setattr(NativeBsim4Backend, "evaluate", boom)
    monkeypatch.setattr(NativeBsim4Backend, "evaluate_batch", staticmethod(boom))
    monkeypatch.setattr(NativeBsim4Backend, "noise_batch", staticmethod(boom))

    result = explore_from_dict(_sky130(), n=8, seed=1, workers=4)
    assert result["summary"]["converged"] == 8
    assert calls["count"] == 0


def test_compiled_explore_is_worker_deterministic():
    data = _sky130()
    serial = explore_from_dict(copy.deepcopy(data), n=12, seed=7, workers=1)
    parallel = explore_from_dict(copy.deepcopy(data), n=12, seed=7, workers=8)
    assert serial["summary"] == parallel["summary"]
    for left, right in zip(serial["candidates"], parallel["candidates"]):
        for key in ("gain_dB", "gain_peak_dB", "bw_Hz", "power_uW", "area"):
            assert left["metrics"][key] == right["metrics"][key]


def test_compiled_explore_cancellation_stops_between_native_chunks():
    checks = {"count": 0}
    progress = []

    def stop():
        checks["count"] += 1
        return checks["count"] > 1

    result = explore_from_dict(
        _sky130(),
        n=40,
        seed=2,
        workers=2,
        should_stop=stop,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert result["stopped_early"] is True
    assert result["summary"]["stopped_early"] is True
    assert result["summary"]["evaluated"] == 8
    assert progress == [(index, 40) for index in range(1, 9)]

