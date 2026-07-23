"""R9: silicon compiled-campaign arm of the dataset builder.

An all-silicon, fixed-topology size/bias grid with only the AC/noise label group
batches its qualifying ``(bias, nf)`` layers through the compiled campaign; the
AC/DC labels come from the batch and ``power_uW`` / ``area`` from the frozen
post-batch reductions (``explore._supply_power_uW`` over the campaign's per-device
channel current ``ich``; ``explore._area``). Every other candidate — a fragmented
layer (< threshold), a periodic label group, a structural / PVT axis, a per-candidate
DC seed, or an AFE circuit — keeps the scalar ``evaluate`` path byte-for-byte.

Gates: campaign vs scalar row-by-row (bit-for-bit on the freepdk45 size grid,
including the post-processed power/area columns), determinism across workers
{1, 2, 8}, the guards routing non-campaign-able configs to scalar, and a zero
Python device/solver frame gate on the layer batch.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("CIRCUIT_ENGINE", "rust")

import pytest

pytest.importorskip("circuitopt_core")

from circuitopt import dataset as D
from circuitopt._campaign_sweep import campaign_enabled

_LABELS = ("gain_dB", "gain_peak_dB", "bw_Hz", "irn_uV", "power_uW", "area")


def _require_rust():
    if not campaign_enabled():
        pytest.skip("dataset campaign arm requires the rust device engine")


def _freepdk45_ready():
    from circuitopt.toolchain import pdk_root
    if not os.path.isfile(os.path.join(pdk_root(), "freepdk45", "models_nom",
                                       "NMOS_VTG.inc")):
        pytest.skip("FreePDK45 cards not present")


def _size_grid_config(extra_labels=None, afe=False):
    base = "examples/afe_explore.json" if afe else "examples/freepdk45_5t_ota.json"
    data = json.load(open(base))
    if afe:
        # A pure size grid over the AFE design (paired devices vary together).
        data["explore"] = {
            "variables": {"W_IN": {"min": 40000.0, "max": 80000.0,
                                   "targets": ["M7.W", "M8.W"]}},
            "objectives": {"power_uW": "min"},
            "band": [0.05, 100.0],
            "freqs": {"start": -2, "stop": 4, "num": 41},
        }
    else:
        data["explore"] = {
            "variables": {
                "W_IN": {"min": 0.3, "max": 1.0, "targets": ["M1.W", "M2.W"]},
                "L_IN": {"min": 0.05, "max": 0.2, "targets": ["M1.L", "M2.L"]},
                "W_LOAD": {"min": 0.3, "max": 1.0, "targets": ["M3.W", "M4.W"]},
            },
            "objectives": {"power_uW": "min"},
            "band": [1000.0, 1000000.0],
            "freqs": {"start": 3, "stop": 7, "num": 25},
        }
    if extra_labels and "transient" in extra_labels:
        data["periodic"] = {"frequency": 1e6, "periods": 2, "steps_per_period": 20,
                            "inputs": {"vinp": {"kind": "sine", "amplitude": 0.01}}}
    return data


def _write(tmp_path, data, name="ds.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def test_dataset_campaign_matches_scalar_reference(tmp_path, monkeypatch):
    """Campaign vs the forced-scalar reference, row-by-row (bit-for-bit incl. power/area)."""
    _require_rust()
    _freepdk45_ready()
    path = _write(tmp_path, _size_grid_config())
    camp = D.run_from_config(path, n=32, seed=1, corner="nom")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(D, "_campaign_dataset_rows", lambda *a, **k: {})
        scal = D.run_from_config(path, n=32, seed=1, corner="nom")
    assert len(camp["rows"]) == len(scal["rows"]) == 32
    for rc, rs in zip(camp["rows"], scal["rows"]):
        assert rc["idx"] == rs["idx"]                       # provenance/row order (§7)
        assert rc["status"]["dc_converged"] == rs["status"]["dc_converged"]
        for m in _LABELS:
            a, b = rc["metrics"][m], rs["metrics"][m]
            assert (a is None) == (b is None), (rc["idx"], m)
            if a is not None:
                rel = abs(a - b) / max(abs(a), abs(b), 1e-30)
                assert rel <= 1e-12, (rc["idx"], m, rel)


def test_dataset_campaign_is_actually_used(tmp_path):
    """Guard against a vacuous parity test: the campaign really covers the grid."""
    _require_rust()
    _freepdk45_ready()
    from circuitopt.circuit_loader import models_from_config
    from circuitopt.device_factory import CircuitBinding
    from circuitopt.explore import sample
    data = _size_grid_config()
    _, topo, sizes, bias, nf, cfg = D.load_dataset_config(_write(tmp_path, data))
    mt, dk = models_from_config(data)
    binding = CircuitBinding(topo=topo, model_types=mt, device_kwargs=dk, nf=nf)
    size_vars, struct_vars, corner_vars = D.split_variables(cfg.variables)
    size_names = {v.name for v in size_vars}
    samples = sample(cfg.variables, 20, seed=1, method="lhs")
    rows = D._campaign_dataset_rows(samples, size_vars, size_names, sizes, bias, nf,
                                    topo, binding, None, cfg.freqs, cfg.band,
                                    list(D.AC_NOISE_LABELS), 1)
    assert len(rows) == 20                                  # one layer, all covered


def test_dataset_campaign_deterministic_across_workers(tmp_path):
    _require_rust()
    _freepdk45_ready()
    path = _write(tmp_path, _size_grid_config())
    base = D.run_from_config(path, n=32, seed=1, corner="nom")
    for w in (1, 2, 8):
        got = D.run_from_config(path, n=32, seed=1, corner="nom", workers=w)
        for rb, rr in zip(base["rows"], got["rows"]):
            assert rb["idx"] == rr["idx"]
            for m in _LABELS:
                a, b = rb["metrics"][m], rr["metrics"][m]
                assert (a is None) == (b is None), (w, rb["idx"], m)
                if a is not None:
                    assert a == b, (w, rb["idx"], m)


def test_dataset_campaign_threshold_keeps_small_layer_scalar(tmp_path, monkeypatch):
    """A layer below the batch threshold stays on the scalar path (no campaign rows)."""
    _require_rust()
    _freepdk45_ready()
    _, topo, sizes, bias, nf, cfg = D.load_dataset_config(
        _write(tmp_path, _size_grid_config()))
    from circuitopt.circuit_loader import models_from_config
    from circuitopt.device_factory import CircuitBinding
    from circuitopt.explore import sample
    mt, dk = models_from_config(_size_grid_config())
    binding = CircuitBinding(topo=topo, model_types=mt, device_kwargs=dk, nf=nf)
    size_vars, _, _ = D.split_variables(cfg.variables)
    size_names = {v.name for v in size_vars}
    samples = sample(cfg.variables, D._MIN_CAMPAIGN_BATCH - 1, seed=1, method="lhs")
    rows = D._campaign_dataset_rows(samples, size_vars, size_names, sizes, bias, nf,
                                    topo, binding, None, cfg.freqs, cfg.band,
                                    list(D.AC_NOISE_LABELS), 1)
    assert rows == {}                                       # fragmented -> scalar


def test_dataset_campaign_guard_periodic_labels(tmp_path):
    """A periodic label group forces the whole build onto the scalar path."""
    _require_rust()
    _freepdk45_ready()
    data = _size_grid_config(extra_labels=("transient",))
    path = _write(tmp_path, data)
    calls = {"n": 0}
    orig = D._campaign_dataset_rows

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    import circuitopt.dataset as dmod
    dmod._campaign_dataset_rows = spy
    try:
        D.run_from_config(path, n=10, seed=1, corner="nom",
                          label_groups=("ac_noise", "transient"))
    finally:
        dmod._campaign_dataset_rows = orig
    assert calls["n"] == 0                                  # guard kept it scalar


def test_dataset_campaign_guard_afe(tmp_path):
    """An AFE (default-PDK) dataset never routes to the campaign."""
    _require_rust()
    data = _size_grid_config(afe=True)
    path = _write(tmp_path, data)
    calls = {"n": 0}
    orig = D._campaign_dataset_rows

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    import circuitopt.dataset as dmod
    dmod._campaign_dataset_rows = spy
    try:
        D.run_from_config(path, n=10, seed=1)
    finally:
        dmod._campaign_dataset_rows = orig
    assert calls["n"] == 0                                  # AFE stays scalar


class _FrameTrap:
    def __init__(self, needles):
        self.needles, self.hits = needles, 0

    def __enter__(self):
        import sys
        sys.setprofile(self._hook)
        return self

    def __exit__(self, *exc):
        import sys
        sys.setprofile(None)

    def _hook(self, frame, event, _arg):
        if event == "call" and any(
                n in frame.f_globals.get("__name__", "") for n in self.needles):
            self.hits += 1


def test_dataset_campaign_layer_batch_zero_python_device_frame(tmp_path, monkeypatch):
    """The layer's solve batch makes no Python BSIM4/solver callback or frame.

    The campaign template build + the post-batch power/area reductions are outside
    the trap; only ``evaluate_batch`` (the detached solve) is wrapped."""
    _require_rust()
    _freepdk45_ready()
    from circuitopt._campaign_sweep import silicon_campaign_for
    from circuitopt.circuit_loader import models_from_config
    from circuitopt.device_factory import CircuitBinding
    from circuitopt.explore import sample
    data = _size_grid_config()
    _, topo, sizes, bias, nf, cfg = D.load_dataset_config(_write(tmp_path, data))
    mt, dk = models_from_config(data)
    binding = CircuitBinding(topo=topo, model_types=mt, device_kwargs=dk, nf=nf)
    size_vars, _, _ = D.split_variables(cfg.variables)
    size_names = {v.name for v in size_vars}
    samples = sample(cfg.variables, 10, seed=1, method="lhs")
    from circuitopt.explore import apply_variables
    grid = [apply_variables(size_vars, {k: v for k, v in vv.items() if k in size_names},
                            sizes, bias, base_nf=nf)[0] for vv in samples]
    camp = silicon_campaign_for(topo, grid[0], bias, nf, binding, cfg.freqs, cfg.band)
    assert camp is not None
    cands = [camp.candidate(g, corner=camp.nominal_corner) for g in grid]

    from circuitopt.compact_models.bsim4 import NativeBsim4Backend
    import circuitopt.ac_solver as acmod
    import circuitopt.noise_solver as nzmod

    def boom(*_a, **_k):
        raise AssertionError("python BSIM4/solver callback during layer batch")

    monkeypatch.setattr(NativeBsim4Backend, "evaluate", boom)
    monkeypatch.setattr(NativeBsim4Backend, "evaluate_batch", staticmethod(boom))
    monkeypatch.setattr(NativeBsim4Backend, "noise_batch", staticmethod(boom))
    monkeypatch.setattr(acmod, "ac_solve", boom)
    monkeypatch.setattr(nzmod, "noise_analysis", boom)
    with _FrameTrap(("compact_models.bsim4", "circuitopt.pdk",
                     "circuitopt.ac_solver", "circuitopt.noise_solver")) as trap:
        out = camp.evaluate_batch(cands, workers=4, analyses=("dc", "ac", "noise"))
    assert all(r["ok"] for r in out)
    assert trap.hits == 0, f"{trap.hits} Python PDK/device frames in the layer batch"
