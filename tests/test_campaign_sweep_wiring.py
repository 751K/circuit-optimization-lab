"""R5-D: the shared sweep-campaign dispatch (`circuitopt._campaign_sweep`).

Covers the wiring the design-space sweep / dataset paths go through:

* family detection (all-silicon model_types -> silicon; AFE OTFT -> afe_otft),
* the **zero Python PDK/device frame** gate — the wired batch is proved to make
  no Python device/backend/solver callback (a monkeypatch counting trap) *and*
  to execute no Python frame inside a PDK/device module (a `sys.setprofile`
  sampling trap), which is what lets `workers` scale under the released GIL,
* index-ordered determinism across worker counts {1, 2, 8}, and
* the fallback contract: no rust engine -> `make_sweep_campaign` returns None.

D12: no PDK value is read or written here; only callback/frame *counts* and
booleans are asserted.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CIRCUIT_ENGINE", "rust")

import numpy as np
import pytest

pytest.importorskip("circuitopt_core")

from circuitopt._campaign_sweep import (campaign_enabled, evaluate_sizes,
                                        make_sweep_campaign)
from circuitopt.circuit_loader import load_circuit_json

_SI_FREQS = np.logspace(3, 7, 25)
_SI_BAND = (1e3, 1e6)
_AFE_FREQS = np.logspace(-2, 4, 61)
_AFE_BAND = (0.05, 100.0)


def _require_rust():
    if not campaign_enabled():
        pytest.skip("sweep-campaign dispatch requires the rust device engine")


def _freepdk45_ready():
    from circuitopt.toolchain import pdk_root

    if not os.path.isfile(os.path.join(pdk_root(), "freepdk45", "models_nom",
                                       "NMOS_VTG.inc")):
        pytest.skip("FreePDK45 cards not present")


def _afe_campaign():
    spec = load_circuit_json("examples/afe_explore.json")
    camp = make_sweep_campaign(spec, _AFE_FREQS, _AFE_BAND)
    assert camp is not None
    return spec, camp


def _freepdk45_campaign():
    _freepdk45_ready()
    spec = load_circuit_json("examples/freepdk45_5t_ota.json")
    camp = make_sweep_campaign(spec, _SI_FREQS, _SI_BAND)
    assert camp is not None
    return spec, camp


def _perturbed(spec, k, rng):
    return [{n: (w * f, l) for n, (w, l), f in
             zip(spec.sizes.keys(), spec.sizes.values(), rng.uniform(0.85, 1.15, len(spec.sizes)))}
            for _ in range(k)]


# ---------------------------------------------------------------------------
# Family detection + fallback contract.
# ---------------------------------------------------------------------------

def test_make_sweep_campaign_detects_families():
    _require_rust()
    _, afe = _afe_campaign()
    assert afe.family == "afe_otft" and afe.nominal_corner is None and afe.needs_seed
    _, si = _freepdk45_campaign()
    assert si.family == "silicon_bsim4" and si.nominal_corner == "nom"
    assert si.needs_seed is False


def test_make_sweep_campaign_none_off_rust(monkeypatch):
    """No rust engine -> None, so callers keep the scalar reference path."""
    import circuitopt._campaign_sweep as cs

    monkeypatch.setattr(cs, "campaign_enabled", lambda: False)
    spec = load_circuit_json("examples/afe_explore.json")
    assert cs.make_sweep_campaign(spec, _AFE_FREQS, _AFE_BAND) is None


# ---------------------------------------------------------------------------
# Zero Python PDK/device frame gate.
# ---------------------------------------------------------------------------

class _FrameTrap:
    """Count Python-frame *entries* into any module whose name contains a needle
    (a sampling-style profiler over the detached batch)."""

    def __init__(self, needles):
        self.needles = needles
        self.hits = 0

    def __enter__(self):
        sys.setprofile(self._hook)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)

    def _hook(self, frame, event, _arg):
        if event != "call":
            return
        name = frame.f_globals.get("__name__", "")
        if any(needle in name for needle in self.needles):
            self.hits += 1


def test_wired_afe_batch_makes_no_python_device_frame(monkeypatch):
    _require_rust()
    _, camp = _afe_campaign()
    sizes = [dict(load_circuit_json("examples/afe_explore.json").sizes)] * 4

    from circuitopt.pmos_tft_model import PMOS_TFT
    import circuitopt.ac_solver as acmod
    import circuitopt.noise_solver as nzmod

    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("python device/solver callback during wired batch")

    monkeypatch.setattr(PMOS_TFT, "get_ss_params", boom)
    monkeypatch.setattr(acmod, "ac_solve", boom)
    monkeypatch.setattr(nzmod, "noise_analysis", boom)

    with _FrameTrap(("pmos_tft_model", "circuitopt.ac_solver",
                     "circuitopt.noise_solver")) as trap:
        out = evaluate_sizes(camp, sizes, workers=4)
    assert all(r["ok"] for r in out)
    assert calls["n"] == 0
    assert trap.hits == 0, f"{trap.hits} Python device/solver frames in the batch"


def test_wired_silicon_batch_makes_no_python_device_frame(monkeypatch):
    _require_rust()
    spec, camp = _freepdk45_campaign()
    sizes = [dict(spec.sizes)] * 4

    from circuitopt.compact_models.bsim4 import NativeBsim4Backend
    import circuitopt.ac_solver as acmod
    import circuitopt.noise_solver as nzmod

    def boom(*_a, **_k):
        raise AssertionError("python BSIM4 backend/solver during wired batch")

    monkeypatch.setattr(NativeBsim4Backend, "evaluate", boom)
    monkeypatch.setattr(NativeBsim4Backend, "evaluate_batch", staticmethod(boom))
    monkeypatch.setattr(NativeBsim4Backend, "noise_batch", staticmethod(boom))
    monkeypatch.setattr(acmod, "ac_solve", boom)
    monkeypatch.setattr(nzmod, "noise_analysis", boom)

    with _FrameTrap(("compact_models.bsim4", "circuitopt.pdk",
                     "circuitopt.ac_solver", "circuitopt.noise_solver")) as trap:
        out = evaluate_sizes(camp, sizes, workers=4)
    assert all(r["ok"] for r in out)
    assert trap.hits == 0, f"{trap.hits} Python PDK/device frames in the batch"


# ---------------------------------------------------------------------------
# Determinism across worker counts (index-ordered).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["afe", "silicon"])
def test_wired_sweep_deterministic_across_workers(family):
    _require_rust()
    if family == "afe":
        spec, camp = _afe_campaign()
    else:
        spec, camp = _freepdk45_campaign()
    rng = np.random.default_rng(5)
    sizes = _perturbed(spec, 12, rng)
    base = evaluate_sizes(camp, sizes, workers=1)
    for workers in (1, 2, 8):
        got = evaluate_sizes(camp, sizes, workers=workers)
        assert len(got) == len(base)
        for i, (a, b) in enumerate(zip(base, got)):
            assert a.get("ok") == b.get("ok"), (family, workers, i)
            if a.get("ok"):
                for key in ("gain_peak_dB", "bw_Hz", "irn_uV", "latch_dV"):
                    assert a[key] == b[key], f"{family} workers={workers} cand={i} {key}"
