"""Compiled SAR conversion batch — bit-exact parity + worker determinism.

heavy_e2e: real FreePDK45 cards and full closed-loop conversions. The compiled
batch (:mod:`circuitopt.sar_rust`, backed by ``co_core::sar``) must reproduce the
frozen :func:`circuitopt.sar.run_sar_conversion` codes *bit-for-bit* — bit
decisions are discrete, so any root shift would flip a code — and be
byte-identical across worker counts. :func:`circuitopt.sar_mc.sar_mismatch_mc`
routes through it with a reference fallback, so its own regressions
(``test_sar_mc``/``test_sar_parallel``) also exercise this path.
"""
from pathlib import Path

import numpy as np
import pytest

from circuitopt.toolchain import pdk_root


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "freepdk45_sar3.json"
_HAVE = (Path(pdk_root()) / "freepdk45" / "models_nom" / "NMOS_VTG.inc").is_file()
pytestmark = pytest.mark.skipif(not _HAVE, reason="FreePDK45 cards not present")


def _spec():
    from circuitopt.circuit_loader import load_circuit_json
    return load_circuit_json(EXAMPLE)


def _draws(spec, over, n, seed):
    from circuitopt.sar import _sar_config
    from circuitopt.sar_mc import (_mismatch_config, draw_device_mismatch,
                                   perturb_capacitors)
    cfg = _sar_config(spec, over)
    mcfg = _mismatch_config(spec, over)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        delvto = draw_device_mismatch(spec, rng, mcfg)
        trial_spec = perturb_capacitors(spec, rng, mcfg)
        draws.append((delvto, trial_spec))
    levels = 1 << cfg["n_bits"]
    vin = (np.arange(levels) + 0.5) / levels * cfg["vref"]
    return cfg, draws, vin


def _trials(draws):
    return [(delvto, [cap[3] for cap in trial_spec.topology.capacitors])
            for delvto, trial_spec in draws]


def test_compiled_batch_matches_reference_codes_bit_for_bit():
    """Each trial's compiled codes equal the frozen run_sar_conversion codes.

    A large Vth/cap sigma forces bit flips and non-monotonic sweeps, so this is a
    discrete-decision (non-tolerance) parity check, not a numeric one.
    """
    from circuitopt.sar import run_sar_conversion
    from circuitopt.sar_rust import build_sar_batch
    spec = _spec()
    cfg, draws, vin = _draws(spec, {"sigma_vth0": 0.05, "sigma_cu": 0.05}, 3, 3)
    reference = [
        np.array([run_sar_conversion(trial_spec, float(v), config=cfg,
                                     mismatch=delvto)["code"] for v in vin],
                 dtype=np.int64)
        for delvto, trial_spec in draws
    ]
    batch = build_sar_batch(spec, cfg)
    got = batch.run(_trials(draws), workers=1)
    assert len(got) == len(reference)
    for expected, actual in zip(reference, got):
        np.testing.assert_array_equal(expected, actual)


def test_compiled_batch_is_worker_count_invariant():
    """workers 1/2/8 return byte-identical codes (index-ordered write-back)."""
    from circuitopt.sar_rust import build_sar_batch
    spec = _spec()
    cfg, draws, _ = _draws(spec, {"sigma_vth0": 0.04, "sigma_cu": 0.04}, 8, 11)
    batch = build_sar_batch(spec, cfg)
    trials = _trials(draws)
    baseline = batch.run(trials, workers=1)
    for workers in (2, 8):
        other = batch.run(trials, workers=workers)
        assert len(other) == len(baseline)
        for expected, actual in zip(baseline, other):
            np.testing.assert_array_equal(expected, actual)


def test_zero_sigma_compiled_batch_is_nominal():
    """An all-zero-sigma batch reproduces the nominal ramp (every code present)."""
    from circuitopt.sar_rust import build_sar_batch
    spec = _spec()
    cfg, draws, _ = _draws(spec, {}, 2, 0)  # sigmas default to 0.0
    batch = build_sar_batch(spec, cfg)
    for codes in batch.run(_trials(draws), workers=1):
        np.testing.assert_array_equal(codes, np.arange(8))


def test_mismatch_mc_takes_the_compiled_path():
    """sar_mismatch_mc resolves a compiled batch for the native FreePDK45 SAR."""
    from circuitopt.sar import _sar_config
    from circuitopt.sar_rust import build_sar_batch
    spec = _spec()
    cfg = _sar_config(spec, {"sigma_vth0": 0.01})
    # build_sar_batch succeeds (does not raise SarRustUnavailable) for this spec,
    # so sar_mismatch_mc's fast path is taken rather than the reference fallback.
    batch = build_sar_batch(spec, cfg)
    assert batch.levels == 8


def test_mismatch_mc_never_calls_python_per_conversion(monkeypatch):
    """The compiled batch does the whole MC in Rust — zero per-bit Python callback.

    The reference path calls :func:`run_sar_conversion` once per conversion (per
    bit inside it). If the compiled fast path is engaged, that Python function is
    never entered during ``sar_mismatch_mc`` — the entire trial batch runs under
    one ``py.detach``. Counting the calls is the "counting trap" that catches a
    silent fallback to the GIL-bound Python loop.
    """
    import circuitopt.sar_mc as sar_mc
    calls = {"n": 0}
    real = sar_mc.run_sar_conversion

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(sar_mc, "run_sar_conversion", counting)
    result = sar_mc.sar_mismatch_mc(_spec(), n=3, seed=2,
                                    config={"sigma_vth0": 0.02, "sigma_cu": 0.02})
    assert len(result["rows"]) == 3
    assert calls["n"] == 0          # compiled path: no Python conversion callback
