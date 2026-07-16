"""Threaded SAR conversions must match the serial path bit-for-bit.

Skip-guarded exactly like ``test_sar.py`` (real ngspice oracle + FreePDK45 cards).
The parallel path only distributes *independent whole conversions* across a thread
pool; these tests pin that it is order-preserving and byte-identical to serial for
both the static sweep and the mismatch Monte-Carlo, and that the mismatch draws stay
seed-deterministic regardless of worker count.
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


def test_sweep_workers_match_serial():
    """A 4-point sweep with workers=2 equals the serial codes and static metrics."""
    from circuitopt.sar import run_sar_sweep
    spec = _spec()
    vin = (np.arange(4) + 0.5) / 4.0
    serial = run_sar_sweep(spec, vin, workers=1)
    parallel = run_sar_sweep(spec, vin, workers=2)
    np.testing.assert_array_equal(serial["codes"], parallel["codes"])
    assert serial["metrics"]["max_abs_dnl"] == parallel["metrics"]["max_abs_dnl"]
    assert serial["metrics"]["max_abs_inl"] == parallel["metrics"]["max_abs_inl"]


def test_sweep_rejects_bad_worker_count():
    from circuitopt.sar import run_sar_sweep
    spec = _spec()
    vin = (np.arange(2) + 0.5) / 2.0
    with pytest.raises(ValueError):
        run_sar_sweep(spec, vin, workers=0)


def test_mismatch_mc_workers_reproduce_serial_codes():
    """Same seed -> identical per-trial codes for workers=1 and workers=2."""
    from circuitopt.sar_mc import sar_mismatch_mc
    spec = _spec()
    cfg = {"sigma_vth0": 0.01, "sigma_cu": 0.02}
    serial = sar_mismatch_mc(spec, n=2, workers=1, seed=7, config=cfg)
    parallel = sar_mismatch_mc(spec, n=2, workers=2, seed=7, config=cfg)
    assert [r["trial"] for r in parallel["rows"]] == [0, 1]   # final order by trial idx
    for a, b in zip(serial["rows"], parallel["rows"]):
        np.testing.assert_array_equal(a["codes"], b["codes"])
    assert serial["summary"]["yield"] == parallel["summary"]["yield"]


def test_mismatch_mc_progress_is_monotonic_under_workers():
    """The progress callback fires with a strictly increasing completed count."""
    from circuitopt.sar_mc import sar_mismatch_mc
    spec = _spec()
    seen = []
    sar_mismatch_mc(spec, n=3, workers=2, seed=1, config={"sigma_vth0": 0.01},
                    progress=lambda i, n, partial: seen.append((i, n, partial["n"])))
    counts = [i for i, _n, _p in seen]
    assert counts == [1, 2, 3]
    assert all(total == 3 for _i, total, _p in seen)
    assert [p for _i, _n, p in seen] == [1, 2, 3]     # summary aggregates finished trials
