"""Per-instance mismatch Monte-Carlo for the FreePDK45 / ngspice SAR ADC path.

Skip-guarded exactly like ``test_sar.py``: these exercise the real ngspice
oracle, so they only run when the FreePDK45 cards and the ngspice binary are
present. The 3-bit example converts fast, so the MC trials here are kept tiny.
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


def test_delvto_targets_only_named_device():
    """The delvto hook stamps the offset on exactly the targeted M-line."""
    from circuitopt.device_factory import resolve_binding
    from circuitopt.ngspice_transient import render_freepdk45_transient_netlist
    from circuitopt.sar import _sar_config, sar_input_waveforms, sar_time_grid
    spec = _spec()
    cfg = _sar_config(spec)
    tgrid = sar_time_grid(spec, cfg)
    waveforms = sar_input_waveforms(spec, 0.5, [None, None, None], 0,
                                    config=cfg, tgrid=tgrid)
    topo, nf, corner, model_types, device_kwargs, _ = resolve_binding(
        spec.binding().at_corner(None))
    kw = dict(topo=topo, output_path="/tmp/unused.dat", nf=nf, inputs=waveforms,
              corner=corner, model_types=model_types, device_kwargs=device_kwargs,
              integration_method="gear2", max_step=cfg["edge_time"])
    nominal = render_freepdk45_transient_netlist(spec.sizes, spec.bias, tgrid, **kw)
    shifted = render_freepdk45_transient_netlist(
        spec.sizes, spec.bias, tgrid, mismatch={"M1": 0.3}, **kw)
    assert "delvto" not in nominal.netlist
    assert shifted.netlist.count("delvto") == 1
    m1_line = next(ln for ln in shifted.netlist.splitlines() if ln.startswith("M1 "))
    assert "delvto" in m1_line
    # An unknown device name is rejected rather than silently dropped.
    with pytest.raises(ValueError):
        render_freepdk45_transient_netlist(
            spec.sizes, spec.bias, tgrid, mismatch={"NOPE": 0.1}, **kw)


def test_delvto_flips_a_bit_decision():
    """A large comparator-input Vth offset flips at least one SAR bit."""
    from circuitopt.sar import run_sar_conversion
    spec = _spec()
    nominal = run_sar_conversion(spec, 0.5)
    shifted = run_sar_conversion(spec, 0.5, mismatch={"M1": 0.3})
    assert not np.array_equal(nominal["bits"], shifted["bits"])


def test_zero_sigma_mc_reproduces_nominal_codes():
    """An all-zero-sigma MC leaves every code-center conversion at its nominal code."""
    from circuitopt.sar_mc import sar_mismatch_mc
    spec = _spec()
    result = sar_mismatch_mc(spec, n=2, seed=0)  # sigmas default to 0.0
    for row in result["rows"]:
        np.testing.assert_array_equal(row["codes"], np.arange(8))
    assert result["summary"]["yield"] == 1.0
    assert result["summary"]["missing_codes"]["worst"] == 0.0


def test_small_sigma_mc_returns_finite_stats_and_yield():
    """A small nonzero-sigma MC yields finite linearity stats and a fractional yield."""
    from circuitopt.sar_mc import sar_mismatch_mc
    spec = _spec()
    result = sar_mismatch_mc(spec, n=3, seed=1,
                             config={"sigma_vth0": 0.01, "sigma_cu": 0.02})
    summary = result["summary"]
    assert summary["n"] == 3
    assert np.isfinite(summary["max_abs_dnl"]["mean"])
    assert np.isfinite(summary["max_abs_inl"]["mean"])
    assert 0.0 <= summary["yield"] <= 1.0
    assert len(result["rows"]) == 3
    # progress callback fires once per trial with a running summary.
    seen = []
    sar_mismatch_mc(spec, n=2, seed=1, config={"sigma_vth0": 0.01},
                    progress=lambda i, n, partial: seen.append((i, n, partial["n"])))
    assert seen == [(1, 2, 1), (2, 2, 2)]
