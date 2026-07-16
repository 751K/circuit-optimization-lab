"""Adversarial semantic tests for the SAR mismatch-MC work package.

Written independently of the implementation (reviewer-side verification): these
pin down the *contract* — seed reproducibility, spec purity, config precedence,
validation, and error propagation — rather than re-testing the happy path.
Skip-guarded like ``test_sar.py``: they exercise the real ngspice oracle.
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


def test_same_seed_gives_identical_mc_results():
    """Same seed -> byte-identical draws and metrics across two independent runs."""
    from circuitopt.sar_mc import sar_mismatch_mc
    cfg = {"sigma_vth0": 0.02, "sigma_cu": 0.05}
    a = sar_mismatch_mc(_spec(), n=1, seed=7, config=cfg)
    b = sar_mismatch_mc(_spec(), n=1, seed=7, config=cfg)
    np.testing.assert_array_equal(a["rows"][0]["codes"], b["rows"][0]["codes"])
    for key in ("max_abs_dnl", "max_abs_inl", "offset_lsb"):
        np.testing.assert_array_equal(a["arrays"][key], b["arrays"][key])


def test_mc_never_mutates_the_loaded_spec():
    """Cap perturbation must act on a copy: the caller's spec stays value-identical."""
    from circuitopt.sar_mc import sar_mismatch_mc
    spec = _spec()
    before = [tuple(item) for item in spec.topology.capacitors]
    sar_mismatch_mc(spec, n=1, seed=3, config={"sigma_cu": 0.2})
    after = [tuple(item) for item in spec.topology.capacitors]
    assert before == after


def test_json_block_is_base_and_function_config_wins():
    """adc.mismatch JSON block seeds the config; the config argument overrides it."""
    from circuitopt.sar_mc import _mismatch_config
    spec = _spec()
    spec.adc["mismatch"] = {"sigma_vth0": 0.011, "dnl_threshold": 0.123}
    resolved = _mismatch_config(spec, None)
    assert resolved["sigma_vth0"] == 0.011
    assert resolved["dnl_threshold"] == 0.123
    resolved = _mismatch_config(spec, {"dnl_threshold": 0.456})
    assert resolved["sigma_vth0"] == 0.011          # JSON base survives
    assert resolved["dnl_threshold"] == 0.456       # explicit override wins
    # Per-polarity override defaults to the flat sigma unless given explicitly.
    assert resolved["sigma_vth0_nmos"] == 0.011
    resolved = _mismatch_config(spec, {"sigma_vth0_pmos": 0.02})
    assert resolved["sigma_vth0_pmos"] == 0.02
    assert resolved["sigma_vth0_nmos"] == 0.011


def test_invalid_mismatch_config_is_rejected():
    from circuitopt.sar_mc import _mismatch_config
    spec = _spec()
    with pytest.raises(ValueError):
        _mismatch_config(spec, {"sigma_vth0": -0.01})
    with pytest.raises(ValueError):
        _mismatch_config(spec, {"sigma_cu": -1.0})
    with pytest.raises(ValueError):
        _mismatch_config(spec, {"w0": 0.0})
    with pytest.raises(ValueError):
        _mismatch_config(spec, {"c_unit": -1e-15})


def test_unknown_mismatch_device_raises_before_simulation():
    from circuitopt.sar import run_sar_conversion
    with pytest.raises(ValueError, match="NOPE"):
        run_sar_conversion(_spec(), 0.5, mismatch={"NOPE": 0.1})


def test_negative_delvto_renders_with_sign():
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
    rendered = render_freepdk45_transient_netlist(
        spec.sizes, spec.bias, tgrid, topo=topo, output_path="/tmp/unused.dat",
        nf=nf, inputs=waveforms, corner=corner, model_types=model_types,
        device_kwargs=device_kwargs, integration_method="gear2",
        max_step=cfg["edge_time"], mismatch={"M2": -0.05})
    m2_line = next(ln for ln in rendered.netlist.splitlines() if ln.startswith("M2 "))
    assert "delvto=-0.05" in m2_line


def test_mismatch_threads_through_the_sweep_api():
    """A gross comparator offset must change sweep codes — i.e. run_sar_sweep really
    forwards ``mismatch`` into every conversion instead of silently dropping it."""
    from circuitopt.sar import run_sar_sweep
    spec = _spec()
    vin = np.array([0.3125, 0.5625])
    nominal = run_sar_sweep(spec, vin)
    shifted = run_sar_sweep(spec, vin, mismatch={"M1": 0.3})
    assert not np.array_equal(nominal["codes"], shifted["codes"])


def test_zero_sigma_passes_arbitrarily_tight_thresholds():
    """Code-center transitions of the nominal 3-bit design sit exactly on the ideal
    grid, so a zero-sigma trial must pass even near-zero DNL/INL yield limits —
    this catches an implementation that leaks noise into the nominal path."""
    from circuitopt.sar_mc import sar_mismatch_mc
    result = sar_mismatch_mc(_spec(), n=1, seed=0,
                             config={"dnl_threshold": 1e-9, "inl_threshold": 1e-9})
    assert result["summary"]["yield"] == 1.0
    assert result["summary"]["max_abs_dnl"]["worst"] <= 1e-9


def test_huge_sigma_stress_never_crashes():
    """A deliberately absurd sigma must degrade gracefully (inf DNL / yield loss /
    missing codes are all acceptable) — never raise from a non-monotonic ramp."""
    from circuitopt.sar_mc import sar_mismatch_mc
    result = sar_mismatch_mc(_spec(), n=1, seed=11,
                             config={"sigma_vth0": 0.4, "sigma_cu": 0.3})
    row = result["rows"][0]
    assert row["codes"].shape == (8,)
    assert result["summary"]["yield"] in (0.0, 1.0)
    assert np.isfinite(result["summary"]["monotonic_rate"])


def test_local_solver_transient_rejects_mismatch():
    """The delvto hook is ngspice-only: the OTFT/local path must refuse it loudly
    rather than silently ignore the offsets."""
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.transient_solver import transient
    spec = load_circuit_json(ROOT / "examples" / "single_stage.json")
    tgrid = np.linspace(0.0, 1e-6, 11)
    with pytest.raises(NotImplementedError):
        transient(spec.sizes, spec.bias, tgrid, binding=spec.binding(),
                  inputs={"vin": np.full(11, 25.0)}, mismatch={"MPU": 0.1})
