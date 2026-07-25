import json
from pathlib import Path

import numpy as np
import pytest

from circuitopt.toolchain import pdk_root


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "freepdk45_sar3.json"
_HAVE = (Path(pdk_root()) / "freepdk45" / "models_nom" / "NMOS_VTG.inc").is_file()
pytestmark = pytest.mark.skipif(not _HAVE, reason="FreePDK45 cards not present")


def test_sar_example_matches_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schemas" / "circuit.schema.json").read_text())
    jsonschema.validate(json.loads(EXAMPLE.read_text()), schema)


def test_differential_sar_waveforms_stay_inside_rails():
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.sar import sar_input_waveforms, sar_time_grid
    spec = load_circuit_json(EXAMPLE)
    tgrid = sar_time_grid(spec)
    wave = sar_input_waveforms(spec, 0.9, [1, 0, None], 2, tgrid=tgrid)
    for value in wave.values():
        assert np.min(value) >= 0.0 and np.max(value) <= 1.0
    np.testing.assert_allclose(wave["sample"] + wave["sample_b"], 1.0)


def test_sar_physical_comparator_conversion(monkeypatch):
    import circuitopt.sar as sar
    from circuitopt.circuit_loader import load_circuit_json

    spec = load_circuit_json(EXAMPLE)
    calls = 0
    real_transient = sar.transient

    def counting_transient(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_transient(*args, **kwargs)

    monkeypatch.setattr(sar, "transient", counting_transient)
    result = sar.run_sar_conversion(spec, 0.7)
    assert result["code"] == 5
    np.testing.assert_array_equal(result["bits"], [1, 0, 1])
    assert result["decision_backend"] == "rust_continuation"
    assert calls == 1
    assert result["transient"]["backend"] == "bsim4_native"
    assert len(result["decisions"]) == 3
    assert result["supply_power"]["total_w"] > 0.0
    assert np.isfinite(result["total_power_w"])


def test_sar_continuation_matches_frozen_replay():
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.sar import (
        _run_sar_conversion_reference,
        run_sar_conversion,
    )

    spec = load_circuit_json(EXAMPLE)
    expected = _run_sar_conversion_reference(spec, 0.7)
    actual = run_sar_conversion(spec, 0.7)

    assert actual["code"] == expected["code"]
    np.testing.assert_array_equal(actual["bits"], expected["bits"])
    np.testing.assert_array_equal(
        actual["transient"]["output"],
        expected["transient"]["output"],
    )
    np.testing.assert_array_equal(
        [item["comparator_v"] for item in actual["decisions"]],
        [item["comparator_v"] for item in expected["decisions"]],
    )
    assert actual["total_power_w"] == expected["total_power_w"]


def test_sar_code_center_sweep_has_every_code(monkeypatch):
    import circuitopt.sar as sar
    from circuitopt.circuit_loader import load_circuit_json

    spec = load_circuit_json(EXAMPLE)
    vin = (np.arange(8) + 0.5) / 8.0
    calls = 0
    real_transient = sar.transient

    def counting_transient(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_transient(*args, **kwargs)

    monkeypatch.setattr(sar, "transient", counting_transient)
    result = sar.run_sar_sweep(spec, vin)
    np.testing.assert_array_equal(result["codes"], np.arange(8))
    assert len(result["metrics"]["missing_codes"]) == 0
    assert calls == len(vin)
    assert all(
        item["decision_backend"] == "rust_continuation"
        for item in result["conversions"]
    )
