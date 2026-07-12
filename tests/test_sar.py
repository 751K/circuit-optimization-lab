import json
from pathlib import Path

import numpy as np
import pytest

from circuitopt.ngspice_char import ngspice_binary
from circuitopt.toolchain import pdk_root


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "freepdk45_sar3.json"
_HAVE = (Path(pdk_root()) / "freepdk45" / "models_nom" / "NMOS_VTG.inc").is_file() \
    and ngspice_binary() is not None
pytestmark = pytest.mark.skipif(not _HAVE, reason="FreePDK45 cards / ngspice not present")


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


def test_sar_physical_comparator_conversion():
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.sar import run_sar_conversion
    spec = load_circuit_json(EXAMPLE)
    result = run_sar_conversion(spec, 0.7)
    assert result["code"] == 5
    np.testing.assert_array_equal(result["bits"], [1, 0, 1])
    assert result["transient"]["backend"] == "ngspice"
    assert len(result["decisions"]) == 3
    assert result["supply_power"]["total_w"] > 0.0
    assert np.isfinite(result["total_power_w"])


def test_sar_code_center_sweep_has_every_code():
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.sar import run_sar_sweep
    spec = load_circuit_json(EXAMPLE)
    vin = (np.arange(8) + 0.5) / 8.0
    result = run_sar_sweep(spec, vin)
    np.testing.assert_array_equal(result["codes"], np.arange(8))
    assert len(result["metrics"]["missing_codes"]) == 0
