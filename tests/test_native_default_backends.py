"""Normal examples and workflows must stay on in-process native backends."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from circuitopt.device_model import get_model_class
from circuitopt.pdk.sky130 import sky130_card_path


ROOT = Path(__file__).resolve().parents[1]


def test_normal_examples_do_not_bind_oracle_models():
    forbidden = ("_ngspice.", "sky130_osdi.")
    violations = []
    for path in sorted((ROOT / "examples").glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        for name, binding in (config.get("models") or {}).items():
            model_type = str(binding.get("type", ""))
            if any(token in model_type for token in forbidden):
                violations.append(f"{path.name}:{name}={model_type}")
    assert not violations, "normal examples bind oracle models: " + ", ".join(violations)


def test_silicon_default_model_classes_use_native_bsim4():
    for model_type in (
        "sky130.nmos",
        "sky130.pmos",
        "freepdk45.nmos",
        "freepdk45.pmos",
        "tsmc28hpcp.nmos",
        "tsmc28hpcp.pmos",
    ):
        model_class = get_model_class(model_type)
        assert model_class is not None
        assert model_class.TRANSIENT_BACKEND == "bsim4_native"


def test_fresh_package_import_registers_only_native_pdks():
    script = """
import json
import circuitopt
print(json.dumps({
    "pdks": circuitopt.list_pdks(),
    "models": circuitopt.registered_models(),
    "top_level_ngspice": hasattr(circuitopt, "ac_ngspice"),
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    state = json.loads(proc.stdout)
    assert state["pdks"] == ["at4000tg", "freepdk45", "sky130", "tsmc28hpcp"]
    assert not any(
        "_ngspice." in name or name.startswith("sky130_osdi.")
        for name in state["models"]
    )
    assert state["top_level_ngspice"] is False


def test_sky130_cards_are_packaged_with_the_adapter():
    path = sky130_card_path("nmos", 1.0, 0.15, "tt")
    assert path.is_file()
    assert path.parent.name == "cards"
    assert path.parent.parent.name == "sky130"


def test_ngspice_experiments_are_explicit_oracles_or_comparisons():
    violations = []
    for path in sorted((ROOT / "experiments").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "circuitopt.ngspice_" not in source:
            continue
        if not any(token in path.stem for token in ("oracle", "compare")):
            violations.append(path.name)
    assert not violations, "implicit ngspice experiments: " + ", ".join(violations)
