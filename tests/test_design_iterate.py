"""Contracts for the design-iteration driver (tools/design_iterate.py).

The campaign itself needs licensed models, so these tests cover the parts that
carry the tool's promises without running one: the override contract (a typo
must fail loudly, never evaluate the unmodified design) and the staging
contract (overrides reach the generated decks, the repository is untouched).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "design_iterate", ROOT / "tools" / "design_iterate.py")
design_iterate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(design_iterate)


def _generator():
    """A minimal stand-in with the three override surfaces and a deck."""
    module = types.SimpleNamespace()
    module.CC = 900e-15
    module.SZ = {"M1": (260.0, 0.18), "M9": (150.0, 0.20)}
    module.MULT = {"M0": 3}
    module.all_testbenches = lambda: {
        "deck.json": {
            "name": "deck",
            "capacitors": [{"name": "CC1", "a": "A", "b": "B", "C": module.CC}],
            "devices": [{"name": "M1", "W": module.SZ["M1"][0],
                         "L": module.SZ["M1"][1], "M": module.MULT["M0"]}],
        }
    }
    return module


# ── override contract ───────────────────────────────────────────────────────────
def test_overrides_reach_constants_sizes_and_multiplicity():
    gen = _generator()
    applied = design_iterate.apply_overrides(
        gen, ["CC=850e-15", "SZ:M9=175/0.25", "MULT:M0=2"])
    assert gen.CC == pytest.approx(850e-15)
    assert gen.SZ["M9"] == pytest.approx((175.0, 0.25))
    assert gen.MULT["M0"] == 2
    # The echo is what appears in the printout, so it must name every override.
    assert any("CC=850e-15" in item for item in applied)
    assert any("SZ[M9]" in item for item in applied)
    assert any("MULT[M0]" in item for item in applied)


@pytest.mark.parametrize("override, needle", [
    ("CCC=1e-12", "constant"),        # typo in a module constant
    ("SZ:M42=10/0.2", "SZ entry"),    # device not in the sizing table
    ("CC", "NAME=VALUE"),             # missing '='
])
def test_unknown_override_target_is_rejected(override, needle):
    """A typo must abort, not silently evaluate the unmodified design.

    Silently ignoring an override is the worst possible failure for this tool:
    every number it prints would describe a design the user did not ask for."""
    gen = _generator()
    with pytest.raises(SystemExit) as excinfo:
        design_iterate.apply_overrides(gen, [override])
    assert needle in str(excinfo.value)
    assert gen.CC == pytest.approx(900e-15)      # untouched


def test_missing_mult_table_is_rejected():
    gen = types.SimpleNamespace(CC=1.0)
    with pytest.raises(SystemExit):
        design_iterate.apply_overrides(gen, ["MULT:M0=2"])


# ── staging contract ────────────────────────────────────────────────────────────
def test_staging_writes_overridden_decks_and_leaves_the_manifest_alone(tmp_path):
    gen = _generator()
    design_iterate.apply_overrides(gen, ["CC=123e-15"])
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"name": "c", "cases": []}), encoding="utf-8")

    staging, staged = design_iterate.staged_manifest(gen, manifest)
    try:
        assert staged.parent == staging
        assert staging != manifest.parent            # repo untouched
        deck = json.loads((staging / "deck.json").read_text())
        assert deck["capacitors"][0]["C"] == pytest.approx(123e-15)
        assert json.loads(staged.read_text())["name"] == "c"
    finally:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
    assert json.loads(manifest.read_text()) == {"name": "c", "cases": []}


# ── point parsing ───────────────────────────────────────────────────────────────
def test_point_parsing_round_trips_and_rejects_malformed():
    assert design_iterate.parse_point("ff/-40/0.85") == ("ff", -40.0, 0.85)
    with pytest.raises(SystemExit):
        design_iterate.parse_point("ff/27")


def test_point_label_matches_the_parse_format():
    label = design_iterate.point_label(
        {"corner": "ss", "temperature_c": 125.0, "supply_v": 0.85})
    assert label == "ss/125/0.85"
    assert design_iterate.parse_point(label) == ("ss", 125.0, 0.85)


def test_cli_requires_a_generator_that_can_drive_a_campaign(monkeypatch):
    module = types.ModuleType("fake_gen_without_decks")
    monkeypatch.setitem(sys.modules, "fake_gen_without_decks", module)
    with pytest.raises(SystemExit) as excinfo:
        design_iterate.main([
            "run", "--generator", "fake_gen_without_decks", "--manifest", "x.json"])
    assert "all_testbenches" in str(excinfo.value)
