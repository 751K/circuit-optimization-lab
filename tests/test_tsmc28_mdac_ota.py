"""Fast structural guards for the TSMC28HPC+ MDAC OTA deliverables.

Foundry-model AC/noise/transient and the 45-point PVT campaign intentionally run
outside CI.  These tests keep the generated, portable netlists and their key ADC
contracts from drifting while requiring no licensed model payload.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
TB_FILES = [
    "tsmc28hpcp_mdac_ota.json",
    "tsmc28hpcp_mdac_ota_ac.json",
    "tsmc28hpcp_mdac_ota_dmloop.json",
    "tsmc28hpcp_mdac_ota_boostn.json",
    "tsmc28hpcp_mdac_ota_boostp.json",
    "tsmc28hpcp_mdac_ota_cmfb1.json",
    "tsmc28hpcp_mdac_ota_cmfb2.json",
    "tsmc28hpcp_mdac_ota_noise.json",
    "tsmc28hpcp_mdac_ota_code_transition.json",
]


def _generator():
    sys.path.insert(0, str(EXAMPLES))
    import tsmc28_mdac_ota_gen
    return tsmc28_mdac_ota_gen


def test_generated_tsmc28_decks_match_source_and_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schemas" / "circuit.schema.json").read_text())
    generated = _generator().all_testbenches()
    assert set(generated) == set(TB_FILES)
    for filename, expected in generated.items():
        actual = json.loads((EXAMPLES / filename).read_text())
        assert actual == expected, f"{filename} is stale; regenerate the TSMC28 decks"
        jsonschema.validate(actual, schema)


def test_dut_uses_one_reference_current_and_no_ideal_active_devices():
    deck = _generator().build_transient()
    assert deck["current_sources"] == [["Iref", "VDD", "IB", 20e-6]]
    assert not deck.get("vcvs")
    assert not deck.get("vccs")
    assert not deck.get("cccs")
    assert not deck.get("ccvs")
    # 46 explicit instances collapsed to 38 (the 8 parallel clones M0B/M0C, M9B,
    # M10B, M11B/M11C, M12B/M12C now render as m= multiplicity); C2b adds the two
    # folded-cascode bottom-mirror sinks MFN1/MFN2 and 8 single-transistor
    # regulated-cascode boosters (4 CS devices + 4 current-source loads): 38+2+8=48.
    assert len(deck["devices"]) == 48
    assert {
        (model["pdk"], model["model"])
        for model in deck["models"].values()
    } == {
        ("tsmc28hpcp", "nmos"), ("tsmc28hpcp", "pmos"),
    }


def test_parallel_multiplicity_replaces_clone_instances():
    deck = _generator().build_transient()
    mult = {dev["name"]: int(dev.get("M", 1)) for dev in deck["devices"]}
    assert {name: m for name, m in mult.items() if m > 1} == {
        "M0": 3, "M9": 2, "M10": 2, "M11": 3, "M12": 3,
    }
    clone_names = {"M0B", "M0C", "M9B", "M10B", "M11B", "M11C", "M12B", "M12C"}
    assert not (set(mult) & clone_names)
    # Multiplied devices carry the generator's per-unit W (the JSON "W" is one
    # unit, never the M-folded total) and the generator's MULT map is the single
    # source of truth for every M value.  Design W/L values themselves are NOT
    # frozen here — sizing iterates (C1 resized M9/M10); the invariant is the
    # multiplicity MECHANISM, not the numbers.
    gen = _generator()
    per_unit = {dev["name"]: dev["W"] for dev in deck["devices"]}
    for name, m in gen.MULT.items():
        assert mult[name] == m
        assert per_unit[name] == pytest.approx(gen.SZ[name][0])
    # C2b: CORE_SAT_DEVICES also carries the folded-cascode bottom-mirror sinks
    # MFN1/MFN2, the Iref diode MBN, and every gain-boost aux transistor, so the
    # campaign region-checks them at every PVT corner.
    assert list(gen.CORE_SAT_DEVICES) == [
        "M0", *[f"M{i}" for i in range(1, 13)],
        *(name for name, *_ in gen.FOLD_DEVICES),
        "MBN", *(name for name, *_ in gen.BOOST_DEVICES)]


def test_gain_boost_aux_amps_and_loop_probes():
    generator = _generator()
    deck = generator.build_transient()
    devices = {dev["name"]: dev for dev in deck["devices"]}
    # Every booster transistor is present and named MB*, and appears in the
    # saturation-checked device set.  C2b: single-transistor regulated-cascode
    # boosters (a CS device + a current-source load per side), replacing the C2
    # quad DDA that could not DC-couple at 0.85 V.
    boost = [name for name, *_ in generator.BOOST_DEVICES]
    assert set(boost) == {
        "MBN1", "MBN2", "MBN1L", "MBN2L",
        "MBP1", "MBP2", "MBP1L", "MBP2L"}
    for name in boost:
        assert name in devices
        assert name.startswith("MB")
        assert name in generator.CORE_SAT_DEVICES
    # Each booster's CS device is source-at-a-rail so its output device stays
    # saturated: PMOS-from-VDD for the NMOS cascodes, NMOS-from-GND for the PMOS.
    assert devices["MBN1"]["source"] == "VDD" and devices["MBN1"]["gate"] == "NN1"
    assert devices["MBP1"]["source"] == "GND" and devices["MBP1"]["gate"] == "A1"
    # The stage-1 cascode gates are regulated by the booster drains, not the fixed
    # VBNC/VBPC bias any more.
    assert devices["M3"]["gate"] == "GBN1" and devices["M4"]["gate"] == "GBN2"
    assert devices["M5"]["gate"] == "GBP1" and devices["M6"]["gate"] == "GBP2"
    # No new ideal sources sneak in with the boost (still one 20 uA reference).
    assert deck["current_sources"] == [["Iref", "VDD", "IB", 20e-6]]

    # Each boost-loop testbench breaks the +side cascode gate with Vinj and mirrors
    # the -side with a unity VCVS whose control pair equals the break pair, so
    # loop_gain_tian_ngspice auto-detects the differential mirror.
    for which, (dp, dn, gp, gn) in generator.BOOST_BREAK.items():
        deck = getattr(generator, f"build_{which}")()
        srcnames = {s[0] for s in deck["vsources"]}
        assert "Vinj" in srcnames and not (srcnames - {"Vinj", "VACP", "VACN"})
        vinj = next(s for s in deck["vsources"] if s[0] == "Vinj")
        assert vinj[1] == gp + "G" and vinj[2] == gp
        (emir,) = deck["vcvs"]
        assert emir["mu"] == 1.0 and {emir["cp"], emir["cn"]} == {gp, gp + "G"}
        assert emir["p"] == gn + "G" and emir["q"] == gn
        gates = {dev["name"]: dev["gate"] for dev in deck["devices"]}
        assert gates[dp] == gp + "G" and gates[dn] == gn + "G"


def test_cmfb1_probe_keeps_compensation_on_physical_node():
    deck = _generator().build_cmfb1()
    cmill = [cap for cap in deck["capacitors"] if cap["name"] == "CMILL1"]
    assert len(cmill) == 1 and {cmill[0]["a"], cmill[0]["b"]} == {"CTRL1", "CMS1"}
    vinj = [src for src in deck["vsources"] if src[0] == "Vinj"]
    assert vinj and {vinj[0][1], vinj[0][2]} == {"CMS1G", "CMS1"}


def test_mdac_capacitance_gain_and_output_load_contracts():
    generator = _generator()
    deck = generator.build_transient()
    caps = {cap["name"]: cap["C"] for cap in deck["capacitors"]}
    assert caps["CS1"] == pytest.approx(2.6e-12)
    assert caps["CS2"] == pytest.approx(2.6e-12)
    assert caps["CF1"] == pytest.approx(325e-15)
    assert caps["CF2"] == pytest.approx(325e-15)
    assert caps["CS1"] / caps["CF1"] == pytest.approx(8.0)
    assert caps["CL1"] == pytest.approx(500e-15)
    assert caps["CL2"] == pytest.approx(500e-15)

    split = generator.build_code_transition()
    for side in ("P", "N"):
        total = sum(cap["C"] for cap in split["capacitors"]
                    if cap["name"].startswith(f"CS{side}"))
        assert total == pytest.approx(2.6e-12)


def test_all_transistors_have_portable_tsmc_bindings_and_native_nf():
    deck = _generator().build_transient()
    models = deck["models"]
    for device in deck["devices"]:
        name = device["name"]
        assert models[name]["pdk"] == "tsmc28hpcp"
        assert models[name]["model"] in {"nmos", "pmos"}
        assert models[name]["section"] == "inherit"
        assert models[name]["bin"] == "auto"
        assert isinstance(device["NF"], int) and device["NF"] >= 1
    encoded = json.dumps(deck)
    assert "/Users/" not in encoded
    assert "cln28hpcp_1d8_elk" not in encoded


def test_code_transition_is_synchronous_complementary_0111_to_1000():
    generator = _generator()
    import numpy as np

    t = np.array([0.0, 10e-12, 20e-12, 30e-12])
    wave = generator.code_transition_inputs(t)
    assert wave["DCH"][0] > wave["DCH"][-1]
    assert wave["DCHB"][0] < wave["DCHB"][-1]
    assert wave["bpp0"][0] > wave["bpp0"][-1]
    assert wave["bpp1"][0] > wave["bpp1"][-1]
    assert wave["bpp2"][0] > wave["bpp2"][-1]
    assert wave["bpp3"][0] < wave["bpp3"][-1]
    for side in ("p", "n"):
        for bit in range(4):
            assert wave[f"bp{side}{bit}"][1] == wave[f"bp{side}{bit}"][0]
            assert wave[f"bp{side}{bit}"][2] == wave[f"bp{side}{bit}"][-1]
