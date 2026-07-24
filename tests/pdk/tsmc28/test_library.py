"""Native, no-ngspice TSMC28 core-model library elaboration."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from circuitopt.pdk.tsmc28 import (
    TSMC28_CORE_CORNERS,
    Tsmc28ModelError,
    load_tsmc28_core_library,
)
from circuitopt.toolchain import tsmc28_model_dir


_PATH = os.path.join(
    tsmc28_model_dir(), "cln28hpcp_1d8_elk_v1d0_2p2.l")
pytestmark = pytest.mark.skipif(
    not os.path.isfile(_PATH),
    reason="licensed TSMC28HPC+ model is not installed",
)


@pytest.mark.parametrize("corner", TSMC28_CORE_CORNERS)
@pytest.mark.parametrize("polarity", ("nmos", "pmos"))
def test_all_core_corners_flatten_without_ngspice(monkeypatch, corner, polarity):
    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/ngspice")
    card = load_tsmc28_core_library(_PATH).core_card(
        polarity,
        width_um=1.0,
        length_um=0.03,
        corner=corner,
        temperature_c=27,
    )
    assert card.corner == corner
    assert card.model_type == polarity
    assert card.model_parameters["level"] == 54
    assert card.model_parameters["version"] == pytest.approx(4.5)
    assert len(card.model_parameters) > 300
    assert card.width_m == pytest.approx(1e-6)
    assert card.length_m == pytest.approx(30e-9)
    model, instance = card.to_bsim4_cards()
    assert model.polarity == (1 if polarity == "nmos" else -1)
    assert "level" not in model.parameters
    assert "mulu0" not in instance.parameters


@pytest.mark.parametrize(
    ("width_um", "length_um", "nf"),
    [
        (0.1, 0.03, 1),
        (1.0, 0.03, 4),
        (10.0, 0.10, 10),
        (300.0, 0.40, 200),
    ],
)
def test_representative_ota_geometries_select_one_bin(width_um, length_um, nf):
    card = load_tsmc28_core_library(_PATH).core_card(
        "nmos",
        width_um=width_um,
        length_um=length_um,
        nf=nf,
    )
    assert card.bin_name.startswith("nch.")
    assert card.instance_parameters["nf"] == nf


def test_temperature_mismatch_and_multiplicity_reach_numeric_card():
    library = load_tsmc28_core_library(_PATH)
    cold = library.core_card(
        "pmos",
        width_um=2.0,
        length_um=0.04,
        temperature_c=-40,
        mismatch_v=0.012,
        mult=3,
    )
    hot = library.core_card(
        "pmos",
        width_um=2.0,
        length_um=0.04,
        temperature_c=125,
        mismatch_v=0.012,
        mult=3,
    )
    assert cold.instance_parameters["delvto"] == pytest.approx(0.012)
    assert cold.instance_parameters["m"] == pytest.approx(3)
    assert cold.temperature_c == -40
    assert hot.temperature_c == 125
    assert cold.model_parameters == hot.model_parameters


def test_card_cache_reuses_immutable_core_model_and_instance_cards():
    library = load_tsmc28_core_library(_PATH)
    library.clear_card_cache()
    request = {
        "pdk": "tsmc28hpcp",
        "model": "nmos",
        "section": "inherit",
        "bin_selector": "auto",
        "width_um": 1.0,
        "length_um": 0.03,
        "nf": 2,
        "mult": 1,
        "corner": "tt",
        "temperature_c": 27.0,
        "mismatch_v": 0.0,
    }

    first = library.device_cards("nmos", **request)
    second = library.device_cards("nmos", **request)
    assert all(left is right for left, right in zip(first, second, strict=True))
    info = library.card_cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (
        1, 1, 1024, 1)
    with pytest.raises(TypeError):
        first[0].model_parameters["vth0"] = 1.0
    with pytest.raises(TypeError):
        first[1].parameters["vth0"] = 1.0
    with pytest.raises(TypeError):
        first[2].parameters["nf"] = 4


def test_card_cache_key_separates_binding_geometry_pvt_and_mismatch():
    library = load_tsmc28_core_library(_PATH)
    library.clear_card_cache()
    base = {
        "pdk": "tsmc28hpcp",
        "model": "pmos",
        "section": "inherit",
        "bin_selector": "auto",
        "width_um": 2.0,
        "length_um": 0.04,
        "nf": 1,
        "mult": 1,
        "corner": "tt",
        "temperature_c": 27.0,
        "mismatch_v": 0.0,
    }
    variants = (
        {},
        {"section": "tt"},
        {"width_um": 2.1},
        {"length_um": 0.05},
        {"nf": 2},
        {"mult": 2},
        {"temperature_c": 125.0},
        {"corner": "ss"},
        {"mismatch_v": 0.01},
    )
    bundles = [
        library.device_cards("pmos", **(base | override))
        for override in variants
    ]
    assert len({id(bundle[0]) for bundle in bundles}) == len(variants)
    assert library.card_cache_info().currsize == len(variants)

    resolved_bin = bundles[0][0].bin_name
    explicit = library.device_cards(
        "pmos", **(base | {"bin_selector": resolved_bin}))
    assert explicit[0].bin_name == resolved_bin
    assert explicit[0] is not bundles[0][0]
    assert library.card_cache_info().currsize == len(variants) + 1


def test_card_cache_same_key_is_identity_stable_across_threads():
    library = load_tsmc28_core_library(_PATH)
    library.clear_card_cache()

    def load():
        return library.device_cards(
            "nmos",
            pdk="tsmc28hpcp",
            model="nmos",
            section="tt",
            bin_selector="auto",
            width_um=3.0,
            length_um=0.06,
            nf=4,
            corner="tt",
            temperature_c=27.0,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        bundles = list(pool.map(lambda _: load(), range(16)))
    assert len({id(bundle[0]) for bundle in bundles}) == 1
    assert len({id(bundle[1]) for bundle in bundles}) == 1
    assert len({id(bundle[2]) for bundle in bundles}) == 1
    assert library.card_cache_info().currsize == 1


def test_invalid_requests_fail_loudly():
    library = load_tsmc28_core_library(_PATH)
    with pytest.raises(Tsmc28ModelError, match="corner"):
        library.core_card("nmos", width_um=1, length_um=0.03, corner="bad")
    with pytest.raises(Tsmc28ModelError, match="positive"):
        library.core_card("nmos", width_um=0, length_um=0.03)
    with pytest.raises(Tsmc28ModelError, match="bins"):
        library.core_card("nmos", width_um=1e9, length_um=0.03)
    with pytest.raises(Tsmc28ModelError, match="PDK"):
        library.core_card(
            "nmos", pdk="freepdk45", width_um=1, length_um=0.03)
    with pytest.raises(Tsmc28ModelError, match="model"):
        library.core_card(
            "nmos", model="pmos", width_um=1, length_um=0.03)
    with pytest.raises(Tsmc28ModelError, match="section"):
        library.core_card(
            "nmos", section="ss", corner="tt", width_um=1, length_um=0.03)
    with pytest.raises(Tsmc28ModelError, match="bin"):
        library.core_card(
            "nmos", bin_selector="not_a_bin", width_um=1, length_um=0.03)
