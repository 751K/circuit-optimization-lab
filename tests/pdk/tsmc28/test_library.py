"""Native, no-ngspice TSMC28 core-model library elaboration."""
from __future__ import annotations

import os

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


def test_invalid_requests_fail_loudly():
    library = load_tsmc28_core_library(_PATH)
    with pytest.raises(Tsmc28ModelError, match="corner"):
        library.core_card("nmos", width_um=1, length_um=0.03, corner="bad")
    with pytest.raises(Tsmc28ModelError, match="positive"):
        library.core_card("nmos", width_um=0, length_um=0.03)
    with pytest.raises(Tsmc28ModelError, match="bins"):
        library.core_card("nmos", width_um=1e9, length_um=0.03)
