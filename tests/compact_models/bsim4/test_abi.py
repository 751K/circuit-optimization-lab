"""Simulator-neutral BSIM4 ABI invariants."""
from __future__ import annotations

import numpy as np
import pytest

from circuitopt.compact_models.bsim4 import (
    Bsim4Bias,
    Bsim4Evaluation,
    Bsim4InstanceCard,
    Bsim4ModelCard,
    Bsim4Noise,
    Bsim4ValidationError,
)


def test_cards_normalize_and_validate_parameters():
    model = Bsim4ModelCard(
        polarity=1,
        parameters={"LEVEL": 54, "VERSION": 4.5, "VTH0": 0.4},
    )
    instance = Bsim4InstanceCard(
        {"W": 1e-6, "L": 30e-9, "NF": 2, "M": 1})
    assert model.parameters == {"vth0": 0.4}
    assert instance.parameters["nf"] == 2
    assert Bsim4Bias(0.9, 0.6, 0.0, 0.0).terminals.shape == (4,)
    compatible = Bsim4ModelCard(
        polarity=-1,
        parameters={"LEVEL": 54, "VERSION": 4.0, "VTH0": -0.4},
        version=4.0,
    )
    assert compatible.version == 4.0


def test_evaluation_enforces_terminal_conservation():
    currents = np.array((1e-3, 0.0, -1e-3, 0.0))
    charges = np.array((1e-15, 2e-15, -2.5e-15, -0.5e-15))
    conductance = np.array([
        [1e-3, 2e-3, -3e-3, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-1e-3, -2e-3, 3e-3, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    capacitance = conductance * 1e-12
    result = Bsim4Evaluation(
        currents,
        conductance,
        charges,
        capacitance,
        {"gm": 2e-3},
        Bsim4Noise(np.zeros((4, 4))),
    )
    assert result.operating_point["gm"] == pytest.approx(2e-3)


def test_invalid_cards_and_nonconservative_results_fail():
    with pytest.raises(Bsim4ValidationError, match="polarity"):
        Bsim4ModelCard(0, {})
    with pytest.raises(Bsim4ValidationError, match="'w'"):
        Bsim4InstanceCard({"w": 0, "l": 30e-9})
    with pytest.raises(Bsim4ValidationError, match="KCL"):
        Bsim4Evaluation(
            np.ones(4),
            np.zeros((4, 4)),
            np.zeros(4),
            np.zeros((4, 4)),
            {},
        )
