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
    with pytest.raises(TypeError):
        model.parameters["vth0"] = 0.5
    with pytest.raises(TypeError):
        instance.parameters["nf"] = 4
    assert Bsim4Bias(0.9, 0.6, 0.0, 0.0).terminals.shape == (4,)
    compatible = Bsim4ModelCard(
        polarity=-1,
        parameters={"LEVEL": 54, "VERSION": 4.0, "VTH0": -0.4},
        version=4.0,
    )
    assert compatible.version == 4.0


def test_cards_precompute_stable_native_cache_key_items():
    model = Bsim4ModelCard(1, {"VTH0": 0.4, "TOXE": 1e-9})
    instance = Bsim4InstanceCard({"W": 2e-6, "L": 1e-6, "NF": 2})

    assert model._parameter_items == (("toxe", 1e-9), ("vth0", 0.4))
    assert instance._parameter_items == (
        ("l", 1e-6),
        ("nf", 2.0),
        ("w", 2e-6),
    )


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


def _tsmc28_cards_available():
    import os

    from circuitopt.toolchain import tsmc28_model_dir

    return os.path.isfile(os.path.join(
        tsmc28_model_dir(), "cln28hpcp_1d8_elk_v1d0_2p2.l"))


@pytest.mark.skipif(
    not _tsmc28_cards_available(), reason="TSMC28 model deck not configured")
def test_whole_card_entry_matches_parameter_at_a_time_setup():
    # `co_bsim4_set_card` exists only to remove FFI crossings and the linear
    # keyword scan behind them. It must land on the same handle state as the
    # single-value setters, applied in the card's own order -- several vendor
    # setters react to what was already set.
    import ctypes as C

    import circuitopt.compact_models.bsim4.native as native
    from circuitopt.pdk.tsmc28.device import Tsmc28NativeNfet

    device = Tsmc28NativeNfet(W=1.0, L=0.03)
    model, instance = device.model_card, device.instance_card
    temperature = float(device.temperature)
    library = native._select_library("rust")

    def evaluate(build):
        pointer = library.co_bsim4_create(model.polarity, temperature)
        assert pointer
        try:
            build(pointer)
            assert library.co_bsim4_setup(pointer) == 0
            terminals = np.asarray([0.7, 0.6, 0.0, 0.0], dtype=np.float64)
            out = [np.zeros(n, dtype=np.float64) for n in (4, 16, 4, 16, 8)]
            double = C.POINTER(C.c_double)
            assert library.co_bsim4_dc(
                pointer,
                terminals.ctypes.data_as(double),
                *[array.ctypes.data_as(double) for array in out],
            ) == 0
            return b"".join(array.tobytes() for array in out)
        finally:
            library.co_bsim4_destroy(pointer)

    def one_at_a_time(pointer):
        for name, value in model.parameters.items():
            assert library.co_bsim4_set_model(
                pointer, name.encode("ascii"), value) == 0
        for name, value in instance.parameters.items():
            assert library.co_bsim4_set_instance(
                pointer, name.encode("ascii"), value) == 0

    def whole_card(pointer):
        failed = C.c_size_t(0)
        for card, attribute, flag in (
            (model, "_native_model_payload", 0),
            (instance, "_native_instance_payload", 1),
        ):
            names, values, count, _ = native._card_payload(card, attribute)
            assert library.co_bsim4_set_card(
                pointer, names, values, count, flag, C.byref(failed)) == 0

    assert evaluate(whole_card) == evaluate(one_at_a_time)


@pytest.mark.skipif(
    not _tsmc28_cards_available(), reason="TSMC28 model deck not configured")
def test_whole_card_entry_reports_the_offending_parameter():
    import ctypes as C

    import circuitopt.compact_models.bsim4.native as native

    library = native._select_library("rust")
    pointer = library.co_bsim4_create(1, 300.15)
    assert pointer
    try:
        names = (C.c_char_p * 3)(b"tnom", b"not_a_bsim4_parameter", b"toxe")
        values = (C.c_double * 3)(27.0, 1.0, 2e-9)
        failed = C.c_size_t(0)
        status = library.co_bsim4_set_card(
            pointer, names, values, 3, 0, C.byref(failed))
        assert status != 0
        assert failed.value == 1
    finally:
        library.co_bsim4_destroy(pointer)
