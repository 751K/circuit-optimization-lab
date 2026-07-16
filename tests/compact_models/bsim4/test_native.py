"""Native Berkeley BSIM4.5 numerical-kernel tests."""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from circuitopt.compact_models.bsim4 import (
    Bsim4Bias,
    Bsim4NativeError,
    NativeBsim4Backend,
)
from circuitopt.pdk.tsmc28 import load_tsmc28_core_library
from circuitopt.toolchain import tsmc28_model_dir


_PATH = os.path.join(
    tsmc28_model_dir(), "cln28hpcp_1d8_elk_v1d0_2p2.l")
pytestmark = [
    pytest.mark.skipif(
        not os.path.isfile(_PATH),
        reason="licensed TSMC28HPC+ model is not installed",
    ),
    pytest.mark.skipif(
        not any(shutil.which(name) for name in ("clang", "cc", "gcc")),
        reason="native BSIM4 tests require a C99 compiler",
    ),
]


def _cards(polarity="nmos", *, corner="tt", temperature_c=27.0):
    card = load_tsmc28_core_library(_PATH).core_card(
        polarity,
        width_um=1.0,
        length_um=0.03,
        corner=corner,
        temperature_c=temperature_c,
    )
    return card.to_bsim4_cards()


@pytest.mark.parametrize(
    ("polarity", "bias", "drain_sign"),
    [
        ("nmos", Bsim4Bias(0.9, 0.9, 0.0, 0.0), 1),
        ("pmos", Bsim4Bias(0.0, 0.0, 0.9, 0.9), -1),
    ],
)
def test_native_tsmc_core_device_is_finite_and_conservative(
    monkeypatch, polarity, bias, drain_sign
):
    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/ngspice")
    model, instance = _cards(polarity)
    result = NativeBsim4Backend(cache_size=0).evaluate(model, instance, bias)
    assert np.all(np.isfinite(result.terminal_currents))
    assert np.sign(result.terminal_currents[0]) == drain_sign
    assert abs(result.terminal_currents[0]) > 1e-6
    assert result.operating_point["gm"] > 0
    assert result.operating_point["gds"] > 0
    assert result.operating_point["internal_nodes"] == 2


@pytest.mark.parametrize(
    ("polarity", "voltage"),
    [
        ("nmos", np.asarray((0.75, 0.68, 0.05, 0.0))),
        ("pmos", np.asarray((0.15, 0.22, 0.85, 0.9))),
    ],
)
def test_native_jacobian_and_charge_derivative_match_finite_difference(
    polarity, voltage
):
    model, instance = _cards(polarity)
    backend = NativeBsim4Backend()

    def evaluate(values):
        return backend.evaluate(model, instance, Bsim4Bias(*values))

    nominal = evaluate(voltage)
    step = 1e-6
    current_fd = np.column_stack([
        (
            evaluate(voltage + np.eye(4)[column] * step).terminal_currents
            - evaluate(voltage - np.eye(4)[column] * step).terminal_currents
        ) / (2 * step)
        for column in range(4)
    ])
    charge_fd = np.column_stack([
        (
            evaluate(voltage + np.eye(4)[column] * step).terminal_charges
            - evaluate(voltage - np.eye(4)[column] * step).terminal_charges
        ) / (2 * step)
        for column in range(4)
    ])
    np.testing.assert_allclose(
        nominal.conductance, current_fd, rtol=1e-3, atol=5e-10)
    np.testing.assert_allclose(
        nominal.capacitance, charge_fd, rtol=5e-3, atol=2e-18)


def test_native_setup_does_not_write_model_check_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model, instance = _cards(corner="ss", temperature_c=-40)
    result = NativeBsim4Backend(cache_size=0).evaluate(
        model,
        instance,
        Bsim4Bias(0.7, 0.6, 0.0, 0.0, temperature_k=233.15),
    )
    assert result.terminal_currents[0] > 0
    assert not (tmp_path / "bsim4v5.out").exists()


def test_native_noise_is_four_terminal_hermitian_and_frequency_dependent():
    model, instance = _cards()
    backend = NativeBsim4Backend()
    bias = Bsim4Bias(0.75, 0.68, 0.05, 0.0)
    low = backend.evaluate(model, instance, bias, frequency_hz=1.0).noise
    high = backend.evaluate(model, instance, bias, frequency_hz=1e6).noise
    assert low is not None and high is not None
    np.testing.assert_allclose(
        low.spectral_density, low.spectral_density.conj().T, atol=1e-30)
    np.testing.assert_allclose(
        low.spectral_density.sum(axis=0), 0.0, atol=1e-28)
    assert np.trace(low.components["white"]).real == pytest.approx(
        np.trace(high.components["white"]).real, rel=1e-5)
    assert np.trace(low.components["flicker"]).real > (
        np.trace(high.components["flicker"]).real * 1e4)
    with pytest.raises(Bsim4NativeError, match="frequency"):
        backend.evaluate(model, instance, bias, frequency_hz=0.0)
