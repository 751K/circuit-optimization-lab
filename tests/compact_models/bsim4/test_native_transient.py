from __future__ import annotations

import os

import numpy as np
import pytest


def test_expanded_grid_does_not_split_mdac_exact_step_multiples():
    from circuitopt.compact_models.bsim4.transient import _expanded_grid

    requested = np.linspace(0.0, 5e-9, 501)
    waveform = np.linspace(0.1, 0.9, len(requested))
    expanded, inputs, indices = _expanded_grid(
        requested, {"vin": waveform}, 1e-11)

    np.testing.assert_array_equal(expanded, requested)
    np.testing.assert_array_equal(inputs["vin"], waveform)
    np.testing.assert_array_equal(indices, np.arange(len(requested)))


def test_expanded_grid_preserves_integer_subdivision_and_real_excess():
    from circuitopt.compact_models.bsim4.transient import _expanded_grid

    exact, _, exact_indices = _expanded_grid(
        np.asarray([0.0, 3e-11]), {}, 1e-11)
    assert len(exact) == 4
    assert exact_indices.tolist() == [0, 3]

    within_tolerance, _, _ = _expanded_grid(
        np.asarray([0.0, 1.0 + 5e-13]), {}, 1.0)
    above_tolerance, _, above_indices = _expanded_grid(
        np.asarray([0.0, 1.0 + 2e-12]), {}, 1.0)
    assert len(within_tolerance) == 2
    assert len(above_tolerance) == 3
    assert above_indices.tolist() == [0, 2]


def _scalar_coefficients(tgrid, method, sample):
    """The per-sample BDF form the vectorized columns replaced.

    Kept as an independent oracle: branch-current reconstruction must stay
    bit-identical to this recurrence, not merely close to it.
    """
    h = float(tgrid[sample] - tgrid[sample - 1])
    if method == "be" or sample == 1:
        return (1.0 / h, -1.0 / h, 0.0)
    h_prev = float(tgrid[sample - 1] - tgrid[sample - 2])
    rho = h / h_prev
    if rho > 2.0:
        return (1.0 / h, -1.0 / h, 0.0)
    return (
        (1.0 + 2.0 * rho) / ((1.0 + rho) * h),
        -(1.0 + rho) / h,
        (rho * rho) / ((1.0 + rho) * h),
    )


@pytest.mark.parametrize("method", ["be", "gear2"])
def test_integration_coefficient_columns_are_bit_identical_to_scalar_form(method):
    from circuitopt.compact_models.bsim4.transient import (
        _integration_coefficient_columns,
    )

    # Uniform, gently stretched (rho <= 2 -> BDF2), and abruptly stretched
    # (rho > 2 -> BE fallback) intervals in one grid.
    steps = [1e-11] * 3 + [1.5e-11, 2.0e-11, 1e-11, 5e-11, 1e-12, 3e-12]
    tgrid = np.concatenate(([0.0], np.cumsum(steps)))
    ratios = np.diff(tgrid)[1:] / np.diff(tgrid)[:-1]
    assert np.any(ratios > 2.0) and np.any(ratios <= 2.0)

    a0, a1, a2 = _integration_coefficient_columns(tgrid, method, None)
    for sample in range(1, len(tgrid)):
        expected = _scalar_coefficients(tgrid, method, sample)
        actual = (a0[sample - 1], a1[sample - 1], a2[sample - 1])
        assert actual == expected, f"sample {sample}: {actual} != {expected}"


def test_integration_coefficient_columns_pass_through_solver_rows():
    from circuitopt.compact_models.bsim4.transient import (
        _integration_coefficient_columns,
    )

    provided = np.asarray([[0.0, 0.0, 0.0], [7.0, -8.0, 9.0], [1.5, -2.5, 3.5]])
    a0, a1, a2 = _integration_coefficient_columns(
        np.asarray([0.0, 1.0, 2.0]), "gear2", provided)

    # The adaptive solver reports its own accepted-step coefficients; sample 0
    # carries no derivative and is dropped.
    np.testing.assert_array_equal(a0, provided[1:, 0])
    np.testing.assert_array_equal(a1, provided[1:, 1])
    np.testing.assert_array_equal(a2, provided[1:, 2])


def test_shift_two_reuses_first_sample_for_the_opening_step():
    from circuitopt.compact_models.bsim4.transient import _shift_two

    values = np.asarray([10.0, 20.0, 30.0, 40.0])
    # sample 1 has no two-back history and reuses values[0]; later samples
    # take values[sample - 2].
    np.testing.assert_array_equal(_shift_two(values), [10.0, 10.0, 20.0])

    pairs = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(_shift_two(pairs), [[1.0, 2.0]])


def test_subdivision_counts_match_the_scalar_rule():
    from circuitopt.compact_models.bsim4.transient import (
        _subdivision_count,
        _subdivision_counts,
    )

    max_step = 1e-11
    intervals = np.asarray([
        1e-11,            # exact single step
        3e-11,            # exact multiple
        1e-11 + 5e-24,    # roundoff-only excess, must not split
        1e-11 + 2e-23,    # real excess, must split
        5e-12,            # below the limit
        7.3e-11,          # fractional multiple
    ])
    expected = [_subdivision_count(value, max_step) for value in intervals]
    assert _subdivision_counts(intervals, max_step).tolist() == expected


def _model_available():
    from circuitopt.toolchain import tsmc28_model_dir

    return os.path.isfile(os.path.join(
        tsmc28_model_dir(),
        "cln28hpcp_1d8_elk_v1d0_2p2.l",
    ))


@pytest.mark.skipif(not _model_available(), reason="TSMC28 model deck not configured")
@pytest.mark.parametrize("adaptive", [False, True])
def test_native_inverter_charge_transient_without_ngspice(monkeypatch, adaptive):
    from circuitopt.circuit_loader import circuit_from_dict
    from circuitopt.transient_solver import transient

    monkeypatch.setenv("NGSPICE_BIN", "/definitely/not/an/executable")
    spec = circuit_from_dict({
        "name": "native_tsmc28_inverter",
        "solved": ["OUT"],
        "rails": {"VDD": "VDD", "GND": 0.0, "IN": 0.0},
        "devices": [
            {"name": "MN", "drain": "OUT", "gate": "IN", "source": "GND",
             "W": 1.0, "L": 0.03},
            {"name": "MP", "drain": "OUT", "gate": "IN", "source": "VDD",
             "W": 2.0, "L": 0.03},
        ],
        "models": {
            "MN": {"pdk": "tsmc28hpcp", "model": "nmos", "section": "inherit", "bin": "auto"},
            "MP": {"pdk": "tsmc28hpcp", "model": "pmos", "section": "inherit", "bin": "auto", "vb": 0.9},
        },
        "bias": {"VDD": 0.9},
        "outputs": ["OUT"],
        "load_caps": [["OUT", "GND", 2e-15]],
        "transient_inputs": {"MN": "vin", "MP": "vin"},
        "dc_guesses": [{"OUT": 0.9}],
    })
    tgrid = np.linspace(0.0, 0.4e-9, 81)
    vin = np.where(tgrid < 0.1e-9, 0.0, 0.9)
    result = transient(
        spec.sizes,
        spec.bias,
        tgrid,
        binding=spec.binding(),
        inputs={"vin": vin},
        corner="tt",
        integration_method="gear2",
        max_step=1e-12,
        adaptive=adaptive,
    )
    output = result["nodes"]["OUT"]
    assert result["backend"] == "bsim4_native"
    assert result["adaptive"] is adaptive
    assert result["nfail"] == 0
    assert output[0] > 0.85
    assert output[-1] < 0.05
    assert np.all(np.isfinite(output))
