from __future__ import annotations

import os
from types import SimpleNamespace

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


def test_integration_coefficient_columns_require_solver_rows():
    from circuitopt.compact_models.bsim4.transient import (
        _integration_coefficient_columns,
    )

    with pytest.raises(RuntimeError, match="did not return"):
        _integration_coefficient_columns(
            np.asarray([0.0, 1.0]), "gear2", None)


def test_integration_coefficient_columns_pass_through_solver_rows():
    from circuitopt.compact_models.bsim4.transient import (
        _integration_coefficient_columns,
    )

    provided = np.asarray([[0.0, 0.0, 0.0], [7.0, -8.0, 9.0], [1.5, -2.5, 3.5]])
    a0, a1, a2 = _integration_coefficient_columns(
        np.asarray([0.0, 1.0, 2.0]), "gear2", provided)

    # Both fixed and adaptive Rust solvers report the coefficients they actually
    # stamped; sample 0 carries no derivative and is dropped.
    np.testing.assert_array_equal(a0, provided[1:, 0])
    np.testing.assert_array_equal(a1, provided[1:, 1])
    np.testing.assert_array_equal(a2, provided[1:, 2])


def test_fixed_grid_wrapper_returns_native_integration_coefficients(monkeypatch):
    from circuitopt.compact_models.bsim4 import rust_transient

    coefficients = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.2, -0.2, 0.0],
    ])
    handles = []

    class FakeHandle:
        def __init__(self):
            self.pointer = 1
            self.closed = False
            handles.append(self)

    class FakeLease:
        def __init__(self):
            self.device = FakeHandle()

        def close(self):
            self.device.closed = True

    class FakeProblem:
        def solve_fixed_grid(self, initial, times, inputs, **kwargs):
            assert kwargs["integration_method"] == "gear2"
            assert kwargs["final_load_tolerance"] == 5e-13
            assert kwargs["model_bypass_tolerance"] == 2e-9
            count = len(times)
            return (
                True,
                np.zeros((count, len(initial))),
                np.zeros((count, 1, 4)),
                np.zeros((count, 1, 4)),
                0,
                -1,
                3,
                4,
                5,
                0,
                [],
            )

        def fixed_grid_coefficients(self, times, **kwargs):
            assert kwargs["integration_method"] == "gear2"
            np.testing.assert_array_equal(times, [0.0, 1.0, 6.0])
            return coefficients

    monkeypatch.setattr(
        rust_transient,
        "build_bsim4_problem",
        lambda *_args, **_kwargs: FakeProblem(),
    )
    wrapper = SimpleNamespace(lease_native_solver_handle=FakeLease)
    plan = SimpleNamespace(devices=[SimpleNamespace(name="M1")])

    result = rust_transient.solve_bsim4_rust(
        plan,
        {"M1": wrapper},
        np.asarray([0.0]),
        np.asarray([0.0, 1.0, 6.0]),
        np.empty((0, 3)),
        (),
        method="gear2",
        newton_maxit=10,
        newton_vtol=1e-8,
        newton_step_limit=0.25,
        gmin=1e-12,
        final_load_tolerance=5e-13,
        model_bypass_tolerance=2e-9,
    )

    np.testing.assert_array_equal(result[2], coefficients)
    assert result[-1]["final_load_tolerance_v"] == 5e-13
    assert result[-1]["model_bypass_tolerance_v"] == 2e-9
    assert handles and handles[0].closed


def test_adaptive_wrapper_passes_final_load_tolerance(monkeypatch):
    from circuitopt.compact_models.bsim4 import rust_transient

    class FakeHandle:
        pointer = 1

    class FakeLease:
        device = FakeHandle()

        @staticmethod
        def close():
            pass

    class FakeProblem:
        def solve_adaptive_gear2(self, initial, times, inputs, **kwargs):
            assert kwargs["final_load_tolerance"] == 2e-13
            assert kwargs["model_bypass_tolerance"] == 1e-9
            count = len(times)
            return (
                True,
                np.asarray(times),
                np.zeros((count, len(initial))),
                np.asarray(inputs).T,
                np.zeros((count, 1, 4)),
                np.zeros((count, 1, 4)),
                np.zeros((count, 3)),
                (count - 1, 0, count - 1, 3, 4, 5, 0, 0, 0, 0, 0),
            )

    monkeypatch.setattr(
        rust_transient,
        "build_bsim4_problem",
        lambda *_args, **_kwargs: FakeProblem(),
    )
    wrapper = SimpleNamespace(lease_native_solver_handle=FakeLease)
    plan = SimpleNamespace(devices=[SimpleNamespace(name="M1")])
    config = SimpleNamespace(
        reltol=1e-4,
        vabstol=1e-6,
        iabstol=1e-12,
        max_steps=100,
        h0=None,
    )

    result = rust_transient.solve_bsim4_rust(
        plan,
        {"M1": wrapper},
        np.asarray([0.0]),
        np.asarray([0.0, 1.0]),
        np.empty((0, 2)),
        (),
        method="gear2",
        newton_maxit=10,
        newton_vtol=1e-8,
        newton_step_limit=0.25,
        gmin=1e-12,
        final_load_tolerance=2e-13,
        model_bypass_tolerance=1e-9,
        adaptive=True,
        adaptive_config=config,
    )

    assert result[-1]["final_load_tolerance_v"] == 2e-13
    assert result[-1]["model_bypass_tolerance_v"] == 1e-9


def test_wrapper_releases_partial_handle_batch_if_leasing_fails():
    from circuitopt.compact_models.bsim4 import rust_transient

    closed = []

    class Lease:
        device = SimpleNamespace(pointer=1)

        @staticmethod
        def close():
            closed.append(True)

    good = SimpleNamespace(lease_native_solver_handle=Lease)

    def fail():
        raise RuntimeError("lease failed")

    bad = SimpleNamespace(lease_native_solver_handle=fail)
    plan = SimpleNamespace(
        devices=[
            SimpleNamespace(name="M1"),
            SimpleNamespace(name="M2"),
        ],
    )

    with pytest.raises(RuntimeError, match="lease failed"):
        rust_transient.solve_bsim4_rust(
            plan,
            {"M1": good, "M2": bad},
            np.asarray([0.0]),
            np.asarray([0.0, 1.0]),
            np.empty((0, 2)),
            (),
            method="gear2",
            newton_maxit=10,
            newton_vtol=1e-8,
            newton_step_limit=0.25,
            gmin=1e-12,
        )

    assert closed == [True]


def test_fast_dc_tries_valid_guesses_and_releases_handles(monkeypatch):
    from circuitopt.compact_models.bsim4 import rust_transient

    closed = []
    calls = []

    class Lease:
        device = SimpleNamespace(pointer=1)

        @staticmethod
        def close():
            closed.append(True)

    class FakeProblem:
        @staticmethod
        def solve_dc(initial, inputs, **kwargs):
            calls.append(np.asarray(initial).copy())
            assert inputs.size == 0
            assert kwargs["voltage_tolerance"] == 1e-10
            if initial[0] == 0.0:
                return False, initial, 2, 1e-3
            return True, np.asarray(initial) + 0.25, 3, 1e-12

    monkeypatch.setattr(
        rust_transient,
        "build_bsim4_problem",
        lambda *_args, **_kwargs: FakeProblem(),
    )
    plan = SimpleNamespace(
        devices=[SimpleNamespace(name="M1")],
        n_aug=2,
    )
    devices = {
        "M1": SimpleNamespace(lease_native_solver_handle=Lease),
    }

    result = rust_transient.solve_bsim4_dc_rust(
        plan,
        devices,
        [
            [np.nan, 0.0],
            [0.0, 0.0],
            [1.0, 2.0],
        ],
    )

    np.testing.assert_array_equal(result, [1.25, 2.25])
    assert len(calls) == 2
    assert closed == [True]


def test_fast_dc_rejects_converged_result_with_loose_residual(monkeypatch):
    from circuitopt.compact_models.bsim4 import rust_transient

    class Lease:
        device = SimpleNamespace(pointer=1)

        @staticmethod
        def close():
            pass

    class FakeProblem:
        @staticmethod
        def solve_dc(initial, _inputs, **_kwargs):
            return True, initial, 1, 1e-6

    monkeypatch.setattr(
        rust_transient,
        "build_bsim4_problem",
        lambda *_args, **_kwargs: FakeProblem(),
    )
    plan = SimpleNamespace(
        devices=[SimpleNamespace(name="M1")],
        n_aug=1,
    )
    devices = {
        "M1": SimpleNamespace(lease_native_solver_handle=Lease),
    }

    assert rust_transient.solve_bsim4_dc_rust(
        plan, devices, [[0.5]], dc_tolerance=1e-10) is None


@pytest.mark.parametrize("value", [-1.0, 1.1e-12, np.inf, np.nan])
def test_final_load_tolerance_rejects_unsafe_values(value):
    from circuitopt.compact_models.bsim4.transient import transient_native_bsim4

    with pytest.raises(ValueError, match=r"within \[0, 1e-12\] V"):
        transient_native_bsim4(
            {},
            {},
            np.asarray([0.0, 1.0]),
            topo=None,
            bsim_final_load_tolerance=value,
        )


@pytest.mark.parametrize("value", [-1.0, 1.1e-8, np.inf, np.nan])
def test_model_bypass_tolerance_rejects_unsafe_values(value):
    from circuitopt.compact_models.bsim4.transient import transient_native_bsim4

    with pytest.raises(ValueError, match=r"within \[0, newton_vtol\] V"):
        transient_native_bsim4(
            {},
            {},
            np.asarray([0.0, 1.0]),
            topo=None,
            newton_vtol=1e-8,
            bsim_model_bypass_tolerance=value,
        )


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
    assert result["bsim_final_load_tolerance"] == 0.0
    assert result["bsim_model_bypass_tolerance"] == 0.0
    assert result["nfail"] == 0
    assert output[0] > 0.85
    assert output[-1] < 0.05
    assert np.all(np.isfinite(output))
