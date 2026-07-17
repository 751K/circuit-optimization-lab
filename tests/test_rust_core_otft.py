"""R3 device-level parity for the Rust AT4000TG OTFT kernels."""
from __future__ import annotations

from dataclasses import astuple

import numpy as np
import pytest

from circuitopt.numba_kernels import (
    _capacitance_charges_impl,
    _eval_currents_impl,
    _newton_internal_fast_impl,
    _newton_internal_impl,
    _residual_pair_jac_internal_impl,
    _solve_dense_neg_rhs_inplace_impl,
    _terminal_derivatives_from_jac_impl,
    _terminal_derivatives_impl,
)
from circuitopt.pmos_tft_model import PMOS_TFT

try:
    import circuitopt_core
except ImportError:  # pragma: no cover - depends on the optional compiled wheel
    circuitopt_core = None


requires_rust_otft = pytest.mark.skipif(
    circuitopt_core is None or not hasattr(circuitopt_core, "otft_device_batch"),
    reason="R3 circuitopt_core OTFT extension is not installed",
)


def _params(device: PMOS_TFT) -> np.ndarray:
    return np.asarray(astuple(device.get_numba_params()), dtype=np.float64)


def _points(device: PMOS_TFT):
    biases = [
        (40.0, 0.0, 20.0),
        (36.32147406780545, 29.07917946549335, 30.65),
        (29.07917946549335, 0.0, 5.5217968040937),
        (38.08434178857114, 5.5217968040937, 29.07917946549335),
        (32.0, 31.7, 40.0),
    ]
    return np.asarray(
        [(*bias, *device.get_op(*bias)) for bias in biases], dtype=np.float64)


@requires_rust_otft
@pytest.mark.parametrize("geometry", [(1000, 20), (5000, 30), (61365, 61)])
def test_rust_otft_device_batch_matches_numba_reference(geometry):
    device = PMOS_TFT(W=geometry[0], L=geometry[1])
    params = _params(device)
    points = _points(device)

    currents, charges, jacobians, derivatives = circuitopt_core.otft_device_batch(
        params, points, True, True, True, 1e-3
    )
    reference_derivatives = circuitopt_core.otft_terminal_derivatives_batch(
        params, points, True, True, True, 1e-3, 1e-6
    )
    assert all(isinstance(value, np.ndarray) for value in (
        currents, charges, jacobians, derivatives, reference_derivatives))

    for point, got_i, got_q, got_j, got_d, got_d_ref in zip(
        points, currents, charges, jacobians, derivatives, reference_derivatives
    ):
        vs, vd, vg, vs1, vd1 = point
        args = (
            vs,
            vd,
            vg,
            vs1,
            vd1,
            device.Vfb,
            device.Vss,
            device.Lc,
            device.lambda_,
            device._contact_scale,
            device._channel_exponent,
            device._current_scale,
            device._inv_Rleak,
        )
        cap_args = (
            vs,
            vd,
            vg,
            vs1,
            vd1,
            device.Vfb,
            device._two_over_pi,
            device._cap_cgs1,
            device._cap_cgd1,
            device._cap_half_wl_ci,
            device._cap_cgs3_base,
            device._cap_cgd3_base,
            device.k1,
        )
        ref_i = _eval_currents_impl(*args)
        ref_q = _capacitance_charges_impl(*cap_args)
        ref_j = _residual_pair_jac_internal_impl(*args)
        idc0 = ref_j[1] - (vs1 - vd1) / 0.1
        ref_d = _terminal_derivatives_from_jac_impl(
            vs,
            vd,
            vg,
            vs1,
            vd1,
            ref_j[0],
            ref_j[1],
            idc0,
            *ref_j[2:],
            True,
            True,
            True,
            1e-3,
            device.Vfb,
            device.Vss,
            device.Lc,
            device.lambda_,
            device._contact_scale,
            device._channel_exponent,
            device._current_scale,
            device._inv_Rleak,
        )
        ref_d_fd = _terminal_derivatives_impl(
            vs,
            vd,
            vg,
            vs1,
            vd1,
            True,
            True,
            True,
            1e-3,
            1e-6,
            device.Vfb,
            device.Vss,
            device.Lc,
            device.lambda_,
            device._contact_scale,
            device._channel_exponent,
            device._current_scale,
            device._inv_Rleak,
        )

        np.testing.assert_allclose(got_i, ref_i, rtol=1e-12, atol=1e-24)
        np.testing.assert_allclose(got_q, ref_q, rtol=1e-12, atol=1e-24)
        np.testing.assert_allclose(got_j, ref_j, rtol=1e-12, atol=1e-18)
        assert got_d[0] == ref_d[0]
        np.testing.assert_allclose(got_d[1:], ref_d[1:], rtol=1e-12, atol=1e-18)
        assert got_d_ref[0] == ref_d_fd[0]
        np.testing.assert_allclose(
            got_d_ref[1:], ref_d_fd[1:], rtol=1e-12, atol=1e-18
        )


@requires_rust_otft
@pytest.mark.parametrize("analytic", [False, True])
def test_rust_otft_internal_newton_matches_numba_reference(analytic):
    device = PMOS_TFT(W=5000, L=30)
    params = _params(device)
    biases = [
        (40.0, 0.0, 20.0),
        (36.32147406780545, 29.07917946549335, 30.65),
        (32.0, 31.7, 40.0),
    ]
    points = np.asarray([
        (vs, vd, vg, vs - 0.01 * (vs - vd), vd + 0.01 * (vs - vd))
        for vs, vd, vg in biases
    ], dtype=np.float64)
    got = circuitopt_core.otft_newton_batch(
        params, points, 1e-12, 40, analytic
    )

    kernel = _newton_internal_fast_impl if analytic else _newton_internal_impl
    for point, result in zip(points, got):
        vs, vd, vg, x0s, x0d = point
        ref = kernel(
            vs,
            vd,
            vg,
            x0s,
            x0d,
            1e-12,
            40,
            device.Vfb,
            device.Vss,
            device.Lc,
            device.lambda_,
            device._contact_scale,
            device._channel_exponent,
            device._current_scale,
            device._inv_Rleak,
        )
        assert bool(result[0]) == ref[0]
        np.testing.assert_allclose(result[1:3], ref[1:3], rtol=1e-12, atol=1e-12)
        if analytic:
            np.testing.assert_array_equal(result[3:], ref[3:])


@requires_rust_otft
def test_rust_otft_rejects_wrong_parameter_count():
    with pytest.raises(RuntimeError, match="exactly 16"):
        circuitopt_core.otft_device_batch(
            np.zeros(15, dtype=np.float64), np.empty((0, 5), dtype=np.float64))


@requires_rust_otft
def test_rust_otft_batch_requires_zero_copy_compatible_inputs():
    device = PMOS_TFT(W=5000, L=30)
    points = np.asfortranarray(_points(device))

    with pytest.raises(ValueError, match="C-contiguous"):
        circuitopt_core.otft_device_batch(_params(device), points)


@requires_rust_otft
def test_public_otft_scalar_methods_dispatch_to_rust(monkeypatch):
    import circuitopt.pmos_tft_model as model_module

    monkeypatch.setattr(model_module, "current_engine", lambda: "numba")
    reference = PMOS_TFT(W=5000, L=30)
    ref_id = reference.get_Idc(40.0, 0.0, 20.0)
    ref_caps = reference.get_capacitances(40.0, 0.0, 20.0)
    ref_ss = reference.get_ss_params(40.0, 0.0, 20.0)

    monkeypatch.setattr(model_module, "current_engine", lambda: "rust")
    device = PMOS_TFT(W=5000, L=30)
    got_id = device.get_Idc(40.0, 0.0, 20.0)
    got_caps = device.get_capacitances(40.0, 0.0, 20.0)
    got_ss = device.get_ss_params(40.0, 0.0, 20.0)

    assert type(device._get_rust_model()).__name__ == "OtftModel"
    np.testing.assert_allclose(got_id, ref_id, rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(got_caps, ref_caps, rtol=1e-12, atol=1e-24)
    for name in ("Cgs", "Cgd", "Ich"):
        np.testing.assert_allclose(got_ss[name], ref_ss[name], rtol=1e-12, atol=1e-24)
    for name in ("gm", "gds"):
        np.testing.assert_allclose(got_ss[name], ref_ss[name], rtol=2e-8, atol=1e-24)


@requires_rust_otft
def test_rust_mna_term_values_match_compiled_token_contract():
    got = circuitopt_core.mna_term_values(
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1, 0, 0], dtype=np.int64),
        np.array([0.0, 0.0, 7.5]),
        np.array([2.0, 3.0]),
        np.array([4.0]),
    )
    np.testing.assert_array_equal(got, [3.0, 4.0, 7.5])
    with pytest.raises(RuntimeError, match="equal lengths"):
        circuitopt_core.mna_term_values(
            np.array([0], dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty(0), np.empty(0), np.empty(0))


@requires_rust_otft
@pytest.mark.parametrize("size", [1, 2, 5, 12])
def test_rust_dense_gepp_matches_numba_reference(size):
    rng = np.random.default_rng(731 + size)
    matrix = rng.normal(size=(size, size))
    matrix += np.eye(size) * (size + 1.0)
    rhs = rng.normal(size=size)
    ref_matrix = matrix.copy()
    ref_rhs = rhs.copy()
    ref_ok, ref_solution = _solve_dense_neg_rhs_inplace_impl(ref_matrix, ref_rhs)

    got_ok, got_matrix, got_solution = circuitopt_core.dense_neg_solve(
        matrix, rhs
    )
    assert got_ok == ref_ok
    np.testing.assert_allclose(got_matrix, ref_matrix, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(got_solution, ref_solution, rtol=1e-14, atol=1e-15)


@requires_rust_otft
def test_rust_dense_gepp_reports_singular_matrix():
    matrix = np.array([[1.0, 2.0], [2.0, 4.0]])
    rhs = np.array([1.0, 2.0])
    got_ok, _, _ = circuitopt_core.dense_neg_solve(matrix, rhs)
    assert not got_ok
