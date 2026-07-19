"""Device-level tests for the compiled AT4000TG OTFT kernels.

History: R3 established these as rust-vs-numba `_impl` parity tests. The Python
`_impl` kernels were removed in v2.0.0 (R7); their equations live on as the
compiled reference oracle (``OtftModel(..., reference=True)``). The tests now
pin (a) batch-vs-scalar marshalling consistency, (b) exact residual identities
at returned Newton roots, (c) analytic-vs-finite-difference derivative
agreement, and (d) production-vs-reference-oracle agreement through the public
model API — with the golden corpus as the frozen numerical oracle.
"""
from __future__ import annotations

from dataclasses import astuple

import numpy as np
import pytest

from circuitopt.pmos_tft_model import PMOS_TFT, otft_reference_mode

try:
    import circuitopt_core
except ImportError:  # pragma: no cover - depends on the optional compiled wheel
    circuitopt_core = None


requires_rust_otft = pytest.mark.skipif(
    circuitopt_core is None or not hasattr(circuitopt_core, "otft_device_batch"),
    reason="circuitopt_core OTFT extension is not installed",
)


def _params(device: PMOS_TFT) -> np.ndarray:
    return np.asarray(astuple(device.get_otft_params()), dtype=np.float64)


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
def test_rust_otft_device_batch_matches_scalar_model(geometry):
    """The GIL-free batch entry must agree with the scalar OtftModel bindings
    bit-for-bit (same kernels, different marshalling), satisfy the residual
    identity, and its analytic derivatives must match the finite-difference
    variant to the established 1e-7."""
    device = PMOS_TFT(W=geometry[0], L=geometry[1])
    params = _params(device)
    points = _points(device)
    scalar = device._get_rust_model()

    currents, charges, jacobians, derivatives = circuitopt_core.otft_device_batch(
        params, points, True, True, True, 1e-3
    )
    fd_derivatives = circuitopt_core.otft_terminal_derivatives_batch(
        params, points, True, True, True, 1e-3, 1e-6
    )
    assert all(isinstance(value, np.ndarray) for value in (
        currents, charges, jacobians, derivatives, fd_derivatives))

    for point, got_i, got_q, got_j, got_d, got_d_fd in zip(
        points, currents, charges, jacobians, derivatives, fd_derivatives
    ):
        vs, vd, vg, vs1, vd1 = point
        # (a) batch == scalar binding, bit-for-bit.
        np.testing.assert_array_equal(
            got_i, scalar.eval_currents(vs, vd, vg, vs1, vd1))
        np.testing.assert_array_equal(
            got_q, scalar.capacitance_charges(vs, vd, vg, vs1, vd1))
        sd = scalar.terminal_derivatives(vs, vd, vg, vs1, vd1, True, True,
                                         True, 1e-3)
        assert bool(got_d[0]) == sd[0]
        np.testing.assert_array_equal(got_d[1:], sd[1:])
        # (b) the stamped residual pair is the exact current identity.
        i_s_s1, i_s1_d1, i_d1_d = got_i[0], got_i[1], got_i[2]
        np.testing.assert_allclose(got_j[0], i_s_s1 - i_s1_d1, rtol=1e-12,
                                   atol=1e-24)
        np.testing.assert_allclose(got_j[1], i_s1_d1 - i_d1_d, rtol=1e-12,
                                   atol=1e-24)
        # Analytic-arm structure: the interior branch conductance is 1/0.1.
        assert got_j[3] == 10.0
        assert got_j[4] == 10.0
        assert np.all(np.isfinite(got_j))
        # (c) analytic vs finite-difference terminal derivatives.
        assert bool(got_d[0]) and bool(got_d_fd[0])
        np.testing.assert_allclose(got_d[1:], got_d_fd[1:], rtol=1e-7,
                                   atol=1e-17)


@requires_rust_otft
@pytest.mark.parametrize("analytic", [False, True])
def test_rust_otft_internal_newton_converges_to_kcl_root(analytic):
    """Both Newton variants (finite-difference and analytic-Jacobian) must
    return internal nodes that satisfy the KCL residual below the tolerance;
    the analytic variant must also match the scalar production binding
    bit-for-bit."""
    device = PMOS_TFT(W=5000, L=30)
    params = _params(device)
    scalar = device._get_rust_model()
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

    for point, result in zip(points, got):
        vs, vd, vg, x0s, x0d = point
        converged, rs1, rd1 = bool(result[0]), result[1], result[2]
        assert converged
        i = scalar.eval_currents(vs, vd, vg, rs1, rd1)
        assert abs(i[0] - i[1]) + abs(i[1] - i[2]) < 1e-9
        if analytic:
            ok, ss1, sd1 = scalar.newton_internal(vs, vd, vg, x0s, x0d,
                                                  1e-12, 40)
            assert ok
            assert (rs1, rd1) == (ss1, sd1)
            np.testing.assert_array_equal(result[3:], result[3:])  # counters finite
            assert np.all(np.isfinite(result[3:]))


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
def test_public_otft_scalar_methods_agree_with_reference_oracle():
    """The public model API under ``otft_reference_mode`` (the root-selection
    recovery oracle, formerly the Python ``_impl`` path) agrees with the
    production path at a healthy operating point — same tolerances the retired
    engine A/B used."""
    reference = PMOS_TFT(W=5000, L=30)
    with otft_reference_mode():
        assert reference._get_rust_model() is reference._rust_model_ref
        ref_id = reference.get_Idc(40.0, 0.0, 20.0)
        ref_caps = reference.get_capacitances(40.0, 0.0, 20.0)
        ref_ss = reference.get_ss_params(40.0, 0.0, 20.0)

    device = PMOS_TFT(W=5000, L=30)
    got_id = device.get_Idc(40.0, 0.0, 20.0)
    got_caps = device.get_capacitances(40.0, 0.0, 20.0)
    got_ss = device.get_ss_params(40.0, 0.0, 20.0)

    assert type(device._get_rust_model()).__name__ == "OtftModel"
    assert device._get_rust_model() is device._rust_model
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
def test_rust_dense_gepp_matches_lapack(size):
    """The in-place GEPP solves ``A x = -b``; LAPACK is the referee now that
    the interpreted `_impl` GEPP is gone."""
    rng = np.random.default_rng(731 + size)
    matrix = rng.normal(size=(size, size))
    matrix += np.eye(size) * (size + 1.0)
    rhs = rng.normal(size=size)
    expected = np.linalg.solve(matrix, -rhs)

    got_ok, _got_matrix, got_solution = circuitopt_core.dense_neg_solve(
        matrix, rhs
    )
    assert got_ok
    np.testing.assert_allclose(got_solution, expected, rtol=1e-10, atol=1e-12)


@requires_rust_otft
def test_rust_dense_gepp_reports_singular_matrix():
    matrix = np.array([[1.0, 2.0], [2.0, 4.0]])
    rhs = np.array([1.0, 2.0])
    got_ok, _, _ = circuitopt_core.dense_neg_solve(matrix, rhs)
    assert not got_ok
