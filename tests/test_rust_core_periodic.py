"""R4 parity and ABI tests for periodic Rust kernels."""
from __future__ import annotations

import numpy as np
import pytest

from circuitopt.numba_kernels import (
    _pac_linearize_orbit_gate1_impl,
    _pac_linearize_orbit_impl,
    _pnoise_fold_psd_impl,
    _pnoise_hb_blocks_impl,
    py_impl,
)
from circuitopt.pmos_tft_model import PMOS_TFT

try:
    import circuitopt_core
except ImportError:  # pragma: no cover - optional compiled wheel
    circuitopt_core = None


requires_rust_periodic = pytest.mark.skipif(
    circuitopt_core is None
    or not hasattr(circuitopt_core, "PeriodicLinearizationProblem"),
    reason="R4 circuitopt_core periodic extension is not installed",
)


def _params(device):
    p = device.get_numba_params()
    return np.array([
        p.Vfb, p.Vss, p.Lc, p.lambda_, p.contact_scale,
        p.channel_exponent, p.current_scale, p.inv_Rleak,
        p.two_over_pi, p.cap_cgs1, p.cap_cgd1, p.cap_half_wl_ci,
        p.cap_cgs3_base, p.cap_cgd3_base, p.k1, p.gate_leak_g,
    ])


@requires_rust_periodic
@pytest.mark.parametrize("charge_caps", [False, True])
def test_hb_blocks_match_reference_exactly(charge_caps):
    rng = np.random.default_rng(1404)
    gf = rng.normal(size=(7, 3, 3)) + 1j * rng.normal(size=(7, 3, 3))
    cf = rng.normal(size=(7, 3, 3)) + 1j * rng.normal(size=(7, 3, 3))
    got = circuitopt_core.periodic_hb_blocks(
        gf, cf, 2, 3.2e6, charge_caps)
    expected = py_impl(_pnoise_hb_blocks_impl)(
        gf, cf, 2, 3.2e6, charge_caps)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])


@requires_rust_periodic
def test_fold_psd_matches_reference_and_rejects_bad_indices():
    rng = np.random.default_rng(1405)
    K, n, nf, ns = 2, 3, 5, 4
    nb = 2 * K + 1
    adjs = rng.normal(size=(nf, nb * n)) + 1j * rng.normal(size=(nf, nb * n))
    freqs = np.geomspace(1.0, 1e5, nf)
    p = rng.integers(-1, nb * n, size=(ns, nb), dtype=np.int64)
    q = rng.integers(-1, nb * n, size=(ns, nb), dtype=np.int64)
    thermal = rng.normal(size=(ns, nb, nb)) + 1j * rng.normal(size=(ns, nb, nb))
    flicker = rng.normal(size=(ns, 4 * K + 1)) + 1j * rng.normal(
        size=(ns, 4 * K + 1))
    got = circuitopt_core.periodic_fold_psd(
        adjs, freqs, K, 3.2e6, p, q, thermal, flicker)
    expected = py_impl(_pnoise_fold_psd_impl)(
        adjs, freqs, K, 3.2e6, p, q, thermal, flicker)
    np.testing.assert_allclose(got[0], expected[0], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-14, atol=1e-14)

    bad = p.copy()
    bad[0, 0] = adjs.shape[1]
    with pytest.raises(ValueError, match="invalid periodic noise"):
        circuitopt_core.periodic_fold_psd(
            adjs, freqs, K, 3.2e6, bad, q, thermal, flicker)
    with pytest.raises(ValueError, match="C-contiguous"):
        circuitopt_core.periodic_hb_blocks(
            np.asfortranarray(thermal[:3]),
            np.asfortranarray(thermal[:3]), 1, 1.0, False)


@requires_rust_periodic
def test_dense_four_terminal_orbit_stamp_maps_state_and_drive_columns():
    problem = circuitopt_core.PeriodicLinearizationProblem({
        "node_count": 1, "state_count": 1, "input_count": 0,
        "drive_count": 1, "devices": [],
        "dense_devices": [[(0, 0), (1, 0), (2, 0), (2, 0)]],
        "resistors": [], "capacitors": [], "gmin": 0.0, "fd_step": 1e-4,
    })
    dense_g = np.zeros((2, 1, 4, 4))
    dense_c = np.zeros_like(dense_g)
    dense_g[:, 0, 0, 0] = 2.0
    dense_g[:, 0, 0, 1] = -3.0
    dense_c[:, 0, 0, 0] = 5.0
    got = problem.linearize(
        np.zeros((2, 1)), np.empty((0, 2)), np.zeros((2, 1)),
        np.empty((0, 2)), dense_g, dense_c)
    np.testing.assert_array_equal(got[0][:, 0, 0], [2.0, 2.0])
    np.testing.assert_array_equal(got[1][:, 0, 0], [5.0, 5.0])
    np.testing.assert_array_equal(got[2][:, 0, 0], [-3.0, -3.0])


def _terminal_arrays():
    # drain is solved node 0; gate/source are fixed rails.
    value_d = (np.array([0]), np.array([0]), np.array([0.0]))
    value_g = (np.array([2]), np.array([0]), np.array([30.0]))
    value_s = (np.array([2]), np.array([0]), np.array([40.0]))
    stamp_d = (np.array([0]), np.array([0]))
    stamp_g = (np.array([2]), np.array([0]))
    stamp_s = (np.array([2]), np.array([0]))
    return value_d, value_g, value_s, stamp_d, stamp_g, stamp_s


@requires_rust_periodic
@pytest.mark.parametrize("gate1", [False, True])
def test_otft_orbit_linearization_matches_numba_reference(gate1):
    device = PMOS_TFT(W=5000, L=30)
    params = _params(device)
    N = 12
    phase = 2 * np.pi * np.arange(N) / N
    node_wave = np.ascontiguousarray((20.0 + 0.2 * np.sin(phase))[:, None])
    node_dot = np.ascontiguousarray((0.2 * 2 * np.pi * np.cos(phase))[:, None])
    input_wave = np.empty((0, N))
    input_dot = np.empty((0, N))
    vd, vg, vs, sd, sg, ss = _terminal_arrays()
    passive_i = np.empty(0, dtype=np.int64)
    passive_f = np.empty(0)
    gate_record = (1, float(device.R_cap), float(device.R_cap2)) if gate1 else None
    problem = circuitopt_core.PeriodicLinearizationProblem({
        "node_count": 1, "state_count": 2 if gate1 else 1,
        "input_count": 0, "drive_count": 0,
        "devices": [
            ((0, 0, 0.0), (2, 0, 30.0), (2, 0, 40.0),
             (0, 0), (2, 0), (2, 0), params.tolist(), gate_record),
        ],
        "dense_devices": [],
        "resistors": [], "capacitors": [], "gmin": 1e-12, "fd_step": 1e-4,
    })
    empty_dense = np.empty((N, 0, 4, 4))
    got = problem.linearize(
        node_wave, input_wave, node_dot, input_dot, empty_dense, empty_dense)

    p = [np.array([value]) for value in params[:15]]
    common = (
        node_wave, input_wave,
        *vd, *vg, *vs, *sd, *sg, *ss,
    )
    passives = (
        passive_i, passive_i, passive_i, passive_i, passive_f,
        passive_i, passive_i, passive_i, passive_i, passive_f,
    )
    if gate1:
        expected = py_impl(_pac_linearize_orbit_gate1_impl)(
            node_wave, input_wave, node_dot, input_dot,
            *vd, *vg, *vs, *sd, *sg, *ss,
            np.array([1]), np.array([device.R_cap]), np.array([device.R_cap2]),
            *p, *passives, 0, 2, 1e-4)
    else:
        expected = py_impl(_pac_linearize_orbit_impl)(
            *common, *p, *passives, 0)
    assert expected[0]
    for actual, reference in zip(got, expected[1:]):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-18)
