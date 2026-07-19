"""Parity and ABI tests for the compiled periodic kernels.

History: R4 established these against the Python/numba ``_impl`` periodic
kernels; those were removed in v2.0.0 (R7). The tests now carry *independent
in-test referees*: the HB conversion and fold-PSD checks reimplement the
published definitions directly in numpy, and the orbit-linearization check
reconstructs the expected stamps per time sample from the scalar
``circuitopt_core.OtftModel`` device bindings (warm-started exactly like the
kernel). The golden corpus pins the full-circuit numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from circuitopt.pmos_tft_model import PMOS_TFT

try:
    import circuitopt_core
except ImportError:  # pragma: no cover - optional compiled wheel
    circuitopt_core = None


requires_rust_periodic = pytest.mark.skipif(
    circuitopt_core is None
    or not hasattr(circuitopt_core, "PeriodicLinearizationProblem"),
    reason="circuitopt_core periodic extension is not installed",
)


def _params(device):
    p = device.get_otft_params()
    return np.array([
        p.Vfb, p.Vss, p.Lc, p.lambda_, p.contact_scale,
        p.channel_exponent, p.current_scale, p.inv_Rleak,
        p.two_over_pi, p.cap_cgs1, p.cap_cgd1, p.cap_half_wl_ci,
        p.cap_cgs3_base, p.cap_cgd3_base, p.k1, p.gate_leak_g,
    ])


def _hb_blocks_reference(Gf, Cf, K, fundamental, charge_caps):
    """Direct numpy statement of the dense HB conversion blocks.

    Row sideband kr, column sideband kc couple through harmonic (kr-kc) mod N;
    the jω term uses the row sideband for charge caps and the column sideband
    for admittance caps."""
    N, n, _ = Gf.shape
    nb = 2 * K + 1
    size = nb * n
    Y = np.zeros((size, size), dtype=np.complex128)
    C = np.zeros((size, size), dtype=np.complex128)
    for kr_i in range(nb):
        kr = kr_i - K
        for kc_i in range(nb):
            kc = kc_i - K
            sideband = kr if charge_caps else kc
            jw = 2.0j * np.pi * sideband * fundamental
            blk = (kr - kc) % N
            r0, c0 = kr_i * n, kc_i * n
            Y[r0:r0 + n, c0:c0 + n] = Gf[blk] + jw * Cf[blk]
            C[r0:r0 + n, c0:c0 + n] = Cf[blk]
    return Y, C


@requires_rust_periodic
@pytest.mark.parametrize("charge_caps", [False, True])
def test_hb_blocks_match_reference_exactly(charge_caps):
    rng = np.random.default_rng(1404)
    gf = rng.normal(size=(7, 3, 3)) + 1j * rng.normal(size=(7, 3, 3))
    cf = rng.normal(size=(7, 3, 3)) + 1j * rng.normal(size=(7, 3, 3))
    got = circuitopt_core.periodic_hb_blocks(
        gf, cf, 2, 3.2e6, charge_caps)
    expected = _hb_blocks_reference(gf, cf, 2, 3.2e6, charge_caps)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])


def _fold_psd_reference(adjs, freqs, K, fundamental, p_idx, q_idx,
                        sth, mfl):
    """Direct numpy statement of the cyclostationary fold.

    Thermal: Z^H S_th Z with the Toeplitz power-harmonic matrix. Flicker:
    sum_a |sum_r Z_r M_{r-a}|^2 / nu_a with nu_a = max(|f + a*f0|, 1e-9)."""
    nfreq = len(freqs)
    nsrc = p_idx.shape[0]
    nb = 2 * K + 1
    out = np.zeros(nfreq)
    dev = np.zeros((nsrc, nfreq))
    for fi, freq in enumerate(freqs):
        nu = np.abs(freq + (np.arange(nb) - K) * fundamental)
        inv_nu = 1.0 / np.maximum(nu, 1e-9)
        adj = adjs[fi]
        for si in range(nsrc):
            Z = np.zeros(nb, dtype=complex)
            for r in range(nb):
                if p_idx[si, r] >= 0:
                    Z[r] += adj[p_idx[si, r]]
                if q_idx[si, r] >= 0:
                    Z[r] -= adj[q_idx[si, r]]
            contrib = float(np.real(Z @ sth[si] @ Z.conj()))
            for a in range(nb):
                u = complex(Z @ mfl[si][2 * K - a:2 * K - a + nb])
                contrib += (u.real ** 2 + u.imag ** 2) * inv_nu[a]
            contrib = max(contrib, 0.0)
            dev[si, fi] = contrib
            out[fi] += contrib
    return out, dev


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
    expected = _fold_psd_reference(
        adjs, freqs, K, 3.2e6, p, q, thermal, flicker)
    np.testing.assert_allclose(got[0], expected[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-13, atol=1e-13)

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


def _orbit_device_quantities(device, node_wave, vg_rail, vs_rail):
    """Per-sample (gm, gds, Cgs, Cgd, root) with the kernel's warm-start order,
    computed through the scalar OtftModel bindings."""
    scalar = device._get_rust_model()
    N = node_wave.shape[0]
    out = []
    seed = None
    for m in range(N):
        Vd = float(node_wave[m, 0])
        Vg, Vs = vg_rail, vs_rail
        if seed is None:
            x0 = (Vs - 0.01 * (Vs - Vd), Vd + 0.01 * (Vs - Vd))
        else:
            x0 = seed
        ok, vs1, vd1 = scalar.newton_internal(Vs, Vd, Vg, x0[0], x0[1],
                                              1e-12, 40)
        assert ok
        seed = (vs1, vd1)
        okd, gm_neg, gds_neg = scalar.terminal_derivatives(
            Vs, Vd, Vg, vs1, vd1, True, True, False, 1e-3)
        assert okd
        gm, gds = -gm_neg, -gds_neg
        if gm < 0.0:
            gm = 0.0
        if gds < 1e-12:
            gds = 1e-12
        cgs, cgd = scalar.capacitances_pair(Vs, Vd, Vg, vs1, vd1)
        out.append((gm, gds, cgs, cgd, (vs1, vd1)))
    return out


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
def test_otft_orbit_linearization_matches_scalar_stamps(gate1):
    """The compiled orbit linearizer must equal the stamp assembly
    reconstructed from the scalar device bindings (same warm-start order).

    Non-gate1 (drain solved, gate/source rails):
        Gt[m] = gmin + gds_m,   Ct[m] = Cgd_m.
    Gate1 (state 1 = internal gate1 node):
        the R_cap/R_cap2 resistive network plus Cgs/Cgd into gate1 and the
        edge-only dC/dVd cross-coupling driven by dVd/dt."""
    device = PMOS_TFT(W=5000, L=30)
    params = _params(device)
    N = 12
    phase = 2 * np.pi * np.arange(N) / N
    node_wave = np.ascontiguousarray((20.0 + 0.2 * np.sin(phase))[:, None])
    node_dot = np.ascontiguousarray((0.2 * 2 * np.pi * np.cos(phase))[:, None])
    input_wave = np.empty((0, N))
    input_dot = np.empty((0, N))
    vd, vg, vs, sd, sg, ss = _terminal_arrays()
    gate_record = (1, float(device.R_cap), float(device.R_cap2)) if gate1 else None
    n_state = 2 if gate1 else 1
    problem = circuitopt_core.PeriodicLinearizationProblem({
        "node_count": 1, "state_count": n_state,
        "input_count": 0, "drive_count": 0,
        "devices": [
            ((0, 0, 0.0), (2, 0, 30.0), (2, 0, 40.0),
             (0, 0), (2, 0), (2, 0), params.tolist(), gate_record),
        ],
        "dense_devices": [],
        "resistors": [], "capacitors": [], "gmin": 1e-12, "fd_step": 1e-4,
    })
    empty_dense = np.empty((N, 0, 4, 4))
    got_gt, got_ct, *_rest = problem.linearize(
        node_wave, input_wave, node_dot, input_dot, empty_dense, empty_dense)

    scalar = device._get_rust_model()
    quantities = _orbit_device_quantities(device, node_wave, 30.0, 40.0)
    fd_step = 1e-4
    inv_rc = 1.0 / device.R_cap
    inv_rc2 = 1.0 / device.R_cap2

    for m, (gm, gds, cgs, cgd, root) in enumerate(quantities):
        if not gate1:
            expected_g = np.array([[1e-12 + gds]])
            expected_c = np.array([[cgd]])
            np.testing.assert_allclose(got_gt[m], expected_g, rtol=1e-12,
                                       atol=1e-18)
            np.testing.assert_allclose(got_ct[m], expected_c, rtol=1e-12,
                                       atol=1e-18)
            continue

        Vd = float(node_wave[m, 0])
        Vg, Vs = 30.0, 40.0
        expected_g = np.zeros((2, 2))
        expected_c = np.zeros((2, 2))
        expected_g[0, 0] += 1e-12                 # gmin on the solved node only
        expected_g[0, 0] += gds                   # channel to source rail
        expected_g[1, 1] += inv_rc                # gate1 <-> gate rail
        expected_g[1, 1] += inv_rc2               # source <-> gate1 leak
        expected_g[0, 0] += inv_rc2               # drain <-> gate1 leak
        expected_g[0, 1] -= inv_rc2
        expected_g[1, 1] += inv_rc2
        expected_g[1, 0] -= inv_rc2
        expected_c[1, 1] += cgs                   # Cgs to gate1 (source is rail)
        expected_c[0, 0] += cgd                   # Cgd to gate1
        expected_c[0, 1] -= cgd
        expected_c[1, 1] += cgd
        expected_c[1, 0] -= cgd

        # Edge-only cross-coupling driven by dVd/dt (dVs/dt = dVg/dt = 0).
        dvd_dt = float(node_dot[m, 0])
        denom = inv_rc + 2.0 * inv_rc2
        dvg1_dt = dvd_dt * inv_rc2 / denom
        vdot_sg1 = -dvg1_dt
        vdot_dg1 = dvd_dt - dvg1_dt
        if abs(vdot_sg1) >= 1e-30 or abs(vdot_dg1) >= 1e-30:
            # Only the Vd axis has a solved control column; re-solve internals
            # warm-seeded from the sample root, like the kernel does.
            okp, vs1p, vd1p = scalar.newton_internal(
                Vs, Vd + fd_step, Vg, root[0], root[1], 1e-12, 40)
            okm, vs1m, vd1m = scalar.newton_internal(
                Vs, Vd - fd_step, Vg, root[0], root[1], 1e-12, 40)
            assert okp and okm
            cgsp, cgdp = scalar.capacitances_pair(
                Vs, Vd + fd_step, Vg, vs1p, vd1p)
            cgsm, cgdm = scalar.capacitances_pair(
                Vs, Vd - fd_step, Vg, vs1m, vd1m)
            d_cgs = (cgsp - cgsm) / (2.0 * fd_step)
            d_cgd = (cgdp - cgdm) / (2.0 * fd_step)
            if vdot_sg1 != 0.0 and d_cgs != 0.0:
                cc = d_cgs * vdot_sg1
                # source row is a rail -> only the gate1 row stamp survives.
                expected_g[1, 0] -= cc
            if vdot_dg1 != 0.0 and d_cgd != 0.0:
                cc = d_cgd * vdot_dg1
                expected_g[0, 0] += cc
                expected_g[1, 0] -= cc

        np.testing.assert_allclose(got_gt[m], expected_g, rtol=1e-9,
                                   atol=1e-18)
        np.testing.assert_allclose(got_ct[m], expected_c, rtol=1e-12,
                                   atol=1e-24)
