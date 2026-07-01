"""Isolated parity for the Numba augmented-system (n_aug > n) branch stamping.

The Numba stamp `_stamp_transient_system_impl` must reproduce the Python
`_k_step_residual` / `_k_build_jac` branch rows/cols (vsource / VCVS / CCCS /
CCVS) bit-closely. Validated here on small *linear* circuits (resistors + an
ideal vsource + one controlled source, no PMOS devices, no caps) so R/J are
exactly computable — this is the safety net guarding the branch math before any
n_aug > n solve path is enabled.
"""
import numpy as np
import pytest

from core import numba_kernels as nk
from core import transient_solver as ts
from tests.test_controlled_sources import (
    _vcvs_topology, _cccs_topology, _ccvs_topology)


def _stamp_parity(topo, seed):
    tgrid = np.linspace(0.0, 1e-3, 3)
    marshal = ts._marshal_transient({}, {}, tgrid, topo=topo,
                                    inputs={}, node_inputs={})
    ctx = marshal.ctx
    n_aug = int(ctx.n_aug)
    assert n_aug > int(ctx.n)            # the vsource adds a branch unknown
    V = np.random.default_rng(seed).standard_normal(n_aug)
    input_now = np.zeros(0)
    h = 1e-4

    # Python reference (no devices -> empty states/history, no load caps).
    ncap = int(ctx.cap_value.shape[0])
    states = ts._k_device_states(ctx, V, input_now)
    R_py = ts._k_step_residual(ctx, V, input_now, states, [], np.zeros(ncap), h)
    J_py = ts._k_build_jac(ctx, V, states, [], h)

    # Numba stamp at the same (V, h); reuse the kernel arg groups for the bulk.
    (solver, device_terms, device_nodes, model_params, op_cache,
     passives, sources, cap_clip, vsources, vcvs, cccs, ccvs) = \
        ts._numba_shared_kernel_arg_groups(ctx)
    nd = int(ctx.dev_d_kind.shape[0])
    z, cz = np.zeros(nd), np.zeros(ncap)
    R = np.zeros(n_aug)
    J = np.zeros((n_aug, n_aug))
    prof = np.zeros(nk.PROFILE_LEN)
    nk._stamp_transient_system_impl(
        V, V, input_now, input_now, h, int(ctx.n), float(ctx.gmin), float(ctx.HH),
        *device_terms, *device_nodes, *model_params, *op_cache,
        *passives, *sources, int(cap_clip[0]),
        z, z, z, z, z, cz,
        R, J, False, prof,
        1.0, -1.0, 0.0, z, z, cz,
        vsources, vcvs, cccs, ccvs)

    assert np.allclose(R, R_py, atol=1e-10, rtol=1e-9), float(np.abs(R - R_py).max())
    assert np.allclose(J, J_py, atol=1e-10, rtol=1e-9), float(np.abs(J - J_py).max())


@pytest.mark.parametrize("seed", [0, 7])
def test_stamp_parity_vcvs(seed):
    _stamp_parity(_vcvs_topology(), seed)


@pytest.mark.parametrize("seed", [0, 7])
def test_stamp_parity_cccs(seed):
    _stamp_parity(_cccs_topology(), seed)


@pytest.mark.parametrize("seed", [0, 7])
def test_stamp_parity_ccvs(seed):
    _stamp_parity(_ccvs_topology(), seed)
