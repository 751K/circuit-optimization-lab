import inspect

from core import numba_kernels as nk
from core import transient_profile as tp
from core import transient_solver as ts


def _param_names(func):
    return tuple(inspect.signature(getattr(func, "py_func", func)).parameters)


def test_transient_profile_slots_are_dense_and_named():
    assert len(tp.TRANSIENT_PROFILE_FIELDS) == tp.PROFILE_LEN
    assert tuple(tp.PROFILE_SLOT_BY_NAME[name]
                 for name in tp.TRANSIENT_PROFILE_FIELDS) == tuple(range(tp.PROFILE_LEN))
    assert tp.PROFILE_NEWTON_ITERS == tp.PROFILE_SLOT_BY_NAME["newton_iters_total"]
    assert tp.PROFILE_FAILED_INTERVALS == tp.PROFILE_SLOT_BY_NAME["failed_intervals"]
    assert tp.PROFILE_STALLED_RESIDUAL_ACCEPTS == tp.PROFILE_LEN - 1


def test_transient_numba_arg_packers_match_kernel_signatures():
    assert ts._NUMBA_GRID_ARG_NAMES == _param_names(nk._transient_solve_grid_impl)
    assert ts._NUMBA_GRID_ARG_NAMES == _param_names(nk._transient_solve_grid_gear2_impl)
    assert ts._NUMBA_ADAPTIVE_GEAR2_ARG_NAMES == _param_names(
        nk._transient_solve_adaptive_gear2_impl)


def test_fixed_grid_numba_boundary_uses_grouped_contract():
    names = _param_names(nk._transient_solve_grid_impl)
    assert names == (
        "run",
        "step",
        "solver",
        "device_terms",
        "device_nodes",
        "model_params",
        "op_cache",
        "passives",
        "sources",
        "cap_clip",
        "vsources",
        "vcvs",
        "cccs",
        "ccvs",
    )
    assert names == _param_names(nk._transient_solve_grid_gear2_impl)
    grouped_names = tuple(name for name, _fields in ts._NUMBA_GRID_ARG_GROUPS)
    assert grouped_names == names
    assert all(len(fields) <= 18 for _name, fields in ts._NUMBA_GRID_ARG_GROUPS)


def test_adaptive_gear2_numba_boundary_uses_grouped_contract():
    names = _param_names(nk._transient_solve_adaptive_gear2_impl)
    assert names == (
        "run",
        "step",
        "solver",
        "device_terms",
        "device_nodes",
        "model_params",
        "op_cache",
        "passives",
        "sources",
        "cap_clip",
        "vsources",
        "vcvs",
        "cccs",
        "ccvs",
    )
    assert len(names) <= 16

    grouped_names = tuple(name for name, _fields in ts._NUMBA_ADAPTIVE_GEAR2_ARG_GROUPS)
    assert grouped_names == names
    assert all(len(fields) <= 18 for _name, fields in ts._NUMBA_ADAPTIVE_GEAR2_ARG_GROUPS)
