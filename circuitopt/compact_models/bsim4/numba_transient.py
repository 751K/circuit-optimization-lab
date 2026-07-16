"""Marshalling for the native BSIM4 fixed-grid Numba transient kernel."""
from __future__ import annotations

import numpy as np

from ...compiled_topology import TERM_RAIL, index_array, term_arrays
from ...numba_kernels import NUMBA_AVAILABLE, _bsim4_transient_grid_impl
from .native import Bsim4NativeError, _raise_status


def solve_bsim4_numba(
    plan,
    devices,
    x0,
    tgrid,
    input_values,
    dynamic_sources,
    *,
    method,
    newton_maxit,
    newton_vtol,
    newton_step_limit,
    gmin,
):
    """Run the reduced four-terminal BSIM4 transient entirely in Numba/C."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("native BSIM4 Numba transient requires Numba")
    wrappers = [devices[item.name] for item in plan.devices]
    if not wrappers:
        raise ValueError("native BSIM4 transient requires at least one device")
    if any(not hasattr(wrapper, "create_native_solver_handle")
           for wrapper in wrappers):
        raise NotImplementedError(
            "native BSIM4 Numba transient requires dedicated C-handle support")

    handles = [wrapper.create_native_solver_handle() for wrapper in wrappers]
    try:
        evaluator = handles[0].kernel_evaluator
        ndev = len(handles)
        terminals = np.zeros((ndev, 4), dtype=np.float64)
        currents = np.zeros((ndev, 4), dtype=np.float64)
        conductance = np.zeros((ndev, 4, 4), dtype=np.float64)
        charges = np.zeros((ndev, 4), dtype=np.float64)
        capacitance = np.zeros((ndev, 4, 4), dtype=np.float64)
        handle_a = np.asarray([handle.pointer for handle in handles], dtype=np.int64)
        terminal_ptr_a = _row_pointers(terminals)
        current_ptr_a = _row_pointers(currents)
        conductance_ptr_a = _row_pointers(conductance)
        charge_ptr_a = _row_pointers(charges)
        capacitance_ptr_a = _row_pointers(capacitance)

        term_kind2d = np.empty((ndev, 4), dtype=np.int64)
        term_ref2d = np.empty((ndev, 4), dtype=np.int64)
        term_val2d = np.empty((ndev, 4), dtype=np.float64)
        term_row2d = np.full((ndev, 4), -1, dtype=np.int64)
        for index, (item, wrapper) in enumerate(zip(plan.devices, wrappers)):
            kinds, refs, values = term_arrays(
                (item.d, item.g, item.s, (TERM_RAIL, wrapper.vb)))
            term_kind2d[index] = kinds
            term_ref2d[index] = refs
            term_val2d[index] = values
            term_row2d[index] = index_array(
                (item.di, item.gi, item.si, None))

        element_args = _element_args(plan, dynamic_sources)
        xhist, nfail, fail_idx, status = _bsim4_transient_grid_impl(
            evaluator,
            handle_a,
            terminal_ptr_a,
            current_ptr_a,
            conductance_ptr_a,
            charge_ptr_a,
            capacitance_ptr_a,
            terminals,
            currents,
            conductance,
            charges,
            capacitance,
            term_kind2d,
            term_ref2d,
            term_val2d,
            term_row2d,
            np.asarray(x0, dtype=np.float64),
            np.asarray(tgrid, dtype=np.float64),
            np.asarray(input_values, dtype=np.float64),
            1 if method in {"gear2", "bdf2"} else 0,
            *element_args,
            int(plan.n),
            int(plan.n_aug),
            int(newton_maxit),
            float(newton_vtol),
            float(newton_step_limit),
            float(gmin),
        )
        if int(status):
            _raise_status(int(status), f"Numba transient at step {int(fail_idx)}")
        if int(nfail) < 0:
            raise Bsim4NativeError("native BSIM4 Numba transient evaluation failed")
        return np.asarray(xhist), int(nfail), int(fail_idx)
    finally:
        for handle in handles:
            handle.close()


def _row_pointers(array):
    row_bytes = int(array.strides[0])
    return np.asarray(
        [array.ctypes.data + index * row_bytes for index in range(array.shape[0])],
        dtype=np.int64,
    )


def _element_args(plan, dynamic_sources):
    resistors = [(item.a, item.b, item.ai, item.bi, item.g)
                 for item in plan.resistors]
    res_a_kind, res_a_ref, res_a_val = term_arrays([row[0] for row in resistors])
    res_b_kind, res_b_ref, res_b_val = term_arrays([row[1] for row in resistors])
    res_ai = index_array(row[2] for row in resistors)
    res_bi = index_array(row[3] for row in resistors)
    res_g = np.asarray([row[4] for row in resistors], dtype=np.float64)

    capacitors = [(item.a, item.b, item.ai, item.bi, item.value)
                  for item in plan.capacitors]
    cap_a_kind, cap_a_ref, cap_a_val = term_arrays(
        [row[0] for row in capacitors])
    cap_b_kind, cap_b_ref, cap_b_val = term_arrays(
        [row[1] for row in capacitors])
    cap_ai = index_array(row[2] for row in capacitors)
    cap_bi = index_array(row[3] for row in capacitors)
    cap_value = np.asarray([row[4] for row in capacitors], dtype=np.float64)

    isrc_pi = index_array(item.pi for item in plan.isources)
    isrc_qi = index_array(item.qi for item in plan.isources)
    isrc_value = np.asarray(
        [item.value for item in plan.isources], dtype=np.float64)
    dyn_pi = index_array(row[0] for row in dynamic_sources)
    dyn_qi = index_array(row[1] for row in dynamic_sources)
    dyn_idx = np.asarray([row[2] for row in dynamic_sources], dtype=np.int64)

    vs_pi = index_array(item.pi for item in plan.vsources)
    vs_qi = index_array(item.qi for item in plan.vsources)
    vs_bi = index_array(item.bi for item in plan.vsources)
    vs_e_const = np.asarray(
        [item.e_const for item in plan.vsources], dtype=np.float64)
    vs_e_idx = index_array(item.e_input_idx for item in plan.vsources)

    vccs_pi = index_array(item.pi for item in plan.vccs)
    vccs_qi = index_array(item.qi for item in plan.vccs)
    vccs_cp_kind, vccs_cp_ref, vccs_cp_val = term_arrays(
        [item.cp for item in plan.vccs])
    vccs_cn_kind, vccs_cn_ref, vccs_cn_val = term_arrays(
        [item.cn for item in plan.vccs])
    vccs_gm = np.asarray([item.gm for item in plan.vccs], dtype=np.float64)

    vcvs_pi = index_array(item.pi for item in plan.vcvs)
    vcvs_qi = index_array(item.qi for item in plan.vcvs)
    vcvs_bi = index_array(item.bi for item in plan.vcvs)
    vcvs_p_kind, vcvs_p_ref, vcvs_p_val = term_arrays(
        [item.p for item in plan.vcvs])
    vcvs_q_kind, vcvs_q_ref, vcvs_q_val = term_arrays(
        [item.q for item in plan.vcvs])
    vcvs_cp_kind, vcvs_cp_ref, vcvs_cp_val = term_arrays(
        [item.cp for item in plan.vcvs])
    vcvs_cn_kind, vcvs_cn_ref, vcvs_cn_val = term_arrays(
        [item.cn for item in plan.vcvs])
    vcvs_mu = np.asarray([item.mu for item in plan.vcvs], dtype=np.float64)

    cccs_pi = index_array(item.pi for item in plan.cccs)
    cccs_qi = index_array(item.qi for item in plan.cccs)
    cccs_ctrl_bi = index_array(item.ctrl_bi for item in plan.cccs)
    cccs_beta = np.asarray(
        [item.beta for item in plan.cccs], dtype=np.float64)

    ccvs_pi = index_array(item.pi for item in plan.ccvs)
    ccvs_qi = index_array(item.qi for item in plan.ccvs)
    ccvs_bi = index_array(item.bi for item in plan.ccvs)
    ccvs_p_kind, ccvs_p_ref, ccvs_p_val = term_arrays(
        [item.p for item in plan.ccvs])
    ccvs_q_kind, ccvs_q_ref, ccvs_q_val = term_arrays(
        [item.q for item in plan.ccvs])
    ccvs_ctrl_bi = index_array(item.ctrl_bi for item in plan.ccvs)
    ccvs_gamma = np.asarray(
        [item.gamma for item in plan.ccvs], dtype=np.float64)

    return (
        res_a_kind, res_a_ref, res_a_val,
        res_b_kind, res_b_ref, res_b_val, res_ai, res_bi, res_g,
        cap_a_kind, cap_a_ref, cap_a_val,
        cap_b_kind, cap_b_ref, cap_b_val, cap_ai, cap_bi, cap_value,
        isrc_pi, isrc_qi, isrc_value, dyn_pi, dyn_qi, dyn_idx,
        vs_pi, vs_qi, vs_bi, vs_e_const, vs_e_idx,
        vccs_pi, vccs_qi,
        vccs_cp_kind, vccs_cp_ref, vccs_cp_val,
        vccs_cn_kind, vccs_cn_ref, vccs_cn_val, vccs_gm,
        vcvs_pi, vcvs_qi, vcvs_bi,
        vcvs_p_kind, vcvs_p_ref, vcvs_p_val,
        vcvs_q_kind, vcvs_q_ref, vcvs_q_val,
        vcvs_cp_kind, vcvs_cp_ref, vcvs_cp_val,
        vcvs_cn_kind, vcvs_cn_ref, vcvs_cn_val, vcvs_mu,
        cccs_pi, cccs_qi, cccs_ctrl_bi, cccs_beta,
        ccvs_pi, ccvs_qi, ccvs_bi,
        ccvs_p_kind, ccvs_p_ref, ccvs_p_val,
        ccvs_q_kind, ccvs_q_ref, ccvs_q_val,
        ccvs_ctrl_bi, ccvs_gamma,
    )
