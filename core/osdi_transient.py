"""Silicon (OSDI) transient — Phase B.

Two layers live here:

* :func:`cs_transient` — the original pure-Python backward-Euler single-stage
  demo (kept as a readable correctness reference).
* :func:`transient_osdi` — the full-fidelity path: marshals a compiled
  topology whose transistors are OSDI devices into
  :func:`core.numba_kernels._osdi_transient_grid_impl`, a numba fixed-grid
  BE/gear2 integrator that calls the compiled ``.osdi`` **inside** the
  nopython loop (fn pointers passed as runtime arguments — the old "OSDI can't
  live in the numba loop" assumption was disproven, see the
  ``osdi-host-perf`` memory). Formulation is a *global fold*: the unknown
  vector is [external solved nodes + ideal-vsource branch currents] ++ every
  device's internal nodes, so internal-node charge dynamics are integrated
  exactly (no quasi-static reduction).

``core.transient_solver.transient`` routes here when the circuit binds OSDI
device models; the OTFT numba kernels are untouched (byte-gate safe).
"""
from __future__ import annotations

import ctypes as C

import numpy as np
from scipy.optimize import brentq

from . import diagnostics
from .numba_kernels import (NUMBA_AVAILABLE, _osdi_transient_adaptive_impl,
                            _osdi_transient_grid_impl)
from .osdi_host import (ANALYSIS_TRAN, CALC_REACT_JACOBIAN,
                        CALC_REACT_RESIDUAL, CALC_RESIST_JACOBIAN,
                        CALC_RESIST_RESIDUAL, Device, OsdiSimInfo, _U32_MAX)

_TRAN_FLAGS = (CALC_RESIST_RESIDUAL | CALC_RESIST_JACOBIAN |
               CALC_REACT_RESIDUAL | CALC_REACT_JACOBIAN | ANALYSIS_TRAN)


def cs_transient(dev, vdd, r_load, c_load, vin, tgrid, *, vmin=1e-4):
    """Backward-Euler ``vout(t)`` of a common-source stage (pure-Python demo).

    Circuit: ``dev`` source at ``vdd``, drain = ``vout``; ``r_load`` and ``c_load``
    from ``vout`` to ground; gate driven by ``vin(t)``.  ``dev`` is an
    :class:`~core.osdi_device.OsdiDevice` biased with its bulk at ``vdd`` (pmos-style,
    so the device sources ``+|Id|`` into the drain node — matching the DC KCL sign).

    KCL at ``vout``:  ``|Id| - dQd/dt - vout/RL - CL·dvout/dt = 0`` (device drain charge
    ``Qd`` stored on the node → ``-dQd/dt`` into it).  Solved per step with a bracketed
    root find (no Jacobian needed).  Returns ``vout`` sampled on ``tgrid``.
    """
    vdd = float(vdd)
    t = np.asarray(tgrid, dtype=float)
    vout = np.zeros(len(t))
    hi = vdd - vmin

    def node(vg, v):
        Id, Qd = dev.id_and_drain_charge(vdd, v, vg)   # (Vs=vdd, Vd=v, Vg=vg)
        return abs(Id), Qd

    # DC initial condition at t0 (no dQ/dt term): |Id| - vout/RL = 0
    vg = float(vin(t[0]))
    vout[0] = brentq(lambda v: node(vg, v)[0] - v / r_load, vmin, hi)
    _, q_prev = node(vg, vout[0])

    for k in range(1, len(t)):
        h = t[k] - t[k - 1]
        vg = float(vin(t[k]))
        vp = vout[k - 1]

        def residual(v, _vp=vp, _vg=vg, _h=h):
            idev, qd = node(_vg, v)
            return idev - (qd - q_prev) / _h - v / r_load - c_load * (v - _vp) / _h

        vout[k] = brentq(residual, vmin, hi)
        _, q_prev = node(vg, vout[k])
    return vout


# ── full-fidelity marshal (Phase B) ──────────────────────────────────────

def _term_triple(term):
    """(kind, payload) topology term → (kind, ref, value) scalars."""
    kind = int(term[0])
    if kind in (0, 1):
        return kind, int(term[1]), 0.0
    return kind, 0, float(term[1])


def _bind_row_buffers(dev, i, v2d, jac2d, react2d):
    """Repoint a *dedicated* host Device's storage at shared 2D-buffer rows.

    The model then reads node voltages from ``v2d[i]`` (via the returned
    OsdiSimInfo's ``prev_solve``) and stamps its resistive/reactive Jacobians
    into ``jac2d[i]`` / ``react2d[i]``. After this, the Device's own
    ``_jac_np`` views are stale — the Device must not be used through the
    normal host API again (the transient context owns it).
    """
    d = dev._d
    n_jac = dev._n_jac
    jac_base = jac2d.ctypes.data + i * jac2d.shape[1] * 8
    slots = (C.c_void_p * n_jac).from_address(
        dev._inst.value + int(d.jacobian_ptr_resist_offset))
    for k in range(n_jac):
        slots[k] = jac_base + 8 * k
    react_base = react2d.ctypes.data + i * react2d.shape[1] * 8
    for k in range(n_jac):
        off = int(d.jacobian_entries[k].react_ptr_off)
        if off != _U32_MAX:
            C.c_void_p.from_address(dev._inst.value + off).value = \
                react_base + 8 * k
    v_addr = v2d.ctypes.data + i * v2d.shape[1] * 8
    return OsdiSimInfo(
        paras=dev._sp, abstime=0.0,
        prev_solve=C.cast(v_addr, C.POINTER(C.c_double)),
        prev_state=None, next_state=None, flags=_TRAN_FLAGS,
        history_ctx=None, query_past_state=None)


def transient_osdi(sizes, bias, tgrid, *, topo, nf=None, V0=None,
                   inputs=None, node_inputs=None, current_inputs=None,
                   corner=None, model_types=None, device_kwargs=None,
                   integration_method="be", newton_maxit=30,
                   newton_vtol=1e-8, newton_step_limit=5.0, gmin=1e-12,
                   adaptive=False, adaptive_reltol=1e-4, adaptive_vabstol=1e-6,
                   adaptive_iabstol=1e-12, adaptive_max_steps=200000):
    """Fixed-grid transient of an OSDI-device circuit (silicon Phase B).

    Same result-dict shape as :func:`core.transient_solver.transient` for the
    common fields (t / nodes / output / vout / nfail). Supports transistors
    (all bound to the *same* compiled ``.osdi``), resistors, capacitors, ideal
    current/voltage sources, controlled sources (VCCS/VCVS/CCCS/CCVS), and
    input-driven nodes/gates/current sources.
    """
    from .ac_solver import ac_solve
    from .device_factory import build_devices
    from .compiled_topology import CompiledTopology, index_array, term_arrays
    from .osdi_device import OsdiDevice, load_model

    method = str(integration_method).lower()
    if method not in ("be", "gear2"):
        raise ValueError(f"integration_method must be 'be' or 'gear2', got {method!r}")
    tgrid = np.asarray(tgrid, float)
    N = len(tgrid)

    inputs = {key: np.asarray(val, float) for key, val in (inputs or {}).items()}
    for key, val in inputs.items():
        if len(val) != N:
            raise ValueError(f"Input waveform {key!r} length {len(val)} != len(tgrid) {N}")
    input_keys = tuple(inputs)
    input_values = (np.vstack([inputs[key] for key in input_keys])
                    if input_keys else np.empty((0, N), float))
    node_inputs = dict(node_inputs or {})
    for node, key in node_inputs.items():
        if key not in inputs:
            raise ValueError(f"node_inputs[{node!r}] references missing waveform {key!r}")

    plan = CompiledTopology(topo, bias, input_keys=input_keys,
                            node_inputs=node_inputs, transient_inputs=True)
    if not plan.devices:
        raise ValueError("transient_osdi needs at least one transistor")

    wrappers = build_devices(sizes, nf=nf, corner=corner, topo=topo,
                             model_types=model_types, device_kwargs=device_kwargs)
    wlist = [wrappers[item.name] for item in plan.devices]
    for w in wlist:
        if not isinstance(w, OsdiDevice):
            raise NotImplementedError(
                "transient_osdi requires every transistor to be an OSDI device; "
                f"got {type(w).__name__} (mixed OTFT/silicon circuits are not supported)")
    # group devices by compiled model: the kernel carries two fn-pointer sets,
    # so up to two distinct .osdi libraries can mix in one circuit (e.g. a
    # BSIM4 amplifier with a BSIM3 auxiliary device)
    lib_keys: list = []
    lib_of = []
    for w in wlist:
        key = (w.VA_PATH, w.MODULE)
        if key not in lib_keys:
            lib_keys.append(key)
        lib_of.append(lib_keys.index(key))
    if len(lib_keys) > 2:
        raise NotImplementedError(
            "silicon transient supports at most two distinct compiled models "
            f"per circuit; got {len(lib_keys)}")
    lib_a = np.array(lib_of, dtype=np.int64)

    # ── initial condition (external nodes) ───────────────────────────────
    n, n_aug = plan.n, plan.n_aug
    if V0 is None:
        ac = ac_solve(sizes, bias, np.array([1.0]), nf=nf, topo=topo, corner=corner,
                      model_types=model_types, device_kwargs=device_kwargs)
        if ac is None:
            raise RuntimeError("transient_osdi: DC solve failed for the initial point")
        dc = ac["dc_op"]
        V0 = np.array([dc[name] for name in topo.solved])
    V0 = np.asarray(V0, float)
    if V0.shape[0] < n_aug:
        V0 = np.concatenate([V0, np.zeros(n_aug - V0.shape[0])])

    # ── dedicated host Devices (fresh, NOT the shared LRU instances) ─────
    ndev = len(wlist)
    libs = [load_model(path) for path, _ in lib_keys]
    devs = [Device(libs[lib_of[i]], w._osdi_card, model_name=w.MODULE,
                   temperature=w._osdi_temperature) for i, w in enumerate(wlist)]
    T = devs[0].n_term
    if any(dev.n_term != T for dev in devs):
        raise NotImplementedError(
            "all compiled models in one transient must have the same terminal count")
    # buffer widths are the max across libs; per-device wiring/scatter below
    # uses each device's own sizes, padding stays inert (-1 indices / zeros)
    n_nodes = max(dev.n_nodes for dev in devs)
    n_jac = max(dev._n_jac for dev in devs)
    # one representative device per lib supplies that lib's fn pointers
    rep = [devs[lib_of.index(k)] for k in range(len(lib_keys))]
    if len(rep) == 1:
        rep = [rep[0], rep[0]]
    input0 = input_values[:, 0] if input_values.size else np.zeros(0)

    def _term_t0(term):
        kind, ref, val = _term_triple(term)
        if kind == 0:
            return float(V0[ref])
        if kind == 1:
            return float(input0[ref])
        return val

    # solve each device's internals at the DC terminals BEFORE rebinding
    internals = []
    for item, w, dev in zip(plan.devices, wlist, devs):
        vd, vg, vs = _term_t0(item.d), _term_t0(item.g), _term_t0(item.s)
        dev.operating_point(vd, vg, vs, w.vb, with_caps=False)
        internals.append(dev._last_redv[T:].copy())

    # ── shared row buffers + rebinding ────────────────────────────────────
    v2d = np.zeros((ndev, n_nodes))
    residR2d = np.zeros((ndev, n_nodes))
    residQ2d = np.zeros((ndev, n_nodes))
    jac2d = np.zeros((ndev, n_jac))
    react2d = np.zeros((ndev, n_jac))
    infos = [_bind_row_buffers(dev, i, v2d, jac2d, react2d)
             for i, dev in enumerate(devs)]
    inst_a = np.array([dev._inst.value for dev in devs], dtype=np.int64)
    model_a = np.array([dev._model.value for dev in devs], dtype=np.int64)
    info_a = np.array([C.addressof(info) for info in infos], dtype=np.int64)
    residR_ptr_a = np.array([residR2d.ctypes.data + i * n_nodes * 8
                             for i in range(ndev)], dtype=np.int64)
    residQ_ptr_a = np.array([residQ2d.ctypes.data + i * n_nodes * 8
                             for i in range(ndev)], dtype=np.int64)

    # ── global unknown layout + precomputed scatter indices ──────────────
    nred_a = np.array([dev._n_red for dev in devs], dtype=np.int64)
    int_base_a = np.zeros(ndev, dtype=np.int64)
    base = n_aug
    for i, dev in enumerate(devs):
        int_base_a[i] = base
        base += dev._n_red - T
    ntot = base

    term_kind2d = np.zeros((ndev, T), dtype=np.int64)
    term_ref2d = np.zeros((ndev, T), dtype=np.int64)
    term_val2d = np.zeros((ndev, T), dtype=float)
    nmap2d = np.full((ndev, n_nodes), -1, dtype=np.int64)
    node_grow2d = np.full((ndev, n_nodes), -1, dtype=np.int64)
    jacR_gflat2d = np.full((ndev, n_jac), -1, dtype=np.int64)
    jacQ_gflat2d = np.full((ndev, n_jac), -1, dtype=np.int64)
    for i, (item, w, dev) in enumerate(zip(plan.devices, wlist, devs)):
        terms = (item.d, item.g, item.s, (2, w.vb))
        rows = (item.di, item.gi, item.si, None)
        for r, term in enumerate(terms):
            term_kind2d[i, r], term_ref2d[i, r], term_val2d[i, r] = _term_triple(term)
        nred = dev._n_red
        grow = np.full(nred, -1, dtype=np.int64)
        for r in range(nred):
            if r < T:
                grow[r] = -1 if rows[r] is None else int(rows[r])
            else:
                grow[r] = int_base_a[i] + (r - T)
        for j in range(dev.n_nodes):
            m = dev._nmap[j]
            if m != _U32_MAX:
                nmap2d[i, j] = m
                node_grow2d[i, j] = grow[m]
        for k in range(dev._n_jac):
            e = dev._jac_entries[k]
            row_m = dev._nmap[e.nodes.node_1]
            col_m = dev._nmap[e.nodes.node_2]
            if row_m == _U32_MAX or col_m == _U32_MAX:
                continue
            gr, gc = grow[row_m], grow[col_m]
            if gr < 0 or gc < 0:
                continue
            jacR_gflat2d[i, k] = gr * ntot + gc
            if int(e.react_ptr_off) != _U32_MAX:
                jacQ_gflat2d[i, k] = gr * ntot + gc

    X0 = np.zeros(ntot)
    X0[:n_aug] = V0[:n_aug]
    for i in range(ndev):
        ni = nred_a[i] - T
        X0[int_base_a[i]:int_base_a[i] + ni] = internals[i]

    # ── passive element arrays (mirrors _marshal_transient's subset) ─────
    res_meta = [(item.a, item.b, item.ai, item.bi, item.g)
                for item in plan.resistors]
    res_a_kind, res_a_ref, res_a_val = term_arrays([m[0] for m in res_meta])
    res_b_kind, res_b_ref, res_b_val = term_arrays([m[1] for m in res_meta])
    res_ai = index_array(m[2] for m in res_meta)
    res_bi = index_array(m[3] for m in res_meta)
    res_g = np.array([m[4] for m in res_meta], dtype=float)

    cap_meta = [(item.a, item.b, item.ai, item.bi, item.value)
                for item in plan.capacitors]
    cap_a_kind, cap_a_ref, cap_a_val = term_arrays([m[0] for m in cap_meta])
    cap_b_kind, cap_b_ref, cap_b_val = term_arrays([m[1] for m in cap_meta])
    cap_ai = index_array(m[2] for m in cap_meta)
    cap_bi = index_array(m[3] for m in cap_meta)
    cap_value = np.array([m[4] for m in cap_meta], dtype=float)

    isrc_pi = index_array(item.pi for item in plan.isources)
    isrc_qi = index_array(item.qi for item in plan.isources)
    isrc_value = np.array([item.value for item in plan.isources], dtype=float)

    dyn_meta = []
    for pos, entry in enumerate(current_inputs or ()):
        if isinstance(entry, dict):
            p_node, q_node, key = entry["p"], entry["q"], entry["input"]
        else:
            p_node, q_node, key = entry
        if key not in plan.input_index:
            raise ValueError(f"current_inputs[{pos}] references missing waveform {key!r}")
        dyn_meta.append((plan.solved_index(plan.compile_term(p_node)),
                         plan.solved_index(plan.compile_term(q_node)),
                         plan.input_index[key]))
    dyn_pi = index_array(m[0] for m in dyn_meta)
    dyn_qi = index_array(m[1] for m in dyn_meta)
    dyn_idx = np.array([m[2] for m in dyn_meta], dtype=np.int64)

    vs_pi = index_array(item.pi for item in plan.vsources)
    vs_qi = index_array(item.qi for item in plan.vsources)
    vs_bi = index_array(item.bi for item in plan.vsources)
    vs_e_const = np.array([item.e_const for item in plan.vsources], dtype=float)
    vs_e_idx = index_array(item.e_input_idx for item in plan.vsources)

    # controlled sources (same stamping conventions as CompiledTopology.dc_residuals)
    vccs_pi = index_array(item.pi for item in plan.vccs)
    vccs_qi = index_array(item.qi for item in plan.vccs)
    vccs_cp_kind, vccs_cp_ref, vccs_cp_val = term_arrays([i.cp for i in plan.vccs])
    vccs_cn_kind, vccs_cn_ref, vccs_cn_val = term_arrays([i.cn for i in plan.vccs])
    vccs_gm = np.array([item.gm for item in plan.vccs], dtype=float)

    vcvs_pi = index_array(item.pi for item in plan.vcvs)
    vcvs_qi = index_array(item.qi for item in plan.vcvs)
    vcvs_bi = index_array(item.bi for item in plan.vcvs)
    vcvs_p_kind, vcvs_p_ref, vcvs_p_val = term_arrays([i.p for i in plan.vcvs])
    vcvs_q_kind, vcvs_q_ref, vcvs_q_val = term_arrays([i.q for i in plan.vcvs])
    vcvs_cp_kind, vcvs_cp_ref, vcvs_cp_val = term_arrays([i.cp for i in plan.vcvs])
    vcvs_cn_kind, vcvs_cn_ref, vcvs_cn_val = term_arrays([i.cn for i in plan.vcvs])
    vcvs_mu = np.array([item.mu for item in plan.vcvs], dtype=float)

    cccs_pi = index_array(item.pi for item in plan.cccs)
    cccs_qi = index_array(item.qi for item in plan.cccs)
    cccs_ctrl_bi = index_array(item.ctrl_bi for item in plan.cccs)
    cccs_beta = np.array([item.beta for item in plan.cccs], dtype=float)

    ccvs_pi = index_array(item.pi for item in plan.ccvs)
    ccvs_qi = index_array(item.qi for item in plan.ccvs)
    ccvs_bi = index_array(item.bi for item in plan.ccvs)
    ccvs_p_kind, ccvs_p_ref, ccvs_p_val = term_arrays([i.p for i in plan.ccvs])
    ccvs_q_kind, ccvs_q_ref, ccvs_q_val = term_arrays([i.q for i in plan.ccvs])
    ccvs_ctrl_bi = index_array(item.ctrl_bi for item in plan.ccvs)
    ccvs_gamma = np.array([item.gamma for item in plan.ccvs], dtype=float)

    fn_args = (
        rep[0]._k_eval, rep[0]._k_load_resid, rep[0]._k_load_jac,
        rep[0]._k_load_resid_react, rep[0]._k_load_jac_react,
        rep[0]._handle_p.value,
        rep[1]._k_eval, rep[1]._k_load_resid, rep[1]._k_load_jac,
        rep[1]._k_load_resid_react, rep[1]._k_load_jac_react,
        rep[1]._handle_p.value, lib_a,
        inst_a, model_a, info_a, residR_ptr_a, residQ_ptr_a,
        v2d, residR2d, residQ2d, jac2d, react2d,
        nmap2d, term_kind2d, term_ref2d, term_val2d,
        nred_a, int_base_a, node_grow2d, jacR_gflat2d, jacQ_gflat2d)
    elem_args = (
        res_a_kind, res_a_ref, res_a_val, res_b_kind, res_b_ref, res_b_val,
        res_ai, res_bi, res_g,
        cap_a_kind, cap_a_ref, cap_a_val, cap_b_kind, cap_b_ref, cap_b_val,
        cap_ai, cap_bi, cap_value,
        isrc_pi, isrc_qi, isrc_value,
        dyn_pi, dyn_qi, dyn_idx,
        vs_pi, vs_qi, vs_bi, vs_e_const, vs_e_idx,
        vccs_pi, vccs_qi, vccs_cp_kind, vccs_cp_ref, vccs_cp_val,
        vccs_cn_kind, vccs_cn_ref, vccs_cn_val, vccs_gm,
        vcvs_pi, vcvs_qi, vcvs_bi,
        vcvs_p_kind, vcvs_p_ref, vcvs_p_val, vcvs_q_kind, vcvs_q_ref, vcvs_q_val,
        vcvs_cp_kind, vcvs_cp_ref, vcvs_cp_val,
        vcvs_cn_kind, vcvs_cn_ref, vcvs_cn_val, vcvs_mu,
        cccs_pi, cccs_qi, cccs_ctrl_bi, cccs_beta,
        ccvs_pi, ccvs_qi, ccvs_bi,
        ccvs_p_kind, ccvs_p_ref, ccvs_p_val, ccvs_q_kind, ccvs_q_ref, ccvs_q_val,
        ccvs_ctrl_bi, ccvs_gamma)
    total_substeps = None
    if adaptive:
        Xhist, nfail, fail_idx, total_substeps = _osdi_transient_adaptive_impl(
            *fn_args, X0, tgrid, input_values, *elem_args,
            n, n_aug, T,
            int(newton_maxit), float(newton_vtol), float(newton_step_limit),
            float(gmin), float(adaptive_reltol), float(adaptive_vabstol),
            float(adaptive_iabstol), int(adaptive_max_steps))
    else:
        Xhist, nfail, fail_idx = _osdi_transient_grid_impl(
            *fn_args, X0, tgrid, input_values,
            1 if method == "gear2" else 0, *elem_args,
            n, n_aug, T,
            int(newton_maxit), float(newton_vtol), float(newton_step_limit),
            float(gmin))
    if nfail < 0:
        raise RuntimeError("OSDI eval returned $fatal during transient")
    if nfail:
        diagnostics.note("osdi_transient.newton_fail",
                         detail=f"{nfail} step(s) not converged, first at {fail_idx}")

    nodes = {nm: Xhist[:, plan.idx[nm]] for nm in plan.solved}
    out = np.zeros(N)
    for node, weight in plan.output_weights.items():
        out += weight * nodes[node]
    result = {"t": tgrid, "output": out, "vout": out, "nodes": nodes,
              "nfail": int(nfail), "nretry": 0,
              "nsubsteps": int(total_substeps) if total_substeps is not None else 0,
              "numba_grid_solver": bool(NUMBA_AVAILABLE),
              "osdi_transient": True,
              "integration_method": "adaptive_bdf2" if adaptive else method,
              "X_final": Xhist[-1].copy()}
    if adaptive:
        result["adaptive"] = True
        result["adaptive_reltol"] = float(adaptive_reltol)
        result["adaptive_vabstol"] = float(adaptive_vabstol)
        result["adaptive_iabstol"] = float(adaptive_iabstol)
    for legacy in ("VOP", "VON"):
        if legacy in nodes:
            result[legacy.lower()] = nodes[legacy]
    # the dedicated Devices/infos/buffers must outlive the kernel call only;
    # keep a reference on the result for debugging/chaining
    result["_osdi_ctx"] = (devs, infos, v2d, residR2d, residQ2d, jac2d, react2d)
    return result
