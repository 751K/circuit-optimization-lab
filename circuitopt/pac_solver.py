"""Generic PSS-assisted periodic AC solver.

The solver operates on an already-computed PSS orbit.  Circuit-specific wrappers
only need to provide the periodic operating point and the small-signal drive
definition; the finite-difference shooting PAC kernel is topology independent.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from scipy import sparse as _sp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import linalg as _spla

from .ac_mna import stamp_adm, stamp_dense_lti, stamp_mos_lti, branch_incidence
from .ac_solver import bw_from_gain, ac_solve
from .compiled_topology import term_arrays
from .device_factory import (
    apply_silicon_corner,
    build_devices,
    dev_corner,
    dev_nf,
    get_ss_params,
    resolve_binding,
)
from .numba_kernels import (_pnoise_hb_blocks_impl, pac_hb_blocks_numba,
                            pac_linearize_orbit_numba,
                            pac_linearize_orbit_gate1_numba, py_impl)
from .topology import Topology
from .transient_solver import transient
from . import diagnostics

if TYPE_CHECKING:
    from .device_factory import CircuitBinding


def _periodic_average(t, values):
    t = np.asarray(t, float)
    values = np.asarray(values)
    period = float(t[-1] - t[0])
    if period <= 0.0:
        return np.mean(values, axis=0)
    return np.trapezoid(values, t, axis=0) / period


_PAC_TD_GROWTH_LIMIT = 1e120


def _max_abs_finite(x):
    if x.size == 0:
        return 0.0
    value = float(np.max(np.abs(x)))
    return value if np.isfinite(value) else np.inf


def _gear2_step_sequence(n_steps):
    return list(range(1, int(n_steps))) + [0]


def _build_gear2_chunk_maps(M, *, chunk_steps=16, growth_limit=_PAC_TD_GROWTH_LIMIT):
    """Condense gear2 one-step maps into short multiple-shooting chunks.

    Directly multiplying all companion matrices can overflow even when the
    quasi-periodic boundary solve is well behaved.  Short chunks keep each
    condensed map finite; the cyclic boundary condition is solved as a sparse
    block system per PAC frequency.
    """
    n_steps = int(M.shape[0])
    sysdim = int(M.shape[1])
    sequence = _gear2_step_sequence(n_steps)
    step = max(1, min(int(chunk_steps), max(1, n_steps)))
    while step >= 1:
        chunks = [sequence[i:i + step] for i in range(0, len(sequence), step)]
        A_chunks = []
        ok = True
        for chunk in chunks:
            A = np.eye(sysdim, dtype=complex)
            for k in chunk:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    A = M[k] @ A
                if (not np.all(np.isfinite(A))) or _max_abs_finite(A) > growth_limit:
                    ok = False
                    break
            if not ok:
                break
            A_chunks.append(A)
        if ok:
            return chunks, A_chunks, step
        if step == 1:
            break
        step = max(1, step // 2)
    raise FloatingPointError("gear2 PAC chunk map overflow")


def _gear2_chunk_forcing(M, s, chunks, *, growth_limit=_PAC_TD_GROWTH_LIMIT):
    sysdim = int(M.shape[1])
    b_chunks = []
    for chunk in chunks:
        b = np.zeros(sysdim, dtype=complex)
        for k in chunk:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                b = M[k] @ b + s[k]
            if (not np.all(np.isfinite(b))) or _max_abs_finite(b) > growth_limit:
                raise FloatingPointError("gear2 PAC chunk forcing overflow")
        b_chunks.append(b)
    return b_chunks


def _gear2_boundary_template(A_chunks):
    count = len(A_chunks)
    if count <= 1:
        return None
    sysdim = int(A_chunks[0].shape[0])
    eye = np.eye(sysdim, dtype=complex)
    data = []
    indices = []
    indptr = [0]
    corner_pos = -1
    for c, A in enumerate(A_chunks):
        if c + 1 < count:
            indices.extend([c, c + 1])
            data.extend([-np.asarray(A, complex), eye])
        else:
            indices.extend([0, c])
            corner_pos = len(data)
            data.extend([eye, -np.asarray(A, complex)])
        indptr.append(len(indices))
    return {
        "data": np.asarray(data, dtype=complex),
        "indices": np.asarray(indices, dtype=np.int32),
        "indptr": np.asarray(indptr, dtype=np.int32),
        "corner_pos": int(corner_pos),
        "eye": eye,
        "shape": (count * sysdim, count * sysdim),
    }


def _solve_gear2_chunk_boundary(M, s, gamma, chunks, A_chunks,
                                boundary_template=None):
    """Solve ``z(T)=gamma*z(0)`` without forming the full monodromy product."""
    count = len(chunks)
    sysdim = int(M.shape[1])
    b_chunks = _gear2_chunk_forcing(M, s, chunks)
    rhs = np.concatenate(b_chunks)
    if count == 1:
        mat = complex(gamma) * np.eye(sysdim, dtype=complex) - A_chunks[0]
        sol = np.linalg.solve(mat, rhs)
    elif boundary_template is not None:
        data = boundary_template["data"].copy()
        data[boundary_template["corner_pos"]] = complex(gamma) * boundary_template["eye"]
        mat = _sp.bsr_matrix(
            (data, boundary_template["indices"], boundary_template["indptr"]),
            shape=boundary_template["shape"],
        ).tocsc()
        sol = _spla.spsolve(mat, rhs)
    else:
        eye = np.eye(sysdim, dtype=complex)
        rows = []
        for c, A in enumerate(A_chunks):
            row = [None] * count
            row[c] = -A
            if c + 1 < count:
                row[c + 1] = eye
            else:
                row[0] = complex(gamma) * eye
            rows.append(row)
        mat = _sp.bmat(rows, format="csc", dtype=complex)
        sol = _spla.spsolve(mat, rhs)
    if not np.all(np.isfinite(sol)):
        dense_mat = mat.toarray() if hasattr(mat, "toarray") else mat
        sol = np.linalg.lstsq(dense_mat, rhs, rcond=None)[0]
    return np.asarray(sol[:sysdim], dtype=complex)


def _merge_sizes_and_nf(sizes, nf, pss_result):
    all_sizes = dict(pss_result.get("all_sizes", sizes))
    all_sizes.update(pss_result.get("switch_sizes", {}))

    if "all_nf" in pss_result and nf is None:
        all_nf = pss_result["all_nf"]
    else:
        if isinstance(nf, dict):
            all_nf = dict(nf)
        elif nf is None:
            all_nf = {}
        else:
            all_nf = {"__global__": nf}
        if "__global__" in all_nf:
            global_nf = all_nf.pop("__global__")
            all_nf = {name: global_nf for name in all_sizes}
        all_nf.update(pss_result.get("switch_nf", {}))
    if not all_nf:
        all_nf = None
    return all_sizes, all_nf


def _input_quadratures(input_drive, tgrid, omega):
    """Real quadrature waveforms q0/q90 such that q0 + j*q90 = a*exp(jwt)."""
    cos_w = np.cos(omega * tgrid)
    sin_w = np.sin(omega * tgrid)
    q0 = {}
    q90 = {}
    for key, amp in input_drive.items():
        amp = complex(amp)
        q0[key] = amp.real * cos_w - amp.imag * sin_w
        q90[key] = amp.real * sin_w + amp.imag * cos_w
    return q0, q90


def _freeze_complex_map(values):
    return tuple(
        (str(key), float(complex(val).real), float(complex(val).imag))
        for key, val in sorted(values.items())
    )


def _freeze_sizes(sizes):
    try:
        return tuple(
            (str(key), float(value[0]), float(value[1]))
            for key, value in sorted(sizes.items())
        )
    except Exception as exc:
        diagnostics.note("pac.freeze_sizes_fail", exc)
        return repr(sizes)


def _freeze_nf(nf):
    if isinstance(nf, dict):
        return tuple((str(k), int(v)) for k, v in sorted(nf.items()))
    return None if nf is None else int(nf)


def _freeze_kwargs(kwargs):
    if kwargs is None:
        return ()
    if not hasattr(kwargs, "items"):
        return (("__value__", repr(kwargs)),)
    return tuple((str(k), repr(v)) for k, v in sorted((kwargs or {}).items()))


def _charge_linearized_caps(pss_result):
    """True when the PSS trajectory came from charge/Q-style transient stamping."""
    mode_id = pss_result.get("transient_cap_mode_id")
    if mode_id is not None:
        return int(mode_id) == 0
    mode = str(pss_result.get("transient_cap_mode", "charge")).lower()
    return mode in {"charge", "q", "qstamp", "q-stamp"}


def _conversion_charge_caps(pss_result, internal_gate_states):
    """Cap operator for the PAC/PNoise *conversion* (separate from the orbit's).

    Cadence's PAC/PNoise linearizes the device's NON-conservative Verilog-A
    operator ``I = C(V)*ddt(V)`` -- including the multi-variable
    ``Cgss(Vs,Vg,Vd)`` cross-coupling (``dC/dVd*dVd``-type terms) -- NOT the
    transient Q-stamp companion used to *integrate* the PSS orbit.  When
    PMOS_TFT gate1 caps are present, fold the conversion with that operator
    (``charge_caps=False``), mirroring :mod:`circuitopt.pnoise_solver` which already
    does this for noise.  Verified on Cadence's own slow chopper orbit: the
    ``C(V)*ddt(V)`` fold gives +0.05% vs the charge/Q collapse's -2.41% (the
    collapse's wrong cross-coupling was the slow-corner -1.9% PAC residual).
    Circuits without those caps keep the orbit's (numba-fast) charge fold --
    for linear caps the row/col folds are identical anyway.
    """
    if internal_gate_states:
        return False
    return _charge_linearized_caps(pss_result)


def _is_constant_wave(values, tol=1e-12):
    arr = np.asarray(values, float)
    if arr.size == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(arr))))
    return float(np.max(arr) - np.min(arr)) <= float(tol) * scale


def _drive_norm(input_drives, ac_drives):
    vals = list(ac_drives.values()) if ac_drives else list(input_drives.values())
    if not vals:
        return 1.0
    if len(vals) > 1 and max(vals) > min(vals):
        return float(max(vals) - min(vals))
    return float(max(abs(v) for v in vals) or 1.0)


def _periodic_wave_derivatives(waves, period):
    """Central periodic time derivative for uniformly sampled wave dictionaries."""
    if not waves:
        return {}
    period = float(period)
    first = next(iter(waves.values()))
    n = len(first)
    if n < 2 or period <= 0.0:
        return {key: np.zeros_like(np.asarray(val, float)) for key, val in waves.items()}
    dt = period / float(n)
    return {
        key: (np.roll(np.asarray(val, float), -1) -
              np.roll(np.asarray(val, float), 1)) / (2.0 * dt)
        for key, val in waves.items()
    }


def _cap_derivatives_fd(dev, Vs, Vd, Vg, step=1e-4):
    """Finite-difference d(Cgs,Cgd)/d(Vs,Vd,Vg) including internal OP dependence."""
    h = float(step)
    if h <= 0.0:
        h = 1e-4

    def caps(vs, vd, vg):
        return dev.get_capacitances(vs, vd, vg)

    out = []
    for axis in range(3):
        plus = [float(Vs), float(Vd), float(Vg)]
        minus = [float(Vs), float(Vd), float(Vg)]
        plus[axis] += h
        minus[axis] -= h
        cp = caps(*plus)
        cm = caps(*minus)
        out.append(((cp[0] - cm[0]) / (2.0 * h),
                    (cp[1] - cm[1]) / (2.0 * h)))
    return out


def _stamp_branch_control(G, p, q, ctrl, coeff):
    """Stamp branch current p->q += coeff*V(ctrl) into an MNA G matrix."""
    coeff = float(coeff)
    if coeff == 0.0 or ctrl[0] != "n":
        return
    col = ctrl[1]
    if p[0] == "n":
        G[p[1], col] += coeff
    if q[0] == "n":
        G[q[1], col] -= coeff


def _stamp_pmos_dynamic_cap_terms(G, d, g, s, dev, Vs, Vd, Vg,
                                  dVs_dt, dVd_dt, dVg_dt, *, fd_step=1e-4):
    """Linearize Verilog-A C(V)*ddt(V) terms around a periodic large-signal orbit.

    Existing C stamps cover C(t)*d(delta_v)/dt.  This adds the conductance-like
    term (dC/dx * delta_x) * dV_large/dt, which is significant on chopper edges.
    """
    vdot_gs = float(dVg_dt) - float(dVs_dt)
    vdot_gd = float(dVg_dt) - float(dVd_dt)
    if abs(vdot_gs) < 1e-30 and abs(vdot_gd) < 1e-30:
        return
    try:
        derivs = _cap_derivatives_fd(dev, Vs, Vd, Vg, step=fd_step)
    except Exception as exc:
        diagnostics.note("pac.cap_deriv_fd_fail", exc)
        return
    controls = (s, d, g)
    for ctrl, (dCgs, dCgd) in zip(controls, derivs):
        if vdot_gs != 0.0:
            _stamp_branch_control(G, g, s, ctrl, -dCgs * vdot_gs)
        if vdot_gd != 0.0:
            _stamp_branch_control(G, g, d, ctrl, -dCgd * vdot_gd)


def _has_gate1_dynamics(dev):
    return (
        hasattr(dev, "R_cap") and hasattr(dev, "R_cap2") and
        callable(getattr(dev, "_gate1_dc", None)) and
        float(getattr(dev, "R_cap", 0.0)) > 0.0 and
        float(getattr(dev, "R_cap2", 0.0)) > 0.0
    )


def _resolve_gate1_instances(all_sizes, all_nf, corner, topo,
                             model_types=None, device_kwargs=None):
    dev_inst = build_devices(all_sizes, nf=all_nf, corner=corner, topo=topo,
                             model_types=model_types, device_kwargs=device_kwargs)
    internal_gate_states = any(
        _has_gate1_dynamics(dev_inst.get(name))
        for name, *_ in topo.devices
    )
    return internal_gate_states, dev_inst


def _stamp_pmos_gate1_lti(G, C, RHS_G, RHS_C, d, g, s, g1,
                          gm, gds, Cgs, Cgd, dev):
    """Stamp the PMOS_TFT small-signal network without collapsing gate1.

    The channel current still uses the external gate (as in the Verilog-A
    ``Ich`` expression), but the displacement-current branches are between
    ``source/drain`` and the internal ``gate1`` node.  Keeping this hidden state
    lets periodic conversion retain the switch-edge charge memory that a static
    terminal {gm,gds,Cgs,Cgd} collapse loses.
    """
    stamp_mos_lti(G, C, RHS_G, RHS_C, d, g, s, gm, gds, 0.0, 0.0)
    stamp_adm(G, RHS_G, g1, g, 1.0 / float(dev.R_cap))
    leak_g = 1.0 / float(dev.R_cap2)
    if leak_g != 0.0:
        stamp_adm(G, RHS_G, s, g1, leak_g)
        stamp_adm(G, RHS_G, d, g1, leak_g)
    stamp_adm(C, RHS_C, s, g1, Cgs)
    stamp_adm(C, RHS_C, d, g1, Cgd)


def _stamp_pmos_gate1_dynamic_cap_terms(
        G, d, g, s, g1, dev, Vs, Vd, Vg,
        dVs_dt, dVd_dt, dVg1_dt, *, fd_step=1e-4):
    """Conductance-like PAC terms for C(V)*ddt(V(source/drain,gate1))."""
    vdot_sg1 = float(dVs_dt) - float(dVg1_dt)
    vdot_dg1 = float(dVd_dt) - float(dVg1_dt)
    if abs(vdot_sg1) < 1e-30 and abs(vdot_dg1) < 1e-30:
        return
    try:
        derivs = _cap_derivatives_fd(dev, Vs, Vd, Vg, step=fd_step)
    except Exception as exc:
        diagnostics.note("pac.cap_deriv_fd_gate1_fail", exc)
        return
    controls = (s, d, g)
    for ctrl, (dCgs, dCgd) in zip(controls, derivs):
        if vdot_sg1 != 0.0:
            _stamp_branch_control(G, s, g1, ctrl, dCgs * vdot_sg1)
        if vdot_dg1 != 0.0:
            _stamp_branch_control(G, d, g1, ctrl, dCgd * vdot_dg1)


def _assemble_pac_linearization_python(
        all_sizes, all_nf, corner, topo, tbias, t_uniform,
        node_wave, input_wave, node_inputs, drive_list, drive_amps, *,
        charge_caps, internal_gate_states=True, dev_inst=None,
        model_types=None, device_kwargs=None):
    """Build time-sampled PAC G/C matrices, optionally retaining PMOS gate1.

    Devices bound to OSDI (compiled Verilog-A) models stamp their full 4×4
    quasi-static terminal (G, C) — see
    :meth:`OsdiDevice.get_terminal_linearization`; OTFT devices keep the
    original gm/gds/Cgs/Cgd (+gate1/dynamic-cap) stamps unchanged.
    """
    N = len(t_uniform)
    n = topo.n
    idx = topo.idx
    rails = topo.rail_values(tbias)
    if dev_inst is None:
        dev_inst = build_devices(all_sizes, nf=all_nf, corner=corner, topo=topo,
                                 model_types=model_types, device_kwargs=device_kwargs)

    gate1_idx = {}
    if internal_gate_states:
        for name, *_ in topo.devices:
            dev = dev_inst.get(name)
            if dev is not None and _has_gate1_dynamics(dev):
                gate1_idx[name] = n + len(gate1_idx)
    n_state = n + len(gate1_idx)

    ext_idx = {node: n_state + i for i, node in enumerate(drive_list)}
    n_ext = n_state + len(drive_list)

    period = float(t_uniform[1] - t_uniform[0]) * float(N) if N > 1 else 0.0
    node_dot = _periodic_wave_derivatives(node_wave, period)
    input_dot = _periodic_wave_derivatives(input_wave, period)

    def term_value(node, m):
        if node in idx:
            return node_wave[node][m]
        if node in node_inputs:
            return input_wave[node_inputs[node]][m]
        return rails[node]

    def term_derivative(node, m):
        if node in idx:
            return node_dot[node][m]
        if node in node_inputs:
            return input_dot[node_inputs[node]][m]
        return 0.0

    def term(node):
        if node in idx:
            return ("n", idx[node])
        if node in ext_idx:
            return ("n", ext_idx[node])
        return ("v", 0.0)

    G_const = np.zeros((n_ext, n_ext))
    C_const = np.zeros((n_ext, n_ext))
    rg = np.zeros(n_ext)
    rc = np.zeros(n_ext)
    for a, b, cap in topo.cap_list():
        stamp_adm(C_const, rc, term(a), term(b), cap)
    for _, a, b, R in topo.resistors:
        stamp_adm(G_const, rg, term(a), term(b), 1.0 / R)
    for k in range(n):
        G_const[k, k] += 1e-12

    Gt_full = np.zeros((N, n_ext, n_ext))
    Ct_full = np.zeros((N, n_ext, n_ext))
    gate1_dc_dot = {}
    if gate1_idx and not charge_caps:
        for name, d, g, s in topo.devices:
            if name not in gate1_idx:
                continue
            dev = dev_inst[name]
            vals = np.empty(N, dtype=float)
            for m in range(N):
                vals[m] = dev._gate1_dc(
                    term_value(s, m), term_value(d, m), term_value(g, m))
            gate1_dc_dot[name] = _periodic_wave_derivatives(
                {name: vals}, period)[name]

    for m in range(N):
        Gt_full[m] += G_const
        Ct_full[m] += C_const
        for name, d, g, s in topo.devices:
            Vs = term_value(s, m)
            Vd = term_value(d, m)
            Vg = term_value(g, m)
            if dev_inst[name].HAS_TERMINAL_LINEARIZATION:
                # silicon: full quasi-static 4×4 terminal stamp (bulk is a
                # constant bias -> known-voltage term, drops). The OTFT
                # dynamic-cap corrections below are model-specific; the OSDI
                # C(V) block is the dQ/dV operator at the orbit point.
                G4, C4 = dev_inst[name].get_terminal_linearization(Vs, Vd, Vg)
                stamp_dense_lti(Gt_full[m], Ct_full[m], rg, rc,
                                 (term(d), term(g), term(s), ("v", 0.0)),
                                 G4, C4)
                continue
            p = get_ss_params(
                all_sizes[name][0], all_sizes[name][1], Vs, Vd, Vg,
                corner=dev_corner(corner, name),
                nf=dev_nf(all_nf, name), dev_inst=dev_inst[name])
            if name in gate1_idx:
                g1 = ("n", gate1_idx[name])
                _stamp_pmos_gate1_lti(
                    Gt_full[m], Ct_full[m], rg, rc, term(d), term(g), term(s), g1,
                    p["gm"], p["gds"], p["Cgs"], p["Cgd"], dev_inst[name])
                if not charge_caps:
                    _stamp_pmos_gate1_dynamic_cap_terms(
                        Gt_full[m], term(d), term(g), term(s), g1, dev_inst[name],
                        Vs, Vd, Vg,
                        term_derivative(s, m), term_derivative(d, m),
                        gate1_dc_dot[name][m])
            else:
                stamp_mos_lti(
                    Gt_full[m], Ct_full[m], rg, rc, term(d), term(g), term(s),
                    p["gm"], p["gds"], p["Cgs"], p["Cgd"])
                if not charge_caps:
                    _stamp_pmos_dynamic_cap_terms(
                        Gt_full[m], term(d), term(g), term(s), dev_inst[name],
                        Vs, Vd, Vg,
                        term_derivative(s, m), term_derivative(d, m),
                        term_derivative(g, m))

    Gt = Gt_full[:, :n_state, :n_state]
    Ct = Ct_full[:, :n_state, :n_state]
    if len(drive_list):
        gdrive = Gt_full[:, :n_state, n_state:] @ np.asarray(drive_amps, complex)
        cdrive = Ct_full[:, :n_state, n_state:] @ np.asarray(drive_amps, complex)
    else:
        gdrive = np.zeros((N, n_state), dtype=complex)
        cdrive = np.zeros((N, n_state), dtype=complex)
    return Gt, Ct, gdrive, cdrive, len(gate1_idx)


def _pac_hb_blocks(Gf, Cf, K, N, n, fundamental, *, charge_caps=False):
    """Dense HB conversion blocks (same kernel as pnoise). Single-sourced onto
    ``_pnoise_hb_blocks_impl`` (jitted for large systems, interpreted `.py_func`
    below the JIT-worthwhile size)."""
    use_numba = (
        pac_hb_blocks_numba is not None and
        (2 * int(K) + 1) * int(n) >= 16
    )
    kernel = _pnoise_hb_blocks_impl if use_numba else py_impl(_pnoise_hb_blocks_impl)
    Y_base, C_block = kernel(
        np.asarray(Gf, dtype=np.complex128),
        np.asarray(Cf, dtype=np.complex128),
        int(K), float(fundamental), bool(charge_caps))
    return Y_base, C_block, use_numba


def _stamp_arrays(terms):
    kind = np.empty(len(terms), dtype=np.int64)
    ref = np.empty(len(terms), dtype=np.int64)
    for pos, term in enumerate(terms):
        kind[pos] = int(term[0])
        ref[pos] = int(term[1])
    return kind, ref


def _try_numba_pac_linearization(
        all_sizes, all_nf, corner, topo, tbias, t_uniform, node_wave,
        input_wave, node_inputs, drive_list, drive_amps, *, charge_caps,
        dev_inst=None):
    if pac_linearize_orbit_numba is None or not charge_caps:
        return None
    if (
        getattr(topo, "vccs", ()) or
        getattr(topo, "vcvs", ()) or
        getattr(topo, "cccs", ()) or
        getattr(topo, "ccvs", ())
    ):
        return None
    # the kernel evaluates the OTFT compact model; silicon (OSDI) devices go
    # through the Python assembler's dense terminal stamp instead
    if dev_inst is not None and any(
            getattr(dev_inst.get(name), "HAS_TERMINAL_LINEARIZATION", False)
            for name, *_ in topo.devices):
        return None

    try:
        N = len(t_uniform)
        n = topo.n
        idx = topo.idx
        rails = topo.rail_values(tbias)
        input_keys = tuple(input_wave)
        input_index = {key: i for i, key in enumerate(input_keys)}
        drive_index = {node: i for i, node in enumerate(drive_list)}

        missing_input = [
            key for key in node_inputs.values()
            if key not in input_index
        ]
        if missing_input:
            return None

        def value_term(node):
            if node in idx:
                return (0, idx[node])
            if node in node_inputs:
                return (1, input_index[node_inputs[node]])
            return (2, float(rails[node]))

        def stamp_term(node):
            if node in idx:
                return (0, idx[node])
            if node in drive_index:
                return (1, drive_index[node])
            return (2, 0)

        node_wave_arr = np.vstack([
            np.asarray(node_wave[node], float) for node in topo.solved
        ]).T
        if input_keys:
            input_wave_arr = np.vstack([
                np.asarray(input_wave[key], float) for key in input_keys
            ])
        else:
            input_wave_arr = np.empty((0, N), dtype=float)

        if dev_inst is None:
            dev_inst = build_devices(all_sizes, nf=all_nf, corner=corner, topo=topo)
        params = [dev_inst[name].get_numba_params() for name, *_ in topo.devices]
        p_Vfb = np.array([p.Vfb for p in params], dtype=float)
        p_Vss = np.array([p.Vss for p in params], dtype=float)
        p_Lc = np.array([p.Lc for p in params], dtype=float)
        p_lambda = np.array([p.lambda_ for p in params], dtype=float)
        p_contact_scale = np.array([p.contact_scale for p in params], dtype=float)
        p_exponent = np.array([p.channel_exponent for p in params], dtype=float)
        p_current_scale = np.array([p.current_scale for p in params], dtype=float)
        p_inv_Rleak = np.array([p.inv_Rleak for p in params], dtype=float)
        p_two_over_pi = np.array([p.two_over_pi for p in params], dtype=float)
        p_cap_cgs1 = np.array([p.cap_cgs1 for p in params], dtype=float)
        p_cap_cgd1 = np.array([p.cap_cgd1 for p in params], dtype=float)
        p_cap_half_wl_ci = np.array([p.cap_half_wl_ci for p in params], dtype=float)
        p_cap_cgs3_base = np.array([p.cap_cgs3_base for p in params], dtype=float)
        p_cap_cgd3_base = np.array([p.cap_cgd3_base for p in params], dtype=float)
        p_k1 = np.array([p.k1 for p in params], dtype=float)

        dev_value_d_kind, dev_value_d_ref, dev_value_d_val = term_arrays(
            [value_term(d) for _name, d, _g, _s in topo.devices])
        dev_value_g_kind, dev_value_g_ref, dev_value_g_val = term_arrays(
            [value_term(g) for _name, _d, g, _s in topo.devices])
        dev_value_s_kind, dev_value_s_ref, dev_value_s_val = term_arrays(
            [value_term(s) for _name, _d, _g, s in topo.devices])
        dev_stamp_d_kind, dev_stamp_d_ref = _stamp_arrays(
            [stamp_term(d) for _name, d, _g, _s in topo.devices])
        dev_stamp_g_kind, dev_stamp_g_ref = _stamp_arrays(
            [stamp_term(g) for _name, _d, g, _s in topo.devices])
        dev_stamp_s_kind, dev_stamp_s_ref = _stamp_arrays(
            [stamp_term(s) for _name, _d, _g, s in topo.devices])

        res_a_kind, res_a_ref = _stamp_arrays(
            [stamp_term(a) for _name, a, _b, _R in topo.resistors])
        res_b_kind, res_b_ref = _stamp_arrays(
            [stamp_term(b) for _name, _a, b, _R in topo.resistors])
        res_g = np.array([1.0 / float(R) for _name, _a, _b, R in topo.resistors],
                         dtype=float)
        caps = topo.cap_list()
        cap_a_kind, cap_a_ref = _stamp_arrays([stamp_term(a) for a, _b, _c in caps])
        cap_b_kind, cap_b_ref = _stamp_arrays([stamp_term(b) for _a, b, _c in caps])
        cap_value = np.array([float(c) for _a, _b, c in caps], dtype=float)

        ok, Gt, Ct, Gin, Cin = pac_linearize_orbit_numba(
            np.asarray(node_wave_arr, float),
            np.asarray(input_wave_arr, float),
            dev_value_d_kind, dev_value_d_ref, dev_value_d_val,
            dev_value_g_kind, dev_value_g_ref, dev_value_g_val,
            dev_value_s_kind, dev_value_s_ref, dev_value_s_val,
            dev_stamp_d_kind, dev_stamp_d_ref,
            dev_stamp_g_kind, dev_stamp_g_ref,
            dev_stamp_s_kind, dev_stamp_s_ref,
            p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
            p_current_scale, p_inv_Rleak,
            p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
            p_cap_cgs3_base, p_cap_cgd3_base, p_k1,
            res_a_kind, res_a_ref, res_b_kind, res_b_ref, res_g,
            cap_a_kind, cap_a_ref, cap_b_kind, cap_b_ref, cap_value,
            len(drive_list),
        )
        if not ok:
            return None
        Gf = np.fft.fft(Gt, axis=0) / N
        Cf = np.fft.fft(Ct, axis=0) / N
        if len(drive_list):
            gdrive = np.tensordot(Gin, np.asarray(drive_amps, complex),
                                  axes=([2], [0]))
            cdrive = np.tensordot(Cin, np.asarray(drive_amps, complex),
                                  axes=([2], [0]))
        else:
            gdrive = np.zeros((N, n), dtype=complex)
            cdrive = np.zeros((N, n), dtype=complex)
        Ginf = np.fft.fft(gdrive, axis=0) / N
        Cinf = np.fft.fft(cdrive, axis=0) / N
        return Gf, Cf, Ginf, Cinf
    except Exception as exc:
        diagnostics.note("pac.numba_linearization_fail", exc)
        return None


def _try_numba_pac_linearization_gate1(
        all_sizes, all_nf, corner, topo, tbias, t_uniform, node_wave,
        input_wave, node_inputs, drive_list, drive_amps, *, dev_inst=None):
    """Numba twin of the gate1-retained Verilog-A (``charge_caps=False``)
    linearization. Returns (Gf, Cf, Ginf, Cinf, n_gate1) or None to fall back to
    the Python assembly. Requires every device to carry gate1 dynamics."""
    if pac_linearize_orbit_gate1_numba is None:
        return None
    if (getattr(topo, "vccs", ()) or getattr(topo, "vcvs", ()) or
            getattr(topo, "cccs", ()) or getattr(topo, "ccvs", ())):
        return None
    try:
        N = len(t_uniform)
        n = topo.n
        idx = topo.idx
        rails = topo.rail_values(tbias)
        if dev_inst is None:
            dev_inst = build_devices(all_sizes, nf=all_nf, corner=corner, topo=topo)
        # The kernel assumes EVERY device has a gate1 node (index n+pos, in topo order,
        # matching _assemble_pac_linearization_python). Mixed topologies -> Python.
        if not all(_has_gate1_dynamics(dev_inst.get(name)) for name, *_ in topo.devices):
            return None

        input_keys = tuple(input_wave)
        input_index = {key: i for i, key in enumerate(input_keys)}
        drive_index = {node: i for i, node in enumerate(drive_list)}
        if any(key not in input_index for key in node_inputs.values()):
            return None

        def value_term(node):
            if node in idx:
                return (0, idx[node])
            if node in node_inputs:
                return (1, input_index[node_inputs[node]])
            return (2, float(rails[node]))

        def stamp_term(node):
            if node in idx:
                return (0, idx[node])
            if node in drive_index:
                return (1, drive_index[node])
            return (2, 0)

        node_wave_arr = np.vstack([
            np.asarray(node_wave[node], float) for node in topo.solved]).T
        input_wave_arr = (np.vstack([np.asarray(input_wave[k], float) for k in input_keys])
                          if input_keys else np.empty((0, N), dtype=float))
        period = float(t_uniform[1] - t_uniform[0]) * float(N) if N > 1 else 0.0
        node_dot = _periodic_wave_derivatives(node_wave, period)
        input_dot = _periodic_wave_derivatives(input_wave, period)
        node_dot_arr = np.vstack([
            np.asarray(node_dot[node], float) for node in topo.solved]).T
        input_dot_arr = (np.vstack([np.asarray(input_dot[k], float) for k in input_keys])
                         if input_keys else np.empty((0, N), dtype=float))

        params = [dev_inst[name].get_numba_params() for name, *_ in topo.devices]
        flt = lambda attr: np.array([getattr(p, attr) for p in params], dtype=float)
        p_Vfb = flt("Vfb"); p_Vss = flt("Vss"); p_Lc = flt("Lc")
        p_lambda = flt("lambda_"); p_contact_scale = flt("contact_scale")
        p_exponent = flt("channel_exponent"); p_current_scale = flt("current_scale")
        p_inv_Rleak = flt("inv_Rleak"); p_two_over_pi = flt("two_over_pi")
        p_cap_cgs1 = flt("cap_cgs1"); p_cap_cgd1 = flt("cap_cgd1")
        p_cap_half_wl_ci = flt("cap_half_wl_ci")
        p_cap_cgs3_base = flt("cap_cgs3_base"); p_cap_cgd3_base = flt("cap_cgd3_base")
        p_k1 = flt("k1")
        p_R_cap = np.array([float(dev_inst[name].R_cap) for name, *_ in topo.devices])
        p_R_cap2 = np.array([float(dev_inst[name].R_cap2) for name, *_ in topo.devices])
        ndev = len(topo.devices)
        gate1_ref = np.arange(n, n + ndev, dtype=np.int64)
        n_state = n + ndev

        vt_d = term_arrays([value_term(d) for _n, d, _g, _s in topo.devices])
        vt_g = term_arrays([value_term(g) for _n, _d, g, _s in topo.devices])
        vt_s = term_arrays([value_term(s) for _n, _d, _g, s in topo.devices])
        st_d = _stamp_arrays([stamp_term(d) for _n, d, _g, _s in topo.devices])
        st_g = _stamp_arrays([stamp_term(g) for _n, _d, g, _s in topo.devices])
        st_s = _stamp_arrays([stamp_term(s) for _n, _d, _g, s in topo.devices])
        res_a = _stamp_arrays([stamp_term(a) for _n, a, _b, _R in topo.resistors])
        res_b = _stamp_arrays([stamp_term(b) for _n, _a, b, _R in topo.resistors])
        res_g = np.array([1.0 / float(R) for _n, _a, _b, R in topo.resistors], dtype=float)
        caps = topo.cap_list()
        cap_a = _stamp_arrays([stamp_term(a) for a, _b, _c in caps])
        cap_b = _stamp_arrays([stamp_term(b) for _a, b, _c in caps])
        cap_value = np.array([float(c) for _a, _b, c in caps], dtype=float)

        ok, Gt, Ct, Gin, Cin = pac_linearize_orbit_gate1_numba(
            np.asarray(node_wave_arr, float), np.asarray(input_wave_arr, float),
            np.asarray(node_dot_arr, float), np.asarray(input_dot_arr, float),
            vt_d[0], vt_d[1], vt_d[2], vt_g[0], vt_g[1], vt_g[2],
            vt_s[0], vt_s[1], vt_s[2],
            st_d[0], st_d[1], st_g[0], st_g[1], st_s[0], st_s[1],
            gate1_ref, p_R_cap, p_R_cap2,
            p_Vfb, p_Vss, p_Lc, p_lambda, p_contact_scale, p_exponent,
            p_current_scale, p_inv_Rleak,
            p_two_over_pi, p_cap_cgs1, p_cap_cgd1, p_cap_half_wl_ci,
            p_cap_cgs3_base, p_cap_cgd3_base, p_k1,
            res_a[0], res_a[1], res_b[0], res_b[1], res_g,
            cap_a[0], cap_a[1], cap_b[0], cap_b[1], cap_value,
            len(drive_list), int(n_state), 1e-4)
        if not ok:
            return None
        Gf = np.fft.fft(Gt, axis=0) / N
        Cf = np.fft.fft(Ct, axis=0) / N
        if len(drive_list):
            damp = np.asarray(drive_amps, complex)
            gdrive = np.tensordot(Gin, damp, axes=([2], [0]))
            cdrive = np.tensordot(Cin, damp, axes=([2], [0]))
        else:
            gdrive = np.zeros((N, n_state), dtype=complex)
            cdrive = np.zeros((N, n_state), dtype=complex)
        Ginf = np.fft.fft(gdrive, axis=0) / N
        Cinf = np.fft.fft(cdrive, axis=0) / N
        return Gf, Cf, Ginf, Cinf, ndev
    except Exception as exc:
        diagnostics.note("pac.numba_linearization_gate1_fail", exc)
        return None


def _try_lti_ac_fast_path(sizes, bias, freqs, pss_result, input_drive, nf,
                          corner=None,
                          compute_condition=False,
                          model_types=None, device_kwargs=None):
    """Use ordinary AC when the supplied PSS orbit is time invariant.

    This is an exact reduction for static operating points.  It is deliberately
    conservative: any time-varying orbit/input/current source or complex phased
    drive falls back to finite-difference PAC.
    """
    topo = pss_result["topology"]
    if pss_result.get("current_inputs"):
        return None
    if any(not _is_constant_wave(v) for v in pss_result.get("inputs", {}).values()):
        return None
    if any(not _is_constant_wave(pss_result["nodes"][node]) for node in topo.solved):
        return None
    drives = {str(k): complex(v) for k, v in input_drive.items()}
    if any(abs(v.imag) > 0.0 for v in drives.values()):
        return None

    tbias = dict(pss_result.get("bias", bias))
    node_inputs = dict(pss_result.get("node_inputs", {}) or {})
    input_drives = {}
    ac_drives = {}
    consumed = set()

    for node, key in node_inputs.items():
        if key not in drives:
            continue
        if node not in topo.rails:
            return None
        ref = topo.rails[node]
        dc_val = float(np.asarray(pss_result["inputs"][key], float)[0])
        if isinstance(ref, str):
            tbias[ref] = dc_val
        elif abs(float(ref) - dc_val) > 1e-9 * max(1.0, abs(dc_val)):
            return None
        ac_drives[str(node)] = float(drives[key].real)
        consumed.add(key)

    transient_inputs = dict(getattr(topo, "transient_inputs", {}) or {})
    dev_by_name = {name: (d, g, s) for name, d, g, s in topo.devices}
    for dev, key in transient_inputs.items():
        if key not in drives:
            continue
        if dev not in dev_by_name:
            return None
        gate = dev_by_name[dev][1]
        if gate in topo.idx or gate not in topo.rails:
            return None
        ref = topo.rails[gate]
        dc_val = float(np.asarray(pss_result["inputs"][key], float)[0])
        if isinstance(ref, str):
            tbias[ref] = dc_val
        elif abs(float(ref) - dc_val) > 1e-9 * max(1.0, abs(dc_val)):
            return None
        input_drives[str(dev)] = float(drives[key].real)
        consumed.add(key)

    if consumed != set(drives):
        return None
    if not input_drives and not ac_drives:
        return None

    fast_topo = Topology(
        solved=topo.solved,
        devices=topo.devices,
        rails=topo.rails,
        outputs=topo.outputs,
        input_drives=input_drives,
        ac_drives=ac_drives,
        load_caps=topo.load_caps,
        dc_guesses=[dict(zip(topo.solved, np.asarray(pss_result["x0"], float)))],
        aliases=topo.aliases,
        transient_inputs=topo.transient_inputs,
        resistors=topo.resistors,
        capacitors=topo.capacitors,
        isources=topo.isources,
        vccs=topo.vccs,
        vsources=topo.vsources,
        vcvs=topo.vcvs,
        cccs=topo.cccs,
        ccvs=topo.ccvs,
        dc_tol=topo.dc_tol,
        require_dc_in_box=topo.require_dc_in_box,
    )
    ac = ac_solve(
        sizes, tbias, freqs, topo=fast_topo, nf=nf, corner=corner,
        x0_guess=dict(zip(topo.solved, np.asarray(pss_result["x0"], float))),
        model_types=model_types, device_kwargs=device_kwargs,
    )
    if ac is None:
        return None
    norm = _drive_norm(input_drives, ac_drives)
    response = np.asarray(ac["response"], complex) * norm
    gains = np.abs(response)
    return {
        "freqs": np.asarray(freqs, float),
        "response": response,
        "gains": gains,
        "Hmag": gains,
        "Av_dc_dB": 20 * np.log10(max(float(gains[0]), 1e-300))
        if len(gains) else np.nan,
        "bw_Hz": bw_from_gain(freqs, gains) if len(gains) else np.nan,
        "pss": pss_result,
        "input_drive": input_drive,
        "pacmag": 1.0,
        "pac_residual": np.zeros(len(freqs)),
        "pac_condition": np.ones(len(freqs)),
        "pac_state_period_runs": 0,
        "pac_input_period_runs": 0,
        "pac_period_runs": 0,
        "pac_state_cache_hit": False,
        "pac_input_cache_hits": 0,
        "pac_cache_enabled": False,
        "pac_condition_computed": bool(compute_condition),
        "nfail": np.zeros(len(freqs), dtype=int),
        "method": "lti_ac_fast_path",
        "ac": ac,
    }


def _analytic_adjoint_pac(all_sizes, tbias, freqs, *, pss_result, input_drive,
                          all_nf, corner=None, n_period_samples=384,
                          max_sideband=10, cache=True, pacmag=1.0,
                          compute_condition=False,
                          model_types=None, device_kwargs=None):
    """Analytic-adjoint PAC from the PSS-orbit small-signal matrices.

    Samples the periodic small-signal G(t)/C(t) (and the input-coupling columns
    G_in(t)/C_in(t)) along the PSS trajectory, forms the harmonic-balance
    conversion matrix Y_HB(f) (blocks G_{kr-kc} + j*2pi*(f+kr*f0)*C_{kr-kc},
    matching the d/dt(C x) LPTV form), and reads the sideband-0 conversion gain
    from a single adjoint solve per frequency:

        out_sb0(f) = adj . b_input(f),   adj = Y_HB(f)^{-T} e,
        b_input[kr] = -(G_in_{kr} + j*2pi*(f+kr*f0)*C_in_{kr}) * drive

    Cost is O(1) (2K+1)n-block solves per frequency instead of the O(n_state)
    one-period transient runs of the finite-difference shooting kernel.
    """
    freqs = np.asarray(freqs, float)
    topo = pss_result["topology"]
    n = topo.n
    idx = topo.idx
    t_orbit = np.asarray(pss_result["t"], float)
    period = float(pss_result.get("period", t_orbit[-1] - t_orbit[0]))
    if period <= 0.0:
        return None
    fundamental = 1.0 / period
    K = int(max_sideband)
    # The HB matrix uses harmonics G_{kr-kc} for kr,kc in [-K,K], i.e. up to the
    # 2K-th harmonic. Sampling the orbit with fewer than 4K points aliases those
    # coefficients (a sharp-switched chopper has slowly-decaying harmonics), so
    # keep the period sampling above the Nyquist limit for the bands we read.
    N = max(int(n_period_samples), 4 * K + 2)
    nb = 2 * K + 1
    node_inputs = dict(pss_result.get("node_inputs", {}) or {})

    # driven input NODES (rails) carrying the small-signal drive
    drive_nodes = {}
    for node, key in node_inputs.items():
        if key in input_drive and node in topo.rails and node not in idx:
            drive_nodes[node] = drive_nodes.get(node, 0j) + complex(input_drive[key])
    dev_by_name = {name: (d, g, s) for name, d, g, s in topo.devices}
    for dev, key in (getattr(topo, "transient_inputs", {}) or {}).items():
        if key in input_drive and dev in dev_by_name:
            gate = dev_by_name[dev][1]
            if gate in topo.rails and gate not in idx:
                drive_nodes[gate] = drive_nodes.get(gate, 0j) + complex(input_drive[key])
    # Small-signal drive on a true-MNA voltage source: the input key matches a
    # vsource whose value is that waveform key.  Unlike a rail drive (which couples
    # through the periodic G_in/C_in node columns), this enters the bordered HB as a
    # baseband forcing on the source's branch-constraint row (V_p - V_q = drive).
    # It is what makes stiff tau>>T switched-cap PAC robust: the conversion matrix is
    # built from the continuous small-signal G(t)/C(t) along the orbit (integration-
    # method independent), instead of the x0-sensitive finite-difference shooting
    # whose (I-Phi)^-1 amplifies the tiny be-vs-gear2 orbit difference into a 24x gain.
    drive_branches = []  # (branch index within topo.vsources, complex amplitude)
    for k, vs in enumerate(topo.vsources):
        val = vs[3] if len(vs) > 3 else None
        if isinstance(val, str) and val in input_drive:
            drive_branches.append((k, complex(input_drive[val])))
    if not drive_nodes and not drive_branches:
        return None
    drive_list = list(drive_nodes)
    drive_amps = np.array([drive_nodes[node] for node in drive_list], dtype=complex)

    t_uniform = np.linspace(0.0, period, N, endpoint=False)
    node_wave = {
        node: np.interp(t_uniform, t_orbit,
                        np.asarray(pss_result["nodes"][node], float), period=period)
        for node in topo.solved
    }
    input_wave = {
        key: np.interp(t_uniform, t_orbit, np.asarray(val, float), period=period)
        for key, val in pss_result.get("inputs", {}).items()
    }

    cache_store = pss_result.setdefault("_pac_analytic_cache", {}) if cache else {}
    internal_gate_states, lin_dev_inst = _resolve_gate1_instances(
        all_sizes, all_nf, corner, topo,
        model_types=model_types, device_kwargs=device_kwargs)
    charge_caps = _conversion_charge_caps(pss_result, internal_gate_states)
    lin_key = (
        "pac_analytic_lin_gate1_v1", tuple(topo.solved), tuple(topo.devices),
        tuple(topo.resistors), tuple(topo.cap_list()), float(period), int(N),
        _freeze_sizes(all_sizes), _freeze_nf(all_nf), _freeze_kwargs(corner or {}),
        "charge_caps" if charge_caps else "cvddt_caps",
        bool(internal_gate_states),
        tuple(sorted(drive_list)),
        _freeze_complex_map(dict(zip(drive_list, drive_amps))),
    )
    lin_cache_hit = bool(cache and lin_key in cache_store)
    lin_numba_used = False
    t_linear0 = time.perf_counter()
    if lin_cache_hit:
        lin = cache_store[lin_key]
        Gf, Cf, Ginf, Cinf = lin["Gf"], lin["Cf"], lin["Ginf"], lin["Cinf"]
        n_state = int(lin.get("n_state", Gf.shape[1]))
        n_gate1 = int(lin.get("n_gate1", max(0, n_state - n)))
        lin_numba_used = bool(lin.get("numba_used", False))
    else:
        fast_lin = None
        gate1_fast = None
        if not internal_gate_states:
            fast_lin = _try_numba_pac_linearization(
                all_sizes, all_nf, corner, topo, tbias, t_uniform,
                node_wave, input_wave, node_inputs, drive_list, drive_amps,
                charge_caps=charge_caps, dev_inst=lin_dev_inst,
            )
        elif not charge_caps:
            gate1_fast = _try_numba_pac_linearization_gate1(
                all_sizes, all_nf, corner, topo, tbias, t_uniform,
                node_wave, input_wave, node_inputs, drive_list, drive_amps,
                dev_inst=lin_dev_inst,
            )
        if fast_lin is not None:
            Gf, Cf, Ginf, Cinf = fast_lin
            n_state = n
            n_gate1 = 0
            lin_numba_used = True
        elif gate1_fast is not None:
            Gf, Cf, Ginf, Cinf, n_gate1 = gate1_fast
            n_state = Gf.shape[1]
            lin_numba_used = True
        else:
            Gt, Ct, gdrive, cdrive, n_gate1 = _assemble_pac_linearization_python(
                all_sizes, all_nf, corner, topo, tbias, t_uniform,
                node_wave, input_wave, node_inputs, drive_list, drive_amps,
                charge_caps=charge_caps,
                internal_gate_states=internal_gate_states,
                dev_inst=lin_dev_inst,
            )
            n_state = Gt.shape[1]
            Gf = np.fft.fft(Gt, axis=0) / N
            Cf = np.fft.fft(Ct, axis=0) / N
            Ginf = np.fft.fft(gdrive, axis=0) / N
            Cinf = np.fft.fft(cdrive, axis=0) / N
        if cache:
            cache_store[lin_key] = {
                "Gf": Gf,
                "Cf": Cf,
                "Ginf": Ginf,
                "Cinf": Cinf,
                "n_state": int(n_state),
                "n_gate1": int(n_gate1),
                "numba_used": bool(lin_numba_used),
            }
    linearization_time_s = time.perf_counter() - t_linear0

    t_hb0 = time.perf_counter()
    Y_base, C_block, hb_numba_used = _pac_hb_blocks(
        Gf, Cf, K, N, n_state, fundamental, charge_caps=charge_caps)
    hb_assembly_time_s = time.perf_counter() - t_hb0
    e = np.zeros(nb * n_state, dtype=complex)
    base0 = K * n_state
    for node, w in topo.output_weights().items():
        e[base0 + idx[node]] = w

    # Ideal voltage sources: border the harmonic-balance system with branch-current
    # unknowns (one per source per harmonic, appended after the nb*n node block). The
    # incidence is constant, so it couples node<->branch only within the same harmonic;
    # branch rows carry no capacitance (C border = 0) and the source is an AC short
    # (no stimulus -> b stays 0 there).
    nbr = topo.n_branches
    if nbr:
        all_branch_sources = list(topo.vsources) + list(topo.vcvs) + list(topo.ccvs)
        Binc = branch_incidence(all_branch_sources, idx, n)
        if n_state != n:
            Bpad = np.zeros((n_state, nbr))
            Bpad[:n, :] = Binc
            Binc = Bpad
        nt = nb * (n_state + nbr)
        Ya = np.zeros((nt, nt), dtype=complex)
        Ca = np.zeros((nt, nt), dtype=complex)
        Ya[:nb * n_state, :nb * n_state] = Y_base
        Ca[:nb * n_state, :nb * n_state] = C_block
        boff = nb * n_state
        for h in range(nb):
            r0, c0 = h * n_state, boff + h * nbr
            Ya[r0:r0 + n_state, c0:c0 + nbr] = Binc      # KCL: node <- branch current
            Ya[c0:c0 + nbr, r0:r0 + n_state] = Binc.T    # constraint: V_p - V_q = 0
        Y_base, C_block = Ya, Ca
        e = np.concatenate([e, np.zeros(nb * nbr, dtype=complex)])

    # Pre-stack the frequency-independent input-coupling harmonics so the per-
    # frequency RHS is one vectorized expression rather than a (2K+1) Python loop.
    # ``om_offset`` carries the per-block kr*f0 term used by charge/Q orbit caps;
    # for the non-conservative C(V)*ddt conversion the input sees only baseband.
    kr_arr = np.arange(-K, K + 1)
    Ginf_stacked = Ginf[kr_arr % N].reshape(nb * n_state)
    Cinf_stacked = Cinf[kr_arr % N].reshape(nb * n_state)
    om_offset = np.repeat(kr_arr * fundamental, n_state) if charge_caps else 0.0
    # ``b`` is rebuilt per frequency only in its node block; the branch rows are
    # frequency-independent (AC short -> 0, plus any driven true-MNA vsource
    # forcing), so set them once and reuse the buffer.
    b = np.zeros(e.shape[0], dtype=complex)
    for br_idx, amp in drive_branches:
        b[nb * n_state + K * nbr + br_idx] = amp
    # Reuse one conversion-matrix buffer instead of allocating Y_base + s*C_block
    # (a full (2K+1)n square complex matrix) every frequency.
    Ybuf = np.empty_like(Y_base)

    response = np.empty(len(freqs), dtype=complex)
    residuals = np.zeros(len(freqs))
    conditions = np.ones(len(freqs))
    for fi, f in enumerate(freqs):
        f = float(f)
        np.multiply(C_block, 2j * np.pi * f, out=Ybuf)   # Ybuf = (2j*pi*f) * C_block
        Ybuf += Y_base                                    # Ybuf = Y_base + (2j*pi*f) C
        if compute_condition:
            try:
                conditions[fi] = float(np.linalg.cond(Ybuf))
            except Exception as exc:
                diagnostics.note("pac.condition_number_fail", exc)
                conditions[fi] = np.inf
        try:
            adj = np.linalg.solve(Ybuf.T, e)
        except np.linalg.LinAlgError:
            adj = np.linalg.lstsq(Ybuf.T, e, rcond=None)[0]
        b[:nb * n_state] = -(Ginf_stacked + (2j * np.pi * (f + om_offset)) * Cinf_stacked)
        response[fi] = adj @ b

    gains = np.abs(response)
    return {
        "freqs": freqs,
        "response": response,
        "gains": gains,
        "Hmag": gains,
        "Av_dc_dB": 20 * np.log10(max(float(gains[0]), 1e-300)) if len(gains) else np.nan,
        "bw_Hz": bw_from_gain(freqs, gains) if len(gains) else np.nan,
        "pss": pss_result,
        "input_drive": input_drive,
        "pacmag": float(pacmag),
        "pac_residual": residuals,
        "pac_condition": conditions,
        "pac_state_period_runs": 0,
        "pac_input_period_runs": 0,
        "pac_period_runs": 0,
        "pac_state_cache_hit": bool(lin_cache_hit),
        "pac_input_cache_hits": 0,
        "pac_cache_enabled": bool(cache),
        "pac_condition_computed": bool(compute_condition),
        "pac_hb_size": int(nb * n_state),
        "pac_state_size": int(n_state),
        "pac_internal_gate1_states": int(n_gate1),
        "pac_numba_linearization_used": bool(lin_numba_used),
        "pac_numba_hb_used": bool(hb_numba_used),
        "pac_linearization_time_s": float(linearization_time_s),
        "pac_hb_assembly_time_s": float(hb_assembly_time_s),
        "nfail": np.zeros(len(freqs), dtype=int),
        "method": "pss_analytic_adjoint",
    }


def _time_domain_pac(all_sizes, tbias, freqs, *, pss_result, input_drive,
                     all_nf, corner=None, n_period_samples=768,
                     integration="gear2", cache=True, pacmag=1.0,
                     model_types=None, device_kwargs=None):
    """Time-domain (shooting) PAC — the formulation Spectre uses, ~10-20x cheaper
    than the harmonic-balance kernel and truncation-free.

    The small-signal response to a baseband input ``exp(j*w*t)`` is, by Floquet,
    ``x(t) = exp(j*w*t)*p(t)`` with ``p`` T-periodic, i.e. the NON-enveloped state
    satisfies the quasi-periodic boundary ``x(T) = exp(j*w*T)*x(0)``.  Integrating
    the linearized orbit one period gives the *frequency-independent* monodromy
    ``Psi`` (built once); per frequency we only integrate the forced response and
    solve the ``(exp(j*w*T)*I - Psi) x0 = g`` boundary system (n×n for backward
    Euler, 2n×2n for gear2/BDF2) — no ``(2K+1)n`` HB matrix and no sideband
    truncation, so it converges to the all-harmonic limit the HB approaches only
    as ``K->inf``.  Sideband-0 output = ``w . <exp(-j*w*t) x(t)>``.

    Scope: the rail-driven chopper (well-conditioned ``I-Psi`` since |Floquet|<1).
    Returns ``None`` — so the caller falls back to the HB adjoint — for true-MNA
    vsource drives, bordered (``n_branches>0``) systems, the stiff tau>>T case,
    or when the numba linearization is unavailable.
    """
    freqs = np.asarray(freqs, float)
    topo = pss_result["topology"]
    n = topo.n
    idx = topo.idx
    t_orbit = np.asarray(pss_result["t"], float)
    period = float(pss_result.get("period", t_orbit[-1] - t_orbit[0]))
    if period <= 0.0:
        return None
    N = int(n_period_samples)
    node_inputs = dict(pss_result.get("node_inputs", {}) or {})

    # Driven input NODES (rails) — identical detection to the HB adjoint path.
    drive_nodes = {}
    for node, key in node_inputs.items():
        if key in input_drive and node in topo.rails and node not in idx:
            drive_nodes[node] = drive_nodes.get(node, 0j) + complex(input_drive[key])
    dev_by_name = {name: (d, g, s) for name, d, g, s in topo.devices}
    for dev, key in (getattr(topo, "transient_inputs", {}) or {}).items():
        if key in input_drive and dev in dev_by_name:
            gate = dev_by_name[dev][1]
            if gate in topo.rails and gate not in idx:
                drive_nodes[gate] = drive_nodes.get(gate, 0j) + complex(input_drive[key])
    # Hand vsource/branch drives and bordered systems back to the HB adjoint (it
    # couples them through the branch-constraint row and is robust on stiff tau>>T).
    has_vsource_drive = any(
        len(vs) > 3 and isinstance(vs[3], str) and vs[3] in input_drive
        for vs in topo.vsources)
    if has_vsource_drive or getattr(topo, "n_branches", 0) or not drive_nodes:
        return None
    drive_list = list(drive_nodes)
    drive_amps = np.array([drive_nodes[node] for node in drive_list], dtype=complex)

    t_uniform = np.linspace(0.0, period, N, endpoint=False)
    node_wave = {
        node: np.interp(t_uniform, t_orbit,
                        np.asarray(pss_result["nodes"][node], float), period=period)
        for node in topo.solved}
    input_wave = {
        key: np.interp(t_uniform, t_orbit, np.asarray(val, float), period=period)
        for key, val in pss_result.get("inputs", {}).items()}

    cache_store = pss_result.setdefault("_pac_td_cache", {}) if cache else {}
    internal_gate_states, lin_dev_inst = _resolve_gate1_instances(
        all_sizes, all_nf, corner, topo,
        model_types=model_types, device_kwargs=device_kwargs)
    charge_caps = _conversion_charge_caps(pss_result, internal_gate_states)
    lin_key = (
        "pac_td_lin_gate1_v1", tuple(topo.solved), tuple(topo.devices),
        tuple(topo.resistors), tuple(topo.cap_list()), float(period), int(N),
        _freeze_sizes(all_sizes), _freeze_nf(all_nf), _freeze_kwargs(corner or {}),
        "charge_caps" if charge_caps else "cvddt_caps",
        bool(internal_gate_states),
        tuple(sorted(drive_list)),
        _freeze_complex_map(dict(zip(drive_list, drive_amps))))
    if cache and lin_key in cache_store:
        lin = cache_store[lin_key]
        Gt, Ct, Gin, Cin = lin["Gt"], lin["Ct"], lin["Gin"], lin["Cin"]
        n_gate1 = int(lin.get("n_gate1", max(0, Gt.shape[1] - n)))
    else:
        fast_lin = None
        gate1_fast = None
        if not internal_gate_states:
            fast_lin = _try_numba_pac_linearization(
                all_sizes, all_nf, corner, topo, tbias, t_uniform,
                node_wave, input_wave, node_inputs, drive_list, drive_amps,
                charge_caps=charge_caps, dev_inst=lin_dev_inst)
        elif not charge_caps:
            gate1_fast = _try_numba_pac_linearization_gate1(
                all_sizes, all_nf, corner, topo, tbias, t_uniform,
                node_wave, input_wave, node_inputs, drive_list, drive_amps,
                dev_inst=lin_dev_inst)
        if fast_lin is None and gate1_fast is None:
            Gt, Ct, Gin, Cin, n_gate1 = _assemble_pac_linearization_python(
                all_sizes, all_nf, corner, topo, tbias, t_uniform,
                node_wave, input_wave, node_inputs, drive_list, drive_amps,
                charge_caps=charge_caps,
                internal_gate_states=internal_gate_states,
                dev_inst=lin_dev_inst)
        else:
            if gate1_fast is not None:
                Gf, Cf, Ginf, Cinf, n_gate1 = gate1_fast
            else:
                Gf, Cf, Ginf, Cinf = fast_lin
                n_gate1 = 0
            # The HB FFTs these harmonics; we keep the time samples (uniform grid).
            Gt = np.fft.ifft(Gf * N, axis=0)
            Ct = np.fft.ifft(Cf * N, axis=0)
            Gin = np.fft.ifft(Ginf * N, axis=0)
            Cin = np.fft.ifft(Cinf * N, axis=0)
        if cache:
            cache_store[lin_key] = {
                "Gt": Gt, "Ct": Ct, "Gin": Gin, "Cin": Cin,
                "n_gate1": int(n_gate1),
            }

    n_state = Gt.shape[1]
    w = np.zeros(n_state, dtype=complex)
    for node, weight in topo.output_weights().items():
        if node in idx:
            w[idx[node]] = weight
    h = period / N
    tm = np.arange(N) * h
    I_n = np.eye(n_state, dtype=complex)

    # Frequency-INDEPENDENT one-period propagators + monodromy (built once).
    t_setup0 = time.perf_counter()
    Cin1 = np.roll(Cin, 1, axis=0)
    td_boundary_mode = "monodromy"
    td_chunk_steps = 0
    gear2_chunks = None
    gear2_A_chunks = None
    gear2_boundary_template = None
    if integration == "be":
        A = Ct / h + Gt
        Cprev = np.roll(Ct, 1, axis=0)
        luA = [lu_factor(A[m]) for m in range(N)]
        P = np.array([lu_solve(luA[m], Cprev[m] / h) for m in range(N)])
        Psi = I_n.copy()
        for m in range(1, N + 1):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                Psi = P[m % N] @ Psi
            if (not np.all(np.isfinite(Psi))) or _max_abs_finite(Psi) > _PAC_TD_GROWTH_LIMIT:
                return None
        sysdim = n_state
    else:  # gear2 / BDF2 on the uniform grid: (a0,a1,a2) = (3/2,-2,1/2)
        a0, a1, a2 = 1.5, -2.0, 0.5
        A = a0 * Ct / h + Gt
        C1 = np.roll(Ct, 1, axis=0); C2 = np.roll(Ct, 2, axis=0)
        Cin2 = np.roll(Cin, 2, axis=0)
        luA = [lu_factor(A[m]) for m in range(N)]
        Zn = np.zeros((n_state, n_state), dtype=complex)
        M = np.empty((N, 2 * n_state, 2 * n_state), dtype=complex)
        for m in range(N):
            B1 = lu_solve(luA[m], a1 * C1[m] / h)
            B2 = lu_solve(luA[m], a2 * C2[m] / h)
            M[m] = np.block([[-B1, -B2], [I_n, Zn]])
        Psi = np.eye(2 * n_state, dtype=complex)
        for m in range(1, N + 1):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                Psi = M[m % N] @ Psi
            if (not np.all(np.isfinite(Psi))) or _max_abs_finite(Psi) > _PAC_TD_GROWTH_LIMIT:
                td_boundary_mode = "multiple_shooting"
                Psi = None
                break
        if td_boundary_mode == "multiple_shooting":
            gear2_chunks, gear2_A_chunks, td_chunk_steps = _build_gear2_chunk_maps(M)
            gear2_boundary_template = _gear2_boundary_template(gear2_A_chunks)
        sysdim = 2 * n_state
    setup_time = time.perf_counter() - t_setup0
    Imat = np.eye(sysdim, dtype=complex)

    response = np.empty(len(freqs), dtype=complex)
    boundary_solve_time = 0.0
    replay_time = 0.0
    for fi, f in enumerate(freqs):
        wf = 2.0 * np.pi * float(f)
        ph = np.exp(1j * wf * tm)
        if integration == "be":
            php = np.exp(1j * wf * (tm - h))
            fm = -((Cin * ph[:, None] - Cin1 * php[:, None]) / h + Gin * ph[:, None])
            r = np.array([lu_solve(luA[m], fm[m]) for m in range(N)])
            g = np.zeros(n_state, dtype=complex)
            for m in range(1, N + 1):
                g = P[m % N] @ g + r[m % N]
            x0 = np.linalg.solve(np.exp(1j * wf * period) * Imat - Psi, g)
            x = np.empty((N, n_state), dtype=complex); x[0] = x0
            for m in range(1, N):
                x[m] = P[m] @ x[m - 1] + r[m]
        else:
            p1 = np.exp(1j * wf * (tm - h)); p2 = np.exp(1j * wf * (tm - 2 * h))
            fm = -((a0 * Cin * ph[:, None] + a1 * Cin1 * p1[:, None]
                    + a2 * Cin2 * p2[:, None]) / h + Gin * ph[:, None])
            s = np.zeros((N, 2 * n_state), dtype=complex)
            for m in range(N):
                s[m, :n_state] = lu_solve(luA[m], fm[m])
            gamma = np.exp(1j * wf * period)
            direct_ok = td_boundary_mode == "monodromy"
            if direct_ok:
                g = np.zeros(2 * n_state, dtype=complex)
                for m in range(1, N + 1):
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        g = M[m % N] @ g + s[m % N]
                    if (not np.all(np.isfinite(g))) or _max_abs_finite(g) > _PAC_TD_GROWTH_LIMIT:
                        direct_ok = False
                        break
            if direct_ok:
                z0 = np.linalg.solve(gamma * Imat - Psi, g)
                x = np.empty((N, n_state), dtype=complex)
                zprev = z0
                x[0] = z0[:n_state]
                for m in range(1, N):
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        zi = M[m] @ zprev + s[m]
                    if (not np.all(np.isfinite(zi))) or _max_abs_finite(zi) > _PAC_TD_GROWTH_LIMIT:
                        direct_ok = False
                        break
                    x[m] = zi[:n_state]
                    zprev = zi
            if not direct_ok:
                if gear2_chunks is None or gear2_A_chunks is None:
                    gear2_chunks, gear2_A_chunks, td_chunk_steps = _build_gear2_chunk_maps(M)
                    gear2_boundary_template = _gear2_boundary_template(gear2_A_chunks)
                    td_boundary_mode = "multiple_shooting"
                tb0 = time.perf_counter()
                z0 = _solve_gear2_chunk_boundary(
                    M, s, gamma, gear2_chunks, gear2_A_chunks,
                    boundary_template=gear2_boundary_template)
                boundary_solve_time += time.perf_counter() - tb0
                x = np.empty((N, n_state), dtype=complex)
                zprev = z0
                x[0] = z0[:n_state]
                tr0 = time.perf_counter()
                for m in range(1, N):
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        zprev = M[m] @ zprev + s[m]
                    if not np.all(np.isfinite(zprev)):
                        raise FloatingPointError("gear2 PAC trajectory overflow")
                    x[m] = zprev[:n_state]
                replay_time += time.perf_counter() - tr0
        X0 = (np.exp(-1j * wf * tm)[:, None] * x).mean(axis=0)
        response[fi] = w @ X0

    gains = np.abs(response)
    return {
        "freqs": freqs, "response": response, "gains": gains, "Hmag": gains,
        "Av_dc_dB": 20 * np.log10(max(float(gains[0]), 1e-300)) if len(gains) else np.nan,
        "bw_Hz": bw_from_gain(freqs, gains) if len(gains) else np.nan,
        "pss": pss_result, "input_drive": input_drive, "pacmag": float(pacmag),
        "pac_residual": np.zeros(len(freqs)), "pac_condition": np.ones(len(freqs)),
        "pac_period_runs": 0, "pac_cache_enabled": bool(cache),
        "pac_td_setup_time_s": float(setup_time),
        "pac_td_boundary_solve_time_s": float(boundary_solve_time),
        "pac_td_replay_time_s": float(replay_time),
        "pac_td_integration": str(integration), "pac_td_samples": int(N),
        "pac_td_boundary_mode": td_boundary_mode,
        "pac_td_chunk_steps": int(td_chunk_steps),
        "pac_state_size": int(n_state),
        "pac_internal_gate1_states": int(n_gate1),
        "nfail": np.zeros(len(freqs), dtype=int),
        "method": "pss_time_domain",
    }


def _resolve_compute_condition(compute_condition, profile=False, debug=False):
    if compute_condition is None:
        return bool(profile or debug)
    return bool(compute_condition)


def pac_solve(sizes: Mapping[str, tuple[float, float]], bias: Mapping[str, float],
              freqs: np.ndarray, *, pss_result: dict,
              input_drive: Mapping[str, complex],
              nf: int | Mapping[str, int] | None = None,
              corner: str | Mapping[str, Any] | None = None,
              fd_state_step: float = 1e-4, fd_input_step: float = 1e-4,
              transient_kwargs: Mapping[str, Any] | None = None,
              pacmag: float = 1.0, rail_margin: float | None = None,
              cache_linearization: bool = True,
              cache_forcing: bool = True, compute_condition: Any = None,
              lti_fast_path: bool = True,
              analytic: bool = True, n_period_samples: int = 384, max_sideband: int = 10,
              time_domain: bool = False, td_integration: str = "gear2",
              td_n_period_samples: int = 768,
              profile: bool = False, debug: bool = False,
              binding: CircuitBinding | None = None) -> dict:
    """Solve sideband-0 PAC around a PSS orbit.

    Parameters
    ----------
    sizes, bias
        Device sizes and bias dictionary for the topology in ``pss_result``.
    freqs
        Baseband PAC frequencies in Hz.
    pss_result
        Result returned by :func:`circuitopt.pss_solver.pss_solve` or a wrapper that
        preserves its fields. It must contain ``topology``, ``t``, ``nodes``,
        ``x0``, ``x_end``, and ``output``.
    input_drive
        Mapping ``input_key -> complex amplitude`` for a 1-unit small-signal
        ``exp(j*w*t)`` drive.  For a differential input, use e.g.
        ``{"vip": 0.5, "vin": -0.5}``.
    binding
        Optional :class:`CircuitBinding`. Only its ``nf`` and ``corner`` fill in
        for the matching signature kwargs — ``topo``/``model_types``/
        ``device_kwargs`` travel with ``pss_result`` and are NOT overridden.
        Explicit non-None ``nf``/``corner`` still win, and both still fall back to
        ``pss_result`` after this. binding=None reproduces the legacy path exactly.
    """
    _, nf, corner, _, _, _ = resolve_binding(binding, nf=nf, corner=corner)
    freqs = np.asarray(freqs, float)
    if np.any(freqs < 0.0):
        raise ValueError("PAC frequencies must be non-negative")
    if not input_drive:
        raise ValueError("input_drive must contain at least one driven input key")
    input_drive = dict(input_drive)
    transient_kwargs = dict(transient_kwargs or {})
    if corner is None:
        corner = pss_result.get("corner")
    # per-device model binding (silicon) travels with the PSS result
    model_types = pss_result.get("model_types")
    device_kwargs = pss_result.get("device_kwargs")
    device_kwargs, corner = apply_silicon_corner(
        model_types, device_kwargs, corner)
    compute_condition = _resolve_compute_condition(
        compute_condition, profile=profile, debug=debug)

    topo = pss_result["topology"]
    tgrid = np.asarray(pss_result["t"], float)
    period = float(pss_result.get("period", tgrid[-1] - tgrid[0]))
    if period <= 0.0:
        raise ValueError("PSS result must span one positive period")
    tbias = dict(pss_result.get("bias", bias))
    base_inputs = {
        key: np.asarray(val, float).copy()
        for key, val in pss_result.get("inputs", {}).items()
    }
    for key in input_drive:
        if key not in base_inputs:
            base_inputs[key] = np.zeros_like(tgrid)
    node_inputs = pss_result.get("node_inputs")
    current_inputs = pss_result.get(
        "current_inputs", pss_result.get("charge_injection_sources", ())
    )
    signed_devices = pss_result.get("signed_devices", ())
    all_sizes, all_nf = _merge_sizes_and_nf(sizes, nf, pss_result)

    solved = list(topo.solved)
    ybase = np.asarray(pss_result["output"], float)
    x0 = np.asarray(pss_result["x0"], float)
    xend_base = np.asarray(pss_result["x_end"], float)
    fd_state_step = float(fd_state_step)
    fd_input_step = float(fd_input_step)
    if fd_state_step <= 0.0 or fd_input_step <= 0.0:
        raise ValueError("finite-difference steps must be positive")
    if lti_fast_path:
        fast = _try_lti_ac_fast_path(all_sizes, tbias, freqs, pss_result,
                                     input_drive, all_nf, corner=corner,
                                     compute_condition=compute_condition,
                                     model_types=model_types,
                                     device_kwargs=device_kwargs)
        if fast is not None:
            fast["pacmag"] = float(pacmag)
            return fast

    if time_domain and analytic:
        td = _time_domain_pac(
            all_sizes, tbias, freqs, pss_result=pss_result,
            input_drive=input_drive, all_nf=all_nf, corner=corner,
            n_period_samples=td_n_period_samples, integration=td_integration,
            cache=cache_linearization, pacmag=pacmag,
            model_types=model_types, device_kwargs=device_kwargs)
        if td is not None:
            return td

    if analytic:
        ana = _analytic_adjoint_pac(
            all_sizes, tbias, freqs, pss_result=pss_result,
            input_drive=input_drive, all_nf=all_nf, corner=corner,
            n_period_samples=n_period_samples, max_sideband=max_sideband,
            cache=cache_linearization, pacmag=pacmag,
            compute_condition=compute_condition,
            model_types=model_types, device_kwargs=device_kwargs)
        if ana is not None:
            return ana

    common_tr = dict(
        topo=topo,
        inputs=base_inputs,
        node_inputs=node_inputs,
        current_inputs=current_inputs,
        nf=all_nf,
        corner=corner,
        model_types=model_types,
        device_kwargs=device_kwargs,
        V0=x0,
        max_step=pss_result.get("transient_max_step"),
        flat_max_step=pss_result.get("transient_flat_max_step"),
        max_retry_subdivisions=0,
        newton_maxit=60,
        newton_step_limit=2.0,
        fallback_least_squares=False,
        signed_devices=signed_devices,
        rail_margin=pss_result.get("rail_margin", 2.0)
        if rail_margin is None else rail_margin,
    )
    common_tr.update(transient_kwargs)
    cache_key_base = (
        "pac_fd_v1",
        tuple(solved),
        len(tgrid),
        float(period),
        _freeze_sizes(all_sizes),
        _freeze_nf(all_nf),
        _freeze_kwargs(transient_kwargs),
        repr(node_inputs),
        repr(current_inputs),
        repr(tuple(signed_devices or ())),
        repr(common_tr.get("max_step")),
        repr(common_tr.get("flat_max_step")),
        repr(common_tr.get("max_retry_subdivisions")),
        repr(common_tr.get("newton_maxit")),
        repr(common_tr.get("newton_step_limit")),
        repr(common_tr.get("newton_vtol")),
        repr(common_tr.get("fallback_full_jacobian")),
        repr(common_tr.get("fallback_least_squares")),
        repr(common_tr.get("fallback_tol")),
        repr(common_tr.get("rail_margin")),
    )
    pac_cache = pss_result.setdefault("_pac_cache", {}) if (
        cache_linearization or cache_forcing
    ) else {}

    def run_with(v0, perturb=None):
        inputs = dict(base_inputs)
        if perturb:
            for key, wave in perturb.items():
                inputs[key] = inputs.get(key, np.zeros_like(tgrid)) + fd_input_step * wave
        tr_kwargs = dict(common_tr)
        tr_kwargs["inputs"] = inputs
        tr_kwargs["V0"] = np.asarray(v0, float)
        return transient(all_sizes, tbias, tgrid, profile=False, **tr_kwargs)

    n = topo.n
    state_key = cache_key_base + ("state", float(fd_state_step))
    state_cached = bool(cache_linearization and state_key in pac_cache)
    state_period_runs = 0
    if state_cached:
        state = pac_cache[state_key]
        phi = np.asarray(state["phi"], float)
        y_cols = np.asarray(state["y_cols"], float)
    else:
        phi = np.empty((n, n), dtype=float)
        y_cols = np.empty((len(tgrid), n), dtype=float)
        for col in range(n):
            step = fd_state_step * max(1.0, abs(float(x0[col])))
            xp = x0.copy()
            xp[col] += step
            trp = run_with(xp)
            state_period_runs += 1
            phi[:, col] = (
                np.asarray([trp["nodes"][node][-1] for node in solved]) - xend_base
            ) / step
            y_cols[:, col] = (np.asarray(trp["output"], float) - ybase) / step
        if cache_linearization:
            pac_cache[state_key] = {"phi": phi.copy(), "y_cols": y_cols.copy()}

    out_response = np.empty(len(freqs), dtype=complex)
    residuals = np.empty(len(freqs), dtype=float)
    conditions = np.empty(len(freqs), dtype=float)
    nfail = np.empty(len(freqs), dtype=int)
    input_period_runs = 0
    input_cache_hits = 0
    input_drive_key = _freeze_complex_map(input_drive)
    for pos, freq in enumerate(freqs):
        omega = 2.0 * np.pi * float(freq)
        forcing_key = cache_key_base + (
            "forcing", float(fd_input_step), input_drive_key, float(freq),
        )
        forcing_cached = bool(cache_forcing and forcing_key in pac_cache)
        if forcing_cached:
            forcing = pac_cache[forcing_key]
            b_end = np.asarray(forcing["b_end"], complex)
            b_y = np.asarray(forcing["b_y"], complex)
            nfail[pos] = int(forcing["nfail"])
            input_cache_hits += 1
        else:
            q0, q90 = _input_quadratures(input_drive, tgrid, omega)
            tr0 = run_with(x0, q0)
            tr90 = run_with(x0, q90)
            input_period_runs += 2
            b_end = (
                np.asarray([tr0["nodes"][node][-1] for node in solved]) - xend_base
                + 1j * (
                    np.asarray([tr90["nodes"][node][-1] for node in solved]) - xend_base
                )
            ) / fd_input_step
            b_y = (
                np.asarray(tr0["output"], float) - ybase
                + 1j * (np.asarray(tr90["output"], float) - ybase)
            ) / fd_input_step
            nfail[pos] = int(tr0.get("nfail", 0)) + int(tr90.get("nfail", 0))
            if cache_forcing:
                pac_cache[forcing_key] = {
                    "b_end": b_end.copy(),
                    "b_y": b_y.copy(),
                    "nfail": int(nfail[pos]),
                }
        gamma = np.exp(1j * omega * period)
        mat = phi.astype(complex) - gamma * np.eye(n, dtype=complex)
        if compute_condition:
            try:
                cond = float(np.linalg.cond(mat))
            except Exception as exc:
                diagnostics.note("pac.boundary_condition_number_fail", exc)
                cond = np.inf
        else:
            cond = np.nan
        conditions[pos] = cond
        try:
            if compute_condition and ((not np.isfinite(cond)) or cond > 1e12):
                raise np.linalg.LinAlgError("ill-conditioned PAC boundary matrix")
            dx0 = np.linalg.solve(mat, -b_end)
        except np.linalg.LinAlgError:
            dx0 = np.linalg.lstsq(mat, -b_end, rcond=None)[0]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            y_env = b_y + y_cols @ dx0
        if not np.all(np.isfinite(y_env)):
            y_env = np.nan_to_num(y_env, nan=0.0, posinf=0.0, neginf=0.0)
        out_response[pos] = _periodic_average(
            tgrid, y_env * np.exp(-1j * omega * tgrid)
        )
        residuals[pos] = float(np.linalg.norm(mat @ dx0 + b_end, ord=np.inf))

    gains = np.abs(out_response)
    return {
        "freqs": freqs,
        "response": out_response,
        "gains": gains,
        "Hmag": gains,
        "Av_dc_dB": 20 * np.log10(max(float(gains[0]), 1e-300))
        if len(gains) else np.nan,
        "bw_Hz": bw_from_gain(freqs, gains) if len(gains) else np.nan,
        "pss": pss_result,
        "input_drive": input_drive,
        "pacmag": float(pacmag),
        "pac_residual": residuals,
        "pac_condition": conditions,
        "pac_state_period_runs": int(state_period_runs),
        "pac_input_period_runs": int(input_period_runs),
        "pac_period_runs": int(state_period_runs + input_period_runs),
        "pac_state_cache_hit": bool(state_cached),
        "pac_input_cache_hits": int(input_cache_hits),
        "pac_cache_enabled": bool(cache_linearization or cache_forcing),
        "pac_condition_computed": bool(compute_condition),
        "nfail": nfail,
        "method": "pss_finite_difference_shooting",
    }
