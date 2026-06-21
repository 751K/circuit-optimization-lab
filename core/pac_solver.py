"""Generic PSS-assisted periodic AC solver.

The solver operates on an already-computed PSS orbit.  Circuit-specific wrappers
only need to provide the periodic operating point and the small-signal drive
definition; the finite-difference shooting PAC kernel is topology independent.
"""
from __future__ import annotations

import numpy as np

try:
    from .ac_mna import _stamp_adm, _stamp_mos_lti
    from .ac_solver import _dev_corner, ac_solve, get_ss_params
    from .pmos_tft_model import PMOS_TFT
    from .topology import Topology
    from .transient_solver import transient
except ImportError:  # pragma: no cover - legacy direct module import
    from ac_mna import _stamp_adm, _stamp_mos_lti
    from ac_solver import _dev_corner, ac_solve, get_ss_params
    from pmos_tft_model import PMOS_TFT
    from topology import Topology
    from transient_solver import transient


def _periodic_average(t, values):
    t = np.asarray(t, float)
    values = np.asarray(values)
    period = float(t[-1] - t[0])
    if period <= 0.0:
        return np.mean(values, axis=0)
    return np.trapezoid(values, t, axis=0) / period


def _bw_from_gain(freqs, gains):
    freqs = np.asarray(freqs, float)
    gains = np.asarray(gains, float)
    if len(freqs) == 0 or len(gains) == 0:
        return np.nan
    peak = float(np.max(gains))
    a3 = peak / np.sqrt(2.0)
    ipk = int(np.argmax(gains))
    bw = float(freqs[-1])
    for i in range(ipk + 1, len(gains)):
        if gains[i] <= a3:
            f0, f1 = float(freqs[i - 1]), float(freqs[i])
            g0, g1 = float(gains[i - 1]), float(gains[i])
            if g1 == g0:
                bw = f1
            elif f0 > 0.0 and f1 > 0.0:
                x0, x1 = np.log10(f0), np.log10(f1)
                x = x0 + (a3 - g0) * (x1 - x0) / (g1 - g0)
                bw = float(10.0 ** np.clip(x, min(x0, x1), max(x0, x1)))
            else:
                bw = float(f0 + (a3 - g0) * (f1 - f0) / (g1 - g0))
            break
    return bw


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
    except Exception:
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
        return int(mode_id) in (0, 3)
    mode = str(pss_result.get("transient_cap_mode", "charge")).lower()
    return mode in {"charge", "q", "qstamp", "q-stamp",
                    "branch", "self", "self-charge"}


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
    except Exception:
        return
    controls = (s, d, g)
    for ctrl, (dCgs, dCgd) in zip(controls, derivs):
        if vdot_gs != 0.0:
            _stamp_branch_control(G, g, s, ctrl, -dCgs * vdot_gs)
        if vdot_gd != 0.0:
            _stamp_branch_control(G, g, d, ctrl, -dCgd * vdot_gd)


def _try_lti_ac_fast_path(sizes, bias, freqs, pss_result, input_drive, nf,
                          corner=None,
                          compute_condition=False):
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
        dc_tol=topo.dc_tol,
        require_dc_in_box=topo.require_dc_in_box,
    )
    ac = ac_solve(
        sizes, tbias, freqs, topo=fast_topo, nf=nf, corner=corner,
        x0_guess=dict(zip(topo.solved, np.asarray(pss_result["x0"], float))),
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
        "bw_Hz": _bw_from_gain(freqs, gains) if len(gains) else np.nan,
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


def _nfval(all_nf, name):
    if isinstance(all_nf, dict):
        return int(all_nf.get(name, 1))
    return int(all_nf) if all_nf else 1


def _analytic_adjoint_pac(all_sizes, tbias, freqs, *, pss_result, input_drive,
                          all_nf, corner=None, n_period_samples=384,
                          max_sideband=10, cache=True, pacmag=1.0,
                          compute_condition=False):
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
    rails = topo.rail_values(tbias)
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
    if not drive_nodes:
        return None
    drive_list = list(drive_nodes)
    ext_idx = {node: n + i for i, node in enumerate(drive_list)}
    n_ext = n + len(drive_list)
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

    charge_caps = _charge_linearized_caps(pss_result)
    cache_store = pss_result.setdefault("_pac_analytic_cache", {}) if cache else {}
    lin_key = (
        "pac_analytic_lin_v3", tuple(topo.solved), tuple(topo.devices),
        tuple(topo.resistors), tuple(topo.cap_list()), float(period), int(N),
        _freeze_sizes(all_sizes), _freeze_nf(all_nf), _freeze_kwargs(corner or {}),
        "charge_caps" if charge_caps else "veriloga_caps",
        tuple(sorted(drive_list)),
        _freeze_complex_map(dict(zip(drive_list, drive_amps))),
    )
    if cache and lin_key in cache_store:
        lin = cache_store[lin_key]
        Gf, Cf, Ginf, Cinf = lin["Gf"], lin["Cf"], lin["Ginf"], lin["Cinf"]
    else:
        dev_inst = {
            name: PMOS_TFT(W=all_sizes[name][0], L=all_sizes[name][1],
                           NF=_nfval(all_nf, name), **_dev_corner(corner, name))
            for name, *_ in topo.devices
        }
        G_const = np.zeros((n_ext, n_ext)); C_const = np.zeros((n_ext, n_ext))
        rg = np.zeros(n_ext); rc = np.zeros(n_ext)
        for a, b, cap in topo.cap_list():
            _stamp_adm(C_const, rc, term(a), term(b), cap)
        for _, a, b, R in topo.resistors:
            _stamp_adm(G_const, rg, term(a), term(b), 1.0 / R)
        for k in range(n):
            G_const[k, k] += 1e-12
        Gt = np.zeros((N, n_ext, n_ext)); Ct = np.zeros((N, n_ext, n_ext))
        for m in range(N):
            Gt[m] += G_const; Ct[m] += C_const
            for name, d, g, s in topo.devices:
                Vs = term_value(s, m); Vd = term_value(d, m); Vg = term_value(g, m)
                p = get_ss_params(all_sizes[name][0], all_sizes[name][1], Vs, Vd, Vg,
                                  corner=_dev_corner(corner, name),
                                  nf=_nfval(all_nf, name), dev_inst=dev_inst[name])
                _stamp_mos_lti(Gt[m], Ct[m], rg, rc, term(d), term(g), term(s),
                               p["gm"], p["gds"], p["Cgs"], p["Cgd"])
                if not charge_caps:
                    _stamp_pmos_dynamic_cap_terms(
                        Gt[m], term(d), term(g), term(s), dev_inst[name],
                        Vs, Vd, Vg,
                        term_derivative(s, m), term_derivative(d, m),
                        term_derivative(g, m))
        Gf = np.fft.fft(Gt[:, :n, :n], axis=0) / N
        Cf = np.fft.fft(Ct[:, :n, :n], axis=0) / N
        Ginf = np.fft.fft(Gt[:, :n, n:] @ drive_amps, axis=0) / N      # (N, n)
        Cinf = np.fft.fft(Ct[:, :n, n:] @ drive_amps, axis=0) / N
        if cache:
            cache_store[lin_key] = {"Gf": Gf, "Cf": Cf, "Ginf": Ginf, "Cinf": Cinf}

    Y_base = np.zeros((nb * n, nb * n), dtype=complex)
    C_block = np.zeros_like(Y_base)
    for kr in range(-K, K + 1):
        br = (kr + K) * n
        for kc in range(-K, K + 1):
            bc = (kc + K) * n
            sideband = kr if charge_caps else kc
            sideband_omega = 2j * np.pi * sideband * fundamental
            g = Gf[(kr - kc) % N]; c = Cf[(kr - kc) % N]
            Y_base[br:br + n, bc:bc + n] = g + sideband_omega * c
            C_block[br:br + n, bc:bc + n] = c
    e = np.zeros(nb * n, dtype=complex)
    base0 = K * n
    for node, w in topo.output_weights().items():
        e[base0 + idx[node]] = w

    response = np.empty(len(freqs), dtype=complex)
    residuals = np.zeros(len(freqs))
    conditions = np.ones(len(freqs))
    for fi, f in enumerate(freqs):
        f = float(f)
        Y = Y_base + (2j * np.pi * f) * C_block
        if compute_condition:
            try:
                conditions[fi] = float(np.linalg.cond(Y))
            except Exception:
                conditions[fi] = np.inf
        try:
            adj = np.linalg.solve(Y.T, e)
        except np.linalg.LinAlgError:
            adj = np.linalg.lstsq(Y.T, e, rcond=None)[0]
        b = np.zeros(nb * n, dtype=complex)
        for kr in range(-K, K + 1):
            input_om = 2j * np.pi * (f + kr * fundamental) if charge_caps else 2j * np.pi * f
            b[(kr + K) * n:(kr + K + 1) * n] = -(
                Ginf[kr % N] + input_om * Cinf[kr % N])
        response[fi] = adj @ b

    gains = np.abs(response)
    return {
        "freqs": freqs,
        "response": response,
        "gains": gains,
        "Hmag": gains,
        "Av_dc_dB": 20 * np.log10(max(float(gains[0]), 1e-300)) if len(gains) else np.nan,
        "bw_Hz": _bw_from_gain(freqs, gains) if len(gains) else np.nan,
        "pss": pss_result,
        "input_drive": input_drive,
        "pacmag": float(pacmag),
        "pac_residual": residuals,
        "pac_condition": conditions,
        "pac_state_period_runs": 0,
        "pac_input_period_runs": 0,
        "pac_period_runs": 0,
        "pac_state_cache_hit": bool(cache and lin_key in cache_store),
        "pac_input_cache_hits": 0,
        "pac_cache_enabled": bool(cache),
        "pac_condition_computed": bool(compute_condition),
        "pac_hb_size": int(nb * n),
        "nfail": np.zeros(len(freqs), dtype=int),
        "method": "pss_analytic_adjoint",
    }


def _resolve_compute_condition(compute_condition, profile=False, debug=False):
    if compute_condition is None:
        return bool(profile or debug)
    return bool(compute_condition)


def pac_solve(sizes, bias, freqs, *, pss_result, input_drive, nf=None,
              corner=None,
              fd_state_step=1e-4, fd_input_step=1e-4, transient_kwargs=None,
              pacmag=1.0, rail_margin=None, cache_linearization=True,
              cache_forcing=True, compute_condition=None, lti_fast_path=True,
              analytic=True, n_period_samples=384, max_sideband=10,
              profile=False, debug=False):
    """Solve sideband-0 PAC around a PSS orbit.

    Parameters
    ----------
    sizes, bias
        Device sizes and bias dictionary for the topology in ``pss_result``.
    freqs
        Baseband PAC frequencies in Hz.
    pss_result
        Result returned by :func:`core.pss_solver.pss_solve` or a wrapper that
        preserves its fields. It must contain ``topology``, ``t``, ``nodes``,
        ``x0``, ``x_end``, and ``output``.
    input_drive
        Mapping ``input_key -> complex amplitude`` for a 1-unit small-signal
        ``exp(j*w*t)`` drive.  For a differential input, use e.g.
        ``{"vip": 0.5, "vin": -0.5}``.
    """
    freqs = np.asarray(freqs, float)
    if np.any(freqs < 0.0):
        raise ValueError("PAC frequencies must be non-negative")
    if not input_drive:
        raise ValueError("input_drive must contain at least one driven input key")
    input_drive = dict(input_drive)
    transient_kwargs = dict(transient_kwargs or {})
    if corner is None:
        corner = pss_result.get("corner")
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
                                     compute_condition=compute_condition)
        if fast is not None:
            fast["pacmag"] = float(pacmag)
            return fast

    if analytic:
        ana = _analytic_adjoint_pac(
            all_sizes, tbias, freqs, pss_result=pss_result,
            input_drive=input_drive, all_nf=all_nf, corner=corner,
            n_period_samples=n_period_samples, max_sideband=max_sideband,
            cache=cache_linearization, pacmag=pacmag,
            compute_condition=compute_condition)
        if ana is not None:
            return ana

    common_tr = dict(
        topo=topo,
        inputs=base_inputs,
        node_inputs=node_inputs,
        current_inputs=current_inputs,
        nf=all_nf,
        corner=corner,
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
            except Exception:
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
        "bw_Hz": _bw_from_gain(freqs, gains) if len(gains) else np.nan,
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
