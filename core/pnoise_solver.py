"""Generic PSS-based periodic-noise solver.

This module contains the topology-independent harmonic-balance conversion
matrix used for PNoise.  Wrappers such as the PMOS chopper only provide a PSS
orbit, optional PAC gains, and device-specific noise policy.
"""
from __future__ import annotations

import numpy as np

try:
    from .ac_mna import _stamp_adm, _stamp_mos_lti
    from .ac_solver import _dev_corner, get_ss_params
    from .noise_solver import band_rms, noise_analysis
    from .numba_kernels import pnoise_fold_psd_numba, pnoise_hb_blocks_numba
    from .pac_solver import (
        _freeze_kwargs,
        _freeze_nf,
        _freeze_sizes,
        _is_constant_wave,
        pac_solve,
    )
    from .pmos_tft_model import PMOS_TFT
except ImportError:  # pragma: no cover - legacy direct module import
    from ac_mna import _stamp_adm, _stamp_mos_lti
    from ac_solver import _dev_corner, get_ss_params
    from noise_solver import band_rms, noise_analysis
    from numba_kernels import pnoise_fold_psd_numba, pnoise_hb_blocks_numba
    from pac_solver import (
        _freeze_kwargs,
        _freeze_nf,
        _freeze_sizes,
        _is_constant_wave,
        pac_solve,
    )
    from pmos_tft_model import PMOS_TFT


_KB = 1.380649e-23
_TEMP = 300.15


def _merge_sizes_and_nf(sizes, nf, pss_result):
    all_sizes = dict(sizes)
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


def _nfval(all_nf, name):
    if isinstance(all_nf, dict):
        return int(all_nf.get(name, 1))
    return int(all_nf) if all_nf else 1


def _periodic_interp(t_src, y_src, t_dst, period):
    return np.interp(t_dst, t_src, np.asarray(y_src, float), period=period)


def _static_bias_from_pss(topo, tbias, pss_result):
    if pss_result.get("current_inputs"):
        return None
    if any(not _is_constant_wave(v) for v in pss_result.get("inputs", {}).values()):
        return None
    if any(not _is_constant_wave(pss_result["nodes"][node]) for node in topo.solved):
        return None

    out = dict(tbias)
    inputs = pss_result.get("inputs", {})
    node_inputs = dict(pss_result.get("node_inputs", {}) or {})
    for node, key in node_inputs.items():
        if key not in inputs or node not in topo.rails:
            return None
        value = float(np.asarray(inputs[key], float)[0])
        ref = topo.rails[node]
        if isinstance(ref, str):
            out[ref] = value
        elif abs(float(ref) - value) > 1e-9 * max(1.0, abs(value)):
            return None

    dev_by_name = {name: (d, g, s) for name, d, g, s in topo.devices}
    for dev, key in getattr(topo, "transient_inputs", {}).items():
        if key not in inputs or dev not in dev_by_name:
            return None
        gate = dev_by_name[dev][1]
        if gate in topo.idx or gate not in topo.rails:
            return None
        value = float(np.asarray(inputs[key], float)[0])
        ref = topo.rails[gate]
        if isinstance(ref, str):
            out[ref] = value
        elif abs(float(ref) - value) > 1e-9 * max(1.0, abs(value)):
            return None
    return out


def _try_lti_noise_fast_path(sizes, bias, freqs, *, pss_result, nf, corner,
                             band, gains, pac_result, input_drive,
                             noise_devices, gds_noise_devices):
    if noise_devices is not None or gds_noise_devices:
        return None
    topo = pss_result["topology"]
    tbias = _static_bias_from_pss(topo, dict(pss_result.get("bias", bias)), pss_result)
    if tbias is None:
        return None
    x0_guess = dict(zip(topo.solved, np.asarray(pss_result["x0"], float)))
    noise = noise_analysis(
        sizes, tbias, freqs, corner=corner, x0_guess=x0_guess, topo=topo, nf=nf,
    )
    if noise is None:
        return None
    if gains is None:
        if pac_result is None:
            if input_drive is None:
                raise ValueError("gains, pac_result, or input_drive is required")
            pac_result = pac_solve(
                sizes, tbias, freqs, pss_result=pss_result,
                input_drive=input_drive, nf=nf,
            )
        gains = pac_result["gains"]
    gains = np.asarray(gains, float)
    out_psd = np.asarray(noise["out_psd"], float)
    irn_psd = out_psd / np.maximum(gains ** 2, 1e-300)
    return {
        "freqs": np.asarray(freqs, float),
        "f_chop": 1.0 / float(pss_result.get("period", 1.0)),
        "fundamental": 1.0 / float(pss_result.get("period", 1.0)),
        "out_psd": out_psd,
        "out_asd": np.sqrt(out_psd),
        "dev_psd": noise.get("dev_psd", {}),
        "gains": gains,
        "irn_psd": irn_psd,
        "irn_uV_band": band_rms(freqs, irn_psd, band[0], band[1]) * 1e6,
        "out_uV_band": band_rms(freqs, out_psd, band[0], band[1]) * 1e6,
        "max_sideband": 0,
        "n_period_samples": 0,
        "pss": pss_result,
        "pac": pac_result,
        "noise": noise,
        "method": "lti_noise_fast_path",
        "pnoise_linearization_cache_hit": False,
        "pnoise_hb_cache_hit": False,
        "pnoise_adjoint_cache_hits": 0,
        "pnoise_cache_enabled": False,
        "pnoise_hb_size": 0,
        "pnoise_hb_solve_count": 0,
        "pnoise_noise_source_count": len(noise.get("dev_psd", {})),
        "pnoise_numba_hb_used": False,
        "pnoise_numba_fold_used": False,
    }


def _hb_blocks(Gf, Cf, K, N, n, fundamental):
    use_numba = (
        pnoise_hb_blocks_numba is not None and
        (2 * int(K) + 1) * int(n) >= 16
    )
    if use_numba:
        return pnoise_hb_blocks_numba(
            np.asarray(Gf, dtype=np.complex128),
            np.asarray(Cf, dtype=np.complex128),
            int(K),
            float(fundamental),
        ) + (True,)

    nb = 2 * K + 1
    Y_base = np.zeros((nb * n, nb * n), dtype=complex)
    C_block = np.zeros_like(Y_base)
    for kr in range(-K, K + 1):
        row_omega = 2j * np.pi * kr * fundamental
        br = (kr + K) * n
        for kc in range(-K, K + 1):
            bc = (kc + K) * n
            g_coeff = Gf[(kr - kc) % N]
            c_coeff = Cf[(kr - kc) % N]
            Y_base[br:br + n, bc:bc + n] = g_coeff + row_omega * c_coeff
            C_block[br:br + n, bc:bc + n] = c_coeff
    return Y_base, C_block, False


def pnoise_solve(sizes, bias, freqs, *, pss_result, fundamental=None, nf=None,
                 corner=None, max_sideband=10, n_period_samples=384,
                 band=(0.05, 100.0), gains=None, pac_result=None,
                 input_drive=None, noise_devices=None,
                 gds_noise_devices=None, switch_noise_conductance_gated=True,
                 cache_linearization=True, lti_fast_path=True):
    """Solve periodic output noise around a PSS orbit.

    The circuit is linearized along the supplied PSS trajectory.  Fourier
    coefficients of ``G(t)`` and ``C(t)`` form the harmonic-balance conversion
    matrix, and an adjoint solve folds cyclostationary device/resistor noise to
    the baseband output.

    If ``gains`` is not supplied, pass either ``pac_result`` or ``input_drive``;
    the latter lets this function run :func:`pac_solve` and compute input-
    referred noise with the same sideband-0 convention.
    """
    freqs = np.asarray(freqs, float)
    if np.any(freqs < 0.0):
        raise ValueError("PNoise frequencies must be non-negative")
    topo = pss_result["topology"]
    n = topo.n
    idx = topo.idx
    t_orbit = np.asarray(pss_result["t"], float)
    period = float(pss_result.get("period", t_orbit[-1] - t_orbit[0]))
    if period <= 0.0:
        raise ValueError("PSS result must span one positive period")
    if fundamental is None:
        fundamental = 1.0 / period
    fundamental = float(fundamental)
    if fundamental <= 0.0:
        raise ValueError("fundamental must be positive")

    N = int(n_period_samples)
    if N < 2:
        raise ValueError("n_period_samples must be at least 2")
    K = int(max_sideband)
    if K < 0:
        raise ValueError("max_sideband must be non-negative")
    tbias = dict(pss_result.get("bias", bias))
    all_sizes, all_nf = _merge_sizes_and_nf(sizes, nf, pss_result)
    if lti_fast_path:
        fast = _try_lti_noise_fast_path(
            all_sizes, tbias, freqs, pss_result=pss_result, nf=all_nf,
            corner=corner, band=band, gains=gains, pac_result=pac_result,
            input_drive=input_drive, noise_devices=noise_devices,
            gds_noise_devices=gds_noise_devices,
        )
        if fast is not None:
            fast["f_chop"] = fundamental
            fast["fundamental"] = fundamental
            return fast

    rails = topo.rail_values(tbias)
    node_inputs = pss_result.get("node_inputs", {})
    t_uniform = np.linspace(0.0, period, N, endpoint=False)

    node_wave = {
        node: _periodic_interp(t_orbit, pss_result["nodes"][node], t_uniform, period)
        for node in topo.solved
    }
    input_wave = {
        key: _periodic_interp(t_orbit, val, t_uniform, period)
        for key, val in pss_result.get("inputs", {}).items()
    }
    devices = list(topo.devices)
    gated_noise = set(gds_noise_devices or ())
    if not switch_noise_conductance_gated:
        gated_noise = set()
    keep = set(noise_devices) if noise_devices is not None else None

    def term_value(node, m):
        if node in idx:
            return node_wave[node][m]
        if node in node_inputs:
            return input_wave[node_inputs[node]][m]
        return rails[node]

    def term(node):
        return ("n", idx[node]) if node in idx else ("v", 0.0)

    lin_key = (
        "pnoise_lin_v1",
        tuple(topo.solved),
        tuple(topo.devices),
        tuple(topo.resistors),
        tuple(topo.cap_list()),
        float(period),
        int(N),
        _freeze_sizes(all_sizes),
        _freeze_nf(all_nf),
        _freeze_kwargs(corner or {}),
        tuple(sorted(gated_noise)),
        bool(switch_noise_conductance_gated),
    )
    cache = pss_result.setdefault("_pnoise_cache", {}) if cache_linearization else {}
    cache_hit = bool(cache_linearization and lin_key in cache)
    if cache_hit:
        lin = cache[lin_key]
        Gf = lin["Gf"]
        Cf = lin["Cf"]
        all_noise_sources = lin["noise_sources"]
    else:
        dev_inst = {
            name: PMOS_TFT(
                W=all_sizes[name][0], L=all_sizes[name][1],
                NF=_nfval(all_nf, name), **_dev_corner(corner, name),
            )
            for name, *_ in devices
        }
        G_const = np.zeros((n, n))
        C_const = np.zeros((n, n))
        rhs_g = np.zeros(n)
        rhs_c = np.zeros(n)
        for a, b, cap in topo.cap_list():
            _stamp_adm(C_const, rhs_c, term(a), term(b), cap)
        for _, a, b, R in topo.resistors:
            _stamp_adm(G_const, rhs_g, term(a), term(b), 1.0 / R)
        for k in range(n):
            G_const[k, k] += 1e-12

        Gt = np.zeros((N, n, n))
        Ct = np.zeros((N, n, n))
        Sth = np.zeros((len(devices), N))
        Sfl = np.zeros((len(devices), N))
        for m in range(N):
            Gm = Gt[m]
            Cm = Ct[m]
            Gm += G_const
            Cm += C_const
            for j, (name, d, g, s) in enumerate(devices):
                Vs = term_value(s, m)
                Vd = term_value(d, m)
                Vg = term_value(g, m)
                p = get_ss_params(
                    all_sizes[name][0], all_sizes[name][1], Vs, Vd, Vg,
                    corner=_dev_corner(corner, name), nf=_nfval(all_nf, name),
                    dev_inst=dev_inst[name],
                )
                _stamp_mos_lti(
                    Gm, Cm, rhs_g, rhs_c, term(d), term(g), term(s),
                    p["gm"], p["gds"], p["Cgs"], p["Cgd"],
                )
                if name in gated_noise:
                    S_th = 4.0 * _KB * _TEMP * abs(p["gds"])
                    S_fl1 = 0.0
                else:
                    try:
                        S_th, S_fl1 = dev_inst[name].get_noise_psd(
                            Vs, Vd, Vg, frequency=1.0
                        )
                    except Exception:
                        S_th, S_fl1 = 0.0, 0.0
                Sth[j, m] = max(float(S_th), 0.0)
                Sfl[j, m] = max(float(S_fl1), 0.0)

        Gf = np.fft.fft(Gt, axis=0) / N
        Cf = np.fft.fft(Ct, axis=0) / N
        Sthf = np.fft.fft(Sth, axis=1) / N
        Sflf = np.fft.fft(Sfl, axis=1) / N

        all_noise_sources = []
        for j, (name, d, _g, s) in enumerate(devices):
            all_noise_sources.append((name, idx.get(d), idx.get(s), Sthf[j], Sflf[j]))
        for name, a, b, R in topo.resistors:
            sth = np.zeros(N, dtype=complex)
            sfl = np.zeros(N, dtype=complex)
            sth[0] = 4.0 * _KB * _TEMP / float(R)
            all_noise_sources.append((name, idx.get(a), idx.get(b), sth, sfl))
        if cache_linearization:
            cache[lin_key] = {
                "Gf": Gf,
                "Cf": Cf,
                "noise_sources": all_noise_sources,
            }

    noise_sources = [
        item for item in all_noise_sources
        if keep is None or item[0] in keep
    ]

    nb = 2 * K + 1
    ks = np.arange(-K, K + 1)
    m_grid = ks[:, None] - ks[None, :]
    out_w = topo.output_weights()

    def coeff(F, k):
        return F[np.asarray(k) % N]

    hb_key = ("pnoise_hb_v1", lin_key, int(K), float(fundamental))
    hb_cache_hit = bool(cache_linearization and hb_key in cache)
    if hb_cache_hit:
        hb = cache[hb_key]
        Y_base = hb["Y_base"]
        C_block = hb["C_block"]
        hb_numba_used = bool(hb.get("numba_used", False))
    else:
        Y_base, C_block, hb_numba_used = _hb_blocks(Gf, Cf, K, N, n, fundamental)
        if cache_linearization:
            cache[hb_key] = {
                "Y_base": Y_base,
                "C_block": C_block,
                "numba_used": bool(hb_numba_used),
            }

    harm_offsets = np.arange(nb, dtype=int) * n
    source_grids = [
        (
            name,
            None if pi is None else harm_offsets + int(pi),
            None if qi is None else harm_offsets + int(qi),
            coeff(sth_coeff, m_grid),
            coeff(sfl_coeff, m_grid),
        )
        for name, pi, qi, sth_coeff, sfl_coeff in noise_sources
    ]
    e = np.zeros(nb * n, dtype=complex)
    base0 = K * n
    for node, weight in out_w.items():
        e[base0 + idx[node]] = weight

    adj_cache = pss_result.setdefault("_pnoise_adjoint_cache", {}) if cache_linearization else {}
    adj_prefix = (
        "pnoise_adj_v1",
        lin_key,
        int(K),
        float(fundamental),
        tuple((node, float(weight)) for node, weight in sorted(out_w.items())),
    )
    adj_cache_hits = 0
    hb_solve_count = 0
    adjs = np.empty((len(freqs), nb * n), dtype=complex)
    for fi, freq in enumerate(freqs):
        freq = float(freq)
        adj_key = adj_prefix + (freq,)
        if cache_linearization and adj_key in adj_cache:
            adj = adj_cache[adj_key]
            adj_cache_hits += 1
        else:
            Y = Y_base + (2j * np.pi * freq) * C_block
            try:
                adj = np.linalg.solve(Y.T, e)
            except np.linalg.LinAlgError:
                adj = np.linalg.lstsq(Y.T, e, rcond=None)[0]
            hb_solve_count += 1
            if cache_linearization:
                adj_cache[adj_key] = adj
        adjs[fi] = adj

    fold_work = len(freqs) * max(1, len(source_grids)) * nb * nb
    use_numba_fold = pnoise_fold_psd_numba is not None and fold_work >= 1000
    source_names = [name for name, *_ in source_grids]
    if use_numba_fold:
        ns = len(source_grids)
        p_indices = np.full((ns, nb), -1, dtype=np.int64)
        q_indices = np.full((ns, nb), -1, dtype=np.int64)
        sth_stack = np.empty((ns, nb, nb), dtype=np.complex128)
        sfl_stack = np.empty((ns, nb, nb), dtype=np.complex128)
        for si, (_name, p_idx, q_idx, sth_grid, sfl_grid) in enumerate(source_grids):
            if p_idx is not None:
                p_indices[si] = np.asarray(p_idx, dtype=np.int64)
            if q_idx is not None:
                q_indices[si] = np.asarray(q_idx, dtype=np.int64)
            sth_stack[si] = np.asarray(sth_grid, dtype=np.complex128)
            sfl_stack[si] = np.asarray(sfl_grid, dtype=np.complex128)
        out_psd, dev_stack = pnoise_fold_psd_numba(
            np.asarray(adjs, dtype=np.complex128),
            np.asarray(freqs, dtype=np.float64),
            int(K),
            float(fundamental),
            p_indices,
            q_indices,
            sth_stack,
            sfl_stack,
        )
        dev_psd = {
            name: np.asarray(dev_stack[si], dtype=float)
            for si, name in enumerate(source_names)
        }
    else:
        out_psd = np.zeros(len(freqs))
        dev_psd = {name: np.zeros(len(freqs)) for name in source_names}
        for fi, freq in enumerate(freqs):
            freq = float(freq)
            adj = adjs[fi]
            nu = np.abs(freq + ks * fundamental)
            nu[nu < 1e-9] = 1e-9
            inv_sqrt_nu = 1.0 / np.sqrt(nu)
            flick_freq = np.outer(inv_sqrt_nu, inv_sqrt_nu)

            for name, p_idx, q_idx, sth_grid, sfl_grid in source_grids:
                if p_idx is None:
                    Z = np.zeros(nb, dtype=complex)
                else:
                    Z = adj[p_idx].copy()
                if q_idx is not None:
                    Z -= adj[q_idx]
                Smat = sth_grid + sfl_grid * flick_freq
                contrib = float(np.real(Z @ (Smat @ np.conj(Z))))
                contrib = max(contrib, 0.0)
                dev_psd[name][fi] += contrib
                out_psd[fi] += contrib

    if gains is None:
        if pac_result is None:
            if input_drive is None:
                raise ValueError("gains, pac_result, or input_drive is required")
            pac_result = pac_solve(
                sizes, bias, freqs, pss_result=pss_result,
                input_drive=input_drive, nf=nf,
            )
        gains = pac_result["gains"]
    gains = np.asarray(gains, float)
    irn_psd = out_psd / np.maximum(gains ** 2, 1e-300)
    return {
        "freqs": freqs,
        "f_chop": fundamental,
        "fundamental": fundamental,
        "out_psd": out_psd,
        "out_asd": np.sqrt(out_psd),
        "dev_psd": dev_psd,
        "gains": gains,
        "irn_psd": irn_psd,
        "irn_uV_band": band_rms(freqs, irn_psd, band[0], band[1]) * 1e6,
        "out_uV_band": band_rms(freqs, out_psd, band[0], band[1]) * 1e6,
        "max_sideband": K,
        "n_period_samples": N,
        "pss": pss_result,
        "pac": pac_result,
        "pnoise_linearization_cache_hit": bool(cache_hit),
        "pnoise_hb_cache_hit": bool(hb_cache_hit),
        "pnoise_adjoint_cache_hits": int(adj_cache_hits),
        "pnoise_cache_enabled": bool(cache_linearization),
        "pnoise_hb_size": int(nb * n),
        "pnoise_hb_solve_count": int(hb_solve_count),
        "pnoise_noise_source_count": int(len(noise_sources)),
        "pnoise_numba_hb_used": bool(hb_numba_used),
        "pnoise_numba_fold_used": bool(use_numba_fold),
        "method": "pss_harmonic_balance_conversion_matrix",
    }
