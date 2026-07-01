"""Generic PSS-based periodic-noise solver.

This module contains the topology-independent PNoise conversion machinery:
the harmonic-balance path used as the generic fallback/comparison path and the
time-domain Floquet adjoint used by the chopper wrapper to avoid HB sideband
truncation. Wrappers such as the PMOS chopper only provide a PSS orbit, optional
PAC gains, and device-specific noise policy.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

try:
    from scipy import linalg as _la
    from scipy import sparse as _sp
    from scipy.sparse import linalg as _spla
except Exception:  # pragma: no cover - scipy is a project dependency
    _la = None
    _sp = None
    _spla = None

from .ac_mna import _branch_incidence
from .ac_solver import _dev_corner, _dev_nf, build_devices, get_ss_params
from .noise_solver import band_rms, noise_analysis
from .numba_kernels import (_pnoise_hb_blocks_impl, pnoise_fold_psd_numba,
                            pnoise_hb_blocks_numba, py_impl)
from .pac_solver import (
    _assemble_pac_linearization_python,
    _conversion_charge_caps,
    _freeze_kwargs,
    _freeze_nf,
    _freeze_sizes,
    _is_constant_wave,
    pac_solve,
)
from . import diagnostics


_KB = 1.380649e-23
_TEMP = 300.15
_HB_SOLVERS = {"auto", "dense", "sparse", "iterative"}


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
                input_drive=input_drive, nf=nf, corner=corner,
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


def _hb_blocks(Gf, Cf, K, N, n, fundamental, *, charge_caps=False):
    """Dense HB conversion blocks. Single-sourced onto ``_pnoise_hb_blocks_impl``
    (jitted for large systems, interpreted `.py_func` below the JIT-worthwhile
    size). See ``docs/single_source_impl_plan.md``."""
    use_numba = (
        pnoise_hb_blocks_numba is not None and
        (2 * int(K) + 1) * int(n) >= 16
    )
    kernel = _pnoise_hb_blocks_impl if use_numba else py_impl(_pnoise_hb_blocks_impl)
    Y_base, C_block = kernel(
        np.asarray(Gf, dtype=np.complex128),
        np.asarray(Cf, dtype=np.complex128),
        int(K), float(fundamental), bool(charge_caps))
    return Y_base, C_block, use_numba


def _hb_blocks_sparse(Gf, Cf, K, N, n, fundamental, drop_tol=0.0,
                      *, charge_caps=False):
    if _sp is None:
        raise RuntimeError("scipy.sparse is required for sparse PNoise HB")
    nb = 2 * K + 1
    size = nb * n
    y_rows = []
    y_cols = []
    y_data = []
    c_rows = []
    c_cols = []
    c_data = []
    for kr_i in range(nb):
        kr = kr_i - K
        br = kr_i * n
        for kc_i in range(nb):
            kc = kc_i - K
            sideband = kr if charge_caps else kc
            sideband_omega = 2.0j * np.pi * sideband * fundamental
            bc = kc_i * n
            coeff_idx = (kr - kc) % N
            c_block = Cf[coeff_idx]
            y_block = Gf[coeff_idx] + sideband_omega * c_block
            if drop_tol > 0.0:
                y_nz = np.nonzero(np.abs(y_block) > drop_tol)
                c_nz = np.nonzero(np.abs(c_block) > drop_tol)
            else:
                y_nz = np.nonzero(y_block)
                c_nz = np.nonzero(c_block)
            if len(y_nz[0]):
                y_rows.extend((br + y_nz[0]).tolist())
                y_cols.extend((bc + y_nz[1]).tolist())
                y_data.extend(y_block[y_nz].tolist())
            if len(c_nz[0]):
                c_rows.extend((br + c_nz[0]).tolist())
                c_cols.extend((bc + c_nz[1]).tolist())
                c_data.extend(c_block[c_nz].tolist())
    Y_base = _sp.csc_matrix((y_data, (y_rows, y_cols)), shape=(size, size),
                            dtype=np.complex128)
    C_block = _sp.csc_matrix((c_data, (c_rows, c_cols)), shape=(size, size),
                             dtype=np.complex128)
    Y_base.sum_duplicates()
    C_block.sum_duplicates()
    Y_base.eliminate_zeros()
    C_block.eliminate_zeros()
    return Y_base, C_block


def _to_sparse_hb(Y_base, C_block):
    if _sp is None:
        raise RuntimeError("scipy.sparse is required for sparse PNoise HB")
    Y_sparse = _sp.csc_matrix(Y_base)
    C_sparse = _sp.csc_matrix(C_block)
    Y_sparse.eliminate_zeros()
    C_sparse.eliminate_zeros()
    return Y_sparse, C_sparse


def _sparse_density(mat):
    if mat is None:
        return 1.0
    total = mat.shape[0] * mat.shape[1]
    return 0.0 if total == 0 else float(mat.nnz) / float(total)


def _estimate_hb_sparse_density(Gf, Cf, K, N, n, drop_tol=0.0):
    nb = 2 * K + 1
    counts = np.zeros(N, dtype=float)
    for k in range(N):
        if drop_tol > 0.0:
            mask = (np.abs(Gf[k]) > drop_tol) | (np.abs(Cf[k]) > drop_tol)
        else:
            mask = (Gf[k] != 0.0) | (Cf[k] != 0.0)
        counts[k] = float(np.count_nonzero(mask))
    total = 0.0
    for kr in range(-K, K + 1):
        for kc in range(-K, K + 1):
            total += counts[(kr - kc) % N]
    denom = float(nb * nb * n * n)
    return 0.0 if denom == 0.0 else total / denom


def _resolve_hb_solver(requested, hb_size, sparse_density,
                       sparse_min_size, sparse_max_density):
    solver = str(requested or "auto").lower()
    if solver not in _HB_SOLVERS:
        raise ValueError(f"hb_solver must be one of {sorted(_HB_SOLVERS)}")
    if solver != "auto":
        if solver in {"sparse", "iterative"} and (_sp is None or _spla is None):
            return "dense"
        return solver
    if (_sp is not None and _spla is not None and
            int(hb_size) >= int(sparse_min_size) and
            float(sparse_density) <= float(sparse_max_density)):
        return "sparse"
    return "dense"


def _block_jacobi_preconditioner(Gf, Cf, K, n, fundamental, freq):
    if _spla is None:
        return None
    nb = 2 * K + 1
    G0 = np.asarray(Gf[0], dtype=np.complex128)
    C0 = np.asarray(Cf[0], dtype=np.complex128)
    factors = []
    for ki in range(nb):
        kh = ki - K
        block = (G0 + (2.0j * np.pi * (float(freq) + kh * fundamental)) * C0).T
        if _la is not None:
            try:
                factors.append(("lu", _la.lu_factor(block)))
                continue
            except Exception as exc:
                diagnostics.note("pnoise.lu_factor_fail", exc)
        factors.append(("dense", block))

    def apply(rhs):
        rhs = np.asarray(rhs, dtype=np.complex128)
        out = np.empty_like(rhs)
        for ki, factor in enumerate(factors):
            lo = ki * n
            hi = lo + n
            mode, data = factor
            try:
                if mode == "lu" and _la is not None:
                    out[lo:hi] = _la.lu_solve(data, rhs[lo:hi])
                else:
                    out[lo:hi] = np.linalg.solve(data, rhs[lo:hi])
            except Exception as exc:
                diagnostics.note("pnoise.block_solve_lstsq", exc)
                mat = data if mode != "lu" else (
                    G0 + (2.0j * np.pi * (float(freq) + (ki - K) * fundamental)) * C0
                ).T
                out[lo:hi] = np.linalg.lstsq(mat, rhs[lo:hi], rcond=None)[0]
        return out

    size = nb * n
    return _spla.LinearOperator((size, size), matvec=apply, dtype=np.complex128)


def _solve_hb_adjoint(Y_base, C_block, Y_sparse, C_sparse, freq, e, solver,
                      iterative_tol, iterative_maxiter, preconditioner=None):
    omega = 2j * np.pi * float(freq)
    info = {
        "solver": solver,
        "iterative_info": 0,
        "iterative_iterations": 0,
        "iterative_fallback": False,
        "dense_fallback": False,
        "preconditioner": "none" if preconditioner is None else "block_jacobi",
    }
    if solver == "dense":
        Y = Y_base + omega * C_block
        try:
            return np.linalg.solve(Y.T, e), info
        except np.linalg.LinAlgError:
            info["dense_fallback"] = True
            return np.linalg.lstsq(Y.T, e, rcond=None)[0], info

    if _sp is None or _spla is None:
        info["solver"] = "dense"
        info["dense_fallback"] = True
        Y = Y_base + omega * C_block
        try:
            return np.linalg.solve(Y.T, e), info
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(Y.T, e, rcond=None)[0], info

    Y = Y_sparse + omega * C_sparse
    A = Y.T.tocsc()
    if solver == "iterative":
        maxiter = None if iterative_maxiter is None else int(iterative_maxiter)
        iter_count = {"n": 0}

        def _cb(_value):
            iter_count["n"] += 1

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    adj, gmres_info = _spla.gmres(
                        A, e, rtol=float(iterative_tol), atol=0.0,
                        maxiter=maxiter, M=preconditioner, callback=_cb,
                        callback_type="pr_norm")
                except TypeError:
                    adj, gmres_info = _spla.gmres(
                        A, e, rtol=float(iterative_tol), atol=0.0,
                        maxiter=maxiter, M=preconditioner, callback=_cb)
            info["iterative_info"] = int(gmres_info)
            info["iterative_iterations"] = int(iter_count["n"])
            if gmres_info == 0 and np.all(np.isfinite(adj)):
                return adj, info
        except Exception as exc:
            diagnostics.note("pnoise.gmres_fail", exc)
            info["iterative_info"] = -1
        info["iterative_fallback"] = True

    try:
        adj = _spla.spsolve(A, e)
        if np.all(np.isfinite(adj)):
            return adj, info
    except Exception as exc:
        diagnostics.note("pnoise.spsolve_fail", exc)
    info["dense_fallback"] = True
    if Y_base is None or C_block is None:
        dense = Y.toarray()
    else:
        dense = Y_base + omega * C_block
    try:
        return np.linalg.solve(dense.T, e), info
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(dense.T, e, rcond=None)[0], info


def _time_domain_pnoise_adjoint(Gf, Cf, e, freqs, K, n_state, fundamental):
    """Truncation-free pnoise adjoint via the Floquet BVP (dual of the TD-PAC).

    The cyclostationary fold only needs the per-device per-sideband adjoint
    transfer ``Z_r``; the HB path gets them from ``Y_HB^H adj = e`` truncated at
    K.  Here we solve the periodic block-bidiagonal ``F^H ζ = c``, where ``F`` is
    the BE time-stepping operator of the SAME ``G(t)/C(t)`` (``Gf/Cf`` ifft'd).
    Real Gt/Ct ⇒ ``F^H = F^T`` except the periodic corner phase (``1/γ``); the
    output weight ``w`` is the baseband block of ``e``; ``Z_r = FFT(ζ·e^{jωt})
    [(-r)%N]`` assembled into the ``[(r+K)·n_state+node]`` vector the fold indexes.
    The adjoint is exact in the sideband index (no K-truncation of the conversion);
    its only error is the BE 1st-order time discretization, controlled by N
    (``n_period_samples``; converged by ~640).  Returns ``adjs`` (nfreq, nb·n).

    **Factor-once (Woodbury).** ``F(γ)`` differs from a reference ``F(γ0)`` ONLY in
    the ``ns×ns`` periodic corner ``-BT[0]/γ`` (the block-bidiagonal bulk is
    frequency-independent).  So we ``splu`` ``F(γ0)`` ONCE and correct each
    frequency with a rank-``ns`` update instead of refactoring the ``N·ns`` sparse
    matrix per frequency (measured 6.6× faster, bit-identical to ~1e-13).  With
    ``F(γ)=F(γ0)+U·d·Wᵀ`` (``U`` = corner block-row ``N-1``, ``W`` = block-col 0,
    ``d=-BT[0](1/γ−1/γ0)``)::

        F(γ)⁻¹c = y0 − Z (I + d·M0)⁻¹ d·q,   y0=F(γ0)⁻¹c, Z=F(γ0)⁻¹U,
                                              M0=Z[block0], q=y0[block0]

    ``γ0`` = the median in-band frequency keeps ``|d|`` small; ``F(γ0)`` is the
    well-conditioned periodic operator (bounded inverse ⇒ no open-loop blow-up, so
    the direct-``splu`` overflow guard is preserved).  Any degeneracy (a Floquet
    resonance ⇒ singular ``I+d·M0`` ⇒ a non-finite ``ζ``) falls back to a fresh
    per-frequency ``splu``, keeping bit-parity with the direct solve.
    """
    if _spla is None or _sp is None:
        return None
    N = int(Gf.shape[0]); ns = int(n_state)
    period = 1.0 / float(fundamental); h = period / N
    Gt = np.fft.ifft(Gf * N, axis=0); Ct = np.fft.ifft(Cf * N, axis=0)
    tm = np.arange(N) * h
    w = np.asarray(e[K * ns:(K + 1) * ns], dtype=complex)  # baseband block = output weights
    A = Ct / h + Gt; Bm = np.roll(Ct, 1, axis=0) / h
    AT = np.transpose(A, (0, 2, 1)); BT = np.transpose(Bm, (0, 2, 1))
    rb = np.arange(ns)[:, None].repeat(ns, 1); cb = np.arange(ns)[None, :].repeat(ns, 0)
    # frequency-independent block-bidiagonal of F^H: diag AT[m], super -BT[m+1].
    ri = []; ci = []; va = []
    for mm in range(N):
        ri.append((mm * ns + rb).ravel()); ci.append((mm * ns + cb).ravel()); va.append(AT[mm].ravel())
        if mm < N - 1:
            ri.append((mm * ns + rb).ravel()); ci.append(((mm + 1) * ns + cb).ravel())
            va.append((-BT[mm + 1]).ravel())
    RI = np.concatenate(ri); CI = np.concatenate(ci); VA = np.concatenate(va)
    crow = ((N - 1) * ns + rb).ravel(); ccol = cb.ravel()      # corner (row N-1, col 0)
    Rall = np.concatenate([RI, crow]); Call = np.concatenate([CI, ccol])
    nb = 2 * K + 1
    freqs = np.asarray(freqs, dtype=float)
    adjs = np.empty((len(freqs), nb * ns), dtype=complex)

    def _rhs(wf):
        return ((1.0 / N) * np.exp(-1j * wf * tm)[:, None] * w[None, :]).ravel()

    def _zeta_direct(wf):
        """Robust reference: build F(γ) and factor it fresh (per-frequency splu)."""
        gamma = np.exp(1j * wf * period)
        Vall = np.concatenate([VA, (-BT[0] / gamma).ravel()])
        F = _sp.csc_matrix((Vall, (Rall, Call)), shape=(N * ns, N * ns))
        return _spla.splu(F).solve(_rhs(wf)).reshape(N, ns)

    def _store(fi, wf, zeta):
        Fh = np.fft.fft(zeta * np.exp(1j * wf * tm)[:, None], axis=0)
        for j, rr in enumerate(range(-K, K + 1)):
            adjs[fi, j * ns:(j + 1) * ns] = Fh[(-rr) % N]

    # Factor F(γ0) once (γ0 = median in-band freq); reuse it across all frequencies.
    lu = None
    try:
        ref = len(freqs) // 2
        g0 = np.exp(1j * 2.0 * np.pi * float(freqs[ref]) * period); inv_g0 = 1.0 / g0
        Vall0 = np.concatenate([VA, (-BT[0] * inv_g0).ravel()])
        lu = _spla.splu(_sp.csc_matrix((Vall0, (Rall, Call)), shape=(N * ns, N * ns)))
        U = np.zeros((N * ns, ns), dtype=complex)              # corner block-row N-1 = e_{N-1} ⊗ I
        U[(N - 1) * ns + np.arange(ns), np.arange(ns)] = 1.0
        Z = lu.solve(U); M0 = Z[:ns]; Ins = np.eye(ns, dtype=complex)
    except Exception as exc:                                   # pragma: no cover
        diagnostics.note("pnoise.td_woodbury_setup_fail", exc)
        lu = None

    n_fallback = 0
    for fi, f in enumerate(freqs):
        wf = 2.0 * np.pi * float(f)
        zeta = None
        if lu is not None:
            try:
                # numpy's BLAS matmul sets spurious FPE flags on padding lanes; the
                # np.isfinite guard below catches any *real* blow-up and falls back.
                with np.errstate(all='ignore'):
                    y0 = lu.solve(_rhs(wf))
                    d = -BT[0] * (1.0 / np.exp(1j * wf * period) - inv_g0)
                    m = np.linalg.solve(Ins + d @ M0, d @ y0[:ns])
                    z = y0 - Z @ m
                if np.isfinite(z).all():
                    zeta = z.reshape(N, ns)
            except Exception:
                zeta = None
        if zeta is None:
            n_fallback += 1
            zeta = _zeta_direct(wf)
        _store(fi, wf, zeta)
    if lu is not None and n_fallback:
        diagnostics.note("pnoise.td_woodbury_freq_fallback",
                         detail=f"{n_fallback}/{len(freqs)} freqs")
    return adjs


def pnoise_solve(sizes, bias, freqs, *, pss_result, fundamental=None, nf=None,
                 corner=None, max_sideband=10, n_period_samples=384,
                 time_domain=False,
                 band=(0.05, 100.0), gains=None, pac_result=None,
                 input_drive=None, noise_devices=None,
                 gds_noise_devices=None, switch_noise_conductance_gated=True,
                 cache_linearization=True, lti_fast_path=True,
                 hb_solver="auto", hb_sparse_min_size=384,
                 hb_sparse_max_density=0.12, hb_sparse_drop_tol=0.0,
                 iterative_tol=1e-10, iterative_maxiter=10,
                 profile=False):
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
    if corner is None:
        corner = pss_result.get("corner")
    if fundamental is None:
        fundamental = 1.0 / period
    fundamental = float(fundamental)
    if fundamental <= 0.0:
        raise ValueError("fundamental must be positive")

    N = int(n_period_samples)
    # The truncation-free TD adjoint is BE 1st-order in time; it needs a finer
    # grid than the (time-resolution-insensitive) HB fold to converge. Bump a
    # default-ish N up so the TD matches Cadence (~640+; see N-convergence note).
    if time_domain and N < 640:
        N = 768
    if N < 2:
        raise ValueError("n_period_samples must be at least 2")
    K = int(max_sideband)
    if K < 0:
        raise ValueError("max_sideband must be non-negative")
    # The HB conversion matrix reads G/C harmonics up to order 2K; sampling the
    # orbit below 4K points aliases those coefficients (sharp switch edges carry
    # slowly-decaying harmonics), so lift N above the Nyquist limit we depend on.
    N = max(N, 4 * K + 2)
    hb_solver_requested = str(hb_solver or "auto").lower()
    if hb_solver_requested not in _HB_SOLVERS:
        raise ValueError(f"hb_solver must be one of {sorted(_HB_SOLVERS)}")
    hb_sparse_min_size = int(hb_sparse_min_size)
    hb_sparse_max_density = float(hb_sparse_max_density)
    hb_sparse_drop_tol = float(hb_sparse_drop_tol)
    pnoise_warnings = []

    def add_warning(code, message):
        item = {"code": str(code), "message": str(message)}
        pnoise_warnings.append(item)
        warnings.warn(f"PNoise: {message}", RuntimeWarning, stacklevel=2)

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
    # Spectre PNoise linearizes the Verilog-A small-signal C(V)*ddt(V) operator
    # (not the local transient Q-stamp companion used to integrate the PSS orbit);
    # the charge fold under-counts the hard-switched chopper noise by ~20%. The
    # operator choice goes through the SAME decision helper as PAC (single source
    # of truth): with internal gate-state retention it returns False (C(V)*ddt(V)).
    internal_gate_states = True
    charge_caps = _conversion_charge_caps(pss_result, internal_gate_states)
    if not switch_noise_conductance_gated:
        gated_noise = set()
    keep = set(noise_devices) if noise_devices is not None else None

    def term_value(node, m):
        if node in idx:
            return node_wave[node][m]
        if node in node_inputs:
            return input_wave[node_inputs[node]][m]
        return rails[node]

    lin_key = (
        "pnoise_lin_gate1_v1",
        tuple(topo.solved),
        tuple(topo.devices),
        tuple(topo.resistors),
        tuple(topo.cap_list()),
        float(period),
        int(N),
        _freeze_sizes(all_sizes),
        _freeze_nf(all_nf),
        _freeze_kwargs(corner or {}),
        "charge_caps" if charge_caps else "cvddt_caps",
        bool(internal_gate_states),
        tuple(sorted(gated_noise)),
        bool(switch_noise_conductance_gated),
    )
    cache = pss_result.setdefault("_pnoise_cache", {}) if cache_linearization else {}
    cache_hit = bool(cache_linearization and lin_key in cache)
    t_linear0 = time.perf_counter()
    if cache_hit:
        lin = cache[lin_key]
        Gf = lin["Gf"]
        Cf = lin["Cf"]
        all_noise_sources = lin["noise_sources"]
        n_state = int(lin.get("n_state", Gf.shape[1]))
        n_gate1 = int(lin.get("n_gate1", max(0, n_state - n)))
        noise_failure_count = int(lin.get("noise_failure_count", 0))
        noise_failure_devices = tuple(lin.get("noise_failure_devices", ()))
        noise_failure_reason = str(lin.get("noise_failure_reason", ""))
    else:
        dev_inst = build_devices(all_sizes, nf=all_nf, corner=corner, topo=topo)
        Gt, Ct, _gdrive, _cdrive, n_gate1 = _assemble_pac_linearization_python(
            all_sizes, all_nf, corner, topo, tbias, t_uniform,
            node_wave, input_wave, node_inputs, (), np.empty(0, dtype=complex),
            charge_caps=charge_caps,
            internal_gate_states=internal_gate_states,
            dev_inst=dev_inst,
        )
        n_state = Gt.shape[1]
        Sth = np.zeros((len(devices), N))
        Sfl = np.zeros((len(devices), N))
        noise_failure_count = 0
        noise_failure_devices = set()
        noise_failure_reason = ""
        for m in range(N):
            for j, (name, d, g, s) in enumerate(devices):
                Vs = term_value(s, m)
                Vd = term_value(d, m)
                Vg = term_value(g, m)
                p = get_ss_params(
                    all_sizes[name][0], all_sizes[name][1], Vs, Vd, Vg,
                    corner=_dev_corner(corner, name), nf=_dev_nf(all_nf, name),
                    dev_inst=dev_inst[name],
                )
                if name in gated_noise:
                    S_th = 4.0 * _KB * _TEMP * abs(p["gds"])
                    S_fl1 = 0.0
                else:
                    try:
                        S_th, S_fl1 = dev_inst[name].get_noise_psd(
                            Vs, Vd, Vg, frequency=1.0
                        )
                    except Exception as exc:
                        diagnostics.note("pnoise.device_noise_zeroed", exc)
                        noise_failure_count += 1
                        noise_failure_devices.add(name)
                        if not noise_failure_reason:
                            noise_failure_reason = type(exc).__name__
                        S_th, S_fl1 = 0.0, 0.0
                Sth[j, m] = max(float(S_th), 0.0)
                Sfl[j, m] = max(float(S_fl1), 0.0)

        Gf = np.fft.fft(Gt, axis=0) / N
        Cf = np.fft.fft(Ct, axis=0) / N
        Sthf = np.fft.fft(Sth, axis=1) / N
        # Flicker is a MODULATED stationary 1/f source i(t)=m(t)*n(t) with the
        # modulation amplitude m(t)=sqrt(PWR(t)); its cyclostationary harmonic
        # matrix must be built from the harmonics of m(t)=sqrt(PWR), NOT of the
        # power PWR(t).  (Thermal is white, so FFT(power) is correct for it.)
        # Using FFT(PWR) + a separable 1/sqrt(nu_k nu_l) weight over-counts a
        # strongly-modulated 1/f source by <PWR>/<sqrt(PWR)>^2 -- ~1 for a
        # constant-bias device (saturated amp), but several-fold for a hard
        # switch whose PWR ∝ Ich(t)^2 spikes during conduction.
        Mflf = np.fft.fft(np.sqrt(Sfl), axis=1) / N

        all_noise_sources = []
        for j, (name, d, _g, s) in enumerate(devices):
            all_noise_sources.append((name, idx.get(d), idx.get(s), Sthf[j], Mflf[j]))
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
                "n_state": int(n_state),
                "n_gate1": int(n_gate1),
                "noise_failure_count": int(noise_failure_count),
                "noise_failure_devices": tuple(sorted(noise_failure_devices)),
                "noise_failure_reason": str(noise_failure_reason),
            }
    noise_failure_devices = tuple(sorted(noise_failure_devices))
    if noise_failure_count:
        devices_s = ", ".join(noise_failure_devices)
        add_warning(
            "device_noise_unsupported",
            "device noise evaluation failed for "
            f"{noise_failure_count} orbit samples on {devices_s}; "
            "those device noise contributions were set to zero. "
            "This degradation path is not fully supported yet.",
        )
    linearization_time_s = time.perf_counter() - t_linear0

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

    hb_key = ("pnoise_hb_v2", lin_key, int(K), float(fundamental),
              float(hb_sparse_drop_tol),
              "charge_caps" if charge_caps else "cvddt_caps")
    hb_cache_hit = bool(cache_linearization and hb_key in cache)
    hb_size = int(nb * n_state)
    sparse_density_estimate = _estimate_hb_sparse_density(
        Gf, Cf, K, N, n_state, drop_tol=hb_sparse_drop_tol)
    prefer_sparse = (
        hb_solver_requested in {"sparse", "iterative"} or
        (hb_solver_requested == "auto" and hb_size >= hb_sparse_min_size and
         _sp is not None and _spla is not None and
         sparse_density_estimate <= hb_sparse_max_density)
    )
    if hb_solver_requested in {"sparse", "iterative"} and (_sp is None or _spla is None):
        add_warning(
            "hb_sparse_unavailable",
            f"hb_solver={hb_solver_requested!r} requested, but scipy.sparse is "
            "unavailable; falling back to dense. This sparse/iterative PNoise "
            "degradation path is not fully supported yet.",
        )
    t_hb0 = time.perf_counter()
    Y_base = None
    C_block = None
    Y_sparse = None
    C_sparse = None
    hb_numba_used = False
    if hb_cache_hit:
        hb = cache[hb_key]
        Y_base = hb.get("Y_base")
        C_block = hb.get("C_block")
        Y_sparse = hb.get("Y_base_sparse")
        C_sparse = hb.get("C_block_sparse")
        hb_numba_used = bool(hb.get("numba_used", False))
    else:
        hb = {}
    if prefer_sparse and Y_sparse is None:
        try:
            if Y_base is not None and C_block is not None and hb_sparse_drop_tol == 0.0:
                Y_sparse, C_sparse = _to_sparse_hb(Y_base, C_block)
            else:
                Y_sparse, C_sparse = _hb_blocks_sparse(
                    Gf, Cf, K, N, n_state, fundamental,
                    drop_tol=hb_sparse_drop_tol, charge_caps=charge_caps)
            hb["Y_base_sparse"] = Y_sparse
            hb["C_block_sparse"] = C_sparse
        except Exception as exc:
            diagnostics.note("pnoise.sparse_assembly_failed", exc)
            add_warning(
                "hb_sparse_assembly_failed",
                "sparse/iterative HB matrix assembly failed "
                f"({type(exc).__name__}); falling back to dense. This PNoise "
                "degradation path is not fully supported yet.",
            )
            Y_sparse = None
            C_sparse = None
    sparse_density = (_sparse_density(Y_sparse)
                      if Y_sparse is not None else sparse_density_estimate)
    solver_used = _resolve_hb_solver(
        hb_solver_requested, hb_size, sparse_density,
        hb_sparse_min_size, hb_sparse_max_density)
    need_dense = solver_used == "dense" or Y_sparse is None or C_sparse is None
    if need_dense and (Y_base is None or C_block is None):
        Y_base, C_block, hb_numba_used = _hb_blocks(
            Gf, Cf, K, N, n_state, fundamental, charge_caps=charge_caps)
        hb["Y_base"] = Y_base
        hb["C_block"] = C_block
        hb["numba_used"] = bool(hb_numba_used)
        if prefer_sparse and (Y_sparse is None or C_sparse is None) and hb_sparse_drop_tol == 0.0:
            try:
                Y_sparse, C_sparse = _to_sparse_hb(Y_base, C_block)
                hb["Y_base_sparse"] = Y_sparse
                hb["C_block_sparse"] = C_sparse
                sparse_density = _sparse_density(Y_sparse)
                solver_used = _resolve_hb_solver(
                    hb_solver_requested, hb_size, sparse_density,
                    hb_sparse_min_size, hb_sparse_max_density)
            except Exception as exc:
                diagnostics.note("pnoise.to_sparse_fail", exc)
    if solver_used in {"sparse", "iterative"} and (Y_sparse is None or C_sparse is None):
        solver_used = "dense"
        add_warning(
            "hb_sparse_matrix_missing",
            f"hb_solver={hb_solver_requested!r} could not build a sparse HB "
            "matrix; falling back to dense. This PNoise degradation path is not "
            "fully supported yet.",
        )
    sparse_nnz = int(Y_sparse.nnz) if Y_sparse is not None else 0
    if cache_linearization:
        cache[hb_key] = hb
    hb_assembly_time_s = time.perf_counter() - t_hb0

    harm_offsets = np.arange(nb, dtype=int) * n_state
    # thermal: Toeplitz of the power harmonics (white -> correct).
    # flicker: the sqrt(PWR) modulation harmonics M_{-2K..2K}; the fold builds the
    # cyclostationary matrix S_kl = sum_a M_{k-a} M*_{l-a} / nu_a from them.
    mvec_idx = np.arange(-2 * K, 2 * K + 1)
    source_grids = [
        (
            name,
            None if pi is None else harm_offsets + int(pi),
            None if qi is None else harm_offsets + int(qi),
            coeff(sth_coeff, m_grid),
            coeff(sfl_coeff, mvec_idx),
        )
        for name, pi, qi, sth_coeff, sfl_coeff in noise_sources
    ]
    e = np.zeros(nb * n_state, dtype=complex)
    base0 = K * n_state
    for node, weight in out_w.items():
        e[base0 + idx[node]] = weight

    hb_preconditioner_kind = (
        "block_jacobi" if solver_used == "iterative" and _spla is not None
        else "none"
    )
    adj_cache = pss_result.setdefault("_pnoise_adjoint_cache", {}) if cache_linearization else {}
    adj_prefix = (
        "pnoise_adj_v1",
        lin_key,
        int(K),
        float(fundamental),
        float(hb_sparse_drop_tol),
        str(solver_used),
        float(iterative_tol) if solver_used == "iterative" else None,
        None if iterative_maxiter is None or solver_used != "iterative" else int(iterative_maxiter),
        hb_preconditioner_kind if solver_used == "iterative" else None,
        tuple((node, float(weight)) for node, weight in sorted(out_w.items())),
    )
    adj_cache_hits = 0
    hb_solve_count = 0
    hb_sparse_direct_count = 0
    hb_iterative_count = 0
    hb_iterative_fallbacks = 0
    hb_dense_fallbacks = 0
    hb_block_preconditioner_count = 0
    hb_iterative_infos = []
    hb_iterative_iterations = []
    # Ideal voltage sources: border the HB adjoint with branch-current unknowns appended
    # after the nb*n_state node block (external node positions in the fold stay
    # unchanged). Force the dense path; vsource circuits are small testbenches,
    # not the large chopper HB.
    nbr = topo.n_branches
    if nbr:
        if Y_base is None or C_block is None:
            Y_base, C_block, _ = _hb_blocks(Gf, Cf, K, N, n_state, fundamental,
                                            charge_caps=charge_caps)
        all_branch_sources = list(topo.vsources) + list(topo.vcvs) + list(topo.ccvs)
        Binc = _branch_incidence(all_branch_sources, idx, n)
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
            Ya[r0:r0 + n_state, c0:c0 + nbr] = Binc
            Ya[c0:c0 + nbr, r0:r0 + n_state] = Binc.T
        Y_base, C_block = Ya, Ca
        Y_sparse = C_sparse = None
        solver_used = "dense"
        e = np.concatenate([e, np.zeros(nb * nbr, dtype=complex)])

    adjs = np.empty((len(freqs), e.shape[0]), dtype=complex)
    t_solve0 = time.perf_counter()
    # Truncation-free time-domain Floquet adjoint (non-bordered case): replaces the
    # K-truncated HB adjoint solve with an exact-in-sideband sparse BVP solve. The
    # existing cyclostationary fold below is reused unchanged.
    td_adjs = (_time_domain_pnoise_adjoint(Gf, Cf, e, freqs, K, n_state, fundamental)
               if (time_domain and nbr == 0) else None)
    pnoise_time_domain_used = td_adjs is not None
    if pnoise_time_domain_used:
        adjs = td_adjs
    for fi, freq in enumerate(freqs):
        if pnoise_time_domain_used:
            break
        freq = float(freq)
        adj_key = adj_prefix + (freq,)
        if cache_linearization and adj_key in adj_cache:
            adj = adj_cache[adj_key]
            adj_cache_hits += 1
        else:
            preconditioner = None
            if hb_preconditioner_kind == "block_jacobi":
                preconditioner = _block_jacobi_preconditioner(
                    Gf, Cf, K, n_state, fundamental, freq)
                if preconditioner is not None:
                    hb_block_preconditioner_count += 1
            adj, solve_info = _solve_hb_adjoint(
                Y_base, C_block, Y_sparse, C_sparse, freq, e, solver_used,
                iterative_tol, iterative_maxiter,
                preconditioner=preconditioner)
            if solve_info["solver"] == "sparse":
                hb_sparse_direct_count += 1
            elif solve_info["solver"] == "iterative":
                hb_iterative_count += 1
                hb_iterative_infos.append(int(solve_info["iterative_info"]))
                hb_iterative_iterations.append(
                    int(solve_info["iterative_iterations"]))
            if solve_info["iterative_fallback"]:
                hb_iterative_fallbacks += 1
                hb_sparse_direct_count += 1
            if solve_info["dense_fallback"]:
                hb_dense_fallbacks += 1
                add_warning(
                    "hb_dense_fallback",
                    f"HB adjoint solve at {freq:g} Hz fell back to a dense or "
                    "least-squares solve. This PNoise degradation path is not "
                    "fully supported yet.",
                )
            hb_solve_count += 1
            if cache_linearization:
                adj_cache[adj_key] = adj
        adjs[fi] = adj
    hb_solve_time_s = time.perf_counter() - t_solve0

    fold_work = len(freqs) * max(1, len(source_grids)) * nb * nb
    # The numba fold accepts explicit source indices into the adjoint vector;
    # bordered (vsource) adjoints are wider, so fold those in Python.
    use_numba_fold = pnoise_fold_psd_numba is not None and fold_work >= 1000 and nbr == 0
    source_names = [name for name, *_ in source_grids]
    t_fold0 = time.perf_counter()
    if use_numba_fold:
        ns = len(source_grids)
        p_indices = np.full((ns, nb), -1, dtype=np.int64)
        q_indices = np.full((ns, nb), -1, dtype=np.int64)
        sth_stack = np.empty((ns, nb, nb), dtype=np.complex128)
        mfl_stack = np.empty((ns, 4 * K + 1), dtype=np.complex128)
        for si, (_name, p_idx, q_idx, sth_grid, mfl_vec) in enumerate(source_grids):
            if p_idx is not None:
                p_indices[si] = np.asarray(p_idx, dtype=np.int64)
            if q_idx is not None:
                q_indices[si] = np.asarray(q_idx, dtype=np.int64)
            sth_stack[si] = np.asarray(sth_grid, dtype=np.complex128)
            mfl_stack[si] = np.asarray(mfl_vec, dtype=np.complex128)
        out_psd, dev_stack = pnoise_fold_psd_numba(
            np.asarray(adjs, dtype=np.complex128),
            np.asarray(freqs, dtype=np.float64),
            int(K),
            float(fundamental),
            p_indices,
            q_indices,
            sth_stack,
            mfl_stack,
        )
        dev_psd = {
            name: np.asarray(dev_stack[si], dtype=float)
            for si, name in enumerate(source_names)
        }
    else:
        out_psd = np.zeros(len(freqs))
        dev_psd = {name: np.zeros(len(freqs)) for name in source_names}
        two_k = 2 * K
        for fi, freq in enumerate(freqs):
            freq = float(freq)
            adj = adjs[fi]
            nu = np.abs(freq + ks * fundamental)        # nu_a per sideband
            nu[nu < 1e-9] = 1e-9
            inv_nu = 1.0 / nu

            for name, p_idx, q_idx, sth_grid, mfl_vec in source_grids:
                if p_idx is None:
                    Z = np.zeros(nb, dtype=complex)
                else:
                    Z = adj[p_idx].copy()
                if q_idx is not None:
                    Z -= adj[q_idx]
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    # thermal (white): Toeplitz quadratic form Z^H S_th Z.
                    contrib = float(np.real(Z @ (sth_grid @ np.conj(Z))))
                    # flicker (1/f, cyclostationary): sum_a |sum_r Z_r M_{r-a}|^2 / nu_a,
                    # with M_{r-a}=mfl_vec[(r-a)+2K] -> for sideband a, slice [2K-a:2K-a+nb].
                    U = np.array([Z @ mfl_vec[two_k - a: two_k - a + nb]
                                  for a in range(nb)])
                    contrib += float(np.sum((U.real ** 2 + U.imag ** 2) * inv_nu))
                contrib = max(contrib, 0.0)
                dev_psd[name][fi] += contrib
                out_psd[fi] += contrib
    fold_time_s = time.perf_counter() - t_fold0

    if gains is None:
        if pac_result is None:
            if input_drive is None:
                raise ValueError("gains, pac_result, or input_drive is required")
            pac_result = pac_solve(
                sizes, bias, freqs, pss_result=pss_result,
                input_drive=input_drive, nf=nf, corner=corner,
            )
        gains = pac_result["gains"]
    gains = np.asarray(gains, float)
    irn_psd = out_psd / np.maximum(gains ** 2, 1e-300)
    method = ("pss_time_domain_floquet_adjoint" if pnoise_time_domain_used
              else "pss_harmonic_balance_conversion_matrix")
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
        "pnoise_hb_size": int(nb * n_state),
        "pnoise_state_size": int(n_state),
        "pnoise_internal_gate1_states": int(n_gate1),
        "pnoise_hb_solve_count": int(hb_solve_count),
        "pnoise_noise_source_count": int(len(noise_sources)),
        "pnoise_numba_hb_used": bool(hb_numba_used),
        "pnoise_numba_fold_used": bool(use_numba_fold),
        "pnoise_time_domain_used": bool(pnoise_time_domain_used),
        "pnoise_conversion": "time_domain" if pnoise_time_domain_used else "harmonic_balance",
        "pnoise_hb_solver_requested": hb_solver_requested,
        "pnoise_hb_solver": str(solver_used),
        "pnoise_hb_sparse_available": bool(_sp is not None and _spla is not None),
        "pnoise_hb_sparse_nnz": int(sparse_nnz),
        "pnoise_hb_sparse_density": float(sparse_density),
        "pnoise_hb_sparse_density_estimate": float(sparse_density_estimate),
        "pnoise_hb_sparse_direct_count": int(hb_sparse_direct_count),
        "pnoise_hb_iterative_count": int(hb_iterative_count),
        "pnoise_hb_iterative_fallbacks": int(hb_iterative_fallbacks),
        "pnoise_hb_dense_fallbacks": int(hb_dense_fallbacks),
        "pnoise_hb_preconditioner": hb_preconditioner_kind,
        "pnoise_hb_block_preconditioner_count": int(hb_block_preconditioner_count),
        "pnoise_hb_iterative_infos": tuple(hb_iterative_infos),
        "pnoise_hb_iterative_iterations": tuple(hb_iterative_iterations),
        "pnoise_noise_failure_count": int(noise_failure_count),
        "pnoise_noise_failure_devices": tuple(noise_failure_devices),
        "pnoise_degraded": bool(pnoise_warnings),
        "pnoise_warnings": tuple(pnoise_warnings),
        "pnoise_linearization_time_s": float(linearization_time_s),
        "pnoise_hb_assembly_time_s": float(hb_assembly_time_s),
        "pnoise_hb_solve_time_s": float(hb_solve_time_s),
        "pnoise_fold_time_s": float(fold_time_s),
        "pnoise_profile_enabled": bool(profile),
        "method": method,
    }
