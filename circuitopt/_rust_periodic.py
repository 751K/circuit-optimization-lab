"""Narrow NumPy boundary for periodic kernels owned by the Rust core."""
from __future__ import annotations

import numpy as np


def _core():
    import circuitopt_core

    return circuitopt_core


def hb_blocks(Gf, Cf, K, fundamental, charge_caps):
    return _core().periodic_hb_blocks(
        np.ascontiguousarray(Gf, dtype=np.complex128),
        np.ascontiguousarray(Cf, dtype=np.complex128),
        int(K),
        float(fundamental),
        bool(charge_caps),
    )


def fold_psd(adjs, freqs, K, fundamental, p_indices, q_indices,
             thermal_grids, flicker_grids):
    return _core().periodic_fold_psd(
        np.ascontiguousarray(adjs, dtype=np.complex128),
        np.ascontiguousarray(freqs, dtype=np.float64),
        int(K),
        float(fundamental),
        np.ascontiguousarray(p_indices, dtype=np.int64),
        np.ascontiguousarray(q_indices, dtype=np.int64),
        np.ascontiguousarray(thermal_grids, dtype=np.complex128),
        np.ascontiguousarray(flicker_grids, dtype=np.complex128),
    )
