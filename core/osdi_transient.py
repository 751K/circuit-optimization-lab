"""Minimal backward-Euler transient for an OSDI (silicon) device — Phase B foundation.

The numba transient kernels are welded to the OTFT compact model (they call
``_eval_currents_impl`` / consume ``NumbaParams``) and cannot call a compiled
``.osdi`` inside the nopython loop.  So a silicon transient lives here as a separate,
pure-Python path: a fixed-step backward-Euler integrator of a common-source stage
(device with source at VDD, drain = ``vout``, gate = ``vin(t)``, plus a load resistor
and load capacitor), evaluating the device current + drain charge through the OSDI
host each step.

This is a *correctness demonstration* of silicon transient, not a replacement for the
OTFT path's adaptive-gear2 machinery; full-fidelity silicon transient/chopper
(adaptive integration, PSS/PAC/PNoise) remains a larger effort.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def cs_transient(dev, vdd, r_load, c_load, vin, tgrid, *, vmin=1e-4):
    """Backward-Euler ``vout(t)`` of a common-source stage.

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
