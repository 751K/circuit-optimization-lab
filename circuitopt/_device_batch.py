"""Evaluate a whole device set at one orbit sample per compiled-backend call.

The periodic linearizations (PAC orbit G/C tensors, PNoise terminal-noise
matrices) walk a fixed orbit sample by sample and need every transistor
evaluated at each one. Driving that through the scalar adapter costs ~73 us per
device-sample of which only ~1-3 us is the compact model: the rest is the
per-call handle lease, the six small array allocations, and the dataclass
contract validation. The stable C batch ABI takes one handle per bias, so a
single call can carry the whole device set at one sample.

The batch is opened only when every device can supply an independent native
handle; anything else (OTFT, a backend without the batch entry point) returns
``None`` and the caller keeps its scalar path.
"""
from __future__ import annotations

import numpy as np

from . import diagnostics


def _hermitian_violation(matrices):
    """Max deviation from Hermitian symmetry over a stack of square matrices."""
    return float(np.max(np.abs(
        matrices - np.conjugate(np.swapaxes(matrices, -1, -2)))))


def _close_at_reference(values, floor, absolute_tolerance, label):
    """Close BSIM's terminal residual at the bulk terminal, batch-wide.

    BSIM's cutoff-state load equations leave an abstol/gmin-scale remainder in
    the four-terminal sum; the scalar adapter absorbs it into the reference
    terminal before enforcing the public KCL contract, and the batch ABI hands
    back the same unreduced kernel output. This is that reduction, applied to a
    leading device axis: ``values`` is ``(device, 4)`` or ``(device, 4, 4)``,
    summed over the terminal axis. A genuinely broken reduction still raises.
    """
    from .compact_models.bsim4 import Bsim4NativeError

    error = np.sum(values, axis=1)
    flat = np.abs(values).reshape(len(values), -1)
    scale = np.maximum(flat.max(axis=1) if flat.size else np.zeros(len(values)),
                       floor)
    limit = np.maximum(1e-8 * scale, absolute_tolerance)
    worst = np.abs(error).reshape(len(values), -1).max(axis=1) if len(values) else ()
    if len(values) and bool(np.any(worst > limit)):
        index = int(np.argmax(worst - limit))
        raise Bsim4NativeError(
            f"BSIM4 terminal-{label} reduction failed at batch index {index}: "
            f"residual={float(worst[index]):.6g}")
    values[:, 3] -= error
    return values


class NativeOrbitBatch:
    """One native BSIM4 handle per device, re-biased sample by sample.

    ``evaluate`` and ``noise`` mirror the scalar adapter's contract, including
    its Hermitian/positive-semidefinite noise validation -- checked across the
    whole batch with one vectorized pass instead of one dataclass construction
    per device.
    """

    def __init__(self, handles, bulk):
        self._handles = list(handles)
        self._bulk = np.asarray(bulk, dtype=float)
        self._terminals = np.empty((len(self._handles), 4), dtype=float)
        self._terminals[:, 3] = self._bulk

    def __len__(self):
        return len(self._handles)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        for handle in self._handles:
            handle.close()
        self._handles = []

    def evaluate(self, vs, vd, vg):
        """Bias every handle and return (currents, conductance, charges, caps).

        ``vs``/``vd``/``vg`` are per-device terminal voltages for one sample.
        The four returned blocks carry the same bulk-terminal reduction the
        scalar adapter applies, so they are interchangeable with
        ``get_terminal_currents`` / ``get_terminal_linearization``.
        """
        from .compact_models.bsim4 import Bsim4NativeError, NativeBsim4Backend

        self._terminals[:, 0] = vd
        self._terminals[:, 1] = vg
        self._terminals[:, 2] = vs
        currents, conductance, charges, capacitance = (
            NativeBsim4Backend.evaluate_batch(self._handles, self._terminals))
        for block in (currents, conductance, charges, capacitance):
            if not np.all(np.isfinite(block)):
                raise Bsim4NativeError(
                    "BSIM4 batch evaluation contains non-finite values")
        _close_at_reference(currents, 1e-18, 1e-9, "current")
        _close_at_reference(conductance, 1e-18, 1e-9, "conductance")
        _close_at_reference(charges, 1e-24, 1e-18, "charge")
        _close_at_reference(capacitance, 1e-24, 1e-18, "capacitance")
        return currents, conductance, charges, capacitance

    def noise(self, frequencies):
        """Terminal-noise (total, flicker) for the last biased operating point.

        Shapes are ``(device, frequency, 4, 4)``. The handles must already have
        been biased through :meth:`evaluate`.
        """
        from .compact_models.bsim4 import Bsim4ValidationError, NativeBsim4Backend

        total, flicker = NativeBsim4Backend.noise_batch(
            self._handles, np.asarray(frequencies, dtype=float))
        white = total - flicker
        if not (np.all(np.isfinite(total)) and np.all(np.isfinite(flicker))):
            raise Bsim4ValidationError("noise matrix contains non-finite values")
        for label, matrices in (("noise", total), ("white", white),
                                ("flicker", flicker)):
            if _hermitian_violation(matrices) > 1e-8 * max(
                    float(np.max(np.abs(matrices))), 1e-30):
                raise Bsim4ValidationError(f"{label} matrix must be Hermitian")
        eigenvalues = np.linalg.eigvalsh(
            (total + np.conjugate(np.swapaxes(total, -1, -2))) * 0.5)
        if float(np.min(eigenvalues)) < -1e-8 * max(
                float(np.max(np.abs(eigenvalues))), 1e-30):
            raise Bsim4ValidationError("noise matrix must be positive semidefinite")
        return total, white, flicker


def open_orbit_batch(devices):
    """Open a :class:`NativeOrbitBatch` over ``devices``, or ``None``.

    ``None`` means at least one device has no independent native handle to
    offer, so the caller must keep evaluating that set one device at a time.
    """
    devices = list(devices)
    if not devices:
        return None
    if any(dev is None or not callable(
            getattr(dev, "create_native_solver_handle", None))
            for dev in devices):
        return None
    handles = []
    try:
        for dev in devices:
            handles.append(dev.create_native_solver_handle())
        return NativeOrbitBatch(
            handles, [float(getattr(dev, "vb", 0.0)) for dev in devices])
    except Exception as exc:
        for handle in handles:
            handle.close()
        diagnostics.note(
            "model.orbit_batch_unavailable", exc,
            detail="native BSIM4 orbit batch unavailable; using scalar evaluation",
        )
        return None
