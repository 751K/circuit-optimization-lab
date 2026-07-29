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


#: Orbit samples carried by one compiled-backend call. One sample is only as
#: wide as the circuit's device count, which leaves the backend's Rayon pool
#: with almost nothing to spread; several samples per call give it real width.
#: Measured on the 13-device chopper (evaluate + 2-frequency noise, 768-sample
#: orbit): 1 sample 121.6 ms, 4 samples 45.3 ms, 16 samples 18.8 ms, 48 samples
#: 12.9 ms. Handles cost 41.2 us each to build and are held for the whole orbit,
#: so 16 pays 8.6 ms to save 103 ms while 48 pays 25.7 ms to save only 5.9 ms
#: more.
ORBIT_BATCH_SAMPLES = 16


class NativeOrbitBatch:
    """Native BSIM4 handles for ``samples`` x ``devices``, re-biased per block.

    ``evaluate`` and ``noise`` mirror the scalar adapter's contract, including
    its bulk-terminal reduction and its Hermitian/positive-semidefinite noise
    validation -- checked across the whole block with one vectorized pass
    instead of one dataclass construction per device.

    Each orbit sample owns its own row of handles, so a device's handle sees a
    strided subsequence of the orbit rather than every sample. BSIM4 evaluation
    carries state between calls on one handle, so this changes the answer at the
    level that state is worth: measured at 1e-13 on tsmc28hpcp_chopper, zero on
    sky130_chopper.
    """

    def __init__(self, handles, bulk, devices_per_sample):
        self._handles = list(handles)
        self._stride = int(devices_per_sample)
        self._samples = len(self._handles) // max(self._stride, 1)
        self._bulk = np.asarray(bulk, dtype=float)
        self._terminals = np.empty((len(self._handles), 4), dtype=float)
        self._terminals[:, 3] = np.tile(self._bulk, self._samples)
        self._active = self._stride

    def __len__(self):
        return len(self._handles)

    @property
    def samples_per_call(self):
        """How many orbit samples one :meth:`evaluate` can carry."""
        return self._samples

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
        """Bias one block of samples; return (currents, conductance, charges, caps).

        ``vs``/``vd``/``vg`` are ``(block, devices)`` or a single sample's
        ``(devices,)``; ``block`` may be shorter than :attr:`samples_per_call`
        for the orbit's last, partial block. The returned arrays are
        ``(block * devices, ...)`` and carry the same bulk-terminal reduction
        the scalar adapter applies, so they are interchangeable with
        ``get_terminal_currents`` / ``get_terminal_linearization``.
        """
        from .compact_models.bsim4 import Bsim4NativeError, NativeBsim4Backend

        flat_d = np.ravel(vd)
        count = flat_d.size
        if count > len(self._handles) or count % self._stride:
            raise ValueError(
                f"orbit batch takes whole samples of {self._stride} devices, "
                f"up to {self._samples}; got {count} values")
        self._active = count
        terminals = self._terminals[:count]
        terminals[:, 0] = flat_d
        terminals[:, 1] = np.ravel(vg)
        terminals[:, 2] = np.ravel(vs)
        currents, conductance, charges, capacitance = (
            NativeBsim4Backend.evaluate_batch(self._handles[:count], terminals))
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
        """Terminal-noise (total, white, flicker) for the last biased block.

        Shapes are ``(block * devices, frequency, 4, 4)``, matching the block
        the preceding :meth:`evaluate` biased.
        """
        from .compact_models.bsim4 import Bsim4ValidationError, NativeBsim4Backend

        total, flicker = NativeBsim4Backend.noise_batch(
            self._handles[:self._active], np.asarray(frequencies, dtype=float))
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


def open_orbit_batch(devices, n_samples=ORBIT_BATCH_SAMPLES):
    """Open a :class:`NativeOrbitBatch` over ``devices``, or ``None``.

    ``None`` means at least one device has no independent native handle to
    offer, so the caller must keep evaluating that set one device at a time.

    ``n_samples`` is how many orbit samples the caller has to walk, NOT the
    batch width: the width is ``min(n_samples, ORBIT_BATCH_SAMPLES)``, because
    each sample in the block costs one handle per device and a whole orbit's
    worth would cost far more to build than the batching saves.
    """
    devices = list(devices)
    if not devices:
        return None
    if any(dev is None or not callable(
            getattr(dev, "create_native_solver_handle", None))
            for dev in devices):
        return None
    samples_per_call = max(1, min(int(n_samples), ORBIT_BATCH_SAMPLES))
    handles = []
    try:
        for _ in range(samples_per_call):
            for dev in devices:
                handles.append(dev.create_native_solver_handle())
        return NativeOrbitBatch(
            handles, [float(getattr(dev, "vb", 0.0)) for dev in devices],
            len(devices))
    except Exception as exc:
        for handle in handles:
            handle.close()
        diagnostics.note(
            "model.orbit_batch_unavailable", exc,
            detail="native BSIM4 orbit batch unavailable; using scalar evaluation",
        )
        return None
