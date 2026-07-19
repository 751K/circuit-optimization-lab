"""In-process Berkeley BSIM4.5 numerical backend.

The vendored compact-model equations are compiled into the ``circuitopt_core``
extension (the ``co-bsim4`` crate builds them at wheel-build time); this module
binds that already-compiled library through a small four-terminal C ABI. It
does not link libngspice, invoke an external circuit simulator, or compile
anything at runtime (the v1.x cc/ctypes runtime-compile path was removed in
v2.0.0). CircuitOpt owns parameter loading, internal-node reduction, and all
circuit-level analyses.
"""
from __future__ import annotations

import ctypes as C
import os
import threading
from collections import OrderedDict

import numpy as np

from .abi import (
    Bsim4Bias,
    Bsim4Evaluation,
    Bsim4InstanceCard,
    Bsim4ModelCard,
    Bsim4Noise,
)


class Bsim4NativeError(RuntimeError):
    """The native BSIM4 kernel could not be built, configured, or evaluated."""


_STATUS = {
    1: "internal panic or singular compact-model matrix",
    7: "unknown or unsupported BSIM4 parameter",
    8: "native compact-model allocation failed",
    10: "requested BSIM4 topology is outside the native core-MOS scope",
    13: "parameters cannot be changed after BSIM4 setup",
}
_ABI_VERSION = 1
_EVAL_VP_T = C.CFUNCTYPE(
    C.c_int,
    C.c_void_p,
    C.c_void_p,
    C.c_void_p,
    C.c_void_p,
    C.c_void_p,
    C.c_void_p,
)
_build_lock = threading.RLock()
_rust_library = None


def _bind_abi(library) -> None:
    """Bind the four-terminal BSIM4 C ABI onto a loaded ``ctypes`` library.

    The compiled ``circuitopt_core`` cdylib exports the ABI. This does not
    affect numerical results — it only declares the argument/return
    marshalling and installs the runtime function pointer.
    """
    double_pointer = C.POINTER(C.c_double)
    library.co_bsim4_create.argtypes = (C.c_int, C.c_double)
    library.co_bsim4_create.restype = C.c_void_p
    library.co_bsim4_destroy.argtypes = (C.c_void_p,)
    library.co_bsim4_destroy.restype = None
    library.co_bsim4_set_model.argtypes = (
        C.c_void_p,
        C.c_char_p,
        C.c_double,
    )
    library.co_bsim4_set_model.restype = C.c_int
    library.co_bsim4_set_instance.argtypes = (
        C.c_void_p,
        C.c_char_p,
        C.c_double,
    )
    library.co_bsim4_set_instance.restype = C.c_int
    library.co_bsim4_setup.argtypes = (C.c_void_p,)
    library.co_bsim4_setup.restype = C.c_int
    library.co_bsim4_dc.argtypes = (
        C.c_void_p,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
    )
    library.co_bsim4_dc.restype = C.c_int
    library.co_bsim4_eval.argtypes = library.co_bsim4_dc.argtypes
    library.co_bsim4_eval.restype = C.c_int
    library.co_bsim4_eval_vp.argtypes = (C.c_void_p,) * 6
    library.co_bsim4_eval_vp.restype = C.c_int
    library.co_bsim4_eval_batch.argtypes = (
        C.POINTER(C.c_void_p),
        C.c_size_t,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        C.POINTER(C.c_int),
    )
    library.co_bsim4_eval_batch.restype = C.c_int
    library.co_bsim4_noise.argtypes = (
        C.c_void_p,
        C.c_double,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
    )
    library.co_bsim4_noise.restype = C.c_int
    if hasattr(library, "co_bsim4_noise_batch"):
        library.co_bsim4_noise_batch.argtypes = (
            C.POINTER(C.c_void_p),
            C.c_size_t,
            double_pointer,
            C.c_size_t,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            C.POINTER(C.c_int),
        )
        library.co_bsim4_noise_batch.restype = C.c_int
    library._co_bsim4_eval_vp = _EVAL_VP_T(
        ("co_bsim4_eval_vp", library))


def _import_circuitopt_core():
    """Import the compiled Rust core. Isolated for testability."""
    import circuitopt_core

    return circuitopt_core


def _rust_extension_path(module) -> str | None:
    """Resolve the on-disk shared object backing ``circuitopt_core``.

    ``maturin develop`` installs an editable *package* whose ``__file__`` is the
    ``__init__.py``; the compiled object is a submodule (``.abi3.so``). A plain
    wheel install exposes the bare extension directly. Handle both, plus a
    directory scan as a last resort.
    """
    suffixes = (".so", ".dylib", ".pyd")
    direct = getattr(module, "__file__", None)
    if direct and direct.endswith(suffixes):
        return direct
    submodule = getattr(module, "circuitopt_core", None)
    submodule_file = getattr(submodule, "__file__", None)
    if submodule_file and submodule_file.endswith(suffixes):
        return submodule_file
    for directory in getattr(module, "__path__", None) or ():
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        for name in names:
            if name.endswith(suffixes):
                return os.path.join(directory, name)
    return None


def _bind_rust_library():
    global _rust_library
    if _rust_library is not None:
        return _rust_library
    with _build_lock:
        if _rust_library is not None:
            return _rust_library
        try:
            module = _import_circuitopt_core()
        except ImportError as exc:
            raise Bsim4NativeError(
                "CIRCUIT_BSIM4_BACKEND=rust requires the compiled circuitopt_core "
                "extension, which is not importable; build it with "
                "`maturin develop --release -m rust/crates/co-py/Cargo.toml`"
            ) from exc
        path = _rust_extension_path(module)
        if not path:
            raise Bsim4NativeError(
                "could not locate the compiled circuitopt_core shared object "
                f"(module {getattr(module, '__file__', module)!r})")
        library = C.CDLL(path)
        library.co_bsim4_abi_version.argtypes = ()
        library.co_bsim4_abi_version.restype = C.c_uint
        abi_version = int(library.co_bsim4_abi_version())
        if abi_version != _ABI_VERSION:
            raise Bsim4NativeError(
                f"rust BSIM4 ABI version {abi_version} != expected {_ABI_VERSION}")
        _bind_abi(library)
        _rust_library = library
    return _rust_library


def _backend_choice() -> str:
    """Validate the backend selector at call time (never baked at import).

    ``rust`` — the compiled ``circuitopt_core`` cdylib — is the only backend in
    v2.0.0. The retired ``cc`` value (v1.x runtime compilation of the vendored
    C with the user's compiler) errors loudly instead of being silently
    remapped, mirroring the engine-switch removals in ``_engine.py``.
    """
    value = os.environ.get("CIRCUIT_BSIM4_BACKEND", "rust").strip().lower()
    if value == "cc":
        raise Bsim4NativeError(
            "CIRCUIT_BSIM4_BACKEND=cc was removed in v2.0.0: the runtime "
            "cc/ctypes build of the vendored BSIM4 sources no longer exists; "
            "rust (the compiled circuitopt_core backend) is the only value")
    if value != "rust":
        raise Bsim4NativeError(
            f"CIRCUIT_BSIM4_BACKEND must be 'rust', got {value!r}")
    return value


def _select_library(backend: str):
    return _bind_rust_library()


def _raise_status(status: int, action: str, parameter: str | None = None) -> None:
    if status == 0:
        return
    detail = _STATUS.get(status, f"native status {status}")
    subject = f" parameter {parameter!r}" if parameter is not None else ""
    raise Bsim4NativeError(f"BSIM4 {action}{subject} failed: {detail}")


class _NativeDevice:
    def __init__(
        self,
        model: Bsim4ModelCard,
        instance: Bsim4InstanceCard,
        temperature_k: float,
        *,
        backend: str = "rust",
    ):
        self._backend = backend
        self._library = _select_library(backend)
        self._pointer = self._library.co_bsim4_create(
            model.polarity, float(temperature_k))
        if not self._pointer:
            raise Bsim4NativeError("BSIM4 native device allocation failed")
        self._lock = threading.RLock()
        try:
            for name, value in model.parameters.items():
                status = self._library.co_bsim4_set_model(
                    self._pointer, name.encode("ascii"), value)
                _raise_status(status, "model setup", name)
            for name, value in instance.parameters.items():
                status = self._library.co_bsim4_set_instance(
                    self._pointer, name.encode("ascii"), value)
                _raise_status(status, "instance setup", name)
            _raise_status(
                self._library.co_bsim4_setup(self._pointer),
                "temperature/setup",
            )
        except Exception:
            self.close()
            raise

    @property
    def pointer(self) -> int:
        """Process-local opaque handle used by the compiled-solver bridge."""
        if not self._pointer:
            raise Bsim4NativeError("BSIM4 native device is closed")
        return int(self._pointer)

    @property
    def kernel_evaluator(self):
        """Runtime ctypes pointer for the all-``void *`` evaluation ABI."""
        return self._library._co_bsim4_eval_vp

    def close(self) -> None:
        with self._lock:
            pointer = getattr(self, "_pointer", None)
            if pointer:
                self._library.co_bsim4_destroy(pointer)
                self._pointer = None

    def __del__(self):  # pragma: no cover - deterministic cache eviction handles normal use
        try:
            self.close()
        except Exception:
            pass

    def evaluate(
        self,
        bias: Bsim4Bias,
        frequency_hz: float | None = None,
    ) -> Bsim4Evaluation:
        terminals = np.ascontiguousarray(bias.terminals, dtype=np.float64)
        currents = np.empty(4, dtype=np.float64)
        conductance = np.empty((4, 4), dtype=np.float64)
        charges = np.empty(4, dtype=np.float64)
        capacitance = np.empty((4, 4), dtype=np.float64)
        op = np.empty(8, dtype=np.float64)
        pointer = C.POINTER(C.c_double)
        with self._lock:
            status = self._library.co_bsim4_eval(
                self._pointer,
                terminals.ctypes.data_as(pointer),
                currents.ctypes.data_as(pointer),
                conductance.ctypes.data_as(pointer),
                charges.ctypes.data_as(pointer),
                capacitance.ctypes.data_as(pointer),
                op.ctypes.data_as(pointer),
            )
            _raise_status(status, "evaluation")
            noise = None
            if frequency_hz is not None:
                if not np.isfinite(frequency_hz) or frequency_hz <= 0:
                    raise Bsim4NativeError(
                        "BSIM4 noise frequency must be positive and finite")
                total_real = np.empty((4, 4), dtype=np.float64)
                total_imag = np.empty((4, 4), dtype=np.float64)
                flicker_real = np.empty((4, 4), dtype=np.float64)
                flicker_imag = np.empty((4, 4), dtype=np.float64)
                status = self._library.co_bsim4_noise(
                    self._pointer,
                    frequency_hz,
                    total_real.ctypes.data_as(pointer),
                    total_imag.ctypes.data_as(pointer),
                    flicker_real.ctypes.data_as(pointer),
                    flicker_imag.ctypes.data_as(pointer),
                )
                _raise_status(status, "noise evaluation")
                total = total_real + 1j * total_imag
                flicker = flicker_real + 1j * flicker_imag
                noise = Bsim4Noise(
                    total,
                    {
                        "white": total - flicker,
                        "flicker": flicker,
                    },
                )
        # BSIM's cutoff-state load equations retain an abstol/gmin-scale
        # terminal residual. Circuit simulators close that numerical remainder
        # at the reference terminal; do the same before enforcing the public
        # four-terminal KCL contract. A genuinely broken reduction still fails.
        current_error = float(np.sum(currents))
        current_scale = max(float(np.max(np.abs(currents))), 1e-18)
        if abs(current_error) > max(1e-8 * current_scale, 1e-9):
            raise Bsim4NativeError(
                "BSIM4 terminal-current reduction failed KCL: "
                f"sum={current_error:.6g} A")
        currents[3] -= current_error
        conductance_error = np.sum(conductance, axis=0)
        conductance_scale = max(
            float(np.max(np.abs(conductance))), 1e-18)
        if float(np.max(np.abs(conductance_error))) > max(
            1e-8 * conductance_scale, 1e-9
        ):
            raise Bsim4NativeError(
                "BSIM4 terminal-conductance reduction failed KCL")
        conductance[3, :] -= conductance_error
        charge_error = float(np.sum(charges))
        charge_scale = max(float(np.max(np.abs(charges))), 1e-24)
        if abs(charge_error) > max(1e-8 * charge_scale, 1e-18):
            raise Bsim4NativeError(
                "BSIM4 terminal-charge reduction failed conservation")
        charges[3] -= charge_error
        capacitance_error = np.sum(capacitance, axis=0)
        capacitance_scale = max(
            float(np.max(np.abs(capacitance))), 1e-24)
        if float(np.max(np.abs(capacitance_error))) > max(
            1e-8 * capacitance_scale, 1e-18
        ):
            raise Bsim4NativeError(
                "BSIM4 terminal-capacitance reduction failed conservation")
        capacitance[3, :] -= capacitance_error
        return Bsim4Evaluation(
            terminal_currents=currents,
            conductance=conductance,
            terminal_charges=charges,
            capacitance=capacitance,
            operating_point={
                "ids": op[0],
                "gm": op[1],
                "gds": op[2],
                "gmb": op[3],
                "vth": op[4],
                "vdsat": op[5],
                "ueff": op[6],
                "internal_nodes": op[7],
            },
            noise=noise,
        )


class _NativeDeviceLease:
    """Pins one cached native handle until the lease leaves scope."""

    def __init__(self, backend, device, cached):
        self._backend = backend
        self.device = device
        self._cached = cached

    def close(self):
        device = self.device
        if device is not None:
            self.device = None
            self._backend._release_device(device, self._cached)

    def __del__(self):  # pragma: no cover - CPython scope exit is the normal path
        try:
            self.close()
        except Exception:
            pass


class NativeBsim4Backend:
    """Berkeley BSIM4.5 evaluator hosted by CircuitOpt in the current process."""

    name = "berkeley-bsim4v5-native"
    version = "4.5.0"
    abi_version = _ABI_VERSION

    def __init__(self, *, cache_size: int | None = None):
        if cache_size is None:
            cache_size = int(os.environ.get("BSIM4_DEVICE_CACHE_SIZE", "128"))
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self._cache_size = cache_size
        self._devices: OrderedDict[tuple, _NativeDevice] = OrderedDict()
        self._active: dict[int, int] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(
        model: Bsim4ModelCard,
        instance: Bsim4InstanceCard,
        temperature_k: float,
        backend: str,
    ) -> tuple:
        return (
            backend,
            model.polarity,
            model.version,
            tuple(sorted(model.parameters.items())),
            tuple(sorted(instance.parameters.items())),
            float(temperature_k),
        )

    def _evict_idle_locked(self) -> None:
        """Trim the LRU without closing a handle leased by another worker."""
        while len(self._devices) > self._cache_size:
            for key, device in self._devices.items():
                if self._active.get(id(device), 0) == 0:
                    self._devices.pop(key)
                    device.close()
                    break
            else:
                # All excess entries are in flight. They are trimmed when their
                # leases return instead of invalidating a live native pointer.
                return

    def _lease_device(
        self,
        model: Bsim4ModelCard,
        instance: Bsim4InstanceCard,
        temperature_k: float,
        backend: str,
    ) -> tuple[_NativeDevice, bool]:
        if self._cache_size == 0:
            return _NativeDevice(model, instance, temperature_k, backend=backend), False
        key = self._key(model, instance, temperature_k, backend)
        with self._lock:
            device = self._devices.get(key)
            if device is not None:
                self._devices.move_to_end(key)
            else:
                device = _NativeDevice(model, instance, temperature_k, backend=backend)
                self._devices[key] = device
            identity = id(device)
            self._active[identity] = self._active.get(identity, 0) + 1
            self._evict_idle_locked()
            return device, True

    def _release_device(self, device: _NativeDevice, cached: bool) -> None:
        if not cached:
            device.close()
            return
        with self._lock:
            identity = id(device)
            remaining = self._active.get(identity, 0) - 1
            if remaining > 0:
                self._active[identity] = remaining
            else:
                self._active.pop(identity, None)
            self._evict_idle_locked()

    def evaluate(
        self,
        model: Bsim4ModelCard,
        instance: Bsim4InstanceCard,
        bias: Bsim4Bias,
        *,
        frequency_hz: float | None = None,
    ) -> Bsim4Evaluation:
        # Backend is chosen per call (CIRCUIT_BSIM4_BACKEND), never at import.
        backend = _backend_choice()
        device, cached = self._lease_device(
            model, instance, bias.temperature_k, backend)
        try:
            return device.evaluate(bias, frequency_hz)
        finally:
            self._release_device(device, cached)

    def create_device(
        self,
        model: Bsim4ModelCard,
        instance: Bsim4InstanceCard,
        temperature_k: float,
    ) -> _NativeDevice:
        """Create an independently owned handle for a compiled solver loop.

        Unlike the ordinary evaluator cache, the caller owns this handle and
        must close it. A dedicated handle avoids sharing mutable BSIM state
        between concurrent transient simulations.
        """
        return _NativeDevice(
            model, instance, float(temperature_k), backend=_backend_choice())

    def lease_device(
        self,
        model: Bsim4ModelCard,
        instance: Bsim4InstanceCard,
        temperature_k: float,
    ) -> _NativeDeviceLease:
        """Pin a cached, already-setup handle for one whole solver call."""
        device, cached = self._lease_device(
            model, instance, float(temperature_k), _backend_choice())
        return _NativeDeviceLease(self, device, cached)

    @staticmethod
    def evaluate_batch(
        devices,
        terminals,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate existing native handles through the stable C batch ABI."""
        device_list = list(devices)
        values = np.ascontiguousarray(terminals, dtype=np.float64)
        count = len(device_list)
        if values.shape != (count, 4):
            raise ValueError(
                f"terminals must have shape ({count}, 4), got {values.shape}")
        if count == 0:
            return (
                np.empty((0, 4)),
                np.empty((0, 4, 4)),
                np.empty((0, 4)),
                np.empty((0, 4, 4)),
            )
        library = device_list[0]._library
        if any(device._library is not library for device in device_list):
            raise ValueError("all BSIM4 batch handles must use the same native library")
        handles = (C.c_void_p * count)(
            *(C.c_void_p(device.pointer) for device in device_list))
        currents = np.empty((count, 4), dtype=np.float64)
        conductance = np.empty((count, 4, 4), dtype=np.float64)
        charges = np.empty((count, 4), dtype=np.float64)
        capacitance = np.empty((count, 4, 4), dtype=np.float64)
        statuses = np.empty(count, dtype=np.int32)
        pointer = C.POINTER(C.c_double)
        status = library.co_bsim4_eval_batch(
            handles,
            count,
            values.ctypes.data_as(pointer),
            currents.ctypes.data_as(pointer),
            conductance.ctypes.data_as(pointer),
            charges.ctypes.data_as(pointer),
            capacitance.ctypes.data_as(pointer),
            statuses.ctypes.data_as(C.POINTER(C.c_int)),
        )
        if status:
            failed = int(np.flatnonzero(statuses)[0])
            _raise_status(
                int(statuses[failed]), f"batch evaluation at index {failed}")
        return currents, conductance, charges, capacitance

    @staticmethod
    def noise_batch(devices, frequencies) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate terminal-noise matrices for independent biased handles.

        The handles must already have been evaluated at their operating points.
        Rust-backed handles execute one frequency sweep per handle outside the
        GIL and distribute independent handles through Rayon.
        """
        device_list = list(devices)
        values = np.ascontiguousarray(frequencies, dtype=np.float64)
        count = len(device_list)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("frequencies must be a finite positive 1D array")
        if count == 0:
            shape = (0, len(values), 4, 4)
            return np.empty(shape, dtype=complex), np.empty(shape, dtype=complex)
        pointers = [device.pointer for device in device_list]
        if len(set(pointers)) != count:
            raise ValueError("BSIM4 noise batch requires independent handles")
        library = device_list[0]._library
        if any(device._library is not library for device in device_list):
            raise ValueError("all BSIM4 noise handles must use the same native library")

        handles = (C.c_void_p * count)(*(C.c_void_p(value) for value in pointers))
        shape = (count, len(values), 4, 4)
        total_real = np.empty(shape, dtype=np.float64)
        total_imag = np.empty(shape, dtype=np.float64)
        flicker_real = np.empty(shape, dtype=np.float64)
        flicker_imag = np.empty(shape, dtype=np.float64)
        statuses = np.empty(count, dtype=np.int32)
        pointer = C.POINTER(C.c_double)

        batch = getattr(library, "co_bsim4_noise_batch", None)
        if batch is not None:
            status = batch(
                handles,
                count,
                values.ctypes.data_as(pointer),
                len(values),
                total_real.ctypes.data_as(pointer),
                total_imag.ctypes.data_as(pointer),
                flicker_real.ctypes.data_as(pointer),
                flicker_imag.ctypes.data_as(pointer),
                statuses.ctypes.data_as(C.POINTER(C.c_int)),
            )
            if status:
                failed = int(np.flatnonzero(statuses)[0])
                _raise_status(int(statuses[failed]), f"noise batch at device {failed}")
        else:
            for device_index, device in enumerate(device_list):
                with device._lock:
                    for frequency_index, frequency in enumerate(values):
                        status = library.co_bsim4_noise(
                            device._pointer,
                            float(frequency),
                            total_real[device_index, frequency_index].ctypes.data_as(pointer),
                            total_imag[device_index, frequency_index].ctypes.data_as(pointer),
                            flicker_real[device_index, frequency_index].ctypes.data_as(pointer),
                            flicker_imag[device_index, frequency_index].ctypes.data_as(pointer),
                        )
                        _raise_status(status, "noise evaluation")
        return total_real + 1j * total_imag, flicker_real + 1j * flicker_imag

    def close(self) -> None:
        with self._lock:
            if self._active:
                raise Bsim4NativeError(
                    "cannot close BSIM4 backend while evaluations are active")
            for device in self._devices.values():
                device.close()
            self._devices.clear()
