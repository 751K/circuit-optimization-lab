"""In-process Berkeley BSIM4.5 numerical backend.

The backend builds the vendored compact-model equations into a private shared
library and calls them through a small four-terminal C ABI. It does not link
libngspice or invoke an external circuit simulator. CircuitOpt owns parameter
loading, internal-node reduction, and all circuit-level analyses.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import platform
import shutil
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ...toolchain import native_model_cache_dir
from .abi import (
    Bsim4Bias,
    Bsim4Evaluation,
    Bsim4InstanceCard,
    Bsim4ModelCard,
    Bsim4Noise,
)


class Bsim4NativeError(RuntimeError):
    """The native BSIM4 kernel could not be built, configured, or evaluated."""


_SOURCE_ROOT = Path(__file__).with_name("native_src")
_VENDOR = _SOURCE_ROOT / "vendor"
_MODEL_DIR = _VENDOR / "bsim4v5"
_INCLUDE_DIR = _VENDOR / "include"
_SUPPORT_DIR = _VENDOR / "support"
_SOURCES = (
    _MODEL_DIR / "b4v5.c",
    _MODEL_DIR / "b4v5par.c",
    _MODEL_DIR / "b4v5mpar.c",
    _MODEL_DIR / "b4v5set.c",
    _MODEL_DIR / "b4v5temp.c",
    _MODEL_DIR / "b4v5ld.c",
    _MODEL_DIR / "b4v5acld.c",
    _MODEL_DIR / "b4v5noi.c",
    _MODEL_DIR / "b4v5geo.c",
    _SUPPORT_DIR / "devsup.c",
    _SOURCE_ROOT / "host.c",
)
_HASH_INPUTS = _SOURCES + tuple(sorted(_MODEL_DIR.glob("*.h"))) + tuple(
    sorted((_INCLUDE_DIR / "ngspice").glob("*.h"))
)
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
_library = None
_rust_library = None
_DEFAULT_BACKEND = "cc"


def _compiler() -> str:
    configured = os.environ.get("BSIM4_CC") or os.environ.get("CC")
    if configured:
        candidate = os.path.abspath(os.path.expanduser(configured))
        if os.path.sep not in configured:
            candidate = shutil.which(configured) or ""
        if candidate and os.access(candidate, os.X_OK):
            return candidate
        raise Bsim4NativeError(f"configured BSIM4 C compiler is not executable: {configured}")
    for name in ("clang", "cc", "gcc"):
        candidate = shutil.which(name)
        if candidate:
            return candidate
    raise Bsim4NativeError(
        "a C99 compiler is required for the native BSIM4.5 backend; "
        "set BSIM4_CC or CC"
    )


def _library_suffix() -> str:
    system = platform.system()
    if system == "Darwin":
        return ".dylib"
    if system == "Linux":
        return ".so"
    raise Bsim4NativeError(
        f"native BSIM4.5 currently supports macOS and Linux, not {system}")


def _source_digest(compiler: str) -> str:
    digest = hashlib.sha256()
    digest.update(platform.platform().encode())
    digest.update(platform.machine().encode())
    digest.update(compiler.encode())
    for path in _HASH_INPUTS:
        if not path.is_file():
            raise Bsim4NativeError(f"packaged BSIM4 source is missing: {path}")
        digest.update(path.relative_to(_SOURCE_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


_build_failure: "Bsim4NativeError | None" = None


def _build_library() -> Path:
    global _build_failure
    compiler = _compiler()
    digest = _source_digest(compiler)
    cache = Path(native_model_cache_dir())
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"libcircuitopt_bsim4v5_{digest}{_library_suffix()}"
    if output.is_file():
        return output

    with _build_lock:
        if output.is_file():
            return output
        # A failed build is deterministic for this process (same sources,
        # same compiler): retrying it for every device/test repeats the
        # full compile just to fail again — a CI run once burned 2.5 h on
        # ~100 such retries. Fail fast after the first attempt.
        if _build_failure is not None:
            raise Bsim4NativeError(
                "native BSIM4.5 build already failed in this process "
                f"(not retrying): {_build_failure}") from _build_failure
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        command = [
            compiler,
            "-O2",
            "-std=c99",
            "-fPIC",
            # clang 16+ promotes implicit-function-declaration to an error in
            # C99 mode; the vendored Berkeley sources rely on implicit libc
            # declarations (e.g. strcmp in b4v5set.c). Keep it a warning so
            # the unmodified vendor tree builds on current Linux toolchains.
            "-Wno-error=implicit-function-declaration",
            "-I",
            str(_INCLUDE_DIR),
            "-I",
            str(_MODEL_DIR),
        ]
        if platform.system() == "Darwin":
            command.append("-dynamiclib")
        else:
            command.append("-shared")
        command.extend(str(path) for path in _SOURCES)
        command.extend(("-lm", "-o", str(temporary)))
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _build_failure = Bsim4NativeError(
                f"failed to run native BSIM4 compiler: {exc}")
            raise _build_failure from exc
        if result.returncode != 0:
            temporary.unlink(missing_ok=True)
            detail = (result.stderr or result.stdout).strip()
            _build_failure = Bsim4NativeError(
                f"native BSIM4.5 build failed with {compiler}:\n{detail}")
            raise _build_failure
        os.replace(temporary, output)
    return output


def _bind_abi(library) -> None:
    """Bind the four-terminal BSIM4 C ABI onto a loaded ``ctypes`` library.

    Both backends export the identical ABI (host.c for ``cc``; the compiled
    ``circuitopt_core`` cdylib for ``rust``), so the binding is shared. This
    does not affect numerical results — it only declares the argument/return
    marshalling and installs the Numba runtime function pointer.
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
    library._co_bsim4_eval_vp = _EVAL_VP_T(
        ("co_bsim4_eval_vp", library))


def _bind_library():
    global _library
    if _library is not None:
        return _library
    with _build_lock:
        if _library is not None:
            return _library
        path = _build_library()
        library = C.CDLL(str(path))
        library.co_bsim4_abi_version.argtypes = ()
        library.co_bsim4_abi_version.restype = C.c_uint
        abi_version = int(library.co_bsim4_abi_version())
        if abi_version != _ABI_VERSION:
            raise Bsim4NativeError(
                f"native BSIM4 ABI version {abi_version} != expected {_ABI_VERSION}")
        _bind_abi(library)
        _library = library
    return _library


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
    """Read the backend selector at call time (never baked at import).

    Mirrors ``ngspice_char.ngspice_chain_enabled``: the environment variable
    wins on every call, so a process can switch backends between evaluations.
    """
    value = os.environ.get("CIRCUIT_BSIM4_BACKEND", _DEFAULT_BACKEND).strip().lower()
    if value not in ("cc", "rust"):
        raise Bsim4NativeError(
            f"CIRCUIT_BSIM4_BACKEND must be 'cc' or 'rust', got {value!r}")
    return value


def _select_library(backend: str):
    return _bind_rust_library() if backend == "rust" else _bind_library()


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
        backend: str = "cc",
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
        """Process-local opaque handle used by the Numba/ctypes runtime bridge."""
        if not self._pointer:
            raise Bsim4NativeError("BSIM4 native device is closed")
        return int(self._pointer)

    @property
    def kernel_evaluator(self):
        """Runtime ctypes pointer for the all-``void *`` evaluation ABI."""
        return self._library._co_bsim4_eval_vp

    def close(self) -> None:
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


class NativeBsim4Backend:
    """Berkeley BSIM4.5 evaluator hosted by CircuitOpt in the current process."""

    name = "berkeley-bsim4v5-native"
    version = "4.5.0"
    abi_version = _ABI_VERSION

    def __init__(self, *, cache_size: int | None = None):
        if cache_size is None:
            cache_size = int(os.environ.get("BSIM4_DEVICE_CACHE_SIZE", "32"))
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self._cache_size = cache_size
        self._devices: OrderedDict[tuple, _NativeDevice] = OrderedDict()
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

    def _device(
        self,
        model: Bsim4ModelCard,
        instance: Bsim4InstanceCard,
        temperature_k: float,
        backend: str,
    ) -> _NativeDevice:
        if self._cache_size == 0:
            return _NativeDevice(model, instance, temperature_k, backend=backend)
        key = self._key(model, instance, temperature_k, backend)
        with self._lock:
            device = self._devices.get(key)
            if device is not None:
                self._devices.move_to_end(key)
                return device
            device = _NativeDevice(model, instance, temperature_k, backend=backend)
            self._devices[key] = device
            while len(self._devices) > self._cache_size:
                _, evicted = self._devices.popitem(last=False)
                evicted.close()
            return device

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
        device = self._device(model, instance, bias.temperature_k, backend)
        try:
            return device.evaluate(bias, frequency_hz)
        finally:
            if self._cache_size == 0:
                device.close()

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

    def close(self) -> None:
        with self._lock:
            for device in self._devices.values():
                device.close()
            self._devices.clear()
