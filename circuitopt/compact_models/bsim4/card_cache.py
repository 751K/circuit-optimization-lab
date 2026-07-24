"""Shared immutable-card cache for native BSIM4 PDK adapters."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import os
import threading
from typing import Callable, Generic, Hashable, Mapping, TypeVar


_BundleT = TypeVar("_BundleT")
_ExtraValue = str | int | float | bool | None


@dataclass(frozen=True)
class Bsim4SourceFingerprint:
    """Identity of one model source used to invalidate derived cards."""

    path: str
    mtime_ns: int
    size: int

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> Bsim4SourceFingerprint:
        resolved = os.path.abspath(os.path.expanduser(os.fspath(path)))
        stat = os.stat(resolved)
        return cls(resolved, stat.st_mtime_ns, stat.st_size)


@dataclass(frozen=True)
class Bsim4CardCacheKey:
    """Complete, simulator-neutral identity of a derived BSIM4 card bundle."""

    source: Bsim4SourceFingerprint
    pdk: str
    model: str
    section: str
    bin: str
    width_um: float
    length_um: float
    nf: int
    mult: int
    temperature_c: float
    corner: str
    mismatch_v: float
    extra: tuple[tuple[str, _ExtraValue], ...] = ()


@dataclass(frozen=True)
class Bsim4CardCacheInfo:
    """Snapshot of a bounded BSIM4 card cache."""

    hits: int
    misses: int
    maxsize: int
    currsize: int


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def freeze_card_parameters(
    parameters: Mapping[str, object] | None,
) -> tuple[tuple[str, _ExtraValue], ...]:
    """Return a deterministic hashable representation of PDK-specific fields."""
    frozen = []
    for raw_name, raw_value in (parameters or {}).items():
        name = str(raw_name).strip().lower()
        if not name:
            raise ValueError("BSIM4 card-cache extra parameter name is empty")
        if raw_value is None or isinstance(raw_value, (str, bool)):
            value: _ExtraValue = raw_value
        elif isinstance(raw_value, int):
            value = raw_value
        else:
            value = _finite_float(raw_value, f"card-cache parameter {name!r}")
        frozen.append((name, value))
    frozen.sort(key=lambda item: item[0])
    if len({name for name, _ in frozen}) != len(frozen):
        raise ValueError("duplicate normalized BSIM4 card-cache parameter")
    return tuple(frozen)


def make_bsim4_card_cache_key(
    *,
    source: Bsim4SourceFingerprint,
    pdk: str,
    model: str,
    section: str,
    bin_selector: str,
    width_um: float,
    length_um: float,
    nf: int = 1,
    mult: int = 1,
    temperature_c: float = 27.0,
    corner: str,
    mismatch_v: float = 0.0,
    extra: Mapping[str, object] | None = None,
) -> Bsim4CardCacheKey:
    """Normalize and validate the common native-BSIM4 cache-key contract."""
    strings = {
        "pdk": str(pdk).strip().lower(),
        "model": str(model).strip().lower(),
        "section": str(section).strip().lower(),
        "bin": str(bin_selector).strip().lower(),
        "corner": str(corner).strip().lower(),
    }
    empty = [name for name, value in strings.items() if not value]
    if empty:
        raise ValueError(
            "BSIM4 card-cache binding fields must be non-empty: "
            + ", ".join(empty))
    width = _finite_float(width_um, "width_um")
    length = _finite_float(length_um, "length_um")
    temperature = _finite_float(temperature_c, "temperature_c")
    mismatch = _finite_float(mismatch_v, "mismatch_v")
    fingers = int(nf)
    multiplicity = int(mult)
    if width <= 0.0 or length <= 0.0:
        raise ValueError("BSIM4 card-cache width and length must be positive")
    if fingers < 1 or fingers != nf or multiplicity < 1 or multiplicity != mult:
        raise ValueError("BSIM4 card-cache nf and mult must be positive integers")
    if temperature <= -273.15:
        raise ValueError("BSIM4 card-cache temperature must exceed absolute zero")
    return Bsim4CardCacheKey(
        source=source,
        pdk=strings["pdk"],
        model=strings["model"],
        section=strings["section"],
        bin=strings["bin"],
        width_um=width,
        length_um=length,
        nf=fingers,
        mult=multiplicity,
        temperature_c=temperature,
        corner=strings["corner"],
        mismatch_v=mismatch,
        extra=freeze_card_parameters(extra),
    )


class Bsim4CardCache(Generic[_BundleT]):
    """Bounded LRU with per-key single-flight construction."""

    def __init__(self, maxsize: int = 1024):
        maxsize = int(maxsize)
        if maxsize < 1:
            raise ValueError("BSIM4 card-cache maxsize must be positive")
        self._maxsize = maxsize
        self._entries: OrderedDict[Hashable, _BundleT] = OrderedDict()
        self._pending: dict[Hashable, threading.Event] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._generation = 0

    def get_or_create(
        self,
        key: Hashable,
        factory: Callable[[], _BundleT],
    ) -> _BundleT:
        """Return one identity-stable value, constructing each cold key once."""
        while True:
            with self._lock:
                cached = self._entries.get(key)
                if cached is not None:
                    self._entries.move_to_end(key)
                    self._hits += 1
                    return cached
                pending = self._pending.get(key)
                if pending is None:
                    pending = threading.Event()
                    self._pending[key] = pending
                    generation = self._generation
                    break
            pending.wait()

        try:
            created = factory()
        except BaseException:
            with self._lock:
                self._pending.pop(key, None)
                pending.set()
            raise

        with self._lock:
            if generation == self._generation:
                self._entries[key] = created
                self._entries.move_to_end(key)
                self._misses += 1
                while len(self._entries) > self._maxsize:
                    self._entries.popitem(last=False)
            self._pending.pop(key, None)
            pending.set()
        return created

    def cache_info(self) -> Bsim4CardCacheInfo:
        with self._lock:
            return Bsim4CardCacheInfo(
                hits=self._hits,
                misses=self._misses,
                maxsize=self._maxsize,
                currsize=len(self._entries),
            )

    def clear(self) -> None:
        """Drop completed entries without disrupting in-flight constructors."""
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0
            self._generation += 1
