"""Shared native-BSIM4 card-cache contract."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from circuitopt.compact_models.bsim4 import (
    Bsim4CardCache,
    Bsim4SourceFingerprint,
    make_bsim4_card_cache_key,
)


_SOURCE = Bsim4SourceFingerprint("/models/test.card", 123, 456)


def _key(**overrides):
    values = {
        "source": _SOURCE,
        "pdk": "test-pdk",
        "model": "nmos",
        "section": "tt",
        "bin_selector": "auto",
        "width_um": 1.0,
        "length_um": 0.05,
        "nf": 2,
        "mult": 1,
        "temperature_c": 27.0,
        "corner": "tt",
        "mismatch_v": 0.0,
        "extra": {"rgeo": 1, "reference": "default"},
    }
    values.update(overrides)
    return make_bsim4_card_cache_key(**values)


def test_common_key_normalizes_binding_and_extra_parameters():
    first = _key(pdk=" TEST-PDK ", model="NMOS", mismatch_v=-0.0)
    second = _key(extra={"reference": "default", "rgeo": 1})
    assert first == second
    assert first.extra == (("reference", "default"), ("rgeo", 1))
    assert _key(temperature_c=125.0) != first
    assert _key(extra={"rgeo": 2}) != first
    with pytest.raises(ValueError, match="finite"):
        _key(mismatch_v=float("nan"))
    with pytest.raises(ValueError, match="non-empty"):
        _key(section="")


def test_common_cache_constructs_one_value_per_cold_key_across_threads():
    cache = Bsim4CardCache[object](maxsize=4)
    calls = 0
    calls_lock = threading.Lock()

    def build():
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.01)
        return object()

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: cache.get_or_create(_key(), build), range(16)))

    assert calls == 1
    assert len({id(value) for value in values}) == 1
    info = cache.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (
        15, 1, 4, 1)


def test_common_cache_is_bounded_lru_and_clear_resets_statistics():
    cache = Bsim4CardCache[object](maxsize=2)
    first = cache.get_or_create(_key(width_um=1.0), object)
    second = cache.get_or_create(_key(width_um=2.0), object)
    assert cache.get_or_create(_key(width_um=1.0), object) is first
    cache.get_or_create(_key(width_um=3.0), object)
    rebuilt_second = cache.get_or_create(_key(width_um=2.0), object)
    assert rebuilt_second is not second
    assert cache.cache_info().currsize == 2
    cache.clear()
    info = cache.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (
        0, 0, 2, 0)
