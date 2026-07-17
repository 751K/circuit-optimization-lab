"""The native BSIM4 build must fail fast after its first failed attempt.

A CI run once spent 2.5 h re-running the identical failing compile for every
silicon test (~100 x full 16k-line vendor build). The build outcome is
deterministic per (sources, compiler, process), so the first failure is
cached and re-raised immediately.
"""
from __future__ import annotations

import time

import pytest

from circuitopt.compact_models.bsim4 import native


@pytest.fixture()
def _isolated_build_state(monkeypatch, tmp_path):
    """Point the builder at a broken compiler and an empty cache dir."""
    monkeypatch.setattr(native, "_build_failure", None)
    monkeypatch.setattr(native, "_compiler", lambda: "/usr/bin/false")
    monkeypatch.setattr(
        native, "native_model_cache_dir", lambda: str(tmp_path))
    yield
    # state is process-global; leave it clean for other tests
    native._build_failure = None


def test_second_build_attempt_fails_fast_with_cached_error(
        _isolated_build_state):
    with pytest.raises(native.Bsim4NativeError):
        native._build_library()
    assert native._build_failure is not None

    start = time.perf_counter()
    with pytest.raises(native.Bsim4NativeError) as second:
        native._build_library()
    elapsed = time.perf_counter() - start

    assert "already failed in this process" in str(second.value)
    assert elapsed < 1.0, "cached failure must not re-run the compiler"


def test_build_failure_cache_does_not_mask_existing_library(
        _isolated_build_state, tmp_path):
    """A pre-existing cached library wins even after an earlier failure."""
    with pytest.raises(native.Bsim4NativeError):
        native._build_library()

    digest = native._source_digest("/usr/bin/false")
    fake = (tmp_path
            / f"libcircuitopt_bsim4v5_{digest}{native._library_suffix()}")
    fake.write_bytes(b"not a real dylib")
    assert native._build_library() == fake
