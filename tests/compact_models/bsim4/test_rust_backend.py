"""Acceptance tests for the R2 Rust BSIM4 backend (CIRCUIT_BSIM4_BACKEND=rust).

Four checks:

(a) golden parity   — three PDKs x both polarities, one geometry/corner/temp
    each (including tsmc28hpcp), evaluated through the full
    ``device_model.create_device`` stack under the rust backend and compared to
    the frozen v1.4.0 corpus (``rel <= 1e-13``); skipped if the corpus is absent.
(b) rust vs cc      — the same bias through both backends, compared directly.
(c) call-time switch — the backend selector is honoured per call within one
    process (never baked at import).
(d) clear error     — a missing ``circuitopt_core`` under the rust backend raises
    an actionable ``Bsim4NativeError``.

The golden corpus lives in the main checkout and is referenced read-only by
absolute path (it is not vendored into per-agent worktrees).
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

from circuitopt.compact_models.bsim4 import native
from circuitopt.compact_models.bsim4.native import Bsim4NativeError, NativeBsim4Backend
from circuitopt.compact_models.bsim4.abi import (
    Bsim4Bias,
    Bsim4InstanceCard,
    Bsim4ModelCard,
)

GOLDEN_PATH = Path(
    "/Users/kong/PycharmProjects/circuit/circuit-optimization-lab"
    "/tests/golden/engine_parity/devices.npz"
)

# One representative (corner, temperature) shared by every PDK/polarity in the
# corpus; the freeze tool records the nominal corner at room temperature.
_SAMPLE_CORNER = "nom"
_SAMPLE_TEMP = 300.15
_PDKS = ("freepdk45", "sky130", "tsmc28hpcp")
_POLARITIES = ("nmos", "pmos")
_NOISE_FREQS = (1e3, 1e6, 1e9)

_TAG_RE = re.compile(r"W(?P<w>[\d.]+)L(?P<l>[\d.]+)NF(?P<nf>\d+)")


def _has_rust_core() -> bool:
    try:
        native._import_circuitopt_core()
    except ImportError:
        return False
    return True


requires_rust = pytest.mark.skipif(
    not _has_rust_core(),
    reason="circuitopt_core (rust backend) not importable; "
    "build with `maturin develop --release -m rust/crates/co-py/Cargo.toml`",
)
requires_golden = pytest.mark.skipif(
    not GOLDEN_PATH.is_file(),
    reason=f"engine-parity golden corpus not present at {GOLDEN_PATH}",
)


def _load_golden():
    return np.load(GOLDEN_PATH)


# Availability keywords: a PDK whose licensed deck / card root is not configured
# in this environment should skip, not fail (mirrors test_tsmc28_5t_ota).
_UNAVAILABLE = ("not found", "not configured", "unavailable", "missing", "no such")


def _create_or_skip(model_name, **kwargs):
    from circuitopt.device_model import create_device

    try:
        return create_device(model_name, **kwargs)
    except Exception as exc:  # noqa: BLE001 - narrow via message inspection
        text = str(exc).lower()
        if any(word in text for word in _UNAVAILABLE):
            pytest.skip(f"{model_name} PDK deck/cards not available: {exc}")
        raise


def _sample_tag(golden, pdk: str, pol: str) -> str | None:
    """Smallest-geometry device tag for ``pdk``/``pol`` at nominal corner/room temp.

    Sorting is numeric on (W, L, NF), so the pick is the freeze tool's first
    ``entries[0]`` geometry — the one the geometry-independent ``noise_typ`` key
    was frozen with — not a lexical accident (``W18.7`` < ``W1L`` as strings).
    """
    marker = f"|{_SAMPLE_CORNER}|T{_SAMPLE_TEMP:g}|"
    tags = [
        key[:-2]
        for key in golden.files
        if key.startswith(f"{pdk}|{pol}|") and key.endswith("|I") and marker in key
    ]
    if not tags:
        return None
    return min(tags, key=lambda tag: (lambda s: (s["w"], s["l"], s["nf"]))(_parse_tag(tag)))


def _parse_tag(tag: str):
    _, pol, geom, corner, temp = tag.split("|")
    match = _TAG_RE.fullmatch(geom)
    assert match, f"unparseable geometry field {geom!r}"
    return {
        "pol": pol,
        "w": float(match["w"]),
        "l": float(match["l"]),
        "nf": int(match["nf"]),
        "corner": corner,
        "temperature": float(temp[1:]),
    }


def _recompute_grid(device, vg_pts, vd_pts, vs=0.0):
    n_g, n_d = len(vg_pts), len(vd_pts)
    grids = {
        "I": np.empty((n_g, n_d, 4)),
        "Q": np.empty((n_g, n_d, 4)),
        "G": np.empty((n_g, n_d, 4, 4)),
        "C": np.empty((n_g, n_d, 4, 4)),
    }
    for i, vg in enumerate(vg_pts):
        for j, vd in enumerate(vd_pts):
            grids["I"][i, j] = device.get_terminal_currents(vs, float(vd), float(vg))
            grids["Q"][i, j] = device.get_terminal_charges(vs, float(vd), float(vg))
            g4, c4 = device.get_terminal_linearization(vs, float(vd), float(vg))
            grids["G"][i, j] = g4
            grids["C"][i, j] = c4
    return grids


def _max_rel(fresh: np.ndarray, gold: np.ndarray, floor: float) -> float:
    """Max-norm relative error, floored so exact-zero references don't divide."""
    scale = max(float(np.max(np.abs(gold))), floor)
    return float(np.max(np.abs(fresh - gold))) / scale


# Per-quantity magnitude floors (A, C, S, F) matching the abi.py conservation
# scales; used only to avoid 0/0 when a whole array is ~0.
_FLOORS = {"I": 1e-18, "Q": 1e-24, "G": 1e-15, "C": 1e-24}


@requires_rust
@requires_golden
@pytest.mark.parametrize("pdk", _PDKS)
@pytest.mark.parametrize("pol", _POLARITIES)
def test_golden_parity_rust(pdk, pol, monkeypatch):
    golden = _load_golden()
    tag = _sample_tag(golden, pdk, pol)
    if tag is None:
        pytest.skip(f"no {pdk}|{pol} device at {_SAMPLE_CORNER}/T{_SAMPLE_TEMP:g} in corpus")
    spec = _parse_tag(tag)
    vg_pts = golden[f"{pdk}|Vg"]
    vd_pts = golden[f"{pdk}|Vd"]

    monkeypatch.setenv("CIRCUIT_BSIM4_BACKEND", "rust")
    device = _create_or_skip(
        f"{pdk}.{pol}",
        W=spec["w"],
        L=spec["l"],
        NF=spec["nf"],
        corner=spec["corner"],
        temperature=spec["temperature"],
    )
    grids = _recompute_grid(device, vg_pts, vd_pts)

    worst = {}
    for name, arr in grids.items():
        gold = golden[f"{tag}|{name}"]
        assert arr.shape == gold.shape
        worst[name] = _max_rel(arr, gold, _FLOORS[name])
        assert worst[name] <= 1e-13, (
            f"{pdk}|{pol} {name} rust-vs-golden rel {worst[name]:.3e} exceeds 1e-13"
        )

    # Scalar-noise PSD at the typical bias (freeze uses defaults = nom/300.15).
    noise_key = f"{pdk}|{pol}|noise_typ"
    if noise_key in golden.files:
        vdd = float(vd_pts[-1])
        noise_dev = _create_or_skip(
            f"{pdk}.{pol}", W=spec["w"], L=spec["l"], NF=spec["nf"])
        pts = np.array(
            [noise_dev.get_noise_psd(0.0, vdd / 2.0, vdd / 2.0, f) for f in _NOISE_FREQS]
        )
        gold_noise = golden[noise_key]
        rel_noise = _max_rel(pts, gold_noise, 1e-30)
        assert rel_noise <= 1e-13, f"{pdk}|{pol} noise rel {rel_noise:.3e} exceeds 1e-13"


def _cards(pol: int):
    model = Bsim4ModelCard(
        polarity=pol, parameters={"toxe": 1.4e-9, "toxp": 1.4e-9, "tnom": 27.0}
    )
    instance = Bsim4InstanceCard(parameters={"w": 1e-6, "l": 1e-7, "nf": 1.0})
    return model, instance


@requires_rust
def test_backend_choice_is_call_time_and_cc_is_removed(monkeypatch):
    """The selector is read on every call; rust is the default and the only
    value. The retired ``cc`` runtime-compile backend errors loudly (v2.0.0
    removal), mirroring the engine-switch removals."""
    model, instance = _cards(1)
    bias = Bsim4Bias(drain=0.5, gate=0.5, source=0.0, bulk=0.0)
    backend = NativeBsim4Backend(cache_size=0)

    monkeypatch.delenv("CIRCUIT_BSIM4_BACKEND", raising=False)
    assert native._backend_choice() == "rust"
    default_eval = backend.evaluate(model, instance, bias)
    assert np.all(np.isfinite(default_eval.terminal_currents))

    monkeypatch.setenv("CIRCUIT_BSIM4_BACKEND", "rust")
    assert native._backend_choice() == "rust"
    rust = backend.evaluate(model, instance, bias)
    assert np.array_equal(rust.terminal_currents, default_eval.terminal_currents)

    monkeypatch.setenv("CIRCUIT_BSIM4_BACKEND", "cc")
    with pytest.raises(Bsim4NativeError, match="removed in v2.0.0"):
        native._backend_choice()
    with pytest.raises(Bsim4NativeError, match="removed in v2.0.0"):
        backend.evaluate(model, instance, bias)


@requires_rust
def test_rust_batch_matches_ordered_scalar_evaluations(monkeypatch):
    monkeypatch.setenv("CIRCUIT_BSIM4_BACKEND", "rust")
    model, instance = _cards(1)
    backend = NativeBsim4Backend(cache_size=0)
    devices = [backend.create_device(model, instance, 300.15) for _ in range(16)]
    terminals = np.asarray([
        (0.25 + 0.02 * index, 0.5, 0.0, 0.0) for index in range(len(devices))
    ])
    try:
        expected = [
            device.evaluate(Bsim4Bias(*values)) for device, values in zip(devices, terminals)
        ]
        currents, conductance, charges, capacitance = backend.evaluate_batch(
            devices, terminals)
        np.testing.assert_array_equal(
            currents, np.asarray([value.terminal_currents for value in expected]))
        np.testing.assert_array_equal(
            conductance, np.asarray([value.conductance for value in expected]))
        np.testing.assert_array_equal(
            charges, np.asarray([value.terminal_charges for value in expected]))
        np.testing.assert_array_equal(
            capacitance, np.asarray([value.capacitance for value in expected]))
    finally:
        for device in devices:
            device.close()


@requires_rust
def test_rust_noise_batch_matches_ordered_scalar_sweeps(monkeypatch):
    monkeypatch.setenv("CIRCUIT_BSIM4_BACKEND", "rust")
    model, instance = _cards(1)
    backend = NativeBsim4Backend(cache_size=0)
    devices = [backend.create_device(model, instance, 300.15) for _ in range(8)]
    terminals = np.asarray([
        (0.3 + 0.03 * index, 0.55, 0.0, 0.0) for index in range(len(devices))
    ])
    frequencies = np.asarray((1e3, 1e6, 1e9))
    try:
        expected_total = []
        expected_flicker = []
        for device, values in zip(devices, terminals):
            total_rows = []
            flicker_rows = []
            for frequency in frequencies:
                result = device.evaluate(Bsim4Bias(*values), float(frequency))
                total_rows.append(result.noise.spectral_density)
                flicker_rows.append(result.noise.components["flicker"])
            expected_total.append(total_rows)
            expected_flicker.append(flicker_rows)

        backend.evaluate_batch(devices, terminals)
        total, flicker = backend.noise_batch(devices, frequencies)
        np.testing.assert_array_equal(total, np.asarray(expected_total))
        np.testing.assert_array_equal(flicker, np.asarray(expected_flicker))
    finally:
        for device in devices:
            device.close()


@requires_rust
def test_rust_cache_never_evicts_active_handles(monkeypatch):
    monkeypatch.setenv("CIRCUIT_BSIM4_BACKEND", "rust")
    model, _ = _cards(1)
    instances = [
        Bsim4InstanceCard(parameters={"w": 1e-6 + index * 1e-8,
                                      "l": 1e-7, "nf": 1.0})
        for index in range(8)
    ]
    bias = Bsim4Bias(drain=0.6, gate=0.55, source=0.0, bulk=0.0)
    reference = [
        NativeBsim4Backend(cache_size=0).evaluate(model, instance, bias)
        for instance in instances
    ]
    backend = NativeBsim4Backend(cache_size=2)
    barrier = Barrier(len(instances))

    def evaluate(index):
        barrier.wait()
        values = [backend.evaluate(model, instances[index], bias) for _ in range(16)]
        return values

    with ThreadPoolExecutor(max_workers=len(instances)) as executor:
        results = list(executor.map(evaluate, range(len(instances))))
    for index, values in enumerate(results):
        for value in values:
            np.testing.assert_array_equal(
                value.terminal_currents, reference[index].terminal_currents)
            np.testing.assert_array_equal(value.conductance, reference[index].conductance)
    assert len(backend._devices) <= 2


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("CIRCUIT_BSIM4_BACKEND", "gpu")
    with pytest.raises(Bsim4NativeError, match="must be 'rust'"):
        native._backend_choice()


def test_missing_core_raises_clear_error(monkeypatch):
    """CIRCUIT_BSIM4_BACKEND=rust with no compiled core gives an actionable error."""

    def _boom():
        raise ImportError("no module named circuitopt_core")

    # Force a rebind and make the import fail.
    monkeypatch.setattr(native, "_rust_library", None)
    monkeypatch.setattr(native, "_import_circuitopt_core", _boom)
    with pytest.raises(Bsim4NativeError, match="requires the compiled circuitopt_core"):
        native._bind_rust_library()
