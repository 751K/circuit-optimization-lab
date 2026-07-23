"""R9: silicon compiled-campaign arm of ``corners.corner_table`` / ``mismatch_mc``.

The silicon (BSIM4) corner sweep and per-device mismatch MC route an all-silicon
circuit through :class:`circuitopt._rust_campaign` (one Rayon pool, per-candidate
corner, ``workers`` scaled, no per-candidate Python callback). The frozen scalar
``metrics`` path — the exact ``ac_solve`` + ``noise_analysis`` the rest of the
stack uses, under the same ``binding`` — is the reference the campaign is
validated against and rolls back to. AFE / default-PDK stays on that scalar path
untouched (the multistable OTFT would let a cold campaign under-report the latch
rate, the R5-D red line).

Gates here:
* per-corner parity of the wired ``corner_table`` campaign arm vs the scalar
  reference across the three PDK 5T OTAs (cold behaviour gate — same convergence,
  metrics well inside the calibration tolerance; the seeded bit-for-bit gate lives
  in ``test_rust_campaign.test_silicon_parity_seeded_bit_for_bit``),
* determinism across worker counts {1, 2, 8} (byte-identical),
* zero Python device/solver frame during the wired batch (a counting + sampling
  trap), and
* the AFE guard: an AFE circuit never routes to the campaign.

D12: only counts / worst-case relative errors are computed; no card text or numeric
value is asserted.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CIRCUIT_ENGINE", "rust")

import numpy as np
import pytest

pytest.importorskip("circuitopt_core")

from circuitopt import corners as C
from circuitopt._campaign_sweep import campaign_enabled, silicon_campaign_for
from circuitopt.circuit_loader import load_circuit_json

_SI_FREQS = np.logspace(3, 7, 25)
_SI_BAND = (1e3, 1e6)

# 5T OTAs + a fully-differential OTA (2 outputs -> a meaningful latch_dV). sky130
# and tsmc28 are restricted to card corners with bundled/​resolvable bins for the
# example geometry (sky130 bundles ff only for a few widths; tsmc28 ff/sf/fs select
# zero bins on some geometries — both documented in co-pdk/PARITY.md).
_CASES = {
    "freepdk45_5t": ("examples/freepdk45_5t_ota.json", ("nom", "ss", "ff")),
    "sky130_5t": ("examples/sky130_5t_ota.json", ("tt", "ss")),
    "tsmc28_5t": ("examples/tsmc28hpcp_5t_ota.json", ("tt", "ss")),
    "freepdk45_fd": ("examples/freepdk45_fd_ota.json", ("nom", "ss", "ff")),
}


def _require_rust():
    if not campaign_enabled():
        pytest.skip("silicon campaign arm requires the rust device engine")


def _ready(path):
    """Skip if the PDK cards for ``path`` are not installed in this checkout."""
    if "tsmc28" in path and not os.environ.get("TSMC28_PDK_ROOT"):
        pytest.skip("TSMC28_PDK_ROOT not set")
    if "freepdk45" in path:
        from circuitopt.toolchain import pdk_root
        if not os.path.isfile(os.path.join(pdk_root(), "freepdk45", "models_nom",
                                           "NMOS_VTG.inc")):
            pytest.skip("FreePDK45 cards not present")


def _load(path):
    spec = load_circuit_json(path)
    return spec, spec.binding()


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


@pytest.mark.parametrize("key", list(_CASES))
def test_corner_table_silicon_matches_scalar_reference(key):
    """Wired ``corner_table`` campaign arm vs the frozen scalar reference, per corner.

    Cold behaviour gate: both converge the same corners, and every metric agrees far
    inside the 1e-3 calibration tolerance (in practice bit-for-bit on freepdk45/sky130
    and ~1e-9 relative on tsmc28, the cold-Newton-vs-fsolve DC-root floor)."""
    _require_rust()
    path, cs = _CASES[key]
    _ready(path)
    spec, binding = _load(path)
    camp = C.corner_table(spec.sizes, spec.bias, nf=spec.nf, topo=spec.topology,
                          corners=cs, freqs=_SI_FREQS, band=_SI_BAND, binding=binding)
    # Force the scalar reference (no campaign) over the same binding/corners.
    scal = C._corner_table_silicon(None, spec.sizes, spec.bias, spec.nf,
                                   spec.topology, cs, _SI_FREQS, _SI_BAND, True, 1,
                                   binding)
    assert set(camp) == set(scal) == set(cs)
    for c in cs:
        a, s = camp[c], scal[c]
        assert (a is None) == (s is None), f"{key} {c}: convergence disagrees"
        if a is None:
            continue
        assert _rel(a["gain_peak_dB"], s["gain_peak_dB"]) <= 1e-7, (key, c, "gain")
        assert _rel(a["bw_Hz"], s["bw_Hz"]) <= 1e-7, (key, c, "bw")
        if np.isfinite(a["irn_uV"]) or np.isfinite(s["irn_uV"]):
            assert _rel(a["irn_uV"], s["irn_uV"]) <= 1e-7, (key, c, "irn")
        assert abs(a["latch_dV"] - s["latch_dV"]) <= 1e-6, (key, c, "latch")


@pytest.mark.parametrize("key", list(_CASES))
def test_corner_table_silicon_deterministic_across_workers(key):
    """corner_table campaign arm is byte-identical for workers in {1, 2, 8}."""
    _require_rust()
    path, cs = _CASES[key]
    _ready(path)
    spec, binding = _load(path)

    def run(w):
        return C.corner_table(spec.sizes, spec.bias, nf=spec.nf, topo=spec.topology,
                              corners=cs, freqs=_SI_FREQS, band=_SI_BAND,
                              workers=w, binding=binding)

    base = run(1)
    for w in (1, 2, 8):
        got = run(w)
        for c in cs:
            a, b = base[c], got[c]
            assert (a is None) == (b is None), (key, w, c)
            if a is None:
                continue
            for k in ("gain_peak_dB", "bw_Hz", "irn_uV", "latch_dV"):
                assert (a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k]))), \
                    f"{key} workers={w} {c} {k}: {a[k]!r} != {b[k]!r}"


class _FrameTrap:
    """Count Python-frame entries into any module whose name contains a needle."""

    def __init__(self, needles):
        self.needles = needles
        self.hits = 0

    def __enter__(self):
        sys.setprofile(self._hook)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)

    def _hook(self, frame, event, _arg):
        if event == "call":
            name = frame.f_globals.get("__name__", "")
            if any(n in name for n in self.needles):
                self.hits += 1


def test_corner_table_silicon_zero_python_device_frame(monkeypatch):
    """The wired silicon corner batch makes no Python BSIM4/solver callback or frame.

    The campaign template is built first (that one build extracts candidate-invariant
    statics); the trap wraps only the batch evaluation, where a per-candidate Python
    device/solver call would break ``workers`` scaling under the released GIL."""
    _require_rust()
    path, cs = _CASES["freepdk45_5t"]
    _ready(path)
    spec, binding = _load(path)
    camp = silicon_campaign_for(spec.topology, spec.sizes, spec.bias, spec.nf,
                                binding, _SI_FREQS, _SI_BAND)
    assert camp is not None and camp.family == "silicon_bsim4"

    from circuitopt.compact_models.bsim4 import NativeBsim4Backend
    import circuitopt.ac_solver as acmod
    import circuitopt.noise_solver as nzmod

    def boom(*_a, **_k):
        raise AssertionError("python BSIM4/solver callback during wired corner batch")

    monkeypatch.setattr(NativeBsim4Backend, "evaluate", boom)
    monkeypatch.setattr(NativeBsim4Backend, "evaluate_batch", staticmethod(boom))
    monkeypatch.setattr(NativeBsim4Backend, "noise_batch", staticmethod(boom))
    monkeypatch.setattr(acmod, "ac_solve", boom)
    monkeypatch.setattr(nzmod, "noise_analysis", boom)

    with _FrameTrap(("compact_models.bsim4", "circuitopt.pdk",
                     "circuitopt.ac_solver", "circuitopt.noise_solver")) as trap:
        out = C._corner_table_silicon(camp, spec.sizes, spec.bias, spec.nf,
                                      spec.topology, cs, _SI_FREQS, _SI_BAND,
                                      True, 4, binding)
    assert all(out[c] is not None for c in cs)
    assert trap.hits == 0, f"{trap.hits} Python PDK/device frames in the corner batch"


def test_corner_table_afe_stays_scalar():
    """AFE circuits never route to the campaign, and the AFE result is binding-invariant.

    ``silicon_campaign_for`` returns None for the AFE family, and ``corner_table``
    with an AFE binding is byte-identical to the legacy no-binding path (the binding
    is not threaded into the scalar OTFT solve, so no default DC seed is injected)."""
    _require_rust()
    spec = load_circuit_json("examples/afe_explore.json")
    binding = spec.binding()
    assert silicon_campaign_for(spec.topology, spec.sizes, spec.bias, spec.nf,
                                binding, C._DEFAULT_FREQS, (0.05, 100.0)) is None

    freqs = np.logspace(-2, 4, 61)
    legacy = C.corner_table(spec.sizes, spec.bias, nf=spec.nf, topo=spec.topology,
                            freqs=freqs)
    with_binding = C.corner_table(spec.sizes, spec.bias, nf=spec.nf,
                                  topo=spec.topology, freqs=freqs, binding=binding)
    assert set(legacy) == set(with_binding)
    for c in legacy:
        a, b = legacy[c], with_binding[c]
        assert (a is None) == (b is None)
        if a is None:
            continue
        for k in ("gain_peak_dB", "bw_Hz", "irn_uV", "latch_dV"):
            assert (a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k]))), (c, k)
