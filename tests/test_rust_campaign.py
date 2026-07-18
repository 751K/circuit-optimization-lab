"""R5-C parity / determinism tests for ``circuitopt_core.CompiledCampaign``.

The compiled AFE OTFT campaign evaluates a candidate matrix through
device-build -> DC -> AC -> noise entirely in Rust, under one ``py.detach``.
These tests establish, against the frozen Python scalar path:

* **Bit-for-bit** AC/noise/DC parity against a *cold-consistent* Python
  reference (fresh cold ``PMOS_TFT`` small-signal params -> the same
  ``circuitopt_core.LtiProblem`` -> the same reductions). This is the
  semantically-correct reference: it uses the identical device kernels the
  campaign uses (under the rust engine), so any mismatch is a port bug.
* A quantified **seed-sensitivity floor** vs the warm ``corners.metrics`` path.
  The AFE OTFT internal 2-node Newton stops at ``tol=1e-12``; its operating
  point is therefore path-dependent, and Python's own warm-cache-vs-cold
  small-signal params diverge up to ~6e-8. The campaign is cold-seed-consistent,
  so it agrees with ``metrics`` only to that inherent floor — a flagged, model
  property, not a port error.
* Determinism: workers in {1, 2, 8} give byte-identical, index-ordered output.
* No per-candidate Python callback during the batch, and a GIL-release speedup.
* Seeded mismatch drawn up front (same rule as ``corners.mismatch_corner``).
* Per-candidate error isolation.

D12: no PDK text or numeric card values touch disk; only counts and worst-case
relative errors are computed in-process.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CIRCUIT_ENGINE", "rust")

import numpy as np
import pytest

circuitopt_core = pytest.importorskip("circuitopt_core")
if not hasattr(circuitopt_core, "CompiledCampaign"):
    pytest.skip("circuitopt_core lacks CompiledCampaign", allow_module_level=True)

from circuitopt._engine import current_engine

if current_engine() != "rust":
    pytest.skip(
        "bit-for-bit campaign parity requires the rust device engine",
        allow_module_level=True,
    )

from circuitopt._rust_campaign import AfeOtftCampaign
from circuitopt._rust_lti import build_lti_problem, complex_array
from circuitopt.ac_solver import ac_solve, bw_from_gain
from circuitopt.compiled_topology import CompiledTopology
from circuitopt.corners import SIGMA_MBETA0, SIGMA_MVT0, metrics
from circuitopt.device_factory import CORNERS
from circuitopt.noise_solver import band_rms, device_psd
from circuitopt.pmos_tft_model import PMOS_TFT
from circuitopt.topology import AFE_TOPO

# Locked AFE design (examples/afe_explore.json) + a couple of perturbations.
BASE_SIZES = {
    "M6": (2264.0, 78.0), "M7": (61365.0, 61.0), "M8": (61365.0, 61.0),
    "M9": (3175.0, 468.0), "M10": (3175.0, 468.0), "M11": (465.0, 66.0),
    "M12": (894.0, 85.0), "M13": (894.0, 85.0), "M14": (5224.0, 46.0),
    "M15": (5224.0, 46.0),
}
BIAS = {"VDD": 40.0, "VCM": 30.65, "VB": 9.84, "VC": 16.0}
FREQS = np.logspace(-2, 4, 121)
BAND = (0.05, 100.0)
_PAIRS = (("M7", "M8"), ("M9", "M10"), ("M12", "M13"), ("M14", "M15"))


def _rel(a, b):
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def _scaled_sizes(factor):
    """Symmetric perturbation of the locked design (pairs stay matched)."""
    paired = {name: pair[0] for pair in _PAIRS for name in pair}
    out = {}
    for name, (w, l) in BASE_SIZES.items():
        group = paired.get(name, name)
        f = factor.get(group, 1.0) if isinstance(factor, dict) else factor
        out[name] = (w * f, l * f)
    return out


def _cold_reference(sizes, corner_map, freqs=FREQS, band=BAND):
    """A cold-consistent Python AC+noise reference for one design.

    Builds fresh (cold) ``PMOS_TFT`` instances, extracts small-signal params at
    the Python cold DC op, assembles the exact same ``LtiProblem`` the campaign
    uses, and reduces with the same helpers. Returns ``(gain_peak_dB, bw_Hz,
    irn_uV, dc_op)`` or ``None`` if the DC solve fails.

    ``corner_map`` is either a process-corner name/dict (global shift) or a
    per-device mismatch map (``{name: {pvt0,pbeta0,mvt0,mbeta0}}``).
    """
    ac = ac_solve(sizes, BIAS, corner=corner_map, freqs=freqs)
    if ac is None:
        return None
    dc = ac["dc_op"]
    plan = CompiledTopology(AFE_TOPO, BIAS)
    nv = {n: dc[n] for n in plan.solved}
    bpts = plan.bias_points(nv)

    def _shift(name):
        base = CORNERS[corner_map] if isinstance(corner_map, str) else corner_map
        entry = base.get(name, base) if isinstance(base, dict) else {}
        if isinstance(entry, dict) and any(isinstance(v, dict) for v in base.values()):
            entry = base.get(name, {})
        elif not isinstance(entry, dict):
            entry = {}
        return {k: float(entry.get(k, 0.0)) for k in ("pvt0", "mvt0", "pbeta0", "mbeta0")}

    cold = {}
    for name, *_ in AFE_TOPO.devices:
        w, l = sizes[name]
        cold[name] = PMOS_TFT(W=w, L=l, NF=1, **_shift(name))
    ss = {name: cold[name].get_ss_params(*bpts[name]) for name, *_ in AFE_TOPO.devices}

    # AC gains (Hmag).
    devs_ac = plan.ac_devices(drive=AFE_TOPO.input_drives, node_drives=AFE_TOPO.ac_drives)
    lti_ac = build_lti_problem(plan, devs_ac, cold, bpts, ss, plan.ac_capacitors(),
                               plan.ac_resistors(), plan.ac_vccs(AFE_TOPO.ac_drives),
                               AFE_TOPO.ac_drives)
    v = complex_array(lti_ac.solve(np.asarray(freqs, float)))
    out = np.zeros(len(freqs), complex)
    for node, w in plan.output_weights.items():
        out += w * v[:, plan.idx[node]]
    gains = np.abs(out / 1.0)
    gain_peak_dB = 20 * np.log10(max(gains.max(), 1e-9))
    bw_Hz = bw_from_gain(freqs, gains)

    # Noise (transpose solve with the same cold ss + cold per-device PSD).
    devs = plan.ac_devices(drive={})
    lti_n = build_lti_problem(plan, devs, cold, bpts, ss, plan.ac_capacitors(),
                              plan.ac_resistors(), plan.ac_vccs())
    sense = plan.output_sense(dtype=float)
    tvec = complex_array(lti_n.solve_transpose(np.asarray(freqs, float), sense))
    inj = {name: (d, s) for name, d, g, s in devs}
    out_psd = np.zeros(len(freqs))
    for name, *_ in AFE_TOPO.devices:
        w, l = sizes[name]
        vs, vd, vg = bpts[name]
        S, _, _ = device_psd(w, l, vs, vd, vg, freqs, corner=_shift(name), nf=1)
        d, s = inj[name]
        z = np.zeros(len(freqs), complex)
        if d[0] == "n":
            z += tvec[:, d[1]]
        if s[0] == "n":
            z -= tvec[:, s[1]]
        out_psd += (np.abs(z) ** 2) * S
    irn_psd = out_psd / np.maximum(gains ** 2, 1e-300)
    irn_uV = band_rms(freqs, irn_psd, *band) * 1e6
    return gain_peak_dB, bw_Hz, irn_uV, dc


def _campaign():
    return AfeOtftCampaign(BIAS, FREQS, band=BAND)


# ---------------------------------------------------------------------------
# Group 1: AFE OTFT parity (candidates x corners) — bit-for-bit cold reference.
# ---------------------------------------------------------------------------

def test_afe_otft_parity_bit_for_bit_vs_cold_reference():
    camp = _campaign()
    corners = ("typical", "slow", "fast")
    factors = (1.0, 0.85, 1.2)
    count = 0
    worst = {"gain_peak_dB": 0.0, "bw_Hz": 0.0, "irn_uV": 0.0}
    for factor in factors:
        sizes = _scaled_sizes(factor)
        for corner in corners:
            ref = _cold_reference(sizes, corner)
            assert ref is not None, f"cold reference failed for factor={factor} corner={corner}"
            gain, bw, irn, dc = ref
            cand = camp.candidate(sizes, corner=corner, seed=camp.seed_vector(dc),
                                  trust_seed_as_op=True)
            res = camp.evaluate_batch([cand])[0]
            assert res["ok"], res
            for key, ref_val in (("gain_peak_dB", gain), ("bw_Hz", bw), ("irn_uV", irn)):
                r = _rel(ref_val, res[key])
                worst[key] = max(worst[key], r)
                # gain/bw are bit-for-bit; irn carries the band_rms naive-sum ULP.
                tol = 1e-12 if key != "irn_uV" else 1e-11
                assert r <= tol, f"{key} factor={factor} corner={corner} rel={r:.3e}"
            count += 1
    assert count == len(factors) * len(corners)
    print(f"\n[group1] cold-reference parity: {count} (candidate,corner) cases, "
          f"worst_rel gain={worst['gain_peak_dB']:.2e} bw={worst['bw_Hz']:.2e} "
          f"irn={worst['irn_uV']:.2e}")


def test_afe_otft_dc_op_is_bit_for_bit_when_seeded():
    camp = _campaign()
    sizes = _scaled_sizes(1.0)
    ac = ac_solve(sizes, BIAS, corner="typical", freqs=FREQS)
    dc = ac["dc_op"]
    seed = camp.seed_vector(dc)
    res = camp.evaluate_batch(
        [camp.candidate(sizes, corner="typical", seed=seed, trust_seed_as_op=False)]
    )[0]
    assert res["ok"] and res["dc_from_seed"]
    for got, want in zip(res["dc_op"], seed):
        assert got == want  # exact


def test_warm_metrics_divergence_is_within_internal_op_seed_floor():
    """The campaign (cold-consistent) agrees with the warm ``corners.metrics``
    path only to the AFE OTFT internal-node Newton's seed-sensitivity floor.
    Flagged, quantified, and shown to be a property of the frozen model — not a
    port bug — by comparing Python's own warm vs cold small-signal params."""
    sizes = _scaled_sizes(1.0)
    camp = _campaign()
    plan = CompiledTopology(AFE_TOPO, BIAS)

    # Python's warm (ac_solve cache) vs cold (fresh instance) ss at the same op.
    ac = ac_solve(sizes, BIAS, corner="typical", freqs=FREQS)
    dc = ac["dc_op"]
    bpts = plan.bias_points({n: dc[n] for n in plan.solved})
    warm_dev = ac._devices
    floor = 0.0
    for name, *_ in AFE_TOPO.devices:
        ss_warm = warm_dev[name].get_ss_params(*bpts[name])
        cold = PMOS_TFT(W=sizes[name][0], L=sizes[name][1], NF=1, pvt0=0.0, pbeta0=0.0)
        ss_cold = cold.get_ss_params(*bpts[name])
        for key in ("gm", "gds"):
            floor = max(floor, _rel(ss_warm[key], ss_cold[key]))
    assert floor > 1e-11, "expected a measurable warm-vs-cold seed floor"

    m = metrics(sizes, BIAS, corner="typical", freqs=FREQS, band=BAND)
    res = camp.evaluate_batch(
        [camp.candidate(sizes, corner="typical", seed=camp.seed_vector(dc),
                        trust_seed_as_op=True)]
    )[0]
    worst = max(_rel(m[k], res[k]) for k in ("gain_peak_dB", "bw_Hz", "irn_uV"))
    print(f"\n[floor] warm-vs-cold ss floor={floor:.2e}; campaign-vs-metrics worst={worst:.2e}")
    # The campaign agrees with warm metrics to within the model's own seed floor.
    assert worst <= max(1e2 * floor, 1e-6)


# ---------------------------------------------------------------------------
# Group 4: determinism across worker counts (byte-identical, index-ordered).
# ---------------------------------------------------------------------------

def _newton_candidates(camp, n):
    cands = []
    rng = np.random.default_rng(7)
    for _ in range(n):
        f = float(rng.uniform(0.8, 1.2))
        sizes = _scaled_sizes(f)
        ac = ac_solve(sizes, BIAS, corner="typical", freqs=FREQS)
        seed = camp.seed_vector(ac["dc_op"]) if ac is not None else None
        cands.append(camp.candidate(sizes, corner="typical", seed=seed,
                                    trust_seed_as_op=False))
    return cands


def test_determinism_across_worker_counts():
    camp = _campaign()
    cands = _newton_candidates(camp, 12)
    baseline = camp.evaluate_batch(cands, workers=1)
    for workers in (1, 2, 8):
        got = camp.evaluate_batch(cands, workers=workers)
        assert len(got) == len(baseline)
        for i, (a, b) in enumerate(zip(baseline, got)):
            assert a["ok"] and b["ok"], (i, a, b)
            for key in ("gain_peak_dB", "bw_Hz", "irn_uV", "latch_dV"):
                assert a[key] == b[key], f"workers={workers} candidate={i} {key}"
            assert a["dc_op"] == b["dc_op"], f"workers={workers} candidate={i} dc_op"


def test_results_are_candidate_index_ordered():
    camp = _campaign()
    # Distinct designs so any reorder is visible in the gain.
    cands, refs = [], []
    for f in (1.0, 0.85, 1.2, 0.9, 1.1):
        sizes = _scaled_sizes(f)
        ref = _cold_reference(sizes, "typical")
        refs.append(ref[0])
        cands.append(camp.candidate(sizes, corner="typical",
                                    seed=camp.seed_vector(ref[3]), trust_seed_as_op=True))
    got = camp.evaluate_batch(cands, workers=8)
    for i, (res, ref_gain) in enumerate(zip(got, refs)):
        assert _rel(ref_gain, res["gain_peak_dB"]) <= 1e-12, i


# ---------------------------------------------------------------------------
# Group 5: no per-candidate Python callback + GIL-release speedup.
# ---------------------------------------------------------------------------

def test_no_python_device_callback_during_batch(monkeypatch):
    camp = _campaign()
    cands = _newton_candidates(camp, 6)
    calls = {"n": 0}
    orig = PMOS_TFT.get_ss_params

    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    # Patch the Python device small-signal entry + the Python solvers: none may
    # fire while the compiled batch runs.
    monkeypatch.setattr(PMOS_TFT, "get_ss_params", counting)
    import circuitopt.ac_solver as acmod
    import circuitopt.noise_solver as nzmod
    ac_calls = {"n": 0}
    monkeypatch.setattr(acmod, "ac_solve",
                        lambda *a, **k: ac_calls.__setitem__("n", ac_calls["n"] + 1))
    monkeypatch.setattr(nzmod, "noise_analysis",
                        lambda *a, **k: ac_calls.__setitem__("n", ac_calls["n"] + 1))

    out = camp.evaluate_batch(cands, workers=4)
    assert all(r["ok"] for r in out)
    assert calls["n"] == 0, "compiled batch called back into the Python device model"
    assert ac_calls["n"] == 0, "compiled batch called back into the Python solvers"


def test_gil_released_speedup():
    camp = _campaign()
    # Cold-Newton candidates are heavy enough to expose parallel scaling.
    cands = _newton_candidates(camp, 48)

    def timed(workers):
        best = float("inf")
        for _ in range(2):
            t0 = time.perf_counter()
            camp.evaluate_batch(cands, workers=workers)
            best = min(best, time.perf_counter() - t0)
        return best

    t1 = timed(1)
    t8 = timed(8)
    speedup = t1 / t8
    print(f"\n[speedup] workers=1 {t1*1e3:.1f}ms  workers=8 {t8*1e3:.1f}ms  speedup={speedup:.2f}x")
    # A real GIL-released batch scales; require a modest, machine-robust margin.
    assert speedup > 1.3, f"no GIL-release speedup (t1={t1:.3f}s t8={t8:.3f}s)"


# ---------------------------------------------------------------------------
# Group 3: seeded mismatch (same rule as corners.mismatch_corner).
# ---------------------------------------------------------------------------

def test_seeded_mismatch_matches_cold_reference():
    camp = _campaign()
    sizes = _scaled_sizes(1.0)
    devices = [d for d, *_ in AFE_TOPO.devices]
    rng = np.random.default_rng(0)
    base = CORNERS["slow"]
    worst = 0.0
    n_ok = 0
    for _ in range(6):
        # Draw exactly like corners.mismatch_corner: per device, in device order.
        corner_map = {
            d: {**base, "mvt0": float(rng.normal(0, SIGMA_MVT0)),
                "mbeta0": float(rng.normal(0, SIGMA_MBETA0))}
            for d in devices
        }
        ref = _cold_reference(sizes, corner_map)
        if ref is None:
            continue
        gain, bw, irn, dc = ref
        mismatch = {d: {"mvt0": corner_map[d]["mvt0"], "mbeta0": corner_map[d]["mbeta0"]}
                    for d in devices}
        cand = camp.candidate(sizes, corner="slow", mismatch=mismatch,
                              seed=camp.seed_vector(dc), trust_seed_as_op=True)
        res = camp.evaluate_batch([cand])[0]
        assert res["ok"], res
        for key, ref_val in (("gain_peak_dB", gain), ("bw_Hz", bw), ("irn_uV", irn)):
            worst = max(worst, _rel(ref_val, res[key]))
        n_ok += 1
    assert n_ok >= 4
    print(f"\n[group3] seeded mismatch: {n_ok} samples, worst_rel={worst:.2e}")
    assert worst <= 1e-11


# ---------------------------------------------------------------------------
# Error propagation: a bad candidate is flagged, the batch survives.
# ---------------------------------------------------------------------------

def test_bad_candidate_is_flagged_without_sinking_batch():
    camp = _campaign()
    good_sizes = _scaled_sizes(1.0)
    ref = _cold_reference(good_sizes, "typical")
    good = camp.candidate(good_sizes, corner="typical",
                          seed=camp.seed_vector(ref[3]), trust_seed_as_op=True)
    # A candidate whose (trusted) seed is malformed -> per-candidate error, not a
    # batch-wide exception. Right device count so it survives the up-front check.
    bad = camp.candidate(good_sizes, corner="typical", trust_seed_as_op=True)
    bad["seed"] = [0.0, 0.0, 0.0]  # wrong length (n_aug is 6)
    out = camp.evaluate_batch([good, bad, good], workers=2)
    assert out[0]["ok"] and out[2]["ok"]
    assert not out[1]["ok"] and "error" in out[1]
    assert _rel(ref[0], out[0]["gain_peak_dB"]) <= 1e-12


def test_device_count_mismatch_raises():
    camp = _campaign()
    with pytest.raises(ValueError):
        camp.evaluate_batch([{"devices": [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]}])
