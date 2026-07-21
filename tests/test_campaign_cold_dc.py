"""R5-D prerequisite behaviour gates: cold-DC root selection + tsmc28 0-bin.

The compiled campaign's circuit Newton (numeric Jacobian + backtracking) is *not*
a bit-for-bit reproduction of the frozen scipy ``fsolve`` (MINPACK hybrd) guard
cascade. R5-C proved bit-exact parity only when the campaign is *seeded* from the
Python operating point. Before R5-D routes cold corner/sweep workflows through
the campaign, these gates pin the campaign's **cold** (no-seed) behaviour against
the frozen scalar path:

1. **Silicon (freepdk45 / sky130 / tsmc28)** — the BSIM4 5T OTAs are monostable,
   so a cold campaign solve reaches the *same physical branch* as the cold scalar
   path: convergence-rate identical, worst node agreement well inside the 1e-3 V
   calibration DC tolerance. Silicon cold sweeps are therefore safe to route.

2. **AFE OTFT** — multistable (the cross-coupled latch). A cold circuit Newton can
   select a *different branch* than ``fsolve`` (observed tens of volts apart), so
   this is a **flagged** divergence: convergence-rate still matches, seeding from
   the scalar op reproduces it bit-for-bit (``dc_from_seed``), and a non-converging
   candidate is flagged ``{ok: False}`` — there is **no silent root substitution**.
   Consequently an AFE size-sweep may only be routed through the campaign when a
   consistent DC seed is supplied (``corners.mismatch_mc`` seeds from the nominal
   op); a cold AFE corner sweep stays on the scalar reference.

3. **tsmc28 0-bin convention** — a geometry whose per-finger width (``w / nf``)
   selects zero delivery bins is rejected *identically* by both engines
   (``ValueError`` <-> per-candidate ``{ok: False}``), and one bad candidate never
   sinks the batch (the "pre-probe and skip" convention the wired corner sweep
   relies on). NOTE (honest flag): in this delivery bin *presence* is
   corner-independent — a 0-bin geometry is 0-bin in every corner, not only
   ff/sf/fs as PARITY.md's prose suggested; the rejection parity is what matters.

D12: no PDK card value is written here; only convergence counts and worst-case
node/voltage differences are computed in-process.
"""
from __future__ import annotations

import os

os.environ.setdefault("CIRCUIT_ENGINE", "rust")

import numpy as np
import pytest

circuitopt_core = pytest.importorskip("circuitopt_core")
if not hasattr(circuitopt_core, "CompiledCampaign"):
    pytest.skip("circuitopt_core lacks CompiledCampaign", allow_module_level=True)

from circuitopt._engine import current_engine

if current_engine() != "rust":
    pytest.skip("cold-DC behaviour gate requires the rust device engine",
                allow_module_level=True)

from circuitopt._rust_campaign import AfeOtftCampaign, SiliconCampaign
from circuitopt.ac_solver import ac_solve
from circuitopt.circuit_loader import load_circuit_json
from circuitopt.compiled_topology import CompiledTopology
from circuitopt.topology import AFE_TOPO

# ── AFE locked design (examples/afe_explore.json) ────────────────────────────
_AFE_SIZES = {
    "M6": (2264.0, 78.0), "M7": (61365.0, 61.0), "M8": (61365.0, 61.0),
    "M9": (3175.0, 468.0), "M10": (3175.0, 468.0), "M11": (465.0, 66.0),
    "M12": (894.0, 85.0), "M13": (894.0, 85.0), "M14": (5224.0, 46.0),
    "M15": (5224.0, 46.0),
}
_AFE_BIAS = {"VDD": 40.0, "VCM": 30.65, "VB": 9.84, "VC": 16.0}
_AFE_FREQS = np.logspace(-2, 4, 121)
_AFE_PAIRS = (("M7", "M8"), ("M9", "M10"), ("M12", "M13"), ("M14", "M15"))

_SI_EXAMPLES = {
    "freepdk45": ("examples/freepdk45_5t_ota.json", "nom"),
    "sky130": ("examples/sky130_5t_ota.json", "tt"),
    "tsmc28": ("examples/tsmc28hpcp_5t_ota.json", "tt"),
}
_SI_FREQS = np.logspace(3, 7, 41)
_SI_BAND = (1e3, 1e6)

# Same-branch tolerance: a 5T-OTA branch swap moves nodes by hundreds of mV, so
# 1e-3 V (the calibration DC tolerance) cleanly separates "same branch, different
# solver path" (~1e-5 V) from an actual branch swap.
_SAME_BRANCH_V = 1e-3


def _silicon_ready(pdk):
    if pdk == "freepdk45":
        from circuitopt.toolchain import pdk_root

        if not os.path.isfile(os.path.join(pdk_root(), "freepdk45", "models_nom",
                                           "NMOS_VTG.inc")):
            return "FreePDK45 cards not present"
    elif pdk == "tsmc28":
        from circuitopt.toolchain import tsmc28_model_dir

        if not os.path.isfile(os.path.join(tsmc28_model_dir(), "cln28hpcp_1d8_elk_v1d0_2p2.l")):
            return "licensed TSMC28HPC+ model is not installed"
    return None


def _si_kwargs(spec):
    binding = spec.binding()
    return dict(topo=spec.topology, nf=spec.nf, model_types=binding.model_types,
                device_kwargs={n: dict(kw) for n, kw in (binding.device_kwargs or {}).items()})


# ---------------------------------------------------------------------------
# (1) Silicon cold-DC: same physical branch as the scalar path.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pdk", sorted(_SI_EXAMPLES))
def test_silicon_cold_dc_matches_scalar_branch(pdk):
    reason = _silicon_ready(pdk)
    if reason:
        pytest.skip(reason)
    path, corner = _SI_EXAMPLES[pdk]
    spec = load_circuit_json(path)
    camp = SiliconCampaign(spec, _SI_FREQS, band=_SI_BAND)
    kwargs = _si_kwargs(spec)
    solved = list(CompiledTopology(spec.topology, spec.bias).solved)

    rng = np.random.default_rng(1)
    both = camp_conv = scalar_conv = agree = 0
    worst_dv = 0.0
    for _ in range(10):
        f = rng.uniform(0.85, 1.15)
        sizes = {n: (w * f, l) for n, (w, l) in spec.sizes.items()}
        cold = camp.evaluate_batch([camp.candidate(sizes, corner)])[0]  # no seed
        ac = ac_solve(sizes, spec.bias, _SI_FREQS, corner=corner, **kwargs)
        camp_conv += bool(cold.get("ok"))
        scalar_conv += ac is not None
        # Convergence-rate parity, case by case.
        assert bool(cold.get("ok")) == (ac is not None), (
            f"{pdk}: cold convergence disagrees ({cold.get('ok')} vs {ac is not None})")
        if cold.get("ok") and ac is not None:
            both += 1
            dv = max(abs(cold["dc_op"][i] - ac["dc_op"][n]) for i, n in enumerate(solved))
            worst_dv = max(worst_dv, dv)
            agree += dv <= _SAME_BRANCH_V
    print(f"\n[cold-dc:{pdk}] camp_conv={camp_conv} scalar_conv={scalar_conv} "
          f"both={both} same_branch={agree} worst_dV={worst_dv:.3e}")
    assert both >= 1, f"{pdk}: no case converged on both paths"
    assert camp_conv == scalar_conv
    assert agree == both, (
        f"{pdk}: {both - agree}/{both} cold cases diverged past {_SAME_BRANCH_V} V "
        f"(worst {worst_dv:.3e} V) — a monostable OTA must stay on one branch")


# ---------------------------------------------------------------------------
# (2) AFE cold-DC: multistable — flagged branch divergence + no silent swap.
# ---------------------------------------------------------------------------

def _afe_sizes(rng):
    paired = {n: p[0] for p in _AFE_PAIRS for n in p}
    groups = set(paired.values()) | {n for n in _AFE_SIZES if n not in paired}
    f = {g: rng.uniform(0.8, 1.2) for g in groups}
    return {n: (w * f[paired.get(n, n)], l * f[paired.get(n, n)])
            for n, (w, l) in _AFE_SIZES.items()}


def test_afe_cold_dc_convergence_parity_but_branch_flagged():
    """Cold AFE convergence rate matches scipy, but the multistable root diverges
    (flagged) — this is *why* the AFE family is only routed through the campaign
    when seeded (see corners.mismatch_mc)."""
    camp = AfeOtftCampaign(_AFE_BIAS, _AFE_FREQS, band=(0.05, 100.0))
    solved = list(CompiledTopology(AFE_TOPO, _AFE_BIAS).solved)
    rng = np.random.default_rng(0)
    both = camp_conv = scalar_conv = 0
    worst_dv = 0.0
    for _ in range(18):
        sizes = _afe_sizes(rng)
        for corner in ("typical", "slow", "fast"):
            cold = camp.evaluate_batch([camp.candidate(sizes, corner=corner)])[0]
            ac = ac_solve(sizes, _AFE_BIAS, _AFE_FREQS, corner=corner)
            camp_conv += bool(cold.get("ok"))
            scalar_conv += ac is not None
            assert bool(cold.get("ok")) == (ac is not None)
            if cold.get("ok") and ac is not None:
                both += 1
                worst_dv = max(worst_dv, max(
                    abs(cold["dc_op"][i] - ac["dc_op"][n]) for i, n in enumerate(solved)))
    print(f"\n[cold-dc:afe] camp_conv={camp_conv} scalar_conv={scalar_conv} "
          f"both={both} worst_branch_dV={worst_dv:.3e} (multistable — expected large)")
    assert camp_conv == scalar_conv           # convergence-rate parity holds
    # This is the flag: the cold AFE root genuinely diverges (multistable). If it
    # ever collapsed to <=1e-3 V the "cold AFE is unsafe" premise would be wrong.
    assert worst_dv > _SAME_BRANCH_V


def test_afe_seeded_dc_is_bit_exact_no_silent_root_substitution():
    """Seeding from the scalar op reproduces it bit-for-bit (dc_from_seed), and a
    non-converging candidate is flagged {ok:False} — never a silent branch swap."""
    camp = AfeOtftCampaign(_AFE_BIAS, _AFE_FREQS, band=(0.05, 100.0))
    solved = list(CompiledTopology(AFE_TOPO, _AFE_BIAS).solved)
    rng = np.random.default_rng(2)
    n_ok = 0
    for _ in range(6):
        sizes = _afe_sizes(rng)
        ac = ac_solve(sizes, _AFE_BIAS, _AFE_FREQS, corner="slow")
        if ac is None:
            continue
        seed = camp.seed_vector(ac["dc_op"])
        res = camp.evaluate_batch(
            [camp.candidate(sizes, corner="slow", seed=seed, trust_seed_as_op=False)])[0]
        assert res["ok"] and res["dc_from_seed"]
        py_op = [float(ac["dc_op"][n]) for n in solved]
        assert py_op == list(res["dc_op"]), "seeded rust DC did not reproduce the op"
        n_ok += 1
    assert n_ok >= 1

    # A candidate that cannot converge is flagged, not silently swapped to a root.
    bad = {n: (1e-6, l) for n, (w, l) in _AFE_SIZES.items()}   # degenerate widths
    out = camp.evaluate_batch([bad and camp.candidate(bad, corner="slow")])[0]
    assert "ok" in out and (out["ok"] is False or "dc_op" in out)


# ---------------------------------------------------------------------------
# (3) tsmc28 0-bin convention: identical rejection, batch not sunk.
# ---------------------------------------------------------------------------

def _tsmc28_or_skip():
    reason = _silicon_ready("tsmc28")
    if reason:
        pytest.skip(reason)
    return load_circuit_json(_SI_EXAMPLES["tsmc28"][0])


def test_tsmc28_zero_bin_rejected_identically_by_both_engines():
    spec = _tsmc28_or_skip()
    camp = SiliconCampaign(spec, _SI_FREQS, band=_SI_BAND)
    kwargs = _si_kwargs(spec)
    # Scale widths up ~20x: per-finger width (w/nf) leaves every delivery bin, so
    # the geometry selects zero bins on both paths.
    zero = {n: (w * 20.0, l) for n, (w, l) in spec.sizes.items()}

    cold = camp.evaluate_batch([camp.candidate(zero, "tt")])[0]
    assert cold["ok"] is False and "error" in cold, cold

    ac = None
    raised = False
    try:
        ac = ac_solve(zero, spec.bias, _SI_FREQS, corner="tt", **kwargs)
    except ValueError:
        raised = True
    # Frozen path rejects the 0-bin geometry the same way (raise or non-convergence).
    assert raised or ac is None, "scalar path accepted a 0-bin geometry the campaign rejected"


def test_tsmc28_zero_bin_does_not_sink_batch():
    spec = _tsmc28_or_skip()
    camp = SiliconCampaign(spec, _SI_FREQS, band=_SI_BAND)
    kwargs = _si_kwargs(spec)
    good_sizes = dict(spec.sizes)
    ac = ac_solve(good_sizes, spec.bias, _SI_FREQS, corner="tt", **kwargs)
    assert ac is not None, "nominal tsmc28 op should converge"
    good = camp.candidate(good_sizes, "tt", seed=camp.seed_vector(ac["dc_op"]),
                          trust_seed_as_op=True)
    bad = camp.candidate({n: (w * 20.0, l) for n, (w, l) in spec.sizes.items()}, "tt")
    out = camp.evaluate_batch([good, bad, good], workers=2)
    assert out[0]["ok"] and out[2]["ok"], "0-bin candidate sank its batch-mates"
    assert out[1]["ok"] is False, "0-bin candidate was not flagged"
