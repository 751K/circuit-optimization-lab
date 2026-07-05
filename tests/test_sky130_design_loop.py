"""Silicon design closed loop: SKY130 5T OTA through dataset → surrogate → optimize.

End-to-end proof that a silicon (SKY130) circuit runs through the *whole* ML design
pipeline via the config ``models`` block: the dataset builder labels it, the surrogate
learns its operating-region metrics, and the optimizer screens a large candidate pool
and verifies the shortlist on the Cadence-calibrated solver — including a cross-corner
(tt vs ss) check. Needs the SKY130 PDK + OpenVAF + ngspice (external drive), so it skips
cleanly in CI. See the ``silicon-pdk-openvaf`` memory.
"""
import json
import os

import numpy as np
import pytest

from core.ac_solver import ac_solve
from core.circuit_loader import load_circuit_json, models_from_config
from core.dataset import run_from_config, to_arrays
from core.device_factory import apply_silicon_corner

_PDK_ROOT = os.environ.get("PDK_ROOT", "/Volumes/MacoutDsik/pdk")
_HAVE = os.path.exists(os.path.join(_PDK_ROOT, "sky130A/libs.tech/ngspice/sky130.lib.spice")) \
    and os.path.exists(os.path.join(
        os.environ.get("OPENVAF_ROOT", "/Volumes/MacoutDsik/Code/VAF/OpenVAF-Reloaded"),
        ".claude/skills/build-openvaf/scripts/vacompile.sh"))
pytestmark = pytest.mark.skipif(not _HAVE, reason="SKY130 PDK / OpenVAF toolchain not present")

CONFIG = "examples/sky130_5t_ota.json"


def _config_dict():
    with open(CONFIG, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def dataset():
    # small labeled silicon dataset built through the ordinary dataset CLI path
    return run_from_config(CONFIG, n=64, seed=0)


def test_dataset_is_silicon(dataset):
    """The dataset routes a mixed nmos+pmos SKY130 circuit and records the binding."""
    m = dataset["manifest"]
    assert m["models"] == {"M1": "sky130.nmos", "M2": "sky130.nmos", "M5": "sky130.nmos",
                           "M3": "sky130.pmos", "M4": "sky130.pmos"}
    # the 5T OTA DC-converges robustly across the whole size box (no dropped rows)
    assert m["counts"]["dc_converged"] == 64
    assert m["counts"]["metrics_finite"] == 64
    _, Y, _, ln, _, _ = to_arrays(dataset)
    gain = Y[:, ln.index("gain_dB")]
    assert int(np.sum((gain > 20) & (gain < 60))) >= 20     # plenty of working OTAs


def test_surrogate_learns_operating_region(dataset):
    """Trained on the operating region, the surrogate predicts silicon gain tightly."""
    sg = pytest.importorskip("core.surrogate")
    X, Y, vn, ln, _, _ = to_arrays(dataset)
    Xf, Yf = sg.filter_rows(X, Y, ln, {"gain_dB": (20.0, 60.0)})   # drop railed corners
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xf))
    cut = int(0.75 * len(Xf))
    model = sg.train(Xf[idx[:cut]], Yf[idx[:cut]], vn, ln)
    sc = sg.score(Yf[idx[cut:]], model.predict(Xf[idx[cut:]]), ln)
    assert sc["gain_dB"]["median_rel_pct"] < 5.0            # gain within a few % of the solver
    assert np.all(np.isfinite(model.predict(Xf[idx[cut:]])))


def test_optimize_screens_and_verifies(dataset, tmp_path):
    """The optimizer screens a big pool with the surrogate and the solver-verify pass
    confirms real feasible silicon OTA designs (the screen-and-verify payoff)."""
    sg = pytest.importorskip("core.surrogate")
    from core import optimize as opt
    X, Y, vn, ln, _, _ = to_arrays(dataset)
    model = sg.train(X, Y, vn, ln)                          # full model: knows the railed region
    pkl = tmp_path / "ota.pkl"
    sg.save(model, str(pkl))
    rep = opt.optimize(CONFIG, str(pkl), n_screen=5000, top_k=6, freqs=np.logspace(1, 8, 71))
    assert rep["n_screen"] == 5000 and rep["screen_seconds"] < 5.0
    feasible = [e for e in rep["verified"] if e["solver_feasible"]]
    assert feasible                                        # ≥1 solver-confirmed feasible design
    for e in feasible:                                    # and they are genuine OTAs
        assert e["solver"]["gain_dB"] >= 25.0
        assert e["solver"]["power_uW"] > 0.0


def test_corner_routing_shifts_silicon():
    """A SKY130 corner routes onto the silicon devices (not the OTFT PVT path) and the
    slow corner physically lowers the tail current and bandwidth."""
    spec = load_circuit_json(CONFIG)
    mt, dk = models_from_config(_config_dict())
    sizes = {n: tuple(v) for n, v in spec.sizes.items()}
    out = {}
    for corner in ("tt", "ss"):
        dk_c, solver_corner = apply_silicon_corner(mt, dk, corner)
        assert solver_corner is None                       # silicon corner ≠ OTFT solver shift
        r = ac_solve(sizes, spec.bias, np.logspace(1, 8, 60), topo=spec.topology,
                     model_types=mt, device_kwargs=dk_c)
        out[corner] = (r["bw_Hz"], abs(r["ss"]["M5"]["Ich"]))
    assert out["ss"][1] < out["tt"][1]                     # slow corner: less tail current
    assert out["ss"][0] < out["tt"][0]                     # → lower bandwidth
