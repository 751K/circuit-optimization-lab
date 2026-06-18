#!/usr/bin/env python3
"""
Interactive AFE Tuner — Flask API Server

Wraps the validated core solvers (DC + AC MNA + noise analysis)
behind a simple REST API so the HTML frontend can query gain, BW, and IRN
for any combination of device sizes and bias voltages.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
from pathlib import Path
import sys
from threading import Lock
from threading import BoundedSemaphore
import warnings

# Suppress known-harmless numpy warnings from validated solvers
# (divide by zero in rout calc when Ich≈0, sqrt of ~0 in noise integration)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def py_type(v):
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return [py_type(x) for x in v.tolist()]
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {k: py_type(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [py_type(x) for x in v]
    return v

from core.ac_solver import ac_solve
from core.noise_solver import noise_analysis, band_rms

app = Flask(__name__, static_folder=str(DEMO_DIR / "static"), static_url_path="/static")

_dc_seed_lock = Lock()
_last_dc_seed = None
_last_dc_bias = None
_preset_seed_cache = {}
SOLVE_TIMEOUT_S = float(os.environ.get("DEMO_SOLVE_TIMEOUT_S", "8.0"))
MAX_SOLVE_WORKERS = int(os.environ.get("DEMO_MAX_SOLVE_WORKERS", "2"))
MAX_NFREQ = int(os.environ.get("DEMO_MAX_NFREQ", "401"))
_solve_executor = ThreadPoolExecutor(max_workers=MAX_SOLVE_WORKERS)
_solve_slots = BoundedSemaphore(MAX_SOLVE_WORKERS)

# ── Frequency grid (shared across all requests) ──────────────────────
FREQ_GRID = np.logspace(-2, 3, 101)  # 0.01 Hz .. 1 kHz, ~Cadence grid
F_LO, F_HI = 0.05, 100.0             # integration bandwidth


# ── Preset designs ───────────────────────────────────────────────────
PRESETS = {
    "base": {
        "label":  "Base",
        "sizes":  {
            "M6": (3000, 150), "M7": (25000, 150), "M8": (25000, 150),
            "M9": (12000, 500), "M10": (12000, 500),
            "M11": (300, 100), "M12": (500, 80), "M13": (500, 80),
            "M14": (2000, 500), "M15": (2000, 500),
        },
        "bias": {"VDD": 40.0, "VCM": 32.0, "VB": 20.0, "VC": 26.0},
    },
    "final": {
        "label":  "Final Locked",
        "sizes":  {
            "M6": (2264, 78), "M7": (61365, 61), "M8": (61365, 61),
            "M9": (3175, 468), "M10": (3175, 468),
            "M11": (465, 66), "M12": (894, 85), "M13": (894, 85),
            "M14": (5224, 46), "M15": (5224, 46),
        },
        "bias": {"VDD": 40.0, "VCM": 30.65, "VB": 9.84, "VC": 16.0},
    },
    "min_area": {
        "label":  "Min Area",
        "sizes":  {
            "M6": (2075, 86), "M7": (57985, 57), "M8": (57985, 57),
            "M9": (2763, 430), "M10": (2763, 430),
            "M11": (484, 59), "M12": (739, 80), "M13": (739, 80),
            "M14": (5120, 46), "M15": (5120, 46),
        },
        "bias": {"VDD": 40.0, "VCM": 30.8, "VB": 9.2, "VC": 16.5},
    },
    "first_feasible": {
        "label":  "First Feasible",
        "sizes":  {
            "M6": (5508, 236), "M7": (83752, 82), "M8": (83752, 82),
            "M9": (13941, 985), "M10": (13941, 985),
            "M11": (993, 256), "M12": (2876, 157), "M13": (2876, 157),
            "M14": (11020, 113), "M15": (11020, 113),
        },
        "bias": {"VDD": 40.0, "VCM": 30.8, "VB": 10.9, "VC": 16.5},
    },
}


def validate_sizes(sizes_dict):
    """Parse and clamp sizes to sane ranges."""
    parsed = {}
    for name in ["M6","M7","M8","M9","M10","M11","M12","M13","M14","M15"]:
        raw = sizes_dict.get(name, [1000, 100])
        w = float(raw[0]) if isinstance(raw, (list, tuple)) else float(raw)
        l = float(raw[1]) if isinstance(raw, (list, tuple)) else 100.0
        # Clamp to DRM + solver stability ranges
        w = max(50, min(200_000, w))
        l = max(10, min(800, l))
        parsed[name] = (w, l)
    return parsed


def validate_bias(bias_dict):
    """Parse and clamp bias voltages."""
    return {
        "VDD": 40.0,  # fixed
        "VCM": max(18.0, min(38.0, float(bias_dict.get("VCM", 30.0)))),
        "VB":  max(1.0,  min(35.0, float(bias_dict.get("VB", 10.0)))),
        "VC":  max(5.0,  min(35.0, float(bias_dict.get("VC", 16.0)))),
    }


_DC_SEED_NODES = ("VOP", "VON", "VFBP", "VFBN", "NET20", "NET2")
_PRESET_SEED_ORDER = ("first_feasible", "final", "min_area", "base")


def _compact_dc_seed(dc):
    return {node: float(dc[node]) for node in _DC_SEED_NODES if node in dc}


def _remap_dc_seed(seed, from_bias, to_bias):
    """Reuse a known AFE branch after bias changes by preserving VCM-relative offsets."""
    if not seed or not from_bias:
        return None
    from_vcm = float(from_bias.get("VCM", 0.0))
    to_vcm = float(to_bias.get("VCM", from_vcm))
    from_vdd = max(float(from_bias.get("VDD", 40.0)), 1e-30)
    to_vdd = float(to_bias.get("VDD", from_vdd))
    scale = to_vdd / from_vdd
    lo, hi = -0.25, to_vdd + 0.25
    out = {}
    for node in _DC_SEED_NODES:
        if node in seed:
            out[node] = float(np.clip(to_vcm + (float(seed[node]) - from_vcm) * scale,
                                      lo, hi))
    return out or None


def _preset_dc_seed(name):
    if name in _preset_seed_cache:
        return _preset_seed_cache[name]
    preset = PRESETS[name]
    ac = ac_solve(preset["sizes"], preset["bias"], np.array([1.0]))
    if ac is None:
        _preset_seed_cache[name] = None
    else:
        _preset_seed_cache[name] = (_compact_dc_seed(ac["dc_op"]), dict(preset["bias"]))
    return _preset_seed_cache[name]


def _demo_dc_attempts(bias):
    """Ordered seed attempts for the interactive demo.

    Cold solve stays first. Fallback seeds are known good physical AFE branches,
    remapped to the current bias, so changing a full size set is less likely to
    land on the wrong branch or show a false DC failure.
    """
    attempts = [("cold", None)]
    with _dc_seed_lock:
        last_seed = dict(_last_dc_seed) if _last_dc_seed is not None else None
        last_bias = dict(_last_dc_bias) if _last_dc_bias is not None else None
    remapped_last = _remap_dc_seed(last_seed, last_bias, bias)
    if remapped_last is not None:
        attempts.append(("warm_rescue", remapped_last))
    for name in _PRESET_SEED_ORDER:
        item = _preset_dc_seed(name)
        if item is None:
            continue
        seed, seed_bias = item
        remapped = _remap_dc_seed(seed, seed_bias, bias)
        if remapped is not None:
            attempts.append((f"preset_{name}", remapped))
    return attempts


def solve_ac_with_retries(sizes, bias, freqs):
    """Use the canonical cold solve first, then demo-safe physical branch seeds."""
    global _last_dc_seed, _last_dc_bias

    for mode, x0 in _demo_dc_attempts(bias):
        ac = ac_solve(sizes, bias, freqs, x0_guess=x0)
        if ac is not None:
            with _dc_seed_lock:
                _last_dc_seed = _compact_dc_seed(ac["dc_op"])
                _last_dc_bias = dict(bias)
            return ac, mode
    return None, "failed"


def solve_payload(data):
    sizes = validate_sizes(data.get("sizes", {}))
    bias = validate_bias(data.get("bias", {}))

    # Optional: allow caller to specify a custom freq grid, bounded so one request
    # cannot accidentally turn the demo into a long batch job.
    nf = int(data.get("nfreq", 121))
    nf = max(2, min(MAX_NFREQ, nf))
    if nf != len(FREQ_GRID):
        freqs = np.logspace(-2, 3, nf)
    else:
        freqs = FREQ_GRID

    # ── AC ────────────────────────────────────────────────
    ac, dc_seed_mode = solve_ac_with_retries(sizes, bias, freqs)

    if ac is None:
        return {"error": "DC Solution did not converge - please adjust sizes or bias",
                "converged": False}

    # ── Noise ─────────────────────────────────────────────
    nr = noise_analysis(sizes, bias, freqs, x0_guess=ac["dc_op"])

    if nr is None:
        return {"error": "Noise analysis failed", "converged": False}

    irn_total = band_rms(freqs, nr["irn_psd"], F_LO, F_HI)  # Vrms
    irn_uV = irn_total * 1e6

    # Per-device noise contributions
    dev_noise = {}
    for name, psd in nr["dev_psd"].items():
        v = band_rms(freqs, psd, F_LO, F_HI)
        dev_noise[name] = round(v * 1e6, 2)  # µVrms

    # Total output noise
    vout_total = band_rms(freqs, nr["out_psd"], F_LO, F_HI)

    # DC op summary
    dc = ac["dc_op"]
    dc_op = {
        "net2": round(dc["net2"], 2),
        "VOP": round(dc["VOP"], 2),
        "VON": round(dc["VOP"], 2),  # symmetric under nominal
        "vfb": round(dc["vfb"], 2),
        "n20": round(dc["n20"], 2),
    }

    # Small-signal summary (gm/gds per device)
    ss_summary = {}
    for name in ["M7", "M9", "M12", "M14", "M6", "M11"]:
        p = ac["ss"].get(name, {})
        ss_summary[name] = {
            "gm_nS": round(p.get("gm", 0) * 1e9, 2),
            "gds_nS": round(p.get("gds", 0) * 1e9, 4),
            "Cgs_pF": round(p.get("Cgs", 0) * 1e12, 3),
            "Cgd_pF": round(p.get("Cgd", 0) * 1e12, 3),
            "Ich_uA": round(p.get("Ich", 0) * 1e6, 4),
        }

    # Subsample frequency response for lighter payload
    keep = np.linspace(0, len(freqs) - 1, min(200, len(freqs))).astype(int)
    keep = sorted(set(keep))

    return py_type({
        "converged": True,
        "gain_dB": round(float(ac["Av_dc_dB"]), 2),
        "peak_dB": round(float(ac["peak_dB"]), 2),
        "bw_Hz": round(float(ac["bw_Hz"]), 2),
        "irn_uV": round(float(irn_uV), 2),
        "vout_uV": round(float(vout_total * 1e6), 1),
        "power_uW": round(float(sum(ss_summary[n]["Ich_uA"] for n in ["M7","M9"]) * bias["VDD"] * 2), 2),
        "dc_op": dc_op,
        "ss": ss_summary,
        "dev_noise": dev_noise,
        "freqs": [round(float(freqs[i]), 4) for i in keep],
        "gains": [round(float(ac["gains"][i]), 4) for i in keep],
        "irn_psd": [float(nr["irn_psd"][i]) for i in keep],
        "out_psd": [float(nr["out_psd"][i]) for i in keep],
        "status": {
            "gain_ok": bool(ac["Av_dc_dB"] >= 20),
            "bw_ok": bool(ac["bw_Hz"] >= 100),
            "irn_ok": bool(irn_uV <= 44.5),
            "all_ok": bool(ac["Av_dc_dB"] >= 20 and ac["bw_Hz"] >= 100 and irn_uV <= 44.5),
            "dc_seed": dc_seed_mode,
            "nfreq": int(len(freqs)),
        },
    })


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/presets")
def get_presets():
    """Return available preset designs (names + labels only, no sizes)."""
    return jsonify({k: {"label": v["label"]} for k, v in PRESETS.items()})


@app.route("/api/solve", methods=["POST"])
def solve():
    """
    Expects JSON: { "sizes": {...}, "bias": {...} }
    Returns: gain_dB, bw_Hz, irn_uV, dc_op, freq_response, noise_breakdown, ...
    """
    data = request.get_json(force=True)
    if not _solve_slots.acquire(blocking=False):
        return jsonify({
            "error": "Solver is busy; please wait for the current request to finish.",
            "converged": False,
            "busy": True,
        }), 429

    future = _solve_executor.submit(solve_payload, data)
    future.add_done_callback(lambda _: _solve_slots.release())
    try:
        return jsonify(future.result(timeout=SOLVE_TIMEOUT_S))
    except TimeoutError:
        return jsonify({
            "error": f"Solve timed out after {SOLVE_TIMEOUT_S:g}s",
            "converged": False,
            "timed_out": True,
        }), 504
    except Exception as exc:
        return jsonify({
            "error": f"Solver error: {exc}",
            "converged": False,
        }), 500


@app.route("/api/preset/<name>")
def load_preset(name):
    """Return a preset's full sizes + bias."""
    p = PRESETS.get(name)
    if p is None:
        return jsonify({"error": f"Unknown preset: {name}"}), 404
    return jsonify({"label": p["label"], "sizes": p["sizes"], "bias": p["bias"]})


if __name__ == "__main__":
    print("AFE Tuner Server starting at http://localhost:5100")
    app.run(host="0.0.0.0", port=5100, debug=True)
