#!/usr/bin/env python3
"""Freeze the engine-parity golden corpus and performance baseline (rewrite phase R0).

The Rust core rewrite (docs/rust_core_rewrite_plan.md, D10) froze a reference
of what the engine computes (originally under the v1.4.0 numba engine; re-frozen
under the rust engine in R6 and under the rust BSIM4 backend in R7 — the corpus
is the permanent reference oracle, docs §4-D4), so every phase diffs against it:

* device-level golden grids   -> tests/golden/engine_parity/devices.npz
  I/G/Q/C (+ scalar noise PSD) over bias grids for the analytic OTFT model and
  the native BSIM4 PDKs (freepdk45 / sky130 / tsmc28hpcp when its licensed
  library resolves).
* circuit-level golden runs   -> tests/golden/engine_parity/circuits/*.json
  ``circuit-opt run`` CLI output (the public serialization) for representative
  examples across analyses, executed in fresh subprocesses.
* performance baseline        -> results/engine_baseline_v140.json
  the five benchmarks/*.py ``--json`` reports plus a timed
  ``python -m circuitopt.calibration --all``.
* manifest                    -> tests/golden/engine_parity/manifest.json
  git commit, environment fingerprint, sha256 of every artifact, skip log.

Usage (from the repository root, project venv)::

    python tools/freeze_engine_golden.py freeze
    python tools/freeze_engine_golden.py verify   # regenerate & compare bit-exact
    python tools/freeze_engine_golden.py freeze --skip-bench   # goldens only

``verify`` recomputes the device grids in-process and re-runs the circuit CLI
cases, then requires bit-exact equality against the stored artifacts (same
machine, same environment). Benchmarks are timing, not goldens, and are never
compared. Exit code 0 = pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden" / "engine_parity"
CIRCUIT_DIR_NAME = "circuits"
BASELINE_PATH = ROOT / "results" / "engine_baseline_v140.json"

# ---------------------------------------------------------------- device grids

# Silicon grids: per PDK -> (vdd, {polarity: [((W_um, L_um, NF), corners)]}).
# SKY130 bundles exact-geometry cards, so its (geometry, corner) pairs must
# match files under circuitopt/pdk/sky130/cards/ (pfet has no ff card at all).
_ALL = ("nom", "ss", "ff")
_SILICON = {
    "freepdk45": (1.1, {
        "nmos": [((0.09, 0.05, 1), _ALL), ((1.0, 0.1, 2), _ALL)],
        "pmos": [((0.09, 0.05, 1), _ALL), ((1.0, 0.1, 2), _ALL)],
    }),
    "sky130": (1.8, {
        "nmos": [((1.0, 0.5, 1), _ALL), ((18.7239, 0.5, 1), _ALL)],
        "pmos": [((12.0, 0.5, 1), ("nom", "ss"))],
    }),
    "tsmc28hpcp": (0.9, {
        "nmos": [((0.3, 0.03, 1), _ALL), ((2.0, 0.1, 2), _ALL)],
        "pmos": [((0.3, 0.03, 1), _ALL), ((2.0, 0.1, 2), _ALL)],
    }),
}
_TEMPS_K = (233.15, 300.15, 398.15)
_NOISE_FREQS = (1e3, 1e6, 1e9)


def _silicon_grid(vdd: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(Vg points, Vd points, Vs) sweep for a silicon device at this rail."""
    vg = np.linspace(0.0, vdd, 6)
    vd = np.linspace(0.0, vdd, 6)
    return vg, vd, 0.0


def _device_case_arrays(dev, vg_pts, vd_pts, vs) -> dict[str, np.ndarray]:
    """Evaluate one BSIM device over the grid; returns named arrays."""
    n_g, n_d = len(vg_pts), len(vd_pts)
    I = np.empty((n_g, n_d, 4))
    Q = np.empty((n_g, n_d, 4))
    G = np.empty((n_g, n_d, 4, 4))
    C = np.empty((n_g, n_d, 4, 4))
    for i, vg in enumerate(vg_pts):
        for j, vd in enumerate(vd_pts):
            I[i, j] = dev.get_terminal_currents(vs, float(vd), float(vg))
            Q[i, j] = dev.get_terminal_charges(vs, float(vd), float(vg))
            g4, c4 = dev.get_terminal_linearization(vs, float(vd), float(vg))
            G[i, j] = g4
            C[i, j] = c4
    return {"I": I, "Q": Q, "G": G, "C": C}


def _freeze_devices(log: list[str]) -> dict[str, np.ndarray]:
    from circuitopt.device_model import create_device

    arrays: dict[str, np.ndarray] = {}

    # --- OTFT (default PDK, analytic model) ---------------------------------
    span = np.linspace(-8.0, 2.0, 11)
    for (w, l) in ((1000.0, 20.0), (500.0, 10.0)):
        dev = create_device("at4000tg.pmos", W=w, L=l)
        n = len(span)
        idc = np.empty((n, n))
        caps = np.empty((n, n, 2))
        ss = np.empty((n, n, 5))
        noise = np.empty((n, n, 2))
        for i, vg in enumerate(span):
            for j, vd in enumerate(span):
                idc[i, j] = dev.get_Idc(0.0, float(vd), float(vg))
                caps[i, j] = dev.get_capacitances(0.0, float(vd), float(vg))
                p = dev.get_ss_params(0.0, float(vd), float(vg))
                ss[i, j] = (p["gm"], p["gds"], p["Cgs"], p["Cgd"], p["Ich"])
                noise[i, j] = dev.get_noise_psd(0.0, float(vd), float(vg), 1e3)
        key = f"otft|pmos|W{w:g}L{l:g}"
        arrays[f"{key}|Vgrid"] = span
        arrays[f"{key}|Idc"] = idc
        arrays[f"{key}|caps"] = caps
        arrays[f"{key}|ss"] = ss
        arrays[f"{key}|noise1k"] = noise
        log.append(f"otft W{w:g} L{l:g}: ok")

    # --- silicon BSIM4 PDKs --------------------------------------------------
    for pdk, (vdd, polmap) in _SILICON.items():
        vg_pts, vd_pts, vs = _silicon_grid(vdd)
        for pol, entries in polmap.items():
            model = f"{pdk}.{pol}"
            for (w, l, nf), corners in entries:
                for corner in corners:
                    for temp in _TEMPS_K:
                        tag = (f"{pdk}|{pol}|W{w:g}L{l:g}NF{nf}|{corner}"
                               f"|T{temp:g}")
                        try:
                            dev = create_device(
                                model, W=w, L=l, NF=nf,
                                corner=corner, temperature=temp)
                            case = _device_case_arrays(dev, vg_pts, vd_pts, vs)
                        except Exception as exc:  # licensed PDK may be absent
                            log.append(f"SKIP {tag}: {type(exc).__name__}: {exc}")
                            continue
                        for name, arr in case.items():
                            arrays[f"{tag}|{name}"] = arr
                        log.append(f"{tag}: ok")
                # scalar noise PSD at typical point, nominal corner/temp
                try:
                    (w0, l0, nf0), _ = entries[0]
                    dev = create_device(model, W=w0, L=l0, NF=nf0)
                    pts = np.array([
                        dev.get_noise_psd(vs, vdd / 2.0, vdd / 2.0, f)
                        for f in _NOISE_FREQS
                    ])
                    arrays[f"{pdk}|{pol}|noise_typ"] = pts
                except Exception as exc:
                    log.append(f"SKIP {pdk}|{pol}|noise_typ: "
                               f"{type(exc).__name__}: {exc}")
        arrays[f"{pdk}|Vg"] = vg_pts
        arrays[f"{pdk}|Vd"] = vd_pts
    return arrays


# ------------------------------------------------------------- circuit goldens

# (example json, analyses, inject) — runs `circuit-opt run <json> -a <analyses>`.
# ``run`` only executes analyses configured in the JSON, so examples without an
# ``analyses`` block get a deterministic default injected into a temp copy.
# sc_lpf pss/pac/pnoise is excluded: the CLI JSON serializer currently crashes
# on tuple keys in PSS results (pre-existing defect, tracked separately); its
# periodic coverage lives in the calibration byte-gate instead.
_AFE_INJECT = {
    "ac": {"freqs": {"start": 1.0, "stop": 1e6, "num": 31, "scale": "log"}},
    "noise": {"freqs": {"start": 1.0, "stop": 1e6, "num": 31, "scale": "log"},
              "band": [10.0, 1e5]},
}
_OTA_INJECT = {
    "ac": {"freqs": {"start": 1e2, "stop": 1e10, "num": 31, "scale": "log"}},
    "noise": {"freqs": {"start": 1e2, "stop": 1e10, "num": 31, "scale": "log"},
              "band": [1e3, 1e8]},
}
_CIRCUIT_CASES = [
    ("afe_explore.json", "ac,noise", _AFE_INJECT),
    ("freepdk45_5t_ota.json", "ac,noise,transient", None),
    ("sky130_5t_ota.json", "ac,noise", _OTA_INJECT),
    ("tsmc28hpcp_5t_ota.json", "ac,noise", None),
    ("periodic_rc.json", "ac,noise,pss,pac,pnoise", None),
]


def _case_slug(example: str, analyses: str) -> str:
    return f"{Path(example).stem}--{analyses.replace(',', '-')}.json"


def _case_input(example: str, inject: dict | None, workdir: Path) -> Path:
    """The JSON the CLI runs: the example itself, or a temp copy with an
    injected ``analyses`` block (merged over any existing one)."""
    src = ROOT / "examples" / example
    if inject is None:
        return src
    data = json.loads(src.read_text(encoding="utf-8"))
    merged = dict(data.get("analyses") or {})
    merged.update(inject)
    data["analyses"] = merged
    dst = workdir / example
    dst.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    return dst


def _run_circuit_case(example: str, analyses: str, out_path: Path,
                      log: list[str], inject: dict | None = None) -> bool:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        circuit_path = _case_input(example, inject, Path(td))
        return _run_circuit_cli(example, analyses, circuit_path, out_path, log)


def _run_circuit_cli(example: str, analyses: str, circuit_path: Path,
                     out_path: Path, log: list[str]) -> bool:
    cmd = [sys.executable, "-m", "circuitopt", "run", str(circuit_path),
           "-a", analyses, "-o", str(out_path), "--quiet"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=3600)
    dt = time.perf_counter() - t0
    if proc.returncode != 0 or not out_path.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        log.append(f"SKIP circuit {example} [{analyses}]: rc={proc.returncode} "
                   f"{' | '.join(tail)}")
        return False
    log.append(f"circuit {example} [{analyses}]: ok ({dt:.1f}s)")
    return True


# ---------------------------------------------------------------- perf baseline

_BENCHES = ("bench_model", "bench_afe", "bench_sweep", "bench_chopper",
            "bench_periodic")


def _parse_trailing_json(text: str):
    """Benchmarks print a JSON object; tolerate leading log lines."""
    idx = text.find("{")
    while idx != -1:
        try:
            return json.loads(text[idx:])
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
    raise ValueError("no JSON object found in benchmark output")


def _run_baseline(log: list[str]) -> dict:
    baseline: dict = {"benchmarks": {}, "calibration": {}}
    for bench in _BENCHES:
        cmd = [sys.executable, "-m", f"benchmarks.{bench}", "--json"]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              timeout=3600)
        wall = time.perf_counter() - t0
        entry: dict = {"wall_s": round(wall, 3), "returncode": proc.returncode}
        if proc.returncode == 0:
            try:
                entry["report"] = _parse_trailing_json(proc.stdout)
            except ValueError as exc:
                entry["error"] = str(exc)
        else:
            entry["stderr_tail"] = (proc.stderr or "").strip().splitlines()[-3:]
        baseline["benchmarks"][bench] = entry
        log.append(f"bench {bench}: rc={proc.returncode} wall={wall:.1f}s")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "circuitopt.calibration", "--all"],
        cwd=ROOT, capture_output=True, text=True, timeout=3600)
    baseline["calibration"] = {
        "wall_s": round(time.perf_counter() - t0, 3),
        "returncode": proc.returncode,
    }
    log.append(f"calibration --all: rc={proc.returncode} "
               f"wall={baseline['calibration']['wall_s']}s")
    return baseline


# -------------------------------------------------------------------- plumbing

def _env_fingerprint() -> dict:
    import numpy
    import scipy
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    return {
        "commit": commit,
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _save_devices_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def freeze(skip_bench: bool) -> int:
    log: list[str] = []
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    circuits_dir = GOLDEN_DIR / CIRCUIT_DIR_NAME
    circuits_dir.mkdir(parents=True, exist_ok=True)

    print("== device grids ==", flush=True)
    arrays = _freeze_devices(log)
    devices_npz = GOLDEN_DIR / "devices.npz"
    _save_devices_npz(devices_npz, arrays)
    print(f"   {len(arrays)} arrays -> {devices_npz}", flush=True)

    print("== circuit cases ==", flush=True)
    frozen_cases = []
    for example, analyses, inject in _CIRCUIT_CASES:
        out = circuits_dir / _case_slug(example, analyses)
        if _run_circuit_case(example, analyses, out, log, inject):
            frozen_cases.append([example, analyses, inject])
        print(f"   {log[-1]}", flush=True)

    baseline = None
    if not skip_bench:
        print("== performance baseline ==", flush=True)
        baseline = {"env": _env_fingerprint(), **_run_baseline(log)}
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True)
                                 + "\n", encoding="utf-8")
        print(f"   -> {BASELINE_PATH}", flush=True)

    files = {str(devices_npz.relative_to(ROOT)): _sha256(devices_npz)}
    for case_file in sorted(circuits_dir.glob("*.json")):
        files[str(case_file.relative_to(ROOT))] = _sha256(case_file)
    manifest = {
        "env": _env_fingerprint(),
        "frozen_circuit_cases": frozen_cases,
        "files": files,
        "log": log,
    }
    (GOLDEN_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== manifest -> {GOLDEN_DIR / 'manifest.json'}", flush=True)
    skips = [line for line in log if line.startswith("SKIP")]
    if skips:
        print(f"NOTE: {len(skips)} skipped entries (recorded in manifest):")
        for line in skips:
            print(f"   {line}")
    return 0


_REPR_RE = re.compile(r"<[\w\.]+ object at 0x[0-9a-fA-F]+>")


def _canonical(obj):
    """Mask serialization artifacts that legitimately differ between runs.

    The CLI serializer currently leaks ``repr()`` of internal objects (e.g.
    ``pss.topology`` -> ``<circuitopt.topology.Topology object at 0x...>``)
    whose memory addresses change per process. Every numeric field is
    bit-reproducible (measured 2026-07-17); only these repr strings are
    masked so goldens stay a strict bit-exact gate for actual data.
    """
    if isinstance(obj, str):
        return _REPR_RE.sub("<obj>", obj)
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_canonical(v) for v in obj]
    return obj


def _json_equal(a, b) -> bool:
    return (json.dumps(_canonical(a), sort_keys=True)
            == json.dumps(_canonical(b), sort_keys=True))


def verify() -> int:
    manifest_path = GOLDEN_DIR / "manifest.json"
    if not manifest_path.exists():
        print("FAIL: no manifest — run `freeze` first")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    print("== verify device grids (recompute, bit-exact) ==", flush=True)
    log: list[str] = []
    fresh = _freeze_devices(log)
    stored = np.load(GOLDEN_DIR / "devices.npz")
    stored_keys = set(stored.files)
    fresh_keys = set(fresh)
    if stored_keys != fresh_keys:
        failures.append(
            f"device key sets differ: only-stored={sorted(stored_keys - fresh_keys)[:5]} "
            f"only-fresh={sorted(fresh_keys - stored_keys)[:5]}")
    for key in sorted(stored_keys & fresh_keys):
        if not np.array_equal(stored[key], fresh[key]):
            failures.append(f"device grid not bit-exact: {key}")
    print(f"   {len(stored_keys & fresh_keys)} arrays compared", flush=True)

    print("== verify circuit cases (re-run CLI, bit-exact) ==", flush=True)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for example, analyses, inject in manifest["frozen_circuit_cases"]:
            slug = _case_slug(example, analyses)
            out = Path(tmp) / slug
            relog: list[str] = []
            if not _run_circuit_case(example, analyses, out, relog, inject):
                failures.append(f"circuit case no longer runs: {slug} "
                                f"({relog[-1]})")
                continue
            golden = json.loads(
                (GOLDEN_DIR / CIRCUIT_DIR_NAME / slug).read_text("utf-8"))
            fresh_case = json.loads(out.read_text("utf-8"))
            if not _json_equal(golden, fresh_case):
                failures.append(f"circuit output drifted: {slug}")
            print(f"   {slug}: {'ok' if _json_equal(golden, fresh_case) else 'DRIFT'}",
                  flush=True)

    if failures:
        print("\nVERIFY FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nVERIFY PASS: goldens are reproducible bit-exactly")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fz = sub.add_parser("freeze", help="write goldens + baseline")
    fz.add_argument("--skip-bench", action="store_true",
                    help="skip the performance baseline (goldens only)")
    sub.add_parser("verify", help="regenerate and require bit-exact equality")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        return freeze(skip_bench=args.skip_bench)
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
