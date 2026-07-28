"""Tests for the background-job layer of the FastAPI service (``circuitopt.service``).

Covers the S2 long-task model: explore / mismatch-MC jobs, their lifecycle
(submit -> poll -> done), WebSocket progress streaming, cooperative cancellation,
the 404 / 409 error cases, and the core ``mismatch_mc`` progress/should_stop
hooks (including byte-equivalence of the default-``None`` path).

Gated on the optional ``fastapi`` dependency (the ``serve`` extra), mirroring
``test_service.py``. Jobs run on real worker threads via ``TestClient`` (which
drives the app in-process), so lifecycle waits poll with a bounded timeout.
Small ``n`` values keep the whole module well under 30s.
"""
import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from circuitopt.circuit_loader import load_circuit_json  # noqa: E402
from circuitopt.corners import mismatch_mc  # noqa: E402
from circuitopt.service.app import create_app  # noqa: E402

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name):
    return json.loads((_EXAMPLES / name).read_text())


@pytest.fixture(scope="module")
def client():
    # Context-manager form runs FastAPI startup/shutdown, so the JobManager pool
    # is cleanly torn down after the module.
    with TestClient(create_app()) as c:
        yield c


def _wait_terminal(client, job_id, timeout=25.0):
    """Poll GET /jobs/{id} until the job reaches a terminal state, or time out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/v1/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not terminate within {timeout}s")


# ── capabilities advertises the job kinds ─────────────────────────────────────

def test_capabilities_lists_jobs(client):
    cap = client.get("/api/v1/capabilities").json()
    assert cap["jobs"] == ["explore", "mc", "pvt"]


# ── explore job lifecycle ─────────────────────────────────────────────────────

def test_explore_job_lifecycle(client):
    r = client.post("/api/v1/jobs/explore",
                    json={"circuit": _load("afe_explore.json"), "n": 6, "seed": 0})
    assert r.status_code == 202
    sub = r.json()
    assert sub["kind"] == "explore"
    job_id = sub["job_id"]

    body = _wait_terminal(client, job_id)
    assert body["status"] == "done"
    res = body["result"]
    # explore output structure
    assert "candidates" in res and "summary" in res
    assert res["summary"]["n"] == 6
    assert len(res["candidates"]) == 6
    assert "objectives" in res
    # whole payload is strict JSON (no NaN/Infinity tokens leaked through)
    dumped = json.dumps(body)
    assert "NaN" not in dumped and "Infinity" not in dumped


# ── mc job lifecycle ──────────────────────────────────────────────────────────

def test_mc_job_lifecycle(client):
    r = client.post("/api/v1/jobs/mc",
                    json={"circuit": _load("afe_explore.json"), "n": 4, "seed": 1,
                          "corner": "typical", "workers": 2})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    body = _wait_terminal(client, job_id)
    assert body["status"] == "done"
    res = body["result"]
    assert "arrays" in res and "summary" in res and "latched" in res
    assert res["summary"]["n"] <= 4


# ── job listing ───────────────────────────────────────────────────────────────

def test_list_jobs(client):
    # submit one so the list is non-empty
    r = client.post("/api/v1/jobs/mc", json={"circuit": _load("afe_explore.json"), "n": 2})
    job_id = r.json()["job_id"]
    _wait_terminal(client, job_id)

    lst = client.get("/api/v1/jobs").json()["jobs"]
    assert any(j["job_id"] == job_id for j in lst)
    entry = next(j for j in lst if j["job_id"] == job_id)
    for key in ("job_id", "kind", "status", "created"):
        assert key in entry
    # listing carries no bulky result payload
    assert "result" not in entry


# ── cancellation ──────────────────────────────────────────────────────────────

def test_cancel_running_job(client):
    # A biggish explore so it is still running when we cancel it.
    r = client.post("/api/v1/jobs/explore",
                    json={"circuit": _load("afe_explore.json"), "n": 400, "seed": 0})
    job_id = r.json()["job_id"]

    d = client.delete(f"/api/v1/jobs/{job_id}")
    assert d.status_code == 200
    assert d.json()["status"] == "cancelling"

    # The worker thread must actually stop within a bounded time.
    body = _wait_terminal(client, job_id, timeout=25.0)
    assert body["status"] == "cancelled"
    # Partial result is preserved and flagged as stopped early (fewer than n ran).
    if body.get("result") is not None:
        assert body["result"]["summary"].get("stopped_early") is True
        assert body["result"]["summary"]["evaluated"] < 400


def test_cancel_unknown_job_404(client):
    d = client.delete("/api/v1/jobs/deadbeef0000")
    assert d.status_code == 404
    assert d.json()["detail"]["stage"] == "job"


def test_cancel_terminal_job_409(client):
    r = client.post("/api/v1/jobs/mc", json={"circuit": _load("afe_explore.json"), "n": 2})
    job_id = r.json()["job_id"]
    _wait_terminal(client, job_id)
    d = client.delete(f"/api/v1/jobs/{job_id}")
    assert d.status_code == 409
    assert "terminal" in d.json()["detail"]["message"]


# ── unknown job -> 404 ────────────────────────────────────────────────────────

def test_get_unknown_job_404(client):
    r = client.get("/api/v1/jobs/nope00000000")
    assert r.status_code == 404
    assert r.json()["detail"]["stage"] == "job"


# ── WebSocket progress + terminal ─────────────────────────────────────────────

def test_ws_progress_and_terminal(client):
    r = client.post("/api/v1/jobs/explore",
                    json={"circuit": _load("afe_explore.json"), "n": 6, "seed": 0})
    job_id = r.json()["job_id"]

    progress_events = 0
    terminal = None
    with client.websocket_connect(f"/api/v1/jobs/{job_id}/events") as ws:
        while True:
            event = ws.receive_json()
            if event.get("type") == "progress":
                progress_events += 1
                assert 0 <= event["frac"] <= 1
            elif event.get("type") == "terminal":
                terminal = event
                break
    assert progress_events >= 1
    assert terminal is not None
    assert terminal["status"] in ("done", "cancelled")


def test_ws_unknown_job_error_frame(client):
    with client.websocket_connect("/api/v1/jobs/nope00000000/events") as ws:
        event = ws.receive_json()
        assert event["type"] == "error"


# ── core mismatch_mc hook: progress + should_stop ─────────────────────────────

def test_mismatch_mc_progress_called_each_sample():
    spec = load_circuit_json(str(_EXAMPLES / "afe_explore.json"))
    freqs = np.logspace(-2, 2, 41)
    calls = []

    def progress(i, n, partial):
        calls.append((i, n))
        assert isinstance(partial, dict) and "n" in partial

    mismatch_mc(spec.sizes, spec.bias, nf=spec.nf, topo=spec.topology,
                base="typical", n=5, seed=1, freqs=freqs, progress=progress)
    # one callback per requested sample, 1-based, total == n
    assert [c[0] for c in calls] == [1, 2, 3, 4, 5]
    assert all(c[1] == 5 for c in calls)


def test_mismatch_mc_should_stop_early():
    spec = load_circuit_json(str(_EXAMPLES / "afe_explore.json"))
    freqs = np.logspace(-2, 2, 41)
    state = {"i": 0}

    def stop():
        state["i"] += 1
        return state["i"] > 3   # allow 3 samples, stop before the 4th

    out = mismatch_mc(spec.sizes, spec.bias, nf=spec.nf, topo=spec.topology,
                      base="typical", n=10, seed=1, freqs=freqs, should_stop=stop)
    assert out["stopped_early"] is True
    assert out["summary"]["stopped_early"] is True
    assert out["summary"]["n"] == 3


def test_mismatch_mc_default_path_bit_identical():
    # progress=None, should_stop=None must reproduce the pre-hook result exactly.
    spec = load_circuit_json(str(_EXAMPLES / "afe_explore.json"))
    freqs = np.logspace(-2, 2, 41)
    a = mismatch_mc(spec.sizes, spec.bias, nf=spec.nf, topo=spec.topology,
                    base="typical", n=6, seed=7, freqs=freqs)
    b = mismatch_mc(spec.sizes, spec.bias, nf=spec.nf, topo=spec.topology,
                    base="typical", n=6, seed=7, freqs=freqs,
                    progress=lambda i, n, p: None, should_stop=lambda: False)
    for k in a["arrays"]:
        assert np.array_equal(a["arrays"][k], b["arrays"][k], equal_nan=True)
    assert a["summary"] == b["summary"]
    assert "stopped_early" not in a and "stopped_early" not in a["summary"]


# ── pvt job ───────────────────────────────────────────────────────────────────

def test_pvt_job_sweeps_the_circuits_own_corner_family(client):
    # An AFE circuit's corners are the OTFT process names. Asking for a corner
    # sweep without naming corners must produce those, not a silicon card set —
    # the two name spaces are disjoint, so the wrong default cannot resolve.
    r = client.post("/api/v1/jobs/pvt", json={"circuit": _load("afe_explore.json")})
    assert r.status_code == 202
    assert r.json()["kind"] == "pvt"

    body = _wait_terminal(client, r.json()["job_id"])
    assert body["status"] == "done", body.get("error")
    res = body["result"]
    assert res["corners"] == ["typical", "slow", "fast"]
    assert set(res["table"]) == {"typical", "slow", "fast"}
    assert res["silicon"] is False
    # A sweep with no PVT axis is one slice, and every corner carries metrics.
    assert res["slices"] == 1
    for name, m in res["table"].items():
        assert m is not None and "gain_peak_dB" in m, name
    assert "NaN" not in json.dumps(body)


def test_pvt_job_nests_the_temperature_axis(client):
    circuit = _load("sky130_fd_ota.json")
    r = client.post("/api/v1/jobs/pvt",
                    json={"circuit": circuit, "temps": [0.0, 85.0], "workers": 2})
    assert r.status_code == 202

    body = _wait_terminal(client, r.json()["job_id"], timeout=90.0)
    assert body["status"] == "done", body.get("error")
    res = body["result"]
    assert res["silicon"] is True
    assert res["corners"] == ["tt", "ss"]          # sky130 bundled-card boundary
    assert res["temps"] == [0.0, 85.0]
    assert res["slices"] == 2
    # Each corner nests one entry per temperature, keyed by the temperature.
    for name, node in res["table"].items():
        assert sorted(float(k) for k in node) == [0.0, 85.0], name

    # A silicon amplifier loses gain as it heats up. This is the axis actually
    # doing something: a grid that ignored `temps` would report equal values.
    hot = res["table"]["tt"]["85.0"]
    cold = res["table"]["tt"]["0.0"]
    assert hot is not None and cold is not None
    assert hot["gain_peak_dB"] < cold["gain_peak_dB"]


def test_pvt_job_reports_progress_per_slice(client):
    circuit = _load("sky130_fd_ota.json")
    job_id = client.post(
        "/api/v1/jobs/pvt",
        json={"circuit": circuit, "temps": [0.0, 27.0, 85.0]},
    ).json()["job_id"]

    with client.websocket_connect(f"/api/v1/jobs/{job_id}/events") as ws:
        seen = []
        while True:
            event = ws.receive_json()
            if event["type"] == "progress":
                seen.append((event["done"], event["total"]))
            if event["type"] == "terminal":
                assert event["status"] == "done"
                break
    # The 0/3 opening frame plus one per completed slice, all against a total of 3.
    assert seen[0] == (0, 3)
    assert sorted(d for d, _ in seen) == [0, 1, 2, 3]
    assert all(total == 3 for _, total in seen)


def test_pvt_axes_are_rejected_before_the_job_is_queued(client):
    # An OTFT circuit has no temperature axis. Rejecting at submit (422) rather
    # than letting a job fail a second later keeps the failure attached to the
    # request that caused it.
    before = len(client.get("/api/v1/jobs").json()["jobs"])
    r = client.post("/api/v1/jobs/pvt",
                    json={"circuit": _load("afe_explore.json"), "temps": [0.0, 85.0]})
    assert r.status_code == 422
    assert "all-silicon" in r.json()["detail"]["message"]
    assert len(client.get("/api/v1/jobs").json()["jobs"]) == before


def test_pvt_sweep_inherits_the_circuits_frequency_range(client):
    # corner_table defaults to the AFE 0.01 Hz - 10 kHz grid. Run a 100 MHz-class
    # silicon OTA over it and `bw_Hz` comes back pinned at 10 kHz -- the top of the
    # grid reported as a measurement. The job must inherit the circuit's own AC
    # sweep instead, so the PVT column agrees with that circuit's AC run.
    circuit = _load("sky130_fd_ota.json")
    ac_stop = circuit["analyses"]["ac"]["freqs"]["stop"]
    assert ac_stop > 1e9                       # the fixture this test relies on

    job_id = client.post("/api/v1/jobs/pvt", json={"circuit": circuit}).json()["job_id"]
    body = _wait_terminal(client, job_id, timeout=90.0)
    assert body["status"] == "done", body.get("error")
    res = body["result"]

    assert res["freq_source"] == "circuit analyses.ac.freqs"
    assert res["freq_range_hz"][1] == ac_stop
    assert res["noise_band_hz"] == circuit["analyses"]["noise"]["band"]
    # The measured bandwidth is a property of the amplifier, not of the grid.
    assert res["table"]["tt"]["bw_Hz"] > 2e4


def test_mc_sweep_inherits_the_circuits_frequency_range(client):
    # The same grid-ceiling trap as the PVT case, and worse in an MC summary:
    # every sample hits the identical ceiling, so the spread collapses to sigma
    # zero and the result reads as a remarkably robust design rather than as a
    # measurement that never happened.
    circuit = _load("sky130_fd_ota.json")
    job_id = client.post(
        "/api/v1/jobs/mc", json={"circuit": circuit, "n": 4, "seed": 0},
    ).json()["job_id"]
    body = _wait_terminal(client, job_id, timeout=120.0)
    assert body["status"] == "done", body.get("error")
    res = body["result"]

    assert res["freq_source"] == "circuit analyses.ac.freqs"
    assert res["noise_band_hz"] == circuit["analyses"]["noise"]["band"]
    # Bandwidth is a property of the amplifier: above the AFE 10 kHz ceiling,
    # and not identical across every sample.
    assert res["summary"]["bw_Hz"]["mean"] > 2e4
    assert res["summary"]["bw_Hz"]["std"] > 0.0


def test_pvt_request_frequency_range_overrides_the_circuit(client):
    circuit = _load("sky130_fd_ota.json")
    job_id = client.post("/api/v1/jobs/pvt", json={
        "circuit": circuit,
        "freqs": {"start": 100.0, "stop": 1e6, "num": 31, "scale": "log"},
        "band": [100.0, 1e5],
    }).json()["job_id"]
    body = _wait_terminal(client, job_id, timeout=90.0)
    assert body["status"] == "done", body.get("error")
    res = body["result"]
    assert res["freq_source"] == "request"
    assert res["freq_range_hz"] == [100.0, 1e6]
    assert res["noise_band_hz"] == [100.0, 1e5]
