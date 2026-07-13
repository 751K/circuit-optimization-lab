"""Smoke tests for the FastAPI service layer (``circuitopt.service``).

Gated on the optional ``fastapi`` dependency (the ``serve`` extra), mirroring
the sklearn guard in ``test_surrogate.py``. Uses ``fastapi.testclient.TestClient``
(in-process, no real server / no open socket). Every case is a thin adapter
check — that the endpoints wire the existing solver stack up correctly — and
stays fast by using the pure-RC ``examples/periodic_rc.json`` circuit.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from circuitopt.service.app import create_app  # noqa: E402
from circuitopt.service.serialize import (  # noqa: E402
    serialize_results,
    to_jsonable,
)

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name):
    return json.loads((_EXAMPLES / name).read_text())


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


# ── health ────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["api"] == "v1"
    import circuitopt
    assert body["version"] == circuitopt.__version__


# ── capabilities ──────────────────────────────────────────────────────────────

def test_capabilities(client):
    r = client.get("/api/v1/capabilities")
    assert r.status_code == 200
    cap = r.json()

    # Registered models: OTFT default + both silicon PDK polarities present.
    models = cap["models"]
    assert "pmos_tft" in models              # OTFT default alias
    assert any(k.startswith("sky130.") for k in models)
    assert any(k.startswith("freepdk45.") for k in models)

    # Analyses + their legal option keys (from analysis_options, not hardcoded).
    analyses = cap["analyses"]
    for name in ("ac", "noise", "transient", "pss", "pac", "pnoise"):
        assert name in analyses
    assert "freqs" in analyses["ac"]
    assert "band" in analyses["noise"]
    assert "max_sideband" in analyses["pnoise"]

    # Corner families.
    corners = cap["corners"]
    assert set(corners["otft"]) == {"typical", "slow", "fast"}
    assert set(corners["sky130"]) == {"tt", "ss", "ff", "sf", "fs"}
    assert set(corners["freepdk45"]) == {"nom", "tt", "ss", "ff", "sf", "fs"}


# ── validate ──────────────────────────────────────────────────────────────────

def test_validate_valid_circuit(client):
    r = client.post("/api/v1/validate", json=_load("periodic_rc.json"))
    assert r.status_code == 200
    assert r.json() == {"valid": True}


def test_validate_valid_voltage_divider(client):
    r = client.post("/api/v1/validate", json=_load("voltage_divider.json"))
    assert r.status_code == 200
    assert r.json()["valid"] is True


def test_validate_unknown_analysis_key(client):
    circuit = _load("periodic_rc.json")
    circuit["analyses"]["ac"]["bogus_key"] = 1
    r = client.post("/api/v1/validate", json=circuit)
    assert r.status_code == 200          # validation outcome is the payload
    body = r.json()
    assert body["valid"] is False
    assert body["errors"]                # non-empty
    assert "bogus_key" in body["errors"][0]


def test_validate_structurally_broken(client):
    circuit = _load("periodic_rc.json")
    del circuit["solved"]                # required field
    r = client.post("/api/v1/validate", json=circuit)
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert body["errors"]


# ── solve ─────────────────────────────────────────────────────────────────────

def test_solve_ac(client):
    r = client.post("/api/v1/solve",
                    json={"circuit": _load("periodic_rc.json"), "selected": ["ac"]})
    assert r.status_code == 200
    body = r.json()
    assert set(body["results"].keys()) == {"ac"}          # subset honored
    assert isinstance(body["elapsed_s"], (int, float))

    ac = body["results"]["ac"]
    assert math.isfinite(ac["Av_dc_dB"])
    assert math.isfinite(ac["bw_Hz"])
    # complex response serialized as {re, im} objects with finite parts
    resp0 = ac["response"][0]
    assert set(resp0) == {"re", "im"}
    assert math.isfinite(resp0["re"]) and math.isfinite(resp0["im"])

    # whole payload is strict JSON (no NaN / Infinity tokens)
    dumped = json.dumps(body)
    assert "NaN" not in dumped and "Infinity" not in dumped


def test_solve_selected_subset(client):
    r = client.post("/api/v1/solve",
                    json={"circuit": _load("periodic_rc.json"), "selected": ["ac", "noise"]})
    assert r.status_code == 200
    assert set(r.json()["results"].keys()) == {"ac", "noise"}


def test_solve_parse_error_is_422(client):
    r = client.post("/api/v1/solve", json={"circuit": {"not": "a circuit"}})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["stage"] == "parse"
    assert detail["message"]             # human-readable message present


def test_solve_broken_analysis_key_is_422(client):
    circuit = _load("periodic_rc.json")
    circuit["analyses"]["ac"]["bogus_key"] = 1
    r = client.post("/api/v1/solve", json={"circuit": circuit, "selected": ["ac"]})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["stage"] == "solve"    # rejected inside run_analysis_suite
    assert "bogus_key" in detail["message"]


# ── serialize (direct unit test of the JSON-safe conventions) ─────────────────

def test_serialize_conventions():
    payload = {
        "ac": {
            "nan": float("nan"),
            "pos_inf": float("inf"),
            "neg_inf": float("-inf"),
            "np_scalar": np.float64(1.5),
            "np_int": np.int64(7),
            "np_bool": np.bool_(True),
            "cplx": complex(1.0, -2.0),
            "cplx_nan_im": complex(3.0, float("nan")),
            "arr": np.array([1.0, float("nan"), 3.0]),
            "carr": np.array([1 + 2j, 3 - 4j]),
            "nested": {"tup": (1, 2), "s": "ok"},
            "_private": "dropped",
            "fn": (lambda: 1),          # callable -> dropped
        },
        "empty": None,                  # None analysis -> dropped by serialize_results
    }
    out = serialize_results(payload)

    assert "empty" not in out           # None-valued analysis skipped
    ac = out["ac"]

    # NaN / +Inf / -Inf -> null
    assert ac["nan"] is None
    assert ac["pos_inf"] is None
    assert ac["neg_inf"] is None

    # numpy scalars -> native
    assert ac["np_scalar"] == 1.5 and isinstance(ac["np_scalar"], float)
    assert ac["np_int"] == 7 and isinstance(ac["np_int"], int)
    assert ac["np_bool"] is True

    # complex -> {re, im}; NaN imaginary part -> null
    assert ac["cplx"] == {"re": 1.0, "im": -2.0}
    assert ac["cplx_nan_im"] == {"re": 3.0, "im": None}

    # ndarray -> list, NaN element -> null
    assert ac["arr"] == [1.0, None, 3.0]
    # complex ndarray -> list of {re, im}
    assert ac["carr"] == [{"re": 1.0, "im": 2.0}, {"re": 3.0, "im": -4.0}]

    # nested containers recursed; tuple -> list
    assert ac["nested"] == {"tup": [1, 2], "s": "ok"}

    # private key + callable dropped
    assert "_private" not in ac
    assert "fn" not in ac

    # the whole thing is strict-JSON serializable
    json.dumps(out)


def test_to_jsonable_bool_not_int():
    # bool must survive as bool, not be coerced to 0/1 (bool is an int subclass).
    assert to_jsonable(True) is True
    assert to_jsonable(False) is False
