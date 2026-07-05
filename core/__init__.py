"""Circuit optimization solver package.

Quick start::

    from core import run_analysis_suite, load_circuit_json
    spec = load_circuit_json("examples/periodic_rc.json")
    results = run_analysis_suite(spec)

Or from the command line::

    python -m core examples/periodic_rc.json
"""

# ── Numba flag pre-scan (MUST run before any solver import below) ──────────────
# core.numba_kernels reads CIRCUIT_USE_NUMBA and bakes USE_NUMBA/NUMBA_AVAILABLE
# at *import time*. Under `python -m core …`, this package __init__ runs (and its
# solver imports below pull numba_kernels in transitively) *before* __main__.py's
# code executes — so a `_cmd_*` handler that sets the env var, or even a pre-scan
# in __main__.py, would be too late and `--no-numba` would silently no-op. Scan
# argv here, at the earliest possible point, so the flag actually takes effect.
import os as _os
import sys as _sys

if "--no-numba" in _sys.argv:
    _os.environ["CIRCUIT_USE_NUMBA"] = "0"

from .ac_solver import ac_solve
from .analysis_dispatch import run_analysis_suite, run_json_analyses
from .circuit_loader import CircuitSpec, load_circuit_json
from .device_factory import CircuitBinding
from .device_model import (TransistorModel, NumbaParams, PDK, create_device,
                           create_transistor, register_model, register_pdk,
                           get_default_model_type, get_default_pdk, get_pdk,
                           list_pdks, transistor_type)
from .noise_solver import band_rms, noise_analysis
from .pac_solver import pac_solve
from . import pmos_tft_model  # noqa: F401 — triggers register_pdk("at4000tg", …)
from . import sky130_model    # noqa: F401 — triggers register_pdk("sky130", …)
from . import freepdk45_model  # noqa: F401 — triggers register_pdk("freepdk45", …)
from .pnoise_solver import pnoise_solve
from .pss_solver import pss_solve
from .topology import Topology
from .transient_solver import transient

# ``explore`` is *not* re-exported here because ``core.explore`` already refers
# to the ``core.explore`` *module*.  Use ``from core.explore import explore``
# or ``core.explore.explore(...)`` to reach the design-exploration function.
# All other top-level names exported below have no module-name collision.

__all__ = [
    # device model abstraction
    "TransistorModel",
    "NumbaParams",
    "PDK",
    "create_device",
    "create_transistor",
    "register_model",
    "register_pdk",
    "get_default_model_type",
    "get_default_pdk",
    "get_pdk",
    "list_pdks",
    "transistor_type",
    # circuit loading
    "load_circuit_json",
    "CircuitSpec",
    "CircuitBinding",
    "Topology",
    # analysis dispatch
    "run_analysis_suite",
    "run_json_analyses",
    # individual solvers
    "ac_solve",
    "noise_analysis",
    "band_rms",
    "transient",
    "pss_solve",
    "pac_solve",
    "pnoise_solve",
]
