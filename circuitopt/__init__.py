"""Circuit optimization solver package.

Quick start::

    from circuitopt import run_analysis_suite, load_circuit_json
    spec = load_circuit_json("examples/periodic_rc.json")
    results = run_analysis_suite(spec)

Or from the command line::

    python -m circuitopt examples/periodic_rc.json

Compute engine: as of v2.0.0 the numerical work runs on a single engine —
``"rust"``, the compiled ``circuitopt_core`` core. The ``--engine`` flag and
``CIRCUIT_ENGINE`` env var are retained but accept only ``rust``; the former
``python``/``numba`` engines (and ``--no-numba`` / ``CIRCUIT_USE_NUMBA``) were
removed and now error. ``current_engine()`` reports the active engine. See
``circuitopt/_engine.py``.
"""

# ── Engine selection pre-scan (MUST run before any solver import below) ────────
# apply_engine_env() resolves the engine from argv/env at the earliest possible
# point — before the solver imports below run. Under `python -m circuitopt …`
# this package __init__ (and its transitive solver imports) executes *before*
# __main__.py's code, so resolving here is what lets `--engine` / CIRCUIT_ENGINE
# be validated at all (and the retired numba switches rejected loudly); the
# resolved name is written back to CIRCUIT_ENGINE for child processes. See
# _engine.py.
from ._engine import apply_engine_env, current_engine

apply_engine_env()

# Single-source version: the number lives only in pyproject.toml. When installed
# (pip / wheel) importlib.metadata reads it back; a bare repo checkout with no
# `pip install` falls back to a local sentinel.
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("circuit-optimization")
except Exception:  # not installed (repo checkout w/o pip install)
    __version__ = "0.0.0+local"

from .ac_solver import ac_solve
from .analysis_dispatch import run_analysis_suite, run_json_analyses
from .circuit_loader import CircuitSpec, load_circuit_json
from .device_factory import CircuitBinding
from .device_model import (TransistorModel, NumbaParams, PDK, create_device,
                           create_transistor, register_model, register_pdk,
                           get_default_model_type, get_default_pdk, get_pdk,
                           list_pdks, registered_models, transistor_type)
from .noise_solver import band_rms, noise_analysis
from .pac_solver import pac_solve
from . import pmos_tft_model  # noqa: F401 — triggers register_pdk("at4000tg", …)
from .pdk.sky130 import device as _sky130_native_device  # noqa: F401
from .pdk.freepdk45 import device as _freepdk45_native_device  # noqa: F401
from .pdk.tsmc28 import device as _tsmc28_native_device  # noqa: F401
from .pnoise_solver import pnoise_solve
from .pss_solver import pss_solve
from .topology import Topology
from .transient_solver import transient
from .adc import (average_supply_power, average_waveform_source_power,
                  code_density_metrics, decode_bit_waveforms, dynamic_metrics,
                  static_ramp_metrics)
from .sar import (run_sar_conversion, run_sar_signal, run_sar_sweep,
                  sar_input_waveforms, sar_time_grid)
from .sar_mc import sar_mismatch_mc
from .sar_explore import (apply_sar_variables, evaluate_sar,
                          load_sar_explore_json, sar_explore_from_dict)

# ``explore`` / ``sar_explore`` (the driver functions) are *not* re-exported here
# because ``circuitopt.explore`` / ``circuitopt.sar_explore`` already refer to the
# *modules*. Use ``from circuitopt.explore import explore`` /
# ``from circuitopt.sar_explore import sar_explore`` to reach the driver functions.
# All other top-level names exported below have no module-name collision.

__all__ = [
    # package metadata
    "__version__",
    # compute-engine switch (rust only, as of v2.0.0)
    "current_engine",
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
    "registered_models",
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
    # SAR ADC workflow
    "run_sar_conversion",
    "run_sar_sweep",
    "run_sar_signal",
    "sar_input_waveforms",
    "sar_time_grid",
    "sar_mismatch_mc",
    "evaluate_sar",
    "apply_sar_variables",
    "load_sar_explore_json",
    "sar_explore_from_dict",
]
