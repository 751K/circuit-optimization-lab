"""Circuit optimization solver package.

Quick start::

    from core import run_analysis_suite, load_circuit_json
    spec = load_circuit_json("examples/periodic_rc.json")
    results = run_analysis_suite(spec)

Or from the command line::

    python -m core examples/periodic_rc.json
"""

try:
    from .ac_solver import ac_solve
    from .analysis_dispatch import run_analysis_suite, run_json_analyses
    from .circuit_loader import CircuitSpec, load_circuit_json
    from .device_model import (TransistorModel, NumbaParams, create_device,
                               register_model, get_default_model_type)
    from .noise_solver import band_rms, noise_analysis
    from .pac_solver import pac_solve
    from . import pmos_tft_model  # noqa: F401 — triggers register_model("pmos_tft")
    from .pnoise_solver import pnoise_solve
    from .pss_solver import pss_solve
    from .topology import Topology
    from .transient_solver import transient
except ImportError:  # pragma: no cover - legacy direct module import
    from ac_solver import ac_solve
    from analysis_dispatch import run_analysis_suite, run_json_analyses
    from circuit_loader import CircuitSpec, load_circuit_json
    from device_model import (TransistorModel, NumbaParams, create_device,
                              register_model, get_default_model_type)
    from noise_solver import band_rms, noise_analysis
    from pac_solver import pac_solve
    import pmos_tft_model  # noqa: F401
    from pnoise_solver import pnoise_solve
    from pss_solver import pss_solve
    from topology import Topology
    from transient_solver import transient

# ``explore`` is *not* re-exported here because ``core.explore`` already refers
# to the ``core.explore`` *module*.  Use ``from core.explore import explore``
# or ``core.explore.explore(...)`` to reach the design-exploration function.
# All other top-level names exported below have no module-name collision.

__all__ = [
    # device model abstraction
    "TransistorModel",
    "NumbaParams",
    "create_device",
    "register_model",
    "get_default_model_type",
    # circuit loading
    "load_circuit_json",
    "CircuitSpec",
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
