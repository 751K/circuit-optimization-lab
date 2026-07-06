"""Central option registry for JSON analysis dispatch.

The dispatch layer, JSON schema, tests, and calibration metadata used to carry
parallel hand-written option lists.  Keep the solver-facing option contract here
and let callers derive allowlists/defaults from this registry.
"""
from __future__ import annotations

from dataclasses import dataclass


_MISSING = object()


def _bool(value):
    return bool(value)


def _int(value):
    return int(value)


def _float(value):
    return float(value)


def _dict(value):
    return dict(value or {})


def _tuple(value):
    return tuple(value)


def _int_or_none(value):
    return None if value is None else int(value)


@dataclass(frozen=True)
class AnalysisOption:
    name: str
    default: object = _MISSING
    cast: object = None
    forward: bool = True
    schema: object = None

    def convert(self, cfg):
        if self.name not in cfg:
            if self.default is _MISSING:
                return _MISSING
            value = self.default
        else:
            value = cfg[self.name]
        return self.cast(value) if self.cast is not None else value


def _opt(name, *, default=_MISSING, cast=None, forward=True, schema=None):
    return AnalysisOption(name, default=default, cast=cast, forward=forward, schema=schema)


POSITIVE_NUMBER = {"$ref": "#/$defs/positiveNumber"}
CORNER = {"$ref": "#/$defs/corner"}
TIME_GRID = {"$ref": "#/$defs/timeGrid"}
PERIODIC = {"$ref": "#/$defs/periodic"}
FREQUENCY_GRID = {"$ref": "#/$defs/frequencyGrid"}
BAND = {"$ref": "#/$defs/band"}
COMPLEX_MAP = {"$ref": "#/$defs/complexMap"}
PSS_ANALYSIS = {"$ref": "#/$defs/pssAnalysis"}
CAP_MODE = {
    "type": "string",
    "enum": ["charge", "q", "qstamp", "q-stamp", "average", "avg", "trapezoid", "trap"],
}
INTEGRATION_METHOD = {"type": "string", "enum": ["gear2", "be"]}
HB_SOLVER = {"enum": ["auto", "dense", "sparse", "iterative"]}


ADAPTIVE_OPTION_NAMES = frozenset({
    "adaptive_reltol", "adaptive_vabstol", "adaptive_iabstol",
    "adaptive_max_steps", "adaptive_h0", "adaptive_freeze_factor",
})

ADAPTIVE_OPTIONS = (
    _opt("adaptive", cast=_bool, schema={"type": "boolean"}),
    _opt("adaptive_config"),
    _opt("adaptive_reltol", cast=_float, schema=POSITIVE_NUMBER),
    _opt("adaptive_vabstol", cast=_float, schema=POSITIVE_NUMBER),
    _opt("adaptive_iabstol", cast=_float, schema=POSITIVE_NUMBER),
    _opt("adaptive_max_steps", cast=_int, schema={"type": "integer", "minimum": 1}),
    _opt("adaptive_h0", cast=_float, schema=POSITIVE_NUMBER),
)

PSS_OPTIONS = (
    _opt("periodic", forward=False, schema=PERIODIC),
    _opt("n_points", forward=False, cast=_int, schema={"type": "integer", "minimum": 2}),
    _opt("tgrid", forward=False, schema=TIME_GRID),
    _opt("tstab_periods", cast=_int, schema={"type": "integer", "minimum": 0}),
    _opt("max_step", cast=_float, schema=POSITIVE_NUMBER),
    _opt("flat_max_step", cast=_float, schema=POSITIVE_NUMBER),
    _opt("max_retry_subdivisions", cast=_int, schema={"type": "integer", "minimum": 0}),
    _opt("newton_maxit", cast=_int, schema={"type": "integer", "minimum": 1}),
    _opt("newton_step_limit", cast=_float, schema=POSITIVE_NUMBER),
    _opt("newton_vtol", cast=_float, schema=POSITIVE_NUMBER),
    _opt("fallback_full_jacobian", cast=_bool, schema={"type": "boolean"}),
    _opt("fallback_least_squares", cast=_bool, schema={"type": "boolean"}),
    _opt("fallback_tol", cast=_float, schema=POSITIVE_NUMBER),
    _opt("residual_tol", cast=_float, schema=POSITIVE_NUMBER),
    _opt("max_shooting_iters", cast=_int, schema={"type": "integer", "minimum": 0}),
    _opt("fd_step", cast=_float, schema=POSITIVE_NUMBER),
    _opt("min_damping", cast=_float, schema=POSITIVE_NUMBER),
    _opt("jacobian_reuse", cast=_bool, schema={"type": "boolean"}),
    _opt("jacobian_rebuild_interval", cast=_int, schema={"type": "integer", "minimum": 0}),
    _opt("analytic_jacobian", cast=_bool, schema={"type": "boolean"}),
    _opt("rail_margin", cast=_float, schema={"type": "number"}),
    _opt("check_periodic_inputs", cast=_bool, schema={"type": "boolean"}),
    _opt("input_periodic_tol", cast=_float, schema=POSITIVE_NUMBER),
    _opt("profile", cast=_bool, schema={"type": "boolean"}),
    _opt("corner", schema=CORNER),
    _opt("integration_method", schema=INTEGRATION_METHOD),
    _opt("physical_factor", cast=_float, schema=POSITIVE_NUMBER),
    _opt("max_stabilization_periods", cast=_int, schema={"type": "integer", "minimum": 0}),
    _opt("levenberg_marquardt", cast=_bool, schema={"type": "boolean"}),
    _opt("cap_mode", schema=CAP_MODE),
    _opt("cap_mode_id", cast=_int, schema={"type": "integer", "minimum": 0, "maximum": 1}),
    _opt("adaptive_freeze_factor", cast=_float, schema=POSITIVE_NUMBER),
) + ADAPTIVE_OPTIONS

TRANSIENT_OPTIONS = (
    _opt("tgrid", forward=False, schema=TIME_GRID),
    _opt("periodic", forward=False, schema=PERIODIC),
    _opt("duration", forward=False, cast=_float, schema=POSITIVE_NUMBER),
    _opt("tstop", forward=False, cast=_float, schema=POSITIVE_NUMBER),
    _opt("n_points", forward=False, cast=_int, schema={"type": "integer", "minimum": 2}),
    _opt("max_step", cast=_float, schema=POSITIVE_NUMBER),
    _opt("flat_max_step", cast=_float, schema=POSITIVE_NUMBER),
    _opt("max_retry_subdivisions", cast=_int, schema={"type": "integer", "minimum": 0}),
    _opt("newton_maxit", cast=_int, schema={"type": "integer", "minimum": 1}),
    _opt("newton_step_limit", cast=_float, schema=POSITIVE_NUMBER),
    _opt("newton_vtol", cast=_float, schema=POSITIVE_NUMBER),
    _opt("fallback_full_jacobian", cast=_bool, schema={"type": "boolean"}),
    _opt("fallback_least_squares", cast=_bool, schema={"type": "boolean"}),
    _opt("fallback_tol", cast=_float, schema=POSITIVE_NUMBER),
    _opt("profile", cast=_bool, schema={"type": "boolean"}),
    _opt("rail_margin", cast=_float, schema={"type": "number"}),
    _opt("corner", schema=CORNER),
    _opt("integration_method", schema=INTEGRATION_METHOD),
    _opt("gear2_be_fallback", cast=_bool, schema={"type": "boolean"}),
    _opt("cap_mode", schema=CAP_MODE),
    _opt("cap_mode_id", cast=_int, schema={"type": "integer", "minimum": 0, "maximum": 1}),
) + ADAPTIVE_OPTIONS

PAC_OPTIONS = (
    _opt("freqs", forward=False, schema=FREQUENCY_GRID),
    _opt("input_drive", forward=False, schema=COMPLEX_MAP),
    _opt("pss", forward=False, schema=PSS_ANALYSIS),
    _opt("corner", forward=False, schema=CORNER),
    _opt("fd_state_step", default=1e-4, cast=_float, schema=POSITIVE_NUMBER),
    _opt("fd_input_step", default=1e-4, cast=_float, schema=POSITIVE_NUMBER),
    _opt("transient_kwargs", default={}, cast=_dict, schema={"type": "object", "additionalProperties": True}),
    _opt("pacmag", default=1.0, cast=_float, schema={"type": "number"}),
    _opt("rail_margin", default=None, schema={"type": ["number", "null"]}),
    _opt("cache_linearization", default=True, cast=_bool, schema={"type": "boolean"}),
    _opt("cache_forcing", default=True, cast=_bool, schema={"type": "boolean"}),
    _opt("compute_condition", default=None, schema={"type": ["boolean", "null"]}),
    _opt("lti_fast_path", default=True, cast=_bool, schema={"type": "boolean"}),
    _opt("analytic", default=True, cast=_bool, schema={"type": "boolean"}),
    _opt("n_period_samples", default=384, cast=_int, schema={"type": "integer", "minimum": 2}),
    _opt("max_sideband", default=10, cast=_int, schema={"type": "integer", "minimum": 0}),
    _opt("time_domain", default=False, cast=_bool, schema={"type": "boolean"}),
    _opt("td_integration", default="gear2", schema=INTEGRATION_METHOD),
    _opt("td_n_period_samples", default=768, cast=_int, schema={"type": "integer", "minimum": 2}),
    _opt("profile", default=False, cast=_bool, schema={"type": "boolean"}),
    _opt("debug", default=False, cast=_bool, schema={"type": "boolean"}),
)

PNOISE_OPTIONS = (
    _opt("freqs", forward=False, schema=FREQUENCY_GRID),
    _opt("band", default=(0.05, 100.0), cast=_tuple, schema=BAND),
    _opt("input_drive", forward=False, schema=COMPLEX_MAP),
    _opt("pss", forward=False, schema=PSS_ANALYSIS),
    _opt("corner", forward=False, schema=CORNER),
    _opt("max_sideband", default=10, cast=_int, schema={"type": "integer", "minimum": 0}),
    _opt("n_period_samples", default=384, cast=_int, schema={"type": "integer", "minimum": 2}),
    _opt("time_domain", default=False, cast=_bool, schema={"type": "boolean"}),
    _opt("noise_devices", default=None, schema={"type": ["array", "null"], "items": {"$ref": "#/$defs/name"}}),
    _opt("gds_noise_devices", default=None, schema={"type": ["array", "null"], "items": {"$ref": "#/$defs/name"}}),
    _opt("switch_noise_conductance_gated", default=True, cast=_bool, schema={"type": "boolean"}),
    _opt("cache_linearization", default=True, cast=_bool, schema={"type": "boolean"}),
    _opt("lti_fast_path", default=True, cast=_bool, schema={"type": "boolean"}),
    _opt("hb_solver", default="auto", schema=HB_SOLVER),
    _opt("hb_sparse_min_size", default=384, cast=_int, schema={"type": "integer", "minimum": 1}),
    _opt("hb_sparse_max_density", default=0.12, cast=_float, schema={"type": "number", "minimum": 0.0, "maximum": 1.0}),
    _opt("hb_sparse_drop_tol", default=0.0, cast=_float, schema={"type": "number", "minimum": 0.0}),
    _opt("iterative_tol", default=1e-10, cast=_float, schema={"type": "number", "exclusiveMinimum": 0.0}),
    _opt("iterative_maxiter", default=10, cast=_int_or_none, schema={"type": ["integer", "null"], "minimum": 1}),
    _opt("profile", default=False, cast=_bool, schema={"type": "boolean"}),
)

ANALYSIS_OPTIONS = {
    "pss": PSS_OPTIONS,
    "transient": TRANSIENT_OPTIONS,
    "pac": PAC_OPTIONS,
    "pnoise": PNOISE_OPTIONS,
}


# Keys the dispatch layer (circuitopt.analysis_dispatch) consumes itself, i.e. legal in
# an ``analyses`` block but NOT part of any solver's option registry above.  These
# are read directly out of ``cfg`` by run_analysis_suite / _run_transient / etc.
# and never reach ``solver_kwargs``.  They must be listed here so that
# ``validate_analysis_cfg`` does not flag them as unknown.  Anything already
# declared in ANALYSIS_OPTIONS (forwarded or not, e.g. ``corner``/``freqs``/
# ``periodic``) is a legal key too and does not need to be repeated here.
#
#   ac      -> _frequency_grid(cfg["freqs"]) + _corner_from_cfg(cfg)
#   noise   -> _frequency_grid(cfg["freqs"]) + _corner_from_cfg(cfg) + band_rms(cfg["band"])
#   transient -> signed_devices (line "signed_devices=tuple(cfg.get('signed_devices', ...))")
#                (freqs is not read; periodic/tgrid/tstop/duration/n_points/corner
#                 all live in TRANSIENT_OPTIONS)
#   pss/pac/pnoise -> every consumed key (freqs/input_drive/pss/corner/band/tgrid/
#                     n_points/periodic) is declared in their *_OPTIONS registries.
DISPATCH_KEYS = {
    "ac": frozenset({"freqs", "corner"}),
    "noise": frozenset({"freqs", "corner", "band"}),
    "transient": frozenset({"signed_devices"}),
    "pss": frozenset(),
    "pac": frozenset(),
    "pnoise": frozenset(),
}


def options_for(analysis):
    return ANALYSIS_OPTIONS.get(str(analysis), ())


def known_keys(analysis):
    """Full set of keys legal in one analysis's ``analyses`` block.

    Union of the solver option registry (:data:`ANALYSIS_OPTIONS`) and the
    dispatch-consumed keys (:data:`DISPATCH_KEYS`).  Analyses without a solver
    registry (``ac``/``noise``) rely entirely on the dispatch set.
    """
    analysis = str(analysis)
    solver = {opt.name for opt in options_for(analysis)}
    return frozenset(solver | DISPATCH_KEYS.get(analysis, frozenset()))


def validate_analysis_cfg(analysis, cfg):
    """Reject unknown keys in one analysis's ``analyses`` block.

    JSON is the only entry point for these options, so a residual key is almost
    always a typo (``max_sidebands`` for ``max_sideband``) that would otherwise
    be silently ignored and run with the default -- the same silent-downgrade
    class this project bans elsewhere.  Raises ``ValueError`` naming the analysis,
    the offending keys, and the legal keys so the user can fix the spelling.
    """
    analysis = str(analysis)
    known = known_keys(analysis)
    extra = sorted(k for k in (cfg or {}) if k not in known)
    if extra:
        raise ValueError(
            f"Unknown option(s) for analysis {analysis!r}: {extra}. "
            f"Valid keys are: {sorted(known)}"
        )


def option_names(analysis, *, forwarded_only=False, schema_only=False):
    out = []
    for opt in options_for(analysis):
        if forwarded_only and not opt.forward:
            continue
        if schema_only and opt.schema is None:
            continue
        out.append(opt.name)
    return frozenset(out)


def solver_kwargs(analysis, cfg, *, include_defaults=False):
    cfg = dict(cfg or {})
    out = {}
    for opt in options_for(analysis):
        if not opt.forward:
            continue
        if include_defaults:
            value = opt.convert(cfg)
        elif opt.name in cfg:
            value = opt.convert(cfg)
        else:
            continue
        if value is not _MISSING:
            out[opt.name] = value
    return out


def schema_properties(analysis):
    return {opt.name: opt.schema for opt in options_for(analysis) if opt.schema is not None}

