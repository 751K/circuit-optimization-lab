"""Shared transient profiling counter schema.

Numba kernels exchange profiling data as a dense float array for speed.  Keep the
slot layout centralized here so Python result assembly and Numba writes cannot
silently drift apart.
"""

PROFILE_NEWTON_ITERS = 0
PROFILE_PMOS_OP_SOLVES = 1
PROFILE_PMOS_INTERNAL_NEWTON_ATTEMPTS = 2
PROFILE_PMOS_INTERNAL_NEWTON_ITERS = 3
PROFILE_INTERNAL_FD_JAC_FALLBACKS = 4
PROFILE_TERMINAL_FD_JAC_FALLBACKS = 5
PROFILE_EDGE_SUBSTEPS = 6
PROFILE_FLAT_SUBSTEPS = 7
PROFILE_EDGE_NEWTON_ITERS = 8
PROFILE_FLAT_NEWTON_ITERS = 9
PROFILE_FAILED_SUBSTEPS = 10
PROFILE_INTERVALS = 11
PROFILE_SUBSTEPS = 12
PROFILE_FAILED_INTERVALS = 13
PROFILE_FAILED_EDGE_INTERVALS = 14
PROFILE_FAILED_FLAT_INTERVALS = 15
PROFILE_FAILED_LAST_RESIDUAL_INF = 16
PROFILE_FAILED_MAX_RESIDUAL_INF = 17
PROFILE_FAILED_LAST_STEP_INF = 18
PROFILE_FAILED_MAX_STEP_INF = 19
PROFILE_FAILED_STAMP_OR_PREV_COUNT = 20
PROFILE_FAILED_LINEAR_SOLVE_COUNT = 21
PROFILE_FAILED_MAXIT_COUNT = 22
PROFILE_STALLED_RESIDUAL_ACCEPTS = 23
PROFILE_LEN = 24

TRANSIENT_PROFILE_FIELDS = (
    "newton_iters_total",
    "pmos_op_solves",
    "pmos_internal_newton_attempts",
    "pmos_internal_newton_iters",
    "internal_fd_jac_fallbacks",
    "terminal_fd_jac_fallbacks",
    "edge_substeps",
    "flat_substeps",
    "edge_newton_iters",
    "flat_newton_iters",
    "failed_substeps",
    "intervals",
    "nsubsteps",
    "failed_intervals",
    "failed_edge_intervals",
    "failed_flat_intervals",
    "failed_last_residual_inf",
    "failed_max_residual_inf",
    "failed_last_step_inf",
    "failed_max_step_inf",
    "failed_stamp_or_prev_count",
    "failed_linear_solve_count",
    "failed_maxit_count",
    "stalled_residual_accepts",
)

PROFILE_SLOT_BY_NAME = {
    name: slot for slot, name in enumerate(TRANSIENT_PROFILE_FIELDS)
}

