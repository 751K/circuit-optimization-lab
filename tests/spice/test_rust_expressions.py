"""Differential parity tests: the Rust ``co-spice`` engine vs the frozen Python
:mod:`circuitopt.spice.expressions` reference.

The Rust evaluator (surfaced through ``circuitopt_core.SpiceScope`` /
``circuitopt_core.spice_eval``) must, for every input, either return a value
bit-identical to (or within a relative error of ``1e-14`` of) the Python
reference, or raise the *same* exception class by name. These tests assert that
on a hand-written adversarial corpus, on the full local TSMC28HPC+ parameter
corpus (when installed), and under concurrent resolution of a shared scope.

Foundry model text is licensed: the large-corpus test reads it only at runtime
and asserts counts / relative error — it never records an expression or a
parameter value in this file, in assertion messages, or in output.
"""
from __future__ import annotations

import math
import os
import threading

import pytest

from circuitopt.spice import (
    EvaluationScope,
    ParameterCycleError,
    SpiceExpressionError,
    UnknownSymbolError,
)

circuitopt_core = pytest.importorskip("circuitopt_core")

if not hasattr(circuitopt_core, "SpiceScope"):
    pytest.skip(
        "circuitopt_core lacks the SpiceScope parity surface; rebuild the wheel",
        allow_module_level=True,
    )


# The relative-error gate for the libm ``pow`` path; everything else is exact.
REL_TOLERANCE = 1e-14


def _relative_error(reference: float, candidate: float) -> float:
    if reference == candidate:
        return 0.0
    if math.isnan(reference) or math.isnan(candidate):
        return 0.0 if (math.isnan(reference) and math.isnan(candidate)) else math.inf
    scale = max(abs(reference), abs(candidate), 1e-300)
    return abs(reference - candidate) / scale


def _reference_outcome(expression: str):
    """``(True, value, None)`` or ``(False, None, exception_class_name)``."""
    try:
        return True, EvaluationScope().evaluate(expression), None
    except Exception as exc:  # noqa: BLE001 - parity requires catching all
        return False, None, type(exc).__name__


def _rust_outcome(expression: str):
    try:
        return True, circuitopt_core.spice_eval(expression), None
    except Exception as exc:  # noqa: BLE001 - parity requires catching all
        return False, None, type(exc).__name__


def _assert_expression_parity(expression: str) -> None:
    ref_ok, ref_value, ref_exc = _reference_outcome(expression)
    rust_ok, rust_value, rust_exc = _rust_outcome(expression)
    assert ref_ok == rust_ok, (
        f"raise/no-raise mismatch for {expression!r}: "
        f"python={'value' if ref_ok else ref_exc}, rust={'value' if rust_ok else rust_exc}"
    )
    if ref_ok:
        assert _relative_error(ref_value, rust_value) <= REL_TOLERANCE, (
            f"value mismatch for {expression!r}: python={ref_value!r}, rust={rust_value!r}"
        )
    else:
        assert ref_exc == rust_exc, (
            f"exception-class mismatch for {expression!r}: python={ref_exc}, rust={rust_exc}"
        )


# ---------------------------------------------------------------------------
# Exception surface
# ---------------------------------------------------------------------------


def test_exception_hierarchy_matches_reference():
    assert issubclass(circuitopt_core.SpiceExpressionError, ValueError)
    assert issubclass(
        circuitopt_core.UnknownSymbolError, circuitopt_core.SpiceExpressionError
    )
    assert issubclass(
        circuitopt_core.ParameterCycleError, circuitopt_core.SpiceExpressionError
    )
    # Names match the reference classes so ``type(exc).__name__`` parity holds.
    assert circuitopt_core.SpiceExpressionError.__name__ == SpiceExpressionError.__name__
    assert circuitopt_core.UnknownSymbolError.__name__ == UnknownSymbolError.__name__
    assert circuitopt_core.ParameterCycleError.__name__ == ParameterCycleError.__name__


# ---------------------------------------------------------------------------
# Hand-written adversarial corpus — values
# ---------------------------------------------------------------------------

_VALUE_CORPUS = [
    # Reference seeds (tests/spice/test_expressions.py).
    "1 + 2 * 3",
    "2**3**2",
    "2.5k + 500",
    "1 ? 2 : 3",
    "0 ? 2 : 3",
    "-2 < -1 ? max(4, 3) : 0",
    "sgn(-3) + int(2.9)",
    "pwr(3, 2) + sqrt(16)",
    "agauss(5, 2, 3)",
    # Right-associative power vs the unary-minus quirk.
    "2**3**2",
    "-2**2",
    "2^3^2",
    # SPICE magnitude suffixes, including mil and case-insensitivity.
    "2.5mil",
    "1mil + 1",
    "2.5MEG",
    "2.5meg + 2.5m",
    "1t + 1g + 1k + 1u + 1n + 1p + 1f",
    "1.1u * 3",
    "3.3n / 2",
    # Boolean / logical, negative-number truthiness, short-circuit.
    "!0",
    "!5",
    "!(-3)",
    "0 && (1/0)",
    "1 || (1/0)",
    "0 || -1",
    "2 && 3",
    "3 == 3",
    "3 != 4",
    "2 <= 2",
    "2 >= 3",
    # Deeply nested, right-associative ternary.
    "0 ? 1 : 0 ? 2 : 1 ? 3 : 4",
    "1 ? (0 ? 1 : 2) : (1 ? 3 : 4)",
    "1 ? 2 ? 3 : 4 : 5",
    # agauss 2- and 3-argument forms.
    "agauss(5, 2)",
    "agauss(-7.5, 1, 3)",
    # Builtins.
    "abs(-4.5)",
    "sqrt(2)",
    "exp(2.5)",
    "log10(7)",
    "int(-2.9)",
    "int(2.9)",
    "int(-0.5)",
    "sgn(0)",
    "sgn(-1e-30)",
    "min(3, 1, 2)",
    "max(3, 1, 2)",
    "selmin(9, 4, 7)",
    "pow(2, 10)",
    "pwr(-8, 2)",
    # Constants.
    "pi",
    "e",
    "true",
    "false",
    "2 * pi",
    # Mixed precedence.
    "1 + 2 * 3 - 4 / 2 ^ 2",
    "(1 + 2) * (3 - 4)",
    "-(3 + 4) * 2",
    # Leading/trailing dot numbers.
    ".5 + 2.",
    "1e3 + 1.5e-2",
]


@pytest.mark.parametrize("expression", _VALUE_CORPUS)
def test_value_corpus_parity(expression):
    _assert_expression_parity(expression)


def test_value_corpus_worst_relative_error_is_bounded():
    worst = 0.0
    for expression in _VALUE_CORPUS:
        ref_ok, ref_value, _ = _reference_outcome(expression)
        rust_ok, rust_value, _ = _rust_outcome(expression)
        if ref_ok and rust_ok:
            worst = max(worst, _relative_error(ref_value, rust_value))
    assert worst <= REL_TOLERANCE


# ---------------------------------------------------------------------------
# Hand-written adversarial corpus — errors (exception-class parity)
# ---------------------------------------------------------------------------

_ERROR_CORPUS = [
    "1 / 0",  # ZeroDivisionError -> SpiceExpressionError
    "missing + 1",  # UnknownSymbolError
    "v(0)",  # unknown function -> UnknownSymbolError
    "sqrt(-1)",  # math domain error -> SpiceExpressionError
    "log10(0)",
    "log10(-1)",
    "exp(1000)",  # overflow -> SpiceExpressionError
    "0 ** -1",  # 0 to a negative power
    "1e300 * 1e300",  # non-finite boundary guard
    "agauss(5)",  # arity error
    "agauss(1, 2, 3, 4)",  # arity error
    "sqrt(1, 2)",  # arity error
    "max()",  # empty variadic
    "1 +",  # parse error
    "1 2",  # parse error
    "max(1,",  # parse error
    "(1",  # parse error
    "1 @ 2",  # unsupported character
    "1/0 < 5",  # eager divide-by-zero, not swallowed by comparison
    "sqrt(-1) + 3",
]


@pytest.mark.parametrize("expression", _ERROR_CORPUS)
def test_error_corpus_parity(expression):
    _assert_expression_parity(expression)


# ---------------------------------------------------------------------------
# Scope semantics — lazy parameters, functions, cycles, evaluate_all
# ---------------------------------------------------------------------------


def test_lazy_case_insensitive_forward_reference_parity():
    py_scope = EvaluationScope(values={"W": 2e-6})
    py_scope.define("area", "w * length")
    py_scope.define("LENGTH", "30n")

    rust_scope = circuitopt_core.SpiceScope({"W": 2e-6})
    rust_scope.define("area", "w * length")
    rust_scope.define("LENGTH", "30n")

    py_value = py_scope.resolve_symbol("AREA")
    rust_value = rust_scope.resolve_symbol("AREA")
    assert _relative_error(py_value, rust_value) <= REL_TOLERANCE
    assert rust_value == pytest.approx(60e-15)


def test_user_defined_function_lexical_parent_parity():
    py_scope = EvaluationScope(values={"offset": 2.0})
    py_scope.define_function("shift", ("x", "gain"), "x * gain + offset")

    rust_scope = circuitopt_core.SpiceScope({"offset": 2.0})
    rust_scope.define_function("shift", ["x", "gain"], "x * gain + offset")

    assert py_scope.evaluate("SHIFT(3, 4)") == rust_scope.evaluate("SHIFT(3, 4)") == 14.0


def test_set_value_and_define_interplay_parity():
    py_scope = EvaluationScope()
    rust_scope = circuitopt_core.SpiceScope()
    for scope in (py_scope, rust_scope):
        scope.define("x", "2 + 3")
    assert py_scope.resolve_symbol("x") == rust_scope.resolve_symbol("x") == 5.0
    for scope in (py_scope, rust_scope):
        scope.set_value("x", 42.0)
    assert py_scope.resolve_symbol("x") == rust_scope.resolve_symbol("x") == 42.0


def test_cycle_detection_parity():
    py_scope = EvaluationScope()
    py_scope.define("a", "b + 1")
    py_scope.define("b", "a + 1")
    rust_scope = circuitopt_core.SpiceScope()
    rust_scope.define("a", "b + 1")
    rust_scope.define("b", "a + 1")

    with pytest.raises(ParameterCycleError, match="a -> b -> a"):
        py_scope.resolve_symbol("a")
    with pytest.raises(circuitopt_core.ParameterCycleError, match="a -> b -> a"):
        rust_scope.resolve_symbol("a")


def test_evaluate_all_parity():
    def build(scope):
        scope.set_value("vdd", 1.8)
        scope.define("half", "vdd / 2")
        scope.define("ratio", "half * 10")
        scope.define("g", "sqrt(ratio) + pwr(2, 3)")
        return scope

    py_values = build(EvaluationScope()).evaluate_all()
    rust_values = build(circuitopt_core.SpiceScope()).evaluate_all()
    assert set(py_values) == set(rust_values)
    for key, py_value in py_values.items():
        assert _relative_error(py_value, rust_values[key]) <= REL_TOLERANCE


# ---------------------------------------------------------------------------
# Concurrency — a shared scope resolved from many threads is deterministic
# ---------------------------------------------------------------------------


def test_shared_scope_resolves_parameters_concurrently():
    chain_length = 96

    def build(scope):
        scope.set_value("base", 2.0)
        for index in range(chain_length):
            if index == 0:
                scope.define("p0", "base + 1")
            else:
                scope.define(f"p{index}", f"p{index - 1} + 0.5")
        return scope

    reference = build(EvaluationScope())
    expected = {f"p{i}": reference.resolve_symbol(f"p{i}") for i in range(chain_length)}

    shared = build(circuitopt_core.SpiceScope())
    results: list[dict[str, float]] = []
    barrier = threading.Barrier(8)
    lock = threading.Lock()

    def worker():
        barrier.wait()
        local = {f"p{i}": shared.resolve_symbol(f"p{i}") for i in range(chain_length)}
        with lock:
            results.append(local)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8
    for local in results:
        assert local == expected


# ---------------------------------------------------------------------------
# Large-scale corpus — the full local TSMC28HPC+ parameter set
# ---------------------------------------------------------------------------


def _tsmc_expression_corpus() -> list[str] | None:
    """All parameter expressions from the licensed model, or ``None`` if absent.

    Reads only at runtime; the returned strings are never persisted or logged.
    """
    from circuitopt.spice import parse_spice_library
    from circuitopt.toolchain import tsmc28_model_dir

    path = os.path.join(tsmc28_model_dir(), "cln28hpcp_1d8_elk_v1d0_2p2.l")
    if not os.path.isfile(path):
        return None

    library = parse_spice_library(path)
    expressions: list[str] = []
    for section in library.sections.values():
        statements = list(section.statements)
        for subcircuit in section.subcircuits.values():
            statements.extend(subcircuit.statements)
            for parameter in subcircuit.parameters:
                expressions.append(parameter.expression)
        for statement in statements:
            for parameter in statement.parameters:
                expressions.append(parameter.expression)
    return expressions


def test_all_real_tsmc_parameter_expressions_match_reference():
    expressions = _tsmc_expression_corpus()
    if expressions is None:
        pytest.skip("licensed TSMC28HPC+ model is not installed")

    total = len(expressions)
    value_matches = 0
    raise_matches = 0
    mismatches = 0
    worst_relative_error = 0.0

    for expression in expressions:
        ref_ok, ref_value, ref_exc = _reference_outcome(expression)
        rust_ok, rust_value, rust_exc = _rust_outcome(expression)
        if ref_ok != rust_ok:
            mismatches += 1
            continue
        if ref_ok:
            relative_error = _relative_error(ref_value, rust_value)
            worst_relative_error = max(worst_relative_error, relative_error)
            if relative_error <= REL_TOLERANCE:
                value_matches += 1
            else:
                mismatches += 1
        else:
            if ref_exc == rust_exc:
                raise_matches += 1
            else:
                mismatches += 1

    # Aggregate assertions only — no expression text or parameter value is ever
    # surfaced, satisfying the licensed-content constraint.
    assert total > 100_000, total
    assert mismatches == 0, (
        f"{mismatches} rust/python disagreements across {total} expressions "
        f"({value_matches} value matches, {raise_matches} identical raises)"
    )
    assert value_matches > 100_000, value_matches
    assert worst_relative_error <= REL_TOLERANCE, worst_relative_error
