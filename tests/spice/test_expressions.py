"""HSPICE expression compiler and lazy-scope tests."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from circuitopt.spice import (
    EvaluationScope,
    ParameterCycleError,
    SpiceExpressionError,
    UnknownSymbolError,
    compile_expression,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", 7.0),
        ("2**3**2", 512.0),
        ("2.5k + 500", 3000.0),
        ("1 ? 2 : 3", 2.0),
        ("0 ? 2 : 3", 3.0),
        ("-2 < -1 ? max(4, 3) : 0", 4.0),
        ("sgn(-3) + int(2.9)", 1.0),
        ("pwr(3, 2) + sqrt(16)", 13.0),
        ("agauss(5, 2, 3)", 5.0),
    ],
)
def test_expression_evaluation(expression, expected):
    assert EvaluationScope().evaluate(expression) == pytest.approx(expected)


def test_lazy_case_insensitive_parameters_and_forward_reference():
    scope = EvaluationScope(values={"W": 2e-6})
    scope.define("area", "w * length")
    scope.define("LENGTH", "30n")
    assert scope.resolve_symbol("AREA") == pytest.approx(60e-15)


def test_user_defined_parameter_function_uses_lexical_parent():
    scope = EvaluationScope(values={"offset": 2.0})
    scope.define_function("shift", ("x", "gain"), "x * gain + offset")
    assert scope.evaluate("SHIFT(3, 4)") == pytest.approx(14.0)


def test_cycle_unknown_function_and_nonfinite_fail_loudly():
    scope = EvaluationScope()
    scope.define("a", "b + 1")
    scope.define("b", "a + 1")
    with pytest.raises(ParameterCycleError, match="a -> b -> a"):
        scope.resolve_symbol("a")
    with pytest.raises(UnknownSymbolError, match="missing"):
        EvaluationScope().evaluate("missing + 1")
    with pytest.raises(UnknownSymbolError, match="v"):
        EvaluationScope().evaluate("v(0)")
    with pytest.raises(SpiceExpressionError):
        EvaluationScope().evaluate("1 / 0")


def test_shared_scope_resolves_parameters_concurrently_without_false_cycles():
    scope = EvaluationScope(values={"base": 2.0})
    scope.define("a", "base + 1")
    scope.define("b", "a * 3")
    scope.define("result", "b + a")
    barrier = Barrier(8)

    def resolve_repeatedly(_index):
        barrier.wait()
        return tuple(scope.resolve_symbol("result") for _ in range(100))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(resolve_repeatedly, range(8)))

    assert results == [(12.0,) * 100] * 8


def test_all_real_tsmc_parameter_expressions_compile_when_locally_installed():
    from circuitopt.spice import parse_spice_library
    from circuitopt.toolchain import tsmc28_model_dir

    path = os.path.join(
        tsmc28_model_dir(), "cln28hpcp_1d8_elk_v1d0_2p2.l")
    if not os.path.isfile(path):
        pytest.skip("licensed TSMC28HPC+ model is not installed")

    library = parse_spice_library(path)
    count = 0
    for section in library.sections.values():
        statements = list(section.statements)
        for subcircuit in section.subcircuits.values():
            statements.extend(subcircuit.statements)
            for parameter in subcircuit.parameters:
                compile_expression(parameter.expression)
                count += 1
        for statement in statements:
            for parameter in statement.parameters:
                compile_expression(parameter.expression)
                count += 1
    assert count > 100_000
