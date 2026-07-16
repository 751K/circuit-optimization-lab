"""Deterministic, sandboxed evaluator for HSPICE parameter expressions.

The implementation uses a small Pratt parser rather than Python ``eval``.  That
keeps foundry model text out of the Python execution environment and gives
precise control over HSPICE number suffixes, case-insensitive symbols and
simulator-specific functions.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Mapping, Protocol


class SpiceExpressionError(ValueError):
    """Base class for expression parse/evaluation failures."""


class UnknownSymbolError(SpiceExpressionError):
    """An expression referenced a symbol or function absent from its scope."""


class ParameterCycleError(SpiceExpressionError):
    """A lazy parameter dependency contains a cycle."""


_TOKEN_NUMBER = re.compile(
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"(?:meg|mil|[tgkmunpf])?(?:[A-Za-z]+)?",
    re.IGNORECASE,
)
_TOKEN_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.$]*")
_SPICE_NUMBER = re.compile(
    r"^(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<suffix>meg|mil|[tgkmunpf])?"
    r"(?P<unit>[A-Za-z]*)$",
    re.IGNORECASE,
)
_SUFFIX = {
    "": 1.0,
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "mil": 25.4e-6,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}
_TWO_CHAR_OPERATORS = {"**", "<=", ">=", "==", "!=", "&&", "||"}
_ONE_CHAR_OPERATORS = set("+-*/^(),?:<>!")


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    offset: int


def _strip_expression_quotes(expression: str) -> str:
    body = expression.strip()
    if len(body) >= 2 and body[0] == body[-1] and body[0] in {"'", '"'}:
        return body[1:-1].strip()
    return body


def tokenize(expression: str) -> tuple[Token, ...]:
    """Tokenize the supported HSPICE expression subset."""
    body = _strip_expression_quotes(expression)
    tokens = []
    index = 0
    while index < len(body):
        if body[index].isspace():
            index += 1
            continue
        pair = body[index:index + 2]
        if pair in _TWO_CHAR_OPERATORS:
            tokens.append(Token("op", pair, index))
            index += 2
            continue
        char = body[index]
        if char in _ONE_CHAR_OPERATORS:
            tokens.append(Token("op", char, index))
            index += 1
            continue
        number = _TOKEN_NUMBER.match(body, index)
        if number:
            tokens.append(Token("number", number.group(0), index))
            index = number.end()
            continue
        identifier = _TOKEN_IDENT.match(body, index)
        if identifier:
            tokens.append(Token("identifier", identifier.group(0), index))
            index = identifier.end()
            continue
        raise SpiceExpressionError(
            f"unsupported character {body[index]!r} at offset {index} in "
            f"{expression!r}")
    tokens.append(Token("eof", "", len(body)))
    return tuple(tokens)


def _number(text: str) -> float:
    match = _SPICE_NUMBER.fullmatch(text)
    if not match:
        raise SpiceExpressionError(f"invalid SPICE number {text!r}")
    suffix = (match.group("suffix") or "").lower()
    return float(match.group("number")) * _SUFFIX[suffix]


class SymbolResolver(Protocol):
    def resolve_symbol(self, name: str) -> float:
        ...

    def call_function(self, name: str, arguments: tuple[float, ...]) -> float:
        ...


class Expression:
    def evaluate(self, resolver: SymbolResolver) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class Number(Expression):
    value: float

    def evaluate(self, resolver: SymbolResolver) -> float:
        del resolver
        return self.value


@dataclass(frozen=True)
class Name(Expression):
    name: str

    def evaluate(self, resolver: SymbolResolver) -> float:
        return float(resolver.resolve_symbol(self.name))


@dataclass(frozen=True)
class Unary(Expression):
    operator: str
    operand: Expression

    def evaluate(self, resolver: SymbolResolver) -> float:
        value = self.operand.evaluate(resolver)
        if self.operator == "+":
            return value
        if self.operator == "-":
            return -value
        if self.operator == "!":
            return float(not bool(value))
        raise SpiceExpressionError(f"unsupported unary operator {self.operator!r}")


@dataclass(frozen=True)
class Binary(Expression):
    operator: str
    left: Expression
    right: Expression

    def evaluate(self, resolver: SymbolResolver) -> float:
        left = self.left.evaluate(resolver)
        if self.operator == "&&":
            return float(bool(left) and bool(self.right.evaluate(resolver)))
        if self.operator == "||":
            return float(bool(left) or bool(self.right.evaluate(resolver)))
        right = self.right.evaluate(resolver)
        if self.operator == "+":
            return left + right
        if self.operator == "-":
            return left - right
        if self.operator == "*":
            return left * right
        if self.operator == "/":
            return left / right
        if self.operator in {"^", "**"}:
            return left ** right
        if self.operator == "<":
            return float(left < right)
        if self.operator == "<=":
            return float(left <= right)
        if self.operator == ">":
            return float(left > right)
        if self.operator == ">=":
            return float(left >= right)
        if self.operator == "==":
            return float(left == right)
        if self.operator == "!=":
            return float(left != right)
        raise SpiceExpressionError(f"unsupported binary operator {self.operator!r}")


@dataclass(frozen=True)
class Conditional(Expression):
    condition: Expression
    when_true: Expression
    when_false: Expression

    def evaluate(self, resolver: SymbolResolver) -> float:
        selected = self.when_true if bool(self.condition.evaluate(resolver)) else self.when_false
        return selected.evaluate(resolver)


@dataclass(frozen=True)
class Call(Expression):
    name: str
    arguments: tuple[Expression, ...]

    def evaluate(self, resolver: SymbolResolver) -> float:
        values = tuple(argument.evaluate(resolver) for argument in self.arguments)
        return float(resolver.call_function(self.name, values))


_BINDING_POWER = {
    "||": 10,
    "&&": 20,
    "==": 30,
    "!=": 30,
    "<": 30,
    "<=": 30,
    ">": 30,
    ">=": 30,
    "+": 40,
    "-": 40,
    "*": 50,
    "/": 50,
    "^": 60,
    "**": 60,
}


class _Parser:
    def __init__(self, tokens: tuple[Token, ...], source: str):
        self.tokens = tokens
        self.source = source
        self.index = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def accept(self, text: str) -> bool:
        if self.current.text == text:
            self.index += 1
            return True
        return False

    def expect(self, text: str) -> None:
        if not self.accept(text):
            raise SpiceExpressionError(
                f"expected {text!r} at offset {self.current.offset} in {self.source!r}")

    def parse(self) -> Expression:
        expression = self.parse_conditional()
        if self.current.kind != "eof":
            raise SpiceExpressionError(
                f"unexpected token {self.current.text!r} at offset "
                f"{self.current.offset} in {self.source!r}")
        return expression

    def parse_conditional(self) -> Expression:
        condition = self.parse_binary(0)
        if not self.accept("?"):
            return condition
        when_true = self.parse_conditional()
        self.expect(":")
        when_false = self.parse_conditional()
        return Conditional(condition, when_true, when_false)

    def parse_binary(self, minimum_binding: int) -> Expression:
        left = self.parse_prefix()
        while True:
            operator = self.current.text
            binding = _BINDING_POWER.get(operator)
            if binding is None or binding < minimum_binding:
                return left
            self.index += 1
            # Power is right-associative; all other operators are left-associative.
            next_binding = binding if operator in {"^", "**"} else binding + 1
            right = self.parse_binary(next_binding)
            left = Binary(operator, left, right)

    def parse_prefix(self) -> Expression:
        if self.current.text in {"+", "-", "!"}:
            operator = self.current.text
            self.index += 1
            return Unary(operator, self.parse_prefix())
        if self.accept("("):
            expression = self.parse_conditional()
            self.expect(")")
            return expression
        token = self.current
        if token.kind == "number":
            self.index += 1
            return Number(_number(token.text))
        if token.kind == "identifier":
            self.index += 1
            name = token.text
            if not self.accept("("):
                return Name(name)
            arguments = []
            if not self.accept(")"):
                while True:
                    arguments.append(self.parse_conditional())
                    if self.accept(")"):
                        break
                    self.expect(",")
            return Call(name, tuple(arguments))
        raise SpiceExpressionError(
            f"expected expression at offset {token.offset} in {self.source!r}")


@lru_cache(maxsize=32768)
def compile_expression(expression: str) -> Expression:
    """Compile one HSPICE expression to a reusable immutable expression tree."""
    return _Parser(tokenize(expression), expression).parse()


def _require_arity(name: str, arguments: tuple[float, ...], count: int) -> None:
    if len(arguments) != count:
        raise SpiceExpressionError(
            f"{name} expects {count} arguments, received {len(arguments)}")


def _builtin(name: str, arguments: tuple[float, ...]) -> float:
    key = name.lower()
    if key == "abs":
        _require_arity(name, arguments, 1)
        return abs(arguments[0])
    if key == "sqrt":
        _require_arity(name, arguments, 1)
        return math.sqrt(arguments[0])
    if key == "exp":
        _require_arity(name, arguments, 1)
        return math.exp(arguments[0])
    if key == "log10":
        _require_arity(name, arguments, 1)
        return math.log10(arguments[0])
    if key == "int":
        _require_arity(name, arguments, 1)
        return float(math.trunc(arguments[0]))
    if key == "sgn":
        _require_arity(name, arguments, 1)
        return float((arguments[0] > 0) - (arguments[0] < 0))
    if key in {"max", "min"}:
        if not arguments:
            raise SpiceExpressionError(f"{name} expects at least one argument")
        return float(max(arguments) if key == "max" else min(arguments))
    if key == "selmin":
        if not arguments:
            raise SpiceExpressionError("selmin expects at least one argument")
        return float(min(arguments))
    if key in {"pwr", "pow"}:
        _require_arity(name, arguments, 2)
        return arguments[0] ** arguments[1]
    if key == "agauss":
        if len(arguments) not in {2, 3}:
            raise SpiceExpressionError(
                f"agauss expects 2 or 3 arguments, received {len(arguments)}")
        # Native PVT elaboration is deterministic nominal evaluation. A future
        # mismatch engine may replace this policy with a seeded random resolver.
        return arguments[0]
    raise UnknownSymbolError(f"unknown HSPICE function {name!r}")


class EvaluationScope(SymbolResolver):
    """Case-insensitive lazy parameter scope with user-defined functions."""

    def __init__(
        self,
        *,
        parent: "EvaluationScope | None" = None,
        values: Mapping[str, float] | None = None,
        functions: Mapping[str, Callable[[tuple[float, ...]], float]] | None = None,
    ):
        self.parent = parent
        self._expressions: dict[str, str] = {}
        self._function_defs: dict[str, tuple[tuple[str, ...], str]] = {}
        self._values = {str(name).lower(): float(value) for name, value in (values or {}).items()}
        self._functions = {str(name).lower(): function for name, function in (functions or {}).items()}
        self._resolving: list[str] = []

    def define(self, name: str, expression: str) -> None:
        key = name.lower()
        self._expressions[key] = expression
        self._values.pop(key, None)

    def set_value(self, name: str, value: float) -> None:
        key = name.lower()
        self._values[key] = float(value)
        self._expressions.pop(key, None)

    def define_function(
        self,
        name: str,
        formal_parameters: tuple[str, ...],
        expression: str,
    ) -> None:
        self._function_defs[name.lower()] = (
            tuple(parameter.lower() for parameter in formal_parameters),
            expression,
        )

    def resolve_symbol(self, name: str) -> float:
        key = name.lower()
        if key in self._values:
            return self._values[key]
        if key in self._expressions:
            if key in self._resolving:
                # SPICE instance/model overrides commonly use ``w=w`` or
                # ``param=param+delta``: the RHS refers to the enclosing scope.
                # Fall back lexically before declaring a real local cycle.
                if self.parent is not None:
                    try:
                        return self.parent.resolve_symbol(name)
                    except UnknownSymbolError:
                        pass
                first = self._resolving.index(key)
                cycle = self._resolving[first:] + [key]
                raise ParameterCycleError(
                    "parameter dependency cycle: " + " -> ".join(cycle))
            self._resolving.append(key)
            try:
                value = compile_expression(self._expressions[key]).evaluate(self)
            except SpiceExpressionError:
                raise
            except (ArithmeticError, ValueError, OverflowError) as exc:
                raise SpiceExpressionError(
                    f"failed to evaluate parameter {name!r}: {exc}") from exc
            finally:
                self._resolving.pop()
            if not math.isfinite(value):
                raise SpiceExpressionError(
                    f"parameter {name!r} evaluated to non-finite value {value}")
            self._values[key] = float(value)
            return float(value)
        if self.parent is not None:
            return self.parent.resolve_symbol(name)
        constants = {"pi": math.pi, "e": math.e, "true": 1.0, "false": 0.0}
        if key in constants:
            return constants[key]
        raise UnknownSymbolError(f"unknown HSPICE symbol {name!r}")

    def call_function(self, name: str, arguments: tuple[float, ...]) -> float:
        key = name.lower()
        if key in self._functions:
            return float(self._functions[key](arguments))
        if key in self._function_defs:
            formals, expression = self._function_defs[key]
            if len(arguments) != len(formals):
                raise SpiceExpressionError(
                    f"{name} expects {len(formals)} arguments, received {len(arguments)}")
            child = EvaluationScope(parent=self, values=dict(zip(formals, arguments)))
            return float(compile_expression(expression).evaluate(child))
        if self.parent is not None:
            try:
                return self.parent.call_function(name, arguments)
            except UnknownSymbolError:
                pass
        return _builtin(name, arguments)

    def evaluate(self, expression: str) -> float:
        try:
            value = float(compile_expression(expression).evaluate(self))
        except SpiceExpressionError:
            raise
        except (ArithmeticError, ValueError, OverflowError) as exc:
            raise SpiceExpressionError(
                f"failed to evaluate expression {expression!r}: {exc}") from exc
        if not math.isfinite(value):
            raise SpiceExpressionError(
                f"expression {expression!r} evaluated to non-finite value {value}")
        return value

    def evaluate_all(self) -> dict[str, float]:
        for name in tuple(self._expressions):
            self.resolve_symbol(name)
        return dict(self._values)
