"""Small, dependency-free parser for SPICE/HSPICE compact-model libraries.

This module deliberately stops at syntax.  It turns a model delivery into an
in-memory AST without evaluating foundry expressions or implementing a compact
model.  Keeping those jobs separate is important:

* the parser handles continuations, library sections, model cards and subcircuits;
* a future elaborator resolves parameters, binning and subcircuit instances;
* an OSDI or native BSIM implementation evaluates the resulting flat model card.

No source text or parsed parameter data is written to disk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping


_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.$]*")
_ASSIGNMENT = re.compile(
    r"([A-Za-z_][A-Za-z0-9_.$]*)"
    r"(?:\s*\(([^()]*)\))?"
    r"\s*=(?!=)"
)
_NUMBER = re.compile(
    r"^[ \t]*"
    r"(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<suffix>meg|mil|[tgkmunpf])?"
    r"(?P<unit>[A-Za-z]*)"
    r"[ \t]*$",
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


class SpiceSyntaxError(ValueError):
    """Raised when structural SPICE syntax is malformed."""


@dataclass(frozen=True)
class SourceLocation:
    path: str
    first_line: int
    last_line: int


@dataclass(frozen=True)
class ParameterAssignment:
    name: str
    expression: str
    formal_parameters: tuple[str, ...] = ()

    @property
    def is_function(self) -> bool:
        return bool(self.formal_parameters)


@dataclass(frozen=True)
class Statement:
    """One logical SPICE statement.

    ``kind`` is lower-case and does not include the leading dot for directives.
    Element statements use their lower-case first character (``m``, ``x``, ...).
    """

    kind: str
    location: SourceLocation
    text: str
    name: str | None = None
    arguments: tuple[str, ...] = ()
    parameters: tuple[ParameterAssignment, ...] = ()

    @property
    def parameter_map(self) -> Mapping[str, str]:
        return {item.name.lower(): item.expression for item in self.parameters}


@dataclass
class Subcircuit:
    name: str
    location: SourceLocation
    terminals: tuple[str, ...]
    parameters: tuple[ParameterAssignment, ...] = ()
    statements: list[Statement] = field(default_factory=list)


@dataclass
class LibrarySection:
    name: str
    location: SourceLocation
    statements: list[Statement] = field(default_factory=list)
    subcircuits: dict[str, Subcircuit] = field(default_factory=dict)

    @property
    def models(self) -> dict[str, Statement]:
        return {
            statement.name.lower(): statement
            for statement in self.statements
            if statement.kind == "model" and statement.name is not None
        }


@dataclass
class SpiceModelLibrary:
    path: str
    top_level: LibrarySection
    sections: dict[str, LibrarySection]

    def section(self, name: str) -> LibrarySection:
        try:
            return self.sections[name.lower()]
        except KeyError as exc:
            raise KeyError(
                f"unknown library section {name!r}; available sections: "
                f"{', '.join(sorted(self.sections))}"
            ) from exc

    def selected_statements(self, names: Iterable[str]) -> Iterator[Statement]:
        """Yield statements from *names* in the requested order."""
        for name in names:
            yield from self.section(name).statements


def parse_spice_number(value: str) -> float:
    """Parse a SPICE number, including the non-SI ``m`` and ``meg`` suffixes.

    Trailing unit text is ignored as SPICE does (``1kOhm`` equals ``1k``).
    """
    match = _NUMBER.match(value)
    if not match:
        raise ValueError(f"not a SPICE numeric literal: {value!r}")
    suffix = (match.group("suffix") or "").lower()
    return float(match.group("number")) * _SUFFIX[suffix]


def _strip_inline_comment(line: str) -> str:
    """Remove an HSPICE ``$`` comment outside quotes."""
    quote = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "$":
            return line[:index]
    return line


def logical_lines(text: str, *, path: str = "<string>") -> Iterator[tuple[str, SourceLocation]]:
    """Join SPICE ``+`` continuation records and attach source locations."""
    current: list[str] = []
    first = 0
    last = 0

    def flush():
        nonlocal current
        if not current:
            return None
        joined = " ".join(part.strip() for part in current if part.strip())
        current = []
        return joined, SourceLocation(path, first, last)

    for line_number, physical in enumerate(text.splitlines(), 1):
        line = _strip_inline_comment(physical).rstrip()
        stripped = line.lstrip()
        if not stripped or stripped.startswith("*"):
            continue
        if stripped.startswith("+"):
            if not current:
                raise SpiceSyntaxError(
                    f"{path}:{line_number}: continuation without a previous statement")
            current.append(stripped[1:])
            last = line_number
            continue
        pending = flush()
        if pending is not None:
            yield pending
        current = [line]
        first = last = line_number
    pending = flush()
    if pending is not None:
        yield pending


def _balanced_outer_parentheses(text: str) -> bool:
    if not (text.startswith("(") and text.endswith(")")):
        return False
    depth = 0
    quote = None
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
            if depth < 0:
                return False
    return depth == 0 and quote is None


def _assignment_starts(text: str) -> list[tuple[int, int, str, tuple[str, ...]]]:
    """Top-level ``name=`` spans, ignoring function arguments and quoted text."""
    starts = []
    depth = 0
    quote = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char in "({[":
            depth += 1
            index += 1
            continue
        if char in ")}]":
            depth = max(depth - 1, 0)
            index += 1
            continue
        if depth == 0:
            match = _ASSIGNMENT.match(text, index)
            if match:
                formals = tuple(
                    item.strip()
                    for item in (match.group(2) or "").split(",")
                    if item.strip()
                )
                starts.append((match.start(), match.end(), match.group(1), formals))
                index = match.end()
                continue
        index += 1
    return starts


def parse_assignments(text: str) -> tuple[ParameterAssignment, ...]:
    """Parse a whitespace-separated HSPICE assignment list.

    Expressions remain strings.  Nested calls, braces and quoted expressions are
    kept intact for the later expression-elaboration stage.
    """
    body = text.strip()
    while _balanced_outer_parentheses(body):
        body = body[1:-1].strip()
    starts = _assignment_starts(body)
    assignments = []
    for index, (begin, value_begin, name, formals) in enumerate(starts):
        value_end = starts[index + 1][0] if index + 1 < len(starts) else len(body)
        expression = body[value_begin:value_end].strip().rstrip(",")
        if not expression:
            raise SpiceSyntaxError(f"missing value for parameter {name!r}")
        assignments.append(ParameterAssignment(
            name=name,
            expression=expression,
            formal_parameters=formals,
        ))
    return tuple(assignments)


def _split_words(text: str) -> tuple[str, ...]:
    """Split words while preserving quoted strings and parenthesized expressions."""
    words = []
    start = None
    depth = 0
    quote = None
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            if start is None:
                start = index
        elif char in "({[":
            depth += 1
            if start is None:
                start = index
        elif char in ")}]":
            depth = max(depth - 1, 0)
        elif char.isspace() and depth == 0:
            if start is not None:
                words.append(text[start:index])
                start = None
        elif start is None:
            start = index
    if start is not None:
        words.append(text[start:])
    return tuple(words)


def _statement(text: str, location: SourceLocation) -> Statement:
    stripped = text.strip()
    if stripped.startswith("."):
        words = _split_words(stripped[1:])
        kind = words[0].lower()
        rest = stripped[1 + len(words[0]):].strip()
        if kind == "model":
            head = _split_words(rest)
            if len(head) < 2:
                raise SpiceSyntaxError(
                    f"{location.path}:{location.first_line}: malformed .model")
            name, model_type = head[:2]
            offset = rest.find(model_type) + len(model_type)
            return Statement(
                kind=kind,
                location=location,
                text=text,
                name=name,
                arguments=(model_type,),
                parameters=parse_assignments(rest[offset:]),
            )
        if kind == "param":
            return Statement(
                kind=kind,
                location=location,
                text=text,
                parameters=parse_assignments(rest),
            )
        return Statement(kind=kind, location=location, text=text, arguments=words[1:])

    words = _split_words(stripped)
    if not words or not _NAME.match(words[0]):
        raise SpiceSyntaxError(
            f"{location.path}:{location.first_line}: malformed element statement")
    name = words[0]
    return Statement(
        kind=name[0].lower(),
        location=location,
        text=text,
        name=name,
        arguments=words[1:],
        parameters=parse_assignments(stripped[len(name):]),
    )


def _subckt_header(statement: Statement) -> tuple[str, tuple[str, ...],
                                                  tuple[ParameterAssignment, ...]]:
    if not statement.arguments:
        raise SpiceSyntaxError(
            f"{statement.location.path}:{statement.location.first_line}: "
            "missing .subckt name")
    name = statement.arguments[0]
    tail = " ".join(statement.arguments[1:])
    assignments = parse_assignments(tail)
    starts = _assignment_starts(tail)
    terminal_text = tail[:starts[0][0]] if starts else tail
    terminals = tuple(
        word for word in _split_words(terminal_text)
        if word.lower() not in {"params:", "param:"}
    )
    return name, terminals, assignments


def parse_spice_library_text(text: str, *, path: str = "<string>") -> SpiceModelLibrary:
    """Parse a SPICE/HSPICE model library from *text*."""
    top = LibrarySection(
        name="<top>",
        location=SourceLocation(path, 1, max(len(text.splitlines()), 1)),
    )
    sections: dict[str, LibrarySection] = {}
    current = top
    current_subckt: Subcircuit | None = None

    for raw, location in logical_lines(text, path=path):
        statement = _statement(raw, location)

        if statement.kind == "lib" and len(statement.arguments) == 1:
            if current is not top:
                raise SpiceSyntaxError(
                    f"{path}:{location.first_line}: nested .lib section")
            name = statement.arguments[0].strip("'\"").lower()
            if name in sections:
                raise SpiceSyntaxError(
                    f"{path}:{location.first_line}: duplicate .lib section {name!r}")
            current = LibrarySection(name=name, location=location)
            sections[name] = current
            continue
        if statement.kind == "endl":
            if current is top:
                raise SpiceSyntaxError(
                    f"{path}:{location.first_line}: .endl outside a .lib section")
            if current_subckt is not None:
                raise SpiceSyntaxError(
                    f"{path}:{location.first_line}: .endl inside .subckt "
                    f"{current_subckt.name!r}")
            current = top
            continue
        if statement.kind == "subckt":
            if current_subckt is not None:
                raise SpiceSyntaxError(
                    f"{path}:{location.first_line}: nested .subckt")
            name, terminals, parameters = _subckt_header(statement)
            key = name.lower()
            current_subckt = Subcircuit(
                name=name,
                location=location,
                terminals=terminals,
                parameters=parameters,
            )
            current.subcircuits[key] = current_subckt
            continue
        if statement.kind == "ends":
            if current_subckt is None:
                raise SpiceSyntaxError(
                    f"{path}:{location.first_line}: .ends outside a .subckt")
            current_subckt = None
            continue

        if current_subckt is not None:
            current_subckt.statements.append(statement)
        else:
            current.statements.append(statement)

    if current_subckt is not None:
        raise SpiceSyntaxError(f"{path}: unterminated .subckt {current_subckt.name!r}")
    if current is not top:
        raise SpiceSyntaxError(f"{path}: unterminated .lib section {current.name!r}")
    return SpiceModelLibrary(path=path, top_level=top, sections=sections)


def parse_spice_library(path: str | Path) -> SpiceModelLibrary:
    """Parse *path* without copying or caching any model content."""
    source = Path(path)
    return parse_spice_library_text(
        source.read_text(encoding="ascii", errors="strict"),
        path=str(source),
    )
