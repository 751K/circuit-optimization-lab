"""Shared, in-memory fixtures for full licensed-deck parity tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True, slots=True)
class TsmcParityDeck:
    path: str
    library: Any
    expressions: tuple[str, ...]


@pytest.fixture(scope="session")
def tsmc_parity_deck() -> TsmcParityDeck:
    """Parse the licensed deck once; consumers must treat the AST as read-only."""
    from circuitopt.spice import parse_spice_library
    from circuitopt.toolchain import tsmc28_model_dir

    path = Path(tsmc28_model_dir()) / "cln28hpcp_1d8_elk_v1d0_2p2.l"
    if not path.is_file():
        pytest.skip("licensed TSMC28HPC+ model is not installed")

    library = parse_spice_library(str(path))
    expressions: list[str] = []
    for section in library.sections.values():
        statements = list(section.statements)
        for subcircuit in section.subcircuits.values():
            statements.extend(subcircuit.statements)
            expressions.extend(
                parameter.expression for parameter in subcircuit.parameters)
        for statement in statements:
            expressions.extend(
                parameter.expression for parameter in statement.parameters)

    return TsmcParityDeck(
        path=str(path),
        library=library,
        expressions=tuple(expressions),
    )
