"""In-process elaboration of the licensed TSMC28HPC+ core MOS library.

The implementation reads the original local HSPICE delivery and returns one
flat numeric BSIM4.5 card plus its numeric instance parameters. It never starts
ngspice and never writes foundry parameter data to disk.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from ...spice import (
    ElaboratedLibrary,
    SpiceElaborationError,
    Statement,
    elaborate_library,
    parse_spice_library,
)
from ...toolchain import tsmc28_model_dir


TSMC28_CORE_CORNERS = ("tt", "ss", "ff", "sf", "fs")
_MODEL_FILE = "cln28hpcp_1d8_elk_v1d0_2p2.l"
_COMMON_SECTIONS = ("setup", "global", "total", "stat")
_MACRO = {"nmos": "nch_mac", "pmos": "pch_mac"}
_MODEL_TYPE = {"nmos": "nmos", "pmos": "pmos"}
_BIN_PARAMETERS = ("lmin", "lmax", "wmin", "wmax")


class Tsmc28ModelError(ValueError):
    """The local delivery cannot produce the requested core-MOS card."""


@dataclass(frozen=True)
class Tsmc28CoreCard:
    """One fully numeric TSMC core-MOS model/instance pair."""

    polarity: str
    corner: str
    temperature_c: float
    macro_name: str
    bin_name: str
    model_type: str
    model_parameters: dict[str, float]
    instance_parameters: dict[str, float]

    @property
    def width_m(self) -> float:
        return self.instance_parameters["w"]

    @property
    def length_m(self) -> float:
        return self.instance_parameters["l"]

    def to_bsim4_cards(self):
        """Convert to the simulator-neutral BSIM4 ABI cards.

        HSPICE's instance-only ``mulu0`` extension is folded into a private copy
        of the selected model's ``u0`` parameter before the standard BSIM4 card is
        constructed.
        """
        from ...compact_models.bsim4 import Bsim4InstanceCard, Bsim4ModelCard

        model_parameters = dict(self.model_parameters)
        instance_parameters = dict(self.instance_parameters)
        mobility_multiplier = instance_parameters.pop("mulu0", 1.0)
        if mobility_multiplier != 1.0:
            if "u0" not in model_parameters:
                raise Tsmc28ModelError(
                    "mulu0 is non-unity but the selected BSIM4 card has no u0")
            model_parameters["u0"] *= mobility_multiplier
        return (
            Bsim4ModelCard(
                polarity=1 if self.polarity == "nmos" else -1,
                parameters=model_parameters,
                version=4.5,
            ),
            Bsim4InstanceCard(instance_parameters),
        )


def _normalize_corner(corner: str) -> str:
    key = str(corner).lower()
    if key == "nom":
        key = "tt"
    if key not in TSMC28_CORE_CORNERS:
        raise Tsmc28ModelError(
            f"unknown TSMC28 core corner {corner!r}; expected one of "
            f"{TSMC28_CORE_CORNERS}")
    return key


def _normalize_polarity(polarity: str) -> str:
    key = str(polarity).lower()
    aliases = {"n": "nmos", "nfet": "nmos", "p": "pmos", "pfet": "pmos"}
    key = aliases.get(key, key)
    if key not in _MACRO:
        raise Tsmc28ModelError(
            f"unknown TSMC28 core polarity {polarity!r}; expected nmos or pmos")
    return key


def _model_path(path: str | None) -> str:
    candidate = path or os.path.join(tsmc28_model_dir(), _MODEL_FILE)
    candidate = os.path.abspath(os.path.expanduser(candidate))
    if not os.path.isfile(candidate):
        raise Tsmc28ModelError(
            f"TSMC28HPC+ HSPICE model not found: {candidate}; set "
            "TSMC28_MODEL_DIR/TSMC28_PDK_ROOT or install the local model")
    return candidate


class Tsmc28CoreLibrary:
    """Parsed model delivery with cached per-corner/temperature elaboration."""

    def __init__(self, path: str | None = None):
        self.path = _model_path(path)
        self._library = parse_spice_library(self.path)
        self._programs: dict[tuple[str, float], ElaboratedLibrary] = {}
        self._lock = threading.RLock()

    def _program(self, corner: str, temperature_c: float) -> ElaboratedLibrary:
        key = (_normalize_corner(corner), float(temperature_c))
        with self._lock:
            program = self._programs.get(key)
            if program is None:
                sections = (
                    _COMMON_SECTIONS[0],
                    key[0],
                    *_COMMON_SECTIONS[1:],
                )
                program = elaborate_library(
                    self._library,
                    sections,
                    initial_values={"temper": key[1]},
                )
                self._programs[key] = program
            return program

    @staticmethod
    def _select_bin(instance, width_m: float, length_m: float) -> Statement:
        candidates = []
        inclusive_candidates = []
        for statement in instance.model_statements:
            bounds = instance.numeric_model(
                statement, names=_BIN_PARAMETERS).parameters
            half_open = (
                bounds["lmin"] <= length_m < bounds["lmax"]
                and bounds["wmin"] <= width_m < bounds["wmax"]
            )
            inclusive = (
                bounds["lmin"] <= length_m <= bounds["lmax"]
                and bounds["wmin"] <= width_m <= bounds["wmax"]
            )
            if half_open:
                candidates.append(statement)
            if inclusive:
                inclusive_candidates.append(statement)
        if not candidates and len(inclusive_candidates) == 1:
            candidates = inclusive_candidates
        if len(candidates) != 1:
            names = [statement.name for statement in candidates]
            raise Tsmc28ModelError(
                f"geometry W={width_m:g} m L={length_m:g} m selects "
                f"{len(candidates)} bins ({names}); expected exactly one")
        return candidates[0]

    def core_card(
        self,
        polarity: str,
        *,
        width_um: float,
        length_um: float,
        nf: int = 1,
        mult: int = 1,
        corner: str = "tt",
        temperature_c: float = 27.0,
        mismatch_v: float = 0.0,
    ) -> Tsmc28CoreCard:
        """Return a flat numeric BSIM4.5 card for one core MOS instance."""
        polarity = _normalize_polarity(polarity)
        corner = _normalize_corner(corner)
        width_um = float(width_um)
        length_um = float(length_um)
        nf = int(nf)
        mult = int(mult)
        if width_um <= 0 or length_um <= 0:
            raise Tsmc28ModelError("core MOS width and length must be positive")
        if nf < 1 or mult < 1:
            raise Tsmc28ModelError("core MOS nf and mult must be positive integers")

        program = self._program(corner, temperature_c)
        macro_name = _MACRO[polarity]
        try:
            instance = program.instantiate(
                macro_name,
                {
                    "w": width_um * 1e-6,
                    "l": length_um * 1e-6,
                    "nf": nf,
                    "multi": mult,
                    "_delvto": float(mismatch_v),
                },
            )
        except SpiceElaborationError as exc:
            raise Tsmc28ModelError(str(exc)) from exc

        mos_elements = [statement for statement in instance.elements if statement.kind == "m"]
        unsupported = [statement.kind for statement in instance.elements if statement.kind != "m"]
        if len(mos_elements) != 1 or unsupported:
            raise Tsmc28ModelError(
                f"{macro_name} must expand to exactly one MOS and no other active "
                f"elements; mos={len(mos_elements)}, unsupported={unsupported}")
        element = mos_elements[0]
        instance_parameters = instance.numeric_parameters(element)
        selected = self._select_bin(
            instance,
            # BSIM4 bins on effective width per finger. The instance ``w`` is
            # total width and ``nf`` is carried separately into the compact model.
            instance_parameters["w"] / instance_parameters["nf"],
            instance_parameters["l"],
        )
        numeric = instance.numeric_model(selected)
        if numeric.model_type.lower() != _MODEL_TYPE[polarity]:
            raise Tsmc28ModelError(
                f"{macro_name} selected {numeric.model_type!r}, expected "
                f"{_MODEL_TYPE[polarity]!r}")
        if int(numeric.parameters.get("level", -1)) != 54:
            raise Tsmc28ModelError(
                f"{selected.name} is not a BSIM4 level-54 model")
        version = numeric.parameters.get("version")
        if version not in {4.5, 4.50}:
            raise Tsmc28ModelError(
                f"{selected.name} uses unsupported BSIM4 version {version!r}")
        return Tsmc28CoreCard(
            polarity=polarity,
            corner=corner,
            temperature_c=float(temperature_c),
            macro_name=macro_name,
            bin_name=selected.name or "",
            model_type=numeric.model_type.lower(),
            model_parameters=numeric.parameters,
            instance_parameters=instance_parameters,
        )


_LIBRARIES: dict[tuple[str, int, int], Tsmc28CoreLibrary] = {}
_LIBRARIES_LOCK = threading.Lock()


def load_tsmc28_core_library(path: str | None = None) -> Tsmc28CoreLibrary:
    """Load/cache the local library by absolute path, mtime and size."""
    resolved = _model_path(path)
    stat = os.stat(resolved)
    key = (resolved, stat.st_mtime_ns, stat.st_size)
    with _LIBRARIES_LOCK:
        library = _LIBRARIES.get(key)
        if library is None:
            library = Tsmc28CoreLibrary(resolved)
            _LIBRARIES.clear()
            _LIBRARIES[key] = library
        return library
