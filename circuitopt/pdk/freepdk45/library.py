"""Flat FreePDK45 BSIM4 model-card loading for the native backend."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from ...compact_models.bsim4 import Bsim4InstanceCard, Bsim4ModelCard
from ...spice import parse_spice_library, parse_spice_number
from ...toolchain import pdk_root


FREEPDK45_CORNERS = ("nom", "tt", "ss", "ff", "sf", "fs")
_CORNER_DIRS = {
    "nom": ("nom", "nom"),
    "tt": ("nom", "nom"),
    "ss": ("ss", "ss"),
    "ff": ("ff", "ff"),
    "sf": ("ss", "ff"),
    "fs": ("ff", "ss"),
}
_MODEL_NAME = {"nmos": "NMOS_VTG", "pmos": "PMOS_VTG"}


class Freepdk45ModelError(ValueError):
    """A local FreePDK45 card cannot be loaded as a native BSIM4 device."""


def normalize_corner(corner) -> str:
    """Return a validated, lower-case FreePDK45 process corner."""
    if corner is None or corner == "":
        return "nom"
    key = corner.lower() if isinstance(corner, str) else corner
    if key not in _CORNER_DIRS:
        raise Freepdk45ModelError(
            f"unknown FreePDK45 corner {corner!r}; expected one of "
            f"{sorted(_CORNER_DIRS)}")
    return key


def normalize_polarity(polarity: str) -> str:
    key = str(polarity).lower()
    key = {"n": "nmos", "nfet": "nmos", "p": "pmos", "pfet": "pmos"}.get(
        key, key)
    if key not in _MODEL_NAME:
        raise Freepdk45ModelError(
            f"unknown FreePDK45 polarity {polarity!r}; expected nmos or pmos")
    return key


def corner_card_dir(polarity: str, corner: str) -> str:
    """Return the per-polarity ``models_*`` suffix for a process corner."""
    polarity = normalize_polarity(polarity)
    nmos_dir, pmos_dir = _CORNER_DIRS[normalize_corner(corner)]
    return nmos_dir if polarity == "nmos" else pmos_dir


def freepdk45_card_path(polarity: str, corner: str = "nom") -> str:
    """Resolve a FreePDK45 VTG card without freezing ``PDK_ROOT`` at import."""
    polarity = normalize_polarity(polarity)
    model = _MODEL_NAME[polarity]
    return os.path.join(
        pdk_root(),
        "freepdk45",
        f"models_{corner_card_dir(polarity, corner)}",
        f"{model}.inc",
    )


@dataclass(frozen=True)
class Freepdk45Card:
    polarity: str
    corner: str
    path: str
    model_name: str
    model_parameters: dict[str, float]
    instance_parameters: dict[str, float]
    source_version: float

    def to_bsim4_cards(self):
        return (
            Bsim4ModelCard(
                polarity=1 if self.polarity == "nmos" else -1,
                parameters=self.model_parameters,
                version=self.source_version,
            ),
            Bsim4InstanceCard(self.instance_parameters),
        )


class Freepdk45Library:
    """One parsed flat FreePDK45 model card."""

    def __init__(self, path: str, polarity: str, corner: str):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.polarity = normalize_polarity(polarity)
        self.corner = normalize_corner(corner)
        if not os.path.isfile(self.path):
            raise Freepdk45ModelError(
                f"FreePDK45 model card not found: {self.path}; set PDK_ROOT")

        library = parse_spice_library(self.path)
        statements = list(library.top_level.models.values())
        if len(statements) != 1:
            raise Freepdk45ModelError(
                f"{self.path} must contain exactly one .model statement")
        statement = statements[0]
        expected_name = _MODEL_NAME[self.polarity]
        model_type = statement.arguments[0].lower() if statement.arguments else ""
        if statement.name != expected_name or model_type != self.polarity:
            raise Freepdk45ModelError(
                f"{self.path} defines {statement.name!r}/{model_type!r}, expected "
                f"{expected_name!r}/{self.polarity!r}")

        parameters = {}
        for assignment in statement.parameters:
            try:
                parameters[assignment.name.lower()] = parse_spice_number(
                    assignment.expression)
            except ValueError as exc:
                raise Freepdk45ModelError(
                    f"{self.path} has non-numeric model parameter "
                    f"{assignment.name!r}={assignment.expression!r}") from exc
        if int(parameters.get("level", -1)) != 54:
            raise Freepdk45ModelError(f"{self.path} is not a BSIM4 level-54 card")
        version = float(parameters.get("version", -1))
        if version != 4.0:
            raise Freepdk45ModelError(
                f"{self.path} uses unsupported BSIM4 version {version!r}")
        self.model_parameters = parameters
        self.model_name = expected_name
        self.source_version = version

    def device_card(
        self,
        *,
        width_um: float,
        length_um: float,
        nf: int = 1,
        mult: int = 1,
        mismatch_v: float = 0.0,
        instance_parameters: dict[str, float] | None = None,
    ) -> Freepdk45Card:
        width_um = float(width_um)
        length_um = float(length_um)
        nf = int(nf)
        mult = int(mult)
        if width_um <= 0 or length_um <= 0:
            raise Freepdk45ModelError("FreePDK45 width and length must be positive")
        if nf < 1 or mult < 1:
            raise Freepdk45ModelError(
                "FreePDK45 nf and mult must be positive integers")
        parameters = {
            "w": width_um * 1e-6,
            "l": length_um * 1e-6,
            "nf": nf,
            "m": mult,
        }
        parameters.update({
            str(name).lower(): float(value)
            for name, value in (instance_parameters or {}).items()
        })
        if mismatch_v:
            parameters["delvto"] = float(mismatch_v)
        return Freepdk45Card(
            polarity=self.polarity,
            corner=self.corner,
            path=self.path,
            model_name=self.model_name,
            model_parameters=dict(self.model_parameters),
            instance_parameters=parameters,
            source_version=self.source_version,
        )


_LIBRARIES: dict[tuple[str, int, int], Freepdk45Library] = {}
_LIBRARIES_LOCK = threading.Lock()


def load_freepdk45_library(
    polarity: str,
    corner: str = "nom",
    path: str | None = None,
) -> Freepdk45Library:
    """Load/cache a flat card by path, mtime and size."""
    polarity = normalize_polarity(polarity)
    corner = normalize_corner(corner)
    resolved = os.path.abspath(os.path.expanduser(
        path or freepdk45_card_path(polarity, corner)))
    if not os.path.isfile(resolved):
        raise Freepdk45ModelError(
            f"FreePDK45 model card not found: {resolved}; set PDK_ROOT")
    stat = os.stat(resolved)
    key = (resolved, stat.st_mtime_ns, stat.st_size)
    with _LIBRARIES_LOCK:
        library = _LIBRARIES.get(key)
        if library is None:
            library = Freepdk45Library(resolved, polarity, corner)
            _LIBRARIES[key] = library
        return library
