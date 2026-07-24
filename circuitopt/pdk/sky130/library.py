"""Resolved SKY130 BSIM4.5 card loading for the native C backend."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ...compact_models.bsim4 import (
    Bsim4CardCache,
    Bsim4CardCacheInfo,
    Bsim4InstanceCard,
    Bsim4ModelCard,
    Bsim4SourceFingerprint,
    make_bsim4_card_cache_key,
)


SKY130_CORNERS = ("tt", "ss", "ff", "sf", "fs")
_SUBCKT = {
    "nmos": "sky130_fd_pr__nfet_01v8",
    "pmos": "sky130_fd_pr__pfet_01v8",
}
_BUNDLED_CARD_DIR = Path(__file__).with_name("cards")


class Sky130ModelError(ValueError):
    """A resolved SKY130 card cannot be loaded as a native BSIM4 device."""


def normalize_corner(corner: str | None) -> str:
    key = "tt" if corner in {None, "", "nom"} else str(corner).lower()
    if key not in SKY130_CORNERS:
        raise Sky130ModelError(
            f"unknown SKY130 corner {corner!r}; expected one of {SKY130_CORNERS}")
    return key


def normalize_polarity(polarity: str) -> str:
    key = str(polarity).lower()
    key = {
        "n": "nmos",
        "nfet": "nmos",
        "p": "pmos",
        "pfet": "pmos",
    }.get(key, key)
    if key not in _SUBCKT:
        raise Sky130ModelError(
            f"unknown SKY130 polarity {polarity!r}; expected nmos or pmos")
    return key


def sky130_card_filename(
    polarity: str,
    width_um: float,
    length_um: float,
    corner: str = "tt",
) -> str:
    polarity = normalize_polarity(polarity)
    corner = normalize_corner(corner)
    return (
        f"{_SUBCKT[polarity]}_{corner}_"
        f"W{float(width_um):g}_L{float(length_um):g}.json"
    )


def sky130_card_dirs() -> tuple[Path, ...]:
    """Return external override and bundled card directories in lookup order."""
    paths = []
    override = os.environ.get("SKY130_CARD_DIR")
    if override:
        paths.append(Path(override).expanduser())
    paths.append(_BUNDLED_CARD_DIR)
    return tuple(paths)


def sky130_card_path(
    polarity: str,
    width_um: float,
    length_um: float,
    corner: str = "tt",
) -> Path:
    filename = sky130_card_filename(polarity, width_um, length_um, corner)
    for directory in sky130_card_dirs():
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in sky130_card_dirs())
    raise Sky130ModelError(
        f"resolved SKY130 BSIM4 card {filename!r} was not found in {searched}. "
        "Use a bundled geometry, set SKY130_CARD_DIR, or explicitly generate an "
        "oracle card with circuitopt.sky130_model.extract_sky130_card().")


@dataclass(frozen=True)
class Sky130Card:
    polarity: str
    corner: str
    path: Path
    model_parameters: Mapping[str, float]
    instance_parameters: Mapping[str, float]
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


_MODEL_CACHE: dict[Bsim4SourceFingerprint, Mapping[str, float]] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_CARD_CACHE = Bsim4CardCache[
    tuple[Sky130Card, Bsim4ModelCard, Bsim4InstanceCard]
](maxsize=1024)


def load_sky130_device_cards(
    polarity: str,
    *,
    pdk: str = "sky130",
    model: str | None = None,
    section: str = "inherit",
    bin_selector: str = "auto",
    width_um: float,
    length_um: float,
    nf: int = 1,
    mult: int = 1,
    corner: str = "tt",
    temperature_c: float = 27.0,
    reference_width_um: float | None = None,
    mismatch_v: float = 0.0,
    instance_parameters: dict[str, float] | None = None,
) -> tuple[Sky130Card, Bsim4ModelCard, Bsim4InstanceCard]:
    """Return cached PDK/model/instance cards without an external simulator."""
    polarity = normalize_polarity(polarity)
    pdk = str(pdk).strip().lower()
    if pdk != "sky130":
        raise Sky130ModelError(f"SKY130 card cannot serve PDK {pdk!r}")
    model = normalize_polarity(model or polarity)
    if model != polarity:
        raise Sky130ModelError(
            f"binding model {model!r} conflicts with polarity {polarity!r}")
    corner = normalize_corner(corner)
    section = str(section).strip().lower()
    if not section:
        raise Sky130ModelError("SKY130 section binding must be non-empty")
    if section != "inherit" and normalize_corner(section) != corner:
        raise Sky130ModelError(
            f"binding section {section!r} conflicts with corner {corner!r}")
    bin_selector = str(bin_selector).strip()
    if not bin_selector:
        raise Sky130ModelError("SKY130 bin binding must be non-empty")
    width_um = float(width_um)
    length_um = float(length_um)
    reference_width_um = (
        width_um if reference_width_um is None else float(reference_width_um))
    nf = int(nf)
    mult = int(mult)
    if width_um <= 0 or length_um <= 0 or reference_width_um <= 0:
        raise Sky130ModelError("SKY130 widths and lengths must be positive")
    if nf < 1 or mult < 1:
        raise Sky130ModelError("SKY130 nf and mult must be positive integers")

    path = sky130_card_path(
        polarity, reference_width_um, length_um, corner)
    resolved_bin = path.stem
    if (
        bin_selector.lower() != "auto"
        and bin_selector.lower() != resolved_bin.lower()
    ):
        raise Sky130ModelError(
            f"requested bin {bin_selector!r}, resolved {resolved_bin!r}")
    source = Bsim4SourceFingerprint.from_path(path)
    with _MODEL_CACHE_LOCK:
        model_parameters = _MODEL_CACHE.get(source)
        if model_parameters is None:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                model_parameters = {
                    str(name).lower(): float(value)
                    for name, value in raw.items()
                }
            except (OSError, ValueError, TypeError) as exc:
                raise Sky130ModelError(
                    f"invalid resolved SKY130 card: {path}") from exc
            if "vth0" not in model_parameters:
                raise Sky130ModelError(
                    f"resolved SKY130 card has no vth0 parameter: {path}")
            version = float(model_parameters.get("version", -1.0))
            if version != 4.5:
                raise Sky130ModelError(
                    f"{path} uses unsupported BSIM4 version {version!r}")
            model_parameters = MappingProxyType(model_parameters)
            _MODEL_CACHE[source] = model_parameters

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
    key = make_bsim4_card_cache_key(
        source=source,
        pdk=pdk,
        model=model,
        section=section,
        bin_selector=bin_selector,
        width_um=width_um,
        length_um=length_um,
        nf=nf,
        mult=mult,
        temperature_c=temperature_c,
        corner=corner,
        mismatch_v=mismatch_v,
        extra={
            "reference_width_um": reference_width_um,
            **parameters,
        },
    )

    def build():
        card = Sky130Card(
            polarity=polarity,
            corner=corner,
            path=path,
            model_parameters=model_parameters,
            instance_parameters=MappingProxyType(dict(parameters)),
            source_version=4.5,
        )
        return (card, *card.to_bsim4_cards())

    return _CARD_CACHE.get_or_create(key, build)


def load_sky130_card(
    polarity: str,
    *,
    width_um: float,
    length_um: float,
    nf: int = 1,
    mult: int = 1,
    corner: str = "tt",
    reference_width_um: float | None = None,
    mismatch_v: float = 0.0,
    instance_parameters: dict[str, float] | None = None,
) -> Sky130Card:
    """Return the PDK-facing card from the shared immutable bundle cache."""
    return load_sky130_device_cards(
        polarity,
        width_um=width_um,
        length_um=length_um,
        nf=nf,
        mult=mult,
        corner=corner,
        reference_width_um=reference_width_um,
        mismatch_v=mismatch_v,
        instance_parameters=instance_parameters,
    )[0]


def sky130_card_cache_info() -> Bsim4CardCacheInfo:
    return _CARD_CACHE.cache_info()


def clear_sky130_card_cache() -> None:
    _CARD_CACHE.clear()
