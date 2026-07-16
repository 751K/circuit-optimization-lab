"""FreePDK45 compatibility exports and optional ngspice oracle devices.

``freepdk45.nmos`` / ``freepdk45.pmos`` are registered by the native BSIM4
adapter under :mod:`circuitopt.pdk.freepdk45`. The historical cached-ngspice
devices remain available explicitly as ``freepdk45_ngspice.*`` for regression
comparisons; normal CircuitOpt analyses do not require ngspice.
"""
from __future__ import annotations

from .device_model import register_pdk
from .ngspice_device import NgspiceDevice
from .pdk.freepdk45 import (
    FREEPDK45_CORNERS,
    Fp45Nfet,
    Fp45Pfet,
    corner_card_dir,
    freepdk45_card_path,
    normalize_corner,
)


_VDD = 1.0
_card_path = freepdk45_card_path


class _Fp45NgspiceFet(NgspiceDevice):
    POLARITY = "nmos"
    MODEL_NAME = "NMOS_VTG"
    VDD = _VDD

    def __init__(
        self,
        W: float = 0.09,
        L: float = 0.05,
        NF: int = 1,
        *,
        corner: str = "nom",
        vb: float = 0.0,
        temperature: float = 300.15,
        extract_w: float | None = None,
        **parameters,
    ):
        corner = normalize_corner(corner)
        self.CARD_PATH = freepdk45_card_path(self.POLARITY, corner)
        super().__init__(
            W=W,
            L=L,
            NF=NF,
            vb=vb,
            corner=corner,
            temperature=temperature,
            extract_w=extract_w,
            **parameters,
        )


class Fp45NgspiceNfet(_Fp45NgspiceFet):
    POLARITY = "nmos"
    MODEL_NAME = "NMOS_VTG"


class Fp45NgspicePfet(_Fp45NgspiceFet):
    POLARITY = "pmos"
    MODEL_NAME = "PMOS_VTG"


register_pdk(
    "freepdk45_ngspice",
    {"nmos": Fp45NgspiceNfet, "pmos": Fp45NgspicePfet},
    default=False,
)


__all__ = [
    "FREEPDK45_CORNERS",
    "Fp45Nfet",
    "Fp45Pfet",
    "Fp45NgspiceNfet",
    "Fp45NgspicePfet",
    "corner_card_dir",
    "freepdk45_card_path",
    "normalize_corner",
]
