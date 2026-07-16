"""TSMC 28HPC+ core MOS adapter backed by the local HSPICE model delivery.

No PDK content is bundled with circuitopt.  Set ``TSMC28_MODEL_DIR`` to the
delivery's ``models/hspice`` directory, or ``TSMC28_PDK_ROOT`` to an installed
PDK root.  The adapter uses the 0.9/1.8-V ``1d8`` deck and exposes the standard
0.9-V core wrappers as ``tsmc28hpcp.nmos`` / ``tsmc28hpcp.pmos``.

The foundry deck wraps BSIM4 devices in ``nch_mac`` / ``pch_mac`` subcircuits and
uses nested HSPICE ``.lib`` sections.  ngspice does not expand those nested
sections itself, so this adapter emits their direct dependency closure and runs
ngspice in HSPICE compatibility mode.  The original model file remains external
and unmodified.
"""
from __future__ import annotations

import os

import numpy as np

from .device_model import register_pdk
from .ngspice_device import NgspiceDevice
from .ngspice_process import NgspiceProcessAdapter
from .toolchain import tsmc28_model_dir


_MODEL_FILE = "cln28hpcp_1d8_elk_v1d0_2p2.l"


class Tsmc28HpcpAdapter(NgspiceProcessAdapter):
    name = "TSMC28HPC+"
    model_prefix = "tsmc28hpcp_ngspice"
    corners = ("tt", "ss", "ff", "sf", "fs")
    default_corner = "tt"
    vdd = 0.9
    cache_namespace = "tsmc28hpcp"
    command_args = ("-D", "ngbehavior=hsa")

    def normalize_corner(self, corner) -> str:
        if isinstance(corner, str) and corner.lower() == "nom":
            corner = "tt"
        return super().normalize_corner(corner)

    def model_path(self) -> str:
        path = os.path.join(tsmc28_model_dir(), _MODEL_FILE)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"TSMC28HPC+ HSPICE model not found: {path}; set TSMC28_MODEL_DIR "
                "to models/hspice, set TSMC28_PDK_ROOT, or install the model at "
                "PDK/tsmc28hpcp/models/hspice")
        return path

    def _library_lines(self, corner: str) -> list[str]:
        # The official embedded-usage deck selects these five sections indirectly.
        # ngspice does not recurse through same-file .lib calls, so select the
        # dependency closure explicitly while leaving the model file untouched.
        path = self.model_path()
        sections = ("setup", self.normalize_corner(corner), "global", "total", "stat")
        return [f'.lib "{path}" {section}' for section in sections]

    def deck_preamble(self, model_types, device_kwargs, device_names):
        self.validate_model_types(model_types, device_names)
        corner = self.common_corner(device_kwargs, device_names)
        return corner, self._library_lines(corner)

    @staticmethod
    def _polarity(model_type: str) -> str:
        polarity = str(model_type).rsplit(".", 1)[-1]
        if polarity not in {"nmos", "pmos"}:
            raise ValueError(f"unsupported TSMC28HPC+ core model type {model_type!r}")
        return polarity

    def render_instance(self, *, name, d, g, s, b, model_type, width_um,
                        length_um, nf, mismatch=0.0, mult=1):
        polarity = self._polarity(model_type)
        model = "nch_mac" if polarity == "nmos" else "pch_mac"
        line = (f"{name} {d} {g} {s} {b} {model} "
                f"w={float(width_um):.17g}u l={float(length_um):.17g}u nf={int(nf)}")
        if int(mult) > 1:
            line += f" m={int(mult)}"
        if float(mismatch) != 0.0:
            line += f" _delvto={float(mismatch):.17g}"
        return line

    def characterization_preamble(self, corner, polarity, card_path):
        del polarity, card_path
        return self._library_lines(corner)

    def characterization_instance(self, *, name, polarity, width_um, length_um, nf=1):
        model = "nch_mac" if polarity == "nmos" else "pch_mac"
        return (f"{name} d g s b {model} w={float(width_um):.17g}u "
                f"l={float(length_um):.17g}u nf={int(nf)}")

    def op_vector(self, instance_name: str, variable: str) -> str:
        # nch_mac/pch_mac name their selected BSIM4 instance ``main``.
        return f"@m.{instance_name.lower()}.main[{variable}]"

    def normalize_op_data(self, variable: str, values):
        # The macro's internal BSIM vectors expose opposite cross-capacitance signs
        # across bias regions. Circuitopt stores effective Cgs/Cgd as negative
        # dQg/dVterminal values and later converts them to positive branch caps.
        if variable in {"cgs", "cgd"}:
            return -np.abs(values)
        return values


TSMC28HPCP_ADAPTER = Tsmc28HpcpAdapter()


class _Tsmc28CoreFet(NgspiceDevice):
    NGSPICE_ADAPTER = TSMC28HPCP_ADAPTER
    VDD = TSMC28HPCP_ADAPTER.vdd
    MODEL_NAME = "nch_mac"
    POLARITY = "nmos"
    NF_SCALES_OP = False

    def __init__(self, W: float = 1.0, L: float = 0.03, NF: int = 1, *,
                 corner: str = "tt", vb: float = 0.0, temperature: float = 300.15,
                 extract_w: float = None, **_ignored):
        corner = TSMC28HPCP_ADAPTER.normalize_corner(corner)
        self.CARD_PATH = TSMC28HPCP_ADAPTER.model_path()
        super().__init__(W=W, L=L, NF=NF, vb=vb, corner=corner,
                         temperature=temperature, extract_w=extract_w)


class Tsmc28Nfet(_Tsmc28CoreFet):
    POLARITY = "nmos"
    MODEL_NAME = "nch_mac"


class Tsmc28Pfet(_Tsmc28CoreFet):
    POLARITY = "pmos"
    MODEL_NAME = "pch_mac"


register_pdk(
    "tsmc28hpcp_ngspice",
    {"nmos": Tsmc28Nfet, "pmos": Tsmc28Pfet},
    default=False,
)
