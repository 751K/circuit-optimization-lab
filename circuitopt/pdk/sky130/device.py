"""CircuitOpt device adapter for native SKY130 BSIM4 MOS models."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np

from ...compact_models.bsim4 import (
    Bsim4Bias,
    Bsim4Evaluation,
    NativeBsim4Backend,
)
from ...device_model import TransistorModel, register_pdk
from .library import load_sky130_card, normalize_corner


_BACKEND = NativeBsim4Backend()


class _Sky130NativeFet(TransistorModel):
    POLARITY = "nmos"
    TYPE = 1
    HAS_TERMINAL_LINEARIZATION = True
    HAS_TERMINAL_NOISE = True
    TRANSIENT_BACKEND = "bsim4_native"
    SUPPORTS_MULTIPLICITY = True
    EXTRACT_W: float | None = None
    _EVAL_CACHE_MAX = 32

    def __init__(
        self,
        W: float = 1.0,
        L: float = 0.15,
        NF: int = 1,
        *,
        corner: str = "tt",
        vb: float = 0.0,
        temperature: float = 300.15,
        mismatch: float = 0.0,
        delvto: float = 0.0,
        mult: int = 1,
        extract_w: float | None = None,
        **parameters,
    ):
        mismatch_v = parameters.pop("_delvto", mismatch or delvto)
        self.W = float(W)
        self.L = float(L)
        self.NF = int(NF)
        self.mult = int(mult)
        self.corner = normalize_corner(corner)
        self.vb = float(vb)
        self.temperature = float(temperature)
        self.extract_w = extract_w
        self.g_area = self.W * self.L * self.mult
        self.kcl_sign = -1.0 if self.TYPE > 0 else 1.0
        reference_width = (
            extract_w if extract_w is not None else self.EXTRACT_W)
        card = load_sky130_card(
            self.POLARITY,
            width_um=self.W,
            length_um=self.L,
            nf=self.NF,
            mult=self.mult,
            corner=self.corner,
            reference_width_um=reference_width,
            mismatch_v=float(mismatch_v),
            instance_parameters=parameters,
        )
        self.model_card, self.instance_card = card.to_bsim4_cards()
        self.card_path = str(card.path)
        self.model_name = card.path.stem
        self._evaluations: OrderedDict[tuple, Bsim4Evaluation] = OrderedDict()

    def _evaluate(
        self,
        Vs: float,
        Vd: float,
        Vg: float,
        frequency_hz: float | None = None,
    ) -> Bsim4Evaluation:
        key = (float(Vs), float(Vd), float(Vg), frequency_hz)
        cached = self._evaluations.get(key)
        if cached is not None:
            self._evaluations.move_to_end(key)
            return cached
        result = _BACKEND.evaluate(
            self.model_card,
            self.instance_card,
            Bsim4Bias(
                drain=Vd,
                gate=Vg,
                source=Vs,
                bulk=self.vb,
                temperature_k=self.temperature,
            ),
            frequency_hz=frequency_hz,
        )
        self._evaluations[key] = result
        while len(self._evaluations) > self._EVAL_CACHE_MAX:
            self._evaluations.popitem(last=False)
        return result

    def get_Idc(self, Vs: float, Vd: float, Vg: float) -> float:
        return float(self._evaluate(Vs, Vd, Vg).terminal_currents[0])

    def get_terminal_currents(self, Vs: float, Vd: float, Vg: float) -> np.ndarray:
        return self._evaluate(Vs, Vd, Vg).terminal_currents

    def get_op(self, Vs: float, Vd: float, Vg: float) -> Tuple:
        self._evaluate(Vs, Vd, Vg)
        return ()

    def get_ss_params(self, Vs: float, Vd: float, Vg: float) -> Dict[str, float]:
        result = self._evaluate(Vs, Vd, Vg)
        matrix = result.conductance
        capacitance = result.capacitance
        return {
            "gm": max(float(matrix[0, 1]), 0.0),
            "gds": max(float(matrix[0, 0]), 1e-15),
            "gmb": float(matrix[0, 3]),
            "Cgs": max(float(-capacitance[1, 2]), 0.0),
            "Cgd": max(float(-capacitance[1, 0]), 0.0),
            "Ich": abs(float(result.terminal_currents[0])),
        }

    def get_terminal_linearization(self, Vs: float, Vd: float, Vg: float):
        result = self._evaluate(Vs, Vd, Vg)
        return result.conductance, result.capacitance

    def get_terminal_noise(
        self,
        Vs: float,
        Vd: float,
        Vg: float,
        frequency: float,
    ):
        noise = self._evaluate(Vs, Vd, Vg, float(frequency)).noise
        if noise is None:
            raise RuntimeError("native BSIM4 backend returned no noise data")
        return noise

    def get_capacitances(self, Vs: float, Vd: float, Vg: float):
        params = self.get_ss_params(Vs, Vd, Vg)
        return params["Cgs"], params["Cgd"]

    def get_noise_psd(
        self,
        Vs: float,
        Vd: float,
        Vg: float,
        frequency: float,
    ):
        noise = self.get_terminal_noise(Vs, Vd, Vg, frequency)
        white = max(float(noise.components["white"][0, 0].real), 0.0)
        flicker = max(float(noise.components["flicker"][0, 0].real), 0.0)
        return white, flicker

    def get_capacitance_charges_from_op(self, *args):
        raise NotImplementedError(
            "native BSIM4 transient uses the four-terminal charge backend")

    def get_capacitance_branch_terms_from_op(self, *args):
        raise NotImplementedError(
            "native BSIM4 transient uses the four-terminal charge backend")

    def get_numba_params(self):
        raise NotImplementedError(
            "native BSIM4 devices do not use the OTFT numba parameter bundle")

    def get_terminal_charges(self, Vs: float, Vd: float, Vg: float) -> np.ndarray:
        return self._evaluate(Vs, Vd, Vg).terminal_charges

    terminal_charges = get_terminal_charges

    def create_native_solver_handle(self):
        return _BACKEND.create_device(
            self.model_card, self.instance_card, self.temperature)


class Sky130Nfet(_Sky130NativeFet):
    POLARITY = "nmos"
    TYPE = 1


class Sky130Pfet(_Sky130NativeFet):
    POLARITY = "pmos"
    TYPE = -1


register_pdk(
    "sky130",
    {"nmos": Sky130Nfet, "pmos": Sky130Pfet},
    default=False,
)
