"""CircuitOpt device-model adapter for native TSMC28HPC+ core MOS."""
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
from .library import load_tsmc28_core_library


_BACKEND = NativeBsim4Backend()


class _Tsmc28NativeCoreFet(TransistorModel):
    POLARITY = "nmos"
    TYPE = 1
    HAS_TERMINAL_LINEARIZATION = True
    HAS_TERMINAL_NOISE = True
    TRANSIENT_BACKEND = "bsim4_native"
    SUPPORTS_MULTIPLICITY = True
    _EVAL_CACHE_MAX = 32

    def __init__(
        self,
        W: float = 1.0,
        L: float = 0.03,
        NF: int = 1,
        *,
        corner: str = "tt",
        vb: float = 0.0,
        temperature: float = 300.15,
        mismatch: float = 0.0,
        delvto: float = 0.0,
        mult: int = 1,
        **parameters,
    ):
        mismatch_v = parameters.pop("_delvto", mismatch or delvto)
        if parameters:
            names = ", ".join(sorted(parameters))
            raise TypeError(f"unsupported native TSMC28 instance parameters: {names}")
        self.W = float(W)
        self.L = float(L)
        self.NF = int(NF)
        self.mult = int(mult)
        self.corner = str(corner).lower()
        self.vb = float(vb)
        self.temperature = float(temperature)
        self.g_area = self.W * self.L * self.mult
        self.kcl_sign = -1.0 if self.TYPE > 0 else 1.0
        card = load_tsmc28_core_library().core_card(
            self.POLARITY,
            width_um=self.W,
            length_um=self.L,
            nf=self.NF,
            mult=self.mult,
            corner=self.corner,
            temperature_c=self.temperature - 273.15,
            mismatch_v=float(mismatch_v),
        )
        self.model_card, self.instance_card = card.to_bsim4_cards()
        self.bin_name = card.bin_name
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


class Tsmc28NativeNfet(_Tsmc28NativeCoreFet):
    POLARITY = "nmos"
    TYPE = 1


class Tsmc28NativePfet(_Tsmc28NativeCoreFet):
    POLARITY = "pmos"
    TYPE = -1


register_pdk(
    "tsmc28hpcp",
    {"nmos": Tsmc28NativeNfet, "pmos": Tsmc28NativePfet},
    default=False,
)
