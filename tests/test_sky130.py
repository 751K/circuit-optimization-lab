"""SKY130 PDK registration + device sanity (skipped without the external toolchain).

SKY130 params are resolved by ngspice and fed to the OpenVAF-compiled BSIM4 (see the
``silicon-pdk-openvaf`` memory). These tests need the SKY130 PDK + OSDI-ngspice +
OpenVAF (external drive), so they skip cleanly in CI. The registration itself
(``register_pdk("sky130", …)``) is import-time and always exercised by ``import core``.
"""
import os

import pytest

import core
from core.osdi_device import openvaf_binary

PDK_ROOT = os.environ.get("PDK_ROOT", "/Volumes/MacoutDsik/pdk")
_NGSPICE_LIB = os.path.join(PDK_ROOT, "sky130A/libs.tech/ngspice/sky130.lib.spice")
_HAVE = os.path.exists(_NGSPICE_LIB) and openvaf_binary() is not None


def test_sky130_registered_but_not_default():
    # import-time registration — no toolchain needed
    assert "sky130" in core.list_pdks()
    assert core.get_default_pdk() == "at4000tg"          # additive; OTFT stays default
    assert core.transistor_type("nmos", pdk="sky130") == "sky130.nmos"
    assert core.transistor_type("pmos", pdk="sky130") == "sky130.pmos"


@pytest.mark.skipif(not _HAVE, reason="SKY130 PDK / OpenVAF toolchain not present")
def test_sky130_nfet_physical():
    nfet = core.create_transistor("nmos", pdk="sky130", W=1.0, L=0.15)
    op_id = nfet.get_Idc(0.0, 1.8, 1.8)                  # get_Idc(Vs, Vd, Vg)
    ss = nfet.get_ss_params(0.0, 1.8, 1.8)
    assert 1e-4 < op_id < 2e-3                           # ~sub-mA at 1.8 V
    assert ss["gm"] > 0 and ss["gds"] > 0 and ss["Cgs"] > 0
    assert nfet.g_area == pytest.approx(0.15)


@pytest.mark.skipif(not _HAVE, reason="SKY130 PDK / OpenVAF toolchain not present")
def test_sky130_pfet_physical():
    pfet = core.create_transistor("pmos", pdk="sky130", W=1.0, L=0.15)
    idp = pfet.get_Idc(1.8, 0.0, 0.0)                    # pmos: source at rail
    assert 1e-5 < abs(idp) < 2e-3
    assert pfet.get_ss_params(1.8, 0.0, 0.0)["gm"] > 0


@pytest.mark.skipif(not _HAVE, reason="SKY130 PDK / OpenVAF toolchain not present")
def test_extract_card_has_key_bsim4_params():
    from core.sky130_model import extract_sky130_card
    card = extract_sky130_card("nmos", 1.0, 0.15, "tt")
    for key in ("vth0", "toxe", "u0", "vsat", "k1", "version"):
        assert key in card
    assert card["version"] == pytest.approx(4.5)         # SKY130 uses BSIM4.5
    assert 0.2 < card["vth0"] < 0.6                       # a sane nfet threshold
