"""SKY130 native C BSIM4 registration and device sanity."""

import pytest

import circuitopt


def test_sky130_registered_but_not_default():
    # import-time registration — no toolchain needed
    assert "sky130" in circuitopt.list_pdks()
    assert circuitopt.get_default_pdk() == "at4000tg"          # additive; OTFT stays default
    assert circuitopt.transistor_type("nmos", pdk="sky130") == "sky130.nmos"
    assert circuitopt.transistor_type("pmos", pdk="sky130") == "sky130.pmos"


def test_sky130_nfet_physical():
    nfet = circuitopt.create_transistor("nmos", pdk="sky130", W=1.0, L=0.15)
    assert nfet.TRANSIENT_BACKEND == "bsim4_native"
    op_id = nfet.get_Idc(0.0, 1.8, 1.8)                  # get_Idc(Vs, Vd, Vg)
    ss = nfet.get_ss_params(0.0, 1.8, 1.8)
    assert 1e-4 < op_id < 2e-3                           # ~sub-mA at 1.8 V
    assert ss["gm"] > 0 and ss["gds"] > 0 and ss["Cgs"] > 0
    assert nfet.g_area == pytest.approx(0.15)


def test_sky130_pfet_physical():
    pfet = circuitopt.create_transistor("pmos", pdk="sky130", W=1.0, L=0.15)
    idp = pfet.get_Idc(1.8, 0.0, 0.0)                    # pmos: source at rail
    assert 1e-5 < abs(idp) < 2e-3
    assert pfet.get_ss_params(1.8, 0.0, 0.0)["gm"] > 0


def test_bundled_card_has_key_bsim4_params():
    from circuitopt.pdk.sky130 import load_sky130_card
    card = load_sky130_card(
        "nmos", width_um=1.0, length_um=0.15).model_parameters
    for key in ("vth0", "toxe", "u0", "vsat", "k1", "version"):
        assert key in card
    assert card["version"] == pytest.approx(4.5)         # SKY130 uses BSIM4.5
    assert 0.2 < card["vth0"] < 0.6                       # a sane nfet threshold
