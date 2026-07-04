"""Validate core.osdi_host against ngspice on an OpenVAF-compiled BSIM4 model.

The OpenVAF compiler, the OSDI-enabled ngspice, and the BSIM4 Verilog-A source
live on the external drive (see the ``silicon-pdk-openvaf`` project memory and the
``build-openvaf`` / ``run-osdi-ngspice`` skills). These tests locate that toolchain,
compile ``bsim4.va`` → ``bsim4.osdi`` once, then check the Python host reproduces
ngspice's DC/AC results for the *same* compiled model (model == oracle). The whole
module skips cleanly when the toolchain is absent (e.g. CI), so it never blocks the
core suite.

Override locations with ``OPENVAF_ROOT`` / ``NGSPICE_BIN`` if they move.
"""
import os
import re
import subprocess

import pytest

VAF_ROOT = os.environ.get("OPENVAF_ROOT", "/Volumes/MacoutDsik/Code/VAF/OpenVAF-Reloaded")
VACOMPILE = os.path.join(VAF_ROOT, ".claude/skills/build-openvaf/scripts/vacompile.sh")
RUN_NGSPICE = os.path.join(VAF_ROOT, ".claude/skills/run-osdi-ngspice/scripts/run-ngspice.sh")
BSIM4_VA = os.path.join(VAF_ROOT, "integration_tests/BSIM4/bsim4.va")

_HAVE_COMPILER = os.path.exists(VACOMPILE) and os.path.exists(BSIM4_VA)
pytestmark = pytest.mark.skipif(
    not _HAVE_COMPILER, reason="OpenVAF/BSIM4 toolchain not present (external-drive only)")

# A minimal but working NMOS card; l/w/nf are BSIM4 *model* params in this VA.
CARD = dict(type=1, l=0.15e-6, w=1.0e-6, toxe=4.148e-9, vth0=0.4, u0=0.04)
_CARD_SPICE = "type=1 l=0.15u w=1u toxe=4.148e-9 vth0=0.4 u0=0.04"
_BIAS = (1.0, 1.1, 0.0, 0.0)   # Vd, Vg, Vs, Vb


@pytest.fixture(scope="module")
def osdi_path(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("osdi") / "bsim4.osdi")
    subprocess.run([VACOMPILE, BSIM4_VA, "-o", out], check=True, capture_output=True)
    assert os.path.exists(out)
    return out


@pytest.fixture(scope="module")
def dev(osdi_path):
    from core.osdi_host import load_osdi, Device
    return Device(load_osdi(osdi_path), CARD)


def test_introspection(osdi_path):
    from core.osdi_host import load_osdi
    m = load_osdi(osdi_path).model()
    assert m.name == "bsim4va"
    assert m.terminals == ["d", "g", "s", "b"]
    assert m.num_terminals == 4 and len(m.nodes) > 4      # has internal nodes


def test_dc_op_is_physical(dev):
    op = dev.operating_point(*_BIAS)
    assert 1e-5 < op["Id"] < 1e-3           # a real saturation current
    assert op["gm"] > 0 and op["gds"] > 0   # positive small-signal conductances
    assert op["gm"] > op["gds"]             # transconductance dominates
    assert op["Cgg"] > 0
    assert op["Cgg"] >= op["Cgs"] > 0       # total gate cap >= partial


def test_id_increases_with_vg(dev):
    ids = [dev.operating_point(1.0, vg, 0.0, 0.0)["Id"] for vg in (0.2, 0.6, 1.0, 1.4)]
    assert all(a < b for a, b in zip(ids, ids[1:]))


def test_deterministic_across_calls(dev):
    a = dev.operating_point(*_BIAS)["Id"]
    _ = dev.operating_point(1.0, 0.7, 0.0, 0.0)     # perturb the instance
    b = dev.operating_point(*_BIAS)["Id"]
    assert a == pytest.approx(b, rel=1e-12)


def test_noise_is_flicker_dominated(dev):
    dev.operating_point(*_BIAS)
    n1, n2 = dev.noise_psd(1.0).sum(), dev.noise_psd(100.0).sum()
    assert n1 > 0 and n2 > 0
    assert n1 / n2 == pytest.approx(100.0, rel=0.05)     # ~1/f over a decade


def _ngspice_op_id(osdi_path, tmp_path):
    """Drain current from ngspice running the same .osdi + card (Id = -i(vd))."""
    net = tmp_path / "op.cir"
    net.write_text(
        f"* osdi op\n.control\npre_osdi {osdi_path}\n.endc\n"
        f"vd d 0 dc {_BIAS[0]}\nvg g 0 dc {_BIAS[1]}\nvs s 0 0\nvb b 0 0\n"
        f"N1 d g s b mn\n.model mn bsim4va {_CARD_SPICE}\n"
        f".control\nop\nprint i(vd)\n.endc\n.end\n")
    runner = RUN_NGSPICE if os.path.exists(RUN_NGSPICE) else \
        os.environ.get("NGSPICE_BIN", "ngspice")
    res = subprocess.run([runner, "-b", str(net)], capture_output=True, text=True)
    m = re.search(r"i\(vd\)\s*=\s*([-\d.eE+]+)", res.stdout)
    assert m, f"could not parse i(vd):\n{res.stdout}\n{res.stderr}"
    return -float(m.group(1))


@pytest.mark.skipif(not os.path.exists(RUN_NGSPICE),
                    reason="OSDI-enabled ngspice not present")
def test_dc_matches_ngspice(dev, osdi_path, tmp_path):
    id_host = dev.operating_point(*_BIAS)["Id"]
    id_ng = _ngspice_op_id(osdi_path, tmp_path)
    assert id_host == pytest.approx(id_ng, rel=1e-3)     # model == oracle


# ── TransistorModel ABC adapter (core.osdi_device) ───────────────────────
class _Nfet:
    """Test binding of the BSIM4 model as an OsdiDevice subclass (built lazily)."""
    @staticmethod
    def cls():
        from core.osdi_device import OsdiDevice

        class Nfet(OsdiDevice):
            VA_PATH = BSIM4_VA
            MODULE = "bsim4va"
            BASE_CARD = dict(toxe=4.148e-9, vth0=0.4, u0=0.04)
            TYPE = 1
        return Nfet


def test_adapter_matches_raw_device(dev):
    adapter = _Nfet.cls()(W=1.0, L=0.15)                  # 1um / 0.15um
    op = dev.operating_point(*_BIAS)
    ss = adapter.get_ss_params(0.0, 1.0, 1.1)
    assert ss["gm"] == pytest.approx(op["gm"], rel=1e-9)
    assert ss["gds"] == pytest.approx(op["gds"], rel=1e-9)
    assert ss["Cgs"] == pytest.approx(op["Cgs"], rel=1e-9)
    assert adapter.g_area == pytest.approx(0.15)


def test_adapter_noise_split_and_deferred_transient():
    adapter = _Nfet.cls()(W=1.0, L=0.15)
    s_th, s_fl = adapter.get_noise_psd(0.0, 1.0, 1.1, 1.0)
    assert s_th > 0 and s_fl > 0                          # thermal (white) + flicker
    assert s_fl > s_th                                    # flicker-dominated at 1 Hz
    with pytest.raises(NotImplementedError):
        adapter.get_numba_params()
