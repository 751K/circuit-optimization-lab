from circuitopt import toolchain


def _executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return str(path)


def test_explicit_pdk_root_has_priority(tmp_path, monkeypatch):
    configured = tmp_path / "custom-pdk"
    monkeypatch.setenv("PDK_ROOT", str(configured))
    assert toolchain.pdk_root() == str(configured)


def test_active_venv_pdk_is_discovered(tmp_path, monkeypatch):
    monkeypatch.delenv("PDK_ROOT", raising=False)
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path))
    (tmp_path / "pdk").mkdir()
    assert toolchain.pdk_root() == str(tmp_path / "pdk")


def test_ngspice_resolution_order(tmp_path, monkeypatch):
    explicit = _executable(tmp_path / "explicit" / "ngspice")
    local = _executable(tmp_path / "venv" / "ngspice" / "bin" / "ngspice")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))
    monkeypatch.setenv("NGSPICE_BIN", explicit)
    assert toolchain.ngspice_binary() == explicit
    monkeypatch.delenv("NGSPICE_BIN")
    assert toolchain.ngspice_binary() == local
