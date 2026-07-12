"""Resolution of optional local simulator and PDK installations."""
from __future__ import annotations

import os
import shutil
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_VENV = os.path.join(PROJECT_ROOT, ".venv")


def _absolute(value: str) -> str:
    return os.path.abspath(os.path.expanduser(value))


def _venv_roots() -> tuple[str, ...]:
    """Active/interpreter/project virtual environments, in priority order."""
    candidates = []
    configured = os.environ.get("VIRTUAL_ENV")
    if configured:
        candidates.append(_absolute(configured))
    if os.path.isfile(os.path.join(sys.prefix, "pyvenv.cfg")):
        candidates.append(_absolute(sys.prefix))
    candidates.append(LOCAL_VENV)
    return tuple(dict.fromkeys(candidates))


def _command(value: str) -> str | None:
    candidate = _absolute(value) if os.path.sep in value else shutil.which(value)
    return candidate if candidate and os.access(candidate, os.X_OK) else None


def pdk_root() -> str:
    """PDK root: explicit environment, then an active/project virtual environment."""
    configured = os.environ.get("PDK_ROOT")
    if configured:
        return _absolute(configured)
    candidates = [os.path.join(root, "pdk") for root in _venv_roots()]
    return next((path for path in candidates if os.path.isdir(path)), candidates[0])


def ngspice_binary() -> str | None:
    """Runnable ngspice: explicit environment, project uv environment, PATH."""
    configured = os.environ.get("NGSPICE_BIN")
    if configured:
        return _command(configured)
    for root in _venv_roots():
        for relative in (("ngspice", "bin", "ngspice"), ("bin", "ngspice")):
            candidate = os.path.join(root, *relative)
            if os.access(candidate, os.X_OK):
                return candidate
    return shutil.which("ngspice")


def openvaf_root() -> str | None:
    """OpenVAF source checkout when configured or vendored locally."""
    configured = os.environ.get("OPENVAF_ROOT")
    if configured:
        return _absolute(configured)
    for relative in (("third_party", "OpenVAF-Reloaded"),
                     ("vendor", "OpenVAF-Reloaded")):
        candidate = os.path.join(PROJECT_ROOT, *relative)
        if os.path.isdir(candidate):
            return candidate
    return None


def openvaf_binary() -> str | None:
    """OpenVAF compiler: explicit binary/root, virtual environment, then PATH."""
    configured = os.environ.get("OPENVAF_BIN")
    if configured:
        return _command(configured)
    root = openvaf_root()
    if root:
        candidate = os.path.join(root, "target", "release", "openvaf-r")
        if os.access(candidate, os.X_OK):
            return candidate
    for venv in _venv_roots():
        for relative in (("bin", "openvaf-r"), ("openvaf", "bin", "openvaf-r")):
            candidate = os.path.join(venv, *relative)
            if os.access(candidate, os.X_OK):
                return candidate
    return shutil.which("openvaf-r")


def bsim4_va_path() -> str | None:
    """BSIM4 Verilog-A source, independently overridable with ``BSIM4_VA``."""
    configured = os.environ.get("BSIM4_VA")
    if configured:
        path = _absolute(configured)
        return path if os.path.isfile(path) else None
    root = openvaf_root()
    if root:
        path = os.path.join(root, "integration_tests", "BSIM4", "bsim4.va")
        return path if os.path.isfile(path) else None
    return None


def openvaf_source_path(*parts: str) -> str | None:
    """A source file inside the configured OpenVAF checkout, when present."""
    root = openvaf_root()
    if not root:
        return None
    path = os.path.join(root, *parts)
    return path if os.path.isfile(path) else None


def osdi_cache_dir() -> str:
    """Writable OSDI cache, defaulting inside the selected virtual environment."""
    configured = os.environ.get("OSDI_CACHE_DIR")
    if configured:
        return _absolute(configured)
    return os.path.join(_venv_roots()[0], "cache", "circuitopt", "osdi")
