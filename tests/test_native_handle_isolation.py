"""Native handle ownership and independent-work cache isolation.

A BSIM instance keeps its internal drain/source node solution as the next
call's warm start and the ``state0`` voltages the next load limits against.
Overlapping backend leases must therefore own different handles. Sequential
leases retain warm-cache reuse by default, while an isolated cache scope also
prevents separate units such as signoff points from inheriting that history.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from circuitopt.compact_models.bsim4 import isolated_native_device_cache
from circuitopt.pdk.freepdk45.device import Fp45Nfet
from circuitopt.toolchain import pdk_root


# Leasing a handle builds a real BSIM instance from a real model card, so the
# whole file needs the FreePDK45 cards installed.
_CARD = os.path.join(pdk_root(), "freepdk45", "models_nom", "NMOS_VTG.inc")
pytestmark = pytest.mark.skipif(
    not os.path.isfile(_CARD), reason="FreePDK45 cards not present")


def _lease_identity(device):
    lease = device.lease_native_solver_handle()
    try:
        return id(lease.device)
    finally:
        lease.close()


def test_leases_reuse_one_handle_inside_a_scope():
    # Isolation is per unit of work, not per call: reuse within the unit is
    # what makes the cache worth having.
    device = Fp45Nfet(W=1.0, L=0.05)
    with isolated_native_device_cache():
        first = _lease_identity(device)
        second = _lease_identity(device)
    assert first == second


def test_a_scope_releases_its_handles_when_it_exits():
    # Nothing may survive a unit of work; otherwise the next unit inherits it
    # and the result depends on execution order again. (Object identity cannot
    # be compared across scopes: CPython reuses the address of a freed handle.)
    from circuitopt.pdk.freepdk45.device import _BACKEND

    device = Fp45Nfet(W=1.0, L=0.05)
    with isolated_native_device_cache():
        _lease_identity(device)
        assert _BACKEND._scoped, "the scope leased into no namespace of its own"
    assert not _BACKEND._scoped


def test_concurrent_scopes_never_share_a_handle():
    # The campaign case: several units running at once, all wanting the same
    # card. Without isolation they all get one handle and interleave on it.
    device = Fp45Nfet(W=1.0, L=0.05)

    def unit(_):
        with isolated_native_device_cache():
            return _lease_identity(device)

    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(unit, range(4)))

    assert len(set(identities)) == len(identities)


def test_unscoped_leases_still_share_the_process_cache():
    # Ordinary single-run callers keep the warm handle they always had.
    device = Fp45Nfet(W=1.0, L=0.05)
    assert _lease_identity(device) == _lease_identity(device)


def test_concurrent_unscoped_leases_never_share_an_active_handle():
    # This is a backend invariant, not something each concurrent driver must
    # remember to request. Per-call native locking would only serialize the
    # individual evaluations while still interleaving two solver histories.
    device = Fp45Nfet(W=1.0, L=0.05)
    barrier = Barrier(2)

    def unit(_):
        lease = device.lease_native_solver_handle()
        try:
            barrier.wait()
            identity = id(lease.device)
            barrier.wait()
            return identity
        finally:
            lease.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        identities = list(pool.map(unit, range(2)))

    assert len(set(identities)) == len(identities)
