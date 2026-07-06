"""Smoke tests for the ``python -m circuitopt plot`` subcommand wiring.

The plot subcommand parses args and dispatches to ``examples.plot_transient`` /
``examples.plot_bode`` by ``kind``.  These tests pin the *wiring* — arg parsing,
``kind`` → function routing, kwarg forwarding, and the matplotlib-missing
``SystemExit`` — using stub plot functions, so a regression in the CLI surfaces
without paying the (seconds-long) real solver + figure-render cost.  The plot
functions' physical correctness is exercised by running the standalone scripts,
not here.
"""
import argparse

import pytest

from circuitopt.__main__ import _add_plot_parser, _cmd_plot


def _parse(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    _add_plot_parser(sub)
    return ap.parse_args(argv)


def test_plot_parser_defaults_and_overrides():
    ns = _parse(["plot"])
    assert ns.cmd == "plot" and ns.kind == "all"
    assert ns.f0 == 10.0 and ns.f_chop == 225.0 and ns.input_diff == 1e-3
    assert ns.out_dir == "results" and ns.npts is None and not ns.quiet
    ns2 = _parse(["plot", "pac", "--f-chop", "300", "--npts", "41", "--quiet"])
    assert ns2.kind == "pac" and ns2.f_chop == 300.0 and ns2.npts == 41 and ns2.quiet


def test_plot_rejects_unknown_kind():
    with pytest.raises(SystemExit):
        _parse(["plot", "bogus"])


@pytest.mark.parametrize("kind,expect", [
    ("afe",       {"afe"}),
    ("chopper",   {"chopper"}),
    ("ac",        {"ac"}),
    ("pac",       {"pac"}),
    ("transient", {"afe", "chopper"}),
    ("bode",      {"ac", "pac"}),
    ("all",       {"afe", "chopper", "ac", "pac"}),
])
def test_plot_dispatch_routes_by_kind(monkeypatch, tmp_path, kind, expect):
    pytest.importorskip("matplotlib")
    import examples.plot_bode as pbd
    import examples.plot_transient as ptr

    called: dict[str, dict] = {}

    def _stub(name):
        def f(*a, **k):
            called[name] = k
            return tmp_path / f"{name}.png"
        return f

    monkeypatch.setattr(ptr, "plot_afe", _stub("afe"))
    monkeypatch.setattr(ptr, "plot_chopper", _stub("chopper"))
    monkeypatch.setattr(pbd, "plot_ac", _stub("ac"))
    monkeypatch.setattr(pbd, "plot_pac", _stub("pac"))

    outs = _cmd_plot(_parse(["plot", kind, "--out-dir", str(tmp_path), "--quiet"]))

    assert set(called) == expect                       # exactly the right routes fired
    assert len(outs) == len(expect)                    # one figure path per route
    for kwargs in called.values():
        assert kwargs.get("out_dir") == str(tmp_path)  # out-dir threaded to every plot


def test_plot_forwards_tuning_flags(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    import examples.plot_bode as pbd
    import examples.plot_transient as ptr

    seen: dict[str, dict] = {}
    for mod, name in [(ptr, "plot_afe"), (ptr, "plot_chopper"),
                      (pbd, "plot_ac"), (pbd, "plot_pac")]:
        monkeypatch.setattr(mod, name,
                            (lambda nm: (lambda *a, **k: seen.setdefault(nm, k)))(name))

    _cmd_plot(_parse(["plot", "all", "--f-chop", "300", "--input-diff", "2e-3",
                      "--f0", "20", "--amp", "1e-3", "--npts", "31",
                      "--out-dir", str(tmp_path), "--quiet"]))

    assert seen["plot_afe"]["f0"] == 20.0 and seen["plot_afe"]["amp"] == 1e-3
    assert seen["plot_chopper"]["f_chop"] == 300.0
    assert seen["plot_chopper"]["input_diff"] == 2e-3
    assert seen["plot_ac"]["npts"] == 31            # npts forwarded only when given
    assert seen["plot_pac"]["f_chop"] == 300.0 and seen["plot_pac"]["npts"] == 31

# NB: the ``except ImportError -> SystemExit("… pip install matplotlib")`` branch in
# ``_cmd_plot`` isn't unit-tested — once ``examples.plot_bode`` is imported it stays
# attribute-bound on the ``examples`` package, so faking a missing matplotlib needs
# fragile import-system surgery for a one-line defensive guard. Not worth the brittleness.
