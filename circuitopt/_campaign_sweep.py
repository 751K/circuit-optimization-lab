"""Shared compiled-campaign dispatch for the batch workflows (rewrite R5-D).

The production sweeps (design-space ``bench_sweep``, dataset build, mismatch MC)
evaluate a *matrix* of candidates. Under the rust engine the compiled campaign
(:mod:`circuitopt._rust_campaign`) runs that matrix — device build, DC, AC,
noise — entirely in Rust under one ``py.detach``, with a single Rayon pool sized
to ``workers`` and **no per-candidate Python callback** (the GIL is released for
the whole batch, so ``workers`` scales). This module is the one place that

  * decides whether a circuit is campaign-able and which device family it is,
  * carries the R5-D cold-DC safety policy, and
  * marshals a list of size dicts into candidates and back.

**Cold-DC safety policy** (see ``tests/test_campaign_cold_dc.py``, the behaviour
gate): the silicon BSIM4 5T OTAs are monostable, so the compiled circuit
Newton reaches the *same physical branch* as the frozen scipy ``fsolve`` path
cold (no seed) — worst-case node agreement ~2e-5 V, convergence-rate identical.
The AFE OTFT is multistable: a *cold* circuit Newton can select a different
branch than ``fsolve`` (observed ~tens of volts apart), so an AFE size-sweep may
only be routed through the campaign when a consistent DC seed is supplied
(``corners.mismatch_mc`` seeds every sample from the shared nominal op). A caller
that has no seed for the AFE family must stay on the scalar reference path.

When the engine is not rust, or the extension lacks ``CompiledCampaign``, or the
circuit is not campaign-able, :func:`make_sweep_campaign` returns ``None`` and the
caller keeps its frozen scalar path (the reference/fallback). No result key,
CLI flag, or JSON contract changes — this only swaps the batch executor.
"""
from __future__ import annotations

from typing import Any, Sequence

from . import diagnostics
from ._engine import current_engine

# Sentinel for "caller did not pass a corner" so ``candidate`` keeps stamping the
# family nominal corner (the AFE/size-sweep default) while an explicit
# ``corner=<name>`` (e.g. a silicon ``corner_table`` arm) can override it.
_NOMINAL = object()


def campaign_enabled() -> bool:
    """True iff the rust engine is active and exposes ``CompiledCampaign``."""
    if current_engine() != "rust":
        return False
    try:
        import circuitopt_core
    except Exception:  # noqa: BLE001 - availability probe
        return False
    return hasattr(circuitopt_core, "CompiledCampaign")


class SweepCampaign:
    """Uniform size-sweep front to the AFE OTFT / silicon compiled campaigns.

    ``family`` is ``"afe_otft"`` or ``"silicon_bsim4"``; ``nominal_corner`` is the
    corner a ``corner=None`` scalar build resolves to (``None`` for AFE, whose
    ``candidate`` treats ``None`` as the nominal process shift). ``needs_seed`` is
    ``True`` for the multistable AFE family — the caller must pass a DC seed for
    correct (non-branch-swapping) results.
    """

    def __init__(self, core, family: str, nominal_corner: str | None, needs_seed: bool):
        self._core = core
        self.family = family
        self.nominal_corner = nominal_corner
        self.needs_seed = needs_seed

    def candidate(self, sizes, *, corner=_NOMINAL, bias=None, seed=None,
                  trust_seed_as_op: bool = False, mismatch=None, nf=None) -> dict:
        """One marshalled candidate.

        ``corner`` defaults to the family nominal (what a plain size-sweep stamps);
        pass an explicit corner name to place the candidate at a specific process
        corner (the silicon ``corner_table`` / ``mismatch_mc`` arms). For the
        silicon family ``mismatch`` is a ``{device: delvto_volts}`` map; for AFE it
        is the ``{device: {mvt0, mbeta0}}`` map. ``bias`` may partially override
        any named template bias for this candidate."""
        chosen = self.nominal_corner if corner is _NOMINAL else corner
        return self._core.candidate(sizes, chosen, mismatch=mismatch,
                                    nf=nf, bias=bias, seed=seed,
                                    trust_seed_as_op=trust_seed_as_op)

    @property
    def solved(self) -> tuple:
        """Solved-node names in DC/seed vector order (for ``dc_op`` reconstruction)."""
        return self._core.solved

    @property
    def band(self) -> tuple[float, float]:
        """Integrated-noise band carried by the immutable analysis template."""
        return self._core.band

    def seed_vector(self, dc_op) -> list[float]:
        """Solved-order DC seed vector from a ``{node: V}`` operating point."""
        return self._core.seed_vector(dc_op)

    def evaluate_batch(self, candidates: Sequence[dict], workers: int = 1,
                       analyses: Sequence[str] = ("dc", "ac", "noise")) -> list[dict]:
        """Run the compiled batch; results are candidate-index ordered."""
        return self._core.evaluate_batch(list(candidates), workers, list(analyses))

    @property
    def supports_prepared(self) -> bool:
        """Whether this family supports reusable DC/AC campaign stages."""
        return callable(getattr(self._core, "prepare_batch", None))

    def prepare_batch(self, candidates: Sequence[dict], workers: int = 1):
        """Prepare reusable DC, linearization, MNA, and forward-AC state."""
        if not self.supports_prepared:
            raise RuntimeError(
                f"campaign family {self.family!r} does not support prepared stages")
        return self._core.prepare_batch(list(candidates), workers)

    def reduce_result(self, row, sizes, bias, nf=None) -> dict:
        """Exact Python-side topology reductions over native campaign outputs."""
        return self._core.reduce_result(row, sizes, bias, nf)


def make_sweep_campaign(spec, freqs, band) -> SweepCampaign | None:
    """Build a :class:`SweepCampaign` for ``spec``, or ``None`` if not applicable.

    ``spec`` is a loaded :class:`circuitopt.circuit_loader.CircuitSpec`. The device
    family is inferred from its binding: an all-silicon ``model_types`` map ->
    silicon BSIM4; an empty map -> the AFE OTFT topology. Any construction failure
    (unsupported topology, mixed PDKs, missing cards) is swallowed to ``None`` so
    the caller transparently falls back to the scalar path.
    """
    if not campaign_enabled():
        return None
    try:
        binding = spec.binding()
        model_types = dict(binding.model_types or {})
        from .device_factory import is_silicon_model_types

        if is_silicon_model_types(model_types):
            from ._rust_campaign import BsimCampaign

            core = BsimCampaign(spec, freqs, band=tuple(band))
            return SweepCampaign(core, "silicon_bsim4", core.nominal_corner,
                                 needs_seed=False)
        if not model_types or not all(
                str(model).startswith("at4000tg.") for model in model_types.values()):
            return None
        from ._rust_campaign import AfeOtftCampaign

        # The compact Rust OTFT evaluator has one physical model implementation;
        # validate every explicit section/bin selector once before compiling so a
        # malformed binding cannot be silently ignored by the fast path.
        binding.build(spec.sizes)
        core = AfeOtftCampaign(spec.bias, freqs, band=tuple(band),
                               topo=spec.topology)
        return SweepCampaign(core, "afe_otft", None, needs_seed=True)
    except Exception as exc:  # noqa: BLE001 - fall back to the scalar reference
        diagnostics.note("campaign_sweep.build_fail", exc)
        return None


def silicon_campaign_for(topo, sizes, bias, nf, binding, freqs, band
                         ) -> SweepCampaign | None:
    """A silicon :class:`SweepCampaign` for a loose ``(topo, sizes, bias, nf)`` + binding.

    The corner/robustness entry points (``corners.corner_table`` /
    ``corners.mismatch_mc``) thread the six-param cluster by hand rather than a
    loaded ``CircuitSpec``, so this synthesises the minimal spec ``make_sweep_campaign``
    needs and returns its campaign **only when the circuit is all-silicon** (an
    explicit ``binding`` with a non-empty ``model_types``). Returns ``None`` for the
    explicitly bound OTFT AFE family, when no binding is supplied, or when the campaign is
    unavailable — the caller then keeps its frozen scalar path. AFE deliberately
    stays scalar here: the multistable OTFT would let a cold campaign under-report
    the latch rate (the R5-D red line), so only silicon (monostable, cold-DC
    consistent) is routed."""
    if binding is None or not campaign_enabled():
        return None
    from .device_factory import is_silicon_model_types

    if not is_silicon_model_types(binding.model_types):
        return None                       # explicitly bound OTFT -> scalar reference
    try:
        from .circuit_loader import CircuitSpec

        spec = CircuitSpec(name="_campaign", topology=topo, sizes=dict(sizes),
                           bias=dict(bias), nf=nf,
                           model_types=binding.model_types,
                           device_kwargs=binding.device_kwargs)
        camp = make_sweep_campaign(spec, freqs, band)
    except Exception as exc:  # noqa: BLE001 - fall back to the scalar reference
        diagnostics.note("campaign_sweep.silicon_build_fail", exc)
        return None
    if camp is None or camp.family != "silicon_bsim4":
        return None
    return camp


def campaign_for(topo, sizes, bias, nf, binding, freqs, band) -> SweepCampaign | None:
    """Build a compiled campaign for a loose fixed-topology circuit cluster.

    Unlike :func:`silicon_campaign_for`, this accepts both all-silicon BSIM4 and
    explicitly bound AT4000TG OTFT circuits. Callers must still honor
    ``campaign.needs_seed`` before routing a multistable AFE batch.
    """
    if binding is None or not campaign_enabled():
        return None
    try:
        from .circuit_loader import CircuitSpec

        spec = CircuitSpec(
            name="_campaign",
            topology=topo,
            sizes=dict(sizes),
            bias=dict(bias),
            nf=nf,
            model_types=binding.model_types or getattr(topo, "model_types", None),
            device_kwargs=(
                binding.device_kwargs or getattr(topo, "device_kwargs", None)
            ),
        )
        return make_sweep_campaign(spec, freqs, band)
    except Exception as exc:  # noqa: BLE001 - caller keeps its scalar fallback
        diagnostics.note("campaign_sweep.generic_build_fail", exc)
        return None


def evaluate_sizes(campaign: SweepCampaign, size_dicts: Sequence[Any], *,
                   workers: int = 1, analyses: Sequence[str] = ("dc", "ac", "noise"),
                   seeds: Sequence[Any] | None = None,
                   biases: Sequence[Any] | None = None) -> list[dict]:
    """Evaluate a list of size dicts through ``campaign`` -> index-ordered results.

    ``seeds`` (optional, one per size dict) supplies a ``{node: V}`` DC seed used
    verbatim as the operating point (``trust_seed_as_op=True``) — the mode that
    keeps the multistable AFE on the reference branch and isolates bit-exact
    AC/noise. ``biases`` optionally supplies one partial named bias mapping per
    candidate. When ``seeds`` is ``None`` the batch runs cold.
    """
    count = len(size_dicts)
    if seeds is not None and len(seeds) != count:
        raise ValueError(f"seeds length {len(seeds)} != candidate count {count}")
    if biases is not None and len(biases) != count:
        raise ValueError(f"biases length {len(biases)} != candidate count {count}")
    cands = []
    for index, sizes in enumerate(size_dicts):
        seed = None if seeds is None else seeds[index]
        bias = None if biases is None else biases[index]
        cands.append(campaign.candidate(
            sizes,
            bias=bias,
            seed=seed,
            trust_seed_as_op=seed is not None,
        ))
    return campaign.evaluate_batch(cands, workers=workers, analyses=analyses)
