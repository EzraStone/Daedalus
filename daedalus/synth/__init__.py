"""The procedural compiler: spec in, verified layout out.

This is both the thing that builds the training corpus and the first baseline
of §07. It will beat the learned model on correctness — a compiler is correct
by construction within its coverage — and that is the point of keeping it. The
model's claim is not correctness; it is compactness, natural-language input,
repair, and layouts outside what this placer can express.

Nothing here trusts itself. Every layout it produces goes through the verifier
before it is called a success, because a buggy placer that emits plausible
grids is exactly the failure mode a synthetic corpus is prone to.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..grid import Grid
from ..redsim import Pass, Verdict, Verifier
from ..spec import PlacedSpec, Spec
from .library import Library, load
from .netlist import Netlist, NetlistError, compile_netlist
from .place import RoutingFailure, Stats, Synthesiser, synthesise

__all__ = [
    "Attempt",
    "Library",
    "Netlist",
    "NetlistError",
    "RoutingFailure",
    "Stats",
    "Synthesiser",
    "compile",
    "compile_netlist",
    "compile_many",
    "load",
    "synthesise",
]


@dataclass(slots=True)
class Attempt:
    """One trip through the compiler, successful or not."""

    grid: Grid | None
    verdict: Verdict | None
    placed: PlacedSpec | None
    stage: str  # "ok" | "netlist" | "placement" | "routing" | "signal" | "verify"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.stage == "ok"


def compile(  # noqa: A001 - the domain word is the right one here
    spec: Spec,
    verifier: Verifier,
    rng: random.Random | None = None,
    attempts: int = 8,
    library: Library | None = None,
    stats: Stats | None = None,
) -> Attempt:
    """Compile a spec into a verified layout.

    Placement is randomised, so a failure is often just bad luck with the
    ordering rather than an impossible spec; ``attempts`` re-rolls before
    giving up. The returned :class:`Attempt` records which stage failed, which
    is what makes the discard rate a diagnosis instead of a mystery.
    """
    rng = rng or random.Random()
    stats = stats if stats is not None else Stats()

    try:
        netlist = compile_netlist(spec)
    except NetlistError as e:
        stats.attempts += 1
        stats.note("netlist")
        return Attempt(None, None, None, "netlist", str(e))

    last: Attempt | None = None
    for _ in range(attempts):
        stats.attempts += 1
        placed = spec.default_placement(rng)
        try:
            grid = synthesise(netlist, placed, rng, library=library) if library else synthesise(
                netlist, placed, rng
            )
        except RoutingFailure as e:
            stats.note(e.stage)
            last = Attempt(None, None, placed, e.stage, e.detail)
            continue
        stats.placed += 1

        verdict = verifier.evaluate(grid, placed)
        if isinstance(verdict, Pass):
            stats.routed += 1
            return Attempt(grid, verdict, placed, "ok")
        stats.note("verify")
        last = Attempt(grid, verdict, placed, "verify", str(verdict))

    return last or Attempt(None, None, None, "placement", "no attempt got as far as routing")


def compile_many(
    spec: Spec,
    verifier: Verifier,
    count: int,
    rng: random.Random | None = None,
    attempts_each: int = 6,
    stats: Stats | None = None,
) -> list[Attempt]:
    """Produce up to ``count`` *distinct* verified layouts for one spec.

    Distinct layouts for the same spec are the augmentation that teaches the
    model there is more than one answer. Without it, the model learns a lookup
    table from spec to layout, which §04 warns is the obvious failure mode.
    """
    rng = rng or random.Random()
    stats = stats if stats is not None else Stats()
    seen: set[bytes] = set()
    out: list[Attempt] = []
    # Give up after a run of duplicates: past a point the placer has exhausted
    # the shapes it knows how to make for this spec.
    misses = 0
    while len(out) < count and misses < count * 3 + 6:
        attempt = compile(spec, verifier, rng, attempts=attempts_each, stats=stats)
        if not attempt.ok:
            misses += 1
            continue
        key = attempt.grid.to_bytes()
        if key in seen:
            misses += 1
            continue
        seen.add(key)
        out.append(attempt)
    return out
