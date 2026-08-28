"""Metrics.

The rule that governs this module: **aggregate token accuracy is never the
headline.** With 85% of every grid being air it reads around 97% while the
model produces nothing that works. Every number here is either functional or
per-class, and the ones that could flatter the model are reported next to the
ones that cannot.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..grid import Grid
from ..redsim import Pass, Verdict, Verifier
from ..spec import PlacedSpec


@dataclass(slots=True)
class SpecResult:
    """What a method produced for one spec."""

    verdicts: list[Verdict]
    passing: list[int] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        return bool(self.passing)

    def best(self) -> Pass | None:
        best = None
        for i in self.passing:
            v = self.verdicts[i]
            assert isinstance(v, Pass)
            if best is None or (v.blocks, v.latency_rt) < (best.blocks, best.latency_rt):
                best = v
        return best


def grade(candidates, placed: PlacedSpec, verifier: Verifier) -> SpecResult:
    verdicts = verifier.evaluate_batch(candidates, placed)
    return SpecResult(
        verdicts=verdicts,
        passing=[i for i, v in enumerate(verdicts) if isinstance(v, Pass)],
    )


def pass_at_k(results: list[SpecResult], k: int) -> float:
    """Fraction of specs with at least one verified-correct sample in ``k`` draws.

    Reported at 1, 8 and 64 because the *gap* is the diagnostic: a small gap
    means the model cannot do these specs; a large one means it can but is
    badly calibrated, and those call for opposite fixes.
    """
    if not results:
        return 0.0
    solved = sum(1 for r in results if any(i < k for i in r.passing))
    return round(solved / len(results), 4)


def verdict_rates(results: list[SpecResult]) -> dict[str, float]:
    """How candidates fail, not just how often.

    Malformed and unstable are tracked apart from plain failure because they
    mean different things: malformed means the model has not learned physical
    validity, and should approach zero fast; a rising unstable rate during
    self-training is an early collapse warning.
    """
    counts: dict[str, int] = {}
    total = 0
    for r in results:
        for v in r.verdicts:
            counts[v.kind] = counts.get(v.kind, 0) + 1
            total += 1
    return {k: round(c / total, 4) for k, c in sorted(counts.items())} if total else {}


def compactness_ratio(results: list[SpecResult], reference: dict[int, int], keys) -> float | None:
    """``blocks(model) / blocks(procedural reference)``, over solved specs.

    Below 1.0 means the model beat the compiler that taught it. This is the
    central empirical claim of the project, so it is measured only on specs
    *both* methods solved — otherwise the model could win by failing on
    everything expensive.
    """
    ratios = []
    for result, key in zip(results, keys):
        best = result.best()
        ref = reference.get(key)
        if best is None or not ref:
            continue
        ratios.append(best.blocks / ref)
    return round(statistics.fmean(ratios), 4) if ratios else None


def latency_ratio(results: list[SpecResult], reference: dict[int, int], keys) -> float | None:
    """The same for propagation delay. Sometimes trades against compactness,
    which is why §07 asks for a Pareto front rather than a single number."""
    ratios = []
    for result, key in zip(results, keys):
        best = result.best()
        ref = reference.get(key)
        if best is None or not ref:
            continue
        ratios.append(best.latency_rt / ref)
    return round(statistics.fmean(ratios), 4) if ratios else None


def novelty(results: list[SpecResult], candidates_per_spec, training_layouts: set[bytes]) -> float:
    """``1 - fraction isomorphic to a training layout``.

    Guards against the whole thing being a retrieval system in a trenchcoat. A
    model with a high pass rate and zero novelty has memorised the corpus, and
    that is worth knowing before anyone writes it up.
    """
    total = 0
    fresh = 0
    for result, cands in zip(results, candidates_per_spec):
        for i in result.passing:
            total += 1
            if bytes(cands[i]) not in training_layouts:
                fresh += 1
    return round(fresh / total, 4) if total else 0.0


def diversity(results: list[SpecResult], candidates_per_spec) -> float:
    """Distinct verified layouts per solved spec.

    A first-class tracked metric, not a curiosity. Self-training on filtered
    samples collapses onto one layout per spec, and when it does, pass@1 keeps
    rising — so this is the number that notices.
    """
    counts = []
    for result, cands in zip(results, candidates_per_spec):
        if not result.passing:
            continue
        counts.append(len({bytes(cands[i]) for i in result.passing}))
    return round(statistics.fmean(counts), 3) if counts else 0.0


def repair_success(results: list[SpecResult]) -> float:
    """Corrupted circuit correct after a single inpainting pass.

    Likely the best demo in the project: it is the thing a player with a broken
    build actually wants, and no existing tool offers it.
    """
    return pass_at_k(results, 1)


def per_class_accuracy(predicted, target) -> dict[int, float]:
    """Token accuracy broken out by block state.

    The aggregate is meaningless here — see the module docstring. Per class it
    is genuinely informative: a model that gets air right and torches wrong is
    a different problem from one that gets everything slightly wrong.
    """
    hits: dict[int, int] = {}
    totals: dict[int, int] = {}
    for p, t in zip(predicted, target):
        totals[t] = totals.get(t, 0) + 1
        if p == t:
            hits[t] = hits.get(t, 0) + 1
    return {t: round(hits.get(t, 0) / n, 4) for t, n in sorted(totals.items())}


def pareto_front(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Non-dominated ``(blocks, latency)`` pairs.

    The figure §07 says will get the project shared: one cloud per method, every
    point a verified-correct circuit. If the model's cloud sits down-left of the
    compiler's, the thesis is demonstrated in one image with no prose.
    """
    front: list[tuple[int, int]] = []
    for p in sorted(set(points)):
        if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in points):
            front.append(p)
    return front


def summarise(
    results: list[SpecResult],
    candidates_per_spec,
    training_layouts: set[bytes] | None = None,
    reference_blocks: dict[int, int] | None = None,
    reference_latency: dict[int, int] | None = None,
    keys=None,
) -> dict:
    """One row of the results table."""
    out: dict = {
        "specs": len(results),
        "pass@1": pass_at_k(results, 1),
        "pass@8": pass_at_k(results, 8),
        "pass@64": pass_at_k(results, 64),
        "diversity": diversity(results, candidates_per_spec),
        **{f"rate_{k}": v for k, v in verdict_rates(results).items()},
    }
    if training_layouts is not None:
        out["novelty"] = novelty(results, candidates_per_spec, training_layouts)
    if reference_blocks and keys is not None:
        out["compactness_ratio"] = compactness_ratio(results, reference_blocks, keys)
    if reference_latency and keys is not None:
        out["latency_ratio"] = latency_ratio(results, reference_latency, keys)
    solved = [r.best() for r in results if r.best()]
    if solved:
        out["mean_blocks"] = round(statistics.fmean(v.blocks for v in solved), 2)
        out["mean_latency_rt"] = round(statistics.fmean(v.latency_rt for v in solved), 2)
        out["pareto"] = pareto_front([(v.blocks, v.latency_rt) for v in solved])
    return out


def grid_of(tokens) -> Grid:
    return Grid.from_tokens(tokens)


def grade_repairs(method, tasks, verifier: Verifier, k: int = 4) -> dict:
    """Score a repair method over damaged circuits.

    Two numbers, because passing is not the whole claim. A method that rebuilds
    the circuit correctly but rewrites half the grid has not repaired anything
    -- it has regenerated, using the damage as an excuse -- and a player who
    wanted their build back would not accept it.

    ``touched_outside`` counts cells changed beyond the damaged region, over
    the repairs that actually verified. Zero is what repair means.
    """
    results, outside, respected = [], [], 0
    for task in tasks:
        candidates = method(task, k)
        if not candidates:
            results.append(SpecResult(verdicts=[]))
            continue
        result = grade(candidates, task.placed, verifier)
        results.append(result)
        allowed = set(task.hit)
        for i in result.passing:
            changed = {
                cell
                for cell, (a, b) in enumerate(zip(candidates[i], task.original))
                if a != b
            }
            stray = changed - allowed
            outside.append(len(stray))
            respected += not stray
    return {
        "tasks": len(tasks),
        "repaired": repair_success(results),
        "mean_cells_touched_outside": (
            round(statistics.fmean(outside), 3) if outside else 0.0
        ),
        "repairs_that_stayed_inside": respected,
    }
