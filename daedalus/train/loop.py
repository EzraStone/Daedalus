"""Verifier-guided self-improvement.

Everything up to here is competent engineering. This is the part that is new,
and it is new for one reason: the reward is *exact and free*. There is no
annotator, no preference model, no aesthetic judgement. A circuit either
implements its truth table or it does not, and a Rust core can decide that in
under two hundred microseconds. So the model can be trained on its own
successful output indefinitely, with a monotone quality floor and a curve you
can plot.

Three things make the loop actually improve rather than merely not get worse:

**Compactness as a secondary objective.** Among passing candidates, prefer
fewer blocks and lower latency. This is what pushes the model past the
procedural placer that made its training data — the placer routes correctly
but wastefully, and the loop can find layouts its heuristics cannot express.
This is the central empirical claim of the whole project.

**Repair pairs from near-misses.** A candidate that fails exactly one
truth-table row is usually one or two blocks from correct. Diffed against a
known-good layout for the same spec, it becomes a masked-inpainting example:
the corrupted region masked, the correct blocks as the target. Failures become
supervision instead of waste.

**Curriculum by acceptance rate.** Advance the difficulty when round-level
pass@1 crosses a threshold. Too early and acceptance collapses and the loop
starves; too late and it overfits easy specs.

The thing that kills loops like this is diversity collapse: the model finds one
layout per gate type and emits it forever, with a beautiful pass rate and zero
novelty. Defended on three fronts here — dedupe accepted samples by canonical
hash, mix in a fixed fraction of original synthetic data every round, and track
distinct-layouts-per-spec as a first-class metric. If that trends down, the
loop is failing even while pass@1 rises.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from ..grid import Grid, diff_cells
from ..redsim import Fail, Pass, Verifier
from ..spec import PlacedSpec, Spec


class Sampler(Protocol):
    """Anything that can propose candidate grids for a spec.

    Deliberately narrow. The loop is the contribution, not the model, and
    keeping the interface to "give me k grids" means it can be exercised with a
    stub — which is how it gets tested without a GPU, and how a new model gets
    dropped in without touching the loop.
    """

    def __call__(self, spec: PlacedSpec, k: int) -> list[list[int]]: ...


class Trainer(Protocol):
    """Anything that can be fine-tuned on accepted pairs."""

    def __call__(self, dataset: TrainingSet, round_index: int) -> dict: ...


@dataclass(slots=True)
class Accepted:
    """A verified sample the loop decided to train on."""

    spec: Spec
    placed: PlacedSpec
    tokens: list[int]
    blocks: int
    latency_rt: int

    def cost(self) -> tuple[int, int]:
        """Ranking key: fewer blocks first, then lower latency."""
        return (self.blocks, self.latency_rt)


@dataclass(slots=True)
class RepairPair:
    """A near-miss and the correct layout it should have been."""

    spec: Spec
    placed: PlacedSpec
    broken: list[int]
    target: list[int]
    #: Cell indices that differ — the region a diffusion model would mask.
    damaged: list[int]


@dataclass(slots=True)
class TrainingSet:
    accepted: list[Accepted] = field(default_factory=list)
    repairs: list[RepairPair] = field(default_factory=list)
    #: Examples carried over from the original procedural corpus.
    anchors: list[Accepted] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.accepted) + len(self.repairs) + len(self.anchors)


@dataclass(slots=True)
class LoopConfig:
    rounds: int = 5
    specs_per_round: int = 20_000
    candidates_per_spec: int = 64
    keep_per_spec: int = 4
    guidance: float = 2.0
    temperature: float = 0.9
    #: Advance the curriculum when round pass@1 clears this.
    advance_threshold: float = 0.6
    #: Fraction of each round's training set drawn from the original corpus.
    #: Self-training on filtered samples is a known mode-collapse machine; this
    #: is the cheapest defence and the one that costs nothing to keep on.
    anchor_fraction: float = 0.25
    #: Gate-count buckets, advanced through as the model earns it.
    curriculum: tuple[tuple[int, int], ...] = ((1, 2), (1, 3), (1, 4), (2, 5), (3, 6))
    seed: int = 0


@dataclass(slots=True)
class RoundReport:
    """One row of the curve that makes this a research contribution."""

    round: int
    difficulty: tuple[int, int]
    specs: int
    candidates: int
    pass_at_1: float
    pass_at_k: float
    accepted: int
    repairs: int
    mean_blocks: float
    mean_latency_rt: float
    #: Distinct verified layouts per spec. The collapse alarm.
    layouts_per_spec: float
    novelty: float
    unstable_rate: float
    malformed_rate: float
    fail_rate: float
    seconds: float
    train: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["difficulty"] = list(self.difficulty)
        return d


def _ratio(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def run_round(
    round_index: int,
    specs: list[tuple[Spec, PlacedSpec]],
    sampler: Sampler,
    verifier: Verifier,
    cfg: LoopConfig,
    known_layouts: set[bytes],
    reference: dict[int, list[int]] | None = None,
) -> tuple[TrainingSet, RoundReport]:
    """Sample, verify, and decide what to keep.

    ``known_layouts`` is every layout the model has already been trained on.
    It is what the novelty metric is measured against, and it is what stops the
    loop congratulating itself for rediscovering its own training set.
    """
    started = time.time()
    out = TrainingSet()
    reference = reference or {}

    solved_at_1 = 0
    solved_at_k = 0
    n_candidates = 0
    verdict_counts = {"pass": 0, "fail": 0, "unstable": 0, "malformed": 0}
    novel = 0
    per_spec_layouts: list[int] = []

    for spec, placed in specs:
        candidates = sampler(placed, cfg.candidates_per_spec)
        if not candidates:
            continue
        verdicts = verifier.evaluate_batch(candidates, placed)
        n_candidates += len(verdicts)
        for v in verdicts:
            verdict_counts[v.kind] = verdict_counts.get(v.kind, 0) + 1

        passing = [
            Accepted(spec, placed, list(tokens), v.blocks, v.latency_rt)
            for tokens, v in zip(candidates, verdicts)
            if isinstance(v, Pass)
        ]
        if verdicts and isinstance(verdicts[0], Pass):
            solved_at_1 += 1
        if passing:
            solved_at_k += 1

            # Dedupe before ranking: a model that emits the same layout sixty
            # times has not found sixty answers, and counting it as such is
            # how a collapsing loop looks healthy.
            unique: dict[bytes, Accepted] = {}
            for a in passing:
                unique.setdefault(bytes(a.tokens), a)
            ranked = sorted(unique.values(), key=Accepted.cost)
            per_spec_layouts.append(len(unique))
            for a in ranked[: cfg.keep_per_spec]:
                key = bytes(a.tokens)
                if key not in known_layouts:
                    novel += 1
                out.accepted.append(a)

            # A known-good layout for this spec makes future near-misses
            # repairable, so remember the best one we have seen.
            best = ranked[0]
            prev = reference.get(spec.semantic_hash())
            if prev is None or best.blocks < sum(1 for t in prev if t):
                reference[spec.semantic_hash()] = best.tokens
        else:
            # Near-misses are the most informative failures: one wrong row is
            # usually one or two blocks from correct.
            near = _closest_miss(candidates, verdicts)
            target = reference.get(spec.semantic_hash())
            if near is not None and target is not None:
                damaged = diff_cells(Grid.from_tokens(near), Grid.from_tokens(target))
                if damaged:
                    out.repairs.append(
                        RepairPair(spec, placed, list(near), list(target), damaged)
                    )

    report = RoundReport(
        round=round_index,
        difficulty=(0, 0),
        specs=len(specs),
        candidates=n_candidates,
        pass_at_1=_ratio(solved_at_1, len(specs)),
        pass_at_k=_ratio(solved_at_k, len(specs)),
        accepted=len(out.accepted),
        repairs=len(out.repairs),
        mean_blocks=round(statistics.fmean([a.blocks for a in out.accepted]), 2)
        if out.accepted
        else 0.0,
        mean_latency_rt=round(statistics.fmean([a.latency_rt for a in out.accepted]), 2)
        if out.accepted
        else 0.0,
        layouts_per_spec=round(statistics.fmean(per_spec_layouts), 2) if per_spec_layouts else 0.0,
        novelty=_ratio(novel, len(out.accepted)),
        unstable_rate=_ratio(verdict_counts["unstable"], n_candidates),
        malformed_rate=_ratio(verdict_counts["malformed"], n_candidates),
        fail_rate=_ratio(verdict_counts["fail"], n_candidates),
        seconds=round(time.time() - started, 2),
    )
    return out, report


def _closest_miss(candidates, verdicts) -> list[int] | None:
    """The candidate that got the most rows right, if it got all but one."""
    best = None
    best_score = None
    for tokens, v in zip(candidates, verdicts):
        if not isinstance(v, Fail) or v.constraint is not None:
            continue
        score = len(v.mismatched_rows)
        if score == 0:
            continue
        if best_score is None or score < best_score:
            best, best_score = tokens, score
    # Only a single wrong row is worth calling a near-miss. Two or more and the
    # circuit is usually wrong in a structural way that inpainting a small
    # region cannot fix, and training on it teaches the model to make small
    # edits to fundamentally broken layouts.
    return list(best) if best is not None and best_score == 1 else None


def run(
    sampler: Sampler,
    verifier: Verifier,
    spec_source,
    cfg: LoopConfig = LoopConfig(),
    trainer: Trainer | None = None,
    anchors: list[Accepted] | None = None,
    out_dir: str | Path | None = None,
) -> list[RoundReport]:
    """Run the loop.

    ``spec_source(rng, count, difficulty)`` supplies specs at a given gate-count
    bucket; ``trainer`` fine-tunes on the accepted set and returns whatever it
    wants logged. Both are injected so the loop can be exercised end to end
    without a GPU — the acceptance logic is the part worth testing, and it does
    not need one.
    """
    rng = random.Random(cfg.seed)
    reports: list[RoundReport] = []
    known: set[bytes] = {bytes(a.tokens) for a in (anchors or [])}
    reference: dict[int, list[int]] = {}
    for a in anchors or []:
        reference.setdefault(a.spec.semantic_hash(), a.tokens)

    stage = 0
    for round_index in range(1, cfg.rounds + 1):
        difficulty = cfg.curriculum[min(stage, len(cfg.curriculum) - 1)]
        specs = spec_source(rng, cfg.specs_per_round, difficulty)
        dataset, report = run_round(
            round_index, specs, sampler, verifier, cfg, known, reference
        )
        report.difficulty = difficulty

        if anchors and cfg.anchor_fraction > 0:
            n = int(len(dataset.accepted) * cfg.anchor_fraction / max(1 - cfg.anchor_fraction, 1e-6))
            dataset.anchors = rng.sample(anchors, min(n, len(anchors)))

        if trainer is not None and len(dataset):
            report.train = trainer(dataset, round_index) or {}

        known |= {bytes(a.tokens) for a in dataset.accepted}
        reports.append(report)

        # Advance only on merit. Advancing on a schedule is how the loop
        # starves: acceptance collapses to near zero and there is nothing left
        # to train on.
        if report.pass_at_1 >= cfg.advance_threshold and stage < len(cfg.curriculum) - 1:
            stage += 1

        if out_dir:
            path = Path(out_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "rounds.jsonl").write_text(
                "\n".join(json.dumps(r.as_dict()) for r in reports) + "\n"
            )
    return reports


def collapse_warning(reports: list[RoundReport]) -> str | None:
    """Read the curve for the failure mode that looks like success.

    Rising pass@1 with falling diversity is a collapsing loop, and it is the
    single most likely way this whole approach fails. Checking for it
    automatically means nobody has to remember to look.
    """
    if len(reports) < 3:
        return None
    recent = reports[-3:]
    rising = recent[-1].pass_at_1 > recent[0].pass_at_1
    diversity_falling = recent[-1].layouts_per_spec < recent[0].layouts_per_spec * 0.7
    novelty_falling = recent[-1].novelty < recent[0].novelty * 0.5
    if rising and (diversity_falling or novelty_falling):
        return (
            f"pass@1 rose {recent[0].pass_at_1:.2f} -> {recent[-1].pass_at_1:.2f} while "
            f"layouts/spec fell {recent[0].layouts_per_spec:.2f} -> "
            f"{recent[-1].layouts_per_spec:.2f} and novelty fell "
            f"{recent[0].novelty:.2f} -> {recent[-1].novelty:.2f}. "
            "The loop is collapsing onto one layout per spec."
        )
    if recent[-1].unstable_rate > recent[0].unstable_rate * 2 + 0.01:
        return (
            f"unstable rate is climbing ({recent[0].unstable_rate:.3f} -> "
            f"{recent[-1].unstable_rate:.3f}); the model is drifting toward oscillators."
        )
    return None
