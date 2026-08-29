"""The four baselines, all run through the identical verifier.

Running everything through one verdict function is the whole point. A
comparison where each method is judged by its own criterion is not a
comparison, and in this domain there is no excuse for one — the criterion is
exact and costs 200 microseconds.

1. **Procedural compiler.** The machinery that made the training data. *It will
   beat the model on correctness, and the README should say so.* A compiler is
   correct by construction within its coverage. The model's argument is not
   correctness — it is compactness, natural language, repair, and layouts
   outside the compiler's expressible space.

2. **Frontier LLM, prompted.** Give a strong general model the spec and the
   grid format and ask for a schematic. Expect a low single-digit pass rate:
   LLMs have essentially no spatial model of signal propagation. A fairly run
   comparison here is a genuinely publishable observation.

3. **Retrieval.** Nearest neighbour over training specs, return that layout
   verbatim. Cheap and surprisingly strong in distribution. If the model cannot
   clearly beat retrieval on the extrapolation split, it is not a generative
   model of circuits.

4. **Unconditional + rejection.** Sample ignoring the spec, keep what verifies.
   An ablation, not a competitor: it isolates how much of the pass rate comes
   from conditioning rather than from the verifier filtering a firehose.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .. import vocab as V
from ..grid import Grid
from ..redsim import Verifier
from ..spec import PlacedSpec, Spec
from ..synth import compile as compile_spec


class Method(Protocol):
    """Everything under test presents the same face: k candidate grids."""

    name: str

    def __call__(self, spec: Spec, placed: PlacedSpec, k: int) -> list[list[int]]: ...


@dataclass
class ProceduralCompiler:
    """Baseline 1. Correct by construction, within its coverage."""

    verifier: Verifier
    rng: random.Random
    attempts: int = 15
    name: str = "compiler"

    def __call__(self, spec: Spec, placed: PlacedSpec, k: int) -> list[list[int]]:
        out = []
        for _ in range(k):
            attempt = compile_spec(
                spec,
                self.verifier,
                self.rng,
                attempts=self.attempts,
                fixed_placement=placed,
            )
            if attempt.ok:
                out.append(attempt.grid.tokens())
            if len(out) >= k:
                break
        return out


@dataclass
class Retrieval:
    """Baseline 3. Nearest neighbour by truth-table distance.

    Distance is Hamming over the truth table rather than embedding cosine,
    which makes this baseline *stronger* than the embedding version and
    therefore a harder thing for the model to beat. Picking a weak form of your
    own baseline is the easiest way to fool yourself.
    """

    #: ``(spec, tokens, input_z, output_z)``. The placement is stored because a
    #: layout built for different port rows is not an answer to this question --
    #: it verifies as a port violation, and counting that as a near miss would
    #: flatter the baseline.
    corpus: list[tuple[Spec, list[int], tuple[int, ...], tuple[int, ...]]]
    name: str = "retrieval"

    def __call__(self, spec: Spec, placed: PlacedSpec, k: int) -> list[list[int]]:
        want = (placed.input_z, placed.output_z)
        scored = []
        for other, tokens, in_z, out_z in self.corpus:
            if other.n_inputs != spec.n_inputs or (in_z, out_z) != want:
                continue
            distance = sum(bin(a ^ b).count("1") for a, b in zip(spec.rows, other.rows))
            scored.append((distance, tokens))
        scored.sort(key=lambda t: t[0])
        return [tokens for _d, tokens in scored[:k]]


@dataclass
class Unconditional:
    """Baseline 4. Sample without looking at the spec.

    Implemented as a plain random-block sampler when no model is given, which
    is the honest floor: it shows what the verifier alone accepts from noise.
    """

    rng: random.Random
    sampler: Callable[[int], list[list[int]]] | None = None
    density: float = 0.08
    name: str = "unconditional"

    def __call__(self, spec: Spec, placed: PlacedSpec, k: int) -> list[list[int]]:
        if self.sampler is not None:
            return self.sampler(k)
        out = []
        legal = [t for t in V.BLOCK_TOKENS if V.legal_at(t, V.LOGIC_Y)]
        for _ in range(k):
            grid = Grid.with_substrate()
            for x, y, z in placed.input_ports:
                grid.set(x, y, z, V.lever(V.Dir4.EAST))
                grid.set(x + 1, y, z, V.SOLID)
            for x, y, z in placed.output_ports:
                grid.set(x, y, z, V.LAMP)
            for z in range(V.SZ):
                for x in range(2, V.SX - 1):
                    if self.rng.random() < self.density:
                        grid.set(x, V.LOGIC_Y, z, self.rng.choice(legal))
            out.append(grid.tokens())
        return out


@dataclass
class ConstrainedRandom:
    """The floor a trained model actually has to clear.

    Same random blocks as :class:`Unconditional`, with the two rules the
    sampler enforces for free: a block that needs holding up gets a solid
    block put under it, and levers and lamps only appear at declared ports.
    Neither costs any training.

    This exists because "97% of random grids are malformed" makes the
    unconditional row look like a low bar that any model clears, when most of
    what clears it is not learning at all. Separating the two says how much of
    a model's well-formedness is the model and how much is the constraint,
    which is the only way the pass rate of a real generator means anything.
    """

    rng: random.Random
    density: float = 0.08
    name: str = "constrained-random"

    def __call__(self, spec: Spec, placed: PlacedSpec, k: int) -> list[list[int]]:
        from ..models.common import _support_offset

        del spec
        # Levers and lamps are excluded outright: their legality is decided by
        # the spec, not by physics, and one anywhere else is a port violation.
        placeable = [
            t
            for t in V.BLOCK_TOKENS
            if V.legal_at(t, V.LOGIC_Y) and V.decode(t).kind not in ("lever", "lamp")
        ]
        out = []
        for _ in range(k):
            grid = Grid.with_substrate()
            for x, y, z in placed.input_ports:
                grid.set(x, y, z, V.lever(V.Dir4.EAST))
                grid.set(x + 1, y, z, V.SOLID)
            for x, y, z in placed.output_ports:
                grid.set(x, y, z, V.LAMP)

            for z in range(V.SZ):
                for x in range(2, V.SX - 1):
                    if self.rng.random() >= self.density:
                        continue
                    token = self.rng.choice(placeable)
                    offset = _support_offset(token)
                    if offset is not None:
                        sx, sy, sz = x + offset[0], V.LOGIC_Y + offset[1], z + offset[2]
                        if not V.in_bounds(sx, sy, sz):
                            continue
                        # Do not prop a block up on a port cell: overwriting one
                        # trades a support failure for a port violation.
                        if (sx, sy, sz) in placed.input_ports or (
                            sx,
                            sy,
                            sz,
                        ) in placed.output_ports:
                            continue
                        grid.set(sx, sy, sz, V.SOLID)
                    grid.set(x, V.LOGIC_Y, z, token)
            out.append(grid.tokens())
        return out


@dataclass
class PromptedLLM:
    """Baseline 2. A frontier model, asked in words.

    The call is injected rather than built in: this repository does not ship an
    API client, a key, or a hidden network dependency. Supply ``call(prompt) ->
    text`` and the parser turns whatever comes back into grids, counting the
    ones that are not even parseable — because "the model emitted prose" is a
    result, and quietly dropping those would flatter the baseline.
    """

    call: Callable[[str], str]
    name: str = "prompted-llm"
    unparseable: int = 0

    def prompt(self, spec: Spec, placed: PlacedSpec) -> str:
        return (
            "Build a Minecraft redstone circuit on a 16x16 grid, one layer.\n"
            "Use these characters: '.' air, '#' solid block, 'd' redstone dust,\n"
            "'t' wall torch (attached to the block on its west side),\n"
            "'>' repeater facing east, 'V' lever, 'L' lamp.\n"
            f"Inputs are levers at x=0 on rows {list(placed.input_z)}, each with a\n"
            f"solid block at x=1. Outputs are lamps at x=15 on rows {list(placed.output_z)}.\n\n"
            f"Specification:\n{spec.source(ascii_only=True)}\n\n"
            f"Truth table:\n{spec.table()}\n\n"
            "Reply with exactly 16 lines of 16 characters and nothing else."
        )

    def __call__(self, spec: Spec, placed: PlacedSpec, k: int) -> list[list[int]]:
        out = []
        for _ in range(k):
            try:
                tokens = parse_ascii_layer(self.call(self.prompt(spec, placed)))
            except ValueError:
                self.unparseable += 1
                continue
            out.append(tokens)
        return out


_ASCII = {
    ".": V.AIR,
    "#": V.SOLID,
    "d": V.WIRE,
    "t": V.torch(V.Attach.WEST),
    ">": V.repeater(V.Dir4.EAST, 1),
    "V": V.lever(V.Dir4.EAST),
    "L": V.LAMP,
    "T": V.TARGET,
    "c": V.comparator(V.Dir4.EAST),
}


def parse_ascii_layer(text: str) -> list[int]:
    """Turn 16 lines of 16 characters into a grid.

    Only the logic layer; the substrate is added underneath. Deliberately
    strict about *blocks* — a lenient parser would silently repair the
    baseline's output and make the comparison meaningless — and deliberately
    forgiving about *packaging*. Prose before and after the grid, a fenced code
    block, and indentation are all dropped, because none of them changes a
    block. Counting an indented grid as unparseable would understate the
    baseline for a reason that has nothing to do with redstone.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    lines = [ln for ln in lines if all(c in _ASCII for c in ln)]
    if len(lines) != V.SZ or any(len(ln) != V.SX for ln in lines):
        raise ValueError(f"expected {V.SZ} lines of {V.SX} characters, got {len(lines)}")
    grid = Grid.with_substrate()
    for z, line in enumerate(lines):
        for x, ch in enumerate(line):
            grid.set(x, V.LOGIC_Y, z, _ASCII[ch])
    return grid.tokens()


def render_ascii_layer(grid: Grid) -> str:
    """The inverse, for building few-shot prompts from real circuits."""
    return grid.layer_text(V.LOGIC_Y)


@dataclass
class RepairTask:
    """One damaged circuit, and the grid it was damaged from.

    Kept apart from generation because it measures a different claim. A method
    that cannot generate a working circuit from nothing may still be able to
    put a broken one back, and that is the capability §06 says a player would
    actually reach for.
    """

    spec: Spec
    placed: PlacedSpec
    original: list[int]
    damaged: list[int]
    #: Cells the corruption touched, which is what a repairer is allowed to
    #: change. Anything outside them is not repair.
    hit: list[int]


def repair_tasks(
    corpus, rng: random.Random, count: int, blocks: int = 6
) -> list[RepairTask]:
    """Damage verified circuits, for measuring inpainting.

    ``corpus`` is the ``(spec, tokens, input_z, output_z)`` tuples the other
    baselines already take, so the same corpus can drive both.
    """
    from ..data.corpus import Example, corrupt

    out = []
    for spec, tokens, input_z, output_z in corpus[:count]:
        placed = spec.place(input_z, output_z)
        example = Example(
            spec_source=spec.source(),
            spec_hash=spec.key(),
            gates=spec.gates,
            n_inputs=spec.n_inputs,
            n_outputs=spec.n_outputs,
            rows=list(spec.rows),
            input_z=list(input_z),
            output_z=list(output_z),
            tokens=list(tokens),
            latency_rt=0,
            blocks=0,
            bbox=[0, 0, 0],
            prompts=[],
        )
        damaged, hit = corrupt(example, rng, blocks=blocks)
        out.append(RepairTask(spec, placed, list(tokens), damaged, list(hit)))
    return out


@dataclass
class OracleRepair:
    """The ceiling: put back exactly what was removed.

    Not a method anyone would ship -- it is handed the answer. It exists so
    repair_success has a 1.0 to sit under, and so a repair harness that scores
    zero can be told apart from one that is wired up wrong.
    """

    name: str = "oracle-repair"

    def __call__(self, task: RepairTask, k: int) -> list[list[int]]:
        return [list(task.original) for _ in range(k)]


@dataclass
class NoRepair:
    """The floor: hand the damage straight back.

    Anything scoring at this level has not repaired anything, and a corruption
    that leaves the circuit working would show up here as a suspiciously high
    number -- which is worth knowing about the damage, not the repairer.
    """

    name: str = "no-repair"

    def __call__(self, task: RepairTask, k: int) -> list[list[int]]:
        return [list(task.damaged) for _ in range(k)]
