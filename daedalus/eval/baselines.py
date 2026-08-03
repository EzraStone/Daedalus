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
    strict — a lenient parser would silently repair the baseline's output and
    make the comparison meaningless.
    """
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
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
