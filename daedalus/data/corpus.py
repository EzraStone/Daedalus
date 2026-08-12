"""Building the corpus, and splitting it honestly.

Order of operations matters here and is not negotiable: sample a spec,
canonicalise it, place and route it, **verify it**, and only then keep it. The
placer is buggy — every placer is — and the verifier is what catches it. The
discard rate is logged as a health metric rather than swallowed, because a
placer that quietly starts failing looks exactly like a corpus that quietly
gets smaller.

Splits are stratified by gate count and held out as whole buckets rather than
as random samples. Random splits measure interpolation. Holding out the 5-6
gate bucket entirely measures whether the model composes, which is the only
question worth asking of a generative model of circuits.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..grid import Grid
from ..redsim import Pass, Verifier
from ..synth import compile_many
from ..synth.place import Stats
from .paraphrase import paraphrase
from .sample import SampleConfig, sample_unique


@dataclass(slots=True)
class Example:
    """One verified (spec, layout) pair, with its language views."""

    spec_source: str
    spec_hash: str
    gates: int
    n_inputs: int
    n_outputs: int
    rows: list[int]
    input_z: list[int]
    output_z: list[int]
    #: The grid, as CELLS token ids.
    tokens: list[int]
    latency_rt: int
    blocks: int
    bbox: list[int]
    prompts: list[str]

    def grid(self) -> Grid:
        return Grid.from_tokens(self.tokens)


@dataclass(slots=True)
class SplitSpec:
    """One row of §04's split table."""

    name: str
    gate_lo: int
    gate_hi: int
    n_specs: int
    layouts_per_spec: int = 1
    purpose: str = ""


#: The default plan. Sizes here are the shape of §04's table scaled to
#: something that runs in a smoke test; `scale` multiplies them.
DEFAULT_SPLITS = (
    SplitSpec("train", 1, 4, 200, 6, "Main corpus. Unique specs x several layouts each."),
    SplitSpec("val", 1, 4, 40, 2, "Unseen specs, seen difficulty. Standard generalisation."),
    SplitSpec("test-extrap", 5, 6, 25, 2, "Harder than anything in training. The headline number."),
    SplitSpec("test-repair", 1, 4, 25, 1, "Working circuits, corrupted later. Measures inpainting."),
)


@dataclass(slots=True)
class BuildReport:
    """Everything needed to tell whether a corpus build went well."""

    started: float = field(default_factory=time.time)
    seconds: float = 0.0
    splits: dict[str, dict] = field(default_factory=dict)
    synth: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "seconds": round(self.seconds, 2),
            "splits": self.splits,
            "synth": self.synth,
        }


def build_split(
    split: SplitSpec,
    verifier: Verifier,
    rng: random.Random,
    seen_specs: set[int],
    cfg: SampleConfig,
    stats: Stats,
    attempts_each: int = 12,
) -> tuple[list[Example], dict]:
    """Sample, compile, verify and describe one split."""
    specs = sample_unique(
        rng,
        split.n_specs,
        cfg.with_gates(split.gate_lo, split.gate_hi),
        seen=seen_specs,
    )
    examples: list[Example] = []
    specs_with_a_layout = 0
    for spec in specs:
        attempts = compile_many(
            spec,
            verifier,
            count=split.layouts_per_spec,
            rng=rng,
            attempts_each=attempts_each,
            stats=stats,
        )
        if not attempts:
            continue
        specs_with_a_layout += 1
        prompts = paraphrase(spec, rng, count=rng.randint(4, 8))
        for a in attempts:
            assert isinstance(a.verdict, Pass), "compile_many must only return verified layouts"
            examples.append(
                Example(
                    spec_source=spec.source(),
                    spec_hash=spec.key(),
                    gates=spec.gates,
                    n_inputs=spec.n_inputs,
                    n_outputs=spec.n_outputs,
                    rows=list(spec.rows),
                    input_z=list(a.placed.input_z),
                    output_z=list(a.placed.output_z),
                    tokens=a.grid.tokens(),
                    latency_rt=a.verdict.latency_rt,
                    blocks=a.verdict.blocks,
                    bbox=list(a.verdict.bbox),
                    prompts=prompts,
                )
            )

    layouts = [e for e in examples]
    summary = {
        "specs_sampled": len(specs),
        "specs_with_a_layout": specs_with_a_layout,
        "spec_yield": round(specs_with_a_layout / max(len(specs), 1), 3),
        "examples": len(layouts),
        "layouts_per_spec": round(len(layouts) / max(specs_with_a_layout, 1), 2),
        "gate_range": [split.gate_lo, split.gate_hi],
        "mean_blocks": round(sum(e.blocks for e in layouts) / max(len(layouts), 1), 1),
        "mean_latency_rt": round(sum(e.latency_rt for e in layouts) / max(len(layouts), 1), 2),
        "distinct_layouts": len({tuple(e.tokens) for e in layouts}),
        "purpose": split.purpose,
    }
    return examples, summary


def build(
    out_dir: str | Path,
    seed: int = 0,
    scale: float = 1.0,
    splits=DEFAULT_SPLITS,
    cfg: SampleConfig | None = None,
    verifier: Verifier | None = None,
) -> BuildReport:
    """Build a corpus on disk.

    Every split draws from a shared set of seen spec hashes, so a spec in
    training can never reappear in validation. That is the whole point of
    hashing by behaviour rather than by text: two different ways of writing the
    same function would otherwise land on both sides of the split and turn
    generalisation into recall.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or SampleConfig()
    report = BuildReport()
    stats = Stats()
    seen: set[int] = set()

    own_verifier = verifier is None
    v = verifier or Verifier()
    if own_verifier:
        v.start()
    try:
        for split in splits:
            scaled = SplitSpec(
                split.name,
                split.gate_lo,
                split.gate_hi,
                max(1, round(split.n_specs * scale)),
                split.layouts_per_spec,
                split.purpose,
            )
            rng = random.Random(f"{seed}:{split.name}")
            examples, summary = build_split(scaled, v, rng, seen, cfg, stats)
            path = out / f"{split.name}.jsonl"
            with path.open("w") as fh:
                for e in examples:
                    fh.write(json.dumps(asdict(e), separators=(",", ":")) + "\n")
            summary["path"] = str(path)
            report.splits[split.name] = summary
    finally:
        if own_verifier:
            v.close()

    report.synth = stats.as_dict()
    report.seconds = time.time() - report.started
    (out / "report.json").write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    _write_dataset_card(out, report)
    return report


def load(path: str | Path) -> list[Example]:
    """Read a split back."""
    out = []
    with Path(path).open() as fh:
        for line in fh:
            if line.strip():
                out.append(Example(**json.loads(line)))
    return out


def corrupt(example: Example, rng: random.Random, blocks: int = 4) -> tuple[list[int], list[int]]:
    """Damage a working circuit, for the repair split.

    Returns the corrupted tokens and the indices that were changed. Corruption
    only touches cells above the substrate and away from the ports: knocking
    out a port would make the grid malformed rather than broken, which measures
    a different thing.
    """
    from .. import vocab as V

    tokens = list(example.tokens)
    protected = set()
    for z in example.input_z:
        protected |= {V.index(0, V.LOGIC_Y, z), V.index(1, V.LOGIC_Y, z)}
    for z in example.output_z:
        protected |= {V.index(V.SX - 1, V.LOGIC_Y, z), V.index(V.SX - 2, V.LOGIC_Y, z)}

    candidates = [
        i
        for i in range(V.CELLS)
        if i not in protected and V.unindex(i)[1] >= V.LOGIC_Y and tokens[i] != V.AIR
    ]
    if not candidates:
        return tokens, []
    hit = rng.sample(candidates, min(blocks, len(candidates)))
    for i in hit:
        # Removing a block is the commonest real-world damage, and it is the
        # one a repair model has to handle; replacing with a random block is
        # the harder case and worth some of the budget.
        tokens[i] = V.AIR if rng.random() < 0.7 else rng.choice(
            [V.WIRE, V.SOLID, V.torch(V.Attach.FLOOR)]
        )
    return tokens, sorted(hit)


def _write_dataset_card(out: Path, report: BuildReport) -> None:
    """A dataset card, because a corpus without one is not a release."""
    lines = [
        "# Daedalus corpus",
        "",
        "Procedurally generated redstone circuits. Every layout in this corpus",
        "was placed by `daedalus.synth`, then **verified** by the `redsim`",
        "simulator before being written out; nothing here is unchecked.",
        "",
        "## Provenance",
        "",
        "Fully synthetic. No community schematics were scraped, so there is no",
        "licensing ambiguity and no unlabelled data. Labels are derived from the",
        "generating spec rather than inferred, so they are exact by construction.",
        "",
        "## Splits",
        "",
        "| split | gates | examples | specs | mean blocks | mean latency |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in report.splits.items():
        lines.append(
            f"| {name} | {s['gate_range'][0]}-{s['gate_range'][1]} | {s['examples']} | "
            f"{s['specs_with_a_layout']} | {s['mean_blocks']} | {s['mean_latency_rt']} rt |"
        )
    lines += [
        "",
        "Splits are held out by whole gate-count bucket, not at random. A random",
        "split measures interpolation; holding out the 5-6 gate bucket measures",
        "whether a model composes.",
        "",
        "Specs are deduplicated by *semantic* hash — the truth table — so two",
        "different ways of writing the same function cannot land on both sides",
        "of a split.",
        "",
        "## Known limits",
        "",
        "- Combinational logic only. 16x6x16. No pistons, no quasi-connectivity.",
        "- Elevated crossings require a clear seven-cell span and are capped at",
        "  two per layout, so dense crossbar functions remain under-represented.",
        "- Natural-language prompts are template-generated by default. They are a",
        "  view of the spec, never its source of truth.",
        "",
        "## Fields",
        "",
        "`spec_source`, `spec_hash`, `gates`, `n_inputs`, `n_outputs`, `rows`,",
        "`input_z`, `output_z`, `tokens` (1536 block-state ids in y-z-x order),",
        "`latency_rt`, `blocks`, `bbox`, `prompts`.",
        "",
    ]
    (out / "DATASET_CARD.md").write_text("\n".join(lines))
