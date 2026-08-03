# Daedalus

**Text-conditioned generation of Minecraft redstone circuits, with a simulator
in the training loop instead of a human.**

Existing voxel generators condition on *shape* — terrain, style, silhouette —
and succeed when a person thinks the result looks right. Redstone breaks that.
A circuit that looks like a circuit and does nothing is worthless, and the
difference between working and broken is often one block.

So the objective changes: **condition generation on function, and let a
simulator decide whether the sample was any good.**

That single change buys three things:

- **A dense, free, non-human reward signal.** No annotators, no aesthetic
  preference model. Every sample gets an exact pass/fail plus continuous scores
  for latency and footprint, in about 150 microseconds.
- **Unlimited perfectly-labelled training data.** Compose known-good primitives
  into random graphs and derive the label from the graph. No scraped
  schematics, no licensing ambiguity, no noisy captions.
- **A self-improvement loop with a hard floor.** Sample, simulate, keep only
  what verifies, retrain on that.

```
$ python -m daedalus compile "inputs A B
outputs Q
Q = !(A & B)"

-- y=1
................
................
..dddd..........
V#d..d.ddddddd>L
...#dd.d........
...t..dd........
...d>dd.........
...d............
...d............
...t............
...#............
...d............
..dd............
..d.............
V#d.............
................

PASS latency=3rt blocks=39 bbox=(16, 1, 13)
```

---

## Status, honestly

This is a **working foundation, not a finished result.** Here is exactly what
exists and what does not, because a README that blurs that line is worse than
no README.

### Built and tested

| Component | State |
|---|---|
| `crates/redsim` — the verifier | Complete. 104-case golden suite, 35 tests, ~153 µs per evaluation in release. |
| `daedalus.spec` — the DSL | Complete. Full grammar, canonicalisation, semantic hashing byte-identical to the Rust side. |
| `daedalus.synth` — the procedural compiler | Working. 88% coverage on the gate set the planar router supports. |
| `daedalus.data` — the corpus engine | Working end to end. Builds, verifies, splits, writes a dataset card. |
| `daedalus.schematic` — export | Complete. `.schem` and `.litematic`, dependency-free NBT. |
| `daedalus.eval` — metrics and baselines | Complete. Three of four baselines runnable today. |
| `daedalus.web` — the local window | Complete. Watch a spec get placed, routed and verified, step by step. |
| `daedalus.train.loop` — the §06 loop | Complete and tested against a stub sampler. |

### Written but not yet run

| Component | Why |
|---|---|
| `daedalus.models` — AR and diffusion | The code is complete and shaped correctly, but **no model has been trained**. There is no checkpoint, no loss curve, no pass@k. `torch` is an optional dependency and the environment this was built in has neither it nor a GPU. |
| `harness/mod` — the Fabric mod | Committed as source with the protocol pinned. Never compiled, never run: no Minecraft server available. |
| sim↔game agreement | **Not measured.** This is the number that would make everything else believable, and it does not exist yet. See [`docs/divergences.md`](docs/divergences.md). |

**So: every number in this repository is internally consistent, and none of it
has been validated against the real game.** That distinction is preserved
deliberately throughout, and closing it is the single highest-value next step.

---

## Quick start

```bash
cargo build --release -p redsim     # the verifier; everything depends on it
python -m daedalus selftest         # builds and verifies a NAND gate

python -m daedalus compile specs/nand.txt --out nand.schem
python -m daedalus corpus data/ --scale 0.1
python -m daedalus baselines --specs 25
```

### The window

```bash
pip install -e ".[web]"
python -m daedalus serve            # http://127.0.0.1:8765
```

A local page that shows the work rather than the result: the spec as parsed,
each placement attempt as it succeeds or fails, the routed grid, the truth
table, the verdict, and a schematic to download.

It takes the **formal DSL**, not English — there is no natural-language parser
and no trained model, and a chat box would be a lie about what the system can
do. The example picker is there to make the DSL learnable instead.

No Python dependencies beyond `pyyaml` for the gate library. `torch` is needed
only to train; `fastapi`/`uvicorn`/`websockets` only for the window.

Run the tests:

```bash
cargo test                          # 35 Rust tests, incl. the golden suite
python -m pytest tests/ -q          # Python
```

---

## How it works

```
                    ┌──────────────────────────────────┐
                    │  A. SPEC SPACE                   │
                    │  random logic DAG ──┐            │
                    │                     ├─► canonical│
                    │  NL paraphrase   ◄──┘   spec DSL │
                    └─────────────┬────────────────────┘
                                  │  (spec, layout) pairs
                    ┌─────────────▼────────────────────┐
                    │  B. LAYOUT SYNTHESISER           │
                    │  gate library + router → voxels  │
                    └─────────────┬────────────────────┘
                                  │  seed corpus
                    ┌─────────────▼────────────────────┐
                    │  C. GENERATOR                    │
      spec ────────►│  masked discrete diffusion over  │
                    │  16×16×6 block-state tokens      │
                    └─────────────┬────────────────────┘
                                  │  N candidate grids
                    ┌─────────────▼────────────────────┐
                    │  D. VERIFIER                     │
                    │  headless tick simulator         │
                    │  → truth table, latency, size    │
                    └──────┬───────────────────┬───────┘
                 pass │                   │ fail
        ┌───────────────▼──────┐   ┌──────▼─────────────────┐
        │ E. EXPORT            │   │ F. REPLAY BUFFER       │
        │ .schem / .litematic  │   │ near-misses, ranked    │
        └──────────────────────┘   └──────┬─────────────────┘
                    ▲                     │
                    └────── retrain ◄─────┘
```

The build order was not negotiable: **the verifier came first, alone, with a
golden suite, before a single line of model code.** It is the only component
with no upstream dependency and the one every other component's correctness is
defined against. If the simulator is wrong, every downstream number is a lie
nobody can detect.

That order paid off immediately. Three of the first hand-built golden fixtures
were wrong — not the simulator, the fixtures — because *dust approaching a
block sideways does not power it*. Finding that on day one rather than after
training a model is the entire argument for writing the verifier first.

## What the physics forced

Three rules shaped more of this codebase than any design decision:

**Adjacent dust is one net.** Two runs that touch have merged, and a merge is an
OR. So the router cannot merely avoid collisions the way a PCB router does — it
carries a one-cell moat around every net.

**Dust powers only what it points at.** A cell's connections decide which blocks
it weakly powers, and with two or more connections it points along them. A gate's
block is solid and can never be a connection, so the *only* way to power it is a
dust cell whose single connection is directly opposite. Every gate input is
therefore routed to as a dead end, and its other three sides are frozen so a
later net cannot give it a second connection.

**Signal dies after fifteen cells.** Long nets need a repeater — inserted only
where cutting the routing tree leaves every driver upstream, because a repeater
conducts one way and a stranded driver is silently disconnected.

The verifier caught every one of these as a bug before it caught them as a
feature.

## The known scope limit

The planar router cannot build netlists that need a **wire crossing**. When a
signal and its complement feed separate branches that later reconverge — XOR,
multiplexers — the netlist is a crossbar, and dust cannot cross dust on one
layer. This is a scope limit, not a bug, and it is pinned by a test
(`test_crossbar_netlists_are_a_known_gap`) so it stays visible rather than
dissolving into a mysterious discard rate.

XOR is compiled through `(a|b) & !(a&b)` instead of the obvious
`(a&!b) | (!a&b)`: one more inverter and one more tick, but it lays out flat.
Genuine crossings need a bridging router that uses the `y` axis, which is v2.

## Honest limitations

- **Combinational logic only.** No latches, counters, clocks or pistons.
- **16×16×6**, ports pinned to the `x=0` and `x=15` faces.
- **No quasi-connectivity.** Modelling it correctly triples simulator complexity
  and makes golden tests version-dependent.
- **Update order differs from Java's.** `redsim` latches and applies
  simultaneously; circuits that depend on Java's quasi-random order are
  *rejected*, not reconciled.
- **~26% of routed layouts fail verification** and are discarded. That is the
  placer being imperfect and the verifier doing its job; the rate is logged in
  every corpus build report rather than hidden.
- **Natural-language prompts are template-generated.** They are a view of the
  spec, never its source of truth. A hand-collected `test-nl` split is specified
  but not yet collected.

Full divergence list: [`docs/divergences.md`](docs/divergences.md).

## Measured today

From `python -m daedalus baselines`, on 6 sampled specs at k=4 — small, and
reported as such:

| method | pass@1 | mean blocks | note |
|---|---|---|---|
| procedural compiler | 0.83 | 33.8 | correct by construction, within coverage |
| retrieval | 0.17 | 16.0 | strong in distribution, as expected |
| unconditional + rejection | 0.00 | — | the honest floor |

The compiler beating everything is the expected and correct result. **A
compiler is correct by construction, and the learned model's argument was never
correctness** — it is compactness, natural-language input, repair, and layouts
outside what the placer can express. Those claims are unevaluated until a model
is trained.

## Repository layout

```
crates/redsim/            Rust verifier — build this first
  src/block.rs            48-token block-state vocabulary
  src/power.rs            multi-source BFS dust solver, weak/strong tiers
  src/tick.rs             latch → update → settle
  src/verdict.rs          truth-table comparison and diagnosis
  tests/golden/           104 hand-built circuits + expected verdicts
daedalus/
  vocab.py, grid.py       the token vocabulary, mirrored and parity-tested
  redsim.py               client for the verifier worker
  spec/                   DSL parser, canonicaliser, semantic hashing
  synth/                  gate library (YAML), placer, congestion-aware router
  data/                   corpus build, splits, paraphrase
  models/                 autoregressive baseline + masked discrete diffusion
  train/loop.py           §06 verifier-guided rounds
  eval/                   metrics + baselines
  schematic/              .schem / .litematic, dependency-free NBT
  web/                    the local window: FastAPI + one static page
harness/                  fidelity harness: compare.py + Fabric mod source
configs/                  25m-ar, 25m-mdm, loop-r5, corpus
docs/                     design spec, divergences, hardware notes
```

## Next steps, in order of value

1. **Measure sim↔game agreement.** Build the Fabric mod, run
   `harness/compare.py --cases 10000`, publish the number. Nothing else in this
   repository means much until this exists.
2. **Train the models.** The code is written; it needs a GPU and a corpus.
3. **Run the loop.** Five rounds, plot pass@1 and `layouts_per_spec` on the same
   axes — the second is the collapse alarm and the whole approach fails quietly
   without it.
4. **Bridging router.** Unlocks XOR, multiplexers and most of the interesting
   half of combinational logic.

## Licence

MIT.
