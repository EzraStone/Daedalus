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
  for latency and footprint, in about 41 microseconds when batched the way the
  loop batches — see [`docs/benchmarks.md`](docs/benchmarks.md), and run
  `daedalus bench` to reproduce it. That figure is the cost of a *verdict*, not
  of a circuit: building one takes about 250 ms, of which the verifier is 0.1%.
  `daedalus bench --compiler` reports the split, and the distinction matters
  because cheap verification is what makes the training loop possible and has
  nothing to do with how fast a corpus builds.
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
| `crates/redsim` — the verifier | Complete. 104-case golden suite, 35 tests, 41 µs per evaluation batched (188 µs one at a time — the difference is pipe round-trip, not simulation). |
| `daedalus.spec` — the DSL | Complete. Full grammar, canonicalisation, semantic hashing byte-identical to the Rust side. |
| `daedalus.synth` — the procedural compiler | Working. Planar routing plus verified two-level bridges for crossbar netlists. About 5.5 specs/second; the yield gap, not the speed, is the open problem. |
| `daedalus.data` — the corpus engine | Working end to end. Builds, verifies, splits, writes a dataset card. |
| `daedalus.schematic` — export | Complete. `.schem` and `.litematic`, dependency-free NBT. |
| `daedalus.eval` — metrics and baselines | Complete. Three of four baselines runnable today. |
| `daedalus.web` — the local window | Complete. Watch a spec get placed, routed and verified, step by step. |
| `daedalus.train.loop` — the §06 loop | Complete. Runs end to end with a real model via `daedalus loop`; no model good enough to make the curve mean anything yet. |
| `daedalus.models` — AR and diffusion | Both train and sample on CPU. Checkpoints save and reload; samples clear every pre-simulation check and come back with real verdicts. |
| `daedalus.text` — prompt conditioning | Works end to end: `train --nl-slots 4`, `sample --prompt "..."`. It is **not** the frozen sentence encoder §05 asks for — see below. |

### Written but not yet run

| Component | Why |
|---|---|
| a model worth the name | The code runs, but **nothing has been trained at a useful size**. The largest run so far is 557K parameters for 48 steps on a CPU, which verifies nothing. There is no pass@k worth quoting. |
| `harness/mod` — the Fabric mod | Committed as source with the protocol pinned. Never compiled, never run: no Minecraft server available. |
| sim↔game agreement | **Not measured.** This is the number that would make everything else believable, and it does not exist yet. See [`docs/divergences.md`](docs/divergences.md). |

**So: every number in this repository is internally consistent, and none of it
has been validated against the real game.** That distinction is preserved
deliberately throughout, and closing it is the single highest-value next step.

---

## Quick start

```bash
cargo build --release -p redsim     # the verifier; everything depends on it
python -m daedalus doctor           # what is installed, what is missing
python -m daedalus selftest         # builds and verifies a NAND gate

python -m daedalus compile specs/nand.txt --out nand.schem
python -m daedalus compile specs/nand.txt --out nand.json   # re-checkable layout
python -m daedalus verify specs/nand.txt nand.json
python -m daedalus corpus data/ --scale 0.1
python -m daedalus baselines --specs 25
python -m daedalus bench                # verifier throughput
python -m daedalus bench --compiler     # end-to-end compiler throughput
python -m daedalus power specs/nand.txt nand.json   # where the signal goes
```

### Seeing why, not just whether

A verdict says a circuit is wrong. It does not say where the signal died,
which is the question anyone staring at a layout is actually asking. `power`
settles the circuit for each row of the truth table and draws dust as its
strength:

```
A=0 B=0  ->  Q=1   (settled after 2 game ticks)
. . . D C B A . . . . . . . . .
. . . E . . 9 9 8 7 6 5 4 3 ▶ ◍
. . . F ◆ . . A . . . . . . . .
⌐ . · · · · . C . . . . . . . .
```

The decay is the diagnostic. Redstone loses a step of strength per block, so
`F` down to `3` across thirteen cells and then a repeater lifting it back up
is a working run — and a run that reaches `0` before it arrives is a routing
bug you can see rather than infer. Both windows draw the same field: press
`p` in the terminal, or use the signal row on the page.

### Training

```bash
pip install -e ".[train]"
python -m daedalus train data/ --out runs/first
python -m daedalus sample runs/first/model.pt specs/nand.txt -k 8
python -m daedalus loop runs/first/model.pt --corpus data/ --rounds 5

# conditioned on the corpus paraphrases as well as the spec
python -m daedalus train data/ --nl-slots 4 --out runs/text
python -m daedalus sample runs/text/model.pt specs/nand.txt \
    --prompt "turn the lamp off when both levers are on"
```

`train` prints a validation loss alongside the training loss, and for the
diffusion model that is the only one worth reading: the objective draws a mask
rate per batch and scales by `1/t`, so the per-batch number swings by an order
of magnitude on a model that has not changed. `sample` sends every candidate
to the verifier and exits non-zero if none of them pass, which on an
undertrained model is the expected outcome and the point — the reward signal
is attached and reporting.

`loop` runs the §06 rounds: sample, verify, keep what passes, retrain, advance
the curriculum on merit. It prints the collapse warning on the way out, since
rising pass@1 with falling diversity is the failure mode that looks like
success.

**On the prompt path.** Every corpus example has always carried four to eight
paraphrases of its spec, and the prefix has always reserved slots for their
embeddings; nothing filled them until now. What fills them is a small encoder
trained jointly with the generator, over hashed word features — *not* the
frozen sentence encoder §05 specifies, which needs a pretrained model this
repository does not ship. The trade is that it costs nothing to install and
generalises only to phrasings like the corpus, which is exactly the
generalisation a frozen encoder would have brought. `daedalus/text.py` states
it in full. The path is now real and measurable; whether it works is a
question that needs a trained model to answer.

### The window

```bash
pip install -e ".[web]"
python -m daedalus serve            # http://127.0.0.1:8765

pip install -e ".[tui]"
python -m daedalus tui              # the same thing, in the terminal
```

Either one shows the work rather than the result: the spec as parsed, each
placement attempt as it succeeds or fails, the routed grid, the truth table,
the verdict, and a schematic to write out. Pick the browser or the terminal;
they run the same compiler and read the same glyphs and colours out of
`daedalus/render.py`, so they cannot drift into disagreeing about what a torch
looks like.

Both take the **formal DSL**, not English — there is no natural-language parser
and no trained model, and a chat box would be a lie about what the system can
do. The example picker is there to make the DSL learnable instead; both views
load the same files from `examples/`.

No Python dependencies beyond `pyyaml` for the gate library. `torch` is needed
only to train; `fastapi`/`uvicorn`/`websockets` only for the page, `textual`
only for the terminal.

Run the tests:

```bash
cargo test                          # 35 Rust tests, incl. the golden suite
python -m pytest tests/ -q          # 270 Python tests
python -m daedalus doctor           # or just ask what is missing
```

The generator tests skip themselves without `torch`, so a run that installs
only the base package reports fewer. CI runs them in a job of their own for
exactly that reason.

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

## Crossbar routing

When a planar search is blocked by another net, the router can now carry one
signal over the other. A bridge climbs from `y=1` to `y=3`, crosses above a
straight lower wire, and descends again. The lower wire remains electrically
independent, which is checked against the Rust verifier over the complete
two-input truth table.

Bridge discovery is deliberately conservative: the underpass must be straight,
the seven-cell span must be clear, and a layout may use at most two bridges.
Placement is still stochastic, so a difficult crossbar can require several
attempts. `Stats.bridged` exposes how many candidates reached the verifier with
an elevated span, and `test_crossbar_netlists_use_the_bridging_router` prevents
the feature from quietly regressing.

XOR is compiled through `(a|b) & !(a&b)` instead of the obvious
`(a&!b) | (!a&b)`: one more inverter and one more tick, but it lays out flat.
The alternate XOR form remains useful because it is smaller and easier to
route, while multiplexers and other genuine crossbars use the bridge fallback.

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
  synth/                  gate library, congestion-aware router, bridge geometry
  data/                   corpus build, splits, paraphrase
  models/                 autoregressive baseline + masked discrete diffusion
  train/loop.py           §06 verifier-guided rounds
  eval/                   metrics + baselines
  schematic/              .schem / .litematic, dependency-free NBT
  render.py               glyphs and colours, shared by both windows
  web/                    the local window: FastAPI + one static page
  tui.py                  the same window in the terminal, on Textual
harness/                  fidelity harness: compare.py + Fabric mod source
configs/                  25m-ar, 25m-mdm, loop-r5, corpus
docs/                     design spec, divergences, hardware notes
```

## Next steps, in order of value

1. **Measure sim↔game agreement.** Build the Fabric mod, run
   `harness/compare.py --cases 10000`, publish the number. Nothing else in this
   repository means much until this exists. Unchanged, and still first.
2. **Train a model at a size that means something.** The pipeline runs
   end to end — `daedalus train`, `sample`, `repair`, `loop` all work, and the
   plumbing is tested — but the largest run so far is 557K parameters for 48
   steps on a CPU. That verifies nothing, and it is supposed to. What is
   missing is a GPU and a few hours, not code.
3. **Then run the loop for real.** Five rounds, plot pass@1 and
   `layouts_per_spec` on the same axes — the second is the collapse alarm and
   the whole approach fails quietly without it. `daedalus loop` already prints
   the warning; nothing has yet given it anything to warn about.
4. **Attack the routing yield, and only the routing yield.** The discard rate
   has been broken down (see [`docs/benchmarks.md`](docs/benchmarks.md)) and it
   is not one problem. Two failure shapes — a net that cannot reach its
   inverter, and a net that cannot be joined into one region — are 151 of 173
   failures, and neither fires because the board is full. They are about pieces
   landing where the router cannot connect them once earlier nets are
   committed, which points at ordering and the greedy commit rather than at the
   volume. The fan-out gap was the other candidate; it is closed, and closing
   it moved the yield number not at all.
5. **Broaden bridge coverage.** Measure crossbar success by topology, then add
   compact vertical primitives for placements the conservative span rejects.
   The seven-cell span is expensive: see the yield figures in
   [`docs/benchmarks.md`](docs/benchmarks.md) for what XOR coverage costs.

## Licence

MIT.
