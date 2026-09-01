# Where the simulator differs from Minecraft

`redsim` is a deterministic model of a **documented subset** of Java-edition
redstone. It is not a reimplementation, and being explicit about the boundary is
what separates a serious repository from a toy. Every item below is a place
where `redsim` and the real game can disagree, what it does instead, and why.

The fidelity harness (`harness/`) exists to keep this list honest. Anything that
disagrees and is not on this list is a bug.

## Modelled faithfully

These are implemented because getting them wrong changes the answer on circuits
people actually build.

| Rule | Behaviour |
|---|---|
| dust decay | Strength = max(neighbour) − 1, floor 0. Solved by multi-source BFS with a bucket queue — exact, and faster than iterating to a fixed point. |
| dust range | 15 blocks from a strength-15 source. Enforced by decay, not special-cased. |
| dust slopes | One-block up and down, with the roofing rule: a run cannot climb if the cell above it is opaque. |
| **dust pointing** | A dust cell weakly powers only the blocks it *points* at. Its pointing comes from its connections; with exactly one connection it renders as a line and points both ways along that axis. A run that passes a block does not power it. |
| **weak vs. strong power** | Two tiers. Only *strong* power re-emits into adjacent dust; weak power still switches torches and lights lamps. Historically the number one source of simulator/game divergence. |
| torch | Output = ¬(support block powered), 1 rt delay. Strongly powers the block above; weakly powers every other adjacent block, which is why two torches on neighbouring blocks interfere. |
| repeater | Restores to 15, directional, delay 1–4 rt, blocks reverse signal, and **locks** when a repeater or comparator faces into its side. |
| comparator | 1 rt delay. Compare: rear if rear ≥ max(sides) else 0. Subtract: max(0, rear − max(sides)). Sides ignore solid blocks, as in the real game. |
| lamp | Lit by either power tier. |

## Deliberate divergences

| Area | Java | `redsim` | Why |
|---|---|---|---|
| **update order** | Quasi-random, depends on chunk and update-queue order. Circuits that depend on it are called *locational*. | Every component latches its input from the pre-update field; all changes apply simultaneously. | Reproducing Java's order would make the simulator non-deterministic in the same way the game is, and every number downstream would be a distribution rather than a value. Instead, `evaluate` runs a Gray-code pass that detects circuits whose output depends on history and **rejects** them. A design that needs a specific update order is a design this project does not want. |
| **torch burnout** | A torch toggling faster than ~8 times in 60 game ticks goes dark and recovers later. | Detected and the circuit is **rejected** with `MALFORMED burnout`. | A design that burns out is a bad design. Simulating the recovery would let the loop accept circuits that work only until someone flips a lever twice. |
| **quasi-connectivity** | Present. A piston "senses" power diagonally below-adjacent. | Absent entirely. | Modelling QC correctly roughly triples simulator complexity and makes the golden tests version-dependent. v1 generates no pistons, so it cannot arise; if the harness ever reports it, something is emitting blocks outside the vocabulary. |
| **target block** | Comparator-readable, outputs a signal when hit by a projectile. | Behaves exactly like a solid block. | Container and projectile reading is v2. The token exists so a v2 model has something to attach to rather than needing a vocabulary change, which would invalidate every checkpoint. |
| **observers** | Edge-triggered pulse on block update. | In the vocabulary, rejected by `check_malformed`. | Edge triggering makes a circuit sequential, and the spec DSL is combinational. There is nothing coherent to condition them on until v2. |
| **build volume** | Infinite. | 16x6x16, outside reads as air. | Fixed volume is what makes the token sequence a fixed length. Treating outside as air rather than stone means a circuit cannot lean on the world border for support. |
| **ports** | Anywhere. | Levers on `x=0`, lamps on `x=15`. | Removes a large nuisance degree of freedom in v1. Free placement is a v2 ablation. |

## Extensions to the verdict enum

The design lists four malformed reasons; the implementation needs three more.
They are extensions, not disagreements with the game:

- `MaskedCell` — a control token (usually `MASK`) inside a grid body. Catches a
  half-denoised diffusion sample before it reaches the simulator.
- `ExcludedBlock` — a block that exists in the vocabulary but is out of scope
  for v1 (observers).
- `HistoryDependent` — the output depends on the order inputs were applied. This
  is the check that turns the update-order divergence from a silent risk into a
  rejection.

## What the signal view is, and is not

`daedalus power` — and the signal buttons in the web and terminal UIs — ask the
simulator for the **settled** dust field under one input assignment: hold the
levers in that position, let the circuit come to rest, then report one strength
per cell. It is a still photograph of a circuit at rest, not a recording of one
switching.

Two things follow, and both matter when reading it:

- **There is no propagation to watch.** The intermediate states a signal passes
  through on its way to rest are computed and discarded. A run of dust does not
  light up cell by cell; it is either at its settled strength or it is not.
  Animating the numbers would be inventing frames the simulator never produced.
- **A cell reading zero is dust with no signal, not the absence of dust.** The
  field is zero everywhere that is not dust, so a reading is only meaningful
  read against the layout it came from. Both UIs draw it over the grid for
  exactly that reason.

`settled` says whether there was a resting state to photograph at all. It is
false when the circuit oscillates, or when it had not come to rest inside the
tick budget — and in both cases the field is a snapshot of something still
moving, which is worth knowing before reading anything into it.

It does **not** cover history dependence. `power` resets the circuit and applies
the assignment directly, so a history-dependent circuit settles, `settled` comes
back true, and the field shows the state reached from all-levers-off — one of
its resting states, not necessarily the one a player flipping levers in some
other order would be looking at. `evaluate` is what detects that case, and it
rejects the circuit; `power` is a viewer and takes the grid as given.

## How this list is checked

`harness/compare.py` replays generated circuits through a real server and diffs
the truth tables, classifying each disagreement by the divergence it implicates.
The targets from the design:

- **100% agreement on the golden set** — 94 of the 104 hand-built circuits,
  plus 2 built in the harness itself, so 96 replayed. The 10 left out are
  malformed by construction (floating dust, a port violation); they are what
  `check_malformed` is for, and a grid that cannot be placed in a world is
  not a question about the game.
- **≥99.5% agreement on 10k random generated circuits.**

Neither has been measured yet: the Fabric mod half of the harness is committed
as source but has never been compiled or run, because the environment this was
built in has no Minecraft server. The Python half is complete and its comparison
path is exercised by the test suite in `--dry-run` mode. Until those two numbers
exist, treat every downstream figure as *internally consistent* rather than
*validated against the game* — which is exactly the distinction this document is
here to preserve.
