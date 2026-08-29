# Measured numbers

Everything here is reproducible with `daedalus bench`. Nothing here has been
validated against Minecraft — see [`divergences.md`](divergences.md) for what
that means and why it is the number that actually matters.

## Verifier throughput

The claim the whole project rests on is that verification is cheap enough to
put inside a training loop. It is, but the figure depends almost entirely on
batching, and quoting one number without saying which is misleading.

```
$ daedalus bench --batch 64
```

Measured on a NAND layout (41 blocks, latency 2rt), release build, 50 repeats,
one x86-64 Linux container:

| grids per request | µs per evaluation | evaluations/second |
|---|---|---|
| 1 | 188.2 | 5,314 |
| 8 | 78.2 | 12,788 |
| 64 | 45.1 | 22,196 |
| 256 | 41.1 | 24,346 |

Read that as two different quantities. At batch 1 you are timing a pipe
round-trip; at batch 256 the round-trip is amortised and you are timing the
simulation, which is roughly **41 µs**. The loop batches — `LoopConfig`
defaults to 64 candidates per spec — so the batched figure is the one that
governs how long a round takes.

A round of 20,000 specs at 64 candidates is 1.28M evaluations, or about a
minute of verification.

### The verifier is not what makes building a corpus slow

The number above is easy to read as the cost of producing a circuit. It is not.
`daedalus bench --compiler` compiles random specs end to end and reports the
split:

```
$ daedalus bench --compiler --specs 50 --seed 7
```

| | |
|---|---|
| median | 354 ms per spec |
| mean | 293 ms |
| throughput | 3.4 specs/second |
| yield | 0.220 |
| time inside the verifier | 10 ms of ~14.7 s, **0.1%** |

Stages over those 50: 11 verified, 35 failed to route, 3 were outside the
primitive set, 1 was rejected by the verifier.

So the ratio is roughly four thousand to one the other way: a verdict costs
tens of microseconds and a layout costs a third of a second, all of it in
Python — netlist construction, placement, and Lee routing. Two things follow.

The verifier being cheap is what makes it usable *inside a training loop*,
where a model produces candidates and every one of them needs a verdict. It
says nothing about how fast a corpus can be built, and optimising it would not
move corpus generation at all.

#### Four changes worth 39%

Profiling that 99.9% put four things at the top, none of them an algorithm.

`neighbours()` — the four orthogonal cells around a cell — was a generator with
a bounds test per step, called **2.3 million times** to compile twelve specs.
There are 256 cells and the answer never changes, so the whole table is now
built once at import and the call is a dict lookup.

`BridgePlan.dust` was a property that rebuilt its seven coordinates on every
read, and `supports`, `footprint`, `wire_hops`, `obstructions` and `place` all
read it. There are `16 x 16 x 2` possible plans; the geometry is now computed
once per plan and cached.

`_sinks_of_driver` walked every net comparing `Driver` dataclasses to find the
sinks one driver feeds — once per candidate site, which is **1.2 million**
equality calls for twelve specs. The netlist does not change during placement,
so the index is built once when the synthesiser is constructed.

`can_hold_dust` — asked for nearly every cell the router considers — called
`net_at` and `component_at` in turn for each neighbour, and each did its own
dictionary lookup for the same cell. A cell can only be one of the two things,
so it is one lookup now. `_site_is_clear` was building a two-element set per
call, 150,000 times a run, to test membership of the gate's own cells.

| | before | after |
|---|---|---|
| median per spec | 354 ms | 217 ms |
| throughput | 3.4 specs/s | 5.5 specs/s |

Checked by fingerprint rather than by eye: compiling 200 specs at seed 11 on
the tree before any of this and on the tree after gives the same SHA-256 over
every stage, every failure detail and every grid produced —
`28bc9fb9ca1e6d3a` both times. Same compiler, 39% less time.

The 0.1% is also flattered by the yield. A spec that never routes never
reaches the verifier, and seven in ten do not, so most of that wall time is
the placer failing rather than the verifier waiting. Fixing the routing gap
would raise the verifier's share — which is the right direction, and still
nowhere near the point where it matters.

## Baselines

`daedalus baselines --specs 15 --k 8 --attempts 12`, seed 0. The prompted-LLM
baseline from §07 needs an API key and has not been run; the rest are here.

| method | pass@1 | pass@8 | diversity | malformed |
|---|---|---|---|---|
| procedural compiler | 0.267 | 0.267 | 4.25 | 0.000 |
| retrieval | 0.200 | 0.200 | 1.00 | 0.000 |
| constrained random | 0.000 | 0.000 | 0.00 | 0.042 |
| unconditional | 0.000 | 0.000 | 0.00 | 0.992 |

The two random rows are the same sampler; the constrained one adds only what
the model's sampler enforces for free — put a solid block under anything that
needs holding up, and keep levers and lamps at declared ports. That alone
takes the malformed rate from **99.2% to 4.2%**, a factor of twenty-four, with
no training whatsoever.

That comparison is the reason the row exists. "97% of random grids are
malformed" makes well-formedness look like an achievement, and it is not one:
almost all of it is available for free from two rules. Both random rows still
score zero on pass@1, because a grid can be perfectly well-formed and compute
nothing. So a trained model reporting a low malformed rate has demonstrated
nothing on its own — the number that means something is the pass rate, and
the floor for that is zero.

Compiler and retrieval land close together for a reason: retrieval searches a
corpus the compiler built, so it inherits the compiler's coverage and cannot
exceed it. Diversity separates them — the compiler finds several distinct
layouts per solved spec, retrieval finds one by construction.

### The prompted-LLM baseline was measuring its own format

The one baseline that cannot be run here is also the one most easily made
unfair by accident, and two ways it was were found by testing the parser
rather than the model.

The parser dropped any line containing a character outside the block
alphabet. That is the right rule for a stray `x` in the middle of a grid, and
the wrong rule for the four spaces markdown puts in front of a fenced code
block: the indentation made every one of the sixteen lines illegal, and the
reply was counted as prose. Indentation moves no blocks, so it now goes the
way trailing whitespace already did. Strictness about the grid itself is
unchanged — one unknown character still rejects the whole reply, and two grids
in one reply are still refused rather than guessed between.

The renderer was worse, because it was silent. `render_ascii_layer` exists to
turn known-good circuits into few-shot examples, and it drew one character per
block *kind*: all five torch attachments as `t`, all sixteen repeater states as
`>`. Rendering a working NAND and reading it straight back produced a grid the
verifier called **malformed** — the torch that had been held up by the block to
its east was now attached to air. Every few-shot prompt built this way was
teaching the model a layout that cannot be graded.

Over twelve compiled circuits, before and after:

| | before | after |
|---|---|---|
| round-trips to byte identity | 0 | 8 |
| refused as inexpressible | 0 | 3 |
| silently changed | 12 | 0 |

The alphabet now spells out torch attachment and repeater facing, which are the
only two states the placer varies. The three refusals are circuits that bridge
over themselves at `y=2`; one layer of characters cannot say so, and saying
nothing was the bug. A test asserts the prompt and the parser describe the same
alphabet, since a character the grader accepts but the prompt never mentions is
a point the baseline cannot score.

None of this changes a measured number — the baseline still has not been run.
It changes whether the number, when it exists, will be about the model.

## Corpus yield

`daedalus corpus` samples a spec, synthesises a layout, and keeps it only if
the verifier passes it. The discard rate is reported rather than hidden,
because it is the honest cost of a procedural compiler that does not always
succeed.

### The headline yield number fell, and it is not a regression

`daedalus corpus /tmp/out --scale 0.05`, seed 0, before and after crossbar
bridges:

| | attempts | routed | bridged | yield |
|---|---|---|---|---|
| `dd18606` (planar only) | 704 | 59 | — | **8.4%** |
| current | 2046 | 31 | 5 | **1.5%** |

That looks like a 5.6× regression and is not one. The bridge work also flipped
`ROUTABLE_GATES` from `("and", "or")` to include XOR — the exclusion existed
precisely *because* a planar router cannot build one — so the two rows are not
sampling the same workload. The second is being asked much harder questions.

Holding the gate set fixed and varying only the code separates them:

| code | gate set | attempts | routed | yield |
|---|---|---|---|---|
| `dd18606` | and/or | 704 | 59 | 8.4% |
| current | and/or | 664 | 59 | **8.9%** |
| current | and/or/xor | 2046 | 31 | 1.5% |

The router did not get worse; on the workload the old one could handle it got
slightly better. What changed is that the corpus now contains a class of spec
that used to be filtered out of it, and those specs are expensive: a crossing
costs a seven-cell span and is capped at two per layout.

Two things follow. Yield is now a property of the spec mix and cannot be
compared across changes to `ROUTABLE_GATES`. And the honest cost of XOR
coverage is that a corpus build does several times the work per example —
worth it for coverage of a function class that was previously impossible,
but worth knowing before budgeting a large build.

An earlier version of this file recorded the 8.4% → 1.5% drop as an
undiagnosed regression, and separately ruled out the gate-spreading change in
`fc93095` as its cause (reverting that line alone gives 1.48% against 1.52%,
and an intermediate value 1.49%). That ruling-out stands; the regression
framing does not.

### Retrying barely helps

Sixty random specs, same seed, varying only the retry budget:

| attempts | solved | |
|---|---|---|
| 3 | 10/60 | 17% |
| 6 | 11/60 | 18% |
| 12 | 11/60 | 18% |
| 25 | 12/60 | 20% |
| 50 | 12/60 | 20% |

A sixteenfold increase in budget buys two more circuits. Failures are
structural, not unlucky: the specs that fail are the same ones every time,
and the router runs out of ideas rather than out of tries.

Where they fail, over 120 specs at 6 attempts each:

```
247  routing    net N: cannot join N fragments
192  routing    net N: cannot reach inverter N
 32  routing    net N: inverter N has no head-on input face
 26  placement  inverter N cannot give net N an output face
```

Two failures are 92% of the routing total, and both are the same shape of
problem: a net that has to reach somewhere the geometry does not allow. That
is where the yield is, and no amount of rerolling will find it.

Every window used to advise "try more attempts or another seed" on a routing
failure. Half of that was wrong and is now corrected — a different seed moves
the port rows and occasionally helps, a bigger budget does not.

### Why retrying does so little, and one fix that did not work

`_topological_order` sorts gates by `(depth, index)`, which is fully
deterministic. Every attempt therefore places gates in the *same order* and a
retry only moves the port rows. Depth is the sole hard constraint — gates at
equal depth are mutually independent — so shuffling within a depth is
topologically valid and looked like the obvious way to make retries mean
something.

It works, in the narrow sense. Sixty random specs:

| attempts | index order | shuffled within depth |
|---|---|---|
| 3 | 17% | 17% |
| 6 | 18% | 20% |
| 12 | 18% | 22% |
| 25 | 20% | 22% |
| 50 | 20% | 22% |

And it is not worth having. On `Q = (A & B) | (!A & C)` — a crossbar netlist,
which is exactly the class bridges were built for — twelve seeds at twenty
attempts each go from **2 builds, both bridged, to 0**. Shuffling only on
retries does not rescue it either: the successes were coming from later
attempts *under the deterministic order*, so any shuffling loses them.

Index order is load-bearing for crossbars and nobody wrote that down. It is
written down now, in `_topological_order`, along with these numbers. One
circuit in sixty is not worth trading the function class the bridge work
exists to cover, so the change is reverted; the finding is not.

## Why specs fail, in the compiler's own words

The yield number has been reported since the corpus engine existed; what it is
made of has not.

```
$ daedalus bench --compiler --specs 200 --seed 11 --attempts 8
```

The command groups failures by *shape* — the placer's own words with the
indices replaced, so "cannot reach inverter 1" and "cannot reach inverter 2"
count as one problem — and reports yield by gate count. Everything below comes
out of that one command, so it can be rerun rather than believed.

| count | shape |
|---|---|
| 91 | `routing: net N: cannot join N fragments` |
| 68 | `routing: net N: cannot reach inverter N` |
| 9 | `routing: net N: inverter N has no head-on input face` |
| 1 | `routing: net N: output N terminal is blocked` |
| 1 | `constraint: region NxN against a budget of NxN` |
| 1 | `verify: Unstable(period_ticks=N)` |

| gates | solved | of | yield |
|---|---|---|---|
| 1 | 1 | 2 | 0.50 |
| 2 | 5 | 14 | 0.36 |
| 3 | 8 | 45 | 0.18 |
| 4 | 7 | 53 | 0.13 |
| 5 | 7 | 51 | 0.14 |
| 6 | 1 | 35 | 0.03 |
| **all** | **29** | **200** | **0.145** |

Three things this says that the single number did not.

**Yield is a function of size, not a constant.** It falls by more than a factor
of ten between one gate and six, and it is still falling at six. A corpus
built this way is therefore biased small, and any model trained on it inherits
that bias — which matters for the extrapolation split specifically, since
"harder than anything in training" is exactly what it is meant to test.

**Twelve percent of specs never reached the placer at all — that one is
fixed, and it bought no yield.** A driver feeding more than three separate
nets was refused outright, because a torch has three free faces. Two
inverters in series are the identity, so a buffer costs two cells and two
torch delays and needs no new primitive; `insert_buffers` now adds one
wherever it is needed and the netlist stage refuses nothing.

The capability gain is real: a spec that wants a signal in four places is
buildable at all now, and the verifier confirms the truth tables. The yield
gain is zero. Measured before and after on a matched run of the same 200
specs — 27 solved either way, with all 24 netlist rejections turning into
routing failures — because the specs that want a fourth face are the dense
ones, and two more gates puts them in the part of the size curve where yield
is 0.03. (Those two runs used a fixed per-spec seed rather than the offset the
command above uses, which is why 27 there and 29 here; the comparison is
between the two trees, not with the table.)

That is worth stating plainly rather than filing as a win. Removing a hard
limit and moving the same specs into a different failure bucket are both
true, and only the first is progress.

**The routing failures are two shapes, both connectivity.** `cannot join
fragments` and `cannot reach inverter` are 159 of the 169 routing failures
between them. Neither is about running out of *space* — the board is nowhere
near full when they fire — they are about a net's pieces ending up in regions
the router cannot connect after earlier nets have been committed. That points
at ordering and at the greedy commit, not at the grid being too small.

### An obvious fix for the biggest shape, which does nothing

`_ensure_sources` joins a net's fragments by sorting them largest first and
trying to connect the largest to each of the others. If the largest reaches
none of them it gives up — even though two smaller fragments might connect to
each other, which would reduce the count and perhaps unblock the next pass.
Trying every pair instead is four lines.

Over the same 200 specs it changes nothing at all:

| | head-to-others | every pair |
|---|---|---|
| solved | 27 | 27 |
| cannot reach inverter | 76 | 76 |
| cannot join fragments | 75 | 75 |
| no head-on input face | 17 | 17 |

(measured with the fixed per-spec seed, as above)

Not "roughly the same" — the same failure on the same spec, every time. When
the largest fragment cannot reach anything, no pair can, because fragments are
separated by committed cells rather than by distance and the separation is
effectively transitive. Reverted. Recorded because it is the first thing
anyone looking at that error message will try.

None of this is fixed here. It is written down because "20% yield" was being
treated as a property of the problem, and it is three separate problems with
three different fixes.

## Signal probing

`daedalus power` settles a circuit for one input assignment and returns the
dust strength of every cell. It costs one settle, so it is the same order as
a single verdict rather than a batch of them -- fine interactively, and not
something to put in a loop over a corpus.

## What is not measured

- **sim↔game agreement.** The number that would make any of the above
  meaningful. Needs the Fabric harness and a Minecraft server; neither has
  been run.
- **pass@k for a trained model.** The generators run and train (see
  `daedalus train`), but no model has been trained at a size or duration where
  a pass rate would mean anything. The tiny CPU configs used in the tests
  verify nothing, which is the expected result and not a finding.
