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

## What is not measured

- **sim↔game agreement.** The number that would make any of the above
  meaningful. Needs the Fabric harness and a Minecraft server; neither has
  been run.
- **pass@k for a trained model.** The generators run and train (see
  `daedalus train`), but no model has been trained at a size or duration where
  a pass rate would mean anything. The tiny CPU configs used in the tests
  verify nothing, which is the expected result and not a finding.
