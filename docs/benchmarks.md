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

`daedalus baselines --specs 20 --k 8 --attempts 12`, seed 0. Three of the four
methods from §07; the prompted-LLM baseline needs an API key and has not been
run.

| method | pass@1 | pass@8 | diversity | malformed |
|---|---|---|---|---|
| procedural compiler | 0.250 | 0.250 | 4.00 | 0.000 |
| retrieval | 0.250 | 0.250 | 1.00 | 0.000 |
| unconditional | 0.000 | 0.000 | 0.00 | 0.969 |

The unconditional row is the floor and it is worth looking at: 97% of random
grids are malformed, so a model that learns nothing but "what a well-formed
circuit looks like" already clears a bar. That is why the sampler's legality
and support constraints matter — they hand that floor to the model for free,
and the interesting question starts above it.

The compiler and retrieval tying at 0.250 is not a coincidence: retrieval is
searching a corpus the compiler built, so it inherits the compiler's coverage.
Diversity separates them — the compiler finds 4 distinct layouts per solved
spec, retrieval finds 1 by construction.

## Corpus yield

`daedalus corpus` samples a spec, synthesises a layout, and keeps it only if
the verifier passes it. The discard rate is reported rather than hidden,
because it is the honest cost of a procedural compiler that does not always
succeed.

### An open regression

Same command, same scale, same seed, before and after crossbar bridges:

| | attempts | routed | bridged | yield |
|---|---|---|---|---|
| `dd18606` (planar only) | 704 | 59 | — | **8.4%** |
| current (with bridges) | 2046 | 31 | 5 | **1.5%** |

```
$ daedalus corpus /tmp/out --scale 0.05
```

Bridges unlocked XOR and multiplexers, which planar routing simply could not
build. They also cost roughly 5.6× in corpus yield, and only 5 of the 31
surviving layouts actually used one. Failures are dominated by routing (1829
of 2046) with a new `ports` category (60) that did not exist before.

**This is not diagnosed.** One hypothesis has been tested and ruled out: the
gate-spreading change in `fc93095` moved the placement cost target from
`3 + (SX-7)` to `5 + (SX-9)`, which looked like the obvious suspect. Reverting
just that line changes nothing —

| `target_x` | attempts | routed | yield |
|---|---|---|---|
| `5 + (SX-9)` (current) | 2046 | 31 | 1.52% |
| `3 + (SX-7)` (reverted) | 2030 | 30 | 1.48% |

— so the cause is elsewhere in the ~230 lines the bridge work changed in
`place.py`. Worth finding: at 1.5% yield a corpus build does about six times
the work for the same number of examples, and the loop's spec throughput is
bounded by the same code.

## What is not measured

- **sim↔game agreement.** The number that would make any of the above
  meaningful. Needs the Fabric harness and a Minecraft server; neither has
  been run.
- **pass@k for a trained model.** The generators run and train (see
  `daedalus train`), but no model has been trained at a size or duration where
  a pass rate would mean anything. The tiny CPU configs used in the tests
  verify nothing, which is the expected result and not a finding.
