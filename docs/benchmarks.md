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

## Corpus yield

`daedalus corpus` samples a spec, synthesises a layout, and keeps it only if
the verifier passes it. The discard rate is reported rather than hidden,
because it is the honest cost of a procedural compiler that does not always
succeed.

At `--scale 0.05`, before crossbar bridges landed:

```
attempts 704 · placed 59 · routed 59
failures: routing 606, placement 25, signal 14
```

An 8% yield, dominated by routing. That is what motivated bridges: a planar
router cannot build a netlist that needs a wire crossing, and a large fraction
of randomly sampled specs need one.

## What is not measured

- **sim↔game agreement.** The number that would make any of the above
  meaningful. Needs the Fabric harness and a Minecraft server; neither has
  been run.
- **pass@k for a trained model.** The generators run and train (see
  `daedalus train`), but no model has been trained at a size or duration where
  a pass rate would mean anything. The tiny CPU configs used in the tests
  verify nothing, which is the expected result and not a finding.
