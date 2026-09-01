# Fidelity harness

The simulator in `crates/redsim` is a model of a documented subset of redstone,
not a reimplementation of Minecraft. Every number in this repository is
measured against it, so "is the model right?" is the question the whole project
rests on — and it cannot be answered by more unit tests, because unit tests
only check the model against itself.

This harness answers it by replaying the same circuits through the real game:

```
compare.py ──.schem──► Fabric mod ──► void world ──► toggle levers ──► lamp states
     │                                                                      │
     └────────────────────── redsim verdict ◄───────────────────────────────┘
                                  agree?
```

## What to report

Two numbers, both in the README:

- **100% agreement on the golden set.** The 104 hand-built circuits with
  hand-derived verdicts, minus the 10 that are malformed by construction and
  cannot be placed in a world at all — so 94, plus the 2 built here, is what
  actually gets replayed. Anything less is a bug in the simulator, not a
  tolerance.

  They live in `crates/redsim/tests/golden.rs` and reach this side through
  `REDSIM_DUMP_GOLDEN`; `compare.py --export-golden` runs that for you and
  prints how many it got. Without the export the golden suite is the two
  hand-built cases, which is a much weaker claim and is reported as such.
- **≥99.5% agreement on 10k random generated circuits.** This is the one that
  makes the rest of the repository believable, and on its own it is a more
  rigorous artifact than most Minecraft-AI projects contain.

Publishing a number below target is still worth doing. Publishing none is not.

## Running it

Three steps, on Linux, macOS or Windows. They need a network and about a
gigabyte of downloads, which is why nothing else in the build depends on them.

```
make harness-check     # everything but Minecraft — start here
make harness-setup     # build the mod, fetch the pinned Fabric server
make harness-server    # run it (leave this going)
make harness-smoke     # ten cases end to end, in another shell
make agreement         # the real run: 10,000 cases
```

`harness-check` runs every golden case through the whole pipeline against a
stand-in that answers with `redsim`'s own verdict. Agreement is 100% by
construction and says nothing about Minecraft — what it checks is that cases
build, schematics serialise, the request and response shapes line up, and the
report adds up. It costs seconds and needs no network, and it is the cheapest
way to find out the harness is miswired before spending an hour on a server.
It found its first bug on its first run.

`server/*.sh` and `server/*.ps1` are the same three steps twice, and they build
the same runtime: `tests/test_harness_scripts.py` diffs the server
configuration the two halves write and fails if they drift apart. A runtime
that differed between platforms would surface as a *fidelity* disagreement,
with nothing pointing at the real cause.

Java comes from the machine where possible. `bootstrap-java.sh` fetches a
Temurin build only when no JDK of the right major version is installed, and
verifies it against the checksum Adoptium publishes for it — which catches a
corrupted download and is a weaker guarantee than the Windows script's pinned
URL and hash. That difference is deliberate and is written down in the script:
inventing checksums for four platforms that nobody could verify would be worse
than not pinning.

## Status

`compare.py` is complete and testable: it speaks the socket protocol, drives the
comparison, and classifies disagreements by the divergence they implicate. The
Fabric mod under `mod/` is the untested half — it needs a Minecraft server, a
Gradle toolchain and a Fabric loader, none of which exist in the environment
this was written in. It is committed as source with the protocol pinned so the
Python side has something concrete to talk to, and it is explicitly **not**
claimed to have been run. The scripts above have not been run end to end
either, for the same reason; what *is* tested is everything in them that does
not need the network, which is most of their logic.

## Protocol

Line-oriented JSON over a TCP socket, one request per line:

```json
{"op": "place", "schematic": "<base64 .schem>", "id": "case-0001"}
{"op": "test",  "id": "case-0001",
 "levers": [[0,1,4],[0,1,8]], "lamps": [[15,1,6]]}
```

The mod places the schematic in a void world, walks every input combination,
waits for the circuit to settle, and reports:

```json
{"id": "case-0001", "rows": [[0,0,1],[0,1,1],[1,0,1],[1,1,0]], "settled": true}
```

Each row is `inputs..., outputs...`. `settled: false` means the circuit was
still changing after the tick cap — which is real-game evidence of an
oscillator, and should agree with an `UNSTABLE` verdict.

## Expected disagreements

These are the four documented divergences, in the order they are likely to bite:

1. **Weak vs. strong power.** Historically the number one source of
   simulator/game divergence. If a disagreement involves a block between two
   dust runs, look here first.
2. **Update order.** Java processes block updates in a quasi-random order;
   `redsim` latches and applies simultaneously. Circuits whose behaviour depends
   on the difference are called *locational* and should be **rejected**, not
   reconciled.
3. **Torch burnout.** `redsim` rejects designs that burn out rather than
   modelling the burnout. The game will report something; the harness should
   count it as agreement when `redsim` said `MALFORMED burnout`.
4. **Quasi-connectivity.** Absent from `redsim` entirely. v1 generates no
   pistons, so this should never fire; if it does, something is emitting blocks
   outside the vocabulary.
