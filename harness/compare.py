"""Simulator vs. real game: the agreement report.

Run the same circuits through `redsim` and through a Minecraft server, and diff
the truth tables. The output is the headline credibility number for the whole
project, so this script is deliberately unforgiving: a disagreement is a
failure, not a warning, and every one is classified by the divergence it
implicates so the fix has somewhere to start.

The server side is `harness/mod`. Without it, `--dry-run` still exercises the
whole comparison path against `redsim` alone, which is what the test suite uses.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import socket
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daedalus import vocab as V  # noqa: E402
from daedalus.data import sample_unique  # noqa: E402
from daedalus.grid import Grid  # noqa: E402
from daedalus.redsim import Malformed, Pass, Unstable, Verifier  # noqa: E402
from daedalus.schematic import write_schem  # noqa: E402
from daedalus.spec import PlacedSpec, Spec  # noqa: E402
from daedalus.synth import compile as compile_spec  # noqa: E402
from daedalus.synth.bridge import BridgePlan  # noqa: E402

#: How a disagreement is classified, in the order they are likely to occur.
DIVERGENCES = (
    "weak-vs-strong-power",
    "update-order",
    "torch-burnout",
    "quasi-connectivity",
    "unclassified",
)


class HarnessError(RuntimeError):
    """The game harness accepted a request but could not execute it."""


@dataclass
class Case:
    name: str
    spec: Spec
    placed: PlacedSpec
    tokens: list[int]
    #: Truth table as `redsim` sees it: one output bitmask per input assignment.
    simulated: list[int] = field(default_factory=list)
    observed: list[int] | None = None
    #: Did each side reach a resting state? Two fields, because they are two
    #: different observations and they can disagree -- which is itself one of
    #: the most interesting things this harness can find. They were one field,
    #: so the game's answer overwrote the simulator's before anything compared
    #: them, and a circuit redsim called UNSTABLE that the real game settles
    #: looked exactly like a circuit both agreed on.
    settled: bool = True
    observed_settled: bool = True

    def agrees(self) -> bool:
        return self.observed is not None and self.observed == self.simulated

    def settling_disagrees(self) -> bool:
        """One side reached a resting state and the other did not."""
        return self.observed is not None and self.settled != self.observed_settled


@dataclass
class Report:
    cases: int = 0
    agreed: int = 0
    unreachable: int = 0
    by_divergence: dict[str, int] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)

    @property
    def agreement(self) -> float:
        checked = self.cases - self.unreachable
        return round(self.agreed / checked, 5) if checked else 0.0

    def as_dict(self) -> dict:
        return {
            "cases": self.cases,
            "checked": self.cases - self.unreachable,
            "agreed": self.agreed,
            "agreement": self.agreement,
            "unreachable": self.unreachable,
            "by_divergence": dict(sorted(self.by_divergence.items())),
            "examples": self.examples[:10],
        }


class GameClient:
    """Talks to the Fabric mod over a socket."""

    def __init__(self, host: str = "127.0.0.1", port: int = 25599, timeout: float = 300.0):
        self.address = (host, port)
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._file = None

    def __enter__(self) -> GameClient:
        self._sock = socket.create_connection(self.address, timeout=self.timeout)
        self._file = self._sock.makefile("rwb")
        return self

    def __exit__(self, *exc) -> None:
        if self._file:
            self._file.close()
        if self._sock:
            self._sock.close()

    def request(self, payload: dict) -> dict:
        assert self._file is not None
        self._file.write((json.dumps(payload) + "\n").encode())
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise ConnectionError("the harness mod closed the connection")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConnectionError("the harness returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise ConnectionError("the harness returned a non-object response")
        if "error" in response:
            raise HarnessError(str(response["error"]))
        return response

    def run_case(self, case: Case) -> tuple[list[int], bool]:
        with tempfile.TemporaryDirectory() as tmp:
            from daedalus.grid import Grid

            path = Path(tmp) / f"{case.name}.schem"
            write_schem(Grid.from_tokens(case.tokens), path)
            blob = base64.b64encode(path.read_bytes()).decode()
        placed = self.request({"op": "place", "id": case.name, "schematic": blob})
        if placed.get("id") != case.name or placed.get("placed") is not True:
            raise ConnectionError("the harness did not acknowledge schematic placement")
        got = self.request(
            {
                "op": "test",
                "id": case.name,
                "levers": [list(p) for p in case.placed.input_ports],
                "lamps": [list(p) for p in case.placed.output_ports],
            }
        )
        if got.get("id") != case.name or not isinstance(got.get("rows"), list):
            raise ConnectionError("the harness returned an invalid test response")
        n_in = len(case.placed.input_ports)
        rows = []
        for row in got["rows"]:
            if not isinstance(row, list) or len(row) != n_in + len(case.placed.output_ports):
                raise ConnectionError("the harness returned a malformed truth-table row")
            outputs = row[n_in:]
            rows.append(sum(bit << j for j, bit in enumerate(outputs)))
        return rows, bool(got.get("settled", True))


def simulate(case: Case, verifier: Verifier) -> Case:
    """Fill in what `redsim` says, row by row.

    The verdict alone is not enough for a comparison: a `FAIL` tells you the
    circuit is wrong but not what it does. The row-level table is what the game
    can be diffed against.
    """
    verdict = verifier.evaluate(case.tokens, case.placed)
    if isinstance(verdict, Pass):
        case.simulated = list(case.spec.rows)
    elif isinstance(verdict, Unstable):
        case.settled = False
        case.simulated = []
    elif isinstance(verdict, Malformed):
        case.simulated = []
    else:
        observed = list(case.spec.rows)
        for row in verdict.mismatched_rows:
            observed[row.inputs] = row.observed
        case.simulated = observed
    return case


def classify(case: Case) -> str:
    """Guess which documented divergence a disagreement implicates.

    A guess, clearly labelled as one. Its job is to point at the right module,
    not to be authoritative — `unclassified` is a perfectly respectable answer
    and is what most genuinely new bugs will land in.
    """
    if case.observed is None:
        return "unclassified"
    if case.settling_disagrees():
        # One side oscillates and the other does not. Update order is the
        # documented reason for that: redsim applies every component's change
        # at once, and the game does not, so a circuit whose resting state
        # depends on the order it got there is exactly the class this catches.
        return "update-order"
    if not case.settled:
        return "update-order"
    from daedalus import vocab as V
    from daedalus.grid import Grid

    grid = Grid.from_tokens(case.tokens)
    # A solid block flanked by dust on opposite sides is the classic weak/strong
    # trap: the game and a naive simulator disagree about whether it conducts.
    for z in range(V.SZ):
        for x in range(1, V.SX - 1):
            if grid.get(x, V.LOGIC_Y, z) != V.SOLID:
                continue
            west = grid.get(x - 1, V.LOGIC_Y, z) == V.WIRE
            east = grid.get(x + 1, V.LOGIC_Y, z) == V.WIRE
            if west and east:
                return "weak-vs-strong-power"
    if any(V.decode(t).kind == "torch" for t in case.tokens if t < V.CONTROL_BASE):
        return "torch-burnout"
    return "unclassified"


def _case_record(case: Case) -> dict:
    return {
        "name": case.name,
        "spec": case.spec.source(ascii_only=True),
        "input_z": list(case.placed.input_z),
        "output_z": list(case.placed.output_z),
        "tokens": case.tokens,
    }


def _case_from_record(record: dict) -> Case:
    spec = Spec.parse(record["spec"])
    placed = spec.place(record["input_z"], record["output_z"])
    tokens = [int(token) for token in record["tokens"]]
    Grid.from_tokens(tokens)
    return Case(str(record["name"]), spec, placed, tokens)


def _load_case_cache(path: Path, seed: int) -> list[Case]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    metadata = json.loads(lines[0])
    if metadata != {"format": "daedalus-fidelity-corpus-v1", "seed": seed}:
        raise ValueError(f"corpus cache metadata does not match seed {seed}")
    return [_case_from_record(json.loads(line)) for line in lines[1:] if line.strip()]


def _append_cached_case(path: Path, seed: int, case: Case) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        metadata = {"format": "daedalus-fidelity-corpus-v1", "seed": seed}
        path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(_case_record(case), separators=(",", ":")) + "\n")
        stream.flush()


def build_cases(
    n: int,
    seed: int,
    verifier: Verifier,
    cache_path: Path | None = None,
) -> list[Case]:
    rng = random.Random(seed)
    cases = _load_case_cache(cache_path, seed) if cache_path else []
    if len(cases) >= n:
        return cases[:n]
    seen = {case.spec.semantic_hash() for case in cases}
    candidate = max(
        (int(case.name.rsplit("-", 1)[1]) + 1 for case in cases if case.name.startswith("case-")),
        default=0,
    )
    max_candidates = candidate + max(50, (n - len(cases)) * 20)
    while len(cases) < n and candidate < max_candidates:
        batch_size = min(256, max_candidates - candidate, max(16, (n - len(cases)) * 4))
        specs = sample_unique(rng, batch_size, seen=seen)
        if not specs:
            break
        for spec in specs:
            if len(cases) >= n or candidate >= max_candidates:
                break
            placed = spec.default_placement(rng)
            attempt = compile_spec(spec, verifier, rng, attempts=10, fixed_placement=placed)
            if attempt.ok:
                case = Case(f"case-{candidate:05d}", spec, placed, attempt.grid.tokens())
                cases.append(case)
                if cache_path:
                    _append_cached_case(cache_path, seed, case)
            candidate += 1
    if len(cases) != n:
        raise RuntimeError(
            f"compiled only {len(cases)} of {n} requested random cases "
            f"after {candidate} candidates"
        )
    return cases


def build_bridge_case() -> Case:
    """The hand-built independent-crossing circuit from the bridge golden test."""
    grid = Grid.with_substrate()
    spec = Spec.parse("inputs A B\noutputs Q R\nQ = A\nR = B")
    placed = spec.place((8, 4), (8, 12))

    for z in (8, 4):
        grid.set(0, 1, z, V.lever(V.Dir4.EAST))
        grid.set(1, 1, z, V.SOLID)
    for z in (8, 12):
        grid.set(14, 1, z, V.repeater(V.Dir4.EAST, 1))
        grid.set(15, 1, z, V.LAMP)

    for x in range(2, 6):
        grid.set(x, 1, 8, V.WIRE)
    BridgePlan((8, 8), "x").place(grid)
    for x in range(11, 14):
        grid.set(x, 1, 8, V.WIRE)

    for x in range(2, 9):
        grid.set(x, 1, 4, V.WIRE)
    for z in range(4, 9):
        grid.set(8, 1, z, V.WIRE)
    grid.set(8, 1, 9, V.repeater(V.Dir4.SOUTH, 1))
    for z in range(10, 13):
        grid.set(8, 1, z, V.WIRE)
    for x in range(8, 14):
        grid.set(x, 1, 12, V.WIRE)

    return Case("golden-bridge-independent", spec, placed, grid.tokens())


def build_direct_case() -> Case:
    """A hand-built full-width signal path with an output repeater."""
    grid = Grid.with_substrate()
    spec = Spec.parse("inputs A\noutputs Q\nQ = A")
    placed = spec.place((8,), (8,))
    grid.set(0, 1, 8, V.lever(V.Dir4.EAST))
    grid.set(1, 1, 8, V.SOLID)
    for x in range(2, 14):
        grid.set(x, 1, 8, V.WIRE)
    grid.set(14, 1, 8, V.repeater(V.Dir4.EAST, 1))
    grid.set(15, 1, 8, V.LAMP)
    return Case("golden-direct-repeater", spec, placed, grid.tokens())


def build_golden_cases() -> list[Case]:
    """Return the deterministic real-game regression suite."""
    return [build_direct_case(), build_bridge_case()]


def run(
    cases: list[Case],
    verifier: Verifier,
    client: GameClient | None,
    progress_every: int = 0,
) -> Report:
    report = Report()
    for index, case in enumerate(cases, 1):
        report.cases += 1
        simulate(case, verifier)
        if client is None:
            report.unreachable += 1
            continue
        try:
            case.observed, case.observed_settled = client.run_case(case)
        except (ConnectionError, OSError, HarnessError) as e:
            # HarnessError belongs here too. It means the mod took the request
            # and could not carry it out -- an unparseable schematic, a fixture
            # it could not build -- which is one unreachable case, not a reason
            # to abandon the other nine thousand. It is a RuntimeError, so
            # before this it escaped the loop and ended the run.
            report.unreachable += 1
            report.examples.append({"id": case.name, "error": str(e)})
            continue
        # Matching truth tables are not enough. A circuit the game never
        # settles and redsim does is a divergence even when the rows happen to
        # line up, and counting it as agreement inflates the one number this
        # harness exists to report.
        if case.agrees() and not case.settling_disagrees():
            report.agreed += 1
        else:
            kind = classify(case)
            report.by_divergence[kind] = report.by_divergence.get(kind, 0) + 1
            example = {
                "id": case.name,
                "spec": case.spec.source(ascii_only=True),
                "simulated": case.simulated,
                "observed": case.observed,
                "divergence": kind,
            }
            if case.settling_disagrees():
                # Which side oscillated, since the tables alone will not say.
                example["settled"] = {
                    "redsim": case.settled,
                    "game": case.observed_settled,
                }
            report.examples.append(example)
        if progress_every > 0 and (index % progress_every == 0 or index == len(cases)):
            print(
                f"checked {index}/{len(cases)}: {report.agreed} agree, "
                f"{report.unreachable} unreachable",
                file=sys.stderr,
                flush=True,
            )
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument(
        "--suite",
        choices=("random", "golden", "combined"),
        default="combined",
        help="select random cases, deterministic golden cases, or both",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=25599)
    ap.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="seconds to allow one real-game truth-table sweep",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build and simulate the cases without contacting a server",
    )
    ap.add_argument("--out", help="write the JSON report here")
    ap.add_argument(
        "--corpus-cache",
        type=Path,
        help="checkpoint accepted random cases as resumable JSON Lines",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="print progress after this many cases; zero disables it",
    )
    args = ap.parse_args(argv)

    with Verifier() as verifier:
        cases = []
        if args.suite in {"golden", "combined"}:
            cases.extend(build_golden_cases())
        if args.suite in {"random", "combined"}:
            cases.extend(build_cases(args.cases, args.seed, verifier, args.corpus_cache))
        if args.dry_run:
            report = run(cases, verifier, None, args.progress_every)
        else:
            try:
                with GameClient(args.host, args.port, args.timeout) as client:
                    report = run(cases, verifier, client, args.progress_every)
            except OSError as e:
                print(
                    f"could not reach the harness mod at {args.host}:{args.port}: {e}\n"
                    "Start the server with harness/mod, or pass --dry-run.",
                    file=sys.stderr,
                )
                return 2

    text = json.dumps(report.as_dict(), indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    # Below the target from the design spec is a failure, not a warning.
    if not args.dry_run and report.agreement < 0.995:
        print(f"\nagreement {report.agreement:.4f} is below the 0.995 target", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
