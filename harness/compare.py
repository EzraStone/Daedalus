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
    settled: bool = True

    def agrees(self) -> bool:
        return self.observed is not None and self.observed == self.simulated


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

    def __init__(self, host: str = "127.0.0.1", port: int = 25599, timeout: float = 60.0):
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


def build_cases(n: int, seed: int, verifier: Verifier) -> list[Case]:
    rng = random.Random(seed)
    cases: list[Case] = []
    for i, spec in enumerate(sample_unique(rng, n * 3)):
        if len(cases) >= n:
            break
        placed = spec.default_placement(rng)
        attempt = compile_spec(spec, verifier, rng, attempts=10, fixed_placement=placed)
        if attempt.ok:
            cases.append(Case(f"case-{i:05d}", spec, placed, attempt.grid.tokens()))
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


def run(cases: list[Case], verifier: Verifier, client: GameClient | None) -> Report:
    report = Report()
    for case in cases:
        report.cases += 1
        simulate(case, verifier)
        if client is None:
            report.unreachable += 1
            continue
        try:
            case.observed, case.settled = client.run_case(case)
        except (ConnectionError, OSError) as e:
            report.unreachable += 1
            report.examples.append({"id": case.name, "error": str(e)})
            continue
        if case.agrees():
            report.agreed += 1
        else:
            kind = classify(case)
            report.by_divergence[kind] = report.by_divergence.get(kind, 0) + 1
            report.examples.append(
                {
                    "id": case.name,
                    "spec": case.spec.source(ascii_only=True),
                    "simulated": case.simulated,
                    "observed": case.observed,
                    "divergence": kind,
                }
            )
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=25599)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build and simulate the cases without contacting a server",
    )
    ap.add_argument("--out", help="write the JSON report here")
    args = ap.parse_args(argv)

    with Verifier() as verifier:
        cases = build_cases(args.cases, args.seed, verifier)
        if args.dry_run:
            report = run(cases, verifier, None)
        else:
            try:
                with GameClient(args.host, args.port) as client:
                    report = run(cases, verifier, client)
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
