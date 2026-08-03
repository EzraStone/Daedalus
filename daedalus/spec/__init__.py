"""Specifications: the machine-readable source of truth for every circuit.

A :class:`Spec` is what a circuit is supposed to *do*. A :class:`PlacedSpec`
adds where its ports live, which is what the verifier needs. Natural language
is a view generated from a spec (see :mod:`daedalus.data.paraphrase`) and never
the other way round — if the prompt and the spec disagree, the spec is right.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .. import vocab as V
from .canon import canonical_source, is_constant, semantic_hash, table_text, truth_table
from .dsl import (
    MAX_INPUTS,
    MAX_OUTPUTS,
    Constraints,
    Expr,
    ParsedSpec,
    SpecSyntaxError,
    format_expr,
    gate_count,
    parse,
)

#: Minimum separation between two port rows.
#:
#: Adjacent dust runs are one net, not two — a router that puts two signals on
#: neighbouring rows has silently built an OR gate. Two rows of clearance is
#: the cheapest way to make that structurally impossible at the ports.
PORT_SPACING = 2

#: Ports live on the logic layer; nothing in v1 puts one higher.
PORT_Y = V.LOGIC_Y


class PlacementError(ValueError):
    """Ports could not be placed on the fixed faces."""


@dataclass(frozen=True, slots=True)
class Spec:
    """A canonical, position-free specification."""

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    #: ``rows[m]`` is the output bitmask for input assignment ``m``.
    rows: tuple[int, ...]
    constraints: Constraints
    #: The rule expressions, kept so a spec can be re-rendered and paraphrased.
    rules: tuple[tuple[str, Expr], ...] = ()
    #: Operator count, the difficulty axis the corpus is stratified on.
    gates: int = 0

    # -- construction ------------------------------------------------------

    @classmethod
    def from_parsed(cls, parsed: ParsedSpec) -> Spec:
        return cls(
            inputs=tuple(parsed.inputs),
            outputs=tuple(parsed.outputs),
            rows=truth_table(parsed),
            constraints=parsed.constraints,
            rules=tuple((name, parsed.rules[name]) for name in parsed.outputs),
            gates=parsed.gate_count(),
        )

    @classmethod
    def parse(cls, source: str) -> Spec:
        return cls.from_parsed(parse(source))

    # -- identity ----------------------------------------------------------

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    @property
    def n_outputs(self) -> int:
        return len(self.outputs)

    def semantic_hash(self) -> int:
        """Fingerprint of the behaviour, independent of names and positions."""
        return semantic_hash(self.n_inputs, self.n_outputs, self.rows)

    def key(self) -> str:
        """Hex form of the semantic hash, for filenames and log lines."""
        return f"{self.semantic_hash():016x}"

    def is_constant(self) -> bool:
        return is_constant(self.rows)

    def expect(self, assignment: int, output: int = 0) -> bool:
        return bool(self.rows[assignment] >> output & 1)

    # -- rendering ---------------------------------------------------------

    def source(self, ascii_only: bool = False) -> str:
        lines = [
            "inputs " + " ".join(self.inputs),
            "outputs " + " ".join(self.outputs),
        ]
        lines += self.constraints.describe()
        for name, expr in self.rules:
            lines.append(f"{name} = {format_expr(expr, ascii_only)}")
        return "\n".join(lines)

    def table(self) -> str:
        return table_text(self.inputs, self.outputs, self.rows)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.source()

    # -- derivation --------------------------------------------------------

    def with_constraints(self, constraints: Constraints) -> Spec:
        return replace(self, constraints=constraints)

    # -- placement ---------------------------------------------------------

    def place(self, input_z, output_z) -> PlacedSpec:
        """Pin the ports to explicit rows on the input and output faces."""
        return PlacedSpec.build(self, tuple(input_z), tuple(output_z))

    def default_placement(self, rng: random.Random | None = None) -> PlacedSpec:
        """Spread the ports evenly over each face.

        With a generator, jitter the rows instead. Port rows are a genuine
        degree of freedom in the layout, and a corpus that always uses the same
        ones teaches the model the placer's habits rather than the physics.
        """
        return self.place(
            _spread(self.n_inputs, rng),
            _spread(self.n_outputs, rng),
        )


def _spread(n: int, rng: random.Random | None = None) -> tuple[int, ...]:
    """``n`` rows in ``1..14``, at least :data:`PORT_SPACING` apart.

    Without a generator the rows are spread evenly over the whole face, which
    is reproducible and fine for tests.

    With one they are sampled uniformly at random subject only to the spacing
    rule, and that difference matters more than it looks. Evenly spread ports
    put the first and last input at opposite ends of the board, so any signal
    that has to meet another one crosses the entire grid — and since every dust
    cell reserves a one-cell moat, two long crossing nets can exhaust the
    routing space on a circuit that would fit easily with the ports six rows
    apart. Sampling produces compact arrangements as often as spread ones, and
    it is better augmentation besides: port rows are a real degree of freedom,
    and a corpus that always uses the same ones teaches the model the placer's
    habits.
    """
    lo, hi = 1, V.SZ - 2
    rows_available = hi - lo + 1
    gap = PORT_SPACING - 1
    if n < 1:
        return ()
    if (n - 1) * PORT_SPACING > rows_available - 1:
        raise PlacementError(
            f"cannot fit {n} ports at least {PORT_SPACING} rows apart in {V.SZ} rows"
        )

    if rng is None:
        if n == 1:
            return (lo + (rows_available - 1) // 2,)
        step = (rows_available - 1) / (n - 1)
        return tuple(round(lo + k * step) for k in range(n))

    # Sample n positions from a range shortened by the mandatory gaps, then
    # push them back apart. Every legal arrangement is equally likely.
    reduced = rows_available - (n - 1) * gap
    picks = sorted(rng.sample(range(reduced), n))
    return tuple(lo + p + k * gap for k, p in enumerate(picks))


@dataclass(frozen=True, slots=True)
class PlacedSpec:
    """A spec with its ports pinned to cells — what the verifier evaluates."""

    spec: Spec
    input_ports: tuple[tuple[int, int, int], ...]
    output_ports: tuple[tuple[int, int, int], ...]

    @classmethod
    def build(cls, spec: Spec, input_z, output_z) -> PlacedSpec:
        if len(input_z) != spec.n_inputs:
            raise PlacementError(
                f"spec has {spec.n_inputs} inputs but {len(input_z)} rows were given"
            )
        if len(output_z) != spec.n_outputs:
            raise PlacementError(
                f"spec has {spec.n_outputs} outputs but {len(output_z)} rows were given"
            )
        _check_rows("input", input_z)
        _check_rows("output", output_z)
        return cls(
            spec=spec,
            input_ports=tuple((V.INPUT_X, PORT_Y, z) for z in input_z),
            output_ports=tuple((V.OUTPUT_X, PORT_Y, z) for z in output_z),
        )

    # -- proxies the verifier client reads --------------------------------

    @property
    def rows(self) -> tuple[int, ...]:
        return self.spec.rows

    @property
    def constraints(self) -> Constraints:
        return self.spec.constraints

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.spec.inputs

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.spec.outputs

    @property
    def input_z(self) -> tuple[int, ...]:
        return tuple(p[2] for p in self.input_ports)

    @property
    def output_z(self) -> tuple[int, ...]:
        return tuple(p[2] for p in self.output_ports)

    def semantic_hash(self) -> int:
        return self.spec.semantic_hash()


def _check_rows(what: str, rows) -> None:
    if len(set(rows)) != len(rows):
        raise PlacementError(f"two {what} ports share a row: {list(rows)}")
    for z in rows:
        if not 0 <= z < V.SZ:
            raise PlacementError(f"{what} row {z} is outside the build volume")
    ordered = sorted(rows)
    for a, b in zip(ordered, ordered[1:]):
        if b - a < PORT_SPACING:
            raise PlacementError(
                f"{what} rows {a} and {b} are closer than {PORT_SPACING} apart; "
                "adjacent dust runs would merge into one net"
            )


__all__ = [
    "MAX_INPUTS",
    "MAX_OUTPUTS",
    "PORT_SPACING",
    "Constraints",
    "Expr",
    "PlacedSpec",
    "PlacementError",
    "Spec",
    "SpecSyntaxError",
    "canonical_source",
    "gate_count",
    "parse",
    "semantic_hash",
    "truth_table",
]
