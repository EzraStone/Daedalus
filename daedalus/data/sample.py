"""Sampling random logic specifications.

Scraping community schematics would give a few thousand unlabelled builds of
wildly varying quality, murky redistribution rights, and no reliable account of
what any of them does. Generating them gives millions of perfectly labelled
examples with a difficulty distribution you chose. This is the rare domain
where synthetic data is strictly better.

The sampler's job is to produce specs that are *interesting*, which means more
than "syntactically valid":

* every declared input must matter. ``Q = (A ∧ ¬A) ∨ B`` mentions ``A`` and
  ignores it; a corpus full of those teaches the model to ignore a port.
* the function must not be constant. A circuit that ignores its inputs is
  legal, trivial, and worth nothing as a training example.
* duplicates must be recognised by *behaviour*, not by source text, or the
  corpus silently concentrates on whatever the sampler finds easy to write.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..spec import Constraints, Spec
from ..spec.canon import irrelevant_inputs, is_constant
from ..spec.dsl import MAX_INPUTS, MAX_OUTPUTS, Binary, Not, ParsedSpec, Ref

#: Operators §04 asks the DAG sampler to draw from.
ALL_GATES = ("and", "or", "xor")

#: All combinational operators are routable. XOR prefers a planar NOR
#: decomposition, while crossbar shapes can fall back to elevated bridges.
ROUTABLE_GATES = ALL_GATES

NAMES = ("A", "B", "C", "D", "E", "F")


@dataclass(frozen=True, slots=True)
class SampleConfig:
    """Difficulty knobs, matching §04's generation procedure."""

    min_gates: int = 1
    max_gates: int = 6
    min_inputs: int = 1
    max_inputs: int = 4
    gate_set: tuple[str, ...] = ROUTABLE_GATES
    #: Probability that any given operator slot becomes a negation.
    negation_rate: float = 0.35
    #: How often to attach a hard constraint, which the verifier enforces.
    constraint_rate: float = 0.15
    #: Upper bound on outputs per spec. The default of one keeps every number
    #: measured so far comparable; above one the sampler draws several rules
    #: over the same inputs, which is the only way the corpus contains a
    #: circuit whose subterm feeds two places.
    max_outputs: int = 1

    def __post_init__(self) -> None:
        """Refuse a configuration that cannot mean what it says.

        `max_outputs` spent a long time being read by nothing, so a value
        that is silently ignored is a mistake this file has already made
        once. The same shape is available through the bounds: `NAMES` holds
        six names and the DSL caps inputs at six, so asking for eight
        produced a six-input spec and no complaint.
        """
        if not 1 <= self.min_inputs <= self.max_inputs:
            raise ValueError(f"inputs {self.min_inputs}..{self.max_inputs} is not a range")
        if self.max_inputs > MAX_INPUTS:
            raise ValueError(
                f"max_inputs={self.max_inputs} exceeds the v1 limit of {MAX_INPUTS}"
            )
        if self.max_inputs > len(NAMES):
            raise ValueError(f"only {len(NAMES)} port names are defined")
        if not 1 <= self.max_outputs <= MAX_OUTPUTS:
            raise ValueError(
                f"max_outputs={self.max_outputs} is outside 1..{MAX_OUTPUTS}"
            )
        if not 0 <= self.min_gates <= self.max_gates:
            raise ValueError(f"gates {self.min_gates}..{self.max_gates} is not a range")
        if not self.gate_set:
            raise ValueError("gate_set is empty; nothing could be sampled")

    def with_gates(self, lo: int, hi: int) -> SampleConfig:
        from dataclasses import replace

        return replace(self, min_gates=lo, max_gates=hi)


def _random_tree(rng: random.Random, leaves: list, gate_set: tuple[str, ...]):
    """Combine ``leaves`` into one expression with ``len(leaves) - 1`` binary ops."""
    pool = list(leaves)
    rng.shuffle(pool)
    while len(pool) > 1:
        a = pool.pop(rng.randrange(len(pool)))
        b = pool.pop(rng.randrange(len(pool)))
        pool.append(Binary(rng.choice(gate_set), a, b))
        rng.shuffle(pool)
    return pool[0]


def _insert_negations(rng: random.Random, expr, count: int):
    """Wrap ``count`` randomly chosen subtrees in a negation.

    Negations are placed after the tree is built rather than during, so the
    binary structure and the inversion structure vary independently. Doing both
    at once correlates them, and the corpus ends up with every NOT sitting in
    the same place relative to its operands.
    """
    def collect(node, setter, sites):
        """Every subtree, paired with a way to replace it in place."""
        sites.append((node, setter))
        if isinstance(node, Not):
            collect(node.operand, _field_setter(node, "operand"), sites)
        elif isinstance(node, Binary):
            collect(node.left, _field_setter(node, "left"), sites)
            collect(node.right, _field_setter(node, "right"), sites)

    for _ in range(count):
        holder = [expr]
        sites: list = []
        collect(expr, _list_setter(holder), sites)
        node, setter = rng.choice(sites)
        # Never stack two negations: the rewriter collapses them, and the
        # sampled gate count would then be a lie the corpus is stratified on.
        if isinstance(node, Not):
            continue
        setter(Not(node))
        expr = holder[0]
    return expr


def _field_setter(node, field_name: str):
    """A setter for one field of a frozen AST node."""
    return lambda value: object.__setattr__(node, field_name, value)


def _list_setter(holder: list):
    """A setter for the root slot, so replacing the whole tree works too."""
    return lambda value: holder.__setitem__(0, value)


def _gate_count(expr) -> int:
    if isinstance(expr, Ref):
        return 0
    if isinstance(expr, Not):
        return 1 + _gate_count(expr.operand)
    if isinstance(expr, Binary):
        return 1 + _gate_count(expr.left) + _gate_count(expr.right)
    return 0


def sample_spec(rng: random.Random, cfg: SampleConfig = SampleConfig()) -> Spec:
    """Draw one interesting spec. Retries internally until it finds one."""
    for _ in range(200):
        n_inputs = rng.randint(cfg.min_inputs, cfg.max_inputs)
        gates = rng.randint(max(cfg.min_gates, n_inputs - 1), cfg.max_gates)
        # A binary tree with `b` operators has `b + 1` leaves, and every input
        # needs at least one leaf, so `b` cannot be below `n_inputs - 1`.
        n_binary = rng.randint(max(0, n_inputs - 1), gates)
        n_unary = gates - n_binary

        names = list(NAMES[:n_inputs])

        def draw(binary: int, unary: int, names=names):
            leaves = [Ref(name) for name in names]
            while len(leaves) < binary + 1:
                leaves.append(Ref(rng.choice(names)))
            return _insert_negations(rng, _random_tree(rng, leaves, cfg.gate_set), unary)

        n_outputs = rng.randint(1, max(1, cfg.max_outputs))
        if n_outputs == 1:
            rules = {"Q": draw(n_binary, n_unary)}
        else:
            # Split the gate budget between the outputs so a two-output spec is
            # not simply twice the circuit. Each output gets at least one gate.
            rules = {}
            remaining_binary, remaining_unary = n_binary, n_unary
            for j in range(n_outputs):
                left = n_outputs - j - 1
                # Clamped: with fewer binary gates than outputs left to fill,
                # the remaining outputs are single-leaf rules like `Q1 = A`,
                # which are legal and are how a spec ends up routing an input
                # straight through beside a computed one.
                take_b = (
                    remaining_binary
                    if left == 0
                    else rng.randint(0, max(0, remaining_binary - left))
                )
                take_u = remaining_unary if left == 0 else rng.randint(0, remaining_unary)
                remaining_binary -= take_b
                remaining_unary -= take_u
                rules[f"Q{j}"] = draw(take_b, take_u)

        parsed = ParsedSpec(inputs=names, outputs=list(rules), rules=rules)
        if irrelevant_inputs(parsed):
            continue
        spec = Spec.from_parsed(parsed)
        if spec.is_constant():
            continue
        if not cfg.min_gates <= spec.gates <= cfg.max_gates:
            continue
        if rng.random() < cfg.constraint_rate:
            spec = spec.with_constraints(_random_constraint(rng, spec))
        return spec
    raise RuntimeError("sampler could not find a non-degenerate spec in 200 tries")


def _random_constraint(rng: random.Random, spec: Spec) -> Constraints:
    """A constraint loose enough that a good layout can satisfy it.

    Sampling constraints blindly would produce specs nothing can satisfy, and
    the corpus would teach the model that constraints are noise.
    """
    kind = rng.choice(("latency", "footprint", "region"))
    if kind == "latency":
        return Constraints(max_latency_rt=spec.gates + rng.randint(2, 5))
    if kind == "footprint":
        return Constraints(max_blocks=rng.randint(60, 120))
    return Constraints(max_region=(16, rng.randint(10, 16)))


def sample_unique(
    rng: random.Random,
    count: int,
    cfg: SampleConfig = SampleConfig(),
    seen: set[int] | None = None,
) -> list[Spec]:
    """Sample ``count`` specs with distinct behaviour.

    Deduplication is by semantic hash, so two specs that compute the same
    function are one spec however differently they are written. That is the
    same key the validation split uses to mean "unseen spec", so a leak here
    would quietly turn generalisation into memorisation.
    """
    seen = seen if seen is not None else set()
    out: list[Spec] = []
    misses = 0
    while len(out) < count and misses < count * 20 + 50:
        spec = sample_spec(rng, cfg)
        key = spec.semantic_hash()
        if key in seen:
            misses += 1
            continue
        seen.add(key)
        out.append(spec)
    return out


def enumerate_small_specs(n_inputs: int) -> list[Spec]:
    """Every distinct boolean function of ``n_inputs`` variables, as a spec.

    Useful as a held-out probe: at two inputs there are only sixteen functions,
    so "did the model learn all of them" is a question with an exact answer
    rather than a sample estimate.
    """
    names = list(NAMES[:n_inputs])
    rows_count = 1 << n_inputs
    out = []
    for table in range(1 << rows_count):
        rows = tuple((table >> m) & 1 for m in range(rows_count))
        if len(set(rows)) <= 1:
            continue
        expr = _sum_of_products(names, rows)
        if expr is None:
            continue
        parsed = ParsedSpec(inputs=names, outputs=["Q"], rules={"Q": expr})
        if irrelevant_inputs(parsed) or is_constant(rows):
            continue
        out.append(Spec.from_parsed(parsed))
    return out


def _sum_of_products(names: list[str], rows: tuple[int, ...]):
    """Canonical DNF for a truth table. Not minimal, and not meant to be — the
    procedural compiler minimises, and the point here is coverage."""
    terms = []
    for m, value in enumerate(rows):
        if not value:
            continue
        literals = []
        for k, name in enumerate(names):
            ref = Ref(name)
            literals.append(ref if (m >> k) & 1 else Not(ref))
        term = literals[0]
        for lit in literals[1:]:
            term = Binary("and", term, lit)
        terms.append(term)
    if not terms:
        return None
    expr = terms[0]
    for t in terms[1:]:
        expr = Binary("or", expr, t)
    return expr
