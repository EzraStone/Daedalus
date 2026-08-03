"""Canonicalisation: truth tables, semantic hashing and normal-form text.

A spec's *meaning* is its truth table. Two specs that agree on every row are
the same problem however differently they are written, and the corpus needs to
know that: it is the deduplication key, it decides what "unseen spec" means
for the validation split, and it is the basis of the novelty metric.
"""

from __future__ import annotations

from itertools import product

from .dsl import Constraints, Expr, ParsedSpec, evaluate_expr, format_expr

#: FNV-1a offset basis and prime, matching ``redsim::spec::Spec::semantic_hash``.
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1


def truth_table(parsed: ParsedSpec) -> tuple[int, ...]:
    """Rows of the truth table, as output bitmasks.

    ``rows[m]`` holds the outputs for the input assignment whose bitmask is
    ``m``: input ``k`` is bit ``k`` of ``m``, output ``j`` is bit ``j`` of the
    row. This is exactly the layout the verifier reads.
    """
    n = len(parsed.inputs)
    rows = []
    for m in range(1 << n):
        env = {name: bool(m >> k & 1) for k, name in enumerate(parsed.inputs)}
        row = 0
        for j, out in enumerate(parsed.outputs):
            if evaluate_expr(parsed.rules[out], env):
                row |= 1 << j
        rows.append(row)
    return tuple(rows)


def semantic_hash(n_inputs: int, n_outputs: int, rows) -> int:
    """A 64-bit fingerprint of what a spec means.

    FNV-1a rather than Python's ``hash`` because the value is written into
    dataset files and compared against the Rust side; it has to be stable
    across processes, releases and languages.
    """
    h = _FNV_OFFSET
    def mix(b: int) -> None:
        nonlocal h
        h = ((h ^ (b & 0xFF)) * _FNV_PRIME) & _MASK64

    mix(n_inputs)
    mix(n_outputs)
    for r in rows:
        for k in range(8):
            mix(r >> (k * 8))
    return h


def canonical_source(parsed: ParsedSpec, ascii_only: bool = False) -> str:
    """Re-render a spec in a normal form.

    Identifiers keep their names — this is a *presentation* normal form for
    diffing and logging, not the semantic key. Use :func:`semantic_hash` when
    identity is the question.
    """
    lines = [
        "inputs " + " ".join(parsed.inputs),
        "outputs " + " ".join(parsed.outputs),
    ]
    lines += parsed.constraints.describe()
    for out in parsed.outputs:
        lines.append(f"{out} = {format_expr(parsed.rules[out], ascii_only)}")
    return "\n".join(lines)


def table_text(inputs, outputs, rows) -> str:
    """A human-readable truth table, for logs and failure messages."""
    header = " ".join(inputs) + " | " + " ".join(outputs)
    lines = [header, "-" * len(header)]
    for m, row in enumerate(rows):
        ins = " ".join(str(m >> k & 1).rjust(len(name)) for k, name in enumerate(inputs))
        outs = " ".join(str(row >> j & 1).rjust(len(name)) for j, name in enumerate(outputs))
        lines.append(f"{ins} | {outs}")
    return "\n".join(lines)


def depends_on(parsed: ParsedSpec, name: str) -> bool:
    """Does any output actually change when ``name`` changes?

    Syntactic reference is not the same as functional dependence: ``Q = A ∧ ¬A``
    mentions ``A`` and ignores it. A spec with a functionally irrelevant input
    is a trap for the corpus — the placer routes a wire that provably cannot
    matter, and the model learns to ignore a port.
    """
    n = len(parsed.inputs)
    k = parsed.inputs.index(name)
    rows = truth_table(parsed)
    return any(rows[m] != rows[m ^ (1 << k)] for m in range(1 << n))


def irrelevant_inputs(parsed: ParsedSpec) -> list[str]:
    return [name for name in parsed.inputs if not depends_on(parsed, name)]


def is_constant(rows) -> bool:
    """True when the outputs never change. Constant specs are legal but
    uninteresting, and the corpus builder drops them."""
    return len(set(rows)) <= 1


def restrict(parsed: ParsedSpec, keep: list[str]) -> tuple[int, ...]:
    """Truth table over a subset of the inputs, the rest held low.

    Used by the ablation reporting in §07 to ask what a circuit does when only
    some of its ports are in play.
    """
    unknown = [k for k in keep if k not in parsed.inputs]
    if unknown:
        raise ValueError(f"{unknown} are not inputs of this spec")
    rows = []
    for bits in product((0, 1), repeat=len(keep)):
        env = dict.fromkeys(parsed.inputs, False)
        env.update(dict(zip(keep, (bool(b) for b in bits))))
        row = 0
        for j, out in enumerate(parsed.outputs):
            if evaluate_expr(parsed.rules[out], env):
                row |= 1 << j
        rows.append(row)
    return tuple(rows)


__all__ = [
    "Constraints",
    "Expr",
    "canonical_source",
    "depends_on",
    "irrelevant_inputs",
    "is_constant",
    "restrict",
    "semantic_hash",
    "table_text",
    "truth_table",
]
