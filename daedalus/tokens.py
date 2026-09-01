"""Sequence construction: spec prefix + grid body.

The model sees a single flat sequence of ``PREFIX_LEN + CELLS`` token ids. The
prefix carries the specification; the body is the grid in ``y -> z -> x``
order, which is the same bytes the verifier reads. Nothing is translated
between "the training representation" and "the real one" — there is only one.

Two conditioning paths share the prefix (§05):

* the **canonical spec path**, which tokenises the DSL directly. Exact, cheap,
  and what the verifier loop uses internally.
* the **natural language path**, where a frozen sentence encoder produces a
  small number of continuous vectors. Those cannot live in a discrete
  sequence, so the prefix reserves slots for them and the model splices the
  projected embeddings in. :func:`spec_prefix` marks the slots.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import vocab as V

#: The prefix has its own small vocabulary, separate from the block states.
#: Sharing one vocabulary would let the model confuse "this cell is a repeater"
#: with "this spec mentions a repeater", which is not a distinction worth
#: making it learn.
PREFIX_VOCAB = {
    "<pad>": 0,
    "<spec>": 1,  # start of the canonical spec
    "<nl>": 2,  # slot for a projected sentence embedding
    "<in>": 3,  # input port count follows
    "<out>": 4,  # output port count follows
    "<row>": 5,  # a truth-table row follows
    "<lat>": 6,  # latency constraint follows
    "<blk>": 7,  # footprint constraint follows
    "<reg>": 8,  # region constraint follows
    "<end>": 9,
}
#: Small integers are encoded literally after the marker tokens.
PREFIX_NUM_BASE = 16
PREFIX_NUM_MAX = 64
PREFIX_VOCAB_SIZE = PREFIX_NUM_BASE + PREFIX_NUM_MAX

PAD = PREFIX_VOCAB["<pad>"]


def _num(v: int) -> int:
    if not 0 <= v < PREFIX_NUM_MAX:
        raise ValueError(f"{v} does not fit the prefix number range 0..{PREFIX_NUM_MAX - 1}")
    return PREFIX_NUM_BASE + v


@dataclass(frozen=True, slots=True)
class Sequence:
    """A tokenised example."""

    prefix: tuple[int, ...]
    body: tuple[int, ...]
    #: Positions in the prefix that hold a sentence embedding rather than a
    #: discrete token.
    nl_slots: tuple[int, ...] = ()

    def __post_init__(self):
        if len(self.prefix) != V.PREFIX_LEN:
            raise ValueError(f"prefix must be {V.PREFIX_LEN} tokens, got {len(self.prefix)}")
        if len(self.body) != V.CELLS:
            raise ValueError(f"body must be {V.CELLS} tokens, got {len(self.body)}")

    def flat(self) -> list[int]:
        """The full sequence. Prefix ids are offset past the block vocabulary
        so a single embedding table can serve both."""
        return [V.VOCAB_SIZE + t for t in self.prefix] + list(self.body)

    @property
    def length(self) -> int:
        return V.SEQ_LEN


#: Total embedding table size when prefix and body share one table.
TOTAL_VOCAB = V.VOCAB_SIZE + PREFIX_VOCAB_SIZE


def spec_prefix(placed_spec, nl_slots: int = 0) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Encode a placed spec into the fixed-width prefix.

    The truth table is written row by row. For up to four inputs that is at
    most sixteen rows and fits comfortably; beyond that the table is truncated
    and the model has to generalise from the port count and the rows it can
    see. That is a real limitation of a 32-token prefix and it is why the
    extrapolation split of §04 is held out by gate count rather than by input
    count.
    """
    toks: list[int] = [PREFIX_VOCAB["<spec>"]]
    slots: list[int] = []
    for _ in range(nl_slots):
        slots.append(len(toks))
        toks.append(PREFIX_VOCAB["<nl>"])

    toks += [PREFIX_VOCAB["<in>"], _num(len(placed_spec.input_ports))]
    toks += [PREFIX_VOCAB["<out>"], _num(len(placed_spec.output_ports))]

    c = placed_spec.constraints
    if c.max_latency_rt is not None:
        toks += [PREFIX_VOCAB["<lat>"], _num(min(c.max_latency_rt, PREFIX_NUM_MAX - 1))]
    if c.max_blocks is not None:
        toks += [PREFIX_VOCAB["<blk>"], _num(min(c.max_blocks, PREFIX_NUM_MAX - 1))]
    if c.max_region is not None:
        toks += [PREFIX_VOCAB["<reg>"], _num(c.max_region[0]), _num(c.max_region[1])]

    for row in placed_spec.rows:
        if len(toks) + 3 > V.PREFIX_LEN:
            break
        toks += [PREFIX_VOCAB["<row>"], _num(row & (PREFIX_NUM_MAX - 1))]

    if len(toks) < V.PREFIX_LEN:
        toks.append(PREFIX_VOCAB["<end>"])
    toks = toks[: V.PREFIX_LEN]
    toks += [PAD] * (V.PREFIX_LEN - len(toks))
    return tuple(toks), tuple(slots)


def encode(grid, placed_spec, nl_slots: int = 0) -> Sequence:
    """Tokenise a ``(grid, spec)`` pair."""
    prefix, slots = spec_prefix(placed_spec, nl_slots)
    body = tuple(grid.cells if hasattr(grid, "cells") else grid)
    return Sequence(prefix=prefix, body=body, nl_slots=slots)


def decode_body(body) -> list[int]:
    """Body tokens as a plain list, for handing to :class:`~daedalus.grid.Grid`."""
    return list(body)


# --------------------------------------------------------------------------
# legality masking (§05)
# --------------------------------------------------------------------------


def legality_mask(placed_spec=None) -> list[list[bool]]:
    """Per-cell mask of block states that are physically possible there.

    Free correctness with no training cost: zeroing the logits of impossible
    states at sample time stops the model wasting probability mass on grids the
    verifier would reject before simulating them.

    Rows are ``TOTAL_VOCAB`` wide, not ``V.VOCAB_SIZE`` — the model's head
    spans blocks *and* prefix tokens, so the mask has to line up with it. The
    prefix half is always false: a spec token in a grid cell is not a block,
    and this is the only thing standing between the sampler and emitting one.

    This is the *position-only* part of legality — what a coordinate can decide
    on its own. Neighbour-dependent rules (dust needs support, a torch needs
    something to hang on) need the whole grid and belong in the sampler, which
    knows what it has committed to so far.

    Pass ``placed_spec`` to also fix the port cells in both directions: levers
    and lamps only at declared ports, and nothing *but* the declared block at
    one. They are the states whose legality is decided entirely by the spec
    rather than by physics, and either mistake is a port violation the
    verifier will reject.
    """
    levers = [t for t in range(V.VOCAB_SIZE) if _is_kind(t, "lever")]
    lamps = [t for t in range(V.VOCAB_SIZE) if _is_kind(t, "lamp")]
    inputs, outputs = set(), set()
    fixed: dict[int, int] = {}
    if placed_spec is not None:
        inputs = {V.index(*p) for p in placed_spec.input_ports}
        outputs = {V.index(*p) for p in placed_spec.output_ports}
        fixed = port_mask(placed_spec)

    mask = []
    for i in range(V.CELLS):
        _x, y, _z = V.unindex(i)
        row = [V.legal_at(t, y) for t in range(V.VOCAB_SIZE)]
        row.extend([False] * (TOTAL_VOCAB - V.VOCAB_SIZE))
        if placed_spec is not None:
            for t in levers:
                row[t] = row[t] and i in inputs
            for t in lamps:
                row[t] = row[t] and i in outputs
            # And the other direction. A port cell is not the model's to
            # choose -- `port_mask` overwrites it anyway -- so leaving air
            # legal at an output port only let the sampler spend probability
            # on grids with no lamp in them, which the verifier then rejects
            # as a port violation. Pinning both directions is what the
            # docstring above claims this mask is for.
            pinned = fixed.get(i)
            if pinned is not None:
                row = [t == pinned for t in range(TOTAL_VOCAB)]
        mask.append(row)
    return mask


def _is_kind(token: int, kind: str) -> bool:
    if V.is_control(token):
        return False
    return V.decode(token).kind == kind


def port_mask(placed_spec) -> dict[int, int]:
    """Cells whose contents the spec already fixes.

    Port positions are not the model's to choose in v1, so pinning them costs
    nothing and removes a whole class of malformed samples.
    """
    fixed: dict[int, int] = {}
    for x, y, z in placed_spec.input_ports:
        fixed[V.index(x, y, z)] = V.lever(V.Dir4.EAST)
        fixed[V.index(x + 1, y, z)] = V.SOLID
    for x, y, z in placed_spec.output_ports:
        fixed[V.index(x, y, z)] = V.LAMP
    return fixed
