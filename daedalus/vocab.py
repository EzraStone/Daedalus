"""The 48-token block-state vocabulary and the grid geometry.

This module is the Python mirror of ``crates/redsim/src/block.rs`` and
``grid.rs``. The token ids are a wire format: they are baked into every
serialised corpus and every trained checkpoint, so the two implementations
have to agree exactly. ``tests/test_vocab_parity.py`` checks that by asking
the Rust binary to round-trip the whole vocabulary.

One token per *fully resolved* block state, not per block type. Two properties
are deliberately excluded because they are derived rather than chosen:

* ``redstone_wire.power`` is simulator output. Including it would let a
  generator assert a power level the physics cannot produce.
* ``redstone_wire``'s connection shape is determined by its neighbours, so
  including it would let a generator emit self-contradictory grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

SX = 16
SY = 6
SZ = 16
CELLS = SX * SY * SZ

#: Layer 0 is always the substrate; dust lives at y >= 1. Fixing that removes
#: an entire class of "floating dust" invalid samples by construction.
SUBSTRATE_Y = 0
LOGIC_Y = 1

#: Inputs on the x=0 face, outputs on x=15.
INPUT_X = 0
OUTPUT_X = SX - 1

#: Spec prefix length, in tokens. §02 budgets 1536 + 32.
PREFIX_LEN = 32
SEQ_LEN = CELLS + PREFIX_LEN


def index(x: int, y: int, z: int) -> int:
    """Flat index in ``y -> z -> x`` raster order (layer-major).

    Layer-major keeps a full planar layer contiguous in the sequence, so a
    local-attention model sees a whole slice at once.
    """
    return (y * SZ + z) * SX + x


def unindex(i: int) -> tuple[int, int, int]:
    return i % SX, i // (SX * SZ), (i // SX) % SZ


def in_bounds(x: int, y: int, z: int) -> bool:
    return 0 <= x < SX and 0 <= y < SY and 0 <= z < SZ


# --------------------------------------------------------------------------
# directions
# --------------------------------------------------------------------------


class Dir4(IntEnum):
    """Horizontal direction. North is -z, South is +z, West is -x, East is +x."""

    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3

    @property
    def delta(self) -> tuple[int, int]:
        return {Dir4.NORTH: (0, -1), Dir4.SOUTH: (0, 1), Dir4.WEST: (-1, 0), Dir4.EAST: (1, 0)}[
            self
        ]

    @property
    def opposite(self) -> Dir4:
        return {
            Dir4.NORTH: Dir4.SOUTH,
            Dir4.SOUTH: Dir4.NORTH,
            Dir4.WEST: Dir4.EAST,
            Dir4.EAST: Dir4.WEST,
        }[self]

    def same_axis(self, other: Dir4) -> bool:
        return (self in (Dir4.NORTH, Dir4.SOUTH)) == (other in (Dir4.NORTH, Dir4.SOUTH))


DIR4 = (Dir4.NORTH, Dir4.SOUTH, Dir4.WEST, Dir4.EAST)


class Dir6(IntEnum):
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3
    UP = 4
    DOWN = 5


class Attach(IntEnum):
    """Direction from a wall-mountable component to the block supporting it."""

    FLOOR = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4

    @property
    def delta(self) -> tuple[int, int, int]:
        return {
            Attach.FLOOR: (0, -1, 0),
            Attach.NORTH: (0, 0, -1),
            Attach.SOUTH: (0, 0, 1),
            Attach.WEST: (-1, 0, 0),
            Attach.EAST: (1, 0, 0),
        }[self]


class CmpMode(IntEnum):
    COMPARE = 0
    SUBTRACT = 1


# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------

VOCAB_SIZE = 48
#: Ids below this are real block states; the rest are sequence control tokens.
CONTROL_BASE = 44
PAD, BOS, EOS, MASK = 44, 45, 46, 47

AIR = 0
SOLID = 1
WIRE = 2
LAMP = 36
TARGET = 37

_TORCH_BASE = 3
_REPEATER_BASE = 8
_COMPARATOR_BASE = 24
_LEVER_BASE = 32
_OBSERVER_BASE = 38


def torch(attach: Attach = Attach.FLOOR) -> int:
    return _TORCH_BASE + int(attach)


def repeater(facing: Dir4, delay: int = 1) -> int:
    if not 1 <= delay <= 4:
        raise ValueError(f"repeater delay must be 1..4, got {delay}")
    return _REPEATER_BASE + int(facing) * 4 + (delay - 1)


def comparator(facing: Dir4, mode: CmpMode = CmpMode.COMPARE) -> int:
    return _COMPARATOR_BASE + int(facing) * 2 + int(mode)


def lever(attach: Dir4 = Dir4.EAST) -> int:
    return _LEVER_BASE + int(attach)


def observer(facing: Dir6) -> int:
    return _OBSERVER_BASE + int(facing)


@dataclass(frozen=True, slots=True)
class Decoded:
    """A token unpacked into its block type and state."""

    kind: str
    attach: Attach | None = None
    facing: Dir4 | Dir6 | None = None
    delay: int | None = None
    mode: CmpMode | None = None


def decode(token: int) -> Decoded:
    """Unpack a token id. Raises for control tokens and out-of-range ids."""
    if token == AIR:
        return Decoded("air")
    if token == SOLID:
        return Decoded("solid")
    if token == WIRE:
        return Decoded("wire")
    if _TORCH_BASE <= token < _REPEATER_BASE:
        return Decoded("torch", attach=Attach(token - _TORCH_BASE))
    if _REPEATER_BASE <= token < _COMPARATOR_BASE:
        k = token - _REPEATER_BASE
        return Decoded("repeater", facing=Dir4(k // 4), delay=(k % 4) + 1)
    if _COMPARATOR_BASE <= token < _LEVER_BASE:
        k = token - _COMPARATOR_BASE
        return Decoded("comparator", facing=Dir4(k // 2), mode=CmpMode(k % 2))
    if _LEVER_BASE <= token < LAMP:
        return Decoded("lever", attach=Dir4(token - _LEVER_BASE))
    if token == LAMP:
        return Decoded("lamp")
    if token == TARGET:
        return Decoded("target")
    if _OBSERVER_BASE <= token < CONTROL_BASE:
        return Decoded("observer", facing=Dir6(token - _OBSERVER_BASE))
    if CONTROL_BASE <= token < VOCAB_SIZE:
        raise ValueError(f"token {token} is a control token, not a block state")
    raise ValueError(f"token {token} is outside the vocabulary")


#: Single-character rendering, matching ``redsim::grid::glyph``.
GLYPH = {
    "air": ".",
    "solid": "#",
    "wire": "d",
    "torch": "t",
    "repeater": ">",
    "comparator": "c",
    "lever": "V",
    "lamp": "L",
    "target": "T",
    "observer": "o",
}


def glyph(token: int) -> str:
    try:
        return GLYPH[decode(token).kind]
    except ValueError:
        return {PAD: "_", BOS: "^", EOS: "$", MASK: "?"}.get(token, "!")


# --------------------------------------------------------------------------
# derived properties, mirroring redsim::block
# --------------------------------------------------------------------------

#: Blocks that occupy their cell as a full opaque cube. Opacity drives three
#: separate rules: dust needs an opaque block beneath it, dust cannot slope up
#: past one, and only opaque blocks carry weak/strong power.
_OPAQUE = frozenset({SOLID, LAMP, TARGET}) | frozenset(
    range(_OBSERVER_BASE, CONTROL_BASE)
)
#: Blocks that can hold weak/strong power as a block rather than as a component.
_CONDUCTIVE = frozenset({SOLID, LAMP, TARGET})


def is_opaque(token: int) -> bool:
    return token in _OPAQUE


def is_conductive(token: int) -> bool:
    return token in _CONDUCTIVE


def supports_dust(token: int) -> bool:
    return is_opaque(token)


def is_material(token: int) -> bool:
    return token != AIR


def is_control(token: int) -> bool:
    return token >= CONTROL_BASE


#: Every legal block-state token, in id order.
BLOCK_TOKENS = tuple(range(CONTROL_BASE))

#: Components v1 puts in the vocabulary but never generates. Observers are
#: edge-triggered, which makes a circuit sequential; the spec DSL is
#: combinational, so there is nothing coherent to condition them on.
EXCLUDED_FROM_GENERATION = frozenset(range(_OBSERVER_BASE, CONTROL_BASE))


def legal_at(token: int, y: int) -> bool:
    """Is this block state placeable at layer ``y`` at all?

    A coarse, position-only legality test used for logit masking at sample
    time (§05). It deliberately ignores neighbour-dependent rules — those need
    the whole grid — and only encodes what a single coordinate can decide.
    """
    if token in EXCLUDED_FROM_GENERATION or is_control(token):
        return False
    if y == SUBSTRATE_Y:
        # The substrate layer is solid or air and nothing else; components
        # there would have no support and dust would be underground.
        return token in (AIR, SOLID)
    return True


def state_string(token: int) -> str:
    """Canonical Minecraft block-state string, for ``.schem`` export."""
    d = decode(token)
    if d.kind == "air":
        return "minecraft:air"
    if d.kind == "solid":
        return "minecraft:stone"
    if d.kind == "wire":
        return "minecraft:redstone_wire[power=0]"
    if d.kind == "torch":
        if d.attach is Attach.FLOOR:
            return "minecraft:redstone_torch[lit=true]"
        # A wall torch's `facing` points away from its support, i.e. the
        # opposite of our attach direction.
        away = {
            Attach.NORTH: "south",
            Attach.SOUTH: "north",
            Attach.WEST: "east",
            Attach.EAST: "west",
        }[d.attach]
        return f"minecraft:redstone_wall_torch[facing={away},lit=true]"
    if d.kind == "repeater":
        return (
            f"minecraft:repeater[facing={_NAME4[d.facing.opposite]},"
            f"delay={d.delay},locked=false,powered=false]"
        )
    if d.kind == "comparator":
        mode = "compare" if d.mode is CmpMode.COMPARE else "subtract"
        return (
            f"minecraft:comparator[facing={_NAME4[d.facing.opposite]},"
            f"mode={mode},powered=false]"
        )
    if d.kind == "lever":
        return f"minecraft:lever[face=wall,facing={_NAME4[d.attach.opposite]},powered=false]"
    if d.kind == "lamp":
        return "minecraft:redstone_lamp[lit=false]"
    if d.kind == "target":
        return "minecraft:target[power=0]"
    if d.kind == "observer":
        return f"minecraft:observer[facing={_NAME6[d.facing]},powered=false]"
    raise AssertionError(f"unreachable kind {d.kind}")


_NAME4 = {Dir4.NORTH: "north", Dir4.SOUTH: "south", Dir4.WEST: "west", Dir4.EAST: "east"}
_NAME6 = {
    Dir6.NORTH: "north",
    Dir6.SOUTH: "south",
    Dir6.WEST: "west",
    Dir6.EAST: "east",
    Dir6.UP: "up",
    Dir6.DOWN: "down",
}
