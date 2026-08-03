"""Writing circuits out in formats Minecraft tooling can read.

Two formats, because the two audiences differ:

``.schem`` (Sponge schematic v2)
    What WorldEdit, Litematica and most servers accept. This is the one that
    matters for the claim that a stranger can go from prompt to a circuit in
    their own world.
``.litematic`` (Litematica)
    A client-side mod format, useful for ghost-overlay building — you see the
    circuit projected into the world and place the blocks yourself.

Both are written through the vendored NBT encoder, so exporting needs no
third-party package.
"""

from __future__ import annotations

from pathlib import Path

from .. import vocab as V
from ..grid import Grid
from .nbt import (
    ByteArray,
    Int,
    IntArray,
    List,
    Long,
    LongArray,
    Short,
    String,
    dumps,
    varint,
)

__all__ = ["write_schem", "write_litematic", "palette_of", "trimmed"]


def palette_of(grid: Grid) -> tuple[dict[str, int], list[int]]:
    """Map the block states present in a grid to palette indices.

    Air is forced to index 0. Most of the grid is air, and a palette where the
    commonest entry encodes as a single zero byte keeps the file small.
    """
    palette: dict[str, int] = {V.state_string(V.AIR): 0}
    indices: list[int] = []
    for token in grid.cells:
        name = V.state_string(token)
        if name not in palette:
            palette[name] = len(palette)
        indices.append(palette[name])
    return palette, indices


def trimmed(grid: Grid) -> tuple[Grid, tuple[int, int, int]]:
    """Crop to the occupied bounding box, including the substrate under it.

    Exporting the full 16x6x16 volume would paste a 256-block stone slab into
    somebody's world along with the circuit. Cropping keeps the substrate that
    the circuit actually stands on and nothing else.
    """
    occupied = [
        (x, y, z)
        for y in range(V.SY)
        for z in range(V.SZ)
        for x in range(V.SX)
        if grid.get(x, y, z) != V.AIR
    ]
    if not occupied:
        return Grid(), (0, 0, 0)
    x0 = min(p[0] for p in occupied)
    x1 = max(p[0] for p in occupied)
    y0 = min(p[1] for p in occupied)
    y1 = max(p[1] for p in occupied)
    z0 = min(p[2] for p in occupied)
    z1 = max(p[2] for p in occupied)
    return grid, (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)


def _sponge_block_data(grid: Grid, palette: dict[str, int]) -> bytes:
    """Blocks in Sponge order: x fastest, then z, then y.

    Note this is *not* the token order, which is ``y -> z -> x``. Getting the
    two confused produces a schematic that is a transposed version of the
    circuit and fails silently — it still places, it just does nothing.
    """
    out = bytearray()
    for y in range(V.SY):
        for z in range(V.SZ):
            for x in range(V.SX):
                out += varint(palette[V.state_string(grid.get(x, y, z))])
    return bytes(out)


def write_schem(grid: Grid, path: str | Path, name: str = "daedalus") -> Path:
    """Write a Sponge schematic v2 (``.schem``)."""
    palette, _ = palette_of(grid)
    root = {
        "Version": Int(2),
        "DataVersion": Int(3465),  # 1.20.1
        "Width": Short(V.SX),
        "Height": Short(V.SY),
        "Length": Short(V.SZ),
        "Offset": IntArray([0, 0, 0]),
        "PaletteMax": Int(len(palette)),
        "Palette": {k: Int(v) for k, v in palette.items()},
        "BlockData": ByteArray(_sponge_block_data(grid, palette)),
        "BlockEntities": List([], element_id=10),
        "Metadata": {
            "Name": String(name),
            "Author": String("Daedalus"),
            "WEOffsetX": Int(0),
            "WEOffsetY": Int(0),
            "WEOffsetZ": Int(0),
        },
    }
    p = Path(path)
    p.write_bytes(dumps(root, "Schematic"))
    return p


def _litematica_states(grid: Grid, palette: dict[str, int]) -> list[int]:
    """Pack palette indices into a long array at the format's bit width.

    Litematica uses the smallest bit count that fits the palette, minimum two,
    and entries may straddle a long boundary. Getting the straddle wrong
    corrupts roughly one block in sixty-four — frequent enough to be obvious,
    rare enough to look like a different bug.
    """
    bits = max(2, (len(palette) - 1).bit_length())
    total = V.CELLS
    longs = [0] * (((total * bits) + 63) // 64)
    for i in range(total):
        # Litematica order is x fastest, then z, then y.
        y = i // (V.SX * V.SZ)
        z = (i // V.SX) % V.SZ
        x = i % V.SX
        value = palette[V.state_string(grid.get(x, y, z))]
        start_bit = i * bits
        start_long = start_bit // 64
        offset = start_bit % 64
        longs[start_long] |= (value << offset) & 0xFFFFFFFFFFFFFFFF
        if offset + bits > 64:
            longs[start_long + 1] |= value >> (64 - offset)
    # NBT longs are signed.
    return [v - (1 << 64) if v >= (1 << 63) else v for v in longs]


def write_litematic(grid: Grid, path: str | Path, name: str = "daedalus") -> Path:
    """Write a Litematica schematic (``.litematic``)."""
    palette, _ = palette_of(grid)
    ordered = [String(k) for k, _ in sorted(palette.items(), key=lambda kv: kv[1])]
    blocks = int(grid.material_blocks())
    region = {
        "Position": {"x": Int(0), "y": Int(0), "z": Int(0)},
        "Size": {"x": Int(V.SX), "y": Int(V.SY), "z": Int(V.SZ)},
        "BlockStatePalette": List(
            [{"Name": s} for s in ordered],
            element_id=10,
        ),
        "BlockStates": LongArray(_litematica_states(grid, palette)),
        "TileEntities": List([], element_id=10),
        "Entities": List([], element_id=10),
        "PendingBlockTicks": List([], element_id=10),
        "PendingFluidTicks": List([], element_id=10),
    }
    root = {
        "MinecraftDataVersion": Int(3465),
        "Version": Int(6),
        "Metadata": {
            "Name": String(name),
            "Author": String("Daedalus"),
            "Description": String("Generated by Daedalus and verified by redsim"),
            "RegionCount": Int(1),
            "TotalVolume": Int(V.CELLS),
            "TotalBlocks": Int(blocks),
            "TimeCreated": Long(0),
            "TimeModified": Long(0),
            "EnclosingSize": {"x": Int(V.SX), "y": Int(V.SY), "z": Int(V.SZ)},
        },
        "Regions": {name: region},
    }
    p = Path(path)
    p.write_bytes(dumps(root, ""))
    return p


def block_summary(grid: Grid) -> dict[str, int]:
    """How many of each block a circuit needs, for a materials list."""
    counts: dict[str, int] = {}
    for token in grid.cells:
        if token == V.AIR:
            continue
        kind = V.decode(token).kind
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))
