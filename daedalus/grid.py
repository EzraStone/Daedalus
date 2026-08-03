"""A 16x6x16 token grid, and the helpers that build one.

The grid is a flat ``bytearray`` of :data:`daedalus.vocab.CELLS` token ids in
``y -> z -> x`` order. That is deliberately the same bytes the model predicts
and the same bytes the verifier reads, so nothing is ever converted between
"the training representation" and "the real one".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import vocab as V


@dataclass(slots=True)
class Grid:
    """A dense grid of block-state tokens."""

    cells: bytearray = field(default_factory=lambda: bytearray(V.CELLS))

    @classmethod
    def with_substrate(cls) -> Grid:
        """All air except a solid layer at ``y = 0``."""
        g = cls()
        for z in range(V.SZ):
            for x in range(V.SX):
                g.cells[V.index(x, V.SUBSTRATE_Y, z)] = V.SOLID
        return g

    @classmethod
    def from_tokens(cls, tokens) -> Grid:
        buf = bytearray(tokens)
        if len(buf) != V.CELLS:
            raise ValueError(f"grid needs exactly {V.CELLS} tokens, got {len(buf)}")
        return cls(buf)

    def copy(self) -> Grid:
        return Grid(bytearray(self.cells))

    # -- access ------------------------------------------------------------

    def get(self, x: int, y: int, z: int) -> int:
        """Read a cell. Outside the build volume reads as air, never as stone,
        so a circuit cannot lean on the world border for support."""
        if not V.in_bounds(x, y, z):
            return V.AIR
        return self.cells[V.index(x, y, z)]

    def set(self, x: int, y: int, z: int, token: int) -> None:
        if V.in_bounds(x, y, z):
            self.cells[V.index(x, y, z)] = token

    def __getitem__(self, pos: tuple[int, int, int]) -> int:
        return self.get(*pos)

    def __setitem__(self, pos: tuple[int, int, int], token: int) -> None:
        self.set(*pos, token)

    # -- measurements ------------------------------------------------------

    def material_blocks(self) -> int:
        """Non-air cells above the substrate.

        The substrate is a fixed 256-block cost every circuit pays; counting it
        would swamp the compactness metric with a constant.
        """
        return sum(
            1
            for y in range(1, V.SY)
            for z in range(V.SZ)
            for x in range(V.SX)
            if self.cells[V.index(x, y, z)] != V.AIR
        )

    def bbox(self) -> tuple[int, int, int]:
        """``(dx, dy, dz)`` extents of the non-air cells above the substrate."""
        xs, ys, zs = [], [], []
        for y in range(1, V.SY):
            for z in range(V.SZ):
                for x in range(V.SX):
                    if self.cells[V.index(x, y, z)] != V.AIR:
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
        if not xs:
            return (0, 0, 0)
        return (max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, max(zs) - min(zs) + 1)

    def occupied_layers(self) -> list[int]:
        return [
            y
            for y in range(V.SY)
            if any(
                self.cells[V.index(x, y, z)] != V.AIR
                for z in range(V.SZ)
                for x in range(V.SX)
            )
        ]

    # -- rendering ---------------------------------------------------------

    def layer_text(self, y: int) -> str:
        return "\n".join(
            "".join(V.glyph(self.cells[V.index(x, y, z)]) for x in range(V.SX))
            for z in range(V.SZ)
        )

    def render(self, skip_substrate: bool = True) -> str:
        """An ASCII view, one block per layer. Loses orientation detail on
        purpose — it is for eyeballing a layout, not for round-tripping one."""
        out = []
        for y in self.occupied_layers():
            if skip_substrate and y == V.SUBSTRATE_Y:
                continue
            out.append(f"-- y={y}")
            out.append(self.layer_text(y))
        return "\n".join(out)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.render()

    # -- serialisation -----------------------------------------------------

    def to_bytes(self) -> bytes:
        return bytes(self.cells)

    def tokens(self) -> list[int]:
        return list(self.cells)


def edit_distance(a: Grid, b: Grid) -> int:
    """Number of cells that differ.

    Used to rank the replay buffer in §06: a candidate a couple of blocks from
    a known-good layout is the most informative kind of failure.
    """
    return sum(1 for x, y in zip(a.cells, b.cells) if x != y)


def diff_cells(a: Grid, b: Grid) -> list[int]:
    """Flat indices where two grids differ, in raster order."""
    return [i for i, (x, y) in enumerate(zip(a.cells, b.cells)) if x != y]
