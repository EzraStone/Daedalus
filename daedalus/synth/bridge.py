"""Geometry for carrying one dust net over another.

The planar router works on ``y=1``.  A safe crossing climbs twice, travels over
the underpassing wire on ``y=3``, then descends twice.  The opaque block above
the lower wire roofs it, but does not interrupt its flat connections.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import vocab as V

Cell = tuple[int, int]
Voxel = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class BridgePlan:
    """A seven-column, two-level crossing centred on one ground cell."""

    crossing: Cell
    axis: str

    def __post_init__(self) -> None:
        if self.axis not in {"x", "z"}:
            raise ValueError("bridge axis must be 'x' or 'z'")

    @property
    def delta(self) -> Cell:
        return (1, 0) if self.axis == "x" else (0, 1)

    def cell(self, offset: int) -> Cell:
        dx, dz = self.delta
        return self.crossing[0] + offset * dx, self.crossing[1] + offset * dz

    @property
    def entry(self) -> Cell:
        return self.cell(-3)

    @property
    def exit(self) -> Cell:
        return self.cell(3)

    @property
    def dust(self) -> tuple[Voxel, ...]:
        heights = (1, 2, 3, 3, 3, 2, 1)
        out = []
        for offset, y in zip(range(-3, 4), heights):
            x, z = self.cell(offset)
            out.append((x, y, z))
        return tuple(out)

    @property
    def supports(self) -> tuple[Voxel, ...]:
        return tuple((x, y - 1, z) for x, y, z in self.dust[1:-1])

    @property
    def footprint(self) -> frozenset[Cell]:
        return frozenset(self.cell(offset) for offset in range(-3, 4))

    @property
    def in_bounds(self) -> bool:
        return all(V.in_bounds(x, y, z) for x, y, z in (*self.dust, *self.supports))
