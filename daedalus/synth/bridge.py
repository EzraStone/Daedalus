"""Geometry for carrying one dust net over another.

The planar router works on ``y=1``.  A safe crossing climbs twice, travels over
the underpassing wire on ``y=3``, then descends twice.  The opaque block above
the lower wire roofs it, but does not interrupt its flat connections.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import vocab as V
from ..grid import Grid

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
    def wire_hops(self) -> int:
        """Signal-strength cost from one ground endpoint to the other."""
        return len(self.dust) - 1

    def other_end(self, endpoint: Cell) -> Cell:
        """Return the endpoint across the elevated span."""
        if endpoint == self.entry:
            return self.exit
        if endpoint == self.exit:
            return self.entry
        raise ValueError(f"{endpoint} is not a bridge endpoint")

    def obstructions(self, grid: Grid) -> tuple[Voxel, ...]:
        """Occupied voxels that prevent this bridge from being stamped.

        Entry and exit may already contain wire because the planar search can
        route to them before it commits the bridge.  Every elevated voxel and
        ramp support must still be empty.
        """
        endpoints = {self.dust[0], self.dust[-1]}
        blocked = []
        for voxel in (*self.supports, *self.dust):
            token = grid.get(*voxel)
            if token != V.AIR and not (voxel in endpoints and token == V.WIRE):
                blocked.append(voxel)
        return tuple(blocked)

    def place(self, grid: Grid) -> None:
        """Stamp the bridge into ``grid`` without overwriting material."""
        if not self.in_bounds:
            raise ValueError("bridge extends outside the build volume")
        blocked = self.obstructions(grid)
        if blocked:
            raise ValueError(f"bridge is obstructed at {blocked[0]}")
        for voxel in self.supports:
            grid.set(*voxel, V.SOLID)
        for voxel in self.dust:
            grid.set(*voxel, V.WIRE)

    @property
    def in_bounds(self) -> bool:
        return all(V.in_bounds(x, y, z) for x, y, z in (*self.dust, *self.supports))


@dataclass(frozen=True, slots=True)
class BridgeRoute:
    """Two planar path segments joined by one elevated crossing."""

    plan: BridgePlan
    lead: tuple[Cell, ...]
    tail: tuple[Cell, ...]

    @property
    def wire_hops(self) -> int:
        """Total signal-strength cost of the new route."""
        planar = max(0, len(self.lead) - 1) + max(0, len(self.tail) - 1)
        return planar + self.plan.wire_hops

    @property
    def ground_cells(self) -> frozenset[Cell]:
        """Planar wire cells added before and after the bridge."""
        return frozenset((*self.lead, *self.tail))
