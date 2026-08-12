"""Unit tests for multilayer routing primitives."""

import pytest

from daedalus import vocab as V
from daedalus.grid import Grid
from daedalus.redsim import Pass, Verifier
from daedalus.spec import Spec
from daedalus.synth.bridge import BridgePlan


@pytest.fixture(scope="module")
def verifier():
    with Verifier() as running:
        yield running


def test_bridge_climbs_two_layers_over_the_crossing():
    bridge = BridgePlan((8, 8), "x")

    assert bridge.entry == (5, 8)
    assert bridge.exit == (11, 8)
    assert [voxel[1] for voxel in bridge.dust] == [1, 2, 3, 3, 3, 2, 1]
    assert bridge.dust[3] == (8, 3, 8)
    assert bridge.supports[2] == (8, 2, 8)


def test_bridge_can_run_along_the_z_axis():
    bridge = BridgePlan((7, 6), "z")

    assert bridge.entry == (7, 3)
    assert bridge.exit == (7, 9)
    assert bridge.footprint == {(7, z) for z in range(3, 10)}


@pytest.mark.parametrize("axis", ["y", "north", ""])
def test_bridge_rejects_unknown_axes(axis):
    with pytest.raises(ValueError, match="axis"):
        BridgePlan((8, 8), axis)


def test_bridge_reports_build_volume_edges():
    assert BridgePlan((8, 8), "x").in_bounds
    assert not BridgePlan((2, 8), "x").in_bounds


def test_bridge_stamps_supported_dust_without_touching_the_underpass():
    grid = Grid.with_substrate()
    bridge = BridgePlan((8, 8), "x")
    grid.set(8, V.LOGIC_Y, 8, V.WIRE)

    bridge.place(grid)

    assert all(grid.get(*voxel) == V.WIRE for voxel in bridge.dust)
    assert all(grid.get(*voxel) == V.SOLID for voxel in bridge.supports)
    assert grid.get(8, V.LOGIC_Y, 8) == V.WIRE


def test_bridge_refuses_to_overwrite_elevated_material():
    grid = Grid.with_substrate()
    bridge = BridgePlan((8, 8), "z")
    grid.set(*bridge.dust[3], V.LAMP)

    with pytest.raises(ValueError, match="obstructed"):
        bridge.place(grid)


def test_bridge_refuses_a_partial_stamp_at_the_world_edge():
    grid = Grid.with_substrate()

    with pytest.raises(ValueError, match="outside the build volume"):
        BridgePlan((2, 8), "x").place(grid)


def test_bridge_keeps_crossing_signals_electrically_independent(verifier):
    grid = Grid.with_substrate()
    placed = Spec.parse("inputs A B\noutputs Q R\nQ = A\nR = B").place((8, 4), (8, 12))

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

    verdict = verifier.evaluate(grid, placed)

    assert isinstance(verdict, Pass), verdict
