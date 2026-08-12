"""Unit tests for multilayer routing primitives."""

import pytest

from daedalus.synth.bridge import BridgePlan


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
