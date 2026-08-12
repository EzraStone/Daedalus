"""Unit tests for multilayer routing primitives."""

import pytest

from daedalus import vocab as V
from daedalus.grid import Grid
from daedalus.redsim import Pass, Verifier
from daedalus.spec import Spec
from daedalus.synth.bridge import BridgePlan
from daedalus.synth.place import Component, Layout, RoutingFailure, Synthesiser


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


def test_bridge_exposes_its_virtual_edge_and_signal_cost():
    bridge = BridgePlan((8, 8), "x")

    assert bridge.wire_hops == 6
    assert bridge.other_end(bridge.entry) == bridge.exit
    assert bridge.other_end(bridge.exit) == bridge.entry
    with pytest.raises(ValueError, match="not a bridge endpoint"):
        bridge.other_end(bridge.crossing)


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


def test_synthesiser_snapshots_preserve_bridge_state():
    synth = Synthesiser.__new__(Synthesiser)
    synth.layout = Layout(
        grid=Grid.with_substrate(),
        components=[Component("test", (1, 1))],
        bridges={0: [BridgePlan((8, 8), "x")]},
    )
    synth.gate_at = {}
    synth.driver_comp = {}
    synth.trees = {}
    synth.net_terminals = {}

    snapshot = synth._snapshot()
    synth.layout.bridges.clear()
    synth._restore(snapshot)

    assert synth.layout.bridges == {0: [BridgePlan((8, 8), "x")]}


def crossing_layout(axis="x"):
    layout = Layout(Grid.with_substrate())
    crossing = (8, 8)
    steps = ((0, -1), (0, 0), (0, 1)) if axis == "x" else ((-1, 0), (0, 0), (1, 0))
    for dx, dz in steps:
        layout.place_dust((crossing[0] + dx, crossing[1] + dz), net=1)
    return layout


@pytest.mark.parametrize("axis", ["x", "z"])
def test_layout_accepts_a_perpendicular_foreign_net_as_an_underpass(axis):
    layout = crossing_layout(axis)

    assert layout.can_place_bridge(BridgePlan((8, 8), axis), net=0)


def test_layout_rejects_a_bridge_over_its_own_net():
    layout = crossing_layout()

    assert not layout.can_place_bridge(BridgePlan((8, 8), "x"), net=1)


def test_layout_rejects_a_bend_under_the_bridge():
    layout = Layout(Grid.with_substrate())
    for cell in ((7, 8), (8, 8), (8, 9)):
        layout.place_dust(cell, net=1)

    assert not layout.can_place_bridge(BridgePlan((8, 8), "x"), net=0)


def test_layout_commits_bridge_endpoints_supports_and_reservations():
    layout = crossing_layout()
    plan = BridgePlan((8, 8), "x")

    layout.place_bridge(plan, net=0)

    assert layout.bridges == {0: [plan]}
    assert layout.net_at(plan.entry) == 0
    assert layout.net_at(plan.exit) == 0
    assert layout.component_at(plan.cell(-2)).kind == "bridge_support"
    assert layout.component_at(plan.cell(2)).kind == "bridge_support"
    assert plan.cell(-1) in layout.frozen
    assert plan.cell(1) in layout.frozen


def test_layout_reports_an_unsafe_bridge_as_a_routing_stage():
    layout = Layout(Grid.with_substrate())

    with pytest.raises(RoutingFailure, match="unsafe crossing") as raised:
        layout.place_bridge(BridgePlan((8, 8), "x"), net=0)

    assert raised.value.stage == "bridge"


def test_router_connectivity_treats_a_bridge_as_one_net():
    layout = crossing_layout()
    plan = BridgePlan((8, 8), "x")
    layout.place_bridge(plan, net=0)
    synth = Synthesiser.__new__(Synthesiser)

    planar = synth._components({plan.entry, plan.exit})
    connected = synth._components({plan.entry, plan.exit}, layout.bridges[0])

    assert len(planar) == 2
    assert connected == [{plan.entry, plan.exit}]


def test_signal_propagation_charges_for_every_elevated_wire_hop():
    layout = crossing_layout()
    plan = BridgePlan((8, 8), "x")
    layout.place_bridge(plan, net=0)
    synth = Synthesiser.__new__(Synthesiser)
    synth.layout = layout
    synth.trees = {0: {plan.entry, plan.exit}}

    assert synth._propagate(0, {plan.entry}, limit=15)[plan.exit] == plan.wire_hops
    assert plan.exit not in synth._propagate(0, {plan.entry}, limit=plan.wire_hops - 1)
