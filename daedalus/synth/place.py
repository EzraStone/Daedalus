"""Placement and routing: turn a netlist into an actual grid of blocks.

This is a small physical-design problem, and the physics is unforgiving in
three specific ways that shape the whole implementation:

**Adjacent dust is one net.** Two runs that touch have merged, and a merge is
an OR. So the router cannot simply avoid collisions the way a PCB router does;
it has to keep a one-cell moat around every net it is not part of.

**Dust only powers what it points at.** A dust cell's connections decide which
blocks it weakly powers. A run that goes *past* a gate's block does not power
it; only a run that dead-ends into it does. So every gate input is routed to as
a leaf, and the three cells around that leaf are frozen so a later net cannot
give it a second connection.

**Signal dies after fifteen blocks.** Long nets need a repeater, which needs
three collinear cells with no branch in the middle.

Nothing here tries to be optimal. It tries to be *correct often enough*, and
then every layout it produces is handed to the verifier, which is the only
thing with the authority to say whether it worked. The discard rate is a
reported health metric, not a hidden failure.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from .. import vocab as V
from ..grid import Grid
from .bridge import BridgePlan
from .library import Library, Orientation, load
from .netlist import Driver, Netlist, Sink

Cell = tuple[int, int]  # (x, z) on the logic layer

_STEPS = ((0, -1), (0, 1), (-1, 0), (1, 0))
#: Direction lookup for turning a step into a block-state facing.
_DIR_OF_STEP = {
    (0, -1): V.Dir4.NORTH,
    (0, 1): V.Dir4.SOUTH,
    (-1, 0): V.Dir4.WEST,
    (1, 0): V.Dir4.EAST,
}


class RoutingFailure(Exception):
    """A layout could not be completed. Expected, common, and counted."""

    def __init__(self, stage: str, detail: str = ""):
        super().__init__(f"{stage}: {detail}" if detail else stage)
        self.stage = stage
        self.detail = detail


def neighbours(c: Cell):
    x, z = c
    for dx, dz in _STEPS:
        nx, nz = x + dx, z + dz
        if 0 <= nx < V.SX and 0 <= nz < V.SZ:
            yield (nx, nz)


@dataclass(slots=True)
class Component:
    """A placed non-dust object, and the nets allowed to touch it."""

    kind: str
    cell: Cell
    nets: set[int] = field(default_factory=set)


@dataclass(slots=True)
class Layout:
    """The evolving floor plan."""

    grid: Grid
    #: cell -> ("dust", net_id) or ("comp", component index)
    occupied: dict[Cell, tuple[str, int]] = field(default_factory=dict)
    components: list[Component] = field(default_factory=list)
    #: Cells frozen so a gate input keeps exactly one connection.
    frozen: set[Cell] = field(default_factory=set)
    #: net id -> the cells its dust occupies
    net_cells: dict[int, set[Cell]] = field(default_factory=dict)
    #: Cells holding an inserted repeater, keyed by net.
    repeaters: dict[int, list[Cell]] = field(default_factory=dict)
    #: Multilayer crossings carried by each net.
    bridges: dict[int, list[BridgePlan]] = field(default_factory=dict)

    # -- placement helpers -------------------------------------------------

    def add_component(self, kind: str, cell: Cell, token: int, nets: set[int]) -> int:
        idx = len(self.components)
        self.components.append(Component(kind, cell, set(nets)))
        self.occupied[cell] = ("comp", idx)
        self.grid.set(cell[0], V.LOGIC_Y, cell[1], token)
        return idx

    def component_at(self, cell: Cell) -> Component | None:
        entry = self.occupied.get(cell)
        if entry and entry[0] == "comp":
            return self.components[entry[1]]
        return None

    def net_at(self, cell: Cell) -> int | None:
        entry = self.occupied.get(cell)
        if entry and entry[0] == "dust":
            return entry[1]
        return None

    def is_free(self, cell: Cell) -> bool:
        return cell not in self.occupied

    def can_hold_dust(self, cell: Cell, net: int) -> bool:
        """May ``net`` place dust here without changing anything else's behaviour?"""
        if cell in self.frozen or cell in self.occupied:
            return False
        for n in neighbours(cell):
            other = self.net_at(n)
            if other is not None and other != net:
                return False  # would merge two nets
            comp = self.component_at(n)
            if comp is not None and net not in comp.nets:
                return False  # would power, or be powered by, a stranger
        return True

    def place_dust(self, cell: Cell, net: int) -> None:
        self.occupied[cell] = ("dust", net)
        self.net_cells.setdefault(net, set()).add(cell)
        self.grid.set(cell[0], V.LOGIC_Y, cell[1], V.WIRE)

    def freeze_around(self, cell: Cell, keep: Cell | None) -> None:
        """Stop anything else attaching to ``cell``, so it stays a dead end."""
        for n in neighbours(cell):
            if n != keep and self.is_free(n):
                self.frozen.add(n)

    def can_place_bridge(self, plan: BridgePlan, net: int) -> bool:
        """Whether ``net`` can safely cross the wire beneath ``plan``."""
        if not plan.in_bounds or plan.obstructions(self.grid):
            return False

        under = self.net_at(plan.crossing)
        if under is None or under == net:
            return False

        # The lower net must pass straight across the bridge, perpendicular
        # to its axis.  A bend or junction underneath would be roofed at its
        # decision point and is too subtle to accept without simulation.
        perpendicular = ((0, -1), (0, 1)) if plan.axis == "x" else ((-1, 0), (1, 0))
        if any(self.net_at(_add(plan.crossing, step)) != under for step in perpendicular):
            return False

        for endpoint in (plan.entry, plan.exit):
            if self.net_at(endpoint) != net and not self.can_hold_dust(endpoint, net):
                return False

        # Ramp blocks occupy y=1.  The cells under the high span stay empty
        # so no later route can create a slope into the crossing structure.
        for offset in (-2, -1, 1, 2):
            cell = plan.cell(offset)
            if not self.is_free(cell) or cell in self.frozen:
                return False
        return True


# --------------------------------------------------------------------------
# the synthesiser
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Stats:
    """Why a batch of layouts failed, so the discard rate is diagnosable
    rather than just a number."""

    attempts: int = 0
    placed: int = 0
    routed: int = 0
    failures: dict[str, int] = field(default_factory=dict)

    def note(self, stage: str) -> None:
        self.failures[stage] = self.failures.get(stage, 0) + 1

    def as_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "placed": self.placed,
            "routed": self.routed,
            "failures": dict(sorted(self.failures.items())),
        }


class Synthesiser:
    """Places and routes one netlist onto one grid."""

    def __init__(
        self,
        netlist: Netlist,
        placed_spec,
        rng: random.Random,
        library: Library | None = None,
    ):
        self.net = netlist
        self.spec = placed_spec
        self.rng = rng
        self.lib = library or load()
        self.layout = Layout(grid=Grid.with_substrate())
        #: inverter index -> (anchor cell, orientation)
        self.gate_at: dict[int, tuple[Cell, Orientation]] = {}
        #: driver -> the component index that realises it
        self.driver_comp: dict[Driver, int] = {}
        #: output index -> the cell feeding its repeater
        self.output_terminal: dict[int, Cell] = {}
        #: net index -> the dust cells routed for it so far
        self.trees: dict[int, set[Cell]] = {}
        #: net index -> the cells that must actually see the signal
        self.net_terminals: dict[int, list[Cell]] = {}
        self._depths: dict[int, int] | None = None

    # -- top level ---------------------------------------------------------

    def run(self) -> Grid:
        """Place and route, interleaved and with backtracking.

        Placing every gate first and routing afterwards does not work: the
        placer has no idea which cells the router will need, so it cheerfully
        parks a gate on top of another gate's only head-on approach. Doing one
        gate at a time in topological order — place it, immediately route the
        net that feeds it, and undo both if the routing fails — means every
        placement decision is made against the space that is actually left.
        """
        self._place_ports()
        for g in self._topological_order():
            self._place_and_route(g)
        self._route_outputs()
        # Signal strength is a whole-net property, so it is settled once every
        # net has its final shape.
        for n in range(len(self.net.nets)):
            if n in self.trees:
                self._repeat_if_too_long(n, self.trees[n])
        return self.layout.grid

    #: How many candidate sites to actually try routing before giving up on a
    #: gate. Trial routing is the expensive part; past a handful of sites the
    #: problem is usually the previous gate's position, not this one's.
    SITE_TRIALS = 10

    def _place_and_route(self, g: int) -> None:
        gate = self.lib.inverter
        nets_of_driver = self._nets_by_driver()
        depths = self._inverter_depths()
        max_depth = max(depths.values(), default=0)

        scored = []
        for anchor, orient in self._candidate_sites(gate):
            if not self._site_is_clear(anchor, orient):
                continue
            scored.append(
                (self._site_cost(g, anchor, orient, depths[g], max_depth), anchor, orient)
            )
        if not scored:
            raise RoutingFailure("placement", f"no room for inverter {g}")
        scored.sort(key=lambda t: t[0])

        in_net = self.net.inverter_input[g]
        last: RoutingFailure | None = None
        for _score, anchor, orient in scored[: self.SITE_TRIALS]:
            saved = self._snapshot()
            try:
                self._commit_gate(g, anchor, orient, nets_of_driver)
                self._ensure_sources(in_net)
                self._route_sink(in_net, Sink("inv", g))
            except RoutingFailure as e:
                self._restore(saved)
                last = e
                continue
            return
        raise last or RoutingFailure("placement", f"no workable site for inverter {g}")

    def _route_outputs(self) -> None:
        for j in range(len(self.net.output_net)):
            n = self.net.output_net[j]
            self._ensure_sources(n)
            self._route_sink(n, Sink("out", j))

    # -- backtracking ------------------------------------------------------

    def _snapshot(self):
        lay = self.layout
        return (
            bytearray(lay.grid.cells),
            dict(lay.occupied),
            [Component(c.kind, c.cell, set(c.nets)) for c in lay.components],
            set(lay.frozen),
            {k: set(v) for k, v in lay.net_cells.items()},
            {k: list(v) for k, v in lay.repeaters.items()},
            {k: list(v) for k, v in lay.bridges.items()},
            dict(self.gate_at),
            dict(self.driver_comp),
            {k: set(v) for k, v in self.trees.items()},
            {k: list(v) for k, v in self.net_terminals.items()},
        )

    def _restore(self, snap) -> None:
        (
            cells,
            occupied,
            components,
            frozen,
            net_cells,
            repeaters,
            bridges,
            gate_at,
            driver_comp,
            trees,
            terminals,
        ) = snap
        lay = self.layout
        lay.grid.cells[:] = cells
        lay.occupied = occupied
        lay.components = components
        lay.frozen = frozen
        lay.net_cells = net_cells
        lay.repeaters = repeaters
        lay.bridges = bridges
        self.gate_at = gate_at
        self.driver_comp = driver_comp
        self.trees = trees
        self.net_terminals = terminals

    # -- ports -------------------------------------------------------------

    def _place_ports(self) -> None:
        p = self.lib.ports
        nets_of_driver = self._nets_by_driver()

        for k, (_x, _y, z) in enumerate(self.spec.input_ports):
            support = (p.support_x, z)
            lever = (p.lever_x, z)
            if not self.layout.is_free(support) or not self.layout.is_free(lever):
                raise RoutingFailure("ports", f"input {k} collides at row {z}")
            self.layout.add_component("lever", lever, V.lever(p.lever_attach), set())
            driven = nets_of_driver.get(Driver("input", k), set())
            idx = self.layout.add_component("support", support, V.SOLID, driven)
            self.driver_comp[Driver("input", k)] = idx
            faces = [(support[0] + 1, z), (support[0], z - 1), (support[0], z + 1)]
            for net in sorted(driven):
                if not self._seed_driver(faces, net):
                    raise RoutingFailure("ports", f"input {k} cannot give net {net} a face")

        for j, (_x, _y, z) in enumerate(self.spec.output_ports):
            rep = (p.repeater_x, z)
            lamp = (p.lamp_x, z)
            if not self.layout.is_free(rep) or not self.layout.is_free(lamp):
                raise RoutingFailure("ports", f"output {j} collides at row {z}")
            net = self.net.output_net[j]
            self.layout.add_component(
                "repeater", rep, V.repeater(p.output_facing, 1), {net}
            )
            self.layout.add_component("lamp", lamp, V.LAMP, set())
            # The repeater reads the cell behind it; that is where the net has
            # to arrive. A repeater takes input along its own axis, so this
            # terminal needs no head-on treatment.
            self.output_terminal[j] = (p.repeater_x - 1, z)

    def _seed_driver(self, faces, net: int) -> bool:
        """Put a dust cell on one of ``faces`` for ``net``, right now.

        Claiming a face with a reservation and routing to it later does not
        work: another net can occupy the cells just outside the claim and
        poison every approach, leaving a face that is free but unreachable.
        Real dust needs no special rule — the moat that keeps nets apart
        already keeps all four of its neighbours clear for its own use, so a
        seeded face can never be walled in.

        A driver's faces are mutually non-adjacent, so seeding one per net
        keeps the nets separate by construction. That is also exactly why the
        fanout budget is three.
        """
        usable = [f for f in faces if _inside(f) and self.layout.can_hold_dust(f, net)]
        if not usable:
            return False
        # Prefer the face with the most room to grow.
        usable.sort(
            key=lambda c: -sum(1 for n in neighbours(c) if self.layout.can_hold_dust(n, net))
        )
        seed = usable[0]
        self.layout.place_dust(seed, net)
        self.trees.setdefault(net, set()).add(seed)
        return True

    def _nets_by_driver(self) -> dict[Driver, set[int]]:
        out: dict[Driver, set[int]] = {}
        for i, net in enumerate(self.net.nets):
            for d in net.drivers:
                out.setdefault(d, set()).add(i)
        return out

    # -- gate placement ----------------------------------------------------

    def _candidate_sites(self, gate):
        """Every anchor/orientation pair, shuffled.

        Shuffling matters even though the sites are scored afterwards: ties are
        common, and a deterministic tie-break would put every circuit of a
        given shape in exactly the same place. The corpus would then teach the
        model the placer's tie-break rule instead of the physics.
        """
        sites = [
            ((x, z), orient)
            for orient in gate.orientations
            for x in range(2, V.SX - 2)
            for z in range(V.SZ)
        ]
        self.rng.shuffle(sites)
        return sites

    def _site_is_clear(self, anchor: Cell, orient: Orientation) -> bool:
        """A gate needs its own two cells and a clear halo.

        The halo is not politeness. A torch weakly powers every adjacent block,
        and a dust cell adjacent to the gate's block powers it — so anything
        that lands next to a gate is part of that gate's behaviour whether it
        meant to be or not.
        """
        cells = [_add(anchor, c) for c in orient.cells()]
        for c in cells:
            if not _inside(c) or not self.layout.is_free(c) or c in self.layout.frozen:
                return False
        occupied = set(cells)
        for c in cells:
            for n in neighbours(c):
                if n in occupied:
                    continue
                if not self.layout.is_free(n):
                    return False
        # Reserve the router's minimum needs at placement time. A gate needs at
        # least one input face whose head-on approach cell is also free, and at
        # least one free output face; a site that has neither is a dead end the
        # router would discover only after doing all the work.
        block = _add(anchor, orient.block)
        has_input = any(
            _inside(face)
            and _inside(approach)
            and self.layout.is_free(face)
            and self.layout.is_free(approach)
            and face not in self.layout.frozen
            and approach not in self.layout.frozen
            for face, approach in (
                (f, (2 * f[0] - block[0], 2 * f[1] - block[1]))
                for f in (_add(anchor, o) for o in orient.input_faces)
            )
        )
        if not has_input:
            return False
        return any(
            _inside(f) and self.layout.is_free(f)
            for f in (_add(anchor, o) for o in orient.output_faces)
        )

    def _site_cost(
        self, g: int, anchor: Cell, orient: Orientation, depth: int, max_depth: int
    ) -> float:
        """Prefer sites that keep the signal flowing left to right and sit
        close to whatever drives them."""
        block = _add(anchor, orient.block)
        span = max(max_depth, 1)
        # Spread the depth levels across the usable columns.
        target_x = 3 + (V.SX - 7) * depth / (span + 1)
        cost = 2.0 * abs(block[0] - target_x)

        in_net = self.net.nets[self.net.inverter_input[g]]
        for d in in_net.drivers:
            src = self._driver_cell(d)
            if src is not None:
                cost += 0.5 * (abs(block[0] - src[0]) + abs(block[1] - src[1]))
        # Sinks that are already pinned (the output ports) pull too.
        for sink in self._sinks_of_driver(Driver("inv", g)):
            if sink.kind == "out":
                _x, _y, z = self.spec.output_ports[sink.idx]
                cost += 0.5 * (abs(block[0] - self.lib.ports.repeater_x) + abs(block[1] - z))
        return cost + self.rng.random() * 0.25

    def _commit_gate(self, g: int, anchor: Cell, orient: Orientation, nets_of_driver) -> None:
        block = _add(anchor, orient.block)
        torch = _add(anchor, orient.torch)
        in_net = self.net.inverter_input[g]
        self.layout.add_component("block", block, V.SOLID, {in_net})
        driven = nets_of_driver.get(Driver("inv", g), set())
        idx = self.layout.add_component("torch", torch, V.torch(orient.attach), driven)
        self.driver_comp[Driver("inv", g)] = idx
        self.gate_at[g] = (anchor, orient)
        faces = [_add(anchor, f) for f in orient.output_faces]
        for net in sorted(driven):
            if not self._seed_driver(faces, net):
                raise RoutingFailure(
                    "placement", f"inverter {g} cannot give net {net} an output face"
                )

    # -- graph helpers -----------------------------------------------------

    def _inverter_depths(self) -> dict[int, int]:
        if self._depths is not None:
            return self._depths
        depth: dict[int, int] = {}

        def net_depth(n: int, seen: frozenset[int]) -> int:
            best = 0
            for d in self.net.nets[n].drivers:
                if d.kind == "inv":
                    best = max(best, 1 + inv_depth(d.idx, seen))
            return best

        def inv_depth(g: int, seen: frozenset[int]) -> int:
            if g in depth:
                return depth[g]
            if g in seen:  # pragma: no cover - netlists are acyclic
                raise RoutingFailure("placement", "cyclic netlist")
            d = net_depth(self.net.inverter_input[g], seen | {g})
            depth[g] = d
            return d

        for g in range(self.net.n_inverters):
            inv_depth(g, frozenset())
        self._depths = depth
        return depth

    def _topological_order(self) -> list[int]:
        depths = self._inverter_depths()
        return sorted(range(self.net.n_inverters), key=lambda g: (depths[g], g))

    def _sinks_of_driver(self, d: Driver) -> list[Sink]:
        out: list[Sink] = []
        for net in self.net.nets:
            if d in net.drivers:
                out.extend(net.sinks)
        return out

    def _driver_cell(self, d: Driver) -> Cell | None:
        idx = self.driver_comp.get(d)
        return self.layout.components[idx].cell if idx is not None else None

    # -- routing -----------------------------------------------------------

    def _components(self, cells: set[Cell]) -> list[set[Cell]]:
        """Split a cell set into connected regions."""
        remaining = set(cells)
        out = []
        while remaining:
            seed = next(iter(remaining))
            comp = {seed}
            queue = deque([seed])
            while queue:
                cur = queue.popleft()
                for nb in neighbours(cur):
                    if nb in remaining and nb not in comp:
                        comp.add(nb)
                        queue.append(nb)
            remaining -= comp
            out.append(comp)
        return out

    def _ensure_sources(self, n: int) -> set[Cell]:
        """Join a net's seeded driver faces into one connected region.

        Every driver has to reach every sink, so a net left in two pieces is an
        OR with a missing term.
        """
        tree = self.trees.setdefault(n, set())
        missing = [d for d, faces in self._source_options(n) if not any(c in tree for c in faces)]
        if missing:
            raise RoutingFailure("routing", f"net {n}: driver {missing[0]} was never seeded")

        while True:
            comps = self._components(tree)
            if len(comps) <= 1:
                return tree
            comps.sort(key=len, reverse=True)
            head = comps[0]
            for other in comps[1:]:
                path = self._search(head, other, n)
                if path is not None:
                    self._commit_path(path, n, tree)
                    break
            else:
                raise RoutingFailure("routing", f"net {n}: cannot join {len(comps)} fragments")

    def _route_sink(self, n: int, sink: Sink) -> None:
        self._route_to_sink(n, sink, self.trees.setdefault(n, set()))

    def _source_options(self, n: int) -> list[tuple[Driver, list[Cell]]]:
        out = []
        for d in self.net.nets[n].drivers:
            idx = self.driver_comp[d]
            comp = self.layout.components[idx]
            if d.kind == "input":
                faces = [c for c in neighbours(comp.cell) if c[0] >= comp.cell[0]]
            else:
                g = d.idx
                anchor, orient = self.gate_at[g]
                faces = [_add(anchor, f) for f in orient.output_faces if _inside(_add(anchor, f))]
            out.append((d, faces))
        return out

    def _seed_candidates(self, options: list[Cell], tree: set[Cell], net: int) -> list[Cell]:
        """Driver faces this net could start from, best first.

        The tie-break is *elbow room*: how many of a face's own neighbours the
        net could still build on. A face with nowhere to go is a dead end even
        when it is the closest one, and the router has no way back out of it.
        """
        usable = [c for c in options if c in tree or self.layout.can_hold_dust(c, net)]

        def room(c: Cell) -> int:
            return sum(1 for n in neighbours(c) if self.layout.can_hold_dust(n, net))

        if tree:
            usable.sort(
                key=lambda c: (c not in tree, min(_manhattan(c, t) for t in tree), -room(c))
            )
        else:
            usable.sort(key=lambda c: (-room(c), c))
        return usable

    def _route_to_sink(self, n: int, sink: Sink, tree: set[Cell]) -> None:
        if sink.kind == "out":
            target = self.output_terminal[sink.idx]
            if target in tree:
                self.net_terminals.setdefault(n, []).append(target)
                return
            if not self.layout.can_hold_dust(target, n):
                raise RoutingFailure("routing", f"net {n}: output {sink.idx} terminal is blocked")
            path = self._search(tree, {target}, n)
            if path is None:
                raise RoutingFailure("routing", f"net {n}: cannot reach output {sink.idx}")
            self._commit_path(path, n, tree)
            self.net_terminals.setdefault(n, []).append(target)
            return

        # A gate input has to arrive *head-on*, and that is a stronger
        # condition than "don't collide".
        #
        # A dust cell weakly powers only the blocks it points at. With two or
        # more connections it points along them, and the gate's block is solid
        # so it can never be one of them. With exactly one connection the cell
        # renders as a straight line and points both ways along that axis — so
        # it powers the block only when its single connection is on the
        # directly opposite side. Every other arrival leaves the gate dark
        # while looking perfectly plausible on screen.
        #
        # That makes the approach cell unique: ``2 * face - block``.
        anchor, orient = self.gate_at[sink.idx]
        block = _add(anchor, orient.block)
        options = []
        for offset in orient.input_faces:
            face = _add(anchor, offset)
            approach = (2 * face[0] - block[0], 2 * face[1] - block[1])
            if not (_inside(face) and _inside(approach)):
                continue
            if not self.layout.can_hold_dust(face, n):
                continue
            # Nothing else on this net may already flank the face, or the
            # single-connection property is already lost.
            if any(c in tree for c in neighbours(face) if c != approach):
                continue
            if approach not in tree and not self.layout.can_hold_dust(approach, n):
                continue
            options.append((face, approach))

        if not options:
            raise RoutingFailure(
                "routing", f"net {n}: inverter {sink.idx} has no head-on input face"
            )

        for face, approach in sorted(
            options, key=lambda fa: min(_manhattan(fa[1], t) for t in tree)
        ):
            if approach in tree:
                path = [approach, face]
            else:
                lead = self._search(
                    tree, {approach}, n, forbid=lambda c, f=face: _adjacent(c, f)
                )
                if lead is None:
                    continue
                path = lead + [face]
            self._commit_path(path, n, tree)
            self.layout.freeze_around(face, keep=approach)
            self.net_terminals.setdefault(n, []).append(face)
            return
        raise RoutingFailure("routing", f"net {n}: cannot reach inverter {sink.idx}")

    def _search(
        self,
        start: set[Cell],
        goal: set[Cell],
        net: int,
        forbid=None,
    ) -> list[Cell] | None:
        """Lee maze search from an existing tree to any goal cell.

        Breadth-first, so the first path found is a shortest one — which is
        also what keeps the signal alive, since every cell of detour costs a
        strength level.

        ``forbid`` marks cells the path may not pass *through*. It is not
        applied to goal cells: the caller uses it to keep a route clear of a
        gate face it is simultaneously trying to reach.
        """
        if not start:
            return None
        import heapq

        prev: dict[Cell, Cell | None] = {c: None for c in start}
        cost: dict[Cell, float] = {c: 0.0 for c in start}
        heap: list[tuple[float, int, Cell]] = []
        tick = 0
        for c in start:
            heapq.heappush(heap, (0.0, tick, c))
            tick += 1
        while heap:
            _c, _t, cur = heapq.heappop(heap)
            if cur in goal and cur not in start:
                path = [cur]
                while (p := prev[path[-1]]) is not None:
                    path.append(p)
                return list(reversed(path))
            for nb in neighbours(cur):
                if nb not in goal:
                    if not self.layout.can_hold_dust(nb, net):
                        continue
                    if forbid is not None and forbid(nb):
                        continue
                # Length still dominates -- every extra cell costs a signal
                # level -- but squeezing through a gap costs a little extra, so
                # the router leaves the narrow lanes free for whoever needs
                # them. Hugging obstacles is what turned one long net into a
                # wall across the board.
                blocked = sum(1 for n in neighbours(nb) if not self.layout.is_free(n))
                step = cost[cur] + 1.0 + 0.35 * blocked
                if step < cost.get(nb, float("inf")):
                    cost[nb] = step
                    prev[nb] = cur
                    heapq.heappush(heap, (step, tick, nb))
                    tick += 1
        return None

    def _commit_path(self, path: list[Cell], net: int, tree: set[Cell]) -> None:
        for cell in path:
            if cell in tree:
                continue
            if self.layout.net_at(cell) == net:
                tree.add(cell)
                continue
            if not self.layout.can_hold_dust(cell, net):
                raise RoutingFailure("routing", f"net {net}: cell {cell} became unusable")
            self.layout.place_dust(cell, net)
            tree.add(cell)

    # -- signal strength ---------------------------------------------------

    def _propagate(self, net: int, starts: set[Cell], limit: int) -> dict[Cell, int]:
        """Hops from one driver's faces, obeying repeaters.

        This is a 0-1 BFS: a dust step costs one level, and passing through a
        repeater resets to zero because the repeater re-emits at full strength.
        A repeater only conducts from its rear to its front, which is exactly
        why an inserted repeater can break a net for a *different* driver — see
        the safety check in :meth:`_insert_repeater`.
        """
        dust = self.trees.get(net, set())
        reps = {c: f for c, f in self.layout.repeaters.get(net, ())}
        dist: dict[Cell, int] = {}
        queue: deque[Cell] = deque()
        for s in starts:
            if s in dust:
                dist[s] = 0
                queue.append(s)
        while queue:
            cur = queue.popleft()
            k = dist[cur]
            if k > limit:
                continue
            for nb in neighbours(cur):
                if nb in dust:
                    if k + 1 <= limit and dist.get(nb, 1 << 30) > k + 1:
                        dist[nb] = k + 1
                        queue.append(nb)
                elif nb in reps:
                    dx, dz = reps[nb].delta
                    if (nb[0] - dx, nb[1] - dz) != cur:
                        continue  # arriving at a side or the front: blocked
                    front = (nb[0] + dx, nb[1] + dz)
                    if front in dust and dist.get(front, 1 << 30) > 0:
                        dist[front] = 0
                        queue.appendleft(front)
        return dist

    def _driver_starts(self, net: int) -> list[tuple[Driver, set[Cell]]]:
        tree = self.trees.get(net, set())
        return [
            (d, {c for c in faces if c in tree}) for d, faces in self._source_options(net)
        ]

    def _unreachable(self, net: int, limit: int) -> list[tuple[set[Cell], Cell]]:
        """Terminals a driver cannot light, as ``(driver faces, terminal)`` pairs.

        Checked **per driver**, not from all drivers pooled. A net is an OR:
        it has to work when any single driver is hot, so a terminal fifteen
        cells from one driver is dead even if another driver sits next to it.
        Pooling the sources hides exactly that bug, and it produces layouts
        that look right and fail one truth-table row.
        """
        bad = []
        for _driver, starts in self._driver_starts(net):
            if not starts:
                continue
            dist = self._propagate(net, starts, limit)
            for t in self.net_terminals.get(net, ()):
                if t not in dist:
                    bad.append((starts, t))
        return bad

    def _repeat_if_too_long(self, net: int, tree: set[Cell]) -> None:
        """Insert repeaters until every driver can light every terminal.

        A driver's face sits at 15 and each cell costs one level, so a terminal
        fifteen cells out is dark. A repeater restores it to 15, which is what
        the redstone tick buys.
        """
        limit = self.lib.max_dust_run - 1
        for _ in range(4):
            bad = self._unreachable(net, limit)
            if not bad:
                return
            if not self._insert_repeater(net, tree, bad, limit):
                raise RoutingFailure(
                    "signal",
                    f"net {net}: {len(bad)} terminal(s) out of reach and no safe repeater site",
                )
        raise RoutingFailure("signal", f"net {net} still too long after four repeaters")

    def _split(self, tree: set[Cell], cut: Cell, seed: Cell) -> set[Cell]:
        """The part of the tree still reachable from ``seed`` once ``cut`` goes."""
        seen = {seed}
        queue = deque([seed])
        while queue:
            cur = queue.popleft()
            for nb in neighbours(cur):
                if nb in tree and nb != cut and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        return seen

    def _insert_repeater(self, net: int, tree: set[Cell], bad, limit: int) -> bool:
        """Turn one straight, unbranched dust cell into a repeater.

        Two conditions, and the second is the one that is easy to miss:

        * the cell must be collinear with exactly two dust neighbours — a
          repeater dropped in a junction would cut off the other branches;
        * cutting there must leave **every** driver on the upstream side. A
          repeater conducts one way only, so a driver stranded downstream would
          be silently disconnected from the sinks it is supposed to feed.
        """
        driver_cells: set[Cell] = set()
        for _driver, starts in self._driver_starts(net):
            driver_cells |= starts
        needy = {t for _starts, t in bad}

        candidates = []
        for cell in tree:
            links = [n for n in neighbours(cell) if n in tree]
            if len(links) != 2:
                continue
            (ax, az), (bx, bz) = links
            if ax != bx and az != bz:
                continue  # a corner, not a straight run
            if cell in driver_cells or cell in needy:
                continue
            if any(
                self.layout.component_at(n) is not None
                for n in neighbours(cell)
                if n not in links
            ):
                continue

            side_a = self._split(tree, cell, links[0])
            side_b = self._split(tree, cell, links[1])
            if side_a & side_b:
                continue  # cutting here does not actually split the net
            if driver_cells <= side_a and needy & side_b:
                upstream, downstream = links[0], side_b
            elif driver_cells <= side_b and needy & side_a:
                upstream, downstream = links[1], side_a
            else:
                continue

            # Prefer a cut that puts the repeater as far downstream as it can
            # go while still being reachable, so one repeater buys the most.
            reach = self._propagate(net, driver_cells, limit)
            if cell not in reach and upstream not in reach:
                continue
            candidates.append((-len(downstream), reach.get(upstream, 0), cell, upstream))

        if not candidates:
            return False
        candidates.sort(key=lambda t: (t[0], -t[1]))
        _, _, cell, upstream = candidates[0]
        facing = _DIR_OF_STEP[(cell[0] - upstream[0], cell[1] - upstream[1])]
        self.layout.grid.set(cell[0], V.LOGIC_Y, cell[1], V.repeater(facing, 1))
        self.layout.occupied[cell] = ("comp", len(self.layout.components))
        self.layout.components.append(Component("repeater", cell, {net}))
        self.layout.net_cells[net].discard(cell)
        tree.discard(cell)
        self.layout.repeaters.setdefault(net, []).append((cell, facing))
        return True


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _add(a: Cell, b: Cell) -> Cell:
    return (a[0] + b[0], a[1] + b[1])


def _inside(c: Cell) -> bool:
    return 0 <= c[0] < V.SX and 0 <= c[1] < V.SZ


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _adjacent(a: Cell, b: Cell) -> bool:
    return _manhattan(a, b) == 1


def synthesise(netlist: Netlist, placed_spec, rng: random.Random) -> Grid:
    """Place and route once. Raises :class:`RoutingFailure` if it does not fit."""
    return Synthesiser(netlist, placed_spec, rng).run()
