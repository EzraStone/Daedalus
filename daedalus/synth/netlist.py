"""Compile a spec into the one active primitive redstone actually has.

Redstone gives you exactly two things for free:

* a **dust merge is an OR** — two runs that touch are one net, and the net is
  hot if any driver is hot;
* a **torch is a NOT** — it inverts whatever powers the block it hangs on.

Everything combinational follows from those, so the netlist has exactly one
gate type: an inverting cell whose input is a dust net and whose output is a
torch. ``NOT`` is that cell with a one-driver net; ``NOR`` is the same cell
with several. There is no separate AND stamp, no XOR stamp, no library of
shapes to keep in step with the simulator.

The other thing that falls out of the physics is fanout. A torch powers dust
on each of its free faces, and two cells on different faces of the same torch
are not adjacent to each other — so one driver can feed up to three *separate*
nets without them merging. That is why a signal used in three places needs no
buffer, and why a signal used in four does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..spec.dsl import Binary, Delay, Expr, Not, Ref, Strength

#: A wall torch has four faces; one holds it up, so three are usable. A lever's
#: support block likewise has three free faces once the lever takes one.
MAX_FANOUT = 3


class NetlistError(ValueError):
    """The spec cannot be expressed within the v1 primitive set."""


@dataclass(frozen=True, slots=True)
class Driver:
    """Something that drives a dust net: an input port or an inverter's torch."""

    kind: str  # "input" | "inv"
    idx: int

    def __str__(self) -> str:
        return f"{self.kind}{self.idx}"


@dataclass(frozen=True, slots=True)
class Sink:
    """Something a dust net feeds: an inverter's block or an output lamp."""

    kind: str  # "inv" | "out"
    idx: int

    def __str__(self) -> str:
        return f"{self.kind}{self.idx}"


@dataclass(slots=True)
class Net:
    """One electrically connected dust region.

    Multiple drivers on one net *is* the OR. Multiple sinks is plain fanout
    within a single net and costs nothing.
    """

    drivers: tuple[Driver, ...]
    sinks: list[Sink] = field(default_factory=list)


@dataclass(slots=True)
class Netlist:
    n_inputs: int
    nets: list[Net]
    #: ``inverter_input[g]`` is the net index inverter ``g`` reads.
    inverter_input: list[int]
    #: ``output_net[j]`` is the net index output ``j`` is driven by.
    output_net: list[int]

    @property
    def n_inverters(self) -> int:
        return len(self.inverter_input)

    def depth(self) -> int:
        """Longest chain of inverters, which is the circuit's latency in
        torch delays before the output repeater is added."""
        memo: dict[int, int] = {}

        def net_depth(n: int) -> int:
            if n in memo:
                return memo[n]
            memo[n] = 0  # guard; the graph is acyclic by construction
            best = 0
            for d in self.nets[n].drivers:
                if d.kind == "inv":
                    best = max(best, 1 + net_depth(self.inverter_input[d.idx]))
            memo[n] = best
            return best

        return max((net_depth(n) for n in self.output_net), default=0)

    def fanout(self) -> dict[Driver, int]:
        counts: dict[Driver, int] = {}
        for net in self.nets:
            for d in net.drivers:
                counts[d] = counts.get(d, 0) + 1
        return counts

    def summary(self) -> str:
        return (
            f"{self.n_inputs} inputs, {self.n_inverters} inverters, "
            f"{len(self.nets)} nets, depth {self.depth()}"
        )


# --------------------------------------------------------------------------
# AST rewriting
# --------------------------------------------------------------------------


def to_nor_form(expr: Expr) -> Expr:
    """Rewrite an expression into ``Ref`` / ``Not`` / ``Or`` only.

    ``a ∧ b`` becomes ``¬(¬a ∨ ¬b)``. Double negations are collapsed as they
    appear, which is what stops the rewriting from doubling the gate count.

    XOR is the interesting case, and the obvious rewrite is the wrong one.
    ``(a ∧ ¬b) ∨ (¬a ∧ b)`` needs only four inverters, but its netlist is a
    crossbar: ``a`` feeds one branch while ``¬a`` feeds the other, and the same
    for ``b``. Two signals have to swap sides, which on a single layer means
    two wire crossings — and dust cannot cross dust. Every XOR built that way
    failed to route, deterministically, at any placement effort.

    ``(a ∨ b) ∧ ¬(a ∧ b)`` costs one more inverter and one more tick of
    latency, and lays out flat: the two inputs merge once near the ports, the
    two inverted inputs merge once more, and the results meet at the end. v2
    can revisit this with a bridging router that uses the y axis to cross.
    """
    if isinstance(expr, Ref):
        return expr
    if isinstance(expr, (Delay, Strength)):
        # Timing and strength annotations are transparent to v1 logic.
        return to_nor_form(expr.operand)
    if isinstance(expr, Not):
        return _negate(to_nor_form(expr.operand))
    if isinstance(expr, Binary):
        left = to_nor_form(expr.left)
        right = to_nor_form(expr.right)
        if expr.op == "or":
            return Binary("or", left, right)
        if expr.op == "and":
            return _negate(Binary("or", _negate(left), _negate(right)))
        # a ^ b  ==  (a | b) & !(a & b)
        #         == !( !(a | b) | !(!a | !b) )
        nor_ab = _negate(Binary("or", left, right))
        and_ab = _negate(Binary("or", _negate(left), _negate(right)))
        return _negate(Binary("or", nor_ab, and_ab))
    raise AssertionError(f"unreachable expression node {expr!r}")


def _negate(expr: Expr) -> Expr:
    """``¬expr``, collapsing ``¬¬x`` to ``x``."""
    if isinstance(expr, Not):
        return expr.operand
    return Not(expr)


# --------------------------------------------------------------------------
# compilation
# --------------------------------------------------------------------------


def compile_netlist(spec) -> Netlist:
    """Turn a :class:`daedalus.spec.Spec` into a netlist.

    Common subexpressions are shared: ``¬A`` appearing in three places is one
    torch, not three. That matters for more than tidiness — the placer has
    twelve usable columns, and a compiler that duplicated every subterm would
    run out of room on circuits that comfortably fit.
    """
    nets: list[Net] = []
    inverter_input: list[int] = []
    # Memo from a rendered expression to the driver set that realises it.
    memo: dict[str, frozenset[Driver]] = {}
    input_index = {name: k for k, name in enumerate(spec.inputs)}

    def net_for(drivers: frozenset[Driver]) -> int:
        """Reuse an existing net with exactly these drivers, or make one.

        Reuse is not just an optimisation: two sinks fed by the same driver set
        genuinely want the same physical net, and creating two would double the
        routing for no behavioural difference.
        """
        key = tuple(sorted(drivers, key=lambda d: (d.kind, d.idx)))
        for i, net in enumerate(nets):
            if net.drivers == key:
                return i
        nets.append(Net(drivers=key))
        return len(nets) - 1

    def build(expr: Expr) -> frozenset[Driver]:
        key = _expr_key(expr)
        if key in memo:
            return memo[key]

        if isinstance(expr, Ref):
            result = frozenset({Driver("input", input_index[expr.name])})
        elif isinstance(expr, Binary) and expr.op == "or":
            # A dust merge. Free: no cell, no delay.
            result = build(expr.left) | build(expr.right)
        elif isinstance(expr, Not):
            inner = build(expr.operand)
            g = len(inverter_input)
            n = net_for(inner)
            nets[n].sinks.append(Sink("inv", g))
            inverter_input.append(n)
            result = frozenset({Driver("inv", g)})
        else:  # pragma: no cover - to_nor_form removes everything else
            raise NetlistError(f"expression node {expr!r} survived NOR rewriting")

        memo[key] = result
        return result

    output_net: list[int] = []
    for j, (_name, expr) in enumerate(spec.rules):
        drivers = build(to_nor_form(expr))
        n = net_for(drivers)
        nets[n].sinks.append(Sink("out", j))
        output_net.append(n)

    netlist = Netlist(
        n_inputs=spec.n_inputs,
        nets=nets,
        inverter_input=inverter_input,
        output_net=output_net,
    )

    insert_buffers(netlist)
    over = {d: c for d, c in netlist.fanout().items() if c > MAX_FANOUT}
    if over:  # pragma: no cover - insert_buffers is supposed to make this unreachable
        raise NetlistError(
            f"driver(s) {[str(d) for d in over]} still feed more than {MAX_FANOUT} "
            "separate nets after buffering"
        )
    return netlist


def _canonical(drivers) -> tuple[Driver, ...]:
    """Drivers in the order :func:`compile_netlist` stores them."""
    return tuple(sorted(set(drivers), key=lambda d: (d.kind, d.idx)))


def insert_buffers(netlist: Netlist) -> int:
    """Give over-fanned drivers a buffer, and return how many were added.

    A torch has three usable faces, so one driver can feed three separate nets
    and no more. A spec that wants a signal in four places used to be refused
    outright -- twelve percent of random specs, and no number of retries could
    touch it, because it is a gap in the primitive set rather than bad luck.

    The primitive set can close it without gaining a primitive. Two inverters
    in series are the identity: feed the crowded driver into a torch, feed that
    torch into another, and the second torch carries the original signal on a
    fresh set of three faces. The cost is two cells and two torch delays, which
    is what a buffer costs in the game as well.

    The crowded driver keeps two of its nets plus the one feeding the buffer;
    everything else moves. If the buffer is itself over-fanned the next pass
    buffers it, and since each pass moves at least two fewer nets than the last
    it terminates.
    """
    added = 0
    while True:
        over = sorted(
            (
                (count, d)
                for d, count in netlist.fanout().items()
                if count > MAX_FANOUT
            ),
            key=lambda t: (-t[0], t[1].kind, t[1].idx),
        )
        if not over:
            return added
        _count, crowded = over[0]

        # The net that feeds the buffer. Reusing an existing single-driver net
        # matters: creating a second one would add to the very fanout being
        # reduced.
        solo = _canonical((crowded,))
        feed = next((i for i, net in enumerate(netlist.nets) if net.drivers == solo), None)
        if feed is None:
            netlist.nets.append(Net(drivers=solo))
            feed = len(netlist.nets) - 1

        first = len(netlist.inverter_input)
        netlist.nets[feed].sinks.append(Sink("inv", first))
        netlist.inverter_input.append(feed)

        middle = len(netlist.nets)
        netlist.nets.append(Net(drivers=_canonical((Driver("inv", first),))))
        second = first + 1
        netlist.nets[middle].sinks.append(Sink("inv", second))
        netlist.inverter_input.append(middle)
        buffered = Driver("inv", second)
        added += 2

        # Leave the crowded driver `feed` and two others; move the rest.
        elsewhere = [
            i
            for i, net in enumerate(netlist.nets)
            if i != feed and i != middle and crowded in net.drivers
        ]
        for i in elsewhere[MAX_FANOUT - 1 :]:
            net = netlist.nets[i]
            net.drivers = _canonical(
                d if d != crowded else buffered for d in net.drivers
            )


def _expr_key(expr: Expr) -> str:
    """A structural key for memoising. Cheaper and clearer than making every
    AST node hashable by identity, and it shares across equal-but-distinct
    subtrees, which is the whole point."""
    if isinstance(expr, Ref):
        return expr.name
    if isinstance(expr, Not):
        return f"!({_expr_key(expr.operand)})"
    if isinstance(expr, Binary):
        # OR is commutative; sorting the operand keys means `a|b` and `b|a`
        # share one net instead of building two identical ones.
        a, b = _expr_key(expr.left), _expr_key(expr.right)
        if expr.op == "or":
            a, b = sorted((a, b))
        return f"({a} {expr.op} {b})"
    if isinstance(expr, (Delay, Strength)):
        return _expr_key(expr.operand)
    raise AssertionError(f"unreachable expression node {expr!r}")
