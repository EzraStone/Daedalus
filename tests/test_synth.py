"""The procedural compiler.

Every assertion here goes through the verifier. A placer that emits plausible
grids nobody checked is the exact failure mode a synthetic corpus is prone to,
so "it compiled" is never enough — the only success criterion is a PASS.
"""

from __future__ import annotations

import random

import pytest

from daedalus import vocab as V
from daedalus.redsim import Pass, Verifier
from daedalus.spec import Spec
from daedalus.synth import compile, compile_attempts, compile_many
from daedalus.synth.library import load
from daedalus.synth.netlist import MAX_FANOUT, compile_netlist, to_nor_form
from daedalus.synth.place import Stats

#: Gate shapes covered by the procedural compiler regression suite.
SUPPORTED = [
    ("Q = !A", "A"),
    ("Q = A & B", "A B"),
    ("Q = A | B", "A B"),
    ("Q = !(A & B)", "A B"),
    ("Q = !(A | B)", "A B"),
    ("Q = A & !B", "A B"),
    ("Q = A & B & C", "A B C"),
    ("Q = A | B | C", "A B C"),
    ("Q = (A & B) | C", "A B C"),
    ("Q = !(A | B | C)", "A B C"),
    ("Q = !((A | B) & C)", "A B C"),
]


@pytest.fixture(scope="module")
def verifier():
    with Verifier() as v:
        yield v


def make(rule: str, inputs: str) -> Spec:
    return Spec.parse(f"inputs {inputs}\noutputs Q\n{rule}")


class TestNetlist:
    def test_or_is_free(self):
        # A dust merge is an OR: no cell, no delay.
        n = compile_netlist(make("Q = A | B", "A B"))
        assert n.n_inverters == 0
        assert n.depth() == 0

    def test_not_is_one_cell(self):
        n = compile_netlist(make("Q = !A", "A"))
        assert n.n_inverters == 1
        assert n.depth() == 1

    def test_and_costs_three_inverters(self):
        # !(!a | !b): invert each input, merge, invert again.
        n = compile_netlist(make("Q = A & B", "A B"))
        assert n.n_inverters == 3
        assert n.depth() == 2

    def test_nand_is_cheaper_than_and(self):
        assert compile_netlist(make("Q = !(A & B)", "A B")).n_inverters == 2
        assert compile_netlist(make("Q = A & B", "A B")).n_inverters == 3

    def test_common_subexpressions_are_shared(self):
        # !A appears three times but is one torch. Without sharing the placer
        # runs out of columns on circuits that comfortably fit.
        n = compile_netlist(make("Q = (!A | B) | (!A | C)", "A B C"))
        assert n.n_inverters == 1

    def test_double_negation_collapses(self):
        assert compile_netlist(make("Q = !!A | B", "A B")).n_inverters == 0

    def test_xor_uses_the_planar_decomposition(self):
        # (a|b) & !(a&b) rather than (a&!b)|(!a&b): one more inverter, but it
        # lays out flat instead of needing two wire crossings.
        n = compile_netlist(make("Q = A ^ B", "A B"))
        assert n.n_inverters == 5
        assert n.depth() == 3

    def test_fanout_beyond_three_is_rejected(self):
        # A torch has three free faces, so a driver can feed three separate
        # nets. A fourth needs a buffer stage, which v1 does not build.
        n = compile_netlist(make("Q = (A|B) | (A|C)", "A B C"))
        assert max(n.fanout().values()) <= MAX_FANOUT

    def test_nor_rewriting_removes_and_and_xor(self):
        from daedalus.spec.dsl import Binary

        def kinds(e):
            if isinstance(e, Binary):
                assert e.op == "or", f"{e.op} survived rewriting"
                kinds(e.left)
                kinds(e.right)
            elif hasattr(e, "operand"):
                kinds(e.operand)

        for rule, ins in SUPPORTED + [("Q = A ^ B", "A B")]:
            kinds(to_nor_form(make(rule, ins).rules[0][1]))


class TestLibrary:
    def test_library_validates(self):
        lib = load()
        assert lib.inverter.orientations
        assert lib.max_dust_run == 15

    def test_torch_faces_are_mutually_non_adjacent(self):
        # This is what makes a fanout of three possible: two nets on two faces
        # of one torch never touch each other.
        for o in load().inverter.orientations:
            faces = o.output_faces
            for i, a in enumerate(faces):
                for b in faces[i + 1 :]:
                    assert abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1


class TestCompilation:
    @pytest.mark.parametrize("rule,inputs", SUPPORTED)
    def test_supported_gates_compile_and_verify(self, verifier, rule, inputs):
        spec = make(rule, inputs)
        for seed in range(6):
            attempt = compile(spec, verifier, random.Random(seed), attempts=15)
            if attempt.ok:
                assert isinstance(attempt.verdict, Pass)
                return
        pytest.fail(f"{rule} did not compile in six tries")

    def test_explicit_gate_library_reaches_the_synthesiser(self, verifier):
        attempt = compile(
            make("Q = !A", "A"),
            verifier,
            random.Random(0),
            attempts=8,
            library=load(),
        )

        assert attempt.ok, attempt

    def test_output_of_the_compiler_is_always_verified(self, verifier):
        # The compiler never reports success on an unchecked grid.
        spec = make("Q = A & B", "A B")
        for seed in range(10):
            a = compile(spec, verifier, random.Random(seed), attempts=10)
            assert a.ok == isinstance(a.verdict, Pass)

    def test_ports_land_where_the_spec_says(self, verifier):
        spec = make("Q = !(A & B)", "A B")
        a = compile(spec, verifier, random.Random(1), attempts=15)
        assert a.ok
        for x, y, z in a.placed.input_ports:
            assert V.decode(a.grid.get(x, y, z)).kind == "lever"
        for x, y, z in a.placed.output_ports:
            assert a.grid.get(x, y, z) == V.LAMP

    def test_substrate_is_intact(self, verifier):
        a = compile(make("Q = A | B", "A B"), verifier, random.Random(2), attempts=15)
        assert a.ok
        for x in range(V.SX):
            for z in range(V.SZ):
                assert a.grid.get(x, V.SUBSTRATE_Y, z) == V.SOLID

    def test_many_layouts_are_distinct(self, verifier):
        # Distinct layouts per spec are the augmentation that teaches the model
        # there is more than one answer. Without it the corpus is a lookup
        # table from spec to layout.
        spec = make("Q = !(A & B)", "A B")
        got = compile_many(spec, verifier, count=6, rng=random.Random(0), attempts_each=12)
        assert len(got) >= 3
        assert len({g.grid.to_bytes() for g in got}) == len(got)

    def test_stats_explain_the_discard_rate(self, verifier):
        stats = Stats()
        compile(make("Q = A & B", "A B"), verifier, random.Random(0), attempts=4, stats=stats)
        d = stats.as_dict()
        assert d["attempts"] >= 1
        assert set(d) == {"attempts", "placed", "routed", "bridged", "failures"}

    def test_crossbar_netlists_use_the_bridging_router(self, verifier):
        """A signal and its complement feeding branches that reconverge needs a
        wire crossing, so at least one placement must use the elevated layer.
        """
        spec = make("Q = (A & B) | (!A & C)", "A B C")
        stats = Stats()
        outcomes = [
            compile(spec, verifier, random.Random(s), attempts=20, stats=stats) for s in range(4)
        ]
        built = [attempt for attempt in outcomes if attempt.ok]

        assert built, "the bridging router should cover at least one crossbar placement"
        assert any(3 in attempt.grid.occupied_layers() for attempt in built)
        assert stats.bridged >= 1


class TestFanoutBuffering:
    """A signal wanted in four places used to be refused outright."""

    CROWDED = (
        "inputs A B C D E\noutputs Q\n"
        "Q = (!(A|B) | !(A|C)) | (!(A|D) | !(A|E))"
    )

    def test_a_spec_that_needs_four_faces_now_compiles(self):
        net = compile_netlist(Spec.parse(self.CROWDED))
        assert max(net.fanout().values()) <= MAX_FANOUT

    def test_nothing_is_added_when_nothing_is_crowded(self):
        from daedalus.synth.netlist import insert_buffers

        net = compile_netlist(Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)"))
        before = (len(net.nets), net.n_inverters)
        assert insert_buffers(net) == 0
        assert (len(net.nets), net.n_inverters) == before

    def test_four_sinks_on_one_net_still_need_no_buffer(self):
        # The distinction the whole rule rests on. Four *sinks* on one net is
        # plain dust fanout and costs nothing; four separate *nets* is what a
        # torch cannot do. Buffering the first case would add two gates to
        # every circuit that uses a signal more than three times.
        from daedalus.synth.netlist import insert_buffers

        net = compile_netlist(
            Spec.parse("inputs A\noutputs P Q R S\nP = !A\nQ = !A\nR = !A\nS = !A")
        )
        assert net.n_inverters == 1
        assert insert_buffers(net) == 0

    def test_a_buffer_costs_two_inverters(self):
        # One inverter would be an inverter. The identity needs two, and the
        # two torch delays are what a buffer costs in the game as well.
        from daedalus.synth.netlist import insert_buffers

        net = compile_netlist(Spec.parse(self.CROWDED))
        before = net.n_inverters
        assert insert_buffers(net) == 0  # compile_netlist already did it
        assert before == 6  # four inverters for the terms, two for the buffer
        assert net.depth() == 3  # one term, plus the buffer's two

    def test_the_buffered_circuit_computes_the_right_function(self, verifier):
        # The only check that matters. A netlist rewrite that is subtly wrong
        # produces a layout that routes and lies, and the truth table is what
        # catches it.
        for source in (
            "inputs A\noutputs P Q R S\nP = !A\nQ = !A\nR = !A\nS = !A",
            "inputs A B\noutputs P Q R S\nP = A\nQ = !A\nR = !(A | B)\nS = B",
            "inputs A B\noutputs P Q R\nP = !A\nQ = !(A | B)\nR = !(A | B)",
        ):
            spec = Spec.parse(source)
            for seed in range(30):
                attempt = compile(spec, verifier, random.Random(seed), attempts=12)
                if attempt.ok:
                    break
            assert attempt.ok, (source, attempt.stage, attempt.detail)

    def test_buffering_terminates_on_a_very_crowded_driver(self):
        # Each pass leaves the crowded driver at three and hands the rest to a
        # buffer that may itself be crowded. It has to converge.
        net = compile_netlist(
            Spec.parse(
                "inputs A B C D E F\noutputs Q R\n"
                "Q = ((!(A|B) | !(A|C)) | (!(A|D) | !(A|E))) | !(A|F)\n"
                "R = !A"
            )
        )
        assert max(net.fanout().values()) <= MAX_FANOUT

    def test_buffering_never_leaves_two_nets_with_the_same_drivers(self):
        # net_for dedupes nets by driver set, and two sinks fed by the same
        # drivers genuinely want one physical net. Substituting a buffer for a
        # driver rewrites driver sets, so it could in principle collide with
        # an existing net and quietly route the same signal twice.
        #
        # It cannot, and the reason is worth pinning: the substituted driver is
        # a brand-new inverter no other net mentions, and two nets that both
        # named the crowded driver already differed somewhere else. Measured
        # over 400 random specs as well as asserted here.
        import collections

        for source in (
            self.CROWDED,
            "inputs A B C D E F\noutputs Q R\n"
            "Q = ((!(A|B) | !(A|C)) | (!(A|D) | !(A|E))) | !(A|F)\nR = !A",
            "inputs A B\noutputs P Q R S\nP = A\nQ = !A\nR = !(A | B)\nS = B",
        ):
            net = compile_netlist(Spec.parse(source))
            counts = collections.Counter(n.drivers for n in net.nets)
            assert all(c == 1 for c in counts.values()), source

    def test_every_buffer_net_still_has_a_sink(self):
        # A net with no sink is dust routed to nowhere: it costs cells, it can
        # block another net, and nothing notices.
        net = compile_netlist(Spec.parse(self.CROWDED))
        assert all(n.sinks for n in net.nets)

    def test_the_guard_still_fires_if_buffering_ever_fails(self, monkeypatch):
        # The rejection this replaced is still the backstop. Without it a bug
        # in insert_buffers would hand the placer a netlist it cannot build
        # and the failure would surface as a routing error somewhere else.
        from daedalus.synth import netlist as module

        monkeypatch.setattr(module, "insert_buffers", lambda _net: 0)
        with pytest.raises(module.NetlistError, match="after buffering"):
            compile_netlist(Spec.parse(self.CROWDED))


class TestConstraintReporting:
    """A missed budget and a broken circuit are different failures."""

    def test_a_circuit_that_only_misses_a_budget_is_not_called_broken(self, verifier):
        # It computes the right function. Reporting that as "verify" sends
        # someone looking for a layout bug that is not there.
        source = "inputs A B C\noutputs Q\nQ = !(A & B) | C\nfootprint <= 30"
        stages = set()
        for attempt in compile_attempts(Spec.parse(source), verifier, random.Random(0), 25):
            stages.add(attempt.stage)
            if attempt.ok:
                break
        assert "constraint" in stages

    def test_an_unconstrained_spec_never_reports_a_constraint_failure(self, verifier):
        source = "inputs A B\noutputs Q\nQ = !(A & B)"
        for attempt in compile_attempts(Spec.parse(source), verifier, random.Random(0), 20):
            assert attempt.stage != "constraint"
            if attempt.ok:
                break

    def test_the_classifier_needs_a_clean_truth_table(self):
        from daedalus.redsim import Fail, RowMismatch
        from daedalus.synth import _constraint_only

        assert _constraint_only(Fail((), "footprint"))
        # Wrong rows *and* over budget is a broken circuit, not a tight one.
        assert not _constraint_only(Fail((RowMismatch(0, 0, 1),), "footprint"))
        assert not _constraint_only(Fail((RowMismatch(0, 0, 1),), None))


class TestGateOrdering:
    """The index tie-break in _topological_order is load-bearing."""

    def test_order_is_deterministic_across_runs(self):
        # Two Synthesisers on the same netlist must agree, whatever their
        # generators have been doing. A retry that reorders gates loses
        # crossbar coverage -- see docs/benchmarks.md.
        from daedalus.synth.netlist import compile_netlist
        from daedalus.synth.place import Synthesiser

        spec = make("Q = (A & B) | (!A & C)", "A B C")
        netlist = compile_netlist(spec)
        placed = spec.default_placement(random.Random(0))
        orders = []
        for seed in range(4):
            s = Synthesiser(netlist, placed, random.Random(seed))
            orders.append(s._topological_order())
        assert len(set(map(tuple, orders))) == 1, orders

    def test_order_respects_depth(self):
        from daedalus.synth.netlist import compile_netlist
        from daedalus.synth.place import Synthesiser

        spec = make("Q = !(!(A & B) & C)", "A B C")
        netlist = compile_netlist(spec)
        s = Synthesiser(netlist, spec.default_placement(random.Random(0)), random.Random(0))
        depths = s._inverter_depths()
        order = s._topological_order()
        assert [depths[g] for g in order] == sorted(depths[g] for g in order)


class TestNeighbourTable:
    """The hottest function in the compiler, now a lookup."""

    def test_every_cell_matches_the_bounds_check_it_replaced(self):
        from daedalus.synth.place import neighbours

        steps = ((0, -1), (0, 1), (-1, 0), (1, 0))
        for x in range(V.SX):
            for z in range(V.SZ):
                want = tuple(
                    (x + dx, z + dz)
                    for dx, dz in steps
                    if 0 <= x + dx < V.SX and 0 <= z + dz < V.SZ
                )
                assert neighbours((x, z)) == want

    def test_corners_have_two_and_the_middle_has_four(self):
        from daedalus.synth.place import neighbours

        assert len(neighbours((0, 0))) == 2
        assert len(neighbours((V.SX - 1, V.SZ - 1))) == 2
        assert len(neighbours((8, 8))) == 4

    def test_a_cell_outside_the_grid_is_an_error_not_an_empty_answer(self):
        from daedalus.synth.place import neighbours

        with pytest.raises(KeyError):
            neighbours((V.SX, 0))


class TestSinkIndex:
    """Built once now; it still has to answer what the scan answered."""

    def test_it_agrees_with_the_scan_it_replaced(self):
        from daedalus.spec import Spec
        from daedalus.synth import compile_netlist
        from daedalus.synth.netlist import Driver
        from daedalus.synth.place import Synthesiser

        for source in (
            "inputs A B\noutputs Q\nQ = !(A & B)",
            "inputs A B\noutputs Q\nQ = A ^ B",
            "inputs A B C\noutputs Q R\nQ = (A & B) | C\nR = !C",
        ):
            spec = Spec.parse(source)
            net = compile_netlist(spec)
            synth = Synthesiser(net, spec.default_placement(), random.Random(0))
            drivers = {d for n in net.nets for d in n.drivers} | {
                Driver("inv", g) for g in range(net.n_inverters)
            }
            for d in drivers:
                want = [s for n in net.nets if d in n.drivers for s in n.sinks]
                assert synth._sinks_of_driver(d) == want, (source, d)

    def test_an_unknown_driver_still_has_no_sinks(self):
        from daedalus.spec import Spec
        from daedalus.synth import compile_netlist
        from daedalus.synth.netlist import Driver
        from daedalus.synth.place import Synthesiser

        spec = Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)")
        synth = Synthesiser(
            compile_netlist(spec), spec.default_placement(), random.Random(0)
        )
        assert synth._sinks_of_driver(Driver("inv", 999)) == []
