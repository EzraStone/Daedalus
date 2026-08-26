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
from daedalus.synth.netlist import MAX_FANOUT, NetlistError, compile_netlist, to_nor_form
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


class TestNetlistErrors:
    def test_a_spec_the_primitive_set_cannot_express_is_reported(self):
        # Four separate nets off one driver: a torch has only three faces.
        spec = Spec.parse(
            "inputs A B C D E\noutputs Q\n"
            "Q = (!(A|B) | !(A|C)) | (!(A|D) | !(A|E))"
        )
        with pytest.raises(NetlistError, match="three free faces|more than"):
            compile_netlist(spec)


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
