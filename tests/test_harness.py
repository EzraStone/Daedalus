"""The fidelity harness, minus the part that needs Minecraft.

Measuring sim↔game agreement is the first item on the next-steps list and the
number the rest of the repository is waiting on. Whoever finally runs it will
read two things: the agreement rate, and `classify`'s guess at which
divergence each disagreement implicates. The second is pure logic, needs no
server, and had no tests — so a misattributed divergence would send someone
to the wrong module with no way to notice.

The client half is not covered here. It needs a Fabric server, and a mock of
it would only assert that the mock behaves like the mock.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from compare import Case, Report, classify  # noqa: E402

from daedalus import vocab as V  # noqa: E402
from daedalus.grid import Grid  # noqa: E402
from daedalus.spec import Spec  # noqa: E402

NAND = "inputs A B\noutputs Q\nQ = !(A & B)"


def make_case(tokens, observed=None, settled=True) -> Case:
    spec = Spec.parse(NAND)
    return Case(
        name="t",
        spec=spec,
        placed=spec.default_placement(random.Random(0)),
        tokens=tokens,
        simulated=[1, 1, 1, 0],
        observed=observed,
        settled=settled,
    )


def empty_grid() -> list[int]:
    return Grid.with_substrate().tokens()


class TestAgreement:
    def test_a_case_the_game_never_answered_does_not_count_as_agreeing(self):
        # `observed is None` means the server never replied. Treating that as
        # agreement would inflate the headline number with cases nobody ran.
        assert not make_case(empty_grid()).agrees()

    def test_matching_truth_tables_agree(self):
        assert make_case(empty_grid(), observed=[1, 1, 1, 0]).agrees()

    def test_one_differing_row_is_a_disagreement(self):
        assert not make_case(empty_grid(), observed=[1, 1, 1, 1]).agrees()


class TestClassification:
    def test_an_unanswered_case_is_not_classified(self):
        assert classify(make_case(empty_grid())) == "unclassified"

    def test_an_unsettled_case_blames_update_order(self):
        case = make_case(empty_grid(), observed=[0, 0, 0, 0], settled=False)
        assert classify(case) == "update-order"

    def test_dust_on_both_sides_of_a_block_blames_weak_versus_strong(self):
        # The classic trap: whether that block conducts is exactly where a
        # naive simulator and the game part company.
        grid = Grid.with_substrate()
        grid.set(5, V.LOGIC_Y, 5, V.SOLID)
        grid.set(4, V.LOGIC_Y, 5, V.WIRE)
        grid.set(6, V.LOGIC_Y, 5, V.WIRE)
        case = make_case(grid.tokens(), observed=[0, 0, 0, 0])
        assert classify(case) == "weak-vs-strong-power"

    def test_a_torch_with_no_flanked_block_blames_burnout(self):
        grid = Grid.with_substrate()
        grid.set(5, V.LOGIC_Y, 5, V.torch(V.Attach.FLOOR))
        case = make_case(grid.tokens(), observed=[0, 0, 0, 0])
        assert classify(case) == "torch-burnout"

    def test_an_ordinary_disagreement_is_left_unclassified(self):
        # The honest default. Most genuinely new bugs land here, and guessing
        # would be worse than admitting the guess failed.
        grid = Grid.with_substrate()
        grid.set(5, V.LOGIC_Y, 5, V.WIRE)
        case = make_case(grid.tokens(), observed=[0, 0, 0, 0])
        assert classify(case) == "unclassified"

    def test_classification_survives_a_masked_cell(self):
        # A half-denoised sample can reach the harness, and decoding a control
        # token raises.
        grid = Grid.with_substrate()
        grid.set(5, V.LOGIC_Y, 5, V.MASK)
        classify(make_case(grid.tokens(), observed=[0, 0, 0, 0]))


class TestReport:
    def test_agreement_is_measured_over_cases_that_were_actually_checked(self):
        # Dividing by total cases would let an unreachable server quietly drag
        # the headline number toward zero.
        report = Report(cases=10, agreed=6, unreachable=4)
        assert report.agreement == 1.0
        assert report.as_dict()["checked"] == 6

    def test_nothing_checked_is_zero_rather_than_a_crash(self):
        assert Report(cases=3, unreachable=3).agreement == 0.0

    def test_examples_are_capped_so_the_report_stays_readable(self):
        report = Report(cases=50, examples=[{"id": str(i)} for i in range(40)])
        assert len(report.as_dict()["examples"]) == 10

    @pytest.mark.parametrize("agreed,checked,expected", [(1, 2, 0.5), (3, 4, 0.75)])
    def test_the_rate_is_what_it_says(self, agreed, checked, expected):
        report = Report(cases=checked, agreed=agreed, unreachable=0)
        assert report.agreement == expected
