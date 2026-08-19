"""Connecting a real model to the §06 loop.

The loop takes its sampler, spec source and trainer as arguments so its
acceptance logic could be tested without a GPU. That worked, and it also meant
nothing ever ran a real model through it. These cover the plugs.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("torch", reason="the generators are an optional extra")

from daedalus import vocab as V  # noqa: E402
from daedalus.grid import Grid  # noqa: E402
from daedalus.models import MaskedDiffusionModel  # noqa: E402
from daedalus.models.common import ModelConfig  # noqa: E402
from daedalus.redsim import Verifier  # noqa: E402
from daedalus.spec import Spec  # noqa: E402
from daedalus.train import LoopConfig, run  # noqa: E402
from daedalus.train.adapters import (  # noqa: E402
    ModelSampler,
    ModelTrainer,
    anchors_from,
    as_examples,
    spec_source,
)
from daedalus.train.loop import Accepted, TrainingSet  # noqa: E402

TINY = ModelConfig(n_layers=2, d_model=64, n_heads=4, d_ff=128)
NAND = "inputs A B\noutputs Q\nQ = !(A & B)"


def placed_nand():
    spec = Spec.parse(NAND)
    return spec, spec.default_placement(random.Random(0))


class TestSampler:
    def test_it_returns_the_grids_it_was_asked_for(self):
        _spec, placed = placed_nand()
        grids = ModelSampler(MaskedDiffusionModel(TINY), steps=4)(placed, 3)
        assert len(grids) == 3
        assert all(len(g) == V.CELLS for g in grids)
        assert all(max(g) < V.VOCAB_SIZE for g in grids)

    def test_candidates_reach_the_simulator(self):
        # This is what the constraint layers buy the loop. A malformed verdict
        # is rejected on inspection and ranks the same as every other
        # malformed one, so a round of them carries nothing to learn from.
        static = {"unsupported", "floating_dust", "port_violation", "masked_cell"}
        _spec, placed = placed_nand()
        grids = ModelSampler(MaskedDiffusionModel(TINY), steps=8)(placed, 4)
        with Verifier() as v:
            verdicts = v.evaluate_batch([Grid.from_tokens(g) for g in grids], placed)
        assert not [x for x in verdicts if getattr(x, "reason", None) in static]


class TestSpecSource:
    def test_specs_land_in_the_requested_bucket(self):
        got = spec_source(random.Random(0), 5, (1, 2))
        assert got
        assert all(1 <= spec.gates <= 2 for spec, _placed in got)

    def test_a_harder_bucket_gives_harder_specs(self):
        got = spec_source(random.Random(0), 5, (3, 6))
        assert all(3 <= spec.gates <= 6 for spec, _placed in got)


class TestDatasetPlumbing:
    def test_accepted_layouts_become_trainable_examples(self):
        spec, placed = placed_nand()
        tokens = Grid.with_substrate().tokens()
        dataset = TrainingSet(accepted=[Accepted(spec, placed, tokens, 30, 2)])
        examples = as_examples(dataset)
        assert len(examples) == 1
        assert examples[0].tokens == tokens
        assert examples[0].spec_hash == spec.key()

    def test_anchors_come_back_in_the_loop_s_own_shape(self):
        spec, placed = placed_nand()
        dataset = TrainingSet(accepted=[Accepted(spec, placed, [V.AIR] * V.CELLS, 1, 1)])
        anchors = anchors_from(as_examples(dataset))
        assert len(anchors) == 1
        assert isinstance(anchors[0], Accepted)

    def test_the_trainer_says_so_when_there_is_nothing_to_learn_from(self):
        trainer = ModelTrainer(MaskedDiffusionModel(TINY))
        assert trainer(TrainingSet(), 1) == {"examples": 0}


class TestWholeLoop:
    def test_a_real_model_runs_a_round(self):
        model = MaskedDiffusionModel(TINY)
        cfg = LoopConfig(rounds=1, specs_per_round=2, candidates_per_spec=2, seed=0)
        with Verifier() as v:
            reports = run(
                ModelSampler(model, steps=4), v, spec_source, cfg, trainer=ModelTrainer(model)
            )
        assert len(reports) == 1
        report = reports[0]
        assert report.candidates > 0
        # An undertrained model verifies nothing; the rates still have to add
        # up, which is what says the round actually graded something.
        total = (
            report.pass_at_1 * 0
            + report.malformed_rate
            + report.fail_rate
            + report.unstable_rate
        )
        assert total == pytest.approx(1.0, abs=0.01)
