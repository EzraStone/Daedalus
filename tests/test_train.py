"""Training: checkpoints that reload, and a metric that means something.

Both concerns here are things a run only discovers the expensive way. The
checkpoint write is the last statement of a training run, so a bug there costs
the whole run; and a progress metric that moves for the wrong reason is worse
than none, because it gets believed.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("torch", reason="training is an optional extra")

import torch  # noqa: E402

from daedalus import vocab as V  # noqa: E402
from daedalus.data.corpus import Example  # noqa: E402
from daedalus.models import AutoregressiveModel, MaskedDiffusionModel  # noqa: E402
from daedalus.models.common import ModelConfig  # noqa: E402
from daedalus.spec import Spec  # noqa: E402
from daedalus.train.pretrain import (  # noqa: E402
    TrainConfig,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train,
)

TINY = ModelConfig(n_layers=2, d_model=64, n_heads=4, d_ff=128)


def make_examples(n: int = 8) -> list[Example]:
    """Cheap stand-ins: a real corpus build is far too slow for a unit test."""
    spec = Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)")
    placed = spec.default_placement(random.Random(0))
    rng = random.Random(0)
    out = []
    for _ in range(n):
        tokens = [V.AIR] * V.CELLS
        for _ in range(20):
            tokens[V.index(rng.randrange(V.SX), V.LOGIC_Y, rng.randrange(V.SZ))] = V.WIRE
        out.append(
            Example(
                spec_source=spec.source(),
                spec_hash=spec.key(),
                gates=spec.gates,
                n_inputs=spec.n_inputs,
                n_outputs=spec.n_outputs,
                rows=list(spec.rows),
                input_z=list(placed.input_z),
                output_z=list(placed.output_z),
                tokens=tokens,
                latency_rt=1,
                blocks=20,
                bbox=[16, 1, 16],
                prompts=[],
            )
        )
    return out


class TestCheckpoints:
    @pytest.mark.parametrize("cls", [MaskedDiffusionModel, AutoregressiveModel])
    def test_a_checkpoint_reloads_into_the_same_model(self, cls, tmp_path):
        torch.manual_seed(0)
        model = cls(TINY)
        model.eval()
        save_checkpoint(model, tmp_path / "model.pt")
        back = load_checkpoint(tmp_path / "model.pt", device="cpu")

        assert type(back) is cls
        tokens = torch.randint(0, V.CONTROL_BASE, (1, V.SEQ_LEN))
        back.eval()
        with torch.no_grad():
            assert torch.allclose(model(tokens), back(tokens))

    def test_a_frozen_slots_config_still_serialises(self, tmp_path):
        # ModelConfig is a slots dataclass, so it has no __dict__ -- reading
        # one to build the blob raises, and it raises at the very end of a
        # training run where it costs the most.
        save_checkpoint(MaskedDiffusionModel(TINY), tmp_path / "m.pt")
        blob = torch.load(tmp_path / "m.pt", map_location="cpu", weights_only=False)
        assert blob["config"]["d_model"] == TINY.d_model
        assert blob["kind"] == "MaskedDiffusionModel"

    def test_an_unknown_kind_is_refused(self, tmp_path):
        torch.save({"kind": "Nonsense", "config": {}, "model": {}}, tmp_path / "m.pt")
        with pytest.raises(ValueError, match="known model"):
            load_checkpoint(tmp_path / "m.pt", device="cpu")


class TestEvaluation:
    def test_the_diffusion_training_loss_is_not_a_progress_signal(self):
        # Establishes why `evaluate` has to exist. The mask rate is drawn per
        # call and the objective is scaled by 1/t, so one unchanged model
        # reports a wide spread of losses on identical input.
        torch.manual_seed(0)
        model = MaskedDiffusionModel(TINY)
        tokens = torch.randint(0, V.CONTROL_BASE, (4, V.SEQ_LEN))
        with torch.no_grad():
            seen = [float(model.loss(tokens)) for _ in range(12)]
        assert max(seen) > 2 * min(seen), "expected the 1/t weighting to spread these"

    def test_evaluation_repeats_exactly(self):
        torch.manual_seed(0)
        model = MaskedDiffusionModel(TINY)
        examples = make_examples(4)
        assert evaluate(model, examples, seed=0) == evaluate(model, examples, seed=0)

    def test_evaluation_leaves_the_model_in_training_mode(self):
        model = MaskedDiffusionModel(TINY)
        model.train()
        evaluate(model, make_examples(2), seed=0)
        assert model.training

    def test_training_moves_the_comparable_metric_down(self):
        torch.manual_seed(0)
        model = MaskedDiffusionModel(TINY)
        examples = make_examples(16)
        before = evaluate(model, examples, seed=0)
        train(model, examples, TrainConfig(epochs=8, batch_size=4, warmup=4, log_every=100))
        assert evaluate(model, examples, seed=0) < before

    def test_history_carries_validation_when_asked(self):
        torch.manual_seed(0)
        model = MaskedDiffusionModel(TINY)
        examples = make_examples(8)
        history = train(
            model,
            examples,
            TrainConfig(epochs=2, batch_size=4, warmup=2, log_every=1),
            val=examples,
        )
        assert history and all("val_loss" in e for e in history)

    def test_evaluating_on_nothing_is_an_error(self):
        with pytest.raises(ValueError):
            evaluate(MaskedDiffusionModel(TINY), [])


class TestPromptConditioning:
    """Training on the paraphrases the corpus has always carried."""

    def prompted(self, slots=4):
        from daedalus.models.common import ModelConfig as MC

        torch.manual_seed(0)
        return MaskedDiffusionModel(
            MC(n_layers=2, d_model=64, n_heads=4, d_ff=128, nl_slots=slots)
        )

    def test_a_spec_only_model_carries_no_prompt_parameters(self):
        # The two have to be a clean comparison, not the same model with a
        # dead branch inflating its parameter count.
        assert self.prompted(slots=0).prompts is None
        assert self.prompted(slots=4).prompts is not None

    def test_the_prompt_encoder_is_trained_along_with_the_body(self):
        model = self.prompted()
        examples = make_examples(8)
        before = model.prompts.embed.weight.detach().clone()
        train(model, examples, TrainConfig(epochs=4, batch_size=4, warmup=2, log_every=100))
        assert not torch.equal(before, model.prompts.embed.weight)

    def test_a_prompted_model_still_improves(self):
        model = self.prompted()
        examples = make_examples(16)
        before = evaluate(model, examples, seed=0)
        train(model, examples, TrainConfig(epochs=8, batch_size=4, warmup=4, log_every=100))
        assert evaluate(model, examples, seed=0) < before

    def test_examples_without_prompts_encode_to_padding(self):
        # The corpus can carry examples with no paraphrases, and those must
        # read as "no condition" rather than as some arbitrary one.
        from daedalus.train.pretrain import to_prompt_features

        examples = make_examples(2)
        for e in examples:
            e.prompts = []
        assert all(set(f) == {0} for f in to_prompt_features(examples, random.Random(0)))

    def test_a_prompt_is_drawn_from_the_example_s_own_paraphrases(self):
        from daedalus.text import encode_prompt
        from daedalus.train.pretrain import to_prompt_features

        examples = make_examples(1)
        examples[0].prompts = ["turn the lamp on", "light it up"]
        got = to_prompt_features(examples, random.Random(0))[0]
        assert got in (encode_prompt("turn the lamp on"), encode_prompt("light it up"))

    def test_evaluation_conditions_on_the_prompt_too(self):
        # Scoring a prompt-trained model without its prompt compares a model
        # that was given the question against one that was not.
        model = self.prompted()
        examples = make_examples(4)
        with torch.no_grad():
            seen = evaluate(model, examples, seed=0)
        assert seen == evaluate(model, examples, seed=0)
