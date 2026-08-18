"""The two generators, exercised on CPU.

These models were written against §05 and then never run — every one of the
bugs asserted against below was live in code that imported cleanly and had a
full docstring. Nothing here is a quality measurement; a two-layer model on
random tokens learns nothing worth reporting. It asserts the plumbing: that a
step runs, that sampling produces block states and not spec tokens, and that
what comes out the far end is something the verifier will read.

Torch is optional, so the whole module skips without it.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("torch", reason="the generators are an optional extra")

import torch  # noqa: E402

from daedalus import tokens as T  # noqa: E402
from daedalus import vocab as V  # noqa: E402
from daedalus.grid import Grid  # noqa: E402
from daedalus.models import AutoregressiveModel, MaskedDiffusionModel  # noqa: E402
from daedalus.models.common import ModelConfig  # noqa: E402
from daedalus.redsim import Verifier  # noqa: E402
from daedalus.spec import Spec  # noqa: E402

#: Small enough to run in a second, wide enough to exercise every code path.
TINY = ModelConfig(n_layers=2, d_model=64, n_heads=4, d_ff=128)

BOTH = [("mdm", MaskedDiffusionModel), ("ar", AutoregressiveModel)]


def placed_nand():
    spec = Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)")
    return spec, spec.default_placement(random.Random(0))


def sample_once(cls, placed, seed: int = 0):
    prefix, _slots = T.spec_prefix(placed)
    torch.manual_seed(seed)
    model = cls(TINY)
    model.eval()
    kwargs = {"steps": 8} if cls is MaskedDiffusionModel else {}
    out = model.sample(
        torch.tensor([prefix], dtype=torch.long),
        legality=T.legality_mask(),
        pinned=T.port_mask(placed),
        **kwargs,
    )
    return out[0].tolist()


class TestTraining:
    @pytest.mark.parametrize("name,cls", BOTH)
    def test_a_step_runs_forward_and_backward(self, name, cls):
        torch.manual_seed(0)
        model = cls(TINY)
        tokens = torch.randint(0, V.CONTROL_BASE, (2, V.SEQ_LEN))
        loss = model.loss(tokens)
        assert torch.isfinite(loss), name
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, f"{name}: nothing received a gradient"
        assert all(torch.isfinite(g).all() for g in grads), name

    def test_the_smoke_run_decreases_the_objective(self):
        from daedalus.train.pretrain import smoke

        out = smoke(seed=0)
        assert out["last_loss"] < out["first_loss"]
        assert out["parameters"] > 0


class TestSampling:
    def test_autoregressive_decoding_starts_from_an_empty_body(self):
        # The body grows from zero cells, so a fixed-size position table makes
        # the very first decode step a shape error. Training never sees this
        # because it always passes a full grid.
        _spec, placed = placed_nand()
        body = sample_once(AutoregressiveModel, placed)
        assert len(body) == V.CELLS

    @pytest.mark.parametrize("name,cls", BOTH)
    def test_samples_are_block_states_not_spec_tokens(self, name, cls):
        # The head spans blocks *and* prefix tokens. If the legality mask is
        # only as wide as the block vocabulary the sampler can put a "<row>"
        # into a grid cell, which is not a block.
        _spec, placed = placed_nand()
        body = sample_once(cls, placed)
        assert max(body) < V.VOCAB_SIZE, f"{name}: emitted a prefix token"
        assert min(body) >= 0

    @pytest.mark.parametrize("name,cls", BOTH)
    def test_pinned_ports_are_left_alone(self, name, cls):
        _spec, placed = placed_nand()
        body = sample_once(cls, placed)
        for cell, token in T.port_mask(placed).items():
            assert body[cell] == token, f"{name}: resampled a pinned port"

    def test_the_legality_mask_lines_up_with_the_head(self):
        mask = T.legality_mask()
        assert len(mask) == V.CELLS
        assert all(len(row) == T.TOTAL_VOCAB for row in mask)
        # The prefix half can never be a block.
        assert all(not any(row[V.VOCAB_SIZE :]) for row in mask)

    def test_the_substrate_layer_admits_only_air_and_stone(self):
        mask = T.legality_mask()
        row = mask[V.index(4, V.SUBSTRATE_Y, 4)]
        assert row[V.AIR] and row[V.SOLID]
        assert not row[V.WIRE]


class TestEndToEnd:
    @pytest.mark.parametrize("name,cls", BOTH)
    def test_the_verifier_reads_what_the_model_writes(self, name, cls):
        # The point of the whole exercise: a sample is a grid the verifier can
        # take a position on. An untrained model earns MALFORMED, and that is
        # a real verdict rather than a crash -- it is the reward signal doing
        # its job on a model that has learned nothing yet.
        _spec, placed = placed_nand()
        body = sample_once(cls, placed)
        with Verifier() as v:
            verdict = v.evaluate(Grid.from_tokens(body), placed)
        assert verdict is not None, name
        assert hasattr(verdict, "is_pass"), name
