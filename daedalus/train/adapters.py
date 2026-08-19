"""Plugs between a torch model and the verifier-guided loop.

:mod:`daedalus.train.loop` takes its sampler, its spec source and its trainer
as arguments, so that the acceptance logic could be tested without a GPU. That
worked -- and it also meant nothing ever connected a real model to it. These
are the three plugs, kept apart from the loop itself so the loop stays
testable without torch.
"""

from __future__ import annotations

import random

from .. import tokens as T
from ..data import sample_unique
from ..models import require_torch
from ..spec import Spec
from .loop import Accepted
from .pretrain import TrainConfig, train


class ModelSampler:
    """The loop's ``Sampler``: draw ``k`` candidate grids for one placed spec.

    Both constraint layers are on. They cost no training and, on an
    undertrained model, they are the difference between candidates the
    verifier grades and candidates it throws out on sight.
    """

    def __init__(self, model, steps: int = 24, temperature: float = 0.9, guidance: float = 2.0):
        require_torch()
        self.model = model
        self.steps = steps
        self.temperature = temperature
        self.guidance = guidance

    def __call__(self, placed, k: int) -> list[list[int]]:
        import torch

        self.model.eval()
        device = next(self.model.parameters()).device
        prefix, _slots = T.spec_prefix(placed)
        batch = torch.tensor([prefix] * k, dtype=torch.long, device=device)
        kwargs = {"temperature": self.temperature}
        if hasattr(self.model, "loss_at"):
            kwargs["steps"] = self.steps
            kwargs["guidance"] = self.guidance
        with torch.no_grad():
            out = self.model.sample(
                batch,
                legality=T.legality_mask(placed),
                pinned=T.port_mask(placed),
                **kwargs,
            )
        return [row.tolist() for row in out.cpu()]


class ModelTrainer:
    """The loop's ``Trainer``: fine-tune on what the round accepted.

    Rounds continue from the weights the last one produced rather than from
    scratch, which is the entire point -- the model is supposed to be climbing.
    """

    def __init__(self, model, cfg: TrainConfig | None = None):
        require_torch()
        self.model = model
        self.cfg = cfg or TrainConfig(epochs=1, batch_size=8)

    def __call__(self, dataset, round_index: int) -> dict:
        examples = as_examples(dataset)
        if not examples:
            return {"examples": 0}
        history = train(self.model, examples, self.cfg)
        return {
            "examples": len(examples),
            "final_loss": history[-1]["loss"] if history else None,
        }


def as_examples(dataset) -> list:
    """Turn a round's accepted layouts into corpus examples to train on."""
    from ..data.corpus import Example

    out = []
    for item in list(dataset.accepted) + list(dataset.anchors):
        spec, placed = item.spec, item.placed
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
                tokens=list(item.tokens),
                latency_rt=item.latency_rt,
                blocks=item.blocks,
                bbox=[0, 0, 0],
                prompts=[],
            )
        )
    return out


def spec_source(rng: random.Random, count: int, difficulty: tuple[int, int]):
    """The loop's spec supply, filtered to the curriculum's gate-count bucket."""
    low, high = difficulty
    out = []
    # Oversample: the gate count of a drawn spec is not something the sampler
    # takes as an argument, so the bucket is a filter rather than a request.
    for spec in sample_unique(rng, count * 4):
        if low <= spec.gates <= high:
            out.append((spec, spec.default_placement(rng)))
        if len(out) >= count:
            break
    return out


def anchors_from(examples) -> list[Accepted]:
    """Original corpus examples, in the form the loop mixes back in.

    Self-training only on filtered samples is a mode-collapse machine, and
    this is the cheapest defence there is.
    """
    out = []
    for e in examples:
        spec = Spec.parse(e.spec_source)
        out.append(
            Accepted(spec, spec.default_placement(), list(e.tokens), e.blocks, e.latency_rt)
        )
    return out
