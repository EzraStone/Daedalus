"""Supervised pretraining on the procedural corpus.

Deliberately unremarkable. The corpus is exact, the objective is standard, and
the only decision worth arguing about is the loss weighting — see
:func:`daedalus.models.common.class_weights` for why an unweighted run
converges to a model that emits empty grids and reports excellent loss.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .. import vocab as V
from ..data.corpus import Example
from ..models import HAVE_TORCH, ModelConfig, class_weights, cosine_schedule, require_torch
from ..spec import Spec
from ..tokens import encode


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 1
    batch_size: int = 24
    lr: float = 3e-4
    warmup: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 25
    device: str = "auto"
    #: fp32 by default. RDNA lacks proper bf16 matrix support, and at 25M
    #: parameters fp32 is genuinely fine while removing a class of NaN
    #: debugging that costs more time than the memory saves.
    dtype: str = "fp32"
    seed: int = 0


def pick_device(requested: str = "auto") -> str:
    require_torch()
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def token_counts(examples: list[Example]) -> list[int]:
    """Block-state frequencies, for inverse-frequency weighting."""
    counts = [0] * V.VOCAB_SIZE
    for e in examples:
        for t in e.tokens:
            counts[t] += 1
    return counts


def to_sequences(examples: list[Example]):
    """Tokenise a corpus split into flat model input."""
    out = []
    for e in examples:
        spec = Spec.parse(e.spec_source)
        placed = spec.place(e.input_z, e.output_z)
        out.append(encode(e.grid(), placed).flat())
    return out


def train(
    model,
    examples: list[Example],
    cfg: TrainConfig = TrainConfig(),
    out_dir: str | Path | None = None,
):
    """Fit ``model`` on ``examples``. Returns the loss history."""
    require_torch()
    import torch

    device = pick_device(cfg.device)
    model = model.to(device)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    sequences = to_sequences(examples)
    weights = class_weights(token_counts(examples)).to(device)
    # The prefix vocabulary is never a prediction target, so it gets no weight.
    full_weights = torch.cat(
        [weights, torch.zeros(model.cfg.vocab_size - V.VOCAB_SIZE, device=device)]
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(sequences) // cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    history = []
    step = 0
    started = time.time()

    for epoch in range(cfg.epochs):
        rng.shuffle(sequences)
        for i in range(steps_per_epoch):
            batch = sequences[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            if not batch:
                continue
            tokens = torch.tensor(batch, dtype=torch.long, device=device)
            loss = model.loss(tokens, weights=full_weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            for group in opt.param_groups:
                group["lr"] = cosine_schedule(step, total, cfg.warmup, cfg.lr)
            opt.step()
            step += 1
            if step % cfg.log_every == 0 or step == total:
                history.append({"step": step, "epoch": epoch, "loss": float(loss.item())})

    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": model.cfg.__dict__}, path / "model.pt")
        (path / "history.json").write_text(
            json.dumps({"seconds": time.time() - started, "history": history}, indent=2)
        )
    return history


def smoke(seed: int = 0) -> dict:
    """A tiny CPU run that proves the objective decreases.

    Not a result — a wiring check. It exists so "the model code runs" is
    something the test suite asserts rather than something the README claims.
    """
    require_torch()
    import torch

    from ..models import MaskedDiffusionModel

    torch.manual_seed(seed)
    cfg = ModelConfig(n_layers=2, d_model=64, n_heads=4, d_ff=128)
    model = MaskedDiffusionModel(cfg)
    tokens = torch.randint(0, V.CONTROL_BASE, (4, V.SEQ_LEN))
    tokens[:, V.PREFIX_LEN :] = 0  # a grid of air, which the model can learn
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = last = None
    for i in range(20):
        loss = model.loss(tokens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss.item())
        last = float(loss.item())
    return {"first_loss": first, "last_loss": last, "parameters": model.body.n_parameters()}


HAVE_TORCH = HAVE_TORCH
