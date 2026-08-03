"""Masked discrete diffusion — the model that earns the repository.

Absorbing-state diffusion over the same token grid: the forward process
replaces tokens with ``MASK`` independently at rate *t*, and the model learns
to fill them back in. Sampling unmasks in confidence order over 16-32 steps.

Why it wins over the autoregressive baseline has little to do with sample
quality and everything to do with what the objective *is*. Because the model
is trained to reconstruct an arbitrary masked subset, **any** subset of the
grid can serve as conditioning at inference time. Repair, extension and
constrained placement stop being three features and become one operation with
a different mask:

* repair — mask the broken region, keep the rest;
* extension — mask empty space, keep the half-built circuit;
* generation — mask everything.

That is the capability that turns this from a demo into something a player
would actually open.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import vocab as V
from ..tokens import TOTAL_VOCAB
from .common import HAVE_TORCH, ModelConfig, require_torch

#: The absorbing state, offset into the shared embedding table.
MASK_ID = V.MASK

if HAVE_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from .common import Body

    class MaskedDiffusionModel(nn.Module):
        """Bidirectional encoder trained to denoise a masked grid."""

        def __init__(self, cfg: ModelConfig | None = None):
            super().__init__()
            require_torch()
            self.cfg = cfg or ModelConfig()
            self.body = Body(self.cfg, causal=False)

        def forward(self, tokens, nl_embeddings=None):
            return self.body(tokens, nl_embeddings)

        # -- training ------------------------------------------------------

        def corrupt(self, body, t):
            """Replace each token with MASK independently with probability t."""
            noise = torch.rand_like(body, dtype=torch.float32)
            masked = noise < t[:, None]
            return torch.where(masked, torch.full_like(body, MASK_ID), body), masked

        def loss(self, tokens, weights=None, nl_embeddings=None, eps: float = 1e-3):
            """Cross-entropy on masked positions only, weighted by ``1/t``.

            The ``1/t`` weight is what makes this a bound on the likelihood
            rather than an arbitrary denoising objective: a batch drawn at a
            low mask rate carries few supervised positions but each is worth
            proportionally more.
            """
            p = self.cfg.prefix_len
            prefix, body = tokens[:, :p], tokens[:, p:]
            t = torch.rand(body.shape[0], device=body.device).clamp_(eps, 1.0)
            noisy, masked = self.corrupt(body, t)
            logits = self(torch.cat([prefix, noisy], dim=1), nl_embeddings)[:, p:]

            per_token = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                body.reshape(-1),
                weight=weights,
                reduction="none",
            ).view_as(body)
            per_token = per_token * masked
            weight = (1.0 / t)[:, None]
            denom = masked.sum().clamp_min(1)
            return (per_token * weight).sum() / denom

        # -- sampling ------------------------------------------------------

        @torch.no_grad()
        def sample(
            self,
            prefix,
            steps: int = 24,
            temperature: float = 0.9,
            guidance: float = 2.0,
            uncond_prefix=None,
            legality=None,
            pinned=None,
            known=None,
        ):
            """Confidence-ordered unmasking.

            ``known`` is what makes this general: a dict of ``{cell: token}``
            that is fixed from the start and never resampled. Pass the ports
            for plain generation, or a whole working circuit minus a damaged
            region for repair.

            ``guidance`` is classifier-free guidance in logit space. §05 expects
            this to be one of the largest single wins, so it is on by default
            and ``uncond_prefix`` makes the ablation a one-line change.
            """
            device = prefix.device
            batch = prefix.shape[0]
            p = self.cfg.prefix_len
            body = torch.full((batch, V.CELLS), MASK_ID, dtype=torch.long, device=device)

            fixed = dict(pinned or {})
            fixed.update(known or {})
            frozen = torch.zeros(V.CELLS, dtype=torch.bool, device=device)
            for cell, token in fixed.items():
                body[:, cell] = token
                frozen[cell] = True

            for step in range(steps):
                logits = self(torch.cat([prefix, body], dim=1))[:, p:]
                if guidance != 1.0 and uncond_prefix is not None:
                    uncond = self(torch.cat([uncond_prefix, body], dim=1))[:, p:]
                    logits = uncond + guidance * (logits - uncond)
                if legality is not None:
                    logits = logits.masked_fill(~legality, float("-inf"))

                probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                confidence, choice = probs.max(dim=-1)
                still_masked = (body == MASK_ID) & ~frozen
                if not still_masked.any():
                    break

                # Unmask a cosine-shaped fraction per step: cautious early,
                # decisive late. Unmasking everything at once is just one-shot
                # prediction and throws away what diffusion is for.
                remaining = still_masked.sum(dim=1)
                keep = torch.ceil(
                    remaining.float() * (1.0 - (step + 1) / steps)
                ).long()
                confidence = confidence.masked_fill(~still_masked, -1.0)
                for b in range(batch):
                    n_reveal = int(remaining[b].item() - keep[b].item())
                    if n_reveal <= 0:
                        continue
                    idx = confidence[b].topk(n_reveal).indices
                    body[b, idx] = choice[b, idx]

            # Anything still masked at the end is a cell the model would not
            # commit to. Leave it as MASK rather than guessing: the verifier
            # reports it as malformed, which is the honest outcome.
            return body

        @torch.no_grad()
        def repair(self, prefix, tokens, damaged, **kwargs):
            """Re-generate ``damaged`` cells, holding everything else fixed.

            The same forward pass as generation. That is the whole argument for
            masked diffusion over an autoregressive model in this domain.
            """
            known = {
                i: int(tokens[0, i].item()) for i in range(V.CELLS) if i not in set(damaged)
            }
            return self.sample(prefix, known=known, **kwargs)

else:  # pragma: no cover - import-time stub

    @dataclass
    class MaskedDiffusionModel:  # type: ignore[no-redef]
        cfg: object = None

        def __post_init__(self):
            require_torch()


__all__ = ["MaskedDiffusionModel", "ModelConfig", "MASK_ID", "TOTAL_VOCAB"]
