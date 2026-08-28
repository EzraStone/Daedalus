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
from .common import (
    HAVE_TORCH,
    ModelConfig,
    PromptEncoder,
    as_legality,
    require_torch,
    support_tables,
)

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
            # Only built when the config asks for it, so a spec-conditioned
            # model carries no prompt parameters at all and the two are a
            # clean comparison rather than the same model with a dead branch.
            self.prompts = (
                PromptEncoder(self.cfg.d_model, self.cfg.nl_slots)
                if self.cfg.nl_slots
                else None
            )
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
            """Training objective: a mask rate drawn fresh for every batch.

            Because ``t`` is random and the result is scaled by ``1/t``, the
            number this returns swings by orders of magnitude from batch to
            batch on an unchanged model. It is the right thing to descend and
            the wrong thing to plot -- use :func:`daedalus.train.evaluate` for
            a figure that means something across steps.
            """
            t = torch.rand(tokens.shape[0], device=tokens.device).clamp_(eps, 1.0)
            return self.loss_at(tokens, t, weights, nl_embeddings)

        def loss_at(self, tokens, t, weights=None, nl_embeddings=None):
            """Cross-entropy on masked positions only, weighted by ``1/t``.

            The ``1/t`` weight is what makes this a bound on the likelihood
            rather than an arbitrary denoising objective: a batch drawn at a
            low mask rate carries few supervised positions but each is worth
            proportionally more.
            """
            p = self.cfg.prefix_len
            prefix, body = tokens[:, :p], tokens[:, p:]
            if not torch.is_tensor(t):
                t = torch.full((body.shape[0],), float(t), device=body.device)
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

        @staticmethod
        def _settle_support(body, frozen, sup_tokens, sup_idx, in_volume, opaque, last=False):
            """Make committed states and their supports agree.

            The veto alone is not enough. Confidence-ordered unmasking reveals
            many cells at once, so a block and the cell holding it up can be
            committed in the same step, each looking permissible because the
            other was still masked when the logits were scored.

            Two repairs, in order: an undecided support becomes stone, which is
            what the model implied by placing the block; and a block whose
            support is already decided against it goes back to MASK, to be
            reconsidered on a later step when the veto can see the conflict.
            """
            batch = body.shape[0]
            lookup = torch.full((opaque.shape[0],), -1, dtype=torch.long, device=body.device)
            lookup[sup_tokens] = torch.arange(len(sup_tokens), device=body.device)
            slot = lookup[body]  # which support rule each committed state uses
            needs = slot >= 0

            rows = torch.arange(body.shape[1], device=body.device).expand(batch, -1)
            where = sup_idx[rows, slot.clamp_min(0)]
            valid = needs & in_volume[rows, slot.clamp_min(0)]

            held = torch.where(valid, body.gather(1, where.clamp_min(0)), body)
            prop = valid & (held == MASK_ID)
            for b in range(batch):
                targets = where[b][prop[b]]
                if targets.numel():
                    body[b, targets] = V.SOLID

            # Re-read: a support just filled in is no longer a conflict.
            held = torch.where(valid, body.gather(1, where.clamp_min(0)), body)
            conflict = valid & ~opaque[held]
            if not last:
                body = body.masked_fill(conflict & ~frozen, MASK_ID)
                return body

            # On the final step there is no later pass to reconsider anything,
            # and a cell left masked is malformed outright. So the support is
            # overwritten instead: losing one block the model chose is a
            # smaller loss than losing the whole grid.
            for b in range(batch):
                targets = where[b][conflict[b]]
                if targets.numel():
                    keep = targets[~frozen[targets]]
                    body[b, keep] = V.SOLID
            return body

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
            enforce_support=True,
            nl_embeddings=None,
        ):
            """Confidence-ordered unmasking.

            ``known`` is what makes this general: a dict of ``{cell: token}``
            that is fixed from the start and never resampled. Pass the ports
            for plain generation, or a whole working circuit minus a damaged
            region for repair.

            ``guidance`` is classifier-free guidance in logit space. §05 expects
            this to be one of the largest single wins, so it is on by default
            and ``uncond_prefix`` makes the ablation a one-line change.

            ``enforce_support`` is the neighbour-dependent half of legality,
            which the position-only mask cannot express. Dust needs an opaque
            block under it; an untrained model gets that wrong on four out of
            five dust cells and every grid comes back malformed, so it is on by
            default and off only for the ablation.
            """
            device = prefix.device
            batch = prefix.shape[0]
            p = self.cfg.prefix_len
            legality = as_legality(legality, device)
            sup_tokens, sup_required, opaque = support_tables(device)
            in_volume = sup_required >= 0
            sup_idx = sup_required.clamp_min(0)
            body = torch.full((batch, V.CELLS), MASK_ID, dtype=torch.long, device=device)

            fixed = dict(pinned or {})
            fixed.update(known or {})
            frozen = torch.zeros(V.CELLS, dtype=torch.bool, device=device)
            for cell, token in fixed.items():
                body[:, cell] = token
                frozen[cell] = True

            for step in range(steps):
                logits = self(torch.cat([prefix, body], dim=1), nl_embeddings)[:, p:]
                if guidance != 1.0 and uncond_prefix is not None:
                    uncond = self(torch.cat([uncond_prefix, body], dim=1))[:, p:]
                    logits = uncond + guidance * (logits - uncond)
                if legality is not None:
                    logits = logits.masked_fill(~legality, float("-inf"))
                if enforce_support:
                    # Veto any state whose support is already committed to
                    # something that will not hold it. A support cell still
                    # masked stays permissive -- it can yet become stone.
                    held = body[:, sup_idx]  # (batch, cells, supported states)
                    blocked = in_volume & (held != MASK_ID) & ~opaque[held]
                    logits[:, :, sup_tokens] = logits[:, :, sup_tokens].masked_fill(
                        blocked | ~in_volume, float("-inf")
                    )

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

                if enforce_support:
                    body = self._settle_support(
                        body,
                        frozen,
                        sup_tokens,
                        sup_idx,
                        in_volume,
                        opaque,
                        last=step == steps - 1,
                    )

            # Anything still masked at the end is a cell the model would not
            # commit to. Leave it as MASK rather than guessing: the verifier
            # reports it as malformed, which is the honest outcome.
            return body

        @torch.no_grad()
        def repair(self, prefix, tokens, damaged, **kwargs):
            """Re-generate ``damaged`` cells, holding everything else fixed.

            The same forward pass as generation. That is the whole argument for
            masked diffusion over an autoregressive model in this domain.

            ``tokens`` is one grid -- the damaged circuit -- given as a
            ``(1, CELLS)`` tensor or any sequence of ``CELLS`` ids. ``prefix``
            may be a batch, which is how you ask for several different repairs
            of the same damage and let the verifier pick.
            """
            if not torch.is_tensor(tokens):
                tokens = torch.tensor(tokens, dtype=torch.long, device=prefix.device)
            if tokens.dim() == 1:
                tokens = tokens[None]
            if tokens.shape[0] != 1:
                raise ValueError("repair takes one grid; batch the prefix instead")
            damaged = set(damaged)
            known = {i: int(tokens[0, i]) for i in range(V.CELLS) if i not in damaged}
            return self.sample(prefix, known=known, **kwargs)

else:  # pragma: no cover - import-time stub

    @dataclass
    class MaskedDiffusionModel:  # type: ignore[no-redef]
        cfg: object = None

        def __post_init__(self):
            require_torch()


__all__ = ["MaskedDiffusionModel", "ModelConfig", "MASK_ID", "TOTAL_VOCAB"]
