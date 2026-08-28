"""The autoregressive baseline.

A decoder-only prefix-LM: the spec tokens are visible to everything, and the
grid is predicted in raster order. Two days of work whose job is to prove the
data pipeline teaches *something*.

Keep it in the repository as a documented baseline rather than deleting it
once the diffusion model wins. Expect mediocre pass rates and no inpainting —
that is not a disappointment, it is the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import vocab as V
from .common import (
    HAVE_TORCH,
    ModelConfig,
    PromptEncoder,
    as_legality,
    require_torch,
    support_tables,
)

if HAVE_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from .common import Body

    class AutoregressiveModel(nn.Module):
        """Prefix-LM over ``[spec | grid]``."""

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
            self.body = Body(self.cfg, causal=True)

        def forward(self, tokens, nl_embeddings=None):
            return self.body(tokens, nl_embeddings)

        def loss(self, tokens, weights=None, nl_embeddings=None):
            """Cross-entropy over the grid positions only.

            The prefix is conditioning, not a prediction target. Including it
            would let the model lower its loss by memorising the spec encoding,
            which teaches it nothing about circuits.
            """
            logits = self(tokens, nl_embeddings)
            p = self.cfg.prefix_len
            pred = logits[:, p - 1 : -1]
            target = tokens[:, p:]
            return F.cross_entropy(
                pred.reshape(-1, pred.shape[-1]),
                target.reshape(-1),
                weight=weights,
            )

        @torch.no_grad()
        def sample(
            self,
            prefix,
            temperature: float = 0.9,
            top_k: int | None = None,
            legality=None,
            pinned=None,
            enforce_support=True,
        ):
            """Generate grids one cell at a time in raster order.

            ``legality`` is the position-only mask from :mod:`daedalus.tokens`
            and ``pinned`` fixes the port cells. Both are free correctness: they
            cost no training and remove whole classes of sample the verifier
            would reject before simulating.
            """
            device = prefix.device
            batch = prefix.shape[0]
            legality = as_legality(legality, device)
            sup_tokens, sup_required, opaque = support_tables(device)
            tokens = torch.cat(
                [prefix, torch.zeros(batch, V.CELLS, dtype=torch.long, device=device)], dim=1
            )
            # Supports that raster order has not reached yet. Placing a torch
            # that hangs east means the block it hangs on is decided later, so
            # the commitment is recorded here and honoured when the loop
            # arrives, rather than being left to luck.
            owed: dict[int, int] = {}

            # Port cells are fixed from the start, so write them in now. Their
            # values are already decided, and a support cell pinned to a lever
            # can never become solid however far ahead it sits -- without this
            # the veto below cannot see that and a torch hangs on nothing.
            settled = torch.zeros(V.CELLS, dtype=torch.bool, device=device)
            for cell, token in (pinned or {}).items():
                tokens[:, self.cfg.prefix_len + cell] = token
                settled[cell] = True

            for i in range(V.CELLS):
                pos = self.cfg.prefix_len + i
                if pinned is not None and i in pinned:
                    tokens[:, pos] = pinned[i]
                    continue
                if i in owed:
                    tokens[:, pos] = owed.pop(i)
                    continue
                logits = self(tokens[:, :pos])[:, -1]
                if legality is not None:
                    logits = logits.masked_fill(~legality[i], float("-inf"))
                if enforce_support:
                    # Raster order is layer-major, so a cell's support sits
                    # either lower down or earlier in the same layer -- decided
                    # already, either way. That makes the rule exactly
                    # checkable here, with no lookahead and nothing to revise.
                    # Supports that read forward are left to the model.
                    where = sup_required[i]
                    decided = (where >= 0) & ((where < i) | settled[where.clamp_min(0)])
                    if decided.any():
                        held = opaque[tokens[:, self.cfg.prefix_len + where.clamp_min(0)]]
                        veto = decided & ~held
                        logits[:, sup_tokens] = logits[:, sup_tokens].masked_fill(
                            veto, float("-inf")
                        )
                    logits[:, sup_tokens[where < 0]] = float("-inf")
                logits = logits / max(temperature, 1e-5)
                if top_k:
                    kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                tokens[:, pos] = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)

                if enforce_support:
                    # Whatever was just placed may owe a support cell further
                    # along. One batch entry deciding this is enough to settle
                    # the cell for all of them, so the veto above keeps the
                    # rest consistent with it.
                    for token in tokens[:, pos].tolist():
                        slot = (sup_tokens == token).nonzero()
                        if not slot.numel():
                            continue
                        target = int(sup_required[i, int(slot[0])])
                        if target > i:
                            owed.setdefault(target, V.SOLID)
            return tokens[:, self.cfg.prefix_len :]

else:  # pragma: no cover - import-time stub

    @dataclass
    class AutoregressiveModel:  # type: ignore[no-redef]
        """Placeholder raising a helpful error when torch is absent."""

        cfg: object = None

        def __post_init__(self):
            require_torch()


__all__ = ["AutoregressiveModel", "ModelConfig"]
