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
from .common import HAVE_TORCH, ModelConfig, as_legality, require_torch

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
            tokens = torch.cat(
                [prefix, torch.zeros(batch, V.CELLS, dtype=torch.long, device=device)], dim=1
            )
            for i in range(V.CELLS):
                pos = self.cfg.prefix_len + i
                logits = self(tokens[:, :pos])[:, -1]
                if pinned is not None and i in pinned:
                    tokens[:, pos] = pinned[i]
                    continue
                if legality is not None:
                    logits = logits.masked_fill(~legality[i], float("-inf"))
                logits = logits / max(temperature, 1e-5)
                if top_k:
                    kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                tokens[:, pos] = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)
            return tokens[:, self.cfg.prefix_len :]

else:  # pragma: no cover - import-time stub

    @dataclass
    class AutoregressiveModel:  # type: ignore[no-redef]
        """Placeholder raising a helpful error when torch is absent."""

        cfg: object = None

        def __post_init__(self):
            require_torch()


__all__ = ["AutoregressiveModel", "ModelConfig"]
