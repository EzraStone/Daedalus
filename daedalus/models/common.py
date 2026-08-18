"""Shared transformer parts.

Both models are the same 25M-parameter body with a different attention mask
and a different training objective, which is deliberate: it makes the
autoregressive/diffusion comparison in §07 a comparison of objectives rather
than of architectures.

``torch`` is an optional dependency. The rest of the package — the verifier,
the compiler, the corpus, the evaluation harness — works without it, so a
clone can run its whole test suite before anyone installs a 2 GB wheel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:  # pragma: no cover - exercised by whichever environment is present
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAVE_TORCH = False

from .. import vocab as V
from ..tokens import TOTAL_VOCAB


def require_torch() -> None:
    if not HAVE_TORCH:
        raise ImportError(
            "this needs PyTorch. Install it for your accelerator, e.g.\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2\n"
            "See docs/hardware.md for the RX 7600 setup."
        )


def as_legality(mask, device):
    """Normalise the position-only legality mask into a bool tensor.

    :mod:`daedalus.tokens` returns plain lists and stays free of torch — the
    corpus, export and evaluation paths use it without a training install — so
    the conversion happens here, once per sample call rather than once per
    denoising step.
    """
    if mask is None:
        return None
    if torch.is_tensor(mask):
        return mask.to(device=device, dtype=torch.bool)
    return torch.tensor(mask, dtype=torch.bool, device=device)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """The 25M-parameter body from §05.

    A 48-token vocabulary is astonishingly small — smaller than a DNA k-mer
    vocabulary — which is exactly why this size is right and why it trains on
    one consumer GPU.
    """

    vocab_size: int = TOTAL_VOCAB
    n_layers: int = 12
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1024  # SwiGLU: two projections of this width
    dropout: float = 0.0
    seq_len: int = V.SEQ_LEN
    prefix_len: int = V.PREFIX_LEN
    #: Alternate layers attending within a y-layer and across layers at fixed
    #: (x, z). Cheaper than full attention over 1536 tokens and matches the
    #: geometry. Off by default so it stays a clean ablation.
    layer_factored_attention: bool = False

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must divide by n_heads")
        return self.d_model // self.n_heads

    def parameter_estimate(self) -> int:
        """Rough parameter count, for sanity-checking a config before it runs."""
        emb = self.vocab_size * self.d_model + V.CELLS * self.d_model
        per_layer = 4 * self.d_model**2 + 3 * self.d_model * self.d_ff
        return emb + self.n_layers * per_layer + self.d_model * self.vocab_size


if HAVE_TORCH:

    class RoPE(nn.Module):
        """Rotary position embeddings over the flat sequence."""

        def __init__(self, d_head: int, max_len: int, base: float = 10000.0):
            super().__init__()
            inv = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
            t = torch.arange(max_len).float()
            freqs = torch.outer(t, inv)
            self.register_buffer("cos", freqs.cos()[None, None], persistent=False)
            self.register_buffer("sin", freqs.sin()[None, None], persistent=False)

        def forward(self, x):
            # x: (batch, heads, seq, d_head)
            seq = x.shape[-2]
            cos = self.cos[..., :seq, :]
            sin = self.sin[..., :seq, :]
            x1, x2 = x[..., ::2], x[..., 1::2]
            out = torch.empty_like(x)
            out[..., ::2] = x1 * cos - x2 * sin
            out[..., 1::2] = x1 * sin + x2 * cos
            return out

    class SwiGLU(nn.Module):
        def __init__(self, d_model: int, d_ff: int):
            super().__init__()
            self.gate = nn.Linear(d_model, d_ff, bias=False)
            self.up = nn.Linear(d_model, d_ff, bias=False)
            self.down = nn.Linear(d_ff, d_model, bias=False)

        def forward(self, x):
            return self.down(F.silu(self.gate(x)) * self.up(x))

    class Attention(nn.Module):
        def __init__(self, cfg: ModelConfig, causal: bool):
            super().__init__()
            self.cfg = cfg
            self.causal = causal
            self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
            self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.rope = RoPE(cfg.d_head, cfg.seq_len)

        def forward(self, x, attn_mask=None):
            b, s, _ = x.shape
            h, dh = self.cfg.n_heads, self.cfg.d_head
            q, k, v = self.qkv(x).split(self.cfg.d_model, dim=-1)
            q = self.rope(q.view(b, s, h, dh).transpose(1, 2))
            k = self.rope(k.view(b, s, h, dh).transpose(1, 2))
            v = v.view(b, s, h, dh).transpose(1, 2)
            # SDPA rather than flash attention: RDNA cards fall back to the
            # math kernel, and at this sequence length that is tolerable.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=self.causal and attn_mask is None
            )
            return self.proj(out.transpose(1, 2).reshape(b, s, -1))

    class Block(nn.Module):
        def __init__(self, cfg: ModelConfig, causal: bool):
            super().__init__()
            self.norm1 = nn.RMSNorm(cfg.d_model)
            self.attn = Attention(cfg, causal)
            self.norm2 = nn.RMSNorm(cfg.d_model)
            self.ff = SwiGLU(cfg.d_model, cfg.d_ff)
            self.drop = nn.Dropout(cfg.dropout)

        def forward(self, x, attn_mask=None):
            x = x + self.drop(self.attn(self.norm1(x), attn_mask))
            return x + self.drop(self.ff(self.norm2(x)))

    class Body(nn.Module):
        """Embeddings, blocks and the output head, shared by both models."""

        def __init__(self, cfg: ModelConfig, causal: bool):
            super().__init__()
            self.cfg = cfg
            self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
            # A learned 3D positional embedding on top of RoPE. RoPE knows the
            # sequence order; this knows the geometry, which is what tells the
            # model that two cells one row apart in z are neighbours even
            # though they are sixteen tokens apart in the sequence.
            self.pos3d = nn.Embedding(V.CELLS, cfg.d_model)
            self.blocks = nn.ModuleList(Block(cfg, causal) for _ in range(cfg.n_layers))
            self.norm = nn.RMSNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.apply(self._init)
            self.register_buffer(
                "cell_ids", torch.arange(V.CELLS), persistent=False
            )

        @staticmethod
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        def embed(self, tokens, nl_embeddings=None):
            x = self.tok(tokens)
            body = x[:, self.cfg.prefix_len :]
            # Slice the position table to the body actually present. Training
            # always passes a full grid, but autoregressive sampling grows the
            # body one cell at a time and starts at zero, so a fixed-size table
            # makes the first decode step a shape error.
            grid_pos = self.pos3d(self.cell_ids[: body.shape[1]])
            x = torch.cat([x[:, : self.cfg.prefix_len], body + grid_pos], dim=1)
            if nl_embeddings is not None:
                # Splice projected sentence embeddings into their prefix slots.
                slots = nl_embeddings.shape[1]
                x = x.clone()
                x[:, 1 : 1 + slots] = nl_embeddings
            return x

        def forward(self, tokens, nl_embeddings=None, attn_mask=None):
            x = self.embed(tokens, nl_embeddings)
            for block in self.blocks:
                x = block(x, attn_mask)
            return self.head(self.norm(x))

        def n_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())


def class_weights(counts, clip: float = 10.0):
    """Inverse-frequency weights over the block vocabulary, clipped.

    85% of tokens are air. Unweighted cross-entropy converges happily to a
    model that emits empty grids and reports excellent loss, so the aggregate
    token accuracy reads about 97% while nothing the model produces works.
    Clipping at 10x stops the rarest states dominating the gradient.
    """
    require_torch()
    total = float(sum(counts))
    weights = []
    for c in counts:
        weights.append(min(clip, total / (len(counts) * c)) if c else clip)
    return torch.tensor(weights, dtype=torch.float32)


def cosine_schedule(step: int, total: int, warmup: int, peak: float, floor: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))
