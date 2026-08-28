"""The two generative models.

Same 25M-parameter body, different objective. The autoregressive model is a
sanity check on the data pipeline; the masked diffusion model is the one that
does repair, and repair is the capability players actually want.
"""

from .ar import AutoregressiveModel
from .common import (
    HAVE_TORCH,
    ModelConfig,
    class_weights,
    cosine_schedule,
    require_torch,
)

if HAVE_TORCH:  # pragma: no cover - the encoder needs torch to exist at all
    from .common import PromptEncoder
else:  # pragma: no cover
    PromptEncoder = None
from .mdm import MASK_ID, MaskedDiffusionModel

__all__ = [
    "HAVE_TORCH",
    "MASK_ID",
    "AutoregressiveModel",
    "MaskedDiffusionModel",
    "ModelConfig",
    "PromptEncoder",
    "class_weights",
    "cosine_schedule",
    "require_torch",
]
