"""The synthetic data engine.

Sample a spec, canonicalise it, place and route it, verify it, keep it. The
verification step is not a formality — it is the only reason a synthetic corpus
can be trusted at all.
"""

from .corpus import BuildReport, Example, SplitSpec, build, corrupt, load
from .paraphrase import LLMParaphraser, name_of, paraphrase
from .sample import (
    ALL_GATES,
    ROUTABLE_GATES,
    SampleConfig,
    enumerate_small_specs,
    sample_spec,
    sample_unique,
)

__all__ = [
    "ALL_GATES",
    "BuildReport",
    "Example",
    "LLMParaphraser",
    "ROUTABLE_GATES",
    "SampleConfig",
    "SplitSpec",
    "build",
    "corrupt",
    "enumerate_small_specs",
    "load",
    "name_of",
    "paraphrase",
    "sample_spec",
    "sample_unique",
]
