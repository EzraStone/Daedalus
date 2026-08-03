"""Daedalus — text-conditioned generation of verified redstone circuits.

The package is layered so that each stage consumes only what the previous one
guarantees:

``daedalus.vocab``
    The 48-token block-state vocabulary and the ``y -> z -> x`` grid order.
    Mirrors ``crates/redsim/src/block.rs`` exactly; the two are checked against
    each other in the test suite.
``daedalus.redsim``
    Client for the Rust verifier. Everything downstream is defined against its
    verdicts.
``daedalus.spec``
    The specification DSL, its canonical form and its semantic hash.
``daedalus.synth``
    Gate library, placer and dust router: the procedural compiler that both
    builds the training corpus and serves as the first baseline.
``daedalus.data``
    Corpus construction, paraphrasing and difficulty-stratified splits.
``daedalus.models``
    The autoregressive baseline and the masked discrete diffusion model.
``daedalus.train``
    The verifier-guided self-improvement loop.
``daedalus.eval``
    Metrics and the four baselines.
``daedalus.schematic``
    ``.schem`` and ``.litematic`` writing, with a dependency-free NBT encoder.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
