"""Turning a prompt into something the model can be conditioned on.

The prefix has always reserved slots for this — :func:`daedalus.tokens.spec_prefix`
marks them, and :meth:`Body.embed` splices vectors into them — and nothing ever
filled them. The canonical-spec path was the only conditioning that worked, so
"text-conditioned" was true of the design and not of the code.

**This is not the frozen sentence encoder §05 describes.** That needs a
pretrained model this repository does not ship and cannot download in the
environments it is built in. What is here instead is a small encoder trained
jointly with the generator, over hashed word features. The trade is explicit:

* it costs nothing to install and nothing to keep in step with an external
  checkpoint;
* it will not generalise to phrasings unlike the corpus, because it has never
  read anything else. A frozen encoder brings that generalisation with it, and
  is the reason §05 asked for one.

So this makes the path real and measurable, and leaves the open question --
does natural language conditioning work as well as the canonical spec? -- in a
state where it can actually be asked.

Hashing rather than a vocabulary file is deliberate. A vocabulary is state
that has to be built, saved beside the corpus, loaded with the checkpoint and
kept in step with all three; a hash needs none of that and cannot fall out of
sync with a model trained against it. The cost is collisions, which at this
size are rare and which the encoder can learn around.
"""

from __future__ import annotations

import re

#: Feature buckets. Corpus prompts draw on a few hundred distinct words, so
#: this leaves collisions unlikely without making the embedding table large.
BUCKETS = 4096

#: Words carrying no information about a circuit. Dropping them shortens the
#: prompt and stops the commonest tokens dominating a bag-of-words average.
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "then", "there", "these", "this", "to", "with", "when", "which", "while"]
)

_WORD = re.compile(r"[a-z0-9]+|[!&|^¬∧∨⊕()]")


def words(prompt: str) -> list[str]:
    """Split a prompt into features.

    Operators are kept as their own tokens: a prompt that quotes the formal
    expression is one of the registers the paraphraser writes, and dropping
    the symbols would turn it into a bag of variable names.
    """
    return [w for w in _WORD.findall(prompt.lower()) if w not in STOPWORDS]


def bucket(word: str) -> int:
    """Stable feature index for a word.

    ``hash()`` is salted per process, so it cannot be used: a model trained on
    Monday would see different features on Tuesday. FNV-1a is the same hash the
    spec side uses for semantic identity, for the same reason.
    """
    h = 0xCBF29CE484222325
    for byte in word.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % BUCKETS


def encode_prompt(prompt: str, length: int = 32) -> list[int]:
    """Feature ids for one prompt, padded or truncated to ``length``.

    Index 0 is padding and never a real feature, so an empty prompt encodes to
    all-padding rather than to whatever word happens to hash to zero.
    """
    ids = [bucket(w) + 1 for w in words(prompt)][:length]
    return ids + [0] * (length - len(ids))


def encode_prompts(prompts, length: int = 32) -> list[list[int]]:
    return [encode_prompt(p, length) for p in prompts]
