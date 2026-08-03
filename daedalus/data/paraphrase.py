"""Natural-language views of a specification.

Language is generated *from* the spec, never the other way round. If a prompt
and a spec ever disagree, the spec is right — that is what makes the reward
signal exact.

§04 asks for four registers per spec, and the reason is that they fail
differently. A model trained only on terse names learns a lookup table from
"NAND gate" to a layout; one trained only on functional descriptions never sees
the words players actually use. So each spec gets a spread:

terse
    ``"NAND gate"`` — the name, if it has one.
functional
    ``"the light is off only when both switches are on"`` — behaviour, in terms
    of what you would see.
procedural
    ``"invert both inputs, then combine them"`` — how you would build it.
vague
    ``"a two-switch safety interlock"`` — what it is *for*, with no logic in it
    at all. This is the register real requests arrive in, and the one a model
    trained on the other three will be worst at.

Templates rather than an LLM by default. An LLM pass is a one-time cost and
§04 budgets for it, but it is also a dependency, an API key and a source of
irreproducibility, so :class:`LLMParaphraser` is a hook rather than the path.
"""

from __future__ import annotations

import random
from typing import Protocol

from ..spec import Spec

#: Behaviour fingerprints for the functions that have ordinary names.
#: Keyed by (n_inputs, rows) so the lookup is by meaning, not by source text.
_NAMED: dict[tuple[int, tuple[int, ...]], str] = {
    (1, (0, 1)): "buffer",
    (1, (1, 0)): "NOT gate",
    (2, (0, 0, 0, 1)): "AND gate",
    (2, (0, 1, 1, 1)): "OR gate",
    (2, (1, 1, 1, 0)): "NAND gate",
    (2, (1, 0, 0, 0)): "NOR gate",
    (2, (0, 1, 1, 0)): "XOR gate",
    (2, (1, 0, 0, 1)): "XNOR gate",
    (2, (1, 0, 1, 1)): "implication gate",
    (3, (0, 0, 0, 0, 0, 0, 0, 1)): "three-input AND gate",
    (3, (0, 1, 1, 1, 1, 1, 1, 1)): "three-input OR gate",
    (3, (1, 1, 1, 1, 1, 1, 1, 0)): "three-input NAND gate",
    (3, (1, 0, 0, 0, 0, 0, 0, 0)): "three-input NOR gate",
    (3, (0, 1, 1, 0, 1, 0, 0, 1)): "three-input XOR gate",
    (3, (0, 0, 0, 1, 0, 1, 1, 1)): "majority gate",
}

_SWITCH_WORDS = ("switch", "lever", "input", "button")
_LIGHT_WORDS = ("lamp", "light", "output", "bulb")

_VAGUE = (
    "a {n}-{sw} safety interlock",
    "a little control panel with {n} {sws}",
    "something that decides when the {lt} should come on, from {n} {sws}",
    "a {n}-way check before the {lt} lights",
    "the logic for a {n}-{sw} door control",
)


def _plural(word: str, n: int) -> str:
    """English plurals, for the handful of nouns this module uses.

    Naive ``word + "es"`` gives "buttones", which is the kind of detail that
    makes generated prompts read as generated. The natural-language split is
    supposed to look like something a person typed.
    """
    if n == 1:
        return word
    return word + ("es" if word.endswith(("s", "x", "ch", "sh")) else "s")


class Paraphraser(Protocol):
    def __call__(self, spec: Spec, rng: random.Random, count: int) -> list[str]: ...


def name_of(spec: Spec) -> str | None:
    """The ordinary name of this function, if it has one."""
    if spec.n_outputs != 1:
        return None
    return _NAMED.get((spec.n_inputs, spec.rows))


def _on_rows(spec: Spec) -> list[int]:
    return [m for m, row in enumerate(spec.rows) if row & 1]


def _describe_assignment(spec: Spec, m: int, sw: str) -> str:
    ons = [spec.inputs[k] for k in range(spec.n_inputs) if (m >> k) & 1]
    offs = [spec.inputs[k] for k in range(spec.n_inputs) if not (m >> k) & 1]
    parts = []
    if ons:
        parts.append(f"{_join(ons)} {'is' if len(ons) == 1 else 'are'} on")
    if offs:
        parts.append(f"{_join(offs)} {'is' if len(offs) == 1 else 'are'} off")
    del sw
    return " and ".join(parts)


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def functional(spec: Spec, rng: random.Random) -> str:
    """Describe what a player would see, not what the gates are."""
    sw, lt = rng.choice(_SWITCH_WORDS), rng.choice(_LIGHT_WORDS)
    ons = _on_rows(spec)
    total = len(spec.rows)
    if len(ons) == 1:
        return f"the {lt} turns on only when {_describe_assignment(spec, ons[0], sw)}"
    if len(ons) == total - 1:
        (off,) = [m for m in range(total) if m not in ons]
        return f"the {lt} stays on except when {_describe_assignment(spec, off, sw)}"
    if len(ons) <= 2:
        cases = " or when ".join(_describe_assignment(spec, m, sw) for m in ons)
        return f"the {lt} lights when {cases}"
    return (
        f"{spec.n_inputs} {_plural(sw, spec.n_inputs)} control one {lt}; it is on for "
        f"{len(ons)} of the {total} combinations"
    )


def procedural(spec: Spec, rng: random.Random) -> str:
    """Describe how you would build it."""
    from ..synth.netlist import compile_netlist

    try:
        netlist = compile_netlist(spec)
    except Exception:  # noqa: BLE001 - a description is never worth an exception
        return f"combine the {spec.n_inputs} inputs into one output"
    inverters, depth = netlist.n_inverters, netlist.depth()
    lt = rng.choice(_LIGHT_WORDS)
    if inverters == 0:
        return f"run every input onto the same wire and into the {lt}"
    if inverters == 1:
        return f"merge the inputs, then invert once with a torch to drive the {lt}"
    return (
        f"invert the inputs, merge them, and invert again — "
        f"{inverters} torches in {depth} stage{'s' if depth != 1 else ''}"
    )


def terse(spec: Spec, rng: random.Random) -> str:
    named = name_of(spec)
    if named:
        return named if rng.random() < 0.5 else named.replace(" gate", "").lower()
    return spec.source(ascii_only=True).splitlines()[-1].split("=", 1)[1].strip()


def vague(spec: Spec, rng: random.Random) -> str:
    template = rng.choice(_VAGUE)
    sw = rng.choice(_SWITCH_WORDS)
    return template.format(
        n=spec.n_inputs,
        sw=sw,
        sws=_plural(sw, spec.n_inputs),
        lt=rng.choice(_LIGHT_WORDS),
    )


REGISTERS = {
    "terse": terse,
    "functional": functional,
    "procedural": procedural,
    "vague": vague,
}


def paraphrase(spec: Spec, rng: random.Random, count: int = 6) -> list[str]:
    """Between four and eight phrasings, spread across the registers."""
    out: list[str] = []
    order = list(REGISTERS.values())
    rng.shuffle(order)
    while len(out) < count:
        for fn in order:
            if len(out) >= count:
                break
            text = fn(spec, rng)
            if text not in out:
                out.append(text)
        else:
            continue
        break
    # A second pass with fresh word choices fills the quota when the registers
    # collide on a very simple spec.
    guard = 0
    while len(out) < count and guard < count * 4:
        guard += 1
        text = rng.choice(order)(spec, rng)
        if text not in out:
            out.append(text)
    return out


class LLMParaphraser:
    """Hook for the one-time LLM pass §04 describes.

    Kept behind an explicit callable so the corpus builder never reaches for a
    network by accident. The cache is the point: paraphrasing is a fixed cost
    per spec, not per layout, and a corpus with ten layouts per spec would
    otherwise pay for it ten times.
    """

    def __init__(self, call, cache: dict[int, list[str]] | None = None):
        self.call = call
        self.cache: dict[int, list[str]] = cache if cache is not None else {}

    def __call__(self, spec: Spec, rng: random.Random, count: int = 6) -> list[str]:
        key = spec.semantic_hash()
        if key not in self.cache:
            self.cache[key] = list(self.call(spec, count))
        got = self.cache[key]
        # Fall back rather than fail: a missing API key should degrade the
        # corpus, not stop it being built.
        return got if got else paraphrase(spec, rng, count)
