"""How a circuit looks, in one place.

There are three views of the same grid — the ASCII dump from
:meth:`daedalus.grid.Grid.render`, the web page, and the terminal UI — and
they were on course to each carry their own private idea of what a torch looks
like. That is the kind of duplication that drifts silently: a new block gets a
colour in one view, a different one in another, and nothing fails.

So the display vocabulary lives here and the views read it.

Note this is deliberately *not* :func:`daedalus.vocab.glyph`. That one is the
ASCII debug character (``d`` for dust, ``t`` for torch) and it has a second job:
it is the format the prompted-LLM baseline reads and writes, so it cannot be
changed for aesthetic reasons. These are for looking at.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import vocab as V

#: One display character per block kind. Air and solid are blank on purpose —
#: at grid scale, colour reads faster than a glyph, and leaving the commonest
#: two blocks unmarked is what lets the components stand out.
KIND_GLYPH = {
    "air": " ",
    "solid": " ",
    "wire": "·",
    "torch": "◆",
    "repeater": "▶",
    "comparator": "◆",
    "lever": "⌐",
    "lamp": "◍",
    "target": "◎",
    "observer": "◇",
}

#: ``(foreground, background)`` per kind, from the design spec's palette.
KIND_COLOUR = {
    "air": ("#3A424C", "#12161B"),
    "solid": ("#8894A2", "#39424E"),
    "wire": ("#F0392C", "#5A1A17"),
    "torch": ("#F0392C", "#40130F"),
    "repeater": ("#5FB3C4", "#2C333C"),
    "comparator": ("#E0A82E", "#2C333C"),
    "lever": ("#DDE4EB", "#2C333C"),
    "lamp": ("#221A05", "#F5C542"),
    "target": ("#A9B4C0", "#3A3040"),
    "observer": ("#77828F", "#2A313A"),
}

#: The kinds worth naming in a legend. Air and solid are self-evident; the rest
#: are what a reader actually has to decode.
LEGEND = ("solid", "wire", "torch", "repeater", "lever", "lamp")


@dataclass(frozen=True, slots=True)
class Cell:
    """One grid cell, ready to draw."""

    token: int
    kind: str
    glyph: str
    foreground: str
    background: str
    state: str
    x: int
    y: int
    z: int

    @property
    def label(self) -> str:
        """``(x, y, z) minecraft:block_state`` — for a tooltip or a status bar."""
        return f"({self.x}, {self.y}, {self.z})  {self.state}"


def describe(token: int, x: int = 0, y: int = 0, z: int = 0) -> Cell:
    try:
        kind = V.decode(token).kind
        state = V.state_string(token)
    except ValueError:
        # A control token in a grid body. Never valid, but a half-denoised
        # diffusion sample is exactly the thing someone will want to look at,
        # so render it rather than raising.
        kind, state = "air", f"control:{token}"
    foreground, background = KIND_COLOUR.get(kind, KIND_COLOUR["air"])
    return Cell(
        token=token,
        kind=kind,
        glyph=KIND_GLYPH.get(kind, "?"),
        foreground=foreground,
        background=background,
        state=state,
        x=x,
        y=y,
        z=z,
    )


def layer(tokens, y: int = V.LOGIC_Y) -> list[list[Cell]]:
    """One horizontal slice, as rows of cells.

    ``tokens`` is the flat ``y -> z -> x`` grid, so a layer is a contiguous
    span — which is the same property that makes the token order right for the
    model's local attention.
    """
    return [
        [describe(tokens[V.index(x, y, z)], x, y, z) for x in range(V.SX)]
        for z in range(V.SZ)
    ]


def occupied_layers(tokens) -> list[int]:
    """Layers with anything in them, so a viewer can skip empty ones."""
    return [
        y
        for y in range(V.SY)
        if any(
            tokens[V.index(x, y, z)] != V.AIR for z in range(V.SZ) for x in range(V.SX)
        )
    ]


def palette() -> dict:
    """The whole display vocabulary, for a client that renders it itself."""
    out = {}
    for token in V.BLOCK_TOKENS:
        cell = describe(token)
        out[str(token)] = {
            "kind": cell.kind,
            "glyph": cell.glyph,
            "ascii": V.glyph(token),
            "foreground": cell.foreground,
            "background": cell.background,
            "state": cell.state,
        }
    return out
