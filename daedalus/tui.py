"""The same window, in a terminal.

Identical shape to :mod:`daedalus.web`: a spec goes in, the placement attempts
stream past as they happen, and a verified circuit comes out. Different reason
to exist — no browser, no port, no second process, and it lives where the rest
of the work already is.

Both views read :mod:`daedalus.render` for glyphs and colours and
:func:`daedalus.synth.compile_attempts` for the work itself, so the only thing
that differs between them is the drawing.

The same honesty constraint applies as on the web page: this takes the formal
DSL, not English. There is no natural-language parser and no trained model.

``pip install 'daedalus[tui]'``, then ``daedalus tui``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TextArea,
)

from . import render
from . import vocab as V
from .redsim import Verifier, VerifierError
from .schematic import block_summary, write_litematic, write_schem
from .spec import Spec, SpecSyntaxError
from .synth import Stats, compile_attempts, scope_hint, stage_rank

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

DEFAULT_SPEC = "inputs A B\noutputs Q\nQ = !(A & B)"

#: The standing note. Same claim the web page makes, and it stays true until
#: there is a model behind this.
BANNER = (
    "Formal specs only — there is no natural-language parser and no trained "
    "model yet. This drives the procedural compiler and the Rust verifier."
)


def load_examples() -> list[tuple[str, str]]:
    """``(title, source)`` for the picker. Same files the web page serves."""
    out = []
    if EXAMPLES.is_dir():
        for path in sorted(EXAMPLES.glob("*.txt")):
            source = path.read_text()
            title = path.stem.replace("-", " ").replace("_", " ")
            for line in source.splitlines():
                if line.startswith("#"):
                    title = line.lstrip("# ").strip()
                    break
            out.append((title, source))
    return out



@dataclass(slots=True)
class Result:
    """What one compile run produced."""

    tokens: list[int] | None = None
    verdict: str = ""
    ok: bool = False
    #: Kept so the signal view can ask the verifier about this exact circuit.
    placed: object = None


class GridView(Static):
    """The logic layer, drawn as coloured cells.

    Two terminal columns per cell: a character is roughly twice as tall as it
    is wide, so a one-block cell would render the grid squashed and a circuit
    would not look like its shape in-game.

    ``layout=True`` is load-bearing. The widget is ``height: auto``, and a
    plain reactive only repaints — the height stays at whatever the placeholder
    measured at compose time, so sixteen rows of circuit get drawn into one
    row of space and the pane looks empty. Nothing in the widget's own output
    shows it; only a rendered screen does.
    """

    tokens: reactive[list[int] | None] = reactive(None, layout=True)
    y: reactive[int] = reactive(V.LOGIC_Y, layout=True)
    #: Dust strengths for one input assignment, or None to draw block kinds.
    dust: reactive[list[int] | None] = reactive(None, layout=True)

    def render(self) -> Text:
        if not self.tokens:
            return Text("no layout — nothing was routed for this spec", style="dim")
        out = Text()
        for z, row in enumerate(render.layer(self.tokens, self.y)):
            for x, cell in enumerate(row):
                if self.dust is not None and cell.kind == "wire":
                    # Strength in place of the glyph, coloured along the ramp.
                    # Where it reaches zero is where the circuit ran out of
                    # signal, which is the whole reason to look at this view.
                    level = self.dust[V.index(x, self.y, z)]
                    out.append(
                        f"{render.POWER_GLYPHS[max(0, min(level, 15))]} ",
                        style=f"{render.power_colour(level)} on {cell.background}",
                    )
                    continue
                out.append(
                    f"{cell.glyph} ", style=f"{cell.foreground} on {cell.background}"
                )
            out.append("\n")
        return out


class Legend(Static):
    def render(self) -> Text:
        out = Text()
        for kind in render.LEGEND:
            foreground, background = render.KIND_COLOUR[kind]
            out.append("  ", style=f"on {background}")
            out.append(f" {kind}  ", style=f"{foreground}")
        return out


class Verdict(Static):
    """The headline. Colour carries the outcome, text carries the detail.

    State lives on the widget rather than in whatever it last rendered:
    reading content back out of a Textual widget means depending on internals
    that move between versions, and it did.
    """

    STYLES = {
        "pass": "bold #4FBF8B",
        "fail": "bold #F0392C",
        "malformed": "bold #F0392C",
        "unstable": "bold #E0A82E",
        "": "dim",
    }

    kind: reactive[str] = reactive("")
    headline: reactive[str] = reactive("idle", layout=True)
    detail: reactive[str] = reactive("", layout=True)

    def show(self, kind: str, headline: str, detail: str = "") -> None:
        self.kind, self.headline, self.detail = kind, headline, detail

    @property
    def text(self) -> str:
        """Everything the widget is saying, for a caller that needs to check."""
        return f"{self.headline}\n{self.detail}".strip()

    def render(self) -> Text:
        out = Text(self.headline, style=self.STYLES.get(self.kind, "dim"))
        if self.detail:
            out.append("\n" + self.detail, style="dim")
        return out


class Detail(Static):
    """Spec facts, plus the materials list once there is a circuit.

    Assembled from stored parts rather than by appending to what is already on
    screen — the latter reads back rendered output, and a second run would
    stack a second copy onto the first.
    """

    spec_text: reactive[str] = reactive("", layout=True)
    materials: reactive[str] = reactive("", layout=True)

    def render(self) -> Text:
        out = Text.from_markup(self.spec_text) if self.spec_text else Text()
        if self.materials:
            out.append_text(Text.from_markup(f"\n\n[#77828F]MATERIALS[/]\n{self.materials}"))
        return out


class DaedalusApp(App):
    """Compile a spec and watch it happen."""

    CSS = """
    Screen { background: #0F1318; }
    #banner {
        background: #241E10; color: #C9BFA6; padding: 0 1;
        border-bottom: solid #29323D;
    }
    #body { height: 1fr; }
    #left { width: 42; border-right: solid #29323D; padding: 0 1; }
    #right { padding: 0 1; }
    .caption { color: #77828F; text-style: bold; margin-top: 1; }
    #source { height: 9; border: solid #29323D; background: #171C23; }
    #log { height: 1fr; min-height: 8; border: solid #29323D; background: #171C23; }
    #verdict { padding: 1; border: solid #29323D; margin-bottom: 1; }
    #grid { height: auto; }
    #legend { height: 1; margin-top: 1; color: #77828F; }
    #detail { height: auto; color: #A9B4C0; }
    Button { width: 100%; margin-top: 1; }
    Input { border: solid #29323D; background: #171C23; }
    Select { margin-bottom: 1; }
    .numbers { height: 3; }
    .numbers Input { width: 1fr; }
    """

    BINDINGS = [
        ("ctrl+r", "compile", "Compile"),
        ("ctrl+e", "export", "Export .schem"),
        # Bridges put dust two and three layers up, so a circuit that uses one
        # is not visible at all from the logic layer alone.
        ("[", "layer_down", "Layer -"),
        ("]", "layer_up", "Layer +"),
        # A verdict says a circuit is wrong; this says where the signal died.
        ("p", "power", "Signal"),
        ("ctrl+q", "quit", "Quit"),
    ]

    TITLE = "Daedalus"
    SUB_TITLE = "compile a spec into a verified redstone circuit"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.examples = load_examples()
        self.result = Result()
        self._power_row: int | None = None
        self._verifier: Verifier | None = None

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(BANNER, id="banner")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Label("SPECIFICATION", classes="caption")
                yield Select(
                    [(title, i) for i, (title, _s) in enumerate(self.examples)],
                    prompt="load an example",
                    id="example",
                )
                yield TextArea(DEFAULT_SPEC, id="source", language=None)
                with Horizontal(classes="numbers"):
                    yield Input("0", placeholder="seed", id="seed", type="integer")
                    yield Input(
                        "20", placeholder="attempts", id="attempts", type="integer"
                    )
                yield Button("Compile  (ctrl+r)", variant="primary", id="run")
                yield Label("PROGRESS", classes="caption")
                yield RichLog(id="log", wrap=True, markup=True)
            with VerticalScroll(id="right"):
                yield Verdict("idle", id="verdict")
                yield Label("LAYOUT", id="layer-caption", classes="caption")
                yield GridView(id="grid")
                yield Legend(id="legend")
                yield Detail(id="detail")
        yield Footer()

    def on_mount(self) -> None:
        try:
            from .redsim import find_binary

            self.log_line("info", "verifier", str(find_binary()))
        except VerifierError as e:
            self.query_one("#verdict", Verdict).show("malformed", "verifier not found", str(e))
            self.log_line("error", "verifier", str(e))
            self.query_one("#run", Button).disabled = True

    # -- helpers -----------------------------------------------------------

    def log_line(self, kind: str, label: str, detail: str = "", n: int | None = None) -> None:
        colour = {
            "ok": "#4FBF8B",
            "fail": "#E0A82E",
            "budget": "#5FB3C4",
            "error": "#F0392C",
            "info": "#A9B4C0",
        }.get(kind, "#A9B4C0")
        prefix = f"[#4C5763]{n:>2}[/] " if n is not None else "   "
        line = f"{prefix}[{colour}]{label}[/]"
        if detail:
            line += f"\n     [#77828F]{detail}[/]"
        self.query_one("#log", RichLog).write(line)

    def verifier(self) -> Verifier:
        if self._verifier is None:
            self._verifier = Verifier()
            self._verifier.start()
        return self._verifier

    def layers(self) -> list[int]:
        """Layers worth showing: the ones with anything in them."""
        if not self.result.tokens:
            return [V.LOGIC_Y]
        return render.occupied_layers(self.result.tokens) or [V.LOGIC_Y]

    def show_layer(self, y: int) -> None:
        grid = self.query_one("#grid", GridView)
        grid.y = y
        occupied = self.layers()
        where = f"{occupied.index(y) + 1} of {len(occupied)}" if y in occupied else "empty"
        name = {V.SUBSTRATE_Y: "substrate", V.LOGIC_Y: "logic"}.get(y, "elevated")
        self.query_one("#layer-caption", Label).update(
            f"LAYOUT · {name.upper()} LAYER (y={y}) · {where} · [ ]"
        )

    def _step_layer(self, delta: int) -> None:
        occupied = self.layers()
        current = self.query_one("#grid", GridView).y
        # Step through occupied layers rather than all six: with a bridge in
        # play the interesting ones are y=1, 2 and 3, and paging through empty
        # layers to find them reads as the view being broken.
        i = occupied.index(current) if current in occupied else 0
        self.show_layer(occupied[max(0, min(i + delta, len(occupied) - 1))])

    def action_power(self) -> None:
        """Cycle through the truth table's rows as signal fields, then off.

        The rows are the point. Watching which cells go dark between "output
        high" and "output low" says more about what a circuit is doing than
        any single snapshot of it.
        """
        grid_view = self.query_one("#grid", GridView)
        if not self.result.ok or not self.result.tokens or self.result.placed is None:
            self.log_line("error", "signal", "nothing verified to probe")
            return

        rows = 1 << len(self.result.placed.input_z)
        self._power_row = 0 if self._power_row is None else self._power_row + 1
        if self._power_row >= rows:
            self._power_row = None
            grid_view.dust = None
            self.show_layer(grid_view.y)
            return

        from .grid import Grid

        try:
            field = self.verifier().power(
                Grid.from_tokens(self.result.tokens), self.result.placed, self._power_row
            )
        except VerifierError as e:
            self._power_row = None
            self.log_line("error", "signal", str(e))
            return
        grid_view.dust = field.dust
        self.show_layer(grid_view.y)
        self.log_line(
            "info",
            f"signal row {self._power_row}",
            f"outputs {field.outputs:b} · {field.reach()} cells carrying signal",
        )

    def action_layer_up(self) -> None:
        self._step_layer(1)

    def action_layer_down(self) -> None:
        self._step_layer(-1)

    def clear_result(self) -> None:
        """Wipe the previous run.

        Without this a spec that fails to route leaves the last successful
        circuit on screen, reading as if the failing spec had produced it.
        """
        self.result = Result()
        self._power_row = None
        grid = self.query_one("#grid", GridView)
        grid.tokens = None
        grid.dust = None
        self.show_layer(V.LOGIC_Y)
        detail = self.query_one("#detail", Detail)
        detail.spec_text = ""
        detail.materials = ""

    # -- actions -----------------------------------------------------------

    @on(Select.Changed, "#example")
    def choose_example(self, event: Select.Changed) -> None:
        if event.value is Select.BLANK:
            return
        self.query_one("#source", TextArea).text = self.examples[int(event.value)][1]

    @on(Button.Pressed, "#run")
    def press_run(self) -> None:
        self.action_compile()

    def action_compile(self) -> None:
        self.query_one("#log", RichLog).clear()
        self.clear_result()
        self.query_one("#verdict", Verdict).show("", "compiling…")
        self.query_one("#run", Button).disabled = True
        self.compile_worker(
            self.query_one("#source", TextArea).text,
            _int(self.query_one("#seed", Input).value, 0),
            max(1, min(_int(self.query_one("#attempts", Input).value, 20), 60)),
        )

    @work(thread=True, exclusive=True)
    def compile_worker(self, source: str, seed: int, attempts: int) -> None:
        """Compile off the UI thread, posting each step back as it happens.

        The compiler is synchronous and CPU-bound; running it inline would
        freeze the interface and every attempt would appear at once at the end,
        which is the experience this view exists to avoid.
        """
        try:
            spec = Spec.parse(source)
        except SpecSyntaxError as e:
            self.call_from_thread(self._finish_error, "parse", str(e))
            return

        placed = spec.default_placement(random.Random(seed))
        self.call_from_thread(self._show_parsed, spec, placed)

        stats = Stats()
        last = None
        failures: list[str] = []
        try:
            for n, attempt in enumerate(
                compile_attempts(
                    spec, self.verifier(), random.Random(seed), attempts, stats=stats
                ),
                start=1,
            ):
                last = attempt
                if not attempt.ok:
                    failures.append(attempt.stage)
                self.call_from_thread(self._show_attempt, n, attempt, spec)
                if attempt.ok:
                    break
        except VerifierError as e:
            self.call_from_thread(self._finish_error, "verifier", str(e))
            return

        self.call_from_thread(self._finish, last, stats, failures)

    # -- UI updates, all on the main thread --------------------------------

    def _show_parsed(self, spec: Spec, placed) -> None:
        from .synth.netlist import NetlistError, compile_netlist

        try:
            summary = compile_netlist(spec).summary()
        except NetlistError as e:
            summary = f"outside the v1 primitive set: {e}"
        self.log_line("info", "parsed", f"{spec.gates} gate(s) · {summary}")
        self.query_one("#detail", Detail).spec_text = (
            f"[#77828F]TRUTH TABLE[/]\n{spec.table()}\n\n"
            f"[#77828F]CANONICAL[/]\n{spec.source()}\n\n"
            f"[#77828F]PORTS[/]\ninputs {' '.join(spec.inputs)} at rows "
            f"{', '.join(map(str, placed.input_z))}\n"
            f"outputs {' '.join(spec.outputs)} at rows "
            f"{', '.join(map(str, placed.output_z))}\n"
            f"spec hash {spec.key()}"
        )

    def _show_attempt(self, n: int, attempt, spec: Spec) -> None:
        if attempt.ok:
            self.log_line("ok", "verified", str(attempt.verdict), n)
            self.result = Result(
                attempt.grid.tokens(), str(attempt.verdict), True, attempt.placed
            )
            self.query_one("#grid", GridView).tokens = self.result.tokens
            self.show_layer(V.LOGIC_Y)
            self.query_one("#verdict", Verdict).show(
                "pass", str(attempt.verdict), f"attempt {n}"
            )
            self.query_one("#detail", Detail).materials = ", ".join(
                f"{k} x{v}" for k, v in block_summary(attempt.grid).items()
            )
        else:
            # A missed budget means the circuit works. Logging it in the same
            # colour as a routing failure hides the one distinction the
            # compiler goes out of its way to draw.
            if attempt.stage == "constraint":
                self.log_line("budget", "over budget", attempt.detail, n)
            else:
                self.log_line("fail", attempt.stage, attempt.detail, n)
            if attempt.grid is not None:
                self.query_one("#grid", GridView).tokens = attempt.grid.tokens()
        del spec

    def _finish(self, last, stats: Stats, failures: list[str]) -> None:
        self.query_one("#run", Button).disabled = False
        if last is not None and last.ok:
            return
        # The most informative failure, not the last one -- same choice
        # `compile` makes, so the pane and the command line agree.
        worst = max(failures, key=stage_rank, default="")
        hint = scope_hint(worst)
        self.query_one("#verdict", Verdict).show(
            "fail",
            "no verified layout",
            hint or f"{stats.as_dict()['attempts']} attempt(s), none verified",
        )
        if hint:
            self.log_line("info", "why", hint)

    def _finish_error(self, where: str, message: str) -> None:
        self.query_one("#run", Button).disabled = False
        self.log_line("error", where, message)
        self.query_one("#verdict", Verdict).show("malformed", f"{where} error", message)

    # -- export ------------------------------------------------------------

    def action_export(self) -> None:
        if not self.result.ok or not self.result.tokens:
            self.log_line("error", "export", "nothing verified to export")
            return
        from .grid import Grid

        path = Path.cwd() / "daedalus.schem"
        write_schem(Grid.from_tokens(self.result.tokens), path)
        self.log_line("ok", "exported", str(path))

    def action_export_litematic(self) -> None:
        if not self.result.ok or not self.result.tokens:
            self.log_line("error", "export", "nothing verified to export")
            return
        from .grid import Grid

        path = Path.cwd() / "daedalus.litematic"
        write_litematic(Grid.from_tokens(self.result.tokens), path)
        self.log_line("ok", "exported", str(path))

    def on_unmount(self) -> None:
        if self._verifier is not None:
            self._verifier.close()
            self._verifier = None


def _int(text: str, fallback: int) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        return fallback


def main() -> int:
    DaedalusApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
