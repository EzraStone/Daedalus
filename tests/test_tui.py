"""The terminal window.

Driven headless through Textual's own harness, which runs the real app and the
real compiler — the worker thread, the verifier subprocess, all of it. The
things worth asserting are the same three the web view got wrong before a
browser caught them: a run that succeeds shows a circuit, a run that fails
shows *no* circuit, and a bad spec does not wedge the app.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual", reason="the terminal window is an optional extra")

from textual.widgets import Button, Input, RichLog, TextArea  # noqa: E402

from daedalus import render  # noqa: E402
from daedalus import vocab as V  # noqa: E402
from daedalus.tui import (  # noqa: E402
    BANNER,
    DaedalusApp,
    Detail,
    GridView,
    Verdict,
    load_examples,
    scope_hint,
)

NAND = "inputs A B\noutputs Q\nQ = !(A & B)"
XOR = "inputs A B\noutputs Q\nQ = A ^ B"


async def compile_in(pilot, source: str, attempts: str = "25") -> None:
    """Type a spec, press Compile, and wait for the worker to finish."""
    pilot.app.query_one("#source", TextArea).text = source
    pilot.app.query_one("#attempts", Input).value = attempts
    await pilot.click("#run")
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


class TestGoldenPath:
    @pytest.mark.asyncio
    async def test_a_verified_circuit_is_drawn(self):
        async with DaedalusApp().run_test() as pilot:
            await compile_in(pilot, NAND)
            app = pilot.app
            assert app.result.ok, app.query_one("#log", RichLog)
            assert app.result.tokens is not None
            assert len(app.result.tokens) == V.CELLS
            assert app.query_one("#grid", GridView).tokens == app.result.tokens

    @pytest.mark.asyncio
    async def test_the_grid_renders_as_coloured_cells(self):
        async with DaedalusApp().run_test() as pilot:
            await compile_in(pilot, NAND)
            text = pilot.app.query_one("#grid", GridView).render()
            # 16 rows, and every cell carries a colour pair from render.py.
            assert str(text).count("\n") == V.SZ
            assert text.spans, "cells should be styled, not plain"

    @pytest.mark.asyncio
    async def test_the_panes_are_given_room_for_what_they_draw(self):
        # These widgets are height:auto, so a reactive that only repaints
        # leaves them the height their placeholder measured at compose time.
        # Everything below line one then gets clipped: render() looked right
        # while the screen showed an empty pane. Assert against the laid-out
        # size, which is the only place it shows.
        async with DaedalusApp().run_test(size=(140, 46)) as pilot:
            await compile_in(pilot, NAND)
            for selector, widget in (("#grid", GridView), ("#detail", Detail)):
                node = pilot.app.query_one(selector, widget)
                lines = str(node.render()).count("\n") + 1
                assert node.size.height == lines, selector

    @pytest.mark.asyncio
    async def test_the_spec_detail_is_shown_before_the_work_starts(self):
        async with DaedalusApp().run_test() as pilot:
            await compile_in(pilot, NAND)
            detail = str(pilot.app.query_one("#detail", Detail).render())
            assert "TRUTH TABLE" in detail
            assert "A B | Q" in detail
            assert "MATERIALS" in detail

    @pytest.mark.asyncio
    async def test_a_second_run_replaces_the_detail_rather_than_stacking(self):
        async with DaedalusApp().run_test() as pilot:
            await compile_in(pilot, NAND)
            await compile_in(pilot, NAND)
            detail = str(pilot.app.query_one("#detail", Detail).render())
            assert detail.count("MATERIALS") == 1
            assert detail.count("TRUTH TABLE") == 1

    @pytest.mark.asyncio
    async def test_export_writes_a_schematic(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        async with DaedalusApp().run_test() as pilot:
            await compile_in(pilot, NAND)
            assert pilot.app.result.ok
            pilot.app.action_export()
            await pilot.pause()
            assert (tmp_path / "daedalus.schem").stat().st_size > 0


class TestFailurePaths:
    @pytest.mark.asyncio
    async def test_a_failed_run_shows_no_circuit(self):
        # The bug the browser caught in the web view: a spec that fails to
        # route must not leave the previous circuit on screen.
        async with DaedalusApp().run_test() as pilot:
            await compile_in(pilot, NAND)
            assert pilot.app.result.ok

            await compile_in(pilot, XOR, attempts="3")
            assert not pilot.app.result.ok
            assert pilot.app.query_one("#grid", GridView).tokens is None
            assert "no layout" in str(pilot.app.query_one("#grid", GridView).render())

    @pytest.mark.asyncio
    async def test_a_routing_failure_explains_itself(self):
        async with DaedalusApp().run_test() as pilot:
            await compile_in(pilot, XOR, attempts="3")
            verdict = pilot.app.query_one("#verdict", Verdict).text
            assert "no verified layout" in verdict
            assert "wire crossing" in verdict

    @pytest.mark.asyncio
    async def test_a_bad_spec_does_not_wedge_the_app(self):
        async with DaedalusApp().run_test() as pilot:
            await compile_in(pilot, "inputs A\noutputs Q\nQ = @")
            assert not pilot.app.result.ok
            assert "parse error" in pilot.app.query_one("#verdict", Verdict).text
            # And the next run still works.
            await compile_in(pilot, NAND)
            assert pilot.app.result.ok

    @pytest.mark.asyncio
    async def test_export_refuses_when_nothing_verified(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        async with DaedalusApp().run_test() as pilot:
            await compile_in(pilot, XOR, attempts="3")
            pilot.app.action_export()
            await pilot.pause()
            assert not (tmp_path / "daedalus.schem").exists()

    @pytest.mark.asyncio
    async def test_the_run_button_comes_back(self):
        # Disabled during a run; a failure that left it disabled would make the
        # app look hung.
        async with DaedalusApp().run_test() as pilot:
            await compile_in(pilot, XOR, attempts="3")
            assert not pilot.app.query_one("#run", Button).disabled


class TestHonesty:
    def test_the_banner_does_not_promise_natural_language(self):
        assert "no natural-language parser" in BANNER
        assert "no trained model" in BANNER

    def test_examples_are_shared_with_the_web_view(self):
        titles = [t for t, _s in load_examples()]
        assert titles, "the picker needs something in it"
        # Same files, so a spec that teaches the DSL badly is fixed once.
        from daedalus.web.app import examples as web_examples

        assert titles == [e["title"] for e in web_examples()]

    def test_scope_hints_cover_every_failure_stage(self):
        for stage in ("routing", "placement", "signal", "netlist", "verify"):
            assert scope_hint(stage), f"{stage} should explain itself"
        assert scope_hint("ok") is None


class TestSharedRendering:
    """The two views must not drift into disagreeing about what a torch is."""

    def test_every_block_kind_has_a_glyph_and_a_colour(self):
        for token in V.BLOCK_TOKENS:
            cell = render.describe(token)
            assert cell.kind in render.KIND_GLYPH, cell.kind
            assert cell.kind in render.KIND_COLOUR, cell.kind
            assert cell.foreground.startswith("#")
            assert cell.background.startswith("#")

    def test_the_web_palette_comes_from_the_same_module(self):
        from daedalus.web.app import palette as web_palette

        served = web_palette()["blocks"]
        for token in V.BLOCK_TOKENS:
            assert served[str(token)]["glyph"] == render.describe(token).glyph

    def test_display_glyphs_are_not_the_ascii_ones(self):
        # V.glyph is the ASCII debug character and doubles as the format the
        # prompted-LLM baseline reads and writes, so it cannot be restyled.
        assert render.KIND_GLYPH["wire"] != V.glyph(V.WIRE)
        assert V.glyph(V.WIRE) == "d"

    def test_layer_is_a_full_slice(self):
        tokens = [V.AIR] * V.CELLS
        rows = render.layer(tokens, V.LOGIC_Y)
        assert len(rows) == V.SZ
        assert all(len(row) == V.SX for row in rows)

    def test_control_tokens_render_rather_than_raise(self):
        # A half-denoised diffusion sample is exactly the thing someone will
        # want to look at, so it must draw instead of blowing up.
        cell = render.describe(V.MASK)
        assert cell.kind == "air"
        assert "control" in cell.state

    def test_occupied_layers_skips_empties(self):
        tokens = [V.AIR] * V.CELLS
        assert render.occupied_layers(tokens) == []
        tokens[V.index(3, 2, 3)] = V.WIRE
        assert render.occupied_layers(tokens) == [2]
