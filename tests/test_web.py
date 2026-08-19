"""The local web window.

Two things are worth testing here and the rest is plumbing:

* the streaming generator and the one-shot compiler must agree — if they ever
  diverge, the page would show something the CLI does not, and there would be
  no way to tell which one was lying;
* the page must not offer an export for a circuit that did not verify, because
  the whole point of the project is that nothing unverified gets treated as a
  result.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi", reason="the web window is an optional extra")
pytest.importorskip("httpx", reason="TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from daedalus import vocab as V  # noqa: E402
from daedalus.redsim import Pass, Verifier  # noqa: E402
from daedalus.spec import Spec  # noqa: E402
from daedalus.synth import compile as compile_spec  # noqa: E402
from daedalus.synth import compile_attempts  # noqa: E402
from daedalus.synth.place import Stats  # noqa: E402
from daedalus.web.app import app  # noqa: E402

NAND = "inputs A B\noutputs Q\nQ = !(A & B)"
# A documented scope gap: the netlist is a crossbar, which one layer cannot route.
XOR = "inputs A B\noutputs Q\nQ = A ^ B"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def verifier():
    with Verifier() as v:
        yield v


class TestStreamingMatchesOneShot:
    """`compile_attempts` is a refactor of `compile`, and has to stay one."""

    @pytest.mark.parametrize(
        "source",
        [
            NAND,
            "inputs A B\noutputs Q\nQ = A & B",
            "inputs A B\noutputs Q\nQ = A | B",
            "inputs A B C\noutputs Q\nQ = A & B & C",
            XOR,
        ],
    )
    def test_final_attempt_is_what_compile_returns(self, verifier, source):
        spec = Spec.parse(source)
        placed = spec.default_placement()

        streamed = None
        for attempt in compile_attempts(
            spec, verifier, random.Random(7), attempts=10, fixed_placement=placed
        ):
            streamed = attempt
            if attempt.ok:
                break
        one_shot = compile_spec(
            spec, verifier, random.Random(7), attempts=10, fixed_placement=placed
        )

        assert streamed is not None
        assert streamed.ok == one_shot.ok
        assert streamed.stage == one_shot.stage
        if streamed.ok:
            assert streamed.grid.tokens() == one_shot.grid.tokens()

    def test_stream_yields_the_failures_too(self, verifier):
        # The one-shot API throws away everything but the last result; the
        # stream keeps them, which is the entire reason it exists.
        spec = Spec.parse(XOR)
        attempts = list(
            compile_attempts(spec, verifier, random.Random(0), attempts=4)
        )
        assert len(attempts) > 1
        assert not any(a.ok for a in attempts)
        assert all(a.stage for a in attempts)

    def test_stops_at_the_first_success(self, verifier):
        spec = Spec.parse(NAND)
        attempts = list(
            compile_attempts(spec, verifier, random.Random(3), attempts=20)
        )
        assert attempts[-1].ok
        assert not any(a.ok for a in attempts[:-1]), "should not keep going after a pass"

    def test_stats_are_threaded_through(self, verifier):
        stats = Stats()
        list(compile_attempts(Spec.parse(NAND), verifier, random.Random(1), 5, stats=stats))
        assert stats.attempts >= 1


class TestRoutes:
    def test_index_serves_the_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Daedalus" in response.text

    def test_health_finds_the_verifier(self, client):
        body = client.get("/api/health").json()
        assert body["ok"], body
        assert Path(body["verifier"]).stem == "redsim"

    def test_examples_carry_usable_source(self, client):
        examples = client.get("/api/examples").json()
        assert examples, "the picker needs something in it"
        for example in examples:
            # Every example must actually parse; a broken one teaches the DSL wrong.
            Spec.parse(example["source"])
            assert example["title"]

    def test_palette_covers_the_vocabulary(self, client):
        body = client.get("/api/palette").json()
        assert len(body["blocks"]) == len(V.BLOCK_TOKENS)
        assert body["geometry"]["sx"] == V.SX
        # Served rather than duplicated in JS precisely so this can be asserted.
        assert body["blocks"][str(V.WIRE)]["kind"] == "wire"

    def test_compile_returns_a_verified_layout(self, client):
        body = client.post(
            "/api/compile", json={"spec_source": NAND, "seed": 0, "attempts": 25}
        ).json()
        assert body["ok"], body
        assert body["spec"]["rows"] == [1, 1, 1, 0]
        final = body["attempts"][-1]
        assert len(final["tokens"]) == V.CELLS
        assert final["blocks"] > 0
        assert "materials" in final

    def test_bad_spec_is_a_400_with_a_position(self, client):
        response = client.post(
            "/api/compile", json={"spec_source": "inputs A\noutputs Q\nQ = @", "seed": 0}
        )
        assert response.status_code == 400
        assert response.json()["where"] == "parse"

    def test_routing_failure_reports_a_retry_hint(self, client):
        body = client.post(
            "/api/compile", json={"spec_source": XOR, "seed": 0, "attempts": 4}
        ).json()
        assert not body["ok"]
        assert "wire crossing" in (body["hint"] or "")
        # Each failed attempt has to say what stage it died at.
        assert all(a["stage"] for a in body["attempts"])

    def test_attempts_budget_is_capped(self, client):
        # A browser tab should not be able to pin a core by typing a big number.
        response = client.post(
            "/api/compile", json={"spec_source": NAND, "attempts": 100_000}
        )
        assert response.status_code == 422


class TestExport:
    def test_export_writes_a_real_schematic(self, client):
        compiled = client.post(
            "/api/compile", json={"spec_source": NAND, "seed": 0, "attempts": 25}
        ).json()
        assert compiled["ok"]
        tokens = compiled["attempts"][-1]["tokens"]

        response = client.post("/api/export", json={"tokens": tokens, "fmt": "schem"})
        assert response.status_code == 200
        assert gzip.decompress(response.content)[:1] == b"\x0a"  # TAG_Compound

        response = client.post("/api/export", json={"tokens": tokens, "fmt": "litematic"})
        assert response.status_code == 200
        assert gzip.decompress(response.content)[:1] == b"\x0a"

    def test_export_rejects_a_wrong_sized_grid(self, client):
        response = client.post("/api/export", json={"tokens": [0, 1, 2], "fmt": "schem"})
        assert response.status_code == 400

    def test_export_rejects_an_unknown_format(self, client):
        response = client.post(
            "/api/export", json={"tokens": [0] * V.CELLS, "fmt": "png"}
        )
        assert response.status_code == 422


class TestStream:
    def test_websocket_reports_each_step_then_done(self, client):
        with client.websocket_connect("/api/compile/stream") as ws:
            ws.send_text(json.dumps({"spec_source": NAND, "seed": 0, "attempts": 25}))
            events = []
            while True:
                message = ws.receive_json()
                events.append(message)
                if message["event"] == "done":
                    break

        kinds = [e["event"] for e in events]
        assert kinds[0] == "parsed", "the spec should be echoed back before any work"
        assert "attempt" in kinds
        assert kinds[-1] == "done"
        assert events[-1]["ok"]
        # The parse event carries what the page needs to render before compiling.
        assert events[0]["spec"]["table"]

    def test_websocket_reports_a_parse_error_without_closing(self, client):
        with client.websocket_connect("/api/compile/stream") as ws:
            ws.send_text(json.dumps({"spec_source": "not a spec", "seed": 0}))
            first = ws.receive_json()
            assert first["event"] == "error"
            assert first["where"] == "parse"

            # The socket has to survive a bad spec: a typo should not require
            # the user to reload the page.
            ws.send_text(json.dumps({"spec_source": NAND, "seed": 0, "attempts": 25}))
            assert ws.receive_json()["event"] == "parsed"

    def test_websocket_reports_a_bad_payload(self, client):
        with client.websocket_connect("/api/compile/stream") as ws:
            ws.send_text("{ not json")
            message = ws.receive_json()
            assert message["event"] == "error"
            assert message["where"] == "request"


class TestPage:
    """Properties of the page itself that are easy to regress.

    Both of these were real bugs, found by driving a browser rather than by the
    TestClient — which is the argument for doing that at least once.
    """

    def test_a_failed_run_clears_the_previous_circuit(self, client):
        # A spec that fails to route used to leave the last successful circuit
        # on screen, which reads as if the failing spec produced it.
        page = client.get("/").text
        assert "function clearResult()" in page
        # Everything from `run()` up to the next top-level function.
        run_body = page.split("function run() {", 1)[1].split("\nasync function", 1)[0]
        assert "clearResult()" in run_body, "run() must wipe the previous result first"
        assert "no-grid" in page, "there should be an explicit empty state"

    def test_page_declares_a_favicon(self):
        # Not cosmetic: a missing one is a 404 on every load, which buries real
        # errors in the console.
        from daedalus.web.app import STATIC

        assert 'rel="icon"' in (STATIC / "index.html").read_text()

    def test_page_does_not_promise_natural_language(self, client):
        # The disclaimer is load-bearing. There is no NL parser and no trained
        # model; a page that implied otherwise would be the single most
        # misleading thing in the repository.
        page = client.get("/").text
        assert "no natural-language parser" in page
        assert "no trained model" in page


class TestSharedVerifier:
    def test_lock_serialises_access(self):
        # The verifier is a pipe with a request/response protocol; two
        # interleaved requests would desynchronise it permanently.
        import threading

        from daedalus.web.app import shared

        errors: list[Exception] = []

        def hammer():
            try:
                spec = Spec.parse(NAND)
                placed = spec.default_placement()
                for _ in range(3):
                    with shared as verifier:
                        result = compile_spec(
                            spec, verifier, random.Random(0), attempts=20,
                            fixed_placement=placed,
                        )
                    assert isinstance(result.verdict, Pass)
            except Exception as e:  # noqa: BLE001 - re-raised on the main thread
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors


class TestLayers:
    """A crossing bridge lives above the logic layer, so one slice is not enough."""

    def test_a_compile_reports_which_layers_are_occupied(self, client):
        response = client.post(
            "/api/compile",
            json={"spec_source": "inputs A B\noutputs Q\nQ = !(A & B)", "attempts": 30},
        )
        assert response.status_code == 200
        body = response.json()
        attempt = body["attempts"][-1]
        assert attempt["ok"], body
        # Substrate and logic at minimum; a bridged layout adds y=2 and y=3.
        assert V.SUBSTRATE_Y in attempt["layers"]
        assert V.LOGIC_Y in attempt["layers"]

    def test_layers_are_only_the_ones_with_something_in_them(self, client):
        response = client.post(
            "/api/compile",
            json={"spec_source": "inputs A B\noutputs Q\nQ = !(A & B)", "attempts": 30},
        )
        attempt = response.json()["attempts"][-1]
        tokens = attempt["tokens"]
        for y in range(V.SY):
            has = any(
                tokens[V.index(x, y, z)] != V.AIR for z in range(V.SZ) for x in range(V.SX)
            )
            assert (y in attempt["layers"]) == has, y
