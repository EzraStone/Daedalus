"""A local window onto the compiler.

The CLI answers "did it work?" after the fact. This answers "what is it doing?"
while it happens — the spec as parsed, each placement attempt as it succeeds or
fails, and the verified grid at the end.

Two things it deliberately does **not** do:

* **It does not accept plain English.** There is no natural-language parser and
  no trained model, so a chat box would be a lie about what the system can do.
  Input is the same DSL the CLI takes; the example picker is there to make that
  approachable without overclaiming.
* **It does not run any model.** Everything here is the procedural compiler and
  the Rust verifier, both of which are finished and tested. Nothing on this page
  should give the impression the generative half exists yet.

Single user, local machine. There is no auth, no rate limiting and no
multi-tenancy, and it binds to loopback by default for exactly that reason.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .. import __version__
from .. import vocab as V
from ..grid import Grid
from ..redsim import Verifier, VerifierError
from ..render import LEGEND, occupied_layers, power_colour
from ..render import palette as display_palette
from ..schematic import block_summary, write_litematic, write_schem
from ..spec import PlacedSpec, Spec, SpecSyntaxError
from ..synth import Attempt, Stats, compile_attempts, stage_rank
from ..synth.netlist import NetlistError, compile_netlist

STATIC = Path(__file__).parent / "static"
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

#: Ceiling on the retry budget a request may ask for. The compiler is fast, but
#: a browser tab should not be able to pin a core for a minute by typing a big
#: number into a form field.
MAX_ATTEMPTS = 60


class SharedVerifier:
    """One worker process, serialised behind a lock.

    :class:`~daedalus.redsim.Verifier` is documented as not thread-safe — it is
    a pipe with a request/response protocol, and two interleaved requests would
    desynchronise it permanently. Since compiling is fast and this is a
    single-user tool, a lock is the right answer; a pool would be complexity
    bought for a problem nobody has.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._verifier: Verifier | None = None

    def __enter__(self) -> Verifier:
        self._lock.acquire()
        try:
            if self._verifier is None:
                self._verifier = Verifier()
                self._verifier.start()
            return self._verifier
        except Exception:
            self._lock.release()
            raise

    def __exit__(self, *exc) -> None:
        self._lock.release()

    def close(self) -> None:
        with self._lock:
            if self._verifier is not None:
                self._verifier.close()
                self._verifier = None


shared = SharedVerifier()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start with nothing and shut the worker down on the way out.

    The verifier is created lazily on first use rather than at startup, so the
    server comes up even when the Rust binary has not been built yet — and
    ``/api/health`` can say so plainly instead of the process refusing to boot.
    """
    yield
    shared.close()


app = FastAPI(title="Daedalus", version=__version__, lifespan=lifespan)


# --------------------------------------------------------------------------
# request/response shapes
# --------------------------------------------------------------------------


class CompileRequest(BaseModel):
    spec_source: str
    seed: int = 0
    attempts: int = Field(default=20, ge=1, le=MAX_ATTEMPTS)


class PowerRequest(BaseModel):
    tokens: list[int]
    input_z: list[int]
    output_z: list[int]
    spec_source: str
    assignment: int = 0


class ExportRequest(BaseModel):
    tokens: list[int]
    fmt: str = Field(default="schem", pattern="^(schem|litematic)$")
    name: str = "daedalus"


@dataclass(slots=True)
class Parsed:
    spec: Spec
    placed: PlacedSpec


def _parse(source: str, seed: int) -> Parsed:
    spec = Spec.parse(source)
    # Deterministic per seed so a rendered circuit can be reproduced from what
    # the page shows: same source, same seed, same layout.
    return Parsed(spec=spec, placed=spec.default_placement(random.Random(seed)))


def _spec_payload(parsed: Parsed) -> dict:
    spec = parsed.spec
    try:
        netlist = compile_netlist(spec)
        netlist_summary = netlist.summary()
    except NetlistError as e:
        netlist_summary = f"cannot be expressed in the v1 primitive set: {e}"
    return {
        "source": spec.source(),
        "ascii_source": spec.source(ascii_only=True),
        "table": spec.table(),
        "inputs": list(spec.inputs),
        "outputs": list(spec.outputs),
        "rows": list(spec.rows),
        "gates": spec.gates,
        "key": spec.key(),
        "netlist": netlist_summary,
        "input_z": list(parsed.placed.input_z),
        "output_z": list(parsed.placed.output_z),
        "constraints": spec.constraints.describe(),
    }


def _attempt_payload(n: int, attempt: Attempt) -> dict:
    payload: dict = {"n": n, "stage": attempt.stage, "detail": attempt.detail, "ok": attempt.ok}
    if attempt.placed is not None:
        # The placement this attempt was built with, not the one the spec
        # payload carries. Ports are re-rolled every attempt, so anything that
        # asks the verifier about this grid has to ask about these ports --
        # the spec's are a different circuit.
        payload["input_z"] = list(attempt.placed.input_z)
        payload["output_z"] = list(attempt.placed.output_z)
    if attempt.grid is not None:
        payload["tokens"] = attempt.grid.tokens()
        # A bridge puts dust on y=2 and y=3. Drawing only the logic layer hides
        # it completely, so the run reads as a circuit with a gap in it.
        payload["layers"] = occupied_layers(payload["tokens"])
    if attempt.verdict is not None:
        payload["verdict"] = str(attempt.verdict)
        payload["verdict_kind"] = attempt.verdict.kind
    if attempt.ok and attempt.grid is not None and attempt.verdict is not None:
        payload["blocks"] = attempt.verdict.blocks
        payload["latency_rt"] = attempt.verdict.latency_rt
        payload["bbox"] = list(attempt.verdict.bbox)
        payload["materials"] = block_summary(attempt.grid)
    return payload


def _worst_stage(attempts: list[dict]) -> str:
    """The stage of the most informative failure, matching what the CLI says.

    Both surfaces run the same compiler over the same spec, so they had better
    give the same advice. Reporting whichever attempt happened to be last means
    a run that built a working circuit two blocks over budget and then rerolled
    into a routing failure is told to try harder, when what it should be told is
    that the budget is the problem.
    """
    failed = [a for a in attempts if not a.get("ok")]
    if not failed:
        return ""
    return max(failed, key=lambda a: stage_rank(a.get("stage", ""))).get("stage", "")


def _scope_hint(stage: str) -> str | None:
    """Turn a failure stage into something a person can act on."""
    if stage in ("routing", "placement", "signal"):
        return (
            "Routing failures here are mostly structural rather than unlucky: "
            "the solved fraction of random specs barely moves between 3 attempts "
            "and 50. Another seed changes the port rows and sometimes helps, but "
            "a bigger attempt count usually will not."
        )
    if stage == "constraint":
        return (
            "The circuit computes the right function but misses a budget the "
            "spec declared. The placer does not aim for latency or size yet, so "
            "it is rerolling and hoping -- loosen the constraint, or raise the "
            "attempt count."
        )
    if stage == "netlist":
        return "The spec is outside the v1 primitive set entirely."
    if stage == "verify":
        return (
            "The placer produced a layout the verifier rejected. This is the "
            "verifier doing its job — roughly a quarter of routed layouts are "
            "discarded this way. Retrying usually finds a good one."
        )
    return None


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text())


@app.get("/api/health")
def health() -> dict:
    """Reports whether the verifier binary is actually reachable.

    Worth its own endpoint: "you forgot to run `cargo build --release`" is by
    far the most likely reason a fresh clone fails, and it should say so rather
    than surfacing as a mysterious error on first compile.
    """
    try:
        from ..redsim import find_binary

        return {"ok": True, "verifier": str(find_binary()), "version": __version__}
    except VerifierError as e:
        return {"ok": False, "error": str(e), "version": __version__}


@app.get("/api/examples")
def examples() -> list[dict]:
    """Example specs for the picker.

    These make the DSL learnable by showing it, which is the honest alternative
    to a text box that pretends to understand English.
    """
    out = []
    if EXAMPLES.is_dir():
        for path in sorted(EXAMPLES.glob("*.txt")):
            source = path.read_text()
            title = path.stem.replace("-", " ").replace("_", " ")
            for line in source.splitlines():
                if line.startswith("#"):
                    title = line.lstrip("# ").strip()
                    break
            out.append({"id": path.stem, "title": title, "source": source})
    return out


@app.post("/api/compile")
def compile_once(request: CompileRequest) -> JSONResponse:
    """Compile and return everything at once.

    The WebSocket is the interesting path; this exists so the API is usable
    from a script or `curl` without speaking WebSocket.
    """
    try:
        parsed = _parse(request.spec_source, request.seed)
    except SpecSyntaxError as e:
        return JSONResponse({"ok": False, "error": str(e), "where": "parse"}, status_code=400)

    stats = Stats()
    attempts: list[dict] = []
    with shared as verifier:
        for n, attempt in enumerate(
            compile_attempts(
                parsed.spec,
                verifier,
                random.Random(request.seed),
                attempts=request.attempts,
                stats=stats,
            ),
            start=1,
        ):
            attempts.append(_attempt_payload(n, attempt))
            if attempt.ok:
                break

    final = attempts[-1] if attempts else {}
    return JSONResponse(
        {
            "ok": bool(final.get("ok")),
            "spec": _spec_payload(parsed),
            "attempts": attempts,
            "stats": stats.as_dict(),
            "hint": None if final.get("ok") else _scope_hint(_worst_stage(attempts)),
        }
    )


@app.websocket("/api/compile/stream")
async def compile_stream(socket: WebSocket) -> None:
    """Push each step as it happens.

    The compiler is synchronous and CPU-bound, so it runs on a worker thread
    and hands results back through a queue. Running it inline would block the
    event loop and the page would receive every message at once at the end —
    which is exactly the experience this endpoint exists to avoid.
    """
    await socket.accept()
    try:
        while True:
            raw = await socket.receive_text()
            try:
                request = CompileRequest(**json.loads(raw))
            except Exception as e:  # noqa: BLE001 - any bad payload is the same to us
                await socket.send_json({"event": "error", "where": "request", "message": str(e)})
                continue
            await _run_stream(socket, request)
    except WebSocketDisconnect:
        return


async def _run_stream(socket: WebSocket, request: CompileRequest) -> None:
    try:
        parsed = _parse(request.spec_source, request.seed)
    except SpecSyntaxError as e:
        await socket.send_json(
            {"event": "error", "where": "parse", "message": str(e), "line": e.line, "column": e.column}
        )
        return

    await socket.send_json({"event": "parsed", "spec": _spec_payload(parsed)})

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stats = Stats()

    def work() -> None:
        try:
            with shared as verifier:
                for n, attempt in enumerate(
                    compile_attempts(
                        parsed.spec,
                        verifier,
                        random.Random(request.seed),
                        attempts=request.attempts,
                        stats=stats,
                    ),
                    start=1,
                ):
                    loop.call_soon_threadsafe(
                        queue.put_nowait, {"event": "attempt", **_attempt_payload(n, attempt)}
                    )
                    if attempt.ok:
                        break
        except VerifierError as e:
            loop.call_soon_threadsafe(
                queue.put_nowait, {"event": "error", "where": "verifier", "message": str(e)}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = loop.run_in_executor(None, work)
    last: dict | None = None
    seen: list[dict] = []
    while True:
        message = await queue.get()
        if message is None:
            break
        if message.get("event") == "attempt":
            last = message
            seen.append(message)
        await socket.send_json(message)
    await task

    await socket.send_json(
        {
            "event": "done",
            "ok": bool(last and last.get("ok")),
            "stats": stats.as_dict(),
            "hint": None if (last and last.get("ok")) else _scope_hint(_worst_stage(seen)),
        }
    )


@app.post("/api/power")
def power(req: PowerRequest) -> JSONResponse:
    """Settle the circuit for one input assignment and return the field.

    The page can already draw what a circuit *is*. This is what it *does*:
    which cells are carrying signal, how strong, and where it runs out.
    """
    try:
        spec = Spec.parse(req.spec_source)
    except SpecSyntaxError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    placed = spec.place(req.input_z, req.output_z)
    try:
        with shared as verifier:
            field = verifier.power(Grid.from_tokens(req.tokens), placed, req.assignment)
    except (VerifierError, ValueError) as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    return JSONResponse(
        {
            "dust": field.dust,
            "settled": field.settled,
            "game_ticks": field.game_ticks,
            "outputs": field.outputs,
            "reach": field.reach(),
            "inputs": [
                {"name": name, "on": bool(req.assignment >> k & 1)}
                for k, name in enumerate(spec.inputs)
            ],
            "lamps": [
                {"name": name, "on": bool(field.outputs >> j & 1)}
                for j, name in enumerate(spec.outputs)
            ],
        }
    )


@app.post("/api/export")
def export(request: ExportRequest) -> FileResponse:
    """Write a schematic and hand it back as a download."""
    if len(request.tokens) != V.CELLS:
        return JSONResponse(
            {"ok": False, "error": f"expected {V.CELLS} tokens, got {len(request.tokens)}"},
            status_code=400,
        )
    grid = Grid.from_tokens(request.tokens)
    suffix = ".litematic" if request.fmt == "litematic" else ".schem"
    safe = "".join(c for c in request.name if c.isalnum() or c in "-_") or "daedalus"

    # The file has to outlive this function: FileResponse streams it after we
    # return, so it cannot be a context manager. mkstemp gives a path that
    # survives, and the OS temp directory is the cleanup story.
    fd, name = tempfile.mkstemp(suffix=suffix, prefix="daedalus-")
    os.close(fd)
    path = Path(name)
    if request.fmt == "litematic":
        write_litematic(grid, path, name=safe)
    else:
        write_schem(grid, path, name=safe)
    return FileResponse(path, filename=f"{safe}{suffix}", media_type="application/octet-stream")


@app.get("/api/palette")
def palette() -> dict:
    """Glyph, colour and kind for every block token.

    Served from :mod:`daedalus.render` rather than duplicated in JavaScript.
    The terminal UI reads the same module, so the two views cannot drift into
    disagreeing about what a torch looks like — which is the sort of thing that
    goes wrong quietly, because nothing fails when it does.
    """
    return {
        "blocks": display_palette(),
        # One colour per dust strength, so the page draws signal the same way
        # the terminal does rather than inventing a second gradient.
        "power_ramp": [power_colour(n) for n in range(16)],
        "legend": list(LEGEND),
        "geometry": {"sx": V.SX, "sy": V.SY, "sz": V.SZ, "logic_y": V.LOGIC_Y},
    }
