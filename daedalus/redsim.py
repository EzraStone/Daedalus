"""Client for the ``redsim`` verifier.

The Rust core runs as a long-lived subprocess and speaks a length-prefixed
binary protocol over a pipe. That is a deliberate choice over a PyO3 extension:
a pipe needs no ABI to keep in step, no compiler toolchain at import time, and
no build step between cloning the repository and running the tests. Batching
amortises the syscall to nothing — §06 evaluates 64 candidates per spec, and a
batch of 64 is one round trip.

Typical use::

    with Verifier() as v:
        verdicts = v.evaluate_batch([grid_a, grid_b], placed_spec)

The client is not thread-safe; give each worker thread its own
:class:`Verifier`, or share one behind a lock.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .grid import Grid
from .vocab import CELLS

MAGIC_REQ = b"RSIM"
MAGIC_RESP = b"RSOK"
PROTOCOL_VERSION = 2
OP_EVALUATE = 1
OP_POWER = 2

_REPO_ROOT = Path(__file__).resolve().parent.parent


class VerifierError(RuntimeError):
    """The worker could not be started, or spoke something unexpected."""


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RowMismatch:
    """One truth-table row that came out wrong."""

    inputs: int
    observed: int
    expected: int


#: Codes returned by the worker for :class:`Malformed`, in the order declared
#: by ``redsim::bin::malformed_code``.
MALFORMED_REASONS = (
    "floating_dust",
    "port_violation",
    "unsupported",
    "excluded_block",
    "masked_cell",
    "burnout",
    "history_dependent",
)

CONSTRAINT_NAMES = (None, "latency", "blocks", "region")


@dataclass(frozen=True, slots=True)
class Pass:
    latency_rt: int
    blocks: int
    bbox: tuple[int, int, int]

    kind = "pass"

    def is_pass(self) -> bool:
        return True

    def mismatch_count(self) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class Fail:
    mismatched_rows: tuple[RowMismatch, ...]
    constraint: str | None
    #: What the layout measured, and the budget it missed. Scalar budgets use
    #: the first component only; a region uses both, as (x, z). Absent when no
    #: constraint was violated.
    got: tuple[int, int] | None = None
    budget: tuple[int, int] | None = None

    kind = "fail"

    def is_pass(self) -> bool:
        return False

    def overshoot(self) -> str | None:
        """The miss in words, for a caller reporting a near miss."""
        if self.constraint is None or self.got is None or self.budget is None:
            return None
        if self.constraint == "region":
            return (
                f"region {self.got[0]}x{self.got[1]} against a budget of "
                f"{self.budget[0]}x{self.budget[1]}"
            )
        unit = {"latency": "rt", "blocks": "blocks"}.get(self.constraint, "")
        measured = f"{self.got[0]} {unit}".strip()
        return f"{measured} against a budget of {self.budget[0]}"

    def mismatch_count(self) -> int:
        return len(self.mismatched_rows) + (1 if self.constraint else 0)


@dataclass(frozen=True, slots=True)
class Unstable:
    """Oscillating. Tracked apart from :class:`Fail` because a rising unstable
    rate during self-training is an early collapse warning, not the same
    signal as a rising failure rate."""

    period_ticks: int

    kind = "unstable"

    def is_pass(self) -> bool:
        return False

    def mismatch_count(self) -> int:
        return 1 << 30


@dataclass(frozen=True, slots=True)
class Malformed:
    reason: str
    at: tuple[int, int, int]

    kind = "malformed"

    def is_pass(self) -> bool:
        return False

    def mismatch_count(self) -> int:
        return 1 << 30


@dataclass(frozen=True, slots=True)
class PowerField:
    """The settled signal strengths for one input assignment.

    ``dust`` is one strength per cell, ``0..=15``, and zero anywhere that is
    not dust. The interesting reading is not the number itself but where it
    reaches zero: dust loses a step of strength per block, so a run that goes
    dark fifteen cells from its source needed a repeater.
    """

    dust: list[int]
    settled: bool
    game_ticks: int
    #: Which output lamps came on, as a bitmask over the declared outputs.
    outputs: int

    def reach(self) -> int:
        """How many cells are carrying any signal at all."""
        return sum(1 for level in self.dust if level)


Verdict = Pass | Fail | Unstable | Malformed


# --------------------------------------------------------------------------
# locating the binary
# --------------------------------------------------------------------------


def _target_binaries(target: Path, os_name: str = os.name) -> tuple[Path, Path]:
    """Return Cargo's release and debug verifier paths for this platform.

    Cargo appends ``.exe`` to Windows binaries.  Keeping the platform detail
    here makes the discovery rule testable on every CI host instead of only on
    a Windows runner.
    """
    filename = "redsim.exe" if os_name == "nt" else "redsim"
    return target / "release" / filename, target / "debug" / filename


def find_binary(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate the ``redsim`` executable.

    Search order: an explicit path, ``$REDSIM_BIN``, the cargo release and
    debug target directories, then ``$PATH``. Release is preferred because the
    debug build is roughly twenty times slower and would quietly make every
    throughput number in the README wrong.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("REDSIM_BIN")
    if env:
        candidates.append(Path(env))
    target = Path(os.environ.get("CARGO_TARGET_DIR", _REPO_ROOT / "target"))
    candidates += _target_binaries(target)
    found = shutil.which("redsim")
    if found:
        candidates.append(Path(found))

    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    raise VerifierError(
        "could not find the redsim binary. Build it with "
        "`cargo build --release -p redsim`, or set $REDSIM_BIN."
    )


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------


class Verifier:
    """A handle on a running ``redsim`` worker."""

    def __init__(self, binary: str | os.PathLike[str] | None = None):
        self.binary = find_binary(binary)
        self._proc: subprocess.Popen[bytes] | None = None
        #: Total grids evaluated, so callers can report verifier throughput as
        #: the scaling axis it actually is.
        self.evaluated = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        try:
            self._proc = subprocess.Popen(
                [str(self.binary), "serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                bufsize=0,
            )
        except OSError as e:  # pragma: no cover - environment specific
            raise VerifierError(f"could not start {self.binary}: {e}") from e

    def close(self) -> None:
        """Shut the worker down and close both pipes.

        Closing stdin alone leaks the stdout descriptor. One leak is harmless;
        a training round that opens a verifier per worker thread runs out of
        descriptors, and it does so a long way from the cause.
        """
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:  # pragma: no cover - best effort teardown
            proc.kill()
            proc.wait(timeout=5)
        finally:
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None and not pipe.closed:
                    pipe.close()

    def __enter__(self) -> Verifier:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- protocol ----------------------------------------------------------

    def _write(self, data: bytes) -> None:
        """Write all of ``data``, however many syscalls that takes.

        The worker is started unbuffered, so this pipe is a raw file object
        and a single write is free to consume only part of the buffer. A
        request is comfortably larger than a pipe (a batch of 64 grids is
        ~98 KB against a 64 KB pipe), so the short write is the normal case
        rather than the rare one.
        """
        assert self._proc is not None and self._proc.stdin is not None
        view = memoryview(data)
        while view:
            written = self._proc.stdin.write(view)
            if not written:
                raise VerifierError("redsim worker stopped accepting input")
            view = view[written:]

    def _read_exact(self, n: int) -> bytes:
        """Read exactly ``n`` bytes, or explain why it could not.

        Same reason as the write side: an unbuffered read returns whatever
        has arrived, which is not necessarily what was asked for. Treating a
        short read as a dead worker misreports a perfectly healthy one, and
        does it intermittently, which is the worst way to find out.
        """
        assert self._proc is not None and self._proc.stdout is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                code = self._proc.poll()
                message = f"redsim worker closed the pipe after {len(buf)}/{n} bytes"
                if code is not None:
                    message += f" (exit {code})"
                if code not in (None, 0) and not buf:
                    # Dying before writing a single byte is what a worker does
                    # when it rejects the request outright, and by far the
                    # commonest reason is a binary older than this client.
                    # The worker says so on stderr, which is easy to miss and
                    # is swallowed entirely by a test runner or a web server.
                    message += (
                        f"\nThe worker refused the request before answering. This "
                        f"client speaks protocol {PROTOCOL_VERSION}; an older "
                        f"binary will not.\nRebuild it: cargo build --release -p redsim"
                    )
                raise VerifierError(message)
            buf += chunk
        return bytes(buf)

    def evaluate_batch(self, grids, spec) -> list[Verdict]:
        """Evaluate many candidate grids against one placed spec.

        ``grids`` may be :class:`~daedalus.grid.Grid` objects, ``bytes`` or any
        iterable of ``CELLS`` token ids.
        """
        payloads = [_as_payload(g) for g in grids]
        if not payloads:
            return []
        self.start()

        req = bytearray(MAGIC_REQ)
        req += bytes([PROTOCOL_VERSION, OP_EVALUATE])
        req += _encode_spec(spec)
        req += struct.pack("<I", len(payloads))
        for p in payloads:
            req += p
        self._write(bytes(req))

        magic = self._read_exact(4)
        if magic != MAGIC_RESP:
            raise VerifierError(f"bad response magic {magic!r}")
        (n,) = struct.unpack("<I", self._read_exact(4))
        if n != len(payloads):
            raise VerifierError(f"asked for {len(payloads)} verdicts, got {n}")
        out = [self._read_verdict() for _ in range(n)]
        self.evaluated += n
        return out

    def evaluate(self, grid, spec) -> Verdict:
        return self.evaluate_batch([grid], spec)[0]

    def power(self, grid, spec, assignment: int = 0) -> PowerField:
        """Settle the circuit with one input assignment and read the levels.

        A verdict says a circuit is wrong. This says *where the signal
        stopped*, which is the question anyone looking at a broken layout is
        actually asking. The simulator computes these levels on the way to
        every verdict already; this stops discarding them.

        ``assignment`` is a bitmask over the declared inputs, in order.
        """
        self.start()
        req = bytearray(MAGIC_REQ)
        req += bytes([PROTOCOL_VERSION, OP_POWER])
        req += _encode_spec(spec)
        req += struct.pack("<Q", assignment)
        req += _as_payload(grid)
        self._write(bytes(req))

        magic = self._read_exact(4)
        if magic != MAGIC_RESP:
            raise VerifierError(f"bad response magic {magic!r}")
        status = self._read_exact(1)[0]
        if status:
            raise VerifierError("this grid is too malformed to simulate")
        settled = bool(self._read_exact(1)[0])
        (ticks,) = struct.unpack("<I", self._read_exact(4))
        (outputs,) = struct.unpack("<Q", self._read_exact(8))
        dust = list(self._read_exact(CELLS))
        return PowerField(dust=dust, settled=settled, game_ticks=ticks, outputs=outputs)

    def _read_verdict(self) -> Verdict:
        kind = self._read_exact(1)[0]
        if kind == 0:
            latency, blocks, bx, by, bz = struct.unpack("<BHBBB", self._read_exact(6))
            return Pass(latency_rt=latency, blocks=blocks, bbox=(bx, by, bz))
        if kind == 1:
            (count,) = struct.unpack("<H", self._read_exact(2))
            rows = []
            for _ in range(count):
                inp, obs, exp = struct.unpack("<QQQ", self._read_exact(24))
                rows.append(RowMismatch(inp, obs, exp))
            code = self._read_exact(1)[0]
            if code >= len(CONSTRAINT_NAMES):
                raise VerifierError(f"unknown constraint code {code}")
            if code == 0:
                return Fail(tuple(rows), None)
            got_a, got_b, max_a, max_b = struct.unpack("<4I", self._read_exact(16))
            return Fail(tuple(rows), CONSTRAINT_NAMES[code], (got_a, got_b), (max_a, max_b))
        if kind == 2:
            return Unstable(period_ticks=self._read_exact(1)[0])
        if kind == 3:
            code, x, y, z = struct.unpack("<BBBB", self._read_exact(4))
            if code >= len(MALFORMED_REASONS):
                raise VerifierError(f"unknown malformed code {code}")
            return Malformed(MALFORMED_REASONS[code], (x, y, z))
        raise VerifierError(f"unknown verdict kind {kind}")


def _as_payload(g) -> bytes:
    if isinstance(g, Grid):
        return g.to_bytes()
    if isinstance(g, (bytes, bytearray, memoryview)):
        b = bytes(g)
    else:
        b = bytes(bytearray(g))
    if len(b) != CELLS:
        raise ValueError(f"grid needs exactly {CELLS} tokens, got {len(b)}")
    return b


def _encode_spec(spec) -> bytes:
    """Serialise a :class:`daedalus.spec.PlacedSpec` for the worker."""
    out = bytearray()
    out.append(len(spec.input_ports))
    for x, y, z in spec.input_ports:
        out += bytes((x, y, z))
    out.append(len(spec.output_ports))
    for x, y, z in spec.output_ports:
        out += bytes((x, y, z))
    rows = spec.rows
    out += struct.pack("<I", len(rows))
    for r in rows:
        out += struct.pack("<Q", r)

    c = spec.constraints
    flags = 0
    if c.max_latency_rt is not None:
        flags |= 1
    if c.max_blocks is not None:
        flags |= 2
    if c.max_region is not None:
        flags |= 4
    region = c.max_region or (0, 0)
    out.append(flags)
    out += struct.pack("<I", c.max_latency_rt or 0)
    out += struct.pack("<H", c.max_blocks or 0)
    out += bytes((region[0], region[1]))
    return bytes(out)


#: A module-level worker for scripts and notebooks that just want to call
#: `evaluate` without managing a context manager.
_shared: Verifier | None = None


def shared() -> Verifier:
    global _shared
    if _shared is None:
        _shared = Verifier()
        _shared.start()
    return _shared


def evaluate(grid, spec) -> Verdict:
    return shared().evaluate(grid, spec)


def evaluate_batch(grids, spec) -> list[Verdict]:
    return shared().evaluate_batch(grids, spec)
