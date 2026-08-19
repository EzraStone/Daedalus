"""Tests for the verifier process boundary."""

from pathlib import Path

import pytest

from daedalus.redsim import Verifier, VerifierError, _target_binaries


def test_cargo_binary_paths_use_the_windows_executable_suffix():
    release, debug = _target_binaries(Path("target"), os_name="nt")

    assert release == Path("target/release/redsim.exe")
    assert debug == Path("target/debug/redsim.exe")


def test_cargo_binary_paths_remain_extensionless_on_unix():
    release, debug = _target_binaries(Path("target"), os_name="posix")

    assert release == Path("target/release/redsim")
    assert debug == Path("target/debug/redsim")


class DribblingPipe:
    """A pipe that hands back one byte at a time, which a real one may do.

    The worker is spawned unbuffered, so its pipes are raw file objects: a
    read returns whatever has arrived rather than what was asked for, and a
    write consumes as much as it feels like. Both are legal, both are
    timing-dependent, and both were being treated as a dead worker.
    """

    def __init__(self, payload: bytes = b""):
        self.payload = payload
        self.written = bytearray()
        self.closed = False

    def read(self, n: int) -> bytes:
        chunk, self.payload = self.payload[:1], self.payload[1:]
        return chunk

    def write(self, data) -> int:
        self.written += bytes(data)[:1]
        return 1


class FakeProc:
    def __init__(self, payload: bytes = b""):
        self.stdin = DribblingPipe()
        self.stdout = DribblingPipe(payload)

    def poll(self):
        return None


def verifier_over(proc) -> Verifier:
    v = Verifier.__new__(Verifier)
    v._proc = proc
    return v


def test_a_short_read_is_reassembled_rather_than_reported_as_a_dead_worker():
    proc = FakeProc(b"abcd")
    assert verifier_over(proc)._read_exact(4) == b"abcd"


def test_a_truly_closed_pipe_still_raises():
    proc = FakeProc(b"ab")
    with pytest.raises(VerifierError, match="2/4 bytes"):
        verifier_over(proc)._read_exact(4)


def test_a_short_write_keeps_going_until_the_whole_request_is_out():
    # A batch of 64 grids is around 98 KB against a 64 KB pipe, so a request
    # routinely does not fit in one write.
    proc = FakeProc()
    payload = bytes(range(256)) * 8
    verifier_over(proc)._write(payload)
    assert bytes(proc.stdin.written) == payload
