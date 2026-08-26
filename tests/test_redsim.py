"""Tests for the verifier process boundary."""

from pathlib import Path

import pytest

from daedalus import vocab as V
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


class TestConstraintDetail:
    """A missed budget should say by how much."""

    def _fail(self, source, attempts=20):
        import random

        from daedalus.spec import Spec
        from daedalus.synth import compile_attempts

        with Verifier() as v:
            for attempt in compile_attempts(Spec.parse(source), v, random.Random(0), attempts):
                if attempt.stage == "constraint" and attempt.verdict is not None:
                    return attempt.verdict
        return None

    def test_a_block_budget_reports_the_measurement_and_the_budget(self):
        # The numbers exist in the simulator; before this they were computed
        # and then dropped at the wire, leaving the caller with the word
        # "blocks" and nothing to act on.
        verdict = self._fail("inputs A B C\noutputs Q\nQ = !(A & B) | C\nfootprint <= 34")
        assert verdict is not None
        assert verdict.got[0] > 34
        assert verdict.budget[0] == 34
        assert "against a budget of 34" in verdict.overshoot()

    def test_a_latency_budget_carries_its_unit(self):
        verdict = self._fail("inputs A B\noutputs Q\nQ = !(A & B)\nlatency <= 1")
        assert verdict is not None
        assert "rt" in verdict.overshoot()

    def test_a_region_budget_reports_both_dimensions(self):
        verdict = self._fail("inputs A B\noutputs Q\nQ = !(A & B)\nregion <= 16 x 4")
        assert verdict is not None
        assert verdict.budget == (16, 4)
        assert "16x4" in verdict.overshoot()

    def test_a_plain_wrong_answer_has_nothing_to_overshoot(self):
        from daedalus.redsim import Fail, RowMismatch

        assert Fail((RowMismatch(0, 0, 1),), None).overshoot() is None


class TestPowerField:
    """Reading the settled signal, not just the verdict."""

    def build(self):
        import random

        from daedalus.spec import Spec
        from daedalus.synth import compile as compile_spec

        spec = Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)")
        with Verifier() as v:
            attempt = compile_spec(spec, v, random.Random(0), attempts=30)
            assert attempt.ok
            return attempt

    def test_the_field_agrees_with_the_truth_table(self):
        # If the levels disagreed with the verdict they would be describing a
        # different circuit, and the whole point is to explain this one.
        attempt = self.build()
        with Verifier() as v:
            got = [v.power(attempt.grid, attempt.placed, m).outputs for m in range(4)]
        assert got == [1, 1, 1, 0]

    def test_dust_strength_never_exceeds_fifteen(self):
        attempt = self.build()
        with Verifier() as v:
            field = v.power(attempt.grid, attempt.placed, 0)
        assert len(field.dust) == V.CELLS
        assert all(0 <= level <= 15 for level in field.dust)

    def test_only_dust_cells_carry_a_level(self):
        attempt = self.build()
        tokens = attempt.grid.tokens()
        with Verifier() as v:
            field = v.power(attempt.grid, attempt.placed, 3)
        for i, level in enumerate(field.dust):
            if level:
                assert tokens[i] == V.WIRE, V.unindex(i)

    def test_turning_the_output_off_lights_less_of_the_circuit(self):
        # The reading that makes this worth having: where the signal stopped.
        attempt = self.build()
        with Verifier() as v:
            on = v.power(attempt.grid, attempt.placed, 0)
            off = v.power(attempt.grid, attempt.placed, 3)
        assert on.outputs == 1 and off.outputs == 0
        assert off.reach() < on.reach()

    def test_a_working_circuit_settles(self):
        attempt = self.build()
        with Verifier() as v:
            field = v.power(attempt.grid, attempt.placed, 0)
        assert field.settled
        assert field.game_ticks > 0

    def test_a_grid_that_cannot_be_simulated_says_so(self):
        import pytest as _pytest

        from daedalus.grid import Grid

        attempt = self.build()
        broken = Grid.from_tokens(attempt.grid.tokens())
        broken.set(3, V.LOGIC_Y, 3, V.WIRE)  # dust with nothing under it
        broken.set(3, V.SUBSTRATE_Y, 3, V.AIR)
        with Verifier() as v, _pytest.raises(VerifierError, match="malformed"):
            v.power(broken, attempt.placed, 0)
