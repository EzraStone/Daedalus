"""The socket protocol, driven end to end against a stand-in for the mod.

`GameClient.run_case` is the code path that runs ten thousand times when
somebody finally measures sim/game agreement, and it had no tests: the real
server needs Minecraft, so the client's whole request/response cycle was
unexercised. A fake that speaks the protocol the mod's HarnessServer speaks
covers all of it except the Java.

This is not a mock of the client. It is a second implementation of the
*server* side, written from the README and checked against HarnessServer.java,
so a change to either end that breaks the contract fails here.
"""

from __future__ import annotations

import base64
import json
import socket
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from compare import Case, GameClient, HarnessError  # noqa: E402

from daedalus.spec import Spec  # noqa: E402
from daedalus.synth import compile as compile_spec  # noqa: E402


class FakeHarness:
    """A server that answers the way the Fabric mod is documented to answer.

    ``rows`` and ``settled`` are what it will report; ``script`` overrides the
    reply for a given op entirely, which is how the protocol violations below
    are provoked.
    """

    def __init__(self, rows=None, settled=True, script=None):
        self.rows = rows
        self.settled = settled
        self.script = script or {}
        self.seen: list[dict] = []
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.address = self._sock.getsockname()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _reply(self, request: dict) -> str | None:
        op = request.get("op")
        if op in self.script:
            return self.script[op]
        if op == "place":
            return json.dumps({"id": request["id"], "placed": True})
        if op == "test":
            n_in, n_out = len(request["levers"]), len(request["lamps"])
            rows = self.rows
            if rows is None:
                rows = [
                    [(m >> k) & 1 for k in range(n_in)] + [0] * n_out
                    for m in range(1 << n_in)
                ]
            return json.dumps({"id": request["id"], "rows": rows, "settled": self.settled})
        return json.dumps({"error": f"unknown op {op}"})

    def _serve(self) -> None:
        try:
            connection, _ = self._sock.accept()
        except OSError:
            return
        with connection, connection.makefile("rwb") as stream:
            while True:
                line = stream.readline()
                if not line:
                    return
                request = json.loads(line)
                self.seen.append(request)
                reply = self._reply(request)
                if reply is None:
                    return  # hang up mid-conversation
                stream.write((reply + "\n").encode())
                stream.flush()

    def close(self) -> None:
        self._sock.close()


@pytest.fixture(scope="module")
def case(verifier_module) -> Case:
    import random

    spec = Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)")
    placed = spec.default_placement(random.Random(0))
    attempt = compile_spec(
        spec, verifier_module, random.Random(0), attempts=30, fixed_placement=placed
    )
    assert attempt.ok
    return Case("case-00000", spec, placed, attempt.grid.tokens())


@pytest.fixture(scope="module")
def verifier_module():
    from daedalus.redsim import Verifier

    with Verifier() as v:
        yield v


def drive(case: Case, **kwargs):
    harness = FakeHarness(**kwargs)
    try:
        with GameClient(*harness.address, timeout=10) as client:
            return client.run_case(case), harness
    finally:
        harness.close()


class TestHappyPath:
    def test_a_full_exchange_is_two_requests_in_order(self, case):
        (_rows, _settled), harness = drive(case)
        assert [r["op"] for r in harness.seen] == ["place", "test"]

    def test_the_schematic_it_sends_is_a_real_one(self, case):
        _result, harness = drive(case)
        blob = base64.b64decode(harness.seen[0]["schematic"])
        # Sponge .schem files are gzipped NBT; the mod parses them as such.
        assert blob[:2] == b"\x1f\x8b"

    def test_the_ports_it_sends_are_the_case_s_own(self, case):
        _result, harness = drive(case)
        request = harness.seen[1]
        assert [tuple(p) for p in request["levers"]] == list(case.placed.input_ports)
        assert [tuple(p) for p in request["lamps"]] == list(case.placed.output_ports)

    def test_rows_come_back_as_output_bitmasks(self, case):
        # The wire format is inputs-then-outputs per row; the comparison wants
        # one integer per row, so this is where a mixed-up split would land.
        n_in = len(case.placed.input_ports)
        rows = [[(m >> k) & 1 for k in range(n_in)] + [1] for m in range(4)]
        (got, _settled), _harness = drive(case, rows=rows)
        assert got == [1, 1, 1, 1]

    def test_a_row_of_zeroes_is_zero_not_missing(self, case):
        n_in = len(case.placed.input_ports)
        rows = [[(m >> k) & 1 for k in range(n_in)] + [0] for m in range(4)]
        (got, _settled), _harness = drive(case, rows=rows)
        assert got == [0, 0, 0, 0]

    def test_an_unsettled_sweep_is_carried_through(self, case):
        # A circuit still changing at the tick cap is real-game evidence of an
        # oscillator and has to reach the comparison, not be flattened to True.
        (_rows, settled), _harness = drive(case, settled=False)
        assert settled is False


class TestProtocolViolations:
    """Every one of these is a wrong answer that must not become a result."""

    def test_an_error_object_raises_rather_than_returning(self, case):
        with pytest.raises(HarnessError, match="no such world"):
            drive(case, script={"place": json.dumps({"error": "no such world"})})

    def test_a_place_that_is_not_acknowledged_is_refused(self, case):
        with pytest.raises(ConnectionError, match="acknowledge"):
            drive(case, script={"place": json.dumps({"id": "case-00000", "placed": False})})

    def test_a_reply_about_a_different_case_is_refused(self, case):
        # Two cases crossing on one connection would silently attribute one
        # circuit's behaviour to another.
        with pytest.raises(ConnectionError, match="acknowledge"):
            drive(case, script={"place": json.dumps({"id": "case-99999", "placed": True})})

    def test_a_test_reply_about_a_different_case_is_refused(self, case):
        with pytest.raises(ConnectionError, match="invalid test response"):
            drive(case, script={"test": json.dumps({"id": "elsewhere", "rows": []})})

    def test_a_row_of_the_wrong_width_is_refused(self, case):
        # Short a column means the input/output split is wrong, and every row
        # after it would be scored against the wrong bits.
        with pytest.raises(ConnectionError, match="malformed truth-table row"):
            drive(case, script={"test": json.dumps({"id": "case-00000", "rows": [[0, 0]]})})

    def test_a_non_object_response_is_refused(self, case):
        with pytest.raises(ConnectionError, match="non-object"):
            drive(case, script={"place": json.dumps([1, 2, 3])})

    def test_invalid_json_is_refused(self, case):
        with pytest.raises(ConnectionError, match="invalid JSON"):
            drive(case, script={"place": "{not json"})

    def test_a_hang_up_mid_conversation_is_refused(self, case):
        with pytest.raises(ConnectionError, match="closed the connection"):
            drive(case, script={"place": None})


class TestOneBadCaseDoesNotEndTheRun:
    """A ten thousand case sweep must survive a circuit the mod refuses."""

    def test_a_refused_case_is_unreachable_not_fatal(self, case, verifier_module):
        from compare import run

        harness = FakeHarness(script={"place": json.dumps({"error": "could not place"})})
        try:
            with GameClient(*harness.address, timeout=10) as client:
                report = run([case], verifier_module, client)
        finally:
            harness.close()
        assert report.unreachable == 1
        assert report.cases == 1
        assert "could not place" in report.examples[0]["error"]

    def test_the_error_is_recorded_against_the_case_that_caused_it(self, case, verifier_module):
        from compare import run

        harness = FakeHarness(script={"place": json.dumps({"error": "boom"})})
        try:
            with GameClient(*harness.address, timeout=10) as client:
                report = run([case], verifier_module, client)
        finally:
            harness.close()
        assert report.examples[0]["id"] == case.name


class TestSettlingIsTwoObservations:
    """redsim's resting state and the game's are different facts.

    They were one field. `run` overwrote the simulator's answer with the
    game's before anything compared them, so the most interesting thing this
    harness can find -- one side oscillating where the other does not -- was
    destroyed on the way in.
    """

    def _report(self, case, verifier, **kwargs):
        from compare import run

        harness = FakeHarness(**kwargs)
        try:
            with GameClient(*harness.address, timeout=10) as client:
                return run([case], verifier, client)
        finally:
            harness.close()

    def _matching_rows(self, case):
        n_in = len(case.placed.input_ports)
        return [
            [(m >> k) & 1 for k in range(n_in)] + [(bit >> 0) & 1]
            for m, bit in enumerate(case.spec.rows)
        ]

    def test_both_sides_settling_on_the_same_table_agrees(self, case, verifier_module):
        report = self._report(
            case, verifier_module, rows=self._matching_rows(case), settled=True
        )
        assert report.agreed == 1
        assert report.by_divergence == {}

    def test_the_same_table_with_the_game_oscillating_is_not_agreement(
        self, case, verifier_module
    ):
        # redsim settles, the game does not, and the rows line up anyway.
        # Scoring that as agreement inflates the headline number with exactly
        # the circuits most worth looking at.
        report = self._report(
            case, verifier_module, rows=self._matching_rows(case), settled=False
        )
        assert report.agreed == 0
        assert report.by_divergence == {"update-order": 1}

    def test_the_report_says_which_side_oscillated(self, case, verifier_module):
        report = self._report(
            case, verifier_module, rows=self._matching_rows(case), settled=False
        )
        assert report.examples[0]["settled"] == {"redsim": True, "game": False}

    def test_the_simulator_s_own_verdict_is_not_overwritten(self, case, verifier_module):
        # The field the game writes is a different one now, so a later reader
        # can still ask what redsim thought.
        report = self._report(
            case, verifier_module, rows=self._matching_rows(case), settled=False
        )
        assert report.cases == 1
        assert report.examples[0]["settled"]["redsim"] is True


class TestSelfCheck:
    """The offline end-to-end path, and the bug it found on its first run."""

    def test_a_hierarchical_case_name_becomes_one_filename(self):
        # Golden names are paths: "invariance/buffer/row08". Writing that under
        # a temporary directory made every slash a directory that did not
        # exist, so every golden case failed against a real server with a
        # FileNotFoundError that read like a Minecraft problem.
        from compare import _schematic_filename

        assert "/" not in _schematic_filename("invariance/buffer/row08")
        assert "\\" not in _schematic_filename("a\\b")
        assert _schematic_filename("golden-direct-repeater") == "golden-direct-repeater.schem"

    def test_distinct_names_stay_distinct(self):
        from compare import _schematic_filename

        from daedalus.spec import Spec as _Spec

        del _Spec
        names = ["and2/2_8_5", "and2/3_9_6", "buffer/row08/d1", "buffer/row08/d2"]
        assert len({_schematic_filename(n) for n in names}) == len(names)

    def test_the_echo_client_agrees_with_the_simulator_by_construction(
        self, case, verifier_module
    ):
        from compare import EchoClient, run

        report = run([case], verifier_module, EchoClient())
        assert report.cases == 1
        assert report.agreed == 1
        assert report.unreachable == 0

    def test_it_still_serialises_the_schematic(self, case, verifier_module, monkeypatch):
        # The point of echoing rather than short-circuiting the client: the
        # self-check has to exercise the same serialisation the real path uses,
        # or it stops being able to find the bug it just found.
        import compare

        calls = []
        real = compare.write_schem
        monkeypatch.setattr(
            compare, "write_schem", lambda grid, path: (calls.append(path), real(grid, path))[1]
        )
        compare.run([case], verifier_module, compare.EchoClient())
        assert len(calls) == 1
        assert str(calls[0]).endswith(".schem")


class TestRandomSuiteAndCache:
    """The paths --suite golden never touches.

    build_cases, the resumable corpus cache, and the report they produce. The
    self-check exercises all of them without a Minecraft server, which is the
    only way any of it gets covered at all.
    """

    def _run(self, tmp_path, *extra):
        import compare

        out = tmp_path / "report.json"
        code = compare.main(
            [
                "--suite", "random", "--cases", "4", "--seed", "3",
                "--self-check", "--out", str(out), *extra,
            ]
        )
        import json as _json

        return code, _json.loads(out.read_text())

    def test_random_cases_go_through_the_whole_pipeline(self, tmp_path):
        code, report = self._run(tmp_path)
        assert code == 0
        assert report["cases"] == 4
        assert report["agreed"] == report["checked"] == 4
        assert report["unreachable"] == 0

    def test_the_cache_is_written_and_then_reused(self, tmp_path):
        cache = tmp_path / "cache.jsonl"
        _code, first = self._run(tmp_path, "--corpus-cache", str(cache))
        assert cache.exists()
        # One metadata line plus one line per case.
        assert len(cache.read_text().splitlines()) == first["cases"] + 1

        before = cache.read_text()
        _code, second = self._run(tmp_path, "--corpus-cache", str(cache))
        assert second["cases"] == first["cases"]
        assert cache.read_text() == before, "a reused cache should not be rewritten"

    def test_a_cache_from_a_different_seed_is_refused(self, tmp_path):
        # Silently mixing two seeds' cases would make the sample something
        # nobody chose, and the agreement figure a number over an unknown set.
        import compare

        cache = tmp_path / "cache.jsonl"
        self._run(tmp_path, "--corpus-cache", str(cache))
        with pytest.raises(ValueError, match="seed"):
            compare.build_cases(2, seed=999, verifier=None, cache_path=cache)
