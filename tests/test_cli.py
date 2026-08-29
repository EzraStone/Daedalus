"""The command line, which is how most people meet this project.

It had no tests at all. The checks here are deliberately shallow -- each
command's real work is covered where that work lives -- but they assert the
things only the CLI can get wrong: that a subcommand is wired to the function
it claims, that a bad spec exits non-zero instead of raising, and that
`train` and `sample` actually hand off to each other.
"""

from __future__ import annotations

import dataclasses
import json
import random

import pytest

from daedalus import vocab as V
from daedalus.cli import main
from daedalus.data.corpus import Example
from daedalus.spec import Spec

NAND = "inputs A B\noutputs Q\nQ = !(A & B)"


def write_corpus(path, n: int = 12):
    """A corpus on disk without paying for a real corpus build."""
    spec = Spec.parse(NAND)
    placed = spec.default_placement(random.Random(0))
    rng = random.Random(0)
    path.mkdir(parents=True, exist_ok=True)
    for name, count in (("train", n), ("val", 4)):
        lines = []
        for _ in range(count):
            tokens = [V.AIR] * V.CELLS
            for _ in range(20):
                tokens[V.index(rng.randrange(V.SX), V.LOGIC_Y, rng.randrange(V.SZ))] = V.WIRE
            lines.append(
                json.dumps(
                    dataclasses.asdict(
                        Example(
                            spec_source=spec.source(),
                            spec_hash=spec.key(),
                            gates=spec.gates,
                            n_inputs=spec.n_inputs,
                            n_outputs=spec.n_outputs,
                            rows=list(spec.rows),
                            input_z=list(placed.input_z),
                            output_z=list(placed.output_z),
                            tokens=tokens,
                            latency_rt=1,
                            blocks=20,
                            bbox=[16, 1, 16],
                            prompts=[],
                        )
                    )
                )
            )
        (path / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
    return path


class TestSurface:
    def test_version_is_reported(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert "daedalus" in capsys.readouterr().out

    def test_every_subcommand_is_reachable(self, capsys):
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        for name in ("compile", "verify", "corpus", "baselines", "train", "sample", "tui"):
            assert name in out, name

    def test_no_subcommand_is_an_error(self):
        with pytest.raises(SystemExit) as exit_info:
            main([])
        assert exit_info.value.code != 0


class TestCompile:
    def test_a_spec_compiles_and_exports(self, tmp_path):
        out = tmp_path / "nand.schem"
        assert main(["compile", NAND, "--out", str(out), "--attempts", "30"]) == 0
        assert out.stat().st_size > 0

    def test_a_bad_spec_exits_cleanly(self, capsys):
        # A syntax error is a user mistake, so it should read as one rather
        # than as a traceback.
        assert main(["compile", "inputs A\noutputs Q\nQ = @"]) == 2
        assert "bad spec" in capsys.readouterr().err


class TestTraining:
    def test_train_then_sample_round_trips(self, tmp_path, capsys):
        pytest.importorskip("torch", reason="training is an optional extra")
        corpus = write_corpus(tmp_path / "corpus")
        run = tmp_path / "run"

        assert main(
            ["train", str(corpus), "--tiny", "--epochs", "2", "--batch-size", "4",
             "--out", str(run)]
        ) == 0
        assert (run / "model.pt").exists()
        assert (run / "history.json").exists()
        assert "validation loss" in capsys.readouterr().out

        # An undertrained model verifies nothing, so a non-zero exit is the
        # honest outcome. What matters is that the checkpoint loads and every
        # candidate reaches the verifier.
        code = main(["sample", str(run / "model.pt"), NAND, "-k", "2", "--steps", "4"])
        assert code in (0, 1)
        assert "verified" in capsys.readouterr().out

    def test_training_history_records_the_comparable_metric(self, tmp_path):
        pytest.importorskip("torch", reason="training is an optional extra")
        corpus = write_corpus(tmp_path / "corpus")
        run = tmp_path / "run"
        main(["train", str(corpus), "--tiny", "--epochs", "2", "--batch-size", "4",
              "--out", str(run)])
        history = json.loads((run / "history.json").read_text())["history"]
        assert history and all("val_loss" in e for e in history)

    def test_an_empty_corpus_is_refused(self, tmp_path, capsys):
        pytest.importorskip("torch", reason="training is an optional extra")
        empty = tmp_path / "corpus"
        empty.mkdir()
        (empty / "train.jsonl").write_text("")
        assert main(["train", str(empty), "--tiny"]) == 1
        assert "no examples" in capsys.readouterr().err


class TestDoctor:
    def test_a_healthy_install_reports_ready(self, capsys):
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "verifier" in out and "compiler" in out
        assert "ready" in out

    def test_a_missing_extra_does_not_fail_the_check(self, capsys, monkeypatch):
        # An extra is a choice. Failing on one would tell a fresh clone its
        # install is broken when it is only minimal.
        monkeypatch.setattr("daedalus.cli._module_present", lambda name: False)
        assert main(["doctor"]) == 0
        assert "optional extras not installed" in capsys.readouterr().out

    def test_a_missing_verifier_is_fatal_and_says_how_to_fix_it(self, capsys, monkeypatch):
        from daedalus.redsim import VerifierError

        def absent(*a, **k):
            raise VerifierError("could not find the redsim binary")

        monkeypatch.setattr("daedalus.redsim.find_binary", absent)
        assert main(["doctor"]) == 1
        out = capsys.readouterr().out
        assert "cargo build --release -p redsim" in out


class TestVerifyRoundTrip:
    """`compile` and `verify` have to agree about where the ports are."""

    def test_a_compiled_layout_verifies(self, tmp_path):
        out = tmp_path / "layout.json"
        assert main(["compile", NAND, "--out", str(out), "--attempts", "30"]) == 0
        # Before the layout carried its port rows this returned 1 with a
        # port_violation: the compiler jitters the rows per attempt, so the
        # grid was being checked against a placement it was never built for.
        assert main(["verify", NAND, str(out)]) == 0

    def test_the_saved_layout_carries_its_port_rows(self, tmp_path):
        out = tmp_path / "layout.json"
        main(["compile", NAND, "--out", str(out), "--attempts", "30"])
        blob = json.loads(out.read_text())
        assert len(blob["tokens"]) == V.CELLS
        assert len(blob["input_z"]) == 2
        assert len(blob["output_z"]) == 1

    def test_a_bare_token_list_still_works_but_says_it_is_guessing(self, tmp_path, capsys):
        # The old format. It cannot be checked exactly, and silently reporting
        # a port violation for a working circuit is the worst way to say so.
        spec = Spec.parse(NAND)
        placed = spec.default_placement()
        out = tmp_path / "bare.json"
        built = tmp_path / "layout.json"
        main(["compile", NAND, "--out", str(built), "--attempts", "30"])
        out.write_text(json.dumps(json.loads(built.read_text())["tokens"]))
        main(["verify", NAND, str(out)])
        assert "no port rows" in capsys.readouterr().err
        del placed


class TestVerifyASchematic:
    """A circuit that has been through the game comes back without port rows."""

    def test_a_schematic_verifies_with_ports_read_off_the_faces(self, tmp_path, capsys):
        out = tmp_path / "c.schem"
        assert main(["compile", NAND, "--out", str(out), "--attempts", "30"]) == 0
        assert main(["verify", NAND, str(out)]) == 0
        printed = capsys.readouterr().out
        assert "ports: inputs at rows" in printed
        assert "Pass(" in printed

    def test_a_spec_with_the_wrong_port_count_says_so(self, tmp_path):
        # Verifying against the wrong spec should read as a mismatch, not as
        # a circuit that mysteriously fails.
        out = tmp_path / "c.schem"
        main(["compile", NAND, "--out", str(out), "--attempts", "30"])
        with pytest.raises(SystemExit, match="input"):
            main(["verify", "inputs A B C\noutputs Q\nQ = A & B & C", str(out)])


class TestPower:
    def test_it_draws_a_layer_per_assignment(self, tmp_path, capsys):
        layout = tmp_path / "p.json"
        main(["compile", NAND, "--out", str(layout), "--attempts", "30"])
        capsys.readouterr()
        assert main(["power", NAND, str(layout)]) == 0
        out = capsys.readouterr().out
        # One block per row of the truth table.
        assert out.count("game ticks") == 4
        assert "A=1 B=1  ->  Q=0" in out

    def test_a_single_assignment_can_be_asked_for(self, tmp_path, capsys):
        layout = tmp_path / "p.json"
        main(["compile", NAND, "--out", str(layout), "--attempts", "30"])
        capsys.readouterr()
        main(["power", NAND, str(layout), "--inputs", "3"])
        out = capsys.readouterr().out
        assert out.count("game ticks") == 1
        assert "A=1 B=1" in out


class TestPromptedSampling:
    def test_a_spec_only_checkpoint_refuses_a_prompt(self, tmp_path, capsys):
        # Silently ignoring it would be worse: the sample would look
        # conditioned and would not be.
        pytest.importorskip("torch", reason="training is an optional extra")
        corpus = write_corpus(tmp_path / "corpus")
        run = tmp_path / "run"
        main(["train", str(corpus), "--tiny", "--epochs", "1", "--batch-size", "4",
              "--out", str(run)])
        capsys.readouterr()
        assert main(["sample", str(run / "model.pt"), NAND, "-k", "1", "--steps", "2",
                     "--prompt", "turn it on"]) == 2
        assert "no prompt" in capsys.readouterr().err

    def test_a_prompted_checkpoint_accepts_one(self, tmp_path, capsys):
        pytest.importorskip("torch", reason="training is an optional extra")
        corpus = write_corpus(tmp_path / "corpus")
        run = tmp_path / "run"
        assert main(["train", str(corpus), "--tiny", "--nl-slots", "4", "--epochs", "1",
                     "--batch-size", "4", "--out", str(run)]) == 0
        capsys.readouterr()
        code = main(["sample", str(run / "model.pt"), NAND, "-k", "1", "--steps", "2",
                     "--prompt", "turn the lamp off when both levers are on"])
        assert code in (0, 1)
        out = capsys.readouterr().out
        assert "prompt: turn the lamp off" in out

    def test_the_slot_count_survives_a_checkpoint_round_trip(self, tmp_path):
        pytest.importorskip("torch", reason="training is an optional extra")
        from daedalus.train import load_checkpoint

        corpus = write_corpus(tmp_path / "corpus")
        run = tmp_path / "run"
        main(["train", str(corpus), "--tiny", "--nl-slots", "4", "--epochs", "1",
              "--batch-size", "4", "--out", str(run)])
        model = load_checkpoint(run / "model.pt", device="cpu")
        assert model.cfg.nl_slots == 4
        assert model.prompts is not None


class TestStaleBinaryCheck:
    """The failure a protocol bump creates, caught before anything runs."""

    def _fake_tree(self, tmp_path, binary_first: bool):
        import os

        src = tmp_path / "crates" / "redsim" / "src"
        src.mkdir(parents=True)
        binary = tmp_path / "target" / "release" / "redsim"
        binary.parent.mkdir(parents=True)
        source = src / "tick.rs"
        if binary_first:
            binary.write_text("")
            source.write_text("")
            os.utime(source, (2_000_000_000, 2_000_000_000))
        else:
            source.write_text("")
            binary.write_text("")
            os.utime(binary, (2_000_000_000, 2_000_000_000))
        return binary, source

    def _check(self, monkeypatch, tmp_path, binary):
        import daedalus.cli as cli

        # _newer_sources walks up from the module's own file to find the
        # checkout, so point it at the fake one.
        monkeypatch.setattr(cli, "__file__", str(tmp_path / "daedalus" / "cli.py"))
        return cli._newer_sources(binary)

    def test_a_source_newer_than_the_binary_is_named(self, tmp_path, monkeypatch):
        binary, source = self._fake_tree(tmp_path, binary_first=True)
        found = self._check(monkeypatch, tmp_path, binary)
        assert found is not None and found.name == source.name

    def test_a_fresh_binary_is_not_flagged(self, tmp_path, monkeypatch):
        binary, _ = self._fake_tree(tmp_path, binary_first=False)
        assert self._check(monkeypatch, tmp_path, binary) is None

    def test_an_installed_package_with_no_sources_is_not_flagged(self, tmp_path, monkeypatch):
        # Most installs have no crates/ directory. Reporting every one of them
        # as stale would make the check noise and train people to ignore it.
        binary = tmp_path / "redsim"
        binary.write_text("")
        assert self._check(monkeypatch, tmp_path, binary) is None

    def test_doctor_still_passes_on_this_checkout(self, capsys):
        # Guards the guard: a check that fires on a correctly built tree is
        # worse than no check.
        assert main(["doctor"]) == 0
        assert "MISS  verifier" not in capsys.readouterr().out


class TestBench:
    def test_compiler_mode_reports_where_the_time_goes(self, capsys):
        # The point of the mode. A verdict costs microseconds and a layout
        # costs milliseconds, and reporting only the first invites the reading
        # that the verifier is what makes corpus building slow.
        assert main(["bench", "--compiler", "--specs", "4", "--attempts", "3"]) == 0
        out = capsys.readouterr().out
        assert "specs/second" in out
        assert "of wall time" in out
        assert "netlist, placement and routing" in out
        assert "stages:" in out

    def test_compiler_mode_groups_failures_by_shape(self, capsys):
        # The stage alone says "routing" for problems needing different fixes,
        # and the raw detail says "net 3" and "net 5" for the same one. The
        # shape is what makes the breakdown countable.
        assert main(["bench", "--compiler", "--specs", "12", "--attempts", "3"]) == 0
        out = capsys.readouterr().out
        assert "yield by gate count" in out
        if "failure shapes:" in out:
            body = out.split("failure shapes:")[1]
            # Indices are replaced, so no shape carries a bare number.
            for line in body.strip().splitlines():
                if not line.strip():
                    break
                _count, shape = line.split(None, 1)
                assert not any(c.isdigit() for c in shape), shape

    def test_failure_shapes_collapse_the_indices(self):
        from daedalus.cli import _failure_shape

        a = _failure_shape("routing", "net 3: cannot reach inverter 1")
        b = _failure_shape("routing", "net 12: cannot reach inverter 7")
        assert a == b == "routing: net N: cannot reach inverter N"

    def test_a_failure_with_no_detail_still_has_a_shape(self):
        from daedalus.cli import _failure_shape

        assert _failure_shape("placement", "") == "placement: placement"

    def test_the_default_mode_still_measures_the_verifier(self, capsys):
        assert main(["bench", "--batch", "4", "--repeats", "2"]) == 0
        assert "evaluations/second" in capsys.readouterr().out


class TestRepairRate:
    def test_a_batch_reports_a_rate_rather_than_one_outcome(self, tmp_path, capsys):
        pytest.importorskip("torch", reason="training is an optional extra")
        corpus = write_corpus(tmp_path / "corpus")
        run = tmp_path / "run"
        main(["train", str(corpus), "--tiny", "--epochs", "1", "--batch-size", "4",
              "--out", str(run)])
        capsys.readouterr()
        # An undertrained model repairs nothing, so a non-zero exit is honest.
        code = main(["repair", str(run / "model.pt"), NAND, "--tasks", "2",
                     "-k", "1", "--steps", "2", "--attempts", "20"])
        assert code in (0, 1)
        out = capsys.readouterr().out
        assert "repaired" in out
        # The number that separates repair from regeneration has to be shown.
        assert "mean_cells_touched_outside" in out
