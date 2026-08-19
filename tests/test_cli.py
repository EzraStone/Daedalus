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
