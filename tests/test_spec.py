"""The spec DSL, its canonical form and its hash."""

from __future__ import annotations

import random

import pytest

from daedalus.spec import PlacementError, Spec, SpecSyntaxError
from daedalus.spec.canon import irrelevant_inputs, is_constant, restrict
from daedalus.spec.dsl import gate_count, parse


def spec(rule: str, inputs: str = "A B", outputs: str = "Q") -> Spec:
    return Spec.parse(f"inputs {inputs}\noutputs {outputs}\n{rule}")


class TestParsing:
    def test_unicode_and_ascii_agree(self):
        a = spec("Q = ¬(A ∧ B)")
        b = spec("Q = !(A & B)")
        c = spec("Q = not (A and B)")
        assert a.rows == b.rows == c.rows == (1, 1, 1, 0)
        assert a.semantic_hash() == b.semantic_hash() == c.semantic_hash()

    def test_precedence_is_not_left_to_right(self):
        # not > and > xor > or
        assert spec("Q = A | B & A").rows == spec("Q = A | (B & A)").rows
        assert spec("Q = !A | B").rows == spec("Q = (!A) | B").rows
        three = spec("Q = A ^ B | C", inputs="A B C")
        assert three.rows == spec("Q = (A ^ B) | C", inputs="A B C").rows

    def test_x_is_usable_as_a_port_name(self):
        # `x` separates the two extents in `region <= 8 x 6`, but reserving it
        # globally would outlaw the most natural name for a boolean input.
        s = spec("o = x & y", inputs="x y", outputs="o")
        assert s.rows == (0, 0, 0, 1)

    def test_region_constraint_parses(self):
        s = Spec.parse("inputs A B\noutputs Q\nregion <= 8 x 6\nQ = A | B")
        assert s.constraints.max_region == (8, 6)

    def test_annotations_are_transparent_to_the_truth_table(self):
        assert spec("Q = delay(A & B, 3)").rows == spec("Q = A & B").rows
        assert spec("Q = strength(A | B, 9)").rows == spec("Q = A | B").rows

    def test_comments_and_blank_lines(self):
        s = Spec.parse("# a nand\ninputs A B\n\noutputs Q\nQ = !(A & B)  # inline\n")
        assert s.rows == (1, 1, 1, 0)

    @pytest.mark.parametrize(
        "source",
        [
            "inputs A\noutputs Q\nQ = B",  # undeclared name
            "inputs A B\noutputs Q\nQ = A",  # unused input
            "inputs A\noutputs Q\nQ = A & (",  # unbalanced
            "inputs A\noutputs Q Z\nQ = A",  # unassigned output
            "inputs A\noutputs A\nA = A",  # port is both
            "inputs A\noutputs Q\nQ = A\nQ = !A",  # assigned twice
            "outputs Q\nQ = A",  # no inputs
            "inputs A\noutputs Q\nR = A",  # rule for a non-output
            "inputs A A\noutputs Q\nQ = A",  # duplicate input
        ],
    )
    def test_rejects_malformed_specs(self, source):
        with pytest.raises(SpecSyntaxError):
            Spec.parse(source)

    def test_error_carries_a_position(self):
        with pytest.raises(SpecSyntaxError) as e:
            Spec.parse("inputs A\noutputs Q\nQ = @")
        assert e.value.line == 3

    def test_seven_inputs_is_too_many(self):
        rule = " | ".join("ABCDEFG")
        with pytest.raises(SpecSyntaxError):
            Spec.parse(f"inputs {' '.join('ABCDEFG')}\noutputs Q\nQ = {rule}")


class TestSemantics:
    def test_truth_table_bit_order(self):
        # input k is bit k of the row index; output j is bit j of the value.
        s = Spec.parse("inputs A B\noutputs P Q\nP = A\nQ = B")
        assert s.rows == (0b00, 0b01, 0b10, 0b11)

    def test_hash_ignores_naming(self):
        assert spec("Q = A & B").semantic_hash() == spec(
            "light = sw1 & sw2", inputs="sw1 sw2", outputs="light"
        ).semantic_hash()

    def test_hash_separates_behaviour(self):
        seen = {}
        for rule in ["Q = A & B", "Q = A | B", "Q = A ^ B", "Q = !(A & B)", "Q = A"]:
            if rule == "Q = A":
                continue  # would be rejected: B unused
            h = spec(rule).semantic_hash()
            assert h not in seen, f"{rule} collides with {seen.get(h)}"
            seen[h] = rule

    def test_gate_count_counts_operators(self):
        assert gate_count(parse("inputs A\noutputs Q\nQ = !A").rules["Q"]) == 1
        assert gate_count(parse("inputs A B\noutputs Q\nQ = A & B").rules["Q"]) == 1
        assert gate_count(parse("inputs A B C\noutputs Q\nQ = (A ^ B) & !C").rules["Q"]) == 3

    def test_syntactic_reference_is_not_functional_dependence(self):
        # `A & !A` mentions A and ignores it. A corpus that trusted the
        # syntax would route a wire that provably cannot matter.
        p = parse("inputs A B\noutputs Q\nQ = (A & !A) | B")
        assert irrelevant_inputs(p) == ["A"]
        assert not is_constant(spec("Q = A | B").rows)

    def test_restrict_holds_the_rest_low(self):
        p = parse("inputs A B\noutputs Q\nQ = A | B")
        assert restrict(p, ["A"]) == (0, 1)

    def test_table_text_is_readable(self):
        text = spec("Q = A & B").table()
        assert "A B | Q" in text
        assert text.strip().endswith("1 1 | 1")


class TestPlacement:
    def test_default_placement_is_deterministic(self):
        s = spec("Q = A & B")
        assert s.default_placement().input_z == s.default_placement().input_z

    def test_ports_land_on_the_fixed_faces(self):
        p = spec("Q = A & B").default_placement()
        assert all(x == 0 for x, _y, _z in p.input_ports)
        assert all(x == 15 for x, _y, _z in p.output_ports)

    def test_sampled_rows_keep_their_spacing(self):
        # Adjacent dust runs are one net. Two ports a single row apart would
        # merge into an OR nobody asked for.
        rng = random.Random(0)
        s = Spec.parse("inputs A B C D\noutputs Q\nQ = A | B | C | D")
        for _ in range(200):
            rows = sorted(s.default_placement(rng).input_z)
            assert all(b - a >= 2 for a, b in zip(rows, rows[1:])), rows
            assert rows[0] >= 1 and rows[-1] <= 14

    def test_sampling_actually_varies(self):
        rng = random.Random(0)
        s = spec("Q = A & B")
        seen = {s.default_placement(rng).input_z for _ in range(100)}
        assert len(seen) > 10, "port rows should be a real degree of freedom"

    def test_rejects_rows_that_are_too_close(self):
        s = spec("Q = A & B")
        with pytest.raises(PlacementError):
            s.place((4, 5), (8,))

    def test_rejects_wrong_number_of_rows(self):
        with pytest.raises(PlacementError):
            spec("Q = A & B").place((2,), (8,))


class TestUnsatisfiableConstraints:
    """Budgets nothing can meet, caught before anything is built."""

    def test_a_narrow_region_is_impossible_not_merely_tight(self):
        # Levers sit on the input face and lamps on the output face, so every
        # layout spans the full width. Any width budget under that is a spec
        # that cannot be built, not one that needs more attempts.
        spec = Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)\nregion <= 10 x 10")
        why = spec.constraints.unsatisfiable()
        assert why and "ports are pinned" in why

    def test_a_full_width_region_is_allowed(self):
        spec = Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)\nregion <= 16 x 12")
        assert spec.constraints.unsatisfiable() is None

    def test_a_footprint_below_what_the_ports_alone_need_is_impossible(self):
        # Two blocks per input, two per output, before any wiring at all. The
        # substrate is not part of this: material_blocks counts from y=1 up.
        spec = Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)\nfootprint <= 5")
        why = spec.constraints.unsatisfiable(spec)
        assert why and "ports" in why

    def test_an_ordinary_footprint_is_allowed(self):
        # 60-120 is the range the corpus sampler draws from, and every one of
        # those has to stay buildable.
        spec = Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)\nfootprint <= 60")
        assert spec.constraints.unsatisfiable(spec) is None

    def test_no_constraints_is_satisfiable(self):
        assert Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)").constraints.unsatisfiable() is None

    def test_the_sampler_never_writes_an_impossible_one(self):
        # The corpus would otherwise carry specs that burn a full retry budget
        # and can never yield an example.
        import random as _random

        from daedalus.data import sample_unique

        for spec in sample_unique(_random.Random(0), 40):
            assert spec.constraints.unsatisfiable(spec) is None, spec.source()


class TestMultiOutputSampling:
    """max_outputs was declared and never read.

    Anyone setting it to three got single-output specs and no complaint, so
    the entire corpus was one output wide -- a class the DSL, the netlist, the
    placer and the verifier all support and the training data never contained.
    """

    def _cfg(self, outputs):
        from dataclasses import replace

        from daedalus.data.sample import SampleConfig

        return replace(SampleConfig(), max_outputs=outputs)

    def test_the_default_is_still_one_output(self):
        # Every measured number in docs/benchmarks.md was taken on
        # single-output specs. The default has to keep them comparable.
        import random as _random

        from daedalus.data.sample import SampleConfig, sample_spec

        assert SampleConfig().max_outputs == 1
        rng = _random.Random(0)
        assert all(sample_spec(rng, SampleConfig()).n_outputs == 1 for _ in range(40))

    def test_raising_it_actually_produces_more_outputs(self):
        import random as _random

        from daedalus.data.sample import sample_spec

        rng = _random.Random(0)
        counts = {sample_spec(rng, self._cfg(3)).n_outputs for _ in range(60)}
        assert counts == {1, 2, 3}

    def test_it_never_exceeds_the_bound(self):
        import random as _random

        from daedalus.data.sample import sample_spec

        rng = _random.Random(1)
        assert all(sample_spec(rng, self._cfg(2)).n_outputs <= 2 for _ in range(60))

    def test_every_output_gets_a_distinct_name(self):
        import random as _random

        from daedalus.data.sample import sample_spec

        rng = _random.Random(2)
        for _ in range(40):
            spec = sample_spec(rng, self._cfg(3))
            assert len(set(spec.outputs)) == spec.n_outputs

    def test_multi_output_specs_are_still_satisfiable(self):
        import random as _random

        from daedalus.data import sample_unique

        for spec in sample_unique(_random.Random(0), 30, self._cfg(3)):
            assert spec.constraints.unsatisfiable(spec) is None, spec.source()

    def test_the_gate_budget_is_split_not_multiplied(self):
        # Giving every output the full budget would make a three-output spec
        # three times the circuit, which is a different difficulty axis from
        # the one the splits are stratified on.
        import random as _random

        from daedalus.data.sample import SampleConfig, sample_spec

        rng = _random.Random(4)
        for _ in range(40):
            spec = sample_spec(rng, self._cfg(3))
            assert spec.gates <= SampleConfig().max_gates, spec.source()

    def test_a_sampled_multi_output_spec_compiles(self):
        # The class is supported everywhere downstream; this is the check that
        # the sampler emits something the rest of the pipeline can take.
        import random as _random

        from daedalus.data import sample_unique
        from daedalus.redsim import Verifier
        from daedalus.synth import compile as compile_spec

        specs = [s for s in sample_unique(_random.Random(5), 40, self._cfg(3)) if s.n_outputs > 1]
        assert specs, "the sampler produced no multi-output spec to try"
        with Verifier() as verifier:
            assert any(
                compile_spec(s, verifier, _random.Random(i), attempts=10).ok
                for i, s in enumerate(specs[:12])
            )


class TestPromptEncoding:
    """Prompts into features, for the conditioning path that was only a slot."""

    def test_words_survive_and_noise_does_not(self):
        from daedalus.text import words

        got = words("4 buttons control one bulb; it is on for 8 of the 16 combinations")
        assert "buttons" in got and "combinations" in got
        # Stopwords carry nothing about a circuit and would dominate an average.
        assert "the" not in got and "of" not in got

    def test_operators_are_features_in_their_own_right(self):
        # One of the paraphraser's registers quotes the formal expression.
        # Dropping the symbols would reduce it to a bag of variable names.
        from daedalus.text import words

        got = words("((!D | !((C | !B))) ^ A)")
        assert "!" in got and "|" in got and "^" in got

    def test_the_same_word_always_lands_in_the_same_bucket(self):
        # Python salts hash() per process, so a model trained on Monday would
        # see different features on Tuesday. This must not use it.
        from daedalus.text import bucket

        assert bucket("lamp") == bucket("lamp")
        assert bucket("lamp") != bucket("lever")

    def test_a_prompt_is_padded_to_a_fixed_width(self):
        from daedalus.text import encode_prompt

        assert len(encode_prompt("turn the lamp on")) == 32
        assert len(encode_prompt("a " * 200)) == 32

    def test_padding_is_never_a_real_feature(self):
        # Index 0 means "nothing here". If a word could hash to it, an empty
        # prompt and a prompt containing that word would look the same.
        from daedalus.text import BUCKETS, bucket, encode_prompt

        assert set(encode_prompt("")) == {0}
        assert all(bucket(w) + 1 > 0 for w in ("lamp", "a", "0"))
        assert max(encode_prompt("lamp lever torch")) <= BUCKETS

    def test_real_corpus_prompts_encode(self):
        import random as _random

        from daedalus.data import sample_unique
        from daedalus.data.paraphrase import paraphrase
        from daedalus.text import encode_prompt

        rng = _random.Random(0)
        for spec in sample_unique(rng, 8):
            for prompt in paraphrase(spec, rng, 3):
                ids = encode_prompt(prompt)
                assert any(ids), prompt
