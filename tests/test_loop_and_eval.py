"""The self-improvement loop and the evaluation harness.

The loop is the part of this project that is actually new, and it is tested
here against a *stub sampler* rather than a trained model. That is on purpose:
the contribution is the acceptance logic — what gets kept, what becomes a
repair pair, when the curriculum advances, and how collapse is detected — and
none of that needs a GPU to be wrong.
"""

from __future__ import annotations

import random

import pytest

from daedalus import vocab as V
from daedalus.data import corrupt, sample_unique
from daedalus.data.corpus import Example
from daedalus.eval import (
    ProceduralCompiler,
    PromptedLLM,
    Retrieval,
    Unconditional,
    grade,
    pareto_front,
    parse_ascii_layer,
    pass_at_k,
    per_class_accuracy,
    render_ascii_layer,
    summarise,
)
from daedalus.eval.metrics import SpecResult
from daedalus.grid import Grid
from daedalus.redsim import Fail, Malformed, Pass, RowMismatch, Verifier
from daedalus.spec import Spec
from daedalus.synth import compile as compile_spec
from daedalus.train.loop import (
    Accepted,
    LoopConfig,
    RoundReport,
    collapse_warning,
    run,
    run_round,
)


@pytest.fixture(scope="module")
def verifier():
    with Verifier() as v:
        yield v


@pytest.fixture(scope="module")
def worked(verifier):
    """A few specs with a known-good layout each."""
    rng = random.Random(4)
    out = []
    for spec in sample_unique(rng, 12):
        placed = spec.default_placement()
        attempt = compile_spec(spec, verifier, rng, attempts=12, fixed_placement=placed)
        if attempt.ok:
            out.append((spec, placed, attempt.grid.tokens()))
        if len(out) >= 3:
            break
    if not out:
        pytest.skip("the compiler produced nothing to build a loop test on")
    return out


class TestLoopAcceptance:
    def test_keeps_only_verified_samples(self, verifier, worked):
        # The stub emits one good layout and a pile of noise. Only the good one
        # may survive, however many candidates there are.
        specs = [(s, p) for s, p, _t in worked]
        good = {s.semantic_hash(): t for s, _p, t in worked}

        def sampler(placed, k):
            target = good[_lookup(specs, placed).semantic_hash()]
            noise = Grid.with_substrate().tokens()
            return [target] + [noise] * (k - 1)

        cfg = LoopConfig(candidates_per_spec=6, keep_per_spec=4)
        dataset, report = run_round(1, specs, sampler, verifier, cfg, set())
        assert report.pass_at_1 == 1.0
        assert dataset.accepted
        assert all(a.tokens in good.values() for a in dataset.accepted)

    def test_ranks_by_compactness_then_latency(self):
        # The central empirical claim depends on this ordering being right.
        a = Accepted(None, None, [], blocks=30, latency_rt=4)
        b = Accepted(None, None, [], blocks=30, latency_rt=2)
        c = Accepted(None, None, [], blocks=20, latency_rt=9)
        assert sorted([a, b, c], key=Accepted.cost) == [c, b, a]

    def test_duplicate_layouts_are_not_counted_as_diversity(self, verifier, worked):
        # A model that emits one layout sixty times has found one answer. Not
        # deduping is how a collapsing loop keeps looking healthy.
        specs = [(s, p) for s, p, _t in worked]
        good = {s.semantic_hash(): t for s, _p, t in worked}

        def sampler(placed, k):
            return [good[_lookup(specs, placed).semantic_hash()]] * k

        _dataset, report = run_round(
            1, specs, sampler, verifier, LoopConfig(candidates_per_spec=8), set()
        )
        assert report.layouts_per_spec == 1.0

    def test_novelty_is_measured_against_what_was_already_seen(self, verifier, worked):
        specs = [(s, p) for s, p, _t in worked]
        good = {s.semantic_hash(): t for s, _p, t in worked}
        known = {bytes(t) for t in good.values()}

        def sampler(placed, k):
            return [good[_lookup(specs, placed).semantic_hash()]] * k

        _d, report = run_round(
            1, specs, sampler, verifier, LoopConfig(candidates_per_spec=4), known
        )
        assert report.novelty == 0.0, "rediscovering the training set is not novelty"

    def test_only_single_row_misses_become_repair_pairs(self, verifier, worked):
        # Two or more wrong rows usually means a structural error that
        # inpainting a small region cannot fix; training on it teaches the model
        # to make small edits to fundamentally broken layouts.
        from daedalus.train.loop import _closest_miss

        one = Fail((RowMismatch(1, 0, 1),), None)
        two = Fail((RowMismatch(1, 0, 1), RowMismatch(2, 1, 0)), None)
        assert _closest_miss([[1], [2]], [one, two]) == [1]
        assert _closest_miss([[2]], [two]) is None
        assert _closest_miss([[1]], [Malformed("burnout", (0, 0, 0))]) is None

    def test_curriculum_advances_only_on_merit(self, verifier, worked):
        specs = [(s, p) for s, p, _t in worked]
        good = {s.semantic_hash(): t for s, _p, t in worked}
        seen_difficulty = []

        def source(rng, count, difficulty):
            seen_difficulty.append(difficulty)
            return specs

        def sampler(placed, k):
            return [good[_lookup(specs, placed).semantic_hash()]] * k

        cfg = LoopConfig(rounds=3, candidates_per_spec=2, advance_threshold=0.6)
        reports = run(sampler, verifier, source, cfg)
        assert len(reports) == 3
        # Every round passes, so the curriculum should climb each time.
        assert seen_difficulty[0] != seen_difficulty[-1]

    def test_curriculum_holds_when_the_model_is_failing(self, verifier, worked):
        specs = [(s, p) for s, p, _t in worked]
        seen_difficulty = []

        def source(rng, count, difficulty):
            seen_difficulty.append(difficulty)
            return specs

        def sampler(placed, k):
            return [Grid.with_substrate().tokens()] * k  # never passes

        run(sampler, verifier, source, LoopConfig(rounds=3, candidates_per_spec=2))
        assert len(set(seen_difficulty)) == 1, "a starving loop must not advance"


class TestCollapseDetection:
    def _report(self, r, pass1, diversity, novelty, unstable=0.0):
        return RoundReport(
            round=r,
            difficulty=(1, 2),
            specs=10,
            candidates=100,
            pass_at_1=pass1,
            pass_at_k=pass1,
            accepted=10,
            repairs=0,
            mean_blocks=30.0,
            mean_latency_rt=2.0,
            layouts_per_spec=diversity,
            novelty=novelty,
            unstable_rate=unstable,
            malformed_rate=0.0,
            fail_rate=0.0,
            seconds=1.0,
        )

    def test_rising_pass_rate_with_falling_diversity_is_flagged(self):
        reports = [
            self._report(1, 0.30, 8.0, 0.9),
            self._report(2, 0.50, 4.0, 0.5),
            self._report(3, 0.70, 1.2, 0.1),
        ]
        warning = collapse_warning(reports)
        assert warning and "collapsing" in warning

    def test_healthy_improvement_is_not_flagged(self):
        reports = [
            self._report(1, 0.30, 6.0, 0.9),
            self._report(2, 0.45, 6.5, 0.9),
            self._report(3, 0.60, 7.0, 0.88),
        ]
        assert collapse_warning(reports) is None

    def test_rising_oscillator_rate_is_flagged(self):
        reports = [
            self._report(1, 0.30, 6.0, 0.9, unstable=0.01),
            self._report(2, 0.35, 6.0, 0.9, unstable=0.05),
            self._report(3, 0.40, 6.0, 0.9, unstable=0.20),
        ]
        warning = collapse_warning(reports)
        assert warning and "unstable" in warning

    def test_needs_three_rounds_before_judging(self):
        assert collapse_warning([]) is None


class TestMetrics:
    def test_pass_at_k_uses_the_first_k_draws(self):
        r = SpecResult(verdicts=[None] * 10, passing=[7])
        assert pass_at_k([r], 1) == 0.0
        assert pass_at_k([r], 8) == 1.0

    def test_pareto_front_drops_dominated_points(self):
        assert pareto_front([(20, 3), (25, 3), (20, 5), (18, 6)]) == [(18, 6), (20, 3)]

    def test_per_class_accuracy_exposes_what_aggregate_hides(self):
        # A model that gets all the air right and every torch wrong reads well
        # in aggregate and is useless. Per class, it is obvious.
        predicted = [V.AIR] * 95 + [V.AIR] * 5
        target = [V.AIR] * 95 + [V.torch(V.Attach.WEST)] * 5
        acc = per_class_accuracy(predicted, target)
        assert acc[V.AIR] == 1.0
        assert acc[V.torch(V.Attach.WEST)] == 0.0

    def test_summarise_reports_failure_modes_separately(self):
        results = [
            SpecResult(verdicts=[Malformed("floating_dust", (0, 0, 0))], passing=[]),
            SpecResult(verdicts=[Pass(2, 30, (4, 1, 4))], passing=[0]),
        ]
        out = summarise(results, [[[0]], [[0]]])
        assert out["rate_malformed"] == 0.5
        assert out["rate_pass"] == 0.5
        assert out["pass@1"] == 0.5


class TestBaselines:
    def test_all_baselines_are_judged_by_the_same_verifier(self, verifier):
        rng = random.Random(6)
        specs = sample_unique(rng, 4)
        corpus = []
        for s in sample_unique(random.Random(77), 6):
            placed = s.default_placement()
            a = compile_spec(s, verifier, rng, attempts=10, fixed_placement=placed)
            if a.ok:
                corpus.append((s, a.grid.tokens(), placed.input_z, placed.output_z))

        methods = [
            ProceduralCompiler(verifier, rng, attempts=12),
            Retrieval(corpus),
            Unconditional(rng),
        ]
        scores = {}
        for method in methods:
            results, cands = [], []
            for s in specs:
                placed = s.default_placement()
                c = method(s, placed, 4)
                cands.append(c)
                results.append(grade(c, placed, verifier))
            scores[method.name] = summarise(results, cands)["pass@1"]

        # The compiler is correct by construction within its coverage, so it has
        # to lead. If it does not, the harness is measuring the wrong thing.
        assert scores["compiler"] >= scores["retrieval"]
        assert scores["compiler"] >= scores["unconditional"]

    def test_unconditional_honours_the_ports_it_was_given(self, verifier):
        # Otherwise the floor baseline fails as a port violation rather than as
        # a wrong circuit, and measures nothing.
        spec = Spec.parse("inputs A B\noutputs Q\nQ = A | B")
        placed = spec.default_placement()
        tokens = Unconditional(random.Random(0))(spec, placed, 1)[0]
        for x, y, z in placed.input_ports:
            assert V.decode(tokens[V.index(x, y, z)]).kind == "lever"
        for x, y, z in placed.output_ports:
            assert tokens[V.index(x, y, z)] == V.LAMP

    def test_llm_baseline_counts_unparseable_replies(self):
        # "The model emitted prose" is a result. Dropping those silently would
        # flatter the baseline.
        llm = PromptedLLM(call=lambda prompt: "Sure! Here is a redstone circuit:")
        spec = Spec.parse("inputs A B\noutputs Q\nQ = A & B")
        out = llm(spec, spec.default_placement(), 3)
        assert out == []
        assert llm.unparseable == 3

    def test_llm_prompt_states_the_grid_contract(self):
        llm = PromptedLLM(call=lambda p: "")
        spec = Spec.parse("inputs A B\noutputs Q\nQ = A & B")
        prompt = llm.prompt(spec, spec.default_placement())
        assert "16 lines of 16 characters" in prompt
        assert "Q = (A & B)" in prompt

    def test_ascii_parser_is_strict(self):
        with pytest.raises(ValueError):
            parse_ascii_layer("too short")
        with pytest.raises(ValueError):
            parse_ascii_layer("\n".join("." * 15 for _ in range(16)))

    def test_ascii_round_trip_preserves_the_logic_layer(self):
        grid = Grid.with_substrate()
        grid.set(4, V.LOGIC_Y, 4, V.WIRE)
        grid.set(5, V.LOGIC_Y, 4, V.SOLID)
        back = Grid.from_tokens(parse_ascii_layer(render_ascii_layer(grid)))
        assert back.get(4, V.LOGIC_Y, 4) == V.WIRE
        assert back.get(5, V.LOGIC_Y, 4) == V.SOLID


class TestRepairData:
    def test_corruption_never_touches_a_port(self, worked):
        # Knocking out a port makes the grid malformed, not broken, and would
        # measure a different thing entirely.
        spec, placed, tokens = worked[0]
        example = Example(
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
            blocks=1,
            bbox=[1, 1, 1],
            prompts=[],
        )
        rng = random.Random(0)
        for _ in range(20):
            damaged, hit = corrupt(example, rng, blocks=6)
            for x, y, z in placed.input_ports:
                assert damaged[V.index(x, y, z)] == tokens[V.index(x, y, z)]
            for x, y, z in placed.output_ports:
                assert damaged[V.index(x, y, z)] == tokens[V.index(x, y, z)]
            assert all(V.unindex(i)[1] >= V.LOGIC_Y for i in hit)

    def test_corruption_actually_breaks_something(self, verifier, worked):
        spec, placed, tokens = worked[0]
        example = Example(
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
            blocks=1,
            bbox=[1, 1, 1],
            prompts=[],
        )
        damaged, hit = corrupt(example, random.Random(3), blocks=8)
        assert hit
        assert not verifier.evaluate(damaged, placed).is_pass()


def _lookup(specs, placed):
    for spec, p in specs:
        if p is placed or p == placed:
            return spec
    raise AssertionError("stub sampler got a placement it did not recognise")
