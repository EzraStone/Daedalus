"""The two generators, exercised on CPU.

These models were written against §05 and then never run — every one of the
bugs asserted against below was live in code that imported cleanly and had a
full docstring. Nothing here is a quality measurement; a two-layer model on
random tokens learns nothing worth reporting. It asserts the plumbing: that a
step runs, that sampling produces block states and not spec tokens, and that
what comes out the far end is something the verifier will read.

Torch is optional, so the whole module skips without it.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("torch", reason="the generators are an optional extra")

import torch  # noqa: E402

from daedalus import tokens as T  # noqa: E402
from daedalus import vocab as V  # noqa: E402
from daedalus.grid import Grid  # noqa: E402
from daedalus.models import AutoregressiveModel, MaskedDiffusionModel  # noqa: E402
from daedalus.models.common import ModelConfig  # noqa: E402
from daedalus.redsim import Verifier  # noqa: E402
from daedalus.spec import Spec  # noqa: E402

#: Small enough to run in a second, wide enough to exercise every code path.
TINY = ModelConfig(n_layers=2, d_model=64, n_heads=4, d_ff=128)

BOTH = [("mdm", MaskedDiffusionModel), ("ar", AutoregressiveModel)]


def placed_nand():
    spec = Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)")
    return spec, spec.default_placement(random.Random(0))


def sample_once(cls, placed, seed: int = 0, **overrides):
    return sample_many(cls, placed, 1, seed, **overrides)[0]


def sample_many(cls, placed, n: int, seed: int = 0, **overrides):
    prefix, _slots = T.spec_prefix(placed)
    torch.manual_seed(seed)
    model = cls(TINY)
    model.eval()
    kwargs = {"steps": 8} if cls is MaskedDiffusionModel else {}
    kwargs.update(overrides)
    kwargs.setdefault("legality", T.legality_mask(placed))
    out = model.sample(
        torch.tensor([prefix] * n, dtype=torch.long),
        pinned=T.port_mask(placed),
        **kwargs,
    )
    return [row.tolist() for row in out]


def unsupported(tokens) -> int:
    """Blocks with nothing holding them up, by the verifier's own four rules."""
    from daedalus.models.common import _support_offset

    count = 0
    for i, token in enumerate(tokens):
        offset = _support_offset(token)
        if offset is None:
            continue
        x, y, z = V.unindex(i)
        nx, ny, nz = x + offset[0], y + offset[1], z + offset[2]
        if not V.in_bounds(nx, ny, nz) or not V.is_opaque(tokens[V.index(nx, ny, nz)]):
            count += 1
    return count


class TestTraining:
    @pytest.mark.parametrize("name,cls", BOTH)
    def test_a_step_runs_forward_and_backward(self, name, cls):
        torch.manual_seed(0)
        model = cls(TINY)
        tokens = torch.randint(0, V.CONTROL_BASE, (2, V.SEQ_LEN))
        loss = model.loss(tokens)
        assert torch.isfinite(loss), name
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, f"{name}: nothing received a gradient"
        assert all(torch.isfinite(g).all() for g in grads), name

    def test_the_smoke_run_decreases_the_objective(self):
        from daedalus.train.pretrain import smoke

        out = smoke(seed=0)
        assert out["last_loss"] < out["first_loss"]
        assert out["parameters"] > 0


class TestSampling:
    def test_autoregressive_decoding_starts_from_an_empty_body(self):
        # The body grows from zero cells, so a fixed-size position table makes
        # the very first decode step a shape error. Training never sees this
        # because it always passes a full grid.
        _spec, placed = placed_nand()
        body = sample_once(AutoregressiveModel, placed)
        assert len(body) == V.CELLS

    @pytest.mark.parametrize("name,cls", BOTH)
    def test_samples_are_block_states_not_spec_tokens(self, name, cls):
        # The head spans blocks *and* prefix tokens. If the legality mask is
        # only as wide as the block vocabulary the sampler can put a "<row>"
        # into a grid cell, which is not a block.
        _spec, placed = placed_nand()
        body = sample_once(cls, placed)
        assert max(body) < V.VOCAB_SIZE, f"{name}: emitted a prefix token"
        assert min(body) >= 0

    @pytest.mark.parametrize("name,cls", BOTH)
    def test_pinned_ports_are_left_alone(self, name, cls):
        _spec, placed = placed_nand()
        body = sample_once(cls, placed)
        for cell, token in T.port_mask(placed).items():
            assert body[cell] == token, f"{name}: resampled a pinned port"

    def test_the_legality_mask_lines_up_with_the_head(self):
        mask = T.legality_mask()
        assert len(mask) == V.CELLS
        assert all(len(row) == T.TOTAL_VOCAB for row in mask)
        # The prefix half can never be a block.
        assert all(not any(row[V.VOCAB_SIZE :]) for row in mask)

    def test_the_substrate_layer_admits_only_air_and_stone(self):
        mask = T.legality_mask()
        row = mask[V.index(4, V.SUBSTRATE_Y, 4)]
        assert row[V.AIR] and row[V.SOLID]
        assert not row[V.WIRE]


class TestSupport:
    """The neighbour-dependent half of legality, which a position-only mask cannot see."""

    def test_the_rules_match_the_verifier(self):
        from daedalus.models.common import _support_offset

        # redsim checks four: dust, repeaters and comparators sit on the block
        # below; torches and levers attach in the direction they carry.
        assert _support_offset(V.WIRE) == (0, -1, 0)
        assert _support_offset(V.repeater(V.Dir4.NORTH, 1)) == (0, -1, 0)
        assert _support_offset(V.comparator(V.Dir4.EAST, False)) == (0, -1, 0)
        assert _support_offset(V.torch(V.Attach.WEST)) == (-1, 0, 0)
        assert _support_offset(V.torch(V.Attach.FLOOR)) == (0, -1, 0)
        assert _support_offset(V.lever(V.Dir4.EAST)) == (1, 0, 0)
        # Everything else stands on its own.
        assert _support_offset(V.SOLID) is None
        assert _support_offset(V.LAMP) is None
        assert _support_offset(V.MASK) is None

    @pytest.mark.parametrize("name,cls", BOTH)
    def test_nothing_floats(self, name, cls):
        _spec, placed = placed_nand()
        for tokens in sample_many(cls, placed, 4):
            assert unsupported(tokens) == 0, name

    @pytest.mark.parametrize("name,cls", BOTH)
    def test_turning_it_off_lets_blocks_float(self, name, cls):
        # The ablation has to still work, and this is also the evidence that
        # the constraint above is doing something rather than being a no-op on
        # a model that happened to get it right.
        _spec, placed = placed_nand()
        loose = sum(
            unsupported(t) for t in sample_many(cls, placed, 4, enforce_support=False)
        )
        assert loose > 0, name

    def test_a_grid_survives_the_checks_that_run_before_simulation(self):
        # The point of the whole constraint. These four reasons are decided by
        # inspecting the grid, so a sample earning one is thrown out before it
        # is ever simulated and the verdict carries nothing the loop can rank
        # -- every candidate is equally, uninformatively bad. Getting past them
        # is what turns a sample into a scoreable one.
        #
        # Burnout is deliberately not in this list: a torch killing itself is
        # something the simulator discovers by running, which means the grid
        # was well-formed enough to run.
        static = {"unsupported", "floating_dust", "port_violation", "masked_cell"}
        _spec, placed = placed_nand()
        with Verifier() as v:
            verdicts = [
                v.evaluate(Grid.from_tokens(t), placed)
                for t in sample_many(MaskedDiffusionModel, placed, 4)
            ]
        offending = [x for x in verdicts if getattr(x, "reason", None) in static]
        assert not offending, offending[0]


class TestPortLegality:
    def test_levers_and_lamps_are_confined_to_declared_ports(self):
        _spec, placed = placed_nand()
        mask = T.legality_mask(placed)
        inputs = {V.index(*p) for p in placed.input_ports}
        outputs = {V.index(*p) for p in placed.output_ports}
        lever, lamp = V.lever(V.Dir4.EAST), V.LAMP
        for cell, row in enumerate(mask):
            assert row[lever] == (cell in inputs)
            assert row[lamp] == (cell in outputs)

    def test_without_a_spec_the_mask_is_position_only(self):
        # The spec-free form is still what the corpus and the ablations use.
        mask = T.legality_mask()
        assert any(row[V.LAMP] for row in mask)

    @pytest.mark.parametrize("name,cls", BOTH)
    def test_no_stray_ports_are_generated(self, name, cls):
        _spec, placed = placed_nand()
        allowed = set(T.port_mask(placed))
        for tokens in sample_many(cls, placed, 2):
            for i, token in enumerate(tokens):
                # A diffusion sample may still hold MASK where the model would
                # not commit, and that is not a block state to decode.
                if V.is_control(token):
                    continue
                if V.decode(token).kind in ("lever", "lamp"):
                    assert i in allowed, f"{name}: stray port at {V.unindex(i)}"


    @pytest.mark.parametrize("name,cls", BOTH)
    def test_every_declared_port_gets_its_block(self, name, cls):
        # The complement of the test above, and the half that was missing. No
        # stray ports says a lamp never appears where it should not; this says
        # one always appears where it must. Before the mask pinned port cells
        # in both directions, air was legal at an output port and a sample
        # could simply leave the lamp out.
        _spec, placed = placed_nand()
        fixed = T.port_mask(placed)
        for tokens in sample_many(cls, placed, 2):
            for cell, token in fixed.items():
                assert tokens[cell] == token, (
                    f"{name}: {V.unindex(cell)} holds "
                    f"{tokens[cell]} rather than the declared port block"
                )


class TestRepair:
    """Inpainting: the operation the diffusion model exists for.

    It was written, documented as the thing that "turns this from a demo into
    something a player would actually open", and then never called by anything
    -- not the evaluation harness, not the CLI, not a test.
    """

    def damaged_grid(self, placed, hurt: int = 8):
        tokens = [V.AIR] * V.CELLS
        for x in range(V.SX):
            for z in range(V.SZ):
                tokens[V.index(x, 0, z)] = V.SOLID
        for cell, token in T.port_mask(placed).items():
            tokens[cell] = token
        broken = [V.index(4 + i, V.LOGIC_Y, 4) for i in range(hurt)]
        for cell in broken:
            tokens[cell] = V.WIRE
        return tokens, broken

    def test_untouched_cells_are_left_exactly_alone(self):
        # The whole contract. If repair rewrites cells outside the damage it is
        # not repair, it is generation with extra steps.
        _spec, placed = placed_nand()
        tokens, broken = self.damaged_grid(placed)
        torch.manual_seed(0)
        model = MaskedDiffusionModel(TINY)
        model.eval()
        prefix, _slots = T.spec_prefix(placed)
        out = model.repair(
            torch.tensor([prefix], dtype=torch.long),
            torch.tensor([tokens], dtype=torch.long),
            broken,
            steps=6,
            legality=T.legality_mask(placed),
        )
        got = out[0].tolist()
        for i in range(V.CELLS):
            if i not in set(broken):
                assert got[i] == tokens[i], f"repair touched {V.unindex(i)}"

    def test_a_batched_prefix_gives_several_repairs_of_one_circuit(self):
        _spec, placed = placed_nand()
        tokens, broken = self.damaged_grid(placed)
        torch.manual_seed(0)
        model = MaskedDiffusionModel(TINY)
        model.eval()
        prefix, _slots = T.spec_prefix(placed)
        out = model.repair(
            torch.tensor([prefix] * 3, dtype=torch.long),
            torch.tensor([tokens], dtype=torch.long),
            broken,
            steps=6,
            legality=T.legality_mask(placed),
        )
        assert out.shape[0] == 3

    def test_it_accepts_a_plain_list_for_the_grid(self):
        _spec, placed = placed_nand()
        tokens, broken = self.damaged_grid(placed)
        model = MaskedDiffusionModel(TINY)
        model.eval()
        prefix, _slots = T.spec_prefix(placed)
        out = model.repair(
            torch.tensor([prefix], dtype=torch.long), tokens, broken, steps=4
        )
        assert out.shape[1] == V.CELLS

    def test_a_batch_of_grids_is_refused(self):
        # Ambiguous: it would silently repair only the first and report k
        # results, which reads as k repairs of k circuits.
        _spec, placed = placed_nand()
        tokens, broken = self.damaged_grid(placed)
        model = MaskedDiffusionModel(TINY)
        prefix, _slots = T.spec_prefix(placed)
        with pytest.raises(ValueError, match="one grid"):
            model.repair(
                torch.tensor([prefix] * 2, dtype=torch.long),
                torch.tensor([tokens, tokens], dtype=torch.long),
                broken,
                steps=2,
            )


class TestEndToEnd:
    @pytest.mark.parametrize("name,cls", BOTH)
    def test_the_verifier_reads_what_the_model_writes(self, name, cls):
        # The point of the whole exercise: a sample is a grid the verifier can
        # take a position on. An untrained model earns MALFORMED, and that is
        # a real verdict rather than a crash -- it is the reward signal doing
        # its job on a model that has learned nothing yet.
        _spec, placed = placed_nand()
        body = sample_once(cls, placed)
        with Verifier() as v:
            verdict = v.evaluate(Grid.from_tokens(body), placed)
        assert verdict is not None, name
        assert hasattr(verdict, "is_pass"), name


class TestPromptEncoder:
    """The other half of the conditioning path the prefix reserved slots for."""

    def encoder(self, slots=4):
        from daedalus.models import PromptEncoder

        torch.manual_seed(0)
        return PromptEncoder(d_model=TINY.d_model, slots=slots)

    def features(self, prompts):
        from daedalus.text import encode_prompts

        return torch.tensor(encode_prompts(prompts))

    def test_it_produces_one_vector_per_slot(self):
        out = self.encoder(slots=4)(self.features(["turn the lamp on", "invert both"]))
        assert out.shape == (2, 4, TINY.d_model)

    def test_an_empty_prompt_is_finite_rather_than_a_division_by_zero(self):
        # Pooling divides by the number of real words, and a prompt of pure
        # padding has none.
        out = self.encoder()(self.features([""]))
        assert torch.isfinite(out).all()

    def test_different_prompts_land_somewhere_different(self):
        out = self.encoder()(self.features(["turn the lamp on", "keep it off"]))
        assert (out[0] != out[1]).any()

    def test_padding_does_not_dilute_a_short_prompt(self):
        # Dividing by the padded width instead of the word count would make a
        # short prompt a quiet one, and length is not meaning.
        encoder = self.encoder()
        short = encoder(self.features(["lamp"]))
        padded = encoder(torch.tensor([[*self.features(["lamp"])[0].tolist()]]))
        assert torch.allclose(short, padded)

    def test_it_splices_into_the_prefix_the_body_reserved(self):
        # The end of the path: encoder output goes where spec_prefix marked.
        from daedalus.models.common import Body

        torch.manual_seed(0)
        body = Body(TINY, causal=False)
        tokens = torch.randint(0, V.CONTROL_BASE, (2, V.SEQ_LEN))
        vectors = self.encoder(slots=4)(self.features(["a", "b"]))
        with torch.no_grad():
            plain = body(tokens)
            conditioned = body(tokens, nl_embeddings=vectors)
        assert not torch.allclose(plain, conditioned), "the prompt changed nothing"

    def test_gradients_reach_the_word_embeddings(self):
        encoder = self.encoder()
        out = encoder(self.features(["turn the lamp on"]))
        out.sum().backward()
        grad = encoder.embed.weight.grad
        assert grad is not None and grad.abs().sum() > 0
        # Padding is held at zero, so it must never accumulate one.
        assert torch.equal(grad[0], torch.zeros_like(grad[0]))


class TestPromptGuidance:
    """Classifier-free guidance over the prompt, not just the spec prefix."""

    def model(self):
        from daedalus.models.common import ModelConfig as MC

        torch.manual_seed(0)
        m = MaskedDiffusionModel(MC(n_layers=2, d_model=64, n_heads=4, d_ff=128, nl_slots=4))
        m.eval()
        return m

    def vectors(self, model, text, n=2):
        from daedalus.text import encode_prompt

        return model.prompts(torch.tensor([encode_prompt(text)] * n))

    def test_guidance_changes_what_is_sampled(self):
        # If the unconditional branch were never taken, guidance would be a
        # no-op and the flag would be decoration.
        _spec, placed = placed_nand()
        model = self.model()
        prefix, _slots = T.spec_prefix(placed)
        nl = self.vectors(model, "turn the lamp off when both levers are on")
        out = []
        for guidance in (1.0, 4.0):
            torch.manual_seed(1)
            out.append(
                model.sample(
                    torch.tensor([prefix] * 2, dtype=torch.long),
                    steps=6,
                    guidance=guidance,
                    nl_embeddings=nl,
                    legality=T.legality_mask(placed),
                    pinned=T.port_mask(placed),
                )
            )
        assert not torch.equal(out[0], out[1])

    def test_guidance_without_a_prompt_is_inert(self):
        # No condition to guide away from, so it must not silently do
        # something arbitrary.
        _spec, placed = placed_nand()
        model = self.model()
        prefix, _slots = T.spec_prefix(placed)
        out = []
        for guidance in (1.0, 4.0):
            torch.manual_seed(1)
            out.append(
                model.sample(
                    torch.tensor([prefix], dtype=torch.long),
                    steps=4,
                    guidance=guidance,
                    legality=T.legality_mask(placed),
                    pinned=T.port_mask(placed),
                )
            )
        assert torch.equal(out[0], out[1])

    def test_the_blank_prompt_is_all_padding(self):
        # "No prompt" has to be the thing training taught with dropout, not
        # some other vector that happens to be handy.
        from daedalus.text import encode_prompt

        assert set(encode_prompt("")) == {0}

    def test_dropout_blanks_some_prompts_and_not_all(self):
        from daedalus.data.corpus import Example
        from daedalus.train.pretrain import to_prompt_features

        examples = [
            Example(
                spec_source="inputs A\noutputs Q\nQ = A",
                spec_hash="x", gates=0, n_inputs=1, n_outputs=1, rows=[0, 1],
                input_z=[1], output_z=[1], tokens=[V.AIR] * V.CELLS,
                latency_rt=1, blocks=1, bbox=[1, 1, 1],
                prompts=["turn the lamp on"],
            )
            for _ in range(200)
        ]
        blanked = sum(
            1
            for f in to_prompt_features(examples, random.Random(0), dropout=0.3)
            if set(f) == {0}
        )
        assert 20 < blanked < 120, blanked
