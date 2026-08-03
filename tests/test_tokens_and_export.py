"""Sequence construction and schematic export."""

from __future__ import annotations

import gzip

import pytest

from daedalus import vocab as V
from daedalus.grid import Grid, diff_cells, edit_distance
from daedalus.schematic import block_summary, palette_of, write_litematic, write_schem
from daedalus.schematic.nbt import Int, String, dumps, varint
from daedalus.spec import Spec
from daedalus.tokens import (
    PREFIX_VOCAB,
    TOTAL_VOCAB,
    Sequence,
    encode,
    legality_mask,
    port_mask,
    spec_prefix,
)


@pytest.fixture
def spec() -> Spec:
    return Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)")


class TestSequences:
    def test_flat_length_matches_the_design(self, spec):
        seq = encode(Grid.with_substrate(), spec.default_placement())
        assert len(seq.flat()) == V.SEQ_LEN == V.CELLS + V.PREFIX_LEN

    def test_prefix_is_fixed_width_and_padded(self, spec):
        prefix, _ = spec_prefix(spec.default_placement())
        assert len(prefix) == V.PREFIX_LEN
        assert prefix[0] == PREFIX_VOCAB["<spec>"]

    def test_prefix_ids_do_not_collide_with_block_ids(self, spec):
        # Prefix and body share one embedding table, so the prefix has to be
        # offset past the block vocabulary. Overlapping them would let the model
        # confuse "this cell is a repeater" with "this spec mentions one".
        seq = encode(Grid.with_substrate(), spec.default_placement())
        flat = seq.flat()
        assert all(t >= V.VOCAB_SIZE for t in flat[: V.PREFIX_LEN])
        assert all(t < V.VOCAB_SIZE for t in flat[V.PREFIX_LEN :])
        assert max(flat) < TOTAL_VOCAB

    def test_nl_slots_are_marked(self, spec):
        prefix, slots = spec_prefix(spec.default_placement(), nl_slots=4)
        assert len(slots) == 4
        assert all(prefix[i] == PREFIX_VOCAB["<nl>"] for i in slots)

    def test_specs_with_different_tables_get_different_prefixes(self):
        a = Spec.parse("inputs A B\noutputs Q\nQ = A & B").default_placement()
        b = Spec.parse("inputs A B\noutputs Q\nQ = A | B").default_placement()
        assert spec_prefix(a)[0] != spec_prefix(b)[0]

    def test_rejects_wrong_sizes(self):
        with pytest.raises(ValueError):
            Sequence(prefix=(0,) * 4, body=(0,) * V.CELLS)
        with pytest.raises(ValueError):
            Sequence(prefix=(0,) * V.PREFIX_LEN, body=(0,) * 10)


class TestLegality:
    def test_mask_covers_every_cell(self):
        mask = legality_mask()
        assert len(mask) == V.CELLS
        assert all(len(row) == V.VOCAB_SIZE for row in mask)

    def test_control_tokens_are_never_legal(self):
        mask = legality_mask()
        for row in (mask[0], mask[V.index(3, 1, 3)]):
            assert not any(row[t] for t in (V.PAD, V.BOS, V.EOS, V.MASK))

    def test_ports_are_pinned(self, spec):
        placed = spec.default_placement()
        fixed = port_mask(placed)
        for x, y, z in placed.input_ports:
            assert V.decode(fixed[V.index(x, y, z)]).kind == "lever"
            assert fixed[V.index(x + 1, y, z)] == V.SOLID
        for x, y, z in placed.output_ports:
            assert fixed[V.index(x, y, z)] == V.LAMP


class TestGridHelpers:
    def test_substrate_is_excluded_from_the_block_count(self):
        # 256 substrate blocks would swamp the compactness metric with a
        # constant, which is the metric the project's central claim rests on.
        assert Grid.with_substrate().material_blocks() == 0

    def test_edit_distance_and_diff_agree(self):
        a = Grid.with_substrate()
        b = a.copy()
        b.set(3, 1, 3, V.WIRE)
        b.set(4, 1, 3, V.SOLID)
        assert edit_distance(a, b) == 2
        assert diff_cells(a, b) == [V.index(3, 1, 3), V.index(4, 1, 3)]

    def test_out_of_bounds_is_air_not_stone(self):
        # A circuit must not be able to lean on the world border for support.
        assert Grid.with_substrate().get(-1, 0, 0) == V.AIR


class TestNBT:
    def test_varint_matches_leb128(self):
        assert varint(0) == b"\x00"
        assert varint(127) == b"\x7f"
        assert varint(128) == b"\x80\x01"
        assert varint(300) == b"\xac\x02"

    def test_dumps_is_gzipped_and_reproducible(self):
        root = {"Version": Int(2), "Name": String("x")}
        a = dumps(root, "Schematic")
        b = dumps(root, "Schematic")
        assert a == b, "identical schematics must produce identical bytes"
        assert gzip.decompress(a)[:1] == b"\x0a"  # TAG_Compound


class TestExport:
    def test_schem_and_litematic_write(self, tmp_path):
        grid = Grid.with_substrate()
        grid.set(4, 1, 4, V.WIRE)
        grid.set(5, 1, 4, V.torch(V.Attach.WEST))
        a = write_schem(grid, tmp_path / "c.schem")
        b = write_litematic(grid, tmp_path / "c.litematic")
        assert a.stat().st_size > 0
        assert b.stat().st_size > 0
        assert gzip.decompress(a.read_bytes())[:1] == b"\x0a"

    def test_air_is_palette_index_zero(self):
        # Most of the grid is air; a palette where the commonest entry encodes
        # as a single zero byte is what keeps the file small.
        palette, _ = palette_of(Grid.with_substrate())
        assert palette[V.state_string(V.AIR)] == 0

    def test_block_summary_counts_kinds(self):
        grid = Grid.with_substrate()
        grid.set(4, 1, 4, V.WIRE)
        grid.set(5, 1, 4, V.WIRE)
        summary = block_summary(grid)
        assert summary["wire"] == 2
        assert summary["solid"] == 256

    def test_every_block_state_has_a_minecraft_name(self):
        for t in V.BLOCK_TOKENS:
            name = V.state_string(t)
            assert name.startswith("minecraft:"), t
