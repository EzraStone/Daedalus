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
        # One row per cell, each as wide as the model's head rather than as
        # wide as the block vocabulary -- the mask is applied to logits over
        # blocks *and* prefix tokens, and a row that stops at VOCAB_SIZE
        # silently leaves the spec tokens unmasked.
        mask = legality_mask()
        assert len(mask) == V.CELLS
        assert all(len(row) == TOTAL_VOCAB for row in mask)

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


class TestNbtReading:
    """The encoder had no counterpart, so nothing it wrote could be read back."""

    def test_every_tag_type_survives_a_round_trip(self):
        from daedalus.schematic.nbt import (
            Byte,
            ByteArray,
            Int,
            IntArray,
            List,
            LongArray,
            Short,
            String,
            dumps,
            loads,
        )

        root = {
            "Version": Int(2),
            "Width": Short(16),
            "BlockData": ByteArray(b"\x01\x02\x7f"),
            "Palette": {"minecraft:air": Int(0)},
            "Names": List([String("a"), String("b")]),
            "Offset": IntArray([1, 2, 3]),
            "Packed": LongArray([1 << 40, -2]),
            "Flag": Byte(1),
        }
        name, back = loads(dumps(root, "Schematic"))
        assert name == "Schematic"
        assert back["Version"] == 2
        assert back["BlockData"] == b"\x01\x02\x7f"
        assert back["Palette"] == {"minecraft:air": 0}
        assert back["Names"] == ["a", "b"]
        assert back["Offset"] == [1, 2, 3]
        assert back["Packed"] == [1 << 40, -2]

    def test_an_empty_list_round_trips(self):
        from daedalus.schematic.nbt import List, dumps, loads

        _name, back = loads(dumps({"BlockEntities": List([], element_id=10)}))
        assert back["BlockEntities"] == []

    def test_ungzipped_nbt_is_accepted_too(self):
        from daedalus.schematic.nbt import Int, dumps, loads

        _name, back = loads(dumps({"V": Int(1)}, "", gzipped=False))
        assert back["V"] == 1

    def test_something_that_is_not_nbt_is_refused(self):
        import pytest as _pytest

        from daedalus.schematic.nbt import NbtError, loads

        with _pytest.raises(NbtError):
            loads(b"not nbt at all")

    def test_varints_round_trip_across_the_byte_boundary(self):
        from daedalus.schematic.nbt import read_varints, varint

        values = [0, 1, 127, 128, 255, 300, 16383, 16384]
        assert read_varints(b"".join(varint(v) for v in values)) == values

    def test_a_truncated_varint_is_an_error_rather_than_a_silent_zero(self):
        import pytest as _pytest

        from daedalus.schematic.nbt import NbtError, read_varints

        with _pytest.raises(NbtError):
            read_varints(b"\x80")


class TestStateParsing:
    """Minecraft block-state strings back into tokens."""

    def test_every_token_survives_its_own_state_string(self):
        # The inverse is derived from state_string rather than restated, so
        # this is what guarantees the two cannot drift apart.
        for token in V.BLOCK_TOKENS:
            assert V.from_state(V.state_string(token)) == token, token

    def test_signal_properties_are_ignored(self):
        # A circuit pulled out of a running world has power in it. That is
        # state the simulator works out for itself, not part of the identity
        # of the block -- keeping it would multiply the vocabulary by sixteen.
        assert V.from_state("minecraft:redstone_wire[power=15]") == V.WIRE
        assert V.from_state("minecraft:redstone_lamp[lit=true]") == V.LAMP
        assert V.from_state("minecraft:repeater[facing=north,delay=2,locked=true,powered=true]") == (
            V.repeater(V.Dir4.SOUTH, 2)
        )

    def test_a_block_outside_the_vocabulary_is_refused(self):
        with pytest.raises(V.UnknownBlock):
            V.from_state("minecraft:piston[facing=up]")

    def test_properties_split_cleanly(self):
        name, props = V.parse_state("minecraft:repeater[facing=north,delay=3]")
        assert name == "minecraft:repeater"
        assert props == {"facing": "north", "delay": "3"}

    def test_a_bare_name_has_no_properties(self):
        assert V.parse_state("minecraft:stone") == ("minecraft:stone", {})


class TestSchematicRoundTrip:
    def test_a_written_schematic_reads_back_identical(self, tmp_path):
        from daedalus.schematic import read_schem

        grid = Grid.with_substrate()
        grid.set(4, 1, 4, V.WIRE)
        grid.set(5, 1, 4, V.torch(V.Attach.WEST))
        grid.set(6, 1, 4, V.repeater(V.Dir4.EAST, 3))
        grid.set(7, 1, 4, V.comparator(V.Dir4.EAST, True))
        path = write_schem(grid, tmp_path / "c.schem")
        assert read_schem(path).tokens() == grid.tokens()

    def test_a_schematic_bigger_than_the_build_volume_is_refused(self, tmp_path):
        from daedalus.schematic import SchematicError, read_schem
        from daedalus.schematic.nbt import ByteArray, Int, Short, dumps

        blob = dumps(
            {
                "Width": Short(64),
                "Height": Short(V.SY),
                "Length": Short(V.SZ),
                "Palette": {"minecraft:air": Int(0)},
                "BlockData": ByteArray(b"\x00"),
            },
            "Schematic",
        )
        path = tmp_path / "big.schem"
        path.write_bytes(blob)
        with pytest.raises(SchematicError, match="larger than"):
            read_schem(path)

    def test_a_truncated_block_run_is_refused(self, tmp_path):
        from daedalus.schematic import SchematicError, read_schem
        from daedalus.schematic.nbt import ByteArray, Int, Short, dumps

        path = tmp_path / "short.schem"
        path.write_bytes(
            dumps(
                {
                    "Width": Short(V.SX),
                    "Height": Short(V.SY),
                    "Length": Short(V.SZ),
                    "Palette": {"minecraft:air": Int(0)},
                    "BlockData": ByteArray(b"\x00\x00"),
                },
                "Schematic",
            )
        )
        with pytest.raises(SchematicError, match="expected"):
            read_schem(path)

    def test_a_missing_field_says_which_one(self, tmp_path):
        from daedalus.schematic import SchematicError, read_schem
        from daedalus.schematic.nbt import Short, dumps

        path = tmp_path / "bad.schem"
        path.write_bytes(dumps({"Width": Short(V.SX)}, "Schematic"))
        with pytest.raises(SchematicError, match="Height"):
            read_schem(path)
