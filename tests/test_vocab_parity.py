"""The Rust and Python vocabularies must agree exactly.

Token ids are a wire format. They are baked into every serialised corpus and
every trained checkpoint, so a silent divergence between the two
implementations would not surface as an error — it would surface as a model
that quietly learns the wrong blocks. This test is the thing that stops that,
by asking the Rust binary to describe its own vocabulary and diffing it
against the Python module's.
"""

from __future__ import annotations

import subprocess

import pytest

from daedalus import redsim
from daedalus import vocab as V
from daedalus.spec.canon import semantic_hash


@pytest.fixture(scope="module")
def dump() -> list[str]:
    binary = redsim.find_binary()
    out = subprocess.run(
        [str(binary), "vocab"], check=True, capture_output=True, text=True
    ).stdout
    return out.splitlines()


def test_geometry_matches(dump):
    (row,) = [ln for ln in dump if ln.startswith("geometry\t")]
    _, sx, sy, sz, cells, substrate, logic, in_x, out_x = row.split("\t")
    assert int(sx) == V.SX
    assert int(sy) == V.SY
    assert int(sz) == V.SZ
    assert int(cells) == V.CELLS
    assert int(substrate) == V.SUBSTRATE_Y
    assert int(logic) == V.LOGIC_Y
    assert int(in_x) == V.INPUT_X
    assert int(out_x) == V.OUTPUT_X


def test_every_token_agrees(dump):
    rows = [ln.split("\t") for ln in dump if ln and ln[0].isdigit()]
    assert len(rows) == V.VOCAB_SIZE, "the dump should cover the whole vocabulary"

    mismatches = []
    for cols in rows:
        tid = int(cols[0])
        got_glyph, got_opaque, got_conductive, got_state = cols[1], cols[2], cols[3], cols[4]
        if tid >= V.CONTROL_BASE:
            # Control tokens have no block state; only their presence matters.
            assert got_state.startswith("control:")
            continue
        want = (
            V.glyph(tid),
            str(int(V.is_opaque(tid))),
            str(int(V.is_conductive(tid))),
            V.state_string(tid),
        )
        got = (got_glyph, got_opaque, got_conductive, got_state)
        if want != got:
            mismatches.append(f"token {tid}: rust={got} python={want}")
    assert not mismatches, "\n".join(mismatches)


def test_semantic_hashes_agree(dump):
    rows = [ln.split("\t") for ln in dump if ln.startswith("hash\t")]
    assert rows, "the dump should carry reference hashes"
    for _, n_in, n_out, cells, value in rows:
        table = [int(c) for c in cells.split(",")]
        assert f"{semantic_hash(int(n_in), int(n_out), table):016x}" == value


def test_control_tokens_are_not_block_states():
    for t in (V.PAD, V.BOS, V.EOS, V.MASK):
        assert V.is_control(t)
        with pytest.raises(ValueError):
            V.decode(t)
        # And nothing may generate them into a grid.
        assert not V.legal_at(t, V.LOGIC_Y)


def test_substrate_layer_admits_only_air_and_stone():
    legal = [t for t in V.BLOCK_TOKENS if V.legal_at(t, V.SUBSTRATE_Y)]
    assert legal == [V.AIR, V.SOLID]


def test_observers_are_in_the_vocabulary_but_never_generated():
    # §02 keeps observers in the vocabulary so a v2 sequential model has a
    # token to attach to, while v1 refuses to emit them.
    assert V.EXCLUDED_FROM_GENERATION
    for t in V.EXCLUDED_FROM_GENERATION:
        assert V.decode(t).kind == "observer"
        assert not V.legal_at(t, V.LOGIC_Y)


def test_index_order_is_layer_major():
    # Two cells in the same row must be adjacent in the sequence; that is the
    # property local attention relies on.
    assert V.index(4, 2, 5) == V.index(3, 2, 5) + 1
    assert V.index(0, 3, 0) == 3 * V.SX * V.SZ
    for i in range(V.CELLS):
        x, y, z = V.unindex(i)
        assert V.index(x, y, z) == i
