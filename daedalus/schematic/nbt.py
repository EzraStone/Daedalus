"""A minimal NBT writer.

``nbtlib`` would do this, but the format is small enough that vendoring a
writer is cheaper than the dependency — and one of the project's claims is that
a stranger can go from clone to a circuit in their world without a package
manager getting involved.

Only the tags the schematic formats actually use are implemented. Reading is
not, because nothing here needs to read one.
"""

from __future__ import annotations

import gzip
import struct
from typing import Any

TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


class Tag:
    """Base for the explicitly typed wrappers.

    NBT distinguishes byte from short from int, and Python does not. Rather
    than guess from magnitude — which silently changes a file's schema when a
    value happens to grow — every numeric value is wrapped in the type it is
    meant to be.
    """

    __slots__ = ("value",)
    tag_id = TAG_END

    def __init__(self, value: Any):
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.value!r})"


class Byte(Tag):
    __slots__ = ()
    tag_id = TAG_BYTE


class Short(Tag):
    __slots__ = ()
    tag_id = TAG_SHORT


class Int(Tag):
    __slots__ = ()
    tag_id = TAG_INT


class Long(Tag):
    __slots__ = ()
    tag_id = TAG_LONG


class ByteArray(Tag):
    __slots__ = ()
    tag_id = TAG_BYTE_ARRAY


class IntArray(Tag):
    __slots__ = ()
    tag_id = TAG_INT_ARRAY


class LongArray(Tag):
    __slots__ = ()
    tag_id = TAG_LONG_ARRAY


class String(Tag):
    __slots__ = ()
    tag_id = TAG_STRING


class List(Tag):
    """A homogeneous list. The element type is taken from the first entry, or
    given explicitly for the empty case."""

    __slots__ = ("element_id",)
    tag_id = TAG_LIST

    def __init__(self, value: list, element_id: int | None = None):
        super().__init__(value)
        if element_id is not None:
            self.element_id = element_id
        elif value:
            self.element_id = _tag_id(value[0])
        else:
            self.element_id = TAG_END


def _tag_id(value: Any) -> int:
    if isinstance(value, Tag):
        return value.tag_id
    if isinstance(value, dict):
        return TAG_COMPOUND
    if isinstance(value, str):
        return TAG_STRING
    if isinstance(value, (bytes, bytearray)):
        return TAG_BYTE_ARRAY
    if isinstance(value, list):
        return TAG_LIST
    raise TypeError(f"cannot infer an NBT tag for {type(value).__name__}; wrap it explicitly")


def _write_string(out: bytearray, s: str) -> None:
    data = s.encode("utf-8")
    out += struct.pack(">H", len(data))
    out += data


def _write_payload(out: bytearray, value: Any) -> None:
    if isinstance(value, Byte):
        out += struct.pack(">b", value.value)
    elif isinstance(value, Short):
        out += struct.pack(">h", value.value)
    elif isinstance(value, Int):
        out += struct.pack(">i", value.value)
    elif isinstance(value, Long):
        out += struct.pack(">q", value.value)
    elif isinstance(value, ByteArray):
        data = bytes(value.value)
        out += struct.pack(">i", len(data))
        out += data
    elif isinstance(value, IntArray):
        out += struct.pack(">i", len(value.value))
        for v in value.value:
            out += struct.pack(">i", v)
    elif isinstance(value, LongArray):
        out += struct.pack(">i", len(value.value))
        for v in value.value:
            out += struct.pack(">q", v)
    elif isinstance(value, String):
        _write_string(out, value.value)
    elif isinstance(value, List):
        out += struct.pack(">b", value.element_id)
        out += struct.pack(">i", len(value.value))
        for item in value.value:
            _write_payload(out, item)
    elif isinstance(value, str):
        _write_string(out, value)
    elif isinstance(value, (bytes, bytearray)):
        data = bytes(value)
        out += struct.pack(">i", len(data))
        out += data
    elif isinstance(value, dict):
        for key, item in value.items():
            out += struct.pack(">b", _tag_id(item))
            _write_string(out, key)
            _write_payload(out, item)
        out += struct.pack(">b", TAG_END)
    elif isinstance(value, list):
        element = _tag_id(value[0]) if value else TAG_END
        out += struct.pack(">b", element)
        out += struct.pack(">i", len(value))
        for item in value:
            _write_payload(out, item)
    else:
        raise TypeError(f"cannot encode {type(value).__name__} as NBT")


def dumps(root: dict, root_name: str = "", gzipped: bool = True) -> bytes:
    """Serialise a compound tag. Minecraft expects gzip; tests do not care."""
    out = bytearray()
    out += struct.pack(">b", TAG_COMPOUND)
    _write_string(out, root_name)
    _write_payload(out, root)
    data = bytes(out)
    # mtime=0 so identical schematics produce identical bytes, which is what
    # makes a corpus of exported circuits deduplicable.
    return gzip.compress(data, mtime=0) if gzipped else data


def varint(value: int) -> bytes:
    """Sponge schematics pack block indices as LEB128 varints."""
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


class NbtError(ValueError):
    """The bytes were not the NBT this expected."""


def _read_string(data: bytes, i: int) -> tuple[str, int]:
    (n,) = struct.unpack_from(">H", data, i)
    i += 2
    return data[i : i + n].decode("utf-8"), i + n


def _read_payload(data: bytes, i: int, tag: int) -> tuple[Any, int]:
    """Decode one payload, returning it and the offset just past it.

    Values come back as plain Python rather than as the ``Tag`` wrappers the
    writer takes. The wrappers exist to pin a width on the way out, which is a
    question the bytes have already answered on the way in.
    """
    if tag == TAG_BYTE:
        return struct.unpack_from(">b", data, i)[0], i + 1
    if tag == TAG_SHORT:
        return struct.unpack_from(">h", data, i)[0], i + 2
    if tag == TAG_INT:
        return struct.unpack_from(">i", data, i)[0], i + 4
    if tag == TAG_LONG:
        return struct.unpack_from(">q", data, i)[0], i + 8
    if tag == TAG_FLOAT:
        return struct.unpack_from(">f", data, i)[0], i + 4
    if tag == TAG_DOUBLE:
        return struct.unpack_from(">d", data, i)[0], i + 8
    if tag == TAG_BYTE_ARRAY:
        (n,) = struct.unpack_from(">i", data, i)
        i += 4
        return data[i : i + n], i + n
    if tag == TAG_STRING:
        return _read_string(data, i)
    if tag == TAG_LIST:
        element = data[i]
        (n,) = struct.unpack_from(">i", data, i + 1)
        i += 5
        out = []
        for _ in range(n):
            value, i = _read_payload(data, i, element)
            out.append(value)
        return out, i
    if tag == TAG_COMPOUND:
        out = {}
        while True:
            kind = data[i]
            i += 1
            if kind == TAG_END:
                return out, i
            name, i = _read_string(data, i)
            out[name], i = _read_payload(data, i, kind)
    if tag in (TAG_INT_ARRAY, TAG_LONG_ARRAY):
        width, code = (4, ">i") if tag == TAG_INT_ARRAY else (8, ">q")
        (n,) = struct.unpack_from(">i", data, i)
        i += 4
        out = [struct.unpack_from(code, data, i + k * width)[0] for k in range(n)]
        return out, i + n * width
    raise NbtError(f"unknown NBT tag id {tag}")


def loads(data: bytes) -> tuple[str, dict]:
    """Parse a compound tag, gzipped or not. Returns ``(root_name, value)``."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if not data or data[0] != TAG_COMPOUND:
        raise NbtError("NBT does not start with a compound tag")
    name, i = _read_string(data, 1)
    value, _ = _read_payload(data, i, TAG_COMPOUND)
    return name, value


def read_varints(data: bytes) -> list[int]:
    """Unpack a run of LEB128 varints, the way Sponge stores block indices."""
    out: list[int] = []
    value = shift = 0
    for byte in data:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            continue
        out.append(value)
        value = shift = 0
    if shift:
        raise NbtError("varint run ended mid-value")
    return out
