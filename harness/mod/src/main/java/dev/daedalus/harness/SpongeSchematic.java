package dev.daedalus.harness;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.minecraft.nbt.NbtCompound;
import net.minecraft.nbt.NbtIo;

/** Decoder for the Sponge schematic v2 subset emitted by Daedalus. */
public final class SpongeSchematic {
    private static final int MAX_DIMENSION = 64;

    private SpongeSchematic() {}

    public static Schematic decode(byte[] compressed) throws IOException {
        NbtCompound root = NbtIo.readCompressed(new ByteArrayInputStream(compressed));
        if (root.getInt("Version") != 2) {
            throw new IllegalArgumentException("only Sponge schematic version 2 is supported");
        }

        int width = dimension(root, "Width");
        int height = dimension(root, "Height");
        int length = dimension(root, "Length");
        int volume = Math.multiplyExact(Math.multiplyExact(width, height), length);

        NbtCompound paletteTag = root.getCompound("Palette");
        Map<Integer, String> palette = new HashMap<>();
        for (String state : paletteTag.getKeys()) {
            int index = paletteTag.getInt(state);
            if (index < 0 || palette.put(index, state) != null) {
                throw new IllegalArgumentException("schematic palette indices must be unique");
            }
        }
        if (palette.isEmpty()) {
            throw new IllegalArgumentException("schematic palette is empty");
        }

        List<Integer> indices = decodeVarints(root.getByteArray("BlockData"), volume);
        List<String> states = new ArrayList<>(volume);
        for (int index : indices) {
            String state = palette.get(index);
            if (state == null) {
                throw new IllegalArgumentException("block data references palette index " + index);
            }
            states.add(state);
        }
        return new Schematic(width, height, length, states);
    }

    private static int dimension(NbtCompound root, String key) {
        int value = root.getShort(key);
        if (value <= 0 || value > MAX_DIMENSION) {
            throw new IllegalArgumentException(
                    key + " must be between 1 and " + MAX_DIMENSION + ", got " + value);
        }
        return value;
    }

    private static List<Integer> decodeVarints(byte[] data, int expected) {
        List<Integer> values = new ArrayList<>(expected);
        int cursor = 0;
        while (cursor < data.length) {
            int value = 0;
            int shift = 0;
            while (true) {
                if (cursor >= data.length) {
                    throw new IllegalArgumentException("truncated varint in block data");
                }
                int next = data[cursor++] & 0xff;
                value |= (next & 0x7f) << shift;
                if ((next & 0x80) == 0) {
                    break;
                }
                shift += 7;
                if (shift >= 32) {
                    throw new IllegalArgumentException("oversized varint in block data");
                }
            }
            values.add(value);
            if (values.size() > expected) {
                throw new IllegalArgumentException("block data exceeds schematic volume");
            }
        }
        if (values.size() != expected) {
            throw new IllegalArgumentException(
                    "expected " + expected + " palette indices, got " + values.size());
        }
        return values;
    }
}
