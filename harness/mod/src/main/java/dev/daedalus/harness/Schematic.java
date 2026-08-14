package dev.daedalus.harness;

import java.util.List;

/** An immutable, decoded block volume in Sponge's x-fastest coordinate order. */
public record Schematic(int width, int height, int length, List<String> states) {
    public Schematic {
        if (width <= 0 || height <= 0 || length <= 0) {
            throw new IllegalArgumentException("schematic dimensions must be positive");
        }
        int volume = Math.multiplyExact(Math.multiplyExact(width, height), length);
        if (states.size() != volume) {
            throw new IllegalArgumentException(
                    "expected " + volume + " blocks, got " + states.size());
        }
        states = List.copyOf(states);
    }

    public String stateAt(int x, int y, int z) {
        if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= length) {
            throw new IndexOutOfBoundsException("block outside schematic volume");
        }
        return states.get(x + z * width + y * width * length);
    }
}
