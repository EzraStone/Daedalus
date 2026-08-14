package dev.daedalus.harness;

import java.util.HashSet;
import java.util.Optional;
import java.util.Set;

import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.registry.Registries;
import net.minecraft.state.property.Property;
import net.minecraft.util.Identifier;

/** Parses canonical block-state strings from a Sponge schematic palette. */
public final class BlockStates {
    private BlockStates() {}

    public static BlockState parse(String encoded) {
        int bracket = encoded.indexOf('[');
        String blockName = bracket < 0 ? encoded : encoded.substring(0, bracket);
        if (blockName.isBlank()) {
            throw new IllegalArgumentException("block state has no identifier");
        }

        Identifier identifier = Identifier.tryParse(blockName);
        if (identifier == null || !Registries.BLOCK.containsId(identifier)) {
            throw new IllegalArgumentException("unknown block " + blockName);
        }
        Block block = Registries.BLOCK.get(identifier);
        BlockState state = block.getDefaultState();

        if (bracket < 0) {
            return state;
        }
        if (!encoded.endsWith("]")) {
            throw new IllegalArgumentException("unterminated properties in " + encoded);
        }
        String properties = encoded.substring(bracket + 1, encoded.length() - 1);
        if (properties.isEmpty()) {
            return state;
        }

        Set<String> seen = new HashSet<>();
        for (String assignment : properties.split(",", -1)) {
            int equals = assignment.indexOf('=');
            if (equals <= 0 || equals == assignment.length() - 1) {
                throw new IllegalArgumentException("invalid property in " + encoded);
            }
            String name = assignment.substring(0, equals);
            String value = assignment.substring(equals + 1);
            if (!seen.add(name)) {
                throw new IllegalArgumentException("duplicate property " + name);
            }
            Property<?> property = block.getStateManager().getProperty(name);
            if (property == null) {
                throw new IllegalArgumentException(blockName + " has no property " + name);
            }
            state = apply(state, property, value);
        }
        return state;
    }

    private static <T extends Comparable<T>> BlockState apply(
            BlockState state, Property<T> property, String encoded) {
        Optional<T> parsed = property.parse(encoded);
        if (parsed.isEmpty()) {
            throw new IllegalArgumentException(
                    "invalid " + property.getName() + " value " + encoded);
        }
        return state.with(property, parsed.get());
    }
}
