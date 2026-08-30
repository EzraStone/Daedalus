package dev.daedalus.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.ByteArrayOutputStream;

import net.minecraft.nbt.NbtCompound;
import net.minecraft.nbt.NbtIo;
import org.junit.jupiter.api.Test;

final class SpongeSchematicTest {
    @Test
    void decodesPaletteAndSpongeCoordinateOrder() throws Exception {
        NbtCompound root = baseRoot(2, 1, 2);
        NbtCompound palette = new NbtCompound();
        palette.putInt("minecraft:air", 0);
        palette.putInt("minecraft:stone", 1);
        root.put("Palette", palette);
        root.putByteArray("BlockData", new byte[] {0, 1, 1, 0});

        Schematic decoded = SpongeSchematic.decode(compress(root));

        assertEquals("minecraft:air", decoded.stateAt(0, 0, 0));
        assertEquals("minecraft:stone", decoded.stateAt(1, 0, 0));
        assertEquals("minecraft:stone", decoded.stateAt(0, 0, 1));
        assertEquals("minecraft:air", decoded.stateAt(1, 0, 1));
    }

    @Test
    void decodesMultiBytePaletteVarints() throws Exception {
        NbtCompound root = baseRoot(1, 1, 1);
        NbtCompound palette = new NbtCompound();
        palette.putInt("minecraft:target", 130);
        root.put("Palette", palette);
        root.putByteArray("BlockData", new byte[] {(byte) 0x82, 0x01});

        assertEquals(
                "minecraft:target",
                SpongeSchematic.decode(compress(root)).stateAt(0, 0, 0));
    }

    @Test
    void rejectsTruncatedVolumes() throws Exception {
        NbtCompound root = baseRoot(2, 1, 1);
        NbtCompound palette = new NbtCompound();
        palette.putInt("minecraft:air", 0);
        root.put("Palette", palette);
        root.putByteArray("BlockData", new byte[] {0});

        assertThrows(IllegalArgumentException.class, () -> SpongeSchematic.decode(compress(root)));
    }

    private static NbtCompound baseRoot(int width, int height, int length) {
        NbtCompound root = new NbtCompound();
        root.putInt("Version", 2);
        root.putShort("Width", (short) width);
        root.putShort("Height", (short) height);
        root.putShort("Length", (short) length);
        return root;
    }

    private static byte[] compress(NbtCompound root) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        NbtIo.writeCompressed(root, output);
        return output.toByteArray();
    }
}
