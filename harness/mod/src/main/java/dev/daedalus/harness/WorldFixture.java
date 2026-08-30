package dev.daedalus.harness;

import java.util.ArrayList;
import java.util.List;

import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.block.LeverBlock;
import net.minecraft.block.RedstoneLampBlock;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.BlockPos;

/** Owns the isolated block volume used by one harness server. */
public final class WorldFixture {
    public static final BlockPos DEFAULT_ORIGIN = new BlockPos(0, 100, 0);

    private final MinecraftServer server;
    private final ServerWorld world;
    private final BlockPos origin;
    private int lastWidth;
    private int lastHeight;
    private int lastLength;

    public WorldFixture(MinecraftServer server) {
        this(server, DEFAULT_ORIGIN);
    }

    WorldFixture(MinecraftServer server, BlockPos origin) {
        this.server = server;
        this.world = server.getOverworld();
        this.origin = origin.toImmutable();
    }

    public void replace(Schematic schematic) {
        requireServerThread();
        clear(Math.max(lastWidth, schematic.width()),
                Math.max(lastHeight, schematic.height()),
                Math.max(lastLength, schematic.length()));

        List<Placement> components = new ArrayList<>();
        for (int y = 0; y < schematic.height(); y++) {
            for (int z = 0; z < schematic.length(); z++) {
                for (int x = 0; x < schematic.width(); x++) {
                    BlockState state = BlockStates.parse(schematic.stateAt(x, y, z));
                    if (state.isAir()) {
                        continue;
                    }
                    Placement placement = new Placement(origin.add(x, y, z), state);
                    if (state.isOf(Blocks.STONE)) {
                        set(placement);
                    } else {
                        components.add(placement);
                    }
                }
            }
        }
        components.forEach(this::set);
        notifyPlacedBlocks(schematic);
        lastWidth = schematic.width();
        lastHeight = schematic.height();
        lastLength = schematic.length();
    }

    public void applyInputs(List<int[]> levers, int assignment) {
        requireServerThread();
        for (int i = 0; i < levers.size(); i++) {
            BlockPos position = absolute(levers.get(i));
            BlockState state = world.getBlockState(position);
            if (!(state.getBlock() instanceof LeverBlock)) {
                throw new IllegalArgumentException("input " + i + " is not a lever at " + position);
            }
            boolean powered = ((assignment >>> i) & 1) != 0;
            if (state.get(LeverBlock.POWERED) != powered) {
                ((LeverBlock) state.getBlock()).togglePower(state, world, position);
            }
        }
    }

    public int[] readRow(List<int[]> levers, List<int[]> lamps, int assignment) {
        requireServerThread();
        int[] row = new int[levers.size() + lamps.size()];
        for (int i = 0; i < levers.size(); i++) {
            row[i] = (assignment >>> i) & 1;
        }
        for (int i = 0; i < lamps.size(); i++) {
            BlockPos position = absolute(lamps.get(i));
            BlockState state = world.getBlockState(position);
            if (!(state.getBlock() instanceof RedstoneLampBlock)) {
                throw new IllegalArgumentException("output " + i + " is not a lamp at " + position);
            }
            row[levers.size() + i] = state.get(RedstoneLampBlock.LIT) ? 1 : 0;
        }
        return row;
    }

    public long fingerprint() {
        requireServerThread();
        long hash = 0xcbf29ce484222325L;
        for (int y = 0; y < lastHeight; y++) {
            for (int z = 0; z < lastLength; z++) {
                for (int x = 0; x < lastWidth; x++) {
                    int state = Block.getRawIdFromState(world.getBlockState(origin.add(x, y, z)));
                    hash ^= state;
                    hash *= 0x100000001b3L;
                }
            }
        }
        return hash;
    }

    public boolean hasScheduledBlockTicks() {
        requireServerThread();
        return world.getBlockTickScheduler().getTickCount() > 0;
    }

    private void clear(int width, int height, int length) {
        for (int y = -1; y <= height; y++) {
            for (int z = -1; z <= length; z++) {
                for (int x = -1; x <= width; x++) {
                    world.setBlockState(origin.add(x, y, z), Blocks.AIR.getDefaultState(),
                            Block.NOTIFY_LISTENERS | Block.FORCE_STATE | Block.SKIP_DROPS);
                }
            }
        }
    }

    private BlockPos absolute(int[] relative) {
        return origin.add(relative[0], relative[1], relative[2]);
    }

    private void set(Placement placement) {
        world.setBlockState(placement.position(), placement.state(),
                Block.NOTIFY_LISTENERS | Block.FORCE_STATE | Block.SKIP_DROPS);
    }

    private void notifyPlacedBlocks(Schematic schematic) {
        for (int y = 0; y < schematic.height(); y++) {
            for (int z = 0; z < schematic.length(); z++) {
                for (int x = 0; x < schematic.width(); x++) {
                    BlockPos position = origin.add(x, y, z);
                    BlockState state = world.getBlockState(position);
                    if (!state.isAir()) {
                        world.updateNeighborsAlways(position, state.getBlock());
                    }
                }
            }
        }
    }

    private void requireServerThread() {
        if (!server.isOnThread()) {
            throw new IllegalStateException("world fixture accessed outside the server thread");
        }
    }

    private record Placement(BlockPos position, BlockState state) {}
}
