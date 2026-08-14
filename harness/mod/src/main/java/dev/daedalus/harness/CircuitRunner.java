package dev.daedalus.harness;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import net.minecraft.server.MinecraftServer;

/**
 * Places a schematic and walks its inputs, on the server thread.
 *
 * <p>The settling rule matters and is the one thing worth getting exactly right:
 * after flipping a lever, wait until the world stops changing rather than
 * waiting a fixed number of ticks. A fixed wait that is too short reads a
 * circuit mid-propagation and reports a disagreement that is really an
 * impatience bug; too long and a 10k-case sweep takes all night. The cap
 * matches the simulator's 200 game ticks so that "did not settle" means the
 * same thing on both sides.
 */
public final class CircuitRunner {
    /** Matches redsim's DEFAULT_MAX_GAME_TICKS. */
    public static final int SETTLE_CAP_TICKS = 200;

    private final MinecraftServer server;
    private final WorldFixture fixture;
    private final Map<String, Schematic> schematics = new ConcurrentHashMap<>();

    public CircuitRunner(MinecraftServer server) {
        this.server = server;
        this.fixture = new WorldFixture(server);
    }

    public static final class Result {
        public final List<int[]> rows = new ArrayList<>();
        public boolean settled = true;
    }

    /**
     * Paste a schematic into the void world at the harness origin, clearing
     * whatever was there. Implemented against the Fabric structure API.
     */
    public void place(String id, byte[] payload) throws Exception {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("schematic id must not be blank");
        }
        Schematic schematic = SpongeSchematic.decode(payload);
        schematics.put(id, schematic);
        runOnServer(() -> fixture.replace(schematic));
    }

    /**
     * Toggle every input combination and read the lamps.
     *
     * <p>Each combination is applied from a cold start, matching the
     * simulator's pass A. Sweeping without resetting would measure the
     * circuit's history dependence instead of its truth table, and the two
     * sides would disagree for a reason that has nothing to do with fidelity.
     */
    public Result sweep(String id, List<int[]> levers, List<int[]> lamps) {
        throw new UnsupportedOperationException(
                "sweep() needs a running Fabric server; see harness/mod/README.md");
    }

    private void runOnServer(Runnable operation) throws Exception {
        CompletableFuture<Void> completed = new CompletableFuture<>();
        server.execute(() -> {
            try {
                operation.run();
                completed.complete(null);
            } catch (Throwable error) {
                completed.completeExceptionally(error);
            }
        });
        completed.get(60, TimeUnit.SECONDS);
    }
}
