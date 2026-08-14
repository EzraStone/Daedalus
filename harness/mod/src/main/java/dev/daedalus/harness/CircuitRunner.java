package dev.daedalus.harness;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
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
    /** Consecutive unchanged ticks required after Minecraft's block queue drains. */
    public static final int QUIET_TICKS = 2;

    private final MinecraftServer server;
    private final WorldFixture fixture;
    private final Map<String, Schematic> schematics = new ConcurrentHashMap<>();
    private SweepJob activeSweep;

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
        callOnServer(() -> {
            fixture.replace(schematic);
            return null;
        });
    }

    /**
     * Toggle every input combination and read the lamps.
     *
     * <p>Each combination is applied from a cold start, matching the
     * simulator's pass A. Sweeping without resetting would measure the
     * circuit's history dependence instead of its truth table, and the two
     * sides would disagree for a reason that has nothing to do with fidelity.
     */
    public Result sweep(String id, List<int[]> levers, List<int[]> lamps) throws Exception {
        Schematic schematic = schematics.get(id);
        if (schematic == null) {
            throw new IllegalArgumentException("unknown schematic id " + id);
        }
        List<int[]> safeLevers = validatePorts("lever", levers, schematic);
        List<int[]> safeLamps = validatePorts("lamp", lamps, schematic);
        if (safeLevers.size() > 12) {
            throw new IllegalArgumentException("at most 12 input levers are supported");
        }

        SweepJob job = callOnServer(() -> {
            if (activeSweep != null) {
                throw new IllegalStateException("another sweep is already running");
            }
            SweepJob created = new SweepJob(schematic, safeLevers, safeLamps);
            activeSweep = created;
            created.beginAssignment();
            return created;
        });
        return job.completed.get(15, TimeUnit.MINUTES);
    }

    /** Advances a sweep once per real game tick. Called by the Fabric tick event. */
    public void tick(MinecraftServer tickingServer) {
        if (tickingServer != server || activeSweep == null) {
            return;
        }
        activeSweep.tick();
    }

    private <T> T callOnServer(Callable<T> operation) throws Exception {
        if (server.isOnThread()) {
            return operation.call();
        }
        CompletableFuture<T> completed = new CompletableFuture<>();
        server.execute(() -> {
            try {
                completed.complete(operation.call());
            } catch (Throwable error) {
                completed.completeExceptionally(error);
            }
        });
        return completed.get(60, TimeUnit.SECONDS);
    }

    private static List<int[]> validatePorts(
            String kind, List<int[]> ports, Schematic schematic) {
        List<int[]> copy = new ArrayList<>(ports.size());
        for (int i = 0; i < ports.size(); i++) {
            int[] position = ports.get(i);
            if (position == null || position.length != 3) {
                throw new IllegalArgumentException(kind + " " + i + " must be an [x,y,z] position");
            }
            if (position[0] < 0 || position[0] >= schematic.width()
                    || position[1] < 0 || position[1] >= schematic.height()
                    || position[2] < 0 || position[2] >= schematic.length()) {
                throw new IllegalArgumentException(kind + " " + i + " lies outside the schematic");
            }
            copy.add(position.clone());
        }
        return List.copyOf(copy);
    }

    private final class SweepJob {
        private final Schematic schematic;
        private final List<int[]> levers;
        private final List<int[]> lamps;
        private final int assignments;
        private final Result result = new Result();
        private final CompletableFuture<Result> completed = new CompletableFuture<>();
        private int assignment;
        private int elapsedTicks;
        private int quietTicks;
        private long lastFingerprint;

        private SweepJob(Schematic schematic, List<int[]> levers, List<int[]> lamps) {
            this.schematic = schematic;
            this.levers = levers;
            this.lamps = lamps;
            this.assignments = 1 << levers.size();
        }

        private void beginAssignment() {
            fixture.replace(schematic);
            fixture.applyInputs(levers, assignment);
            lastFingerprint = fixture.fingerprint();
            elapsedTicks = 0;
            quietTicks = 0;
        }

        private void tick() {
            try {
                elapsedTicks++;
                long fingerprint = fixture.fingerprint();
                if (fingerprint == lastFingerprint) {
                    quietTicks++;
                } else {
                    quietTicks = 0;
                    lastFingerprint = fingerprint;
                }

                if (quietTicks >= QUIET_TICKS && !fixture.hasScheduledBlockTicks()) {
                    result.rows.add(fixture.readRow(levers, lamps, assignment));
                    assignment++;
                    if (assignment == assignments) {
                        finish();
                    } else {
                        beginAssignment();
                    }
                } else if (elapsedTicks >= SETTLE_CAP_TICKS) {
                    result.settled = false;
                    finish();
                }
            } catch (Throwable error) {
                completed.completeExceptionally(error);
                activeSweep = null;
            }
        }

        private void finish() {
            completed.complete(result);
            activeSweep = null;
        }
    }
}
