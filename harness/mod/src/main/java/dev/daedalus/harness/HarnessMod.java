package dev.daedalus.harness;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;

/**
 * Entry point. Starts the socket listener when the server comes up.
 *
 * <p>The port is read from {@code daedalus.harness.port} so several servers can
 * run in parallel — verifier throughput is the bottleneck in
 * {@code daedalus.train.loop}, but game throughput is the bottleneck here, and
 * the only way to raise it is more servers.
 */
public final class HarnessMod implements ModInitializer {
    public static final int DEFAULT_PORT = 25599;

    private HarnessServer harnessServer;
    private CircuitRunner circuitRunner;

    @Override
    public void onInitialize() {
        ServerLifecycleEvents.SERVER_STARTED.register(server -> {
            int port = Integer.getInteger("daedalus.harness.port", DEFAULT_PORT);
            circuitRunner = new CircuitRunner(server);
            harnessServer = new HarnessServer(port, circuitRunner);
            Thread thread = new Thread(harnessServer, "daedalus-harness");
            thread.setDaemon(true);
            thread.start();
            System.out.println("[daedalus] fidelity harness listening on " + port);
        });
        ServerTickEvents.END_SERVER_TICK.register(server -> {
            if (circuitRunner != null) {
                circuitRunner.tick(server);
            }
        });
        ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
            if (harnessServer != null) {
                harnessServer.stop();
            }
            circuitRunner = null;
        });
    }
}
