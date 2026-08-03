package dev.daedalus.harness;

/**
 * Entry point. Starts the socket listener when the server comes up.
 *
 * <p>The port is read from {@code daedalus.harness.port} so several servers can
 * run in parallel — verifier throughput is the bottleneck in
 * {@code daedalus.train.loop}, but game throughput is the bottleneck here, and
 * the only way to raise it is more servers.
 */
public final class HarnessMod {
    public static final int DEFAULT_PORT = 25599;

    private static HarnessServer server;

    public static void onInitializeServer() {
        int port = Integer.getInteger("daedalus.harness.port", DEFAULT_PORT);
        server = new HarnessServer(port, new CircuitRunner());
        Thread thread = new Thread(server, "daedalus-harness");
        thread.setDaemon(true);
        thread.start();
        System.out.println("[daedalus] fidelity harness listening on " + port);
    }

    public static void onStopServer() {
        if (server != null) {
            server.stop();
        }
    }
}
