package dev.daedalus.harness;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Socket front end for the fidelity harness.
 *
 * <p>One JSON object per line, request and response, so the Python side needs
 * no framing logic and a human can drive it with netcat when something is
 * wrong.
 *
 * <p>Everything that touches the world is handed to the server thread. Minecraft
 * is not thread-safe, and placing blocks from the socket thread is the kind of
 * bug that shows up as a wrong agreement number rather than as a crash — which
 * is far worse, because the agreement number is the thing this harness exists
 * to establish.
 */
public final class HarnessServer implements Runnable {
    private final int port;
    private final CircuitRunner runner;
    private volatile boolean running = true;
    private volatile ServerSocket listener;

    public HarnessServer(int port, CircuitRunner runner) {
        this.port = port;
        this.runner = runner;
    }

    public void stop() {
        running = false;
        ServerSocket socket = listener;
        if (socket != null) {
            try {
                socket.close();
            } catch (Exception ignored) {
                // Closing an already closed listener is harmless during shutdown.
            }
        }
    }

    @Override
    public void run() {
        try (ServerSocket server = new ServerSocket(port)) {
            listener = server;
            while (running) {
                try (Socket socket = server.accept()) {
                    handle(socket);
                } catch (Exception e) {
                    // One bad client must not take the harness down mid-run;
                    // a 10k-case sweep is expensive to restart.
                    System.err.println("[daedalus] client error: " + e.getMessage());
                }
            }
        } catch (SocketException e) {
            if (running) {
                System.err.println("[daedalus] listener failed: " + e.getMessage());
            }
        } catch (Exception e) {
            System.err.println("[daedalus] listener failed: " + e.getMessage());
        } finally {
            listener = null;
        }
    }

    private void handle(Socket socket) throws Exception {
        BufferedReader in = new BufferedReader(
                new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
        BufferedWriter out = new BufferedWriter(
                new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
        String line;
        while ((line = in.readLine()) != null) {
            String response;
            try {
                response = dispatch(Json.parse(line));
            } catch (Exception e) {
                response = "{\"error\":" + Json.quote(String.valueOf(e.getMessage())) + "}";
            }
            out.write(response);
            out.write('\n');
            out.flush();
        }
    }

    private String dispatch(Json.Obj request) throws Exception {
        String op = request.string("op");
        switch (op) {
            case "place": {
                String id = request.string("id");
                byte[] schematic = java.util.Base64.getDecoder()
                        .decode(request.string("schematic"));
                runner.place(id, schematic);
                return "{\"id\":" + Json.quote(id) + ",\"placed\":true}";
            }
            case "test": {
                String id = request.string("id");
                List<int[]> levers = request.positions("levers");
                List<int[]> lamps = request.positions("lamps");
                CircuitRunner.Result result = runner.sweep(id, levers, lamps);
                StringBuilder sb = new StringBuilder();
                sb.append("{\"id\":").append(Json.quote(id)).append(",\"rows\":[");
                List<String> rows = new ArrayList<>();
                for (int[] row : result.rows) {
                    StringBuilder r = new StringBuilder("[");
                    for (int i = 0; i < row.length; i++) {
                        if (i > 0) {
                            r.append(',');
                        }
                        r.append(row[i]);
                    }
                    rows.add(r.append(']').toString());
                }
                sb.append(String.join(",", rows));
                sb.append("],\"settled\":").append(result.settled).append('}');
                return sb.toString();
            }
            case "ping":
                return "{\"pong\":true}";
            default:
                throw new IllegalArgumentException("unknown op " + op);
        }
    }
}
