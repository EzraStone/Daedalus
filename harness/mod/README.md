# Daedalus fidelity mod

A Fabric server mod that accepts a `.schem` over a socket, places it in a void
world, walks the input levers through every combination, and reports the output
lamp states.

The module targets Minecraft 1.20.1, Yarn build 10, Fabric Loader 0.19.3, and
Fabric API 0.92.11. Those versions are pinned in `gradle.properties` so local
and CI builds resolve the same game API.

## Building

```bash
cd harness/mod
./gradlew build
```

The current Loom toolchain runs Gradle on JDK 21 or newer. The compiled mod
still targets Java 17, which is the runtime required by Minecraft 1.20.1.

Drop the jar in a server's `mods/` alongside Fabric API, run the server with a
void world, then:

```bash
python harness/compare.py --cases 10000 --out agreement.json
```

## Protocol

See `../README.md`. One JSON object per line, request and response.
