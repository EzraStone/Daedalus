# Daedalus fidelity mod

A Fabric server mod that accepts a `.schem` over a socket, places it in a void
world, walks the input levers through every combination, and reports the output
lamp states.

**This is the untested half of the harness.** It was written against the Fabric
API but has never been compiled or run — the environment it was authored in has
no Minecraft server, no Gradle and no Fabric loader. It is committed as source,
with the protocol pinned, so `harness/compare.py` has something concrete to talk
to and so the remaining work is visible rather than implied. Treat the Java here
as a specification of the protocol, not as a working artifact.

## Building

```bash
cd harness/mod
./gradlew build          # needs JDK 17+ and network access to the Fabric maven
```

Drop the jar in a server's `mods/` alongside Fabric API, run the server with a
void world, then:

```bash
python harness/compare.py --cases 10000 --out agreement.json
```

## Protocol

See `../README.md`. One JSON object per line, request and response.
