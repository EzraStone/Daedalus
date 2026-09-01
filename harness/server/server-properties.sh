#!/usr/bin/env bash
# The server configuration, in one place, printed to stdout.
#
# A separate script rather than a heredoc inside setup.sh so that the parity
# test can diff it against the block in setup.ps1 without running either. The
# settings are not arbitrary: a void superflat world with no mobs, no
# structures and no spawn protection is what makes a placed circuit the only
# thing in the world that can change, and max-tick-time=-1 stops the watchdog
# killing a server that is deliberately being asked to settle an oscillator.

set -euo pipefail

cat <<'PROPERTIES'
allow-flight=true
difficulty=peaceful
enable-command-block=false
enable-query=false
enable-rcon=false
enforce-secure-profile=false
force-gamemode=true
function-permission-level=2
gamemode=creative
generate-structures=false
generator-settings={"layers":[{"block":"minecraft:air","height":1}],"biome":"minecraft:the_void"}
hardcore=false
level-name=harness-world
level-type=minecraft:flat
max-players=1
max-tick-time=-1
motd=Daedalus fidelity harness
online-mode=false
pause-when-empty-seconds=-1
player-idle-timeout=0
prevent-proxy-connections=false
pvp=false
server-ip=127.0.0.1
server-port=25565
simulation-distance=2
spawn-animals=false
spawn-monsters=false
spawn-npcs=false
spawn-protection=0
sync-chunk-writes=true
view-distance=2
PROPERTIES
