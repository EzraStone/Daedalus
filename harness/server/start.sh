#!/usr/bin/env bash
# Run the pinned Fabric server with the harness mod loaded.
#
# The POSIX counterpart of start.ps1. Minecraft 1.20.1 runs on Java 17; the
# mod is built with 21. Asking for the wrong one here is the failure that
# looks like a corrupt jar, so it is checked before anything launches.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

maximum_heap="2G"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --max-heap) maximum_heap="${2:?--max-heap needs a size}"; shift 2 ;;
        --max-heap=*) maximum_heap="${1#*=}"; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

runtime="$(harness_dir)/runtime"
if [ ! -f "${runtime}/fabric-server-launch.jar" ]; then
    echo "server runtime is missing; run setup.sh --accept-eula first" >&2
    exit 1
fi

java="$(require_java 17)"
port="$(version_of harness_port)"

cd "$runtime"
exec "$java" \
    -Xms512M \
    "-Xmx${maximum_heap}" \
    "-Ddaedalus.harness.port=${port}" \
    -jar fabric-server-launch.jar nogui
