#!/usr/bin/env bash
# Build the harness mod and assemble the pinned Fabric server runtime.
#
# The POSIX counterpart of setup.ps1. Same versions, same server.properties,
# same layout on disk -- a runtime built by either script should be
# indistinguishable, and tests/test_harness_scripts.py checks that the two
# agree rather than trusting this comment.

set -euo pipefail
# shellcheck source=harness/server/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

accept_eula=0
for argument in "$@"; do
    case "$argument" in
        --accept-eula) accept_eula=1 ;;
        *) echo "unknown option: ${argument}" >&2; exit 2 ;;
    esac
done
if [ "$accept_eula" -ne 1 ]; then
    echo "Pass --accept-eula after reviewing https://aka.ms/MinecraftEULA" >&2
    exit 2
fi

repository="$(repository_dir)"
runtime="$(harness_dir)/runtime"
mods="${runtime}/mods"
mkdir -p "$mods"

minecraft="$(version_of minecraft)"
fabric_loader="$(version_of fabric_loader)"
fabric_launcher="$(version_of fabric_launcher)"
fabric_api="$(version_of fabric_api)"

# Gradle needs 21; the server itself runs on 17. Resolved separately so a
# machine with only one of them fails on the step that actually needs the other.
build_java="$(require_java 21)"
echo "Building the harness mod with ${build_java}"
JAVA_HOME="$(cd "$(dirname "$build_java")/.." && pwd)" \
    "${repository}/harness/mod/gradlew" -p "${repository}/harness/mod" build

harness_jar=""
for candidate in "${repository}"/harness/mod/build/libs/*.jar; do
    case "$candidate" in
        *-sources.jar) continue ;;
    esac
    [ -f "$candidate" ] || continue
    harness_jar="$candidate"
    break
done
if [ -z "$harness_jar" ]; then
    echo "the built harness jar was not found" >&2
    exit 1
fi
cp -f "$harness_jar" "${mods}/daedalus-harness.jar"

launcher="${runtime}/fabric-server-launch.jar"
if [ ! -f "$launcher" ]; then
    echo "Fetching the Fabric server launcher"
    fetch "https://meta.fabricmc.net/v2/versions/loader/${minecraft}/${fabric_loader}/${fabric_launcher}/server/jar" \
        "$launcher"
fi

if [ ! -f "${mods}/fabric-api.jar" ]; then
    echo "Fetching Fabric API ${fabric_api}"
    fetch "https://maven.fabricmc.net/net/fabricmc/fabric-api/fabric-api/${fabric_api}/fabric-api-${fabric_api}.jar" \
        "${mods}/fabric-api.jar"
fi

printf 'eula=true\n' > "${runtime}/eula.txt"
"$(harness_dir)/server-properties.sh" > "${runtime}/server.properties"

echo "Pinned Fabric server is ready at ${runtime}"
