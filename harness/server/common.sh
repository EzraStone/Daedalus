#!/usr/bin/env bash
# Shared helpers for the POSIX half of the harness bootstrap.
#
# The Windows scripts next door are the original and stay authoritative for
# what the runtime looks like; these exist because the number this harness is
# meant to produce -- sim/game agreement -- was reachable only from a Windows
# desktop, which rules out Linux, macOS and every CI runner. Nothing here
# changes the server's configuration; it reproduces it.

set -euo pipefail

harness_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

repository_dir() {
    cd "$(harness_dir)/../.." && pwd
}

# Read one key from versions.properties. Both halves read the same file, so the
# Minecraft, loader and API versions cannot drift apart between platforms.
version_of() {
    local key="$1" file
    file="$(harness_dir)/versions.properties"
    local value
    value="$(sed -n "s/^${key}=\(.*\)$/\1/p" "$file" | head -n 1)"
    if [ -z "$value" ]; then
        echo "no '${key}' in ${file}" >&2
        return 1
    fi
    printf '%s\n' "$value"
}

# The major version of a java executable, or nothing if it cannot be run.
#
# Scans every line rather than the first. A JVM with JAVA_TOOL_OPTIONS or
# _JAVA_OPTIONS set -- which is the normal state inside a container or on a
# CI runner -- prints "Picked up ..." before the version, and reading only
# line one there reports no Java at all on a machine that has one.
java_major() {
    local java="$1" output
    output="$("$java" -version 2>&1)" || return 0
    printf '%s\n' "$output" | sed -n 's/.*version "\([0-9][0-9]*\).*/\1/p' | head -n 1
}

# Find a java of exactly this major version: an explicitly pinned toolchain
# first, then JAVA_HOME, then whatever is on PATH.
#
# Preferring an installed JDK is not laziness. The Windows script pins two
# Temurin builds by URL and SHA-256, which is the right thing when you are
# fetching a binary -- but the checksums are per platform, and inventing
# Linux and macOS ones without being able to verify them would be worse than
# not pinning at all. A JDK the machine already has needs no checksum.
find_java() {
    local want="$1" candidate
    for candidate in \
        "$(repository_dir)/.tools/jdk-${want}/bin/java" \
        "${JAVA_HOME:-/nonexistent}/bin/java" \
        "$(command -v java 2>/dev/null || true)"; do
        [ -x "$candidate" ] || continue
        if [ "$(java_major "$candidate")" = "$want" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

require_java() {
    local want="$1" java
    if java="$(find_java "$want")"; then
        printf '%s\n' "$java"
        return 0
    fi
    cat >&2 <<MSG
No Java ${want} found. Looked in .tools/jdk-${want}, \$JAVA_HOME and \$PATH.

Install one from your package manager or https://adoptium.net/, or unpack a
Temurin ${want} build into $(repository_dir)/.tools/jdk-${want}.
MSG
    return 1
}

fetch() {
    local url="$1" destination="$2"
    mkdir -p "$(dirname "$destination")"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --silent --show-error --output "$destination" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --quiet --output-document "$destination" "$url"
    else
        echo "need curl or wget to download ${url}" >&2
        return 1
    fi
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}
