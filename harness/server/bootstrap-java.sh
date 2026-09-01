#!/usr/bin/env bash
# Fetch a Temurin JDK into .tools when the machine has none of the right major.
#
# The Windows counterpart pins two builds by URL and SHA-256, which is the
# right shape when you are fetching a binary. It cannot be copied here: those
# checksums are for the Windows x64 archives, and Linux and macOS on x64 and
# arm64 are four more platforms whose hashes are not written down anywhere in
# this repository.
#
# Rather than invent them, this resolves the build through the Adoptium API and
# verifies the download against the checksum the API reports for it. Be clear
# about what that is worth: it detects a corrupted or truncated download, and
# it does not pin the artifact the way the Windows script does -- a compromised
# API would hand out a matching pair. On a machine that already has a JDK of
# the right major version, none of this runs at all, which is the better path
# and the one setup.sh prefers.

set -euo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

API="${ADOPTIUM_API:-https://api.adoptium.net}"

adoptium_os() {
    case "$(uname -s)" in
        Linux) echo linux ;;
        Darwin) echo mac ;;
        *) echo "unsupported operating system: $(uname -s)" >&2; return 1 ;;
    esac
}

adoptium_arch() {
    case "$(uname -m)" in
        x86_64 | amd64) echo x64 ;;
        aarch64 | arm64) echo aarch64 ;;
        *) echo "unsupported architecture: $(uname -m)" >&2; return 1 ;;
    esac
}

install_jdk() {
    local major="$1"
    local tools destination
    tools="$(repository_dir)/.tools"
    destination="${tools}/jdk-${major}"

    if [ -x "${destination}/bin/java" ]; then
        echo "Java ${major} is already unpacked at ${destination}"
        return 0
    fi

    local os arch url_query metadata archive expected actual
    os="$(adoptium_os)"
    arch="$(adoptium_arch)"
    url_query="${API}/v3/assets/latest/${major}/hotspot?os=${os}&architecture=${arch}&image_type=jdk"

    mkdir -p "$tools"
    metadata="${tools}/jdk-${major}.json"
    echo "Resolving Temurin ${major} for ${os}/${arch}"
    fetch "$url_query" "$metadata"

    local link
    link="$(PY_METADATA="$metadata" python3 - <<'PY'
import json
import os

with open(os.environ["PY_METADATA"], encoding="utf-8") as stream:
    assets = json.load(stream)
if not assets:
    raise SystemExit("the Adoptium API returned no build for this platform")
package = assets[0]["binary"]["package"]
print(package["link"])
print(package.get("checksum", ""))
PY
)"
    url="$(printf '%s\n' "$link" | sed -n 1p)"
    expected="$(printf '%s\n' "$link" | sed -n 2p)"
    if [ -z "$expected" ]; then
        echo "the Adoptium API gave no checksum for ${url}; refusing to install" >&2
        return 1
    fi

    archive="${tools}/jdk-${major}.tar.gz"
    echo "Downloading ${url}"
    fetch "$url" "$archive"
    actual="$(sha256_of "$archive")"
    if [ "$actual" != "$expected" ]; then
        rm -f "$archive"
        echo "checksum mismatch for Java ${major}: expected ${expected}, got ${actual}" >&2
        return 1
    fi

    # Temurin archives carry a versioned top directory; strip it so the result
    # lands where find_java looks.
    rm -rf "$destination"
    mkdir -p "$destination"
    tar -xzf "$archive" -C "$destination" --strip-components 1
    # macOS builds nest the runtime under Contents/Home.
    if [ ! -x "${destination}/bin/java" ] && [ -x "${destination}/Contents/Home/bin/java" ]; then
        mv "${destination}/Contents/Home" "${destination}.home"
        rm -rf "$destination"
        mv "${destination}.home" "$destination"
    fi
    if [ ! -x "${destination}/bin/java" ]; then
        echo "Java ${major} did not unpack to ${destination}" >&2
        return 1
    fi
    rm -f "$archive" "$metadata"
    echo "Java ${major} is ready at ${destination}"
}

wanted=("$@")
if [ "${#wanted[@]}" -eq 0 ]; then
    wanted=(17 21)
fi
for major in "${wanted[@]}"; do
    if find_java "$major" >/dev/null 2>&1; then
        echo "Java ${major} is already available; nothing to fetch"
        continue
    fi
    install_jdk "$major"
done
