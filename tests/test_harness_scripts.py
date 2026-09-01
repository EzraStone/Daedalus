"""The two halves of the harness bootstrap have to agree.

`setup.ps1` and `setup.sh` build the same server two ways. Nothing stops them
drifting apart except a test that reads both, and a runtime that differs
between platforms would show up as a *fidelity* disagreement -- the one number
this repository is waiting on -- with nothing pointing at the real cause.

These are text-level checks on purpose. Running either script needs a network,
a Gradle toolchain and a Minecraft server, none of which are available here;
what can be checked without them is that the two scripts say the same thing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[1] / "harness" / "server"
POSIX = ("common.sh", "setup.sh", "start.sh", "smoke.sh", "server-properties.sh")


def properties_from_powershell() -> dict[str, str]:
    """The server.properties block embedded in setup.ps1."""
    text = (SERVER / "setup.ps1").read_text(encoding="utf-8")
    body = re.search(r"@'\n(.*?)\n'@", text, re.DOTALL)
    assert body, "no here-string in setup.ps1"
    return dict(
        line.split("=", 1) for line in body.group(1).splitlines() if "=" in line
    )


def properties_from_posix() -> dict[str, str]:
    out = subprocess.run(
        ["bash", str(SERVER / "server-properties.sh")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


class TestParity:
    def test_both_halves_write_the_same_server_properties(self):
        assert properties_from_posix() == properties_from_powershell()

    def test_the_settings_that_make_a_circuit_the_only_moving_thing_are_set(self):
        # Not parity for its own sake. A world with mobs, structures or a
        # ticking watchdog is one where a lamp can change for a reason that is
        # not the circuit, and every such row would be scored as a divergence.
        properties = properties_from_posix()
        assert properties["level-type"] == "minecraft:flat"
        assert properties["spawn-monsters"] == "false"
        assert properties["spawn-animals"] == "false"
        assert properties["generate-structures"] == "false"
        assert properties["spawn-protection"] == "0"
        assert properties["max-tick-time"] == "-1"
        assert '"biome":"minecraft:the_void"' in properties["generator-settings"]

    def test_the_server_is_not_reachable_from_off_the_machine(self):
        # It runs with online-mode off and secure profiles disabled, which is
        # fine on loopback and an open door anywhere else.
        properties = properties_from_posix()
        assert properties["server-ip"] == "127.0.0.1"
        assert properties["online-mode"] == "false"
        assert properties["enable-rcon"] == "false"
        assert properties["enable-query"] == "false"


class TestScripts:
    @pytest.mark.parametrize("name", POSIX)
    def test_it_parses(self, name):
        subprocess.run(["bash", "-n", str(SERVER / name)], check=True)

    @pytest.mark.parametrize("name", POSIX)
    def test_it_is_executable(self, name):
        assert (SERVER / name).stat().st_mode & 0o111, name

    @pytest.mark.parametrize("name", POSIX)
    def test_it_stops_on_the_first_error(self, name):
        # Without `set -e` a failed download leaves a half-built runtime that
        # then fails much later, somewhere unrelated.
        assert "set -euo pipefail" in (SERVER / name).read_text(encoding="utf-8")

    def test_every_windows_script_has_a_posix_counterpart(self):
        windows = {p.stem for p in SERVER.glob("*.ps1")}
        posix = {p.stem for p in SERVER.glob("*.sh")}
        assert windows - posix == set(), f"no POSIX equivalent for {windows - posix}"


class TestVersions:
    def test_both_halves_read_the_same_versions_file(self):
        assert "versions.properties" in (SERVER / "setup.ps1").read_text(encoding="utf-8")
        assert "versions.properties" in (SERVER / "common.sh").read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "key", ["minecraft", "fabric_loader", "fabric_launcher", "fabric_api", "harness_port"]
    )
    def test_the_posix_reader_returns_what_the_file_holds(self, key):
        wanted = dict(
            line.split("=", 1)
            for line in (SERVER / "versions.properties").read_text().splitlines()
            if "=" in line
        )[key]
        got = subprocess.run(
            ["bash", "-c", f'source "{SERVER}/common.sh"; version_of {key}'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert got == wanted

    def test_a_missing_key_fails_loudly(self):
        result = subprocess.run(
            ["bash", "-c", f'source "{SERVER}/common.sh"; version_of nonesuch'],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "nonesuch" in result.stderr


class TestJavaDiscovery:
    def test_it_reads_past_a_jvm_options_notice(self, tmp_path):
        # A JVM with JAVA_TOOL_OPTIONS set -- the normal state in a container
        # or on a CI runner -- prints "Picked up ..." before the version line.
        # Reading only the first line reports no Java on a machine that has one.
        fake = tmp_path / "java"
        fake.write_text(
            '#!/usr/bin/env bash\n'
            'echo "Picked up JAVA_TOOL_OPTIONS: -Dfoo=bar" >&2\n'
            'echo \'openjdk version "21.0.10" 2026-01-20\' >&2\n'
        )
        fake.chmod(0o755)
        got = subprocess.run(
            ["bash", "-c", f'source "{SERVER}/common.sh"; java_major "{fake}"'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert got == "21"

    def test_an_unrunnable_java_reports_nothing_rather_than_failing(self, tmp_path):
        fake = tmp_path / "java"
        fake.write_text("#!/usr/bin/env bash\nexit 1\n")
        fake.chmod(0o755)
        result = subprocess.run(
            ["bash", "-c", f'source "{SERVER}/common.sh"; java_major "{fake}"'],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    @pytest.mark.skipif(shutil.which("java") is None, reason="no java on this machine")
    def test_it_finds_the_java_this_machine_has(self):
        major = subprocess.run(
            ["bash", "-c", f'source "{SERVER}/common.sh"; java_major "$(command -v java)"'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        found = subprocess.run(
            ["bash", "-c", f'source "{SERVER}/common.sh"; find_java {major}'],
            capture_output=True,
            text=True,
        )
        assert found.returncode == 0
        assert Path(found.stdout.strip()).exists()

    def test_a_version_that_is_not_installed_explains_where_it_looked(self):
        result = subprocess.run(
            ["bash", "-c", f'source "{SERVER}/common.sh"; require_java 99'],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert ".tools/jdk-99" in result.stderr
        assert "adoptium" in result.stderr.lower()


class TestJavaBootstrap:
    """The download path, driven against a local stand-in for the Adoptium API.

    Worth the setup: this is the one script here that fetches a binary and
    then trusts it, so the interesting cases are the ones where the bytes are
    wrong. `$ADOPTIUM_API` exists precisely so they can be provoked.
    """

    @staticmethod
    def _fake_jdk(root: Path, major: str) -> Path:
        """A tarball shaped like a Temurin archive: one versioned top directory."""
        import tarfile

        top = root / f"jdk-{major}.0.1+9"
        (top / "bin").mkdir(parents=True)
        java = top / "bin" / "java"
        java.write_text(f'#!/usr/bin/env bash\necho \'openjdk version "{major}.0.1"\' >&2\n')
        java.chmod(0o755)
        archive = root / "jdk.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(top, arcname=top.name)
        return archive

    @staticmethod
    def _serve(directory: Path):
        import functools
        import http.server
        import threading

        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(directory)
        )
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def _run(self, tmp_path, checksum: str | None, major: str = "42"):
        import hashlib
        import json
        import os

        served = tmp_path / "served"
        served.mkdir()
        archive = self._fake_jdk(tmp_path, major)
        shutil.copy(archive, served / "jdk.tar.gz")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()

        server = self._serve(served)
        port = server.server_address[1]
        package: dict[str, str] = {"link": f"http://127.0.0.1:{port}/jdk.tar.gz"}
        if checksum is not None:
            package["checksum"] = digest if checksum == "good" else checksum
        asset = [{"binary": {"package": package}}]

        api = served / "v3" / "assets" / "latest" / major / "hotspot"
        api.parent.mkdir(parents=True)
        api.write_text(json.dumps(asset))

        repository = tmp_path / "repo"
        (repository / "harness").mkdir(parents=True)
        shutil.copytree(SERVER, repository / "harness" / "server")

        environment = dict(os.environ)
        # SimpleHTTPRequestHandler ignores the query string, so the resolver
        # lands on the file regardless of the os/architecture it appends.
        environment["ADOPTIUM_API"] = f"http://127.0.0.1:{port}"
        environment["NO_PROXY"] = "127.0.0.1,localhost"
        environment["no_proxy"] = "127.0.0.1,localhost"
        try:
            return subprocess.run(
                ["bash", str(repository / "harness" / "server" / "bootstrap-java.sh"), major],
                capture_output=True,
                text=True,
                env=environment,
            ), repository
        finally:
            # Both, and in this order. shutdown() stops the accept loop but
            # leaves the listening socket open, and pytest turns the resulting
            # unraisable ResourceWarning into a failure.
            server.shutdown()
            server.server_close()

    def test_a_good_download_unpacks_where_find_java_looks(self, tmp_path):
        result, repository = self._run(tmp_path, "good")
        assert result.returncode == 0, result.stderr
        assert (repository / ".tools" / "jdk-42" / "bin" / "java").exists()

    def test_a_corrupted_download_is_rejected_and_not_left_behind(self, tmp_path):
        result, repository = self._run(tmp_path, "0" * 64)
        assert result.returncode != 0
        assert "checksum mismatch" in result.stderr
        assert not (repository / ".tools" / "jdk-42" / "bin" / "java").exists()
        assert not (repository / ".tools" / "jdk-42.tar.gz").exists()

    def test_a_build_with_no_published_checksum_is_refused(self, tmp_path):
        # Installing it anyway would mean running an unverified binary, which
        # is worse than telling someone to fetch a JDK themselves.
        result, _repository = self._run(tmp_path, None)
        assert result.returncode != 0
        assert "no checksum" in result.stderr

    def test_it_does_not_download_a_java_the_machine_already_has(self):
        major = subprocess.run(
            ["bash", "-c", f'source "{SERVER}/common.sh"; java_major "$(command -v java)"'],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not major:
            pytest.skip("no java on this machine")
        result = subprocess.run(
            ["bash", str(SERVER / "bootstrap-java.sh"), major],
            capture_output=True,
            text=True,
            env={**dict(__import__("os").environ), "ADOPTIUM_API": "http://127.0.0.1:1"},
        )
        assert result.returncode == 0
        assert "already available" in result.stdout
