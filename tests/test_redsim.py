"""Tests for the verifier process boundary."""

from pathlib import Path

from daedalus.redsim import _target_binaries


def test_cargo_binary_paths_use_the_windows_executable_suffix():
    release, debug = _target_binaries(Path("target"), os_name="nt")

    assert release == Path("target/release/redsim.exe")
    assert debug == Path("target/debug/redsim.exe")


def test_cargo_binary_paths_remain_extensionless_on_unix():
    release, debug = _target_binaries(Path("target"), os_name="posix")

    assert release == Path("target/release/redsim")
    assert debug == Path("target/debug/redsim")
