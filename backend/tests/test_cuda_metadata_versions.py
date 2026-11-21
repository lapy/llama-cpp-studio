"""CUDA repository/archive mapping for the supported Ubuntu generations."""

from __future__ import annotations

import io

import pytest

from backend import cuda_metadata


@pytest.mark.parametrize(
    ("version", "repo"),
    [
        ("22.04", "ubuntu2204"),
        ("22.04.5", "ubuntu2204"),
        ("24.04", "ubuntu2404"),
        ("24.04.1", "ubuntu2404"),
        ("25.10", "ubuntu2404"),
        ("26.04", "ubuntu2604"),
        ("26.04.1", "ubuntu2604"),
        ("26.10", "ubuntu2604"),
    ],
)
def test_repo_for_ubuntu_version(version, repo):
    assert cuda_metadata.repo_for_ubuntu_version(version) == repo


def test_invalid_version_falls_back_to_the_oldest_repo():
    assert cuda_metadata.repo_for_ubuntu_version("") == "ubuntu2204"
    assert cuda_metadata.repo_for_ubuntu_version("24") == "ubuntu2204"


@pytest.mark.parametrize(
    ("repo", "target"),
    [
        ("ubuntu2204", "22.04"),
        ("ubuntu2404", "24.04"),
        ("ubuntu2604", "24.04"),
    ],
)
def test_archive_target_version(repo, target):
    assert cuda_metadata.get_archive_target_version(repo) == target


def test_get_ubuntu_version_reads_os_release(monkeypatch):
    monkeypatch.setattr(cuda_metadata.os.path, "exists", lambda path: path == "/etc/os-release")
    monkeypatch.setattr(
        "builtins.open",
        lambda *args, **kwargs: io.StringIO('NAME="Ubuntu"\nVERSION_ID="26.04"\n'),
    )
    assert cuda_metadata.get_ubuntu_version() == "ubuntu2604"


def test_get_ubuntu_version_defaults_without_os_release(monkeypatch):
    monkeypatch.setattr(cuda_metadata.os.path, "exists", lambda path: False)
    assert cuda_metadata.get_ubuntu_version() == "ubuntu2404"
