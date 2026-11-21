"""CUDA host compiler selection must stay inside nvcc's supported range."""

from __future__ import annotations

import pytest

from backend import cuda_host_compiler


@pytest.mark.parametrize(
    ("root", "expected"),
    [
        ("/app/data/cuda/cuda-12.8.2", (12, 8)),
        ("/app/data/cuda/cuda-13.0", (13, 0)),
        ("/usr/local/cuda-12.4.1", (12, 4)),
        ("/app/data/cuda/current", None),
        ("", None),
    ],
)
def test_cuda_toolkit_version(root, expected):
    assert cuda_host_compiler.cuda_toolkit_version(root) == expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ((12, 0), 12),
        ((12, 3), 12),
        ((12, 4), 13),
        ((12, 6), 13),
        ((12, 8), 14),
        ((12, 9), 14),
        ((13, 0), 15),
        (None, 12),
    ],
)
def test_max_supported_gcc(version, expected):
    assert cuda_host_compiler.max_supported_gcc(version) == expected


def _which(mapping):
    return lambda name: mapping.get(name)


def test_select_prefers_the_newest_supported_compiler(monkeypatch):
    monkeypatch.setattr(
        cuda_host_compiler.shutil,
        "which",
        _which(
            {
                "g++-14": "/usr/bin/g++-14",
                "g++-12": "/usr/bin/g++-12",
                "g++": "/usr/bin/g++",
            }
        ),
    )
    assert (
        cuda_host_compiler.select_cuda_host_compiler("/app/data/cuda/cuda-12.8.2")
        == "/usr/bin/g++-14"
    )


def test_select_falls_back_to_an_older_supported_compiler(monkeypatch):
    monkeypatch.setattr(
        cuda_host_compiler.shutil,
        "which",
        _which({"g++-12": "/usr/bin/g++-12", "g++": "/usr/bin/g++"}),
    )
    assert (
        cuda_host_compiler.select_cuda_host_compiler("/app/data/cuda/cuda-12.8.2")
        == "/usr/bin/g++-12"
    )


def test_select_does_not_use_unsupported_unversioned_gcc(monkeypatch):
    monkeypatch.setattr(
        cuda_host_compiler.shutil,
        "which",
        _which({"g++": "/usr/bin/g++"}),
    )
    assert cuda_host_compiler.select_cuda_host_compiler("/app/data/cuda/cuda-12.8.2") is None


def test_select_allows_unversioned_gcc_for_cuda_13(monkeypatch):
    monkeypatch.setattr(
        cuda_host_compiler.shutil,
        "which",
        _which({"g++": "/usr/bin/g++"}),
    )
    assert (
        cuda_host_compiler.select_cuda_host_compiler("/app/data/cuda/cuda-13.0") == "/usr/bin/g++"
    )
