"""CUDA/glibc C23 math compatibility decision and shim wiring."""

from __future__ import annotations

import os

import pytest

from backend import cuda_glibc_compat


@pytest.mark.parametrize(
    ("glibc", "cuda", "expected"),
    [
        ((2, 39), (12, 8), False),
        ((2, 41), (12, 8), True),
        ((2, 43), (12, 8), True),
        ((2, 43), (12, 9), True),
        ((2, 43), (13, 0), True),
        ((2, 43), (13, 1), True),
        ((2, 43), (13, 2), False),
        ((2, 43), (13, 3), False),
        ((2, 43), None, True),
    ],
)
def test_needs_cuda_c23_shim(glibc, cuda, expected):
    assert cuda_glibc_compat.needs_cuda_c23_shim(glibc, cuda) is expected


def test_shim_flag_points_at_an_existing_header():
    flag = cuda_glibc_compat.cuda_c23_shim_flag("/app/data/cuda/cuda-12.8.2")
    assert flag.startswith("-Xcompiler=-include,")
    path = flag.split(",", 1)[1]
    assert path.endswith("glibc_c23_math_compat.h")
    assert os.path.isfile(path)


def test_unknown_glibc_on_linux_is_safe_to_shim(monkeypatch):
    monkeypatch.setattr(cuda_glibc_compat.sys, "platform", "linux")
    monkeypatch.setattr(cuda_glibc_compat, "host_glibc_version", lambda: None)
    assert cuda_glibc_compat.needs_cuda_c23_shim(None, (12, 8)) is True


def test_non_linux_never_shims(monkeypatch):
    monkeypatch.setattr(cuda_glibc_compat.sys, "platform", "win32")
    assert cuda_glibc_compat.needs_cuda_c23_shim((2, 43), (12, 8)) is False
