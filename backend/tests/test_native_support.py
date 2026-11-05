"""Unit pins for the shared native binary discovery (Phase-7 extraction)."""

from __future__ import annotations

import os

from backend.engines.native_support import native_binary_candidates, native_find_binary


def test_candidates_cover_common_cmake_layouts(tmp_path):
    candidates = native_binary_candidates(str(tmp_path), "audio_server")
    executable = "audio_server.exe" if os.name == "nt" else "audio_server"
    assert candidates == [
        os.path.join(str(tmp_path), "bin", executable),
        os.path.join(str(tmp_path), executable),
        os.path.join(str(tmp_path), "Release", executable),
        os.path.join(str(tmp_path), "bin", "Release", executable),
    ]


def test_find_binary_prefers_bin_dir_and_returns_absolute(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    binary = bin_dir / ("audio_server.exe" if os.name == "nt" else "audio_server")
    binary.write_bytes(b"")
    assert native_find_binary(str(tmp_path), "audio_server") == os.path.abspath(binary)


def test_find_binary_raises_with_layout_context(tmp_path):
    try:
        native_find_binary(str(tmp_path), "audio_server")
    except RuntimeError as exc:
        assert "audio_server" in str(exc)
        assert str(tmp_path) in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected RuntimeError")
