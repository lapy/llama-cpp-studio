"""Per-engine CUDA toolkit binding for llama-family source builds."""

from types import SimpleNamespace

from backend.engines.llama_cpp.build_options import (
    coerce_build_settings,
    default_build_settings,
    settings_to_field_kwargs,
)
from backend.engines.python.cmake_build import CmakeBuildHost


def _fake_installer(root):
    class _Installer:
        def get_cuda_env(self, version=None):
            if version == "12.9":
                return {"CUDA_HOME": str(root), "PATH": f"{root}/bin"}
            return {}

    return _Installer()


def test_build_settings_round_trip_the_bound_toolkit():
    defaults = default_build_settings()
    assert defaults["cuda_toolkit"] == ""

    coerced = coerce_build_settings({"cuda_toolkit": "12.9"})
    assert coerced["cuda_toolkit"] == "12.9"
    assert settings_to_field_kwargs({"cuda_toolkit": "12.9"})["cuda_toolkit"] == "12.9"


def test_preferred_toolkit_resolves_and_is_validated(monkeypatch, tmp_path):
    cuda_root = tmp_path / "cuda-12.9"
    (cuda_root / "bin").mkdir(parents=True)
    (cuda_root / "bin" / "nvcc").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(
        "backend.cuda_installer.get_cuda_installer", lambda: _fake_installer(cuda_root)
    )

    host = SimpleNamespace(_verify_cuda_toolkit_complete=lambda _path: (True, []))
    available, root, error = CmakeBuildHost._check_cuda_toolkit_available(host, "12.9")

    assert available is True
    assert root == str(cuda_root)
    assert error is None


def test_preferred_toolkit_rejects_missing_or_incomplete(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "backend.cuda_installer.get_cuda_installer", lambda: _fake_installer(tmp_path / "missing")
    )
    host = SimpleNamespace(_verify_cuda_toolkit_complete=lambda _path: (True, []))

    available, root, error = CmakeBuildHost._check_cuda_toolkit_available(host, "12.9")
    assert available is False
    assert root is None
    assert "not installed" in error

    cuda_root = tmp_path / "cuda-12.9"
    (cuda_root / "bin").mkdir(parents=True)
    (cuda_root / "bin" / "nvcc").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(
        "backend.cuda_installer.get_cuda_installer", lambda: _fake_installer(cuda_root)
    )
    host = SimpleNamespace(_verify_cuda_toolkit_complete=lambda _path: (False, ["cudart"]))
    available, root, error = CmakeBuildHost._check_cuda_toolkit_available(host, "12.9")
    assert available is False
    assert "incomplete" in error
    assert "cudart" in error
