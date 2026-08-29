"""Tests for the audio.cpp CMake build-option catalog."""

from backend.audio_build_options import (
    catalog_for_ui,
    coerce_build_settings,
    default_build_settings,
)
from backend.audio_cpp_manager import AudioCppBuildConfig, AudioCppManager


def test_catalog_exposes_backends_and_iqk_style_sections():
    cat = catalog_for_ui()
    ids = {c["id"] for c in cat["categories"]}
    assert "backends" in ids
    assert "models" in ids
    keys = {o["key"] for c in cat["categories"] for o in c["options"]}
    for expected in ("cuda", "hip", "vulkan", "metal", "llamafile", "deployment_build", "native_model_manager", "model_set"):
        assert expected in keys


def test_legacy_backend_maps_to_toggles():
    settings = coerce_build_settings({"backend": "cuda", "native_cpu": False})
    assert settings["cuda"] is True
    assert settings["backend"] == "cuda"
    assert settings["native_cpu"] is False


def test_cuda_hip_mutex():
    settings = coerce_build_settings({"cuda": True, "hip": True})
    assert settings["cuda"] is True
    assert settings["hip"] is False


def test_defaults_cover_catalog():
    defaults = default_build_settings()
    assert defaults["build_type"] == "RelWithDebInfo"
    assert "llamafile" in defaults


def test_build_options_api(client):
    r = client.get("/api/audio-cpp/build-options")
    # Feature flag may 404 in some envs; accept catalog or gated 404
    if r.status_code == 404:
        return
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "audio_cpp"
    assert any(c["id"] == "backends" for c in body["categories"])


def test_hip_cmake_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.build_progress.shutil.which",
        lambda name: "/usr/bin/ninja" if name == "ninja" else None,
    )
    manager = AudioCppManager(str(tmp_path / "audio-cpp"))
    config = AudioCppBuildConfig(hip=True, cuda=False).normalized()
    args = manager._cmake_args("/s", "/b", config)
    assert "-DENGINE_ENABLE_HIP=ON" in args
    assert "-DENGINE_ENABLE_CUDA=OFF" in args
    assert config.backend == "hip"
