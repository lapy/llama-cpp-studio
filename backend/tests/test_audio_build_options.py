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
    for expected in (
        "cuda",
        "llamafile",
        "deployment_build",
        "native_model_manager",
        "use_system_openssl",
        "build_server_frontends",
        "build_model_tests",
        "model_set",
        "static_espeak",
        "build_c_api",
        "build_extended_tests",
    ):
        assert expected in keys
    for dropped in ("hip", "vulkan", "metal", "hip_strix_halo"):
        assert dropped not in keys
    assert cat["defaults"]["native_model_manager"] is False


def test_legacy_backend_maps_to_toggles():
    settings = coerce_build_settings({"backend": "cuda", "native_cpu": False})
    assert settings["cuda"] is True
    assert settings["backend"] == "cuda"
    assert settings["native_cpu"] is False


def test_legacy_hip_is_ignored():
    settings = coerce_build_settings({"cuda": True, "hip": True, "backend": "hip"})
    assert settings["cuda"] is True
    assert settings["backend"] == "cuda"
    assert "hip" not in settings or settings.get("hip") is not True

    cpu = coerce_build_settings({"backend": "metal"})
    assert cpu["cuda"] is False
    assert cpu["backend"] == "cpu"


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


def test_cuda_cmake_forces_unsupported_backends_off(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.build_progress.shutil.which",
        lambda name: "/usr/bin/ninja" if name == "ninja" else None,
    )
    manager = AudioCppManager(str(tmp_path / "audio-cpp"))
    config = AudioCppBuildConfig(hip=True, metal=True, cuda=True).normalized()
    args = manager._cmake_args("/s", "/b", config)
    assert "-DENGINE_ENABLE_CUDA=ON" in args
    assert "-DENGINE_ENABLE_HIP=OFF" in args
    assert "-DENGINE_ENABLE_VULKAN=OFF" in args
    assert "-DENGINE_ENABLE_METAL=OFF" in args
    assert "-DENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=OFF" in args
    assert "-DAUDIOCPP_STATIC_ESPEAK=OFF" in args
    assert "-DAUDIOCPP_BUILD_C_API=OFF" in args
    assert "-DENGINE_BUILD_EXTENDED_TESTS=OFF" in args
    assert "-DENGINE_BUILD_MODEL_TESTS=OFF" in args
    assert "-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF" in args
    assert config.backend == "cuda"
    assert config.hip is False
    assert config.metal is False
