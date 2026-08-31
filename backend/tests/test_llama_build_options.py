"""Tests for the llama.cpp CMake build-option catalog."""

from backend.llama_build_options import (
    BUILD_OPTIONS,
    append_generic_cmake_flags,
    catalog_for_ui,
    coerce_build_settings,
    default_build_settings,
    settings_to_field_kwargs,
    stored_config_to_settings,
)
from backend.llama_manager import BuildConfig
from backend.routes.llama_versions import (
    _build_config_from_any,
    _build_config_from_settings,
)


def test_catalog_exposes_major_backends():
    cat = catalog_for_ui("llama_cpp")
    assert cat["categories"]
    backends = next(c for c in cat["categories"] if c["id"] == "backends")
    keys = {o["key"] for o in backends["options"]}
    for expected in ("cuda", "hip", "vulkan", "metal", "sycl", "opencl", "blas"):
        assert expected in keys
    assert "openblas" not in keys  # legacy alias hidden from UI


def test_ik_catalog_includes_iqk_excludes_blas():
    ik = catalog_for_ui("ik_llama")
    ids = {c["id"] for c in ik["categories"]}
    assert "iqk" in ids
    assert "opencl" not in ids
    keys = {o["key"] for c in ik["categories"] for o in c["options"]}
    assert "iqk_mul_mat" in keys
    assert "iqk_flash_attention" in keys
    assert "blas" not in keys
    assert "build_tools" not in keys
    assert "build_examples" in keys


def test_ik_cmake_flags_use_hipblas_and_cuda_use_graphs():
    args = []

    def set_flag(flag, value):
        args.append(f"-D{flag}={'ON' if value else 'OFF'}")

    cfg = BuildConfig(enable_cuda=True, enable_hip=True, enable_cuda_graphs=True)
    append_generic_cmake_flags(args, cfg, set_flag=set_flag, engine="ik_llama")
    joined = " ".join(args)
    assert "-DGGML_HIPBLAS=ON" in joined
    assert "-DGGML_HIP=ON" not in joined
    assert "-DGGML_CUDA_USE_GRAPHS=ON" in joined
    assert "-DGGML_CUDA_GRAPHS=ON" not in joined
    assert "-DGGML_IQK_MUL_MAT=ON" in joined


def test_catalog_marks_advanced_collapsed():
    cat = catalog_for_ui("llama_cpp")
    backends = next(c for c in cat["categories"] if c["id"] == "backends")
    assert backends["collapsed"] is False
    primary = {o["key"] for o in backends["options"] if o.get("primary")}
    extra = {o["key"] for o in backends["options"] if not o.get("primary")}
    assert "cuda" in primary
    assert "rpc" in extra
    for c in cat["categories"]:
        if c["id"] == "backends":
            continue
        assert c["collapsed"] is True, c["id"]


def test_defaults_cover_all_options():
    defaults = default_build_settings()
    assert defaults["build_type"] == "Release"
    for opt in BUILD_OPTIONS:
        assert opt.key in defaults


def test_openblas_legacy_enables_blas():
    settings = coerce_build_settings({"openblas": True})
    assert settings["blas"] is True
    assert settings["blas_vendor"] == "OpenBLAS"
    cfg = BuildConfig(**settings_to_field_kwargs(settings))
    assert cfg.enable_blas is True
    assert cfg.blas_vendor == "OpenBLAS"


def test_settings_to_build_config_backends():
    cfg = _build_config_from_settings(
        {"cuda": True, "vulkan": True, "flash_attention": True, "native": False}
    )
    assert cfg.enable_cuda is True
    assert cfg.enable_vulkan is True
    assert cfg.enable_flash_attention is True
    assert cfg.enable_native is False


def test_stored_enable_shape_still_loads():
    cfg = _build_config_from_any(
        {"enable_cuda": True, "enable_native": False, "build_type": "Debug"}
    )
    assert cfg.enable_cuda is True
    assert cfg.enable_native is False
    assert cfg.build_type == "Debug"


def test_stored_config_to_settings_maps_enable_fields():
    settings = stored_config_to_settings(
        {"enable_cuda": True, "enable_native": False, "build_type": "Debug"}
    )
    assert settings["cuda"] is True
    assert settings["native"] is False
    assert settings["build_type"] == "Debug"

    both = stored_config_to_settings({"cuda": False, "enable_cuda": True})
    assert both["cuda"] is False


def test_append_generic_cmake_flags_gates_cuda_children():
    args = []

    def set_flag(flag, value):
        args.append(f"-D{flag}={'ON' if value else 'OFF'}")

    cfg = BuildConfig(enable_cuda=False, enable_vulkan=True)
    append_generic_cmake_flags(args, cfg, set_flag=set_flag, engine="llama_cpp")
    joined = " ".join(args)
    assert "-DGGML_VULKAN=ON" in joined
    # CUDA FA_ALL is special-cased elsewhere; FA itself is gated off when CUDA off
    assert "-DGGML_CUDA_FA=OFF" in joined
    assert "GGML_CUDA_PEER_MAX_BATCH_SIZE" not in joined


def test_build_options_api(client):
    r = client.get("/api/llama-versions/build-options", params={"engine": "ik_llama"})
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "ik_llama"
    assert any(c["id"] == "iqk" for c in body["categories"])
    assert body["defaults"]["build_server"] is True