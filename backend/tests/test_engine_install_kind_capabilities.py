"""R74 pins: install-kind capabilities follow the declaration, not id tuples.

`_PYTHON_VENV_ENGINES` (orphan discovery + venv retry rule) and
`_CMAKE_BUILD_ENGINES` (cmake-editability) were engine-id tuples in shared code.
Both now derive from `EngineSpec.install_kind` via
`engine_ids_with_install_kind`, so a YAML engine that declares the same kind gets
the same behavior with no code edit.
"""

from __future__ import annotations

import pytest

from backend.engines import lifecycle as lifecycle_module
from backend.engines.registry import (
    ENGINE_REGISTRY,
    SHIPPED_ENGINE_IDS,
    EngineSpec,
    engine_ids_with_install_kind,
)
from backend.routes.llama_versions import _cmake_build_engines

VENV_ENGINES = {
    "lmdeploy",
    "1cat_vllm",
    "sglang",
    "sglang_v100",
    "sglang_sxm2",
    "sglang_sm70",
    "vllm",
}
CMAKE_ENGINES = {"llama_cpp", "ik_llama", "audio_cpp"}


def test_install_kind_helpers_match_the_historical_families():
    # Fixture personas may register at startup and add themselves to these
    # sets; the SHIPPED families (generated + definition-owned) stay exactly as
    # shipped.
    venv_ids = set(engine_ids_with_install_kind("python_venv"))
    assert venv_ids & SHIPPED_ENGINE_IDS == VENV_ENGINES
    assert set(engine_ids_with_install_kind("cmake")) & SHIPPED_ENGINE_IDS == CMAKE_ENGINES
    assert set(engine_ids_with_install_kind("prebuilt")) & SHIPPED_ENGINE_IDS == {"unsloth_llama"}
    assert engine_ids_with_install_kind("not-a-kind") == ()
    assert set(lifecycle_module.python_venv_engine_ids()) & SHIPPED_ENGINE_IDS == VENV_ENGINES
    assert set(_cmake_build_engines()) & SHIPPED_ENGINE_IDS == CMAKE_ENGINES


def _fake_spec(engine_id: str, install_kind: str) -> EngineSpec:
    return EngineSpec(
        id=engine_id,
        label=engine_id,
        repository_source=engine_id,
        install_kind=install_kind,
        runtime_kind="llama_server",
        scanner_kind="llama_help",
        active_path_fields=("binary_path",),
        artifact_formats=frozenset({"gguf"}),
        package_kinds=frozenset({"builtin"}),
        tasks=frozenset({"text-generation"}),
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
    )


def test_a_declared_venv_engine_becomes_retryable(monkeypatch):
    monkeypatch.setitem(ENGINE_REGISTRY, "fixture_venv", _fake_spec("fixture_venv", "python_venv"))
    row = {
        "version": "0.1.0",
        "install_type": "pip",
        "venv_path": "/opt/studio/fixture/0.1.0/venv",
    }
    assert lifecycle_module.engine_version_is_retryable("fixture_venv", row) is True
    # A cmake declaration keeps the cmake rule (no venv retryability).
    monkeypatch.setitem(ENGINE_REGISTRY, "fixture_cmake", _fake_spec("fixture_cmake", "cmake"))
    assert lifecycle_module.engine_version_is_retryable("fixture_cmake", row) is False


def test_a_declared_cmake_engine_becomes_cmake_editable(monkeypatch):
    monkeypatch.setitem(ENGINE_REGISTRY, "fixture_cmake", _fake_spec("fixture_cmake", "cmake"))
    assert "fixture_cmake" in _cmake_build_engines()
    monkeypatch.setitem(
        ENGINE_REGISTRY, "fixture_prebuilt", _fake_spec("fixture_prebuilt", "prebuilt")
    )
    assert "fixture_prebuilt" not in _cmake_build_engines()


def test_no_family_tuples_remain_in_shared_code():
    """The historical tuples must not come back as literals."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    lifecycle_source = (root / "backend/engines/lifecycle.py").read_text(encoding="utf-8")
    routes_source = (root / "backend/routes/llama_versions.py").read_text(encoding="utf-8")
    assert "_PYTHON_VENV_ENGINES" not in lifecycle_source
    assert "_CMAKE_BUILD_ENGINES" not in routes_source
