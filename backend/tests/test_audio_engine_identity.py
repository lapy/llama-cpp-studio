"""Audio engine identity comes from definitions, not a literal engine id."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import backend.engines.audio_cpp.identity as identity
import backend.model_catalog.audio_cpp_provider as provider_module
import backend.services.audio_model_installer as installer_module


def test_audio_capability_resolution_is_definition_driven(monkeypatch):
    assert "audio_cpp" in identity.audio_engine_ids()
    assert identity.default_audio_engine_id() == "audio_cpp"
    assert identity.is_audio_engine_id("audio_cpp") is True

    import backend.engines.registry as registry

    monkeypatch.setitem(
        registry.ENGINE_REGISTRY,
        "fixture_audio",
        SimpleNamespace(runtime_kind="audio_cpp"),
    )
    assert "fixture_audio" in identity.audio_engine_ids()
    assert identity.is_audio_engine_id("fixture_audio") is True
    assert identity.is_audio_engine_id("llama_cpp") is False


def test_catalog_provider_uses_the_injected_engine_id(monkeypatch):
    captured = {}

    def fake_version_entry(store, engine_id, version):
        captured.update(engine_id=engine_id, version=version)
        return {}

    monkeypatch.setattr(provider_module, "get_version_entry", fake_version_entry)
    provider = provider_module.AudioCppCatalogProvider(store=object(), engine_id="fixture_audio")
    assert provider.id == "fixture_audio"
    assert provider._normalize_packages([], {"version": "v1"}) == []
    assert captured == {"engine_id": "fixture_audio", "version": "v1"}


def test_model_installer_uses_the_injected_engine_id(monkeypatch):
    calls = []

    class Store:
        def get_active_engine_version(self, engine_id):
            calls.append(engine_id)
            return {}

    monkeypatch.setattr(installer_module, "audio_cpp_enabled", lambda: True)
    installer = installer_module.AudioModelInstaller(
        store=Store(), engine_id="fixture_audio", manager=SimpleNamespace()
    )
    with pytest.raises(RuntimeError, match="Install and activate"):
        installer._active_version()
    assert calls == ["fixture_audio"]


def test_catalog_service_accepts_a_declared_audio_provider(monkeypatch):
    import backend.engines.registry as registry
    from backend.model_catalog.service import ModelCatalogService

    monkeypatch.setitem(
        registry.ENGINE_REGISTRY,
        "fixture_audio",
        SimpleNamespace(runtime_kind="audio_cpp", catalog={"providers": ["fixture_audio"]}),
    )
    assert ModelCatalogService._provider_ids({"provider": "fixture_audio"}) == ["fixture_audio"]


def test_extension_prefix_and_context_follow_the_declared_engine(monkeypatch):
    import backend.engines.registry as registry
    from backend.routes import audio_cpp_versions as route

    monkeypatch.setitem(
        registry.ENGINE_REGISTRY,
        "fixture_audio",
        SimpleNamespace(runtime_kind="audio_cpp"),
    )
    assert route.segment_for_engine("fixture_audio") == "fixture-audio"
    assert route.engine_for_path("/api/fixture-audio/contract") == "fixture_audio"
    assert route.engine_for_path("/api/health") == "audio_cpp"

    token = route.set_engine_context("fixture_audio")
    try:
        assert route.current_engine_id() == "fixture_audio"
    finally:
        route.reset_engine_context(token)
    assert route.current_engine_id() == "audio_cpp"
