"""The native audio.cpp WebUI gateway is leftover surface and must stay gone."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


_REMOVED_MODULES = (
    "backend.audio_cpp_ui_gateway",
    "backend.audio_cpp_ui_rewrite",
    "backend.routes.audio_cpp_ui_proxy",
)


@pytest.mark.parametrize("module_name", _REMOVED_MODULES)
def test_native_ui_modules_are_not_importable(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_native_ui_source_files_are_gone():
    root = Path(__file__).resolve().parents[2]
    leftover = [
        root / "backend" / "audio_cpp_ui_gateway.py",
        root / "backend" / "audio_cpp_ui_rewrite.py",
        root / "backend" / "routes" / "audio_cpp_ui_proxy.py",
    ]
    missing = [str(path) for path in leftover if path.exists()]
    assert missing == []


def test_studio_does_not_mount_audio_cpp_ui_routes():
    from backend.main import app

    mounted = [
        getattr(route, "path", "")
        for route in app.routes
        if "audio-cpp-ui" in str(getattr(route, "path", ""))
        or "audio-cpp-ui" in str(getattr(route, "tags", ()))
    ]
    assert mounted == []


def test_legacy_audio_cpp_ui_path_is_not_a_llama_swap_redirect():
    from backend.main import app

    client = TestClient(app, follow_redirects=False)
    response = client.get("/audio-cpp-ui/audio-cpp-pocket_tts_english_q8_0/")
    assert response.status_code != 307
    location = response.headers.get("location", "")
    assert "/upstream/" not in location
    schema = client.get("/openapi.json").json()
    assert not any(path.startswith("/audio-cpp-ui") for path in schema.get("paths", {}))
