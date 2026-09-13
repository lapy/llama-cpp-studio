from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.audio_cpp_proxy_routing import resolve_audio_upstream_target


@pytest.fixture
def store():
    return SimpleNamespace(
        get_llama_swap_routing=lambda: {"selectors": {"auto-audio": {}}},
        list_models=lambda: [
            {
                "id": "audio-demo",
                "proxy_name": "audio-demo",
                "config": {
                    "engine": "audio_cpp",
                    "engines": {"audio_cpp": {
                        "model_alias": "voice",
                        "swap_aliases": ["other-voice"],
                        "set_params_by_id": [{"sub_id": "warm", "params": {"temperature": 0.2}}],
                    }},
                },
            },
        ],
    )


@pytest.mark.parametrize("model", ["audio-demo", "voice", "other-voice"])
@pytest.mark.parametrize("path", [
    "/v1/audio/transcriptions/details",
    "/v1/audio/alignments",
    "/v1/models/audio-demo/capabilities",
])
def test_native_target_resolves_ordinary_aliases(store, model, path):
    target = resolve_audio_upstream_target(model, path, store=store)
    assert target.path == f"/upstream/audio-demo{path}"
    assert target.model == "audio-demo"


@pytest.mark.parametrize("model,status,detail", [
    ("", 400, "requires a model"),
    ("auto-audio", 400, "selectors"),
    ("voice:warm", 400, "parameter aliases"),
    ("unknown", 404, "Unknown Studio audio model"),
])
def test_native_target_never_silently_bypasses_unsupported_routing(store, model, status, detail):
    with pytest.raises(HTTPException) as caught:
        resolve_audio_upstream_target(model, "/v1/audio/alignments", store=store)
    assert caught.value.status_code == status
    assert detail in caught.value.detail


@pytest.mark.parametrize("path", ["https://example.com/v1/audio", "/v1/../api", "/v1/audio?model=other"])
def test_native_target_rejects_paths_outside_api(store, path):
    with pytest.raises(ValueError):
        resolve_audio_upstream_target("voice", path, store=store)
