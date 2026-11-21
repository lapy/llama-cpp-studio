from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.engines.audio_cpp.proxy_routing import resolve_audio_upstream_target


@pytest.fixture
def store():
    return SimpleNamespace(
        get_native_proxy_routing=lambda: {"selectors": {"auto-audio": {}}},
        list_models=lambda: [
            {
                "id": "audio-demo",
                "proxy_name": "audio-demo",
                "config": {
                    "engine": "audio_cpp",
                    "engines": {
                        "audio_cpp": {
                            "model_alias": "voice",
                            "swap_aliases": ["other-voice"],
                            "set_params_by_id": [
                                {"sub_id": "warm", "params": {"temperature": 0.2}}
                            ],
                        }
                    },
                },
            },
        ],
    )


NATIVE_PATHS = [
    "/v1/audio/transcriptions/details",
    "/v1/audio/alignments",
    "/v1/models/audio-demo/capabilities",
    # v0.9.5 generic transports and engine-memory management.
    "/v1/tasks/stream",
    "/v1/tasks/batch",
    "/v1/tasks/unload_models",
    "/v1/tasks/unload_all_models",
    "/v1/models/load",
    "/v1/recipes",
    "/v1/recipes/run",
]


@pytest.mark.parametrize("model", ["audio-demo", "voice", "other-voice"])
@pytest.mark.parametrize("path", NATIVE_PATHS)
def test_native_target_resolves_ordinary_aliases(store, model, path):
    target = resolve_audio_upstream_target(model, path, store=store)
    assert target.path == f"/upstream/audio-demo{path}"
    assert target.model == "audio-demo"


@pytest.mark.parametrize("model", ["", "   "])
def test_native_target_requires_a_model(store, model):
    with pytest.raises(HTTPException) as caught:
        resolve_audio_upstream_target(model, "/v1/tasks/stream", store=store)
    assert caught.value.status_code == 400
    assert "requires a model" in caught.value.detail


@pytest.mark.parametrize(
    "path",
    [
        "/v1/audio/transcriptions/details",
        "/v1/tasks/stream",
        "/v1/tasks/unload_models",
        "/v1/recipes/run",
    ],
)
@pytest.mark.parametrize(
    "model,status,detail",
    [
        ("auto-audio", 404, "Unknown Studio audio model"),
        ("voice:warm", 400, "does not apply parameter aliases"),
        ("unknown", 404, "Unknown Studio audio model"),
    ],
)
def test_native_target_never_silently_bypasses_unsupported_routing(
    store, path, model, status, detail
):
    with pytest.raises(HTTPException) as caught:
        resolve_audio_upstream_target(model, path, store=store)
    assert caught.value.status_code == status
    assert detail in caught.value.detail


@pytest.mark.parametrize(
    "path",
    [
        "https://example.com/v1/audio",
        "/v1/../api",
        "/v1/audio?model=other",
        "/v1/tasks/stream?model=x",
        "/v1/recipes#fragment",
        "/audioapi/v1/tasks/run",
        "",
    ],
)
def test_native_target_rejects_paths_outside_api(store, path):
    with pytest.raises(ValueError):
        resolve_audio_upstream_target("voice", path, store=store)


def test_default_store_lookup_is_request_scoped(monkeypatch):
    """The resolver reads the current store on every call and caches nothing."""
    from backend import data_store

    first = SimpleNamespace(
        list_models=lambda: [
            {
                "id": "audio-first",
                "config": {"engine": "audio_cpp", "engines": {"audio_cpp": {}}},
            }
        ]
    )
    second = SimpleNamespace(
        list_models=lambda: [
            {
                "id": "audio-second",
                "config": {"engine": "audio_cpp", "engines": {"audio_cpp": {}}},
            }
        ]
    )
    monkeypatch.setattr(data_store, "get_store", lambda: first)
    assert (
        resolve_audio_upstream_target("audio-first", "/v1/tasks/stream").path
        == "/upstream/audio-first/v1/tasks/stream"
    )
    monkeypatch.setattr(data_store, "get_store", lambda: second)
    assert (
        resolve_audio_upstream_target("audio-second", "/v1/tasks/stream").path
        == "/upstream/audio-second/v1/tasks/stream"
    )
    # The first store's model is gone from the resolver's view with it.
    with pytest.raises(HTTPException) as caught:
        resolve_audio_upstream_target("audio-first", "/v1/tasks/stream")
    assert caught.value.status_code == 404


def test_non_audio_engine_models_are_not_resolvable(store):
    llama = SimpleNamespace(
        list_models=lambda: [
            {
                "id": "text-model",
                "config": {"engine": "llama_cpp", "engines": {"llama_cpp": {}}},
            }
        ]
    )
    with pytest.raises(HTTPException) as caught:
        resolve_audio_upstream_target("text-model", "/v1/tasks/stream", store=llama)
    assert caught.value.status_code == 404
