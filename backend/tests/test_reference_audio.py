"""Reference audio storage helpers and API routes."""

from __future__ import annotations

import io

import pytest
from fastapi import HTTPException

from backend import reference_audio
from backend.model_config import effective_model_config

WAV_BYTES = b"RIFF....WAVE"


@pytest.fixture(autouse=True)
def isolate_reference_audio_data(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_audio, "_data_root", lambda: str(tmp_path / "data"))


def test_sanitize_reference_filename_rejects_non_wav():
    with pytest.raises(HTTPException) as exc:
        reference_audio.sanitize_reference_filename("clip.mp3")
    assert exc.value.status_code == 400


def test_sanitize_reference_filename_strips_unsafe_chars():
    assert reference_audio.sanitize_reference_filename("my voice (1).wav") == "my_voice_1.wav"


def test_sanitize_reference_filename_rejects_blank_stem():
    with pytest.raises(HTTPException) as exc:
        reference_audio.sanitize_reference_filename("---.wav")
    assert exc.value.status_code == 400


def test_get_audio_model_bundle_root_missing_path():
    with pytest.raises(HTTPException) as exc:
        reference_audio.get_audio_model_bundle_root({})
    assert exc.value.status_code == 400
    assert "bundle path" in exc.value.detail


def test_save_list_and_delete_reference_audio(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    content = WAV_BYTES

    saved = reference_audio.save_reference_audio(
        str(bundle),
        filename="My Voice.wav",
        content=content,
    )
    assert saved["filename"] == "My_Voice.wav"
    assert saved["path"].endswith("/refs/My_Voice.wav")
    assert saved["relative_path"] == "refs/My_Voice.wav"
    assert saved["display_path"] == "refs/My_Voice.wav"
    assert str(tmp_path / "data") in saved["path"]
    assert saved["size_bytes"] == len(content)
    assert saved["modified_at"]

    items = reference_audio.list_reference_audio(str(bundle))
    assert len(items) == 1
    assert items[0]["size_bytes"] == len(content)

    reference_audio.delete_reference_audio(str(bundle), filename="My_Voice.wav")
    assert reference_audio.list_reference_audio(str(bundle)) == []


def test_list_reference_audio_empty_without_refs_dir(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    assert reference_audio.list_reference_audio(str(bundle)) == []


def test_list_reference_audio_ignores_non_wav(tmp_path):
    bundle = tmp_path / "bundle"
    refs = bundle / "refs"
    refs.mkdir(parents=True)
    (refs / "note.txt").write_text("ignore", encoding="utf-8")
    reference_audio.save_reference_audio(
        str(bundle),
        filename="voice.wav",
        content=WAV_BYTES,
    )
    assert len(reference_audio.list_reference_audio(str(bundle))) == 1


def test_save_reference_audio_deduplicates_filename(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    reference_audio.save_reference_audio(
        str(bundle),
        filename="voice.wav",
        content=WAV_BYTES,
    )
    second = reference_audio.save_reference_audio(
        str(bundle),
        filename="voice.wav",
        content=WAV_BYTES,
    )
    assert second["filename"] == "voice_1.wav"


def test_save_reference_audio_rejects_empty(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(HTTPException) as exc:
        reference_audio.save_reference_audio(
            str(bundle),
            filename="voice.wav",
            content=b"",
        )
    assert exc.value.status_code == 400


def test_save_reference_audio_rejects_oversized(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(reference_audio, "MAX_REFERENCE_AUDIO_BYTES", 4)
    with pytest.raises(HTTPException) as exc:
        reference_audio.save_reference_audio(
            str(bundle),
            filename="voice.wav",
            content=b"12345",
        )
    assert exc.value.status_code == 400
    assert "maximum size" in exc.value.detail


def test_reference_audio_limit_is_60mb():
    assert reference_audio.MAX_REFERENCE_AUDIO_BYTES == 60 * 1024 * 1024


def test_save_reference_audio_rejects_malformed_wav(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(HTTPException) as exc:
        reference_audio.save_reference_audio(
            str(bundle),
            filename="voice.wav",
            content=b"not-a-wav",
        )
    assert exc.value.status_code == 400
    assert "valid WAV" in exc.value.detail


def test_delete_reference_audio_not_found(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(HTTPException) as exc:
        reference_audio.delete_reference_audio(str(bundle), filename="missing.wav")
    assert exc.value.status_code == 404


def test_delete_rejects_in_use_reference(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    saved = reference_audio.save_reference_audio(
        str(bundle),
        filename="voice.wav",
        content=WAV_BYTES,
    )
    config = effective_model_config(
        {
            "engine": "audio_cpp",
            "engines": {
                "audio_cpp": {
                    "family": "omnivoice",
                    "task": "tts",
                    "voice_presets": {
                        "assistant": {"voice_ref": saved["path"]},
                    },
                }
            },
        }
    )
    with pytest.raises(HTTPException) as exc:
        reference_audio.delete_reference_audio(
            str(bundle),
            filename=saved["filename"],
            effective_config=config,
        )
    assert exc.value.status_code == 409


def test_find_config_references_nested_paths():
    config = {
        "voice_presets": {"clone": {"voice_ref": "refs/a.wav"}},
        "speech_defaults": {"voice_ref": "refs/b.wav"},
    }
    assert reference_audio.find_config_references(config, "refs/a.wav") == [
        "voice_presets.clone.voice_ref"
    ]
    assert reference_audio.find_config_references(config, "refs/b.wav") == [
        "speech_defaults.voice_ref"
    ]
    assert reference_audio.find_config_references(config, "/app/data/audio/refs/a.wav") == [
        "voice_presets.clone.voice_ref"
    ]


def _seed_audio_model(store, bundle, *, config=None):
    store.add_model(
        {
            "id": "audio/demo",
            "name": "Demo",
            "display_name": "Demo",
            "format": "audio_cpp",
            "compatible_engines": ["audio_cpp"],
            "artifact": {"path": str(bundle), "package_kind": "prepared_bundle"},
            "config": config
            or {
                "engine": "audio_cpp",
                "engines": {
                    "audio_cpp": {
                        "family": "omnivoice",
                        "task": "tts",
                        "voice_presets": {},
                    }
                },
            },
        }
    )


def test_reference_audio_routes(client, monkeypatch, tmp_path):
    from backend import data_store

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    _seed_audio_model(store, bundle)

    upload = client.post(
        "/api/models/audio%2Fdemo/reference-audio",
        files={"file": ("voice.wav", io.BytesIO(WAV_BYTES), "audio/wav")},
    )
    assert upload.status_code == 200
    payload = upload.json()
    assert payload["path"].endswith("/refs/voice.wav")
    assert payload["relative_path"] == "refs/voice.wav"
    assert str(tmp_path / "data") in payload["path"]

    listing = client.get("/api/models/audio%2Fdemo/reference-audio")
    assert listing.status_code == 200
    assert len(listing.json()["items"]) == 1

    delete = client.delete("/api/models/audio%2Fdemo/reference-audio/voice.wav")
    assert delete.status_code == 200
    assert client.get("/api/models/audio%2Fdemo/reference-audio").json()["items"] == []


def test_reference_audio_routes_rejects_non_audio_model(client, monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    store.add_model(
        {
            "id": "org/gguf",
            "name": "GGUF",
            "display_name": "GGUF",
            "format": "gguf",
            "compatible_engines": ["llama_cpp"],
            "config": {},
        }
    )

    response = client.get("/api/models/org%2Fgguf/reference-audio")
    assert response.status_code == 400
    assert "audio.cpp" in response.json()["detail"]


def test_reference_audio_routes_list_includes_used_by(client, monkeypatch, tmp_path):
    from backend import data_store

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    _seed_audio_model(
        store,
        bundle,
        config={
            "engine": "audio_cpp",
            "engines": {
                "audio_cpp": {
                    "family": "omnivoice",
                    "task": "tts",
                    "voice_presets": {
                        "assistant": {"voice_ref": "refs/voice.wav"},
                    },
                }
            },
        },
    )
    reference_audio.save_reference_audio(
        str(bundle),
        filename="voice.wav",
        content=WAV_BYTES,
        storage_key="audio/demo",
    )

    listing = client.get("/api/models/audio%2Fdemo/reference-audio")
    assert listing.status_code == 200
    item = listing.json()["items"][0]
    assert item["path"].endswith("/refs/voice.wav")
    assert item["relative_path"] == "refs/voice.wav"
    assert item["used_by"] == ["voice_presets.assistant.voice_ref"]


def test_reference_audio_routes_delete_in_use_returns_409(client, monkeypatch, tmp_path):
    from backend import data_store

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    _seed_audio_model(
        store,
        bundle,
        config={
            "engine": "audio_cpp",
            "engines": {
                "audio_cpp": {
                    "family": "omnivoice",
                    "task": "tts",
                    "voice_presets": {
                        "assistant": {"voice_ref": "refs/voice.wav"},
                    },
                }
            },
        },
    )
    reference_audio.save_reference_audio(
        str(bundle),
        filename="voice.wav",
        content=WAV_BYTES,
        storage_key="audio/demo",
    )

    delete = client.delete("/api/models/audio%2Fdemo/reference-audio/voice.wav")
    assert delete.status_code == 409
    assert "voice_presets.assistant.voice_ref" in delete.json()["detail"]
