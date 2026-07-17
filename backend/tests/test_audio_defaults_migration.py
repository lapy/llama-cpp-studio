"""Batch migration of audio.cpp request defaults after contract drift."""

from backend import data_store
from backend.audio_defaults_migration import (
    migrate_audio_models_defaults,
    migrate_request_defaults_section,
)


def test_migrate_request_defaults_moves_speech_to_task():
    section, changed, notes = migrate_request_defaults_section(
        {
            "family": "vevo2",
            "task": "tts",
            "speech_defaults": {"instructions": "warm"},
            "task_defaults": {},
        },
        expected_key="task_defaults",
        mark_reviewed_fingerprint="abc",
    )
    assert changed is True
    assert section["task_defaults"]["instructions"] == "warm"
    assert section["speech_defaults"] == {}
    assert section["last_reviewed_fingerprint"] == "abc"
    assert any("Moved speech_defaults" in note for note in notes)


def test_migrate_models_batch(monkeypatch, tmp_path):
    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    store.add_model(
        {
            "id": "audio-vevo",
            "name": "Vevo",
            "family": "vevo2",
            "compatible_engines": ["audio_cpp"],
            "config": {
                "engine": "audio_cpp",
                "engines": {
                    "audio_cpp": {
                        "family": "vevo2",
                        "task": "tts",
                        "speech_defaults": {"temperature": 0.4},
                    }
                },
            },
        }
    )
    monkeypatch.setattr(
        store,
        "get_active_engine_version",
        lambda engine: (
            {"version": "v1", "source_path": "/tmp"} if engine == "audio_cpp" else None
        ),
    )
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_version_entry",
        lambda *a, **k: {"contract_fingerprint": "f" * 64},
    )
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_model_profile_entry",
        lambda *a, **k: {
            "inspection": {
                "family": "vevo2",
                "tasks": [{"task": "tts"}, {"task": "vc"}],
            }
        },
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.audio_cpp_model_profile_fingerprint",
        lambda *a, **k: "fp",
    )

    result = migrate_audio_models_defaults(store, mark_reviewed=True)
    assert result["migrated_count"] == 1
    model = store.get_model("audio-vevo")
    audio = model["config"]["engines"]["audio_cpp"]
    assert audio["task_defaults"]["temperature"] == 0.4
    assert not audio.get("speech_defaults")
    assert audio["last_reviewed_fingerprint"] == "f" * 64


def test_migrate_defaults_api(client, monkeypatch, tmp_path):
    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    store.add_model(
        {
            "id": "audio-asr",
            "name": "ASR",
            "family": "qwen3_asr",
            "compatible_engines": ["audio_cpp"],
            "config": {
                "engine": "audio_cpp",
                "engines": {
                    "audio_cpp": {
                        "family": "qwen3_asr",
                        "task": "asr",
                        "speech_defaults": {"language": "en"},
                    }
                },
            },
        }
    )
    monkeypatch.setattr(
        store,
        "get_active_engine_version",
        lambda engine: (
            {"version": "v1", "source_path": "/tmp"} if engine == "audio_cpp" else None
        ),
    )
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_version_entry",
        lambda *a, **k: {"contract_fingerprint": "c" * 64},
    )
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_model_profile_entry",
        lambda *a, **k: {
            "inspection": {"family": "qwen3_asr", "tasks": [{"task": "asr"}]}
        },
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.audio_cpp_model_profile_fingerprint",
        lambda *a, **k: "fp",
    )

    r = client.post(
        "/api/audio-cpp/migrate-defaults",
        json={"model_ids": ["audio-asr"], "mark_reviewed": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["migrated_count"] == 1
    audio = store.get_model("audio-asr")["config"]["engines"]["audio_cpp"]
    assert audio["transcription_defaults"]["language"] == "en"
    assert audio["last_reviewed_fingerprint"] == "c" * 64
