"""Demo-voice library derived from the active audio.cpp checkout."""

from __future__ import annotations

import hashlib
import io

import pytest

from backend import reference_audio
from backend.audio import community_voices
from backend.audio.community_voices import (
    CommunityVoiceError,
    install_community_voices,
    library_status,
    list_reference_entries,
    prompt_text_bytes,
    voice_catalog,
)


WAV_BYTES = b"RIFF....WAVE"
CHECKOUT = ("webui", "native", "demo_voices")


@pytest.fixture(autouse=True)
def isolate_voice_data(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_audio, "_data_root", lambda: str(tmp_path / "data"))


def _checkout(tmp_path, wav_bytes=WAV_BYTES, prompt=b"demo_1_man|hello there\n"):
    source = tmp_path.joinpath(*("audio.cpp",) + CHECKOUT)
    source.mkdir(parents=True)
    (source / "demo_1_man.wav").write_bytes(wav_bytes)
    if prompt is not None:
        (source / "prompt_text").write_bytes(prompt)
    return tmp_path / "audio.cpp"


def test_catalog_is_derived_from_the_checkout(tmp_path):
    root = _checkout(tmp_path)
    catalog = voice_catalog(source_root=str(root))
    assert [item["id"] for item in catalog] == ["demo_1_man"]
    assert catalog[0]["reference_text"] == "hello there"
    assert catalog[0]["sha256"] == hashlib.sha256(WAV_BYTES).hexdigest()
    assert prompt_text_bytes(source_root=str(root)) == b"demo_1_man|hello there\n"


def test_empty_without_checkout_or_installed_copy():
    status = library_status()
    assert status["installed"] is False
    assert status["voice_dir"] is None
    assert status["source"]["path"] == "webui/native/demo_voices"
    assert status["items"] == []
    assert list_reference_entries() == []


@pytest.mark.asyncio
async def test_install_copies_the_active_checkout(tmp_path):
    root = _checkout(tmp_path)
    status = await install_community_voices(source_root=str(root))

    assert status["installed"] is True
    assert status["voice_dir"]
    assert status["items"][0]["path"].endswith("demo_1_man.wav")
    entries = list_reference_entries()
    assert entries[0]["storage"] == "community"
    assert entries[0]["display_path"] == "community/demo_1_man.wav"
    assert entries[0]["reference_text"] == "hello there"
    assert entries[0]["voice_id"] == "demo_1_man"


@pytest.mark.asyncio
async def test_install_refreshes_when_the_checkout_changes(tmp_path):
    root = _checkout(tmp_path)
    await install_community_voices(source_root=str(root))
    updated = b"RIFFxxxxWAVE-new"
    (tmp_path / "audio.cpp" / "webui" / "native" / "demo_voices" / "demo_1_man.wav").write_bytes(updated)

    stale = library_status(source_root=str(root))
    assert stale["installed"] is False
    assert stale["items"][0]["installed"] is False

    await install_community_voices(source_root=str(root))
    fresh = library_status(source_root=str(root))
    assert fresh["installed"] is True
    assert fresh["items"][0]["size_bytes"] == len(updated)


@pytest.mark.asyncio
async def test_install_requires_a_checkout_pack(tmp_path):
    with pytest.raises(CommunityVoiceError, match="No demo voices found"):
        await install_community_voices(source_root=str(tmp_path / "missing"))
    assert library_status()["installed"] is False


@pytest.mark.asyncio
async def test_install_rejects_non_wav_members(tmp_path):
    root = _checkout(tmp_path, wav_bytes=b"not-a-wav-at-all")
    with pytest.raises(CommunityVoiceError, match="not a WAV"):
        await install_community_voices(source_root=str(root))


@pytest.mark.asyncio
async def test_install_requires_prompt_text(tmp_path):
    root = _checkout(tmp_path, prompt=None)
    with pytest.raises(CommunityVoiceError, match="prompt_text"):
        await install_community_voices(source_root=str(root))


def test_installed_copy_reports_complete_without_a_checkout(tmp_path):
    import asyncio

    root = _checkout(tmp_path)
    asyncio.run(install_community_voices(source_root=str(root)))

    status = library_status(source_root=str(tmp_path / "pruned"))
    assert status["installed"] is True
    assert status["source"]["checkout"] == ""


def test_audio_runtime_sets_voice_dir_when_library_is_installed(tmp_path, monkeypatch):
    import backend.engines.audio_cpp.runtime as audio_runtime
    from backend.tests.test_audio_cpp_runtime import _fixture

    store, model, config = _fixture(tmp_path)
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(tmp_path / "sidecars"))
    monkeypatch.setattr(audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)
    monkeypatch.setattr(audio_runtime, "installed_voice_dir", lambda: "/data/community-voices")

    runtime = audio_runtime.build_audio_cpp_runtime(store, model, config, "audio-demo")
    assert runtime["sidecar"]["voice_dir"] == "/data/community-voices"


def test_audio_runtime_omits_voice_dir_without_library(tmp_path, monkeypatch):
    import backend.engines.audio_cpp.runtime as audio_runtime
    from backend.tests.test_audio_cpp_runtime import _fixture

    store, model, config = _fixture(tmp_path)
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(tmp_path / "sidecars"))
    monkeypatch.setattr(audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)
    monkeypatch.setattr(audio_runtime, "installed_voice_dir", lambda: "")

    runtime = audio_runtime.build_audio_cpp_runtime(store, model, config, "audio-demo")
    assert "voice_dir" not in runtime["sidecar"]


def test_community_voice_routes(client, monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)

    listing = client.get("/api/audio-cpp/community-voices")
    assert listing.status_code == 200
    body = listing.json()
    assert body["installed"] is False
    assert body["source"]["repository"] == "https://github.com/0xShug0/audio.cpp"
    assert body["items"] == []

    async def fake_install(*, source_root=""):
        assert source_root == ""
        return {"installed": True, "voice_dir": "/data/community-voices", "items": body["items"]}

    monkeypatch.setattr(community_voices, "install_community_voices", fake_install)
    monkeypatch.setattr(
        "backend.proxy.llama_swap.manager.mark_swap_config_stale",
        lambda: None,
    )
    installed = client.post("/api/audio-cpp/community-voices/install")
    assert installed.status_code == 200
    assert installed.json()["installed"] is True


def test_community_voice_install_reports_checkout_failure(client, monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)

    async def fail_install(*, source_root=""):
        raise CommunityVoiceError("No demo voices found in the active audio.cpp checkout")

    monkeypatch.setattr(community_voices, "install_community_voices", fail_install)
    response = client.post("/api/audio-cpp/community-voices/install")
    assert response.status_code == 502
    assert "No demo voices found" in response.json()["detail"]


def test_reference_audio_list_appends_installed_community_clips(client, monkeypatch, tmp_path):
    from backend import data_store
    from backend.tests.test_reference_audio import WAV_BYTES as route_wav
    from backend.tests.test_reference_audio import _seed_audio_model

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    _seed_audio_model(store, bundle)

    monkeypatch.setattr(
        community_voices,
        "list_reference_entries",
        lambda: [
            {
                "filename": "demo_1_man.wav",
                "path": "/data/community/demo_1_man.wav",
                "relative_path": "community/demo_1_man.wav",
                "display_path": "community/demo_1_man.wav",
                "size_bytes": 12,
                "storage": "community",
                "voice_id": "demo_1_man",
                "reference_text": "hello",
            }
        ],
    )
    upload = client.post(
        "/api/models/audio%2Fdemo/reference-audio",
        files={"file": ("voice.wav", io.BytesIO(route_wav), "audio/wav")},
    )
    assert upload.status_code == 200
    listing = client.get("/api/models/audio%2Fdemo/reference-audio")
    items = listing.json()["items"]
    assert [item["filename"] for item in items] == ["voice.wav", "demo_1_man.wav"]
    assert items[1]["storage"] == "community"
