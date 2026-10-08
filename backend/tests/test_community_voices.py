"""Pinned audio.cpp demo-voice library."""

from __future__ import annotations

import hashlib
import io

import pytest

from backend import reference_audio
from backend.audio import community_voices
from backend.audio.community_voices import (
    PROMPT_TEXT_SHA256,
    CommunityVoiceError,
    install_community_voices,
    library_status,
    list_reference_entries,
    prompt_text_bytes,
    voice_catalog,
)


WAV_BYTES = b"RIFF....WAVE"


@pytest.fixture(autouse=True)
def isolate_voice_data(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_audio, "_data_root", lambda: str(tmp_path / "data"))


def test_prompt_text_matches_pinned_upstream_pack():
    payload = prompt_text_bytes()
    assert hashlib.sha256(payload).hexdigest() == PROMPT_TEXT_SHA256
    assert payload.startswith(b"demo_1_man|okay,I'm Cemo")
    assert len(voice_catalog()) == 4


def test_library_is_not_installed_until_every_file_matches(tmp_path):
    status = library_status()
    assert status["installed"] is False
    assert status["voice_dir"] is None
    assert status["source"]["path"] == "webui/native/demo_voices"
    assert [item["id"] for item in status["items"]] == [
        "demo_1_man",
        "demo_2_man",
        "demo_3_woman",
        "demo_4_woman",
    ]
    assert list_reference_entries() == []


@pytest.mark.asyncio
async def test_install_copies_matching_local_checkout(tmp_path, monkeypatch):
    spec = {
        "id": "demo_1_man",
        "filename": "demo_1_man.wav",
        "label": "Demo 1 (man)",
        "reference_text": "hello there",
        "size_bytes": len(WAV_BYTES),
        "sha256": hashlib.sha256(WAV_BYTES).hexdigest(),
    }
    prompt = b"demo_1_man|hello there\n"
    monkeypatch.setattr(community_voices, "_CATALOG", (spec,))
    monkeypatch.setattr(community_voices, "PROMPT_TEXT_SHA256", hashlib.sha256(prompt).hexdigest())

    source = tmp_path / "audio.cpp" / "webui" / "native" / "demo_voices"
    source.mkdir(parents=True)
    (source / "demo_1_man.wav").write_bytes(WAV_BYTES)
    (source / "prompt_text").write_bytes(prompt)

    async def fail_download(_url):
        raise AssertionError("local pack should not be downloaded")

    monkeypatch.setattr(community_voices, "_fetch_bytes", fail_download)
    status = await install_community_voices(source_root=str(tmp_path / "audio.cpp"))

    assert status["installed"] is True
    assert status["voice_dir"]
    assert status["items"][0]["path"].endswith("demo_1_man.wav")
    entries = list_reference_entries()
    assert entries[0]["storage"] == "community"
    assert entries[0]["display_path"] == "community/demo_1_man.wav"
    assert entries[0]["reference_text"] == "hello there"
    assert entries[0]["voice_id"] == "demo_1_man"


@pytest.mark.asyncio
async def test_install_downloads_when_local_pack_does_not_match(tmp_path, monkeypatch):
    spec = {
        "id": "demo_1_man",
        "filename": "demo_1_man.wav",
        "label": "Demo 1 (man)",
        "reference_text": "hello there",
        "size_bytes": len(WAV_BYTES),
        "sha256": hashlib.sha256(WAV_BYTES).hexdigest(),
    }
    prompt = b"demo_1_man|hello there\n"
    monkeypatch.setattr(community_voices, "_CATALOG", (spec,))
    monkeypatch.setattr(community_voices, "PROMPT_TEXT_SHA256", hashlib.sha256(prompt).hexdigest())

    source = tmp_path / "audio.cpp" / "webui" / "native" / "demo_voices"
    source.mkdir(parents=True)
    (source / "demo_1_man.wav").write_bytes(b"RIFF....WAVE-wrong")

    fetched = []

    async def fake_download(url):
        fetched.append(url)
        if url.endswith("/prompt_text"):
            return prompt
        return WAV_BYTES

    monkeypatch.setattr(community_voices, "_fetch_bytes", fake_download)
    status = await install_community_voices(source_root=str(tmp_path / "audio.cpp"))
    assert status["installed"] is True
    assert len(fetched) == 2


@pytest.mark.asyncio
async def test_install_rejects_a_download_that_does_not_match(tmp_path, monkeypatch):
    spec = {
        "id": "demo_1_man",
        "filename": "demo_1_man.wav",
        "label": "Demo 1 (man)",
        "reference_text": "hello there",
        "size_bytes": len(WAV_BYTES),
        "sha256": hashlib.sha256(WAV_BYTES).hexdigest(),
    }
    prompt = b"demo_1_man|hello there\n"
    monkeypatch.setattr(community_voices, "_CATALOG", (spec,))
    monkeypatch.setattr(community_voices, "PROMPT_TEXT_SHA256", hashlib.sha256(prompt).hexdigest())

    async def fake_download(_url):
        if str(_url).endswith("/prompt_text"):
            return prompt
        return b"RIFF....WAVE-nope"

    monkeypatch.setattr(community_voices, "_fetch_bytes", fake_download)
    with pytest.raises(CommunityVoiceError, match="does not match"):
        await install_community_voices(source_root="")
    assert library_status()["installed"] is False


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
    assert len(body["items"]) == 4

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


def test_community_voice_install_reports_download_failure(client, monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)

    async def fail_install(*, source_root=""):
        raise CommunityVoiceError("demo_1_man.wav does not match the pinned audio.cpp demo voice")

    monkeypatch.setattr(community_voices, "install_community_voices", fail_install)
    response = client.post("/api/audio-cpp/community-voices/install")
    assert response.status_code == 502
    assert "does not match" in response.json()["detail"]


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
