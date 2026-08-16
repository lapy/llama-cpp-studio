"""Studio /audio-cpp-ui redirects to llama-swap's /upstream/{model}/ UI."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from backend.main import app
    from backend.routes import audio_cpp_ui_proxy as proxy

    monkeypatch.setattr(proxy, "_get_proxy_port", lambda: 2000)
    return TestClient(app, follow_redirects=False)


def test_audio_ui_redirects_index_to_llama_swap_upstream(client):
    response = client.get("/audio-cpp-ui/audio-cpp-pocket_tts_english_q8_0")
    assert response.status_code == 307
    assert (
        response.headers["location"]
        == "http://testserver:2000/upstream/audio-cpp-pocket_tts_english_q8_0/"
    )


def test_audio_ui_redirects_trailing_slash_to_llama_swap_upstream(client):
    response = client.get("/audio-cpp-ui/audio-cpp-pocket_tts_english_q8_0/")
    assert response.status_code == 307
    assert (
        response.headers["location"]
        == "http://testserver:2000/upstream/audio-cpp-pocket_tts_english_q8_0/"
    )


def test_audio_ui_redirects_nested_path_and_query(client):
    response = client.get(
        "/audio-cpp-ui/audio-cpp-pocket_tts_english_q8_0/v1/ui/models-root?x=1"
    )
    assert response.status_code == 307
    assert (
        response.headers["location"]
        == "http://testserver:2000/upstream/audio-cpp-pocket_tts_english_q8_0/v1/ui/models-root?x=1"
    )


def test_audio_ui_redirects_use_request_hostname(client):
    response = client.get(
        "/audio-cpp-ui/audio-cpp-pocket_tts_english_q8_0/",
        headers={"host": "10.0.0.129:8080"},
    )
    assert response.status_code == 307
    assert (
        response.headers["location"]
        == "http://10.0.0.129:2000/upstream/audio-cpp-pocket_tts_english_q8_0/"
    )
