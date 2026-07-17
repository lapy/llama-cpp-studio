"""CI smoke: activate fixture + speech/ASR through llama-swap proxy paths.

Does not require a live llama-swap process or `/v1/tasks/run` routing.
"""

from __future__ import annotations

from backend import data_store
import backend.llama_swap_config as swap_config
from backend.audio_request_policy import build_request_policy
from backend.engine_param_scanner import compute_audio_cpp_capability_delta


def test_activate_fixture_persists_capability_delta_shape():
    previous = {
        "capabilities": {"families": ["omnivoice"], "tasks": ["tts"]},
    }
    current = {
        "capabilities": {
            "families": ["omnivoice", "qwen3_asr"],
            "tasks": ["tts", "asr"],
        },
    }
    delta = compute_audio_cpp_capability_delta(previous, current)
    assert delta["added_families"] == ["qwen3_asr"]
    assert delta["added_tasks"] == ["asr"]
    assert delta["removed_families"] == []
    assert delta["removed_tasks"] == []


def test_activate_api_fixture_exposes_delta_and_affected_models(
    client, monkeypatch, tmp_path
):
    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    server = tmp_path / "audiocpp_server"
    cli = tmp_path / "audiocpp_cli"
    server.write_text("x")
    cli.write_text("x")
    store.add_engine_version(
        "audio_cpp",
        {
            "version": "smoke-v1",
            "server_binary_path": str(server),
            "cli_binary_path": str(cli),
            "repository_source": "audio.cpp",
        },
    )
    store.add_model(
        {
            "id": "audio-omnivoice-smoke",
            "name": "OmniVoice Smoke",
            "family": "omnivoice",
            "compatible_engines": ["audio_cpp"],
            "config": {
                "engine": "audio_cpp",
                "engines": {
                    "audio_cpp": {"family": "omnivoice", "task": "tts"},
                },
            },
        }
    )

    from backend.engine_param_catalog import upsert_version_entry

    upsert_version_entry(
        store,
        "audio_cpp",
        "smoke-v1",
        {
            "contract_fingerprint": "0" * 64,
            "contract_changed": False,
            "capabilities": {"families": ["old"], "tasks": ["tts"]},
        },
    )

    def fake_scan(store_obj, engine, row):
        entry = {
            "contract_fingerprint": "s" * 64,
            "previous_contract_fingerprint": "0" * 64,
            "contract_changed": True,
            "capabilities": {
                "families": ["omnivoice", "qwen3_asr"],
                "tasks": ["tts", "asr"],
                "discovery_source": "json",
            },
            "capability_delta": {
                "added_families": ["omnivoice", "qwen3_asr"],
                "removed_families": ["old"],
                "added_tasks": ["asr"],
                "removed_tasks": [],
            },
        }
        upsert_version_entry(store_obj, "audio_cpp", "smoke-v1", entry)
        return entry

    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_engine_version",
        fake_scan,
    )
    monkeypatch.setattr(
        "backend.llama_swap_manager.mark_swap_config_stale",
        lambda: None,
    )

    class FakeSwap:
        async def start_proxy(self):
            return None

    monkeypatch.setattr(
        "backend.llama_swap_manager.get_llama_swap_manager",
        lambda: FakeSwap(),
    )

    r = client.post(
        "/api/audio-cpp/versions/activate",
        json={"version_id": "smoke-v1"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["capability_delta"]["added_tasks"] == ["asr"]
    assert any(m["id"] == "audio-omnivoice-smoke" for m in data["affected_models"])

    status = client.get("/api/audio-cpp/status")
    assert status.status_code == 200
    body = status.json()
    assert body["contract_changed"] is True
    assert "omnivoice" in body["families"]
    assert "asr" in body["tasks"]


def test_speech_model_routes_through_llama_swap_proxy_path():
    policy = build_request_policy(
        task="tts",
        family="omnivoice",
        inspection={"family": "omnivoice", "tasks": [{"task": "tts"}]},
    )
    assert policy["api_endpoint"] == "/v1/audio/speech"
    assert policy["request_defaults_key"] == "speech_defaults"

    filters, _aliases = swap_config._yaml_filters_and_aliases(
        stable_id="audio-omnivoice-smoke",
        config={
            "engine": "audio_cpp",
            "family": "omnivoice",
            "task": "tts",
            "speech_defaults": {"instructions": "warm narrator"},
        },
    )
    assert filters["setParams"]["instructions"] == "warm narrator"
    # Proxy-facing OpenAI path (not /upstream/.../tasks/run)
    assert policy["api_endpoint"].startswith("/v1/audio/")


def test_asr_model_routes_through_llama_swap_proxy_path():
    policy = build_request_policy(
        task="asr",
        family="qwen3_asr",
        inspection={"family": "qwen3_asr", "tasks": [{"task": "asr"}]},
    )
    assert policy["api_endpoint"] == "/v1/audio/transcriptions"
    assert policy["request_defaults_key"] == "transcription_defaults"

    filters, _aliases = swap_config._yaml_filters_and_aliases(
        stable_id="audio-asr-smoke",
        config={
            "engine": "audio_cpp",
            "family": "qwen3_asr",
            "task": "asr",
            "transcription_defaults": {"language": "en"},
        },
    )
    assert filters["setParams"]["language"] == "en"


def test_smoke_does_not_require_tasks_run_proxy_route():
    """Document deferred llama-swap gap: generic tasks stay on /upstream/… fallback."""
    policy = build_request_policy(
        task="vad",
        family="silero_vad",
        inspection={"family": "silero_vad", "tasks": [{"task": "vad"}]},
    )
    assert policy["api_endpoint"] == "/v1/tasks/run"
    assert policy["api_endpoint"] != "/v1/audio/speech"
    assert policy["api_endpoint"] != "/v1/audio/transcriptions"
