"""Public inference URL and bounded connect-test relay."""

from urllib.parse import quote

from backend.inference_url import normalize_public_inference_url
from backend.tests.test_api_routes import _install_temp_store, _seed_model


def test_public_inference_url_rejects_credentials_and_bad_schemes():
    assert normalize_public_inference_url("  ") == ""
    assert normalize_public_inference_url("https://infer.example.test/proxy/") == (
        "https://infer.example.test/proxy"
    )
    for value in ("javascript:alert(1)", "https://user:secret@infer.example.test"):
        try:
            normalize_public_inference_url(value)
        except ValueError:
            continue
        raise AssertionError(f"expected rejection for {value}")


def test_inference_settings_round_trip(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    missing = client.get("/api/settings/inference")
    assert missing.status_code == 200
    assert missing.json()["public_inference_url"] == ""

    rejected = client.put(
        "/api/settings/inference",
        json={"public_inference_url": "ftp://files.example.test"},
    )
    assert rejected.status_code == 400

    saved = client.put(
        "/api/settings/inference",
        json={"public_inference_url": "https://infer.example.test/base/"},
    )
    assert saved.status_code == 200
    assert saved.json()["public_inference_url"] == "https://infer.example.test/base"
    assert store.get_settings()["public_inference_url"] == "https://infer.example.test/base"

    status = client.get("/api/status")
    assert status.json()["proxy_status"]["public_inference_url"] == (
        "https://infer.example.test/base"
    )


def test_connect_test_uses_embeddings_for_embedding_models(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)
    store.update_model(
        "org/model",
        {"pipeline_tag": "feature-extraction", "config": {"embedding": True}},
    )

    seen = {}

    class FakeClient:
        async def request(self, method, path, *, timeout=10.0, json=None):
            seen["method"] = method
            seen["path"] = path
            seen["timeout"] = timeout
            seen["json"] = json

            class Response:
                status_code = 200
                text = '{"data":[{"embedding":[0.1]}]}'

            return Response()

    monkeypatch.setattr(
        "backend.proxy.llama_swap.client.get_llama_swap_client",
        lambda: FakeClient(),
    )
    response = client.post(f"/api/models/{quote('org/model', safe='')}/connect-test")
    assert response.status_code == 200
    assert seen["method"] == "POST"
    assert seen["path"].endswith("/v1/embeddings")
    assert seen["json"]["input"] == "ping"
    assert seen["timeout"] == 10
    assert "max_tokens" not in seen["json"]


def test_connect_test_bounds_chat_and_download_is_not_reviewed(
    client, monkeypatch, tmp_path
):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)

    listed = client.get("/api/models")
    quant = listed.json()[0]["quantizations"][0]
    assert quant["config_reviewed"] is False

    seen = {}

    class FakeClient:
        async def request(self, method, path, *, timeout=10.0, json=None):
            seen["path"] = path
            seen["json"] = json

            class Response:
                status_code = 200
                text = "pong"

            return Response()

    monkeypatch.setattr(
        "backend.proxy.llama_swap.client.get_llama_swap_client",
        lambda: FakeClient(),
    )
    response = client.post(f"/api/models/{quote('org/model', safe='')}/connect-test")
    assert response.status_code == 200
    assert seen["path"].endswith("/v1/chat/completions")
    assert seen["json"]["max_tokens"] == 16

    saved = client.put(
        f"/api/models/{quote('org/model', safe='')}/config",
        json={"engine": "llama_cpp", "engines": {"llama_cpp": {"threads": 2}}},
    )
    assert saved.status_code == 200
    reviewed = client.get("/api/models").json()[0]["quantizations"][0]
    assert reviewed["config_reviewed"] is True
    assert reviewed["config_reviewed_at"]


def _audio_install_record(tmp_path, store):
    """The record ``AudioModelInstaller._model_record`` stores after a package install."""
    from backend.services.audio_model_installer import AudioModelInstaller

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    installer = AudioModelInstaller(store)
    return installer._model_record(
        {
            "id": "demo-voice",
            "display_name": "Demo Voice",
            "source": {"kind": "direct"},
        },
        str(bundle),
        str(bundle),
        {
            "family": "asr",
            "task_names": ["asr"],
            "tasks": [{"task": "asr", "modes": ["offline"]}],
            "capabilities": {},
        },
        "direct",
        {"version": "v1", "source_commit": "abc", "build_config": {"backend": "cpu"}},
    )


def test_audio_install_record_is_not_reviewed_until_config_save(
    client, monkeypatch, tmp_path
):
    from urllib.parse import quote

    from backend.models.config import config_was_reviewed

    store = _install_temp_store(monkeypatch, tmp_path)
    record = _audio_install_record(tmp_path, store)
    assert isinstance(record["config"].get("engines"), dict)
    assert "config_reviewed_at" not in record
    assert config_was_reviewed(record) is False

    stored = store.add_model(record)
    assert config_was_reviewed(stored) is False
    listed = client.get("/api/models")
    assert listed.status_code == 200
    quant = listed.json()[0]["quantizations"][0]
    assert quant["id"] == record["id"]
    assert quant["config_reviewed"] is False
    assert quant["config_reviewed_at"] is None

    monkeypatch.setattr(
        "backend.audio.model_config.validate_audio_model_config",
        lambda *args, **kwargs: {"errors": [], "warnings": []},
    )
    saved = client.put(
        f"/api/models/{quote(record['id'], safe='')}/config",
        json=record["config"],
    )
    assert saved.status_code == 200
    reviewed = client.get("/api/models").json()[0]["quantizations"][0]
    assert reviewed["config_reviewed"] is True
    assert reviewed["config_reviewed_at"]
    assert store.get_model(record["id"])["config_review_source"] == "user"


def test_unstamped_engine_map_is_not_reviewed(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    models_path = tmp_path / "config" / "models.yaml"
    models_path.write_text(
        "schema_version: 2\n"
        "models:\n"
        "  - id: unstamped-model\n"
        "    display_name: Unstamped\n"
        "    format: gguf\n"
        "    config:\n"
        "      engine: llama_cpp\n"
        "      engines:\n"
        "        llama_cpp:\n"
        "          threads: 4\n",
        encoding="utf-8",
    )

    listed = client.get("/api/models")
    assert listed.status_code == 200
    quant = listed.json()[0]["quantizations"][0]
    assert quant["config_reviewed"] is False
    assert quant["config_reviewed_at"] is None

    store.update_model("unstamped-model", {"display_name": "Unstamped renamed"})
    migrated = store.get_model("unstamped-model")
    assert "config_review_source" not in migrated
    assert client.get("/api/models").json()[0]["quantizations"][0]["config_reviewed"] is False

    fresh = client.get("/api/models")
    assert fresh.json()[0]["quantizations"][0]["name"] == "Unstamped renamed"
