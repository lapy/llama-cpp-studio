"""Search metadata regressions for backend.huggingface."""

import asyncio
from types import SimpleNamespace

import backend.huggingface as huggingface


def test_search_with_api_uses_configured_client_full_metadata_for_likes(monkeypatch):
    observed = {}

    async def fake_rate_limit():
        return None

    class FakeApi:
        def list_models(self, **kwargs):
            observed["kwargs"] = kwargs
            return [
                SimpleNamespace(
                    id="org/model",
                    likes=321,
                    downloads=12345,
                    tags=[],
                    siblings=[],
                )
            ]

    async def fake_process(models, limit, model_format):
        observed["models"] = models
        observed["limit"] = limit
        observed["format"] = model_format
        return [{"id": models[0].id, "likes": models[0].likes}]

    monkeypatch.setattr(huggingface, "_rate_limit", fake_rate_limit)
    monkeypatch.setattr(huggingface, "hf_api", FakeApi())
    monkeypatch.setattr(huggingface, "_process_models_parallel", fake_process)
    huggingface._search_cache.clear()

    result = asyncio.run(huggingface._search_with_api("Qwen", 5, "gguf"))

    assert result == [{"id": "org/model", "likes": 321}]
    assert observed["kwargs"]["search"] == "Qwen"
    assert observed["kwargs"]["filter"] == "gguf"
    assert observed["kwargs"]["sort"] == "downloads"
    assert observed["kwargs"]["full"] is True
    assert "expand" not in observed["kwargs"]
    assert observed["models"][0].likes == 321
