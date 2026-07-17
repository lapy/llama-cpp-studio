"""Search metadata regressions for backend.huggingface."""

import asyncio
from types import SimpleNamespace

import backend.huggingface as huggingface


def test_is_mtp_filename_and_excludes_from_quant_grouping():
    from backend.huggingface import (
        is_mtp_filename,
        is_mmproj_filename,
        is_auxiliary_gguf_filename,
        mtp_option_label,
        _process_single_model,
    )

    assert is_mtp_filename("MTP/mtp-gemma-4-31B-it-Q8_0.gguf") is True
    assert is_mtp_filename("mtp-gemma-4-31B-it.gguf") is True
    assert is_mtp_filename("MTP/gemma-4-31B-it-Q8_0-MTP.gguf") is True
    assert is_mtp_filename("gemma-4-31B-it-Q8_0.gguf") is False
    # Model family name contains MTP — these are main weights, not companions.
    assert is_mtp_filename("Qwen3.6-27B-MTP.gguf") is False
    assert is_mtp_filename("Qwen3.6-27B-MTP-Q8_0.gguf") is False
    assert is_mtp_filename("Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf") is False
    assert is_mmproj_filename("mmproj-F16.gguf") is True
    assert is_auxiliary_gguf_filename("MTP/mtp-x.gguf") is True
    assert mtp_option_label("MTP/mtp-gemma-4-31B-it-Q8_0.gguf") == "Q8_0"
    assert mtp_option_label("mtp-gemma-4-31B-it.gguf") == "Default"

    model = SimpleNamespace(
        id="unsloth/gemma-4-31B-it-GGUF",
        modelId="unsloth/gemma-4-31B-it-GGUF",
        author="unsloth",
        downloads=10,
        likes=1,
        tags=[],
        siblings=[
            SimpleNamespace(rfilename="gemma-4-31B-it-Q8_0.gguf", size=1000),
            SimpleNamespace(rfilename="MTP/mtp-gemma-4-31B-it-Q8_0.gguf", size=200),
            SimpleNamespace(rfilename="mtp-gemma-4-31B-it.gguf", size=180),
            SimpleNamespace(rfilename="mmproj-F16.gguf", size=50),
        ],
    )

    result = asyncio.run(_process_single_model(model, "gguf"))
    assert result is not None
    assert set(result["quantizations"].keys()) == {"Q8_0"}
    assert len(result["quantizations"]["Q8_0"]["files"]) == 1
    assert result["quantizations"]["Q8_0"]["files"][0]["filename"] == (
        "gemma-4-31B-it-Q8_0.gguf"
    )
    assert len(result["mtp_files"]) == 2
    assert len(result["mmproj_files"]) == 1

    mtp_named_model = SimpleNamespace(
        id="unsloth/Qwen3.6-27B-MTP-GGUF",
        modelId="unsloth/Qwen3.6-27B-MTP-GGUF",
        author="unsloth",
        downloads=10,
        likes=1,
        tags=[],
        siblings=[
            SimpleNamespace(rfilename="Qwen3.6-27B-MTP-Q8_0.gguf", size=1000),
            SimpleNamespace(rfilename="Qwen3.6-27B-MTP-Q4_K_M.gguf", size=800),
            SimpleNamespace(rfilename="Qwen3.6-27B-MTP.gguf", size=2000),
        ],
    )
    mtp_named = asyncio.run(_process_single_model(mtp_named_model, "gguf"))
    assert mtp_named is not None
    assert mtp_named["mtp_files"] == []
    assert set(mtp_named["quantizations"].keys()) >= {"Q8_0", "Q4_K_M"}
    assert all(
        not f["filename"].lower().startswith("mtp")
        for entry in mtp_named["quantizations"].values()
        for f in entry["files"]
    )


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
