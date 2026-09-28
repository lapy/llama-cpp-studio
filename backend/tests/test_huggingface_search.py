"""Search metadata regressions for backend.models.hub."""

import asyncio
from types import SimpleNamespace

import backend.models.hub as huggingface


def test_is_mtp_filename_and_excludes_from_quant_grouping():
    from backend.models.hub import (
        is_mtp_filename,
        is_dflash_filename,
        is_mmproj_filename,
        is_auxiliary_gguf_filename,
        mtp_option_label,
        dflash_option_label,
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
    assert is_dflash_filename("laguna-s-2.1-DFlash-BF16.gguf") is True
    assert is_dflash_filename("DFlash/laguna-draft.gguf") is True
    assert is_dflash_filename("dflash-laguna-BF16.gguf") is True
    assert is_dflash_filename("laguna-s-2.1-Q4_K_M.gguf") is False
    assert is_auxiliary_gguf_filename("MTP/mtp-x.gguf") is True
    assert is_auxiliary_gguf_filename("laguna-s-2.1-DFlash-BF16.gguf") is True
    assert mtp_option_label("MTP/mtp-gemma-4-31B-it-Q8_0.gguf") == "Q8_0"
    assert mtp_option_label("mtp-gemma-4-31B-it.gguf") == "Default"
    assert dflash_option_label("laguna-s-2.1-DFlash-BF16.gguf") == "BF16"

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

    laguna_model = SimpleNamespace(
        id="poolside/Laguna-S-2.1-GGUF",
        modelId="poolside/Laguna-S-2.1-GGUF",
        author="poolside",
        downloads=10,
        likes=1,
        tags=[],
        siblings=[
            SimpleNamespace(rfilename="laguna-s-2.1-Q4_K_M.gguf", size=68000),
            SimpleNamespace(rfilename="laguna-s-2.1-DFlash-BF16.gguf", size=2200),
        ],
    )
    laguna = asyncio.run(_process_single_model(laguna_model, "gguf"))
    assert laguna is not None
    assert set(laguna["quantizations"].keys()) == {"Q4_K_M"}
    assert len(laguna["dflash_files"]) == 1
    assert laguna["dflash_files"][0]["filename"] == "laguna-s-2.1-DFlash-BF16.gguf"
    assert laguna["dflash_files"][0]["label"] == "BF16"

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


def test_apply_remote_sizes_fills_quant_and_safetensors_totals():
    from backend.models.hub import _apply_remote_sizes

    row = {
        "id": "org/model",
        "quantizations": {
            "Q4_K_M": {
                "files": [
                    {"filename": "a-q4_k_m.gguf", "size": 0},
                    {"filename": "b-q4_k_m.gguf", "size": 0},
                ],
                "total_size": 0,
            }
        },
        "mmproj_files": [{"filename": "mmproj-f16.gguf", "size": 0}],
        "safetensors_files": [{"filename": "model.safetensors", "size": 0}],
    }
    _apply_remote_sizes(
        row,
        {
            "a-q4_k_m.gguf": 10,
            "b-q4_k_m.gguf": 5,
            "mmproj-f16.gguf": 3,
            "model.safetensors": 40,
        },
    )
    assert row["quantizations"]["Q4_K_M"]["total_size"] == 15
    assert row["mmproj_files"][0]["size"] == 3
    assert row["total_size"] == 40


def test_extract_quantization_accepts_lowercase_hub_filenames():
    from backend.models.hub import _extract_quantization, _process_single_model

    assert _extract_quantization("qwen2.5-7b-instruct-q4_k_m.gguf") == "Q4_K_M"
    assert _extract_quantization("qwen2.5-7b-instruct-fp16.gguf") == "FP16"
    assert _extract_quantization("model-Q4_K_M.gguf") == "Q4_K_M"

    model = SimpleNamespace(
        id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        modelId="Qwen/Qwen2.5-7B-Instruct-GGUF",
        author="Qwen",
        downloads=10,
        likes=1,
        tags=[],
        siblings=[
            SimpleNamespace(
                rfilename="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                size=3993201344,
            ),
            SimpleNamespace(
                rfilename="qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
                size=689872288,
            ),
            SimpleNamespace(rfilename="qwen2.5-7b-instruct-fp16.gguf", size=100),
        ],
    )
    result = asyncio.run(_process_single_model(model, "gguf"))
    assert result is not None
    assert set(result["quantizations"]) == {"Q4_K_M", "FP16"}
    assert result["quantizations"]["Q4_K_M"]["total_size"] == 3993201344 + 689872288


def test_repo_id_from_query_accepts_only_org_name():
    from backend.models.hub import repo_id_from_query

    assert repo_id_from_query("Qwen/Qwen2.5-7B-Instruct-GGUF") == "Qwen/Qwen2.5-7B-Instruct-GGUF"
    assert repo_id_from_query("qwen 7b") == ""
    assert repo_id_from_query("org/name/extra") == ""


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
    assert "full" not in observed["kwargs"]
    assert "likes" in observed["kwargs"]["expand"]
    assert "cardData" in observed["kwargs"]["expand"]
    assert "gguf" in observed["kwargs"]["expand"]
    assert "safetensors" in observed["kwargs"]["expand"]
    assert observed["models"][0].likes == 321


def test_process_single_model_keeps_live_card_and_gguf_fields():
    from backend.models.hub import _process_single_model

    card = SimpleNamespace(
        to_dict=lambda: {
            "license": "apache-2.0",
            "language": ["en"],
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "pipeline_tag": "text-generation",
        }
    )
    model = SimpleNamespace(
        id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        modelId="Qwen/Qwen2.5-7B-Instruct-GGUF",
        author="Qwen",
        downloads=100,
        likes=12,
        tags=["gguf"],
        pipeline_tag="text-generation",
        sha="abc123",
        last_modified="2024-09-20T06:38:28+00:00",
        gated=False,
        private=False,
        library_name=None,
        card_data=card,
        gguf={
            "architecture": "qwen2",
            "context_length": 131072,
            "total": 7615616512,
        },
        siblings=[SimpleNamespace(rfilename="model-Q4_K_M.gguf", size=None)],
    )

    result = asyncio.run(_process_single_model(model, "gguf"))

    assert result["likes"] == 12
    assert result["license"] == "apache-2.0"
    assert result["language"] == ["en"]
    assert result["base_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert result["architecture"] == "qwen2"
    assert result["context_length"] == 131072
    assert result["parameters"] == "7.6B"
    assert result["sha"] == "abc123"
    assert result["updated_at"] == "2024-09-20T06:38:28+00:00"
    assert result["quantizations"]["Q4_K_M"]["files"][0]["filename"] == "model-Q4_K_M.gguf"

    from backend.model_catalog.huggingface_provider import HuggingFaceCatalogProvider

    item = HuggingFaceCatalogProvider._normalize(result, "gguf")
    assert item["family"] == "qwen2"
    assert item["languages"] == ["en"]
    assert item["source"]["revision"] == "abc123"
    assert item["metadata"]["last_modified"] == "2024-09-20T06:38:28+00:00"
    assert item["metadata"]["parameters"] == "7.6B"
    assert item["metadata"]["context_length"] == 131072
    assert item["metadata"]["license"] == "apache-2.0"
