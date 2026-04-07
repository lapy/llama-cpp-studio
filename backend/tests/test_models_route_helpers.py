"""Unit coverage for compact helpers in backend.routes.models."""

import asyncio

import pytest
from fastapi import HTTPException

import backend.routes.models as models_routes
import backend.services.model_downloads as model_downloads
import backend.services.model_metadata as model_metadata
from backend.utils.coercion import coerce_positive_int


def test_embedding_and_filename_helpers():
    assert models_routes._is_mmproj_filename("Some-MMProj.gguf") is True
    assert models_routes._is_mmproj_filename("weights.gguf") is False

    assert (
        model_metadata.looks_like_embedding_model("text-embedding", "plain model")
        is True
    )
    assert model_metadata.looks_like_embedding_model(None, "bge-small-en-v1.5") is True
    assert model_metadata.looks_like_embedding_model(None, "chat model") is False

    assert model_metadata.model_is_embedding({"config": {"embedding": True}}) is True
    assert (
        model_metadata.model_is_embedding(
            {"pipeline_tag": "feature-extraction", "config": {}}
        )
        is True
    )


def test_get_model_or_404_and_mmproj_sharing():
    class FakeStore:
        def get_model(self, model_id):
            return {"id": model_id} if model_id == "x" else None

        def list_models(self):
            return [
                {
                    "id": "a",
                    "huggingface_id": "org/model",
                    "mmproj_filename": "mmproj.gguf",
                },
                {
                    "id": "b",
                    "huggingface_id": "org/model",
                    "mmproj_filename": "other.gguf",
                },
            ]

    assert models_routes._get_model_or_404(FakeStore(), "x") == {"id": "x"}
    with pytest.raises(HTTPException):
        models_routes._get_model_or_404(FakeStore(), None)
    with pytest.raises(HTTPException):
        models_routes._get_model_or_404(FakeStore(), "missing")

    assert (
        models_routes._other_models_share_mmproj(
            FakeStore(), "org/model", "mmproj.gguf", "z"
        )
        is True
    )
    assert (
        models_routes._other_models_share_mmproj(
            FakeStore(), "org/model", "missing.gguf", "z"
        )
        is False
    )


def test_remove_model_from_disk_and_manifests_calls_expected_backend_actions(
    monkeypatch,
):
    calls = {}

    monkeypatch.setattr(
        models_routes,
        "purge_safetensors_repo_completely",
        lambda hf_id: calls.setdefault("safetensors", hf_id),
    )
    monkeypatch.setattr(
        models_routes,
        "purge_gguf_store_model",
        lambda hf_id, model_id, quantization: calls.setdefault(
            "gguf", (hf_id, model_id, quantization)
        ),
    )
    monkeypatch.setattr(
        models_routes,
        "delete_cached_model_file",
        lambda hf_id, filename: calls.setdefault("deleted", (hf_id, filename)),
    )
    monkeypatch.setattr(
        models_routes, "_other_models_share_mmproj", lambda *args, **kwargs: False
    )

    class FakeStore:
        pass

    asyncio.run(
        models_routes._remove_model_from_disk_and_manifests(
            FakeStore(),
            {
                "format": "safetensors",
                "huggingface_id": "org/repo",
                "id": "repo-model",
            },
        )
    )
    assert calls["safetensors"] == "org/repo"

    asyncio.run(
        models_routes._remove_model_from_disk_and_manifests(
            FakeStore(),
            {
                "format": "gguf",
                "huggingface_id": "org/model",
                "id": "gguf-model",
                "quantization": "Q4_K_M",
                "mmproj_filename": "mmproj.gguf",
            },
        )
    )
    assert calls["gguf"] == ("org/model", "gguf-model", "Q4_K_M")
    assert calls["deleted"] == ("org/model", "mmproj.gguf")


def test_architecture_and_config_helpers():
    assert model_metadata.normalize_architecture(" llama ") == "llama"
    assert model_metadata.normalize_architecture(None) == "unknown"
    assert model_metadata.detect_architecture_from_name("Qwen 2.5 Instruct") == "qwen2"
    assert model_metadata.detect_architecture_from_name("Phi-3 mini") == "phi-2"
    assert (
        models_routes._coerce_model_config('{"engine":"llama_cpp","threads":4}')[
            "threads"
        ]
        == 4
    )
    assert coerce_positive_int("4,096") == 4096
    assert coerce_positive_int(2_000_000_000) is None
    assert models_routes._apply_prompt_reservation(20000) == 11808
    assert models_routes._apply_prompt_reservation(8000) == 8000


def test_normalize_hf_overrides_validates_structure():
    assert models_routes._normalize_hf_overrides('{"rope":{"type":"yarn"}}') == {
        "rope": {"type": "yarn"}
    }
    assert models_routes._normalize_hf_overrides({"a": 1, "b": {"c": True}}) == {
        "a": 1,
        "b": {"c": True},
    }
    with pytest.raises(HTTPException, match="valid JSON"):
        models_routes._normalize_hf_overrides("{bad json}")
    with pytest.raises(HTTPException, match="non-empty strings"):
        models_routes._normalize_hf_overrides({"": 1})
    with pytest.raises(HTTPException, match="scalars or nested objects"):
        models_routes._normalize_hf_overrides({"x": []})


def test_refresh_gguf_model_metadata_updates_store_and_falls_back_on_name(
    monkeypatch, tmp_path
):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_text("gguf", encoding="utf-8")

    monkeypatch.setattr(
        model_metadata,
        "get_model_layer_info",
        lambda path: {
            "architecture": "unknown",
            "layer_count": 32,
            "context_length": 8192,
            "parameter_count": 123,
        },
    )

    class FakeStore:
        def __init__(self):
            self.updated = None

        def update_model(self, model_id, fields):
            self.updated = (model_id, fields)

    store = FakeStore()
    result = model_metadata.refresh_gguf_model_metadata(
        {
            "id": "m1",
            "display_name": "Qwen helper",
            "huggingface_id": "org/model",
            "model_type": "old",
        },
        store,
        str(gguf_path),
    )

    assert result["updated_fields"] == {"model_type": "qwen2"}
    assert result["metadata"]["layer_count"] == 32
    assert store.updated == ("m1", {"model_type": "qwen2"})


def test_collect_safetensors_runtime_metadata_merges_details_and_tensor_summary(
    monkeypatch,
):
    async def fake_get_model_details(hf_id):
        return {
            "architecture": "llama",
            "base_model": "base",
            "pipeline_tag": "text-generation",
            "parameters": "7B",
            "model_max_length": "16384",
            "config": {"max_position_embeddings": "8192"},
            "language": ["en"],
            "license": "apache-2.0",
        }

    async def fake_get_safetensors_metadata_summary(hf_id):
        return {
            "files": [
                {
                    "filename": "model-00001-of-00002.safetensors",
                    "tensor_count": 123,
                    "dtype_counts": {"F16": 12},
                }
            ]
        }

    monkeypatch.setattr(model_downloads, "get_model_details", fake_get_model_details)
    monkeypatch.setattr(
        model_downloads,
        "get_safetensors_metadata_summary",
        fake_get_safetensors_metadata_summary,
    )
    monkeypatch.setattr(
        model_downloads,
        "get_tokenizer_config",
        lambda hf_id: {"model_max_length": "32768"},
    )

    metadata, tensor_summary, max_context = asyncio.run(
        model_downloads.collect_safetensors_runtime_metadata(
            "org/repo",
            "model-00001-of-00002.safetensors",
        )
    )

    assert metadata["architecture"] == "llama"
    assert metadata["model_max_length"] == 32768
    assert metadata["max_context_length"] == 16384
    assert metadata["tokenizer_config"]["model_max_length"] == "32768"
    assert tensor_summary == {"tensor_count": 123, "dtype_counts": {"F16": 12}}
    assert max_context == 16384
