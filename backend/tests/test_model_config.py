"""Model config normalization, merge, and helpers."""

from backend.model_config import (
    config_api_response,
    default_engine_for_format,
    effective_model_config,
    effective_model_config_from_raw,
    merge_model_config_put,
    normalize_model_config,
    set_embedding_flag,
)


def test_normalize_empty():
    n = normalize_model_config(None)
    assert n["engine"] == "llama_cpp"
    assert n["engines"] == {}


def test_normalize_flat_dict_migrates_to_engines():
    n = normalize_model_config({"ctx_size": 4096, "engine": "llama_cpp"})
    assert n["engine"] == "llama_cpp"
    assert n["engines"]["llama_cpp"]["ctx_size"] == 4096


def test_normalize_multi_engine():
    raw = {
        "engine": "lmdeploy",
        "engines": {
            "llama_cpp": {"ctx_size": 2048},
            "lmdeploy": {"session_len": 8192},
        },
    }
    n = normalize_model_config(raw)
    assert n["engine"] == "lmdeploy"
    assert n["engines"]["lmdeploy"]["session_len"] == 8192


def test_normalize_invalid_engine_id_falls_back():
    n = normalize_model_config({"engine": "not_valid", "engines": {}})
    assert n["engine"] == "llama_cpp"


def test_effective_model_config():
    n = normalize_model_config(
        {"engine": "llama_cpp", "engines": {"llama_cpp": {"temperature": 0.7}}}
    )
    eff = effective_model_config(n)
    assert eff["engine"] == "llama_cpp"
    assert eff["temperature"] == 0.7


def test_effective_from_raw_string_json():
    eff = effective_model_config_from_raw('{"engine":"llama_cpp","ctx_size":1024}')
    assert eff["ctx_size"] == 1024


def test_config_api_response_includes_engines_map():
    n = normalize_model_config(
        {
            "engine": "llama_cpp",
            "engines": {"llama_cpp": {"n_gpu_layers": 10}},
        }
    )
    api = config_api_response(n)
    assert api["n_gpu_layers"] == 10
    assert "engines" in api
    assert api["engines"]["llama_cpp"]["n_gpu_layers"] == 10


def test_merge_put_updates_flat_params():
    existing = normalize_model_config({"engine": "llama_cpp", "threads": 4})
    merged = merge_model_config_put(existing, {"threads": 8})
    assert merged["engine"] == "llama_cpp"
    assert merged["engines"]["llama_cpp"]["threads"] == 8


def test_merge_put_replaces_engines_sections():
    existing = normalize_model_config(
        {
            "engine": "llama_cpp",
            "engines": {
                "llama_cpp": {"ctx_size": 2048},
                "lmdeploy": {"session_len": 4096},
            },
        }
    )
    merged = merge_model_config_put(
        existing,
        {
            "engine": "lmdeploy",
            "engines": {"lmdeploy": {"session_len": 8192}},
        },
    )
    assert merged["engine"] == "lmdeploy"
    assert merged["engines"]["lmdeploy"]["session_len"] == 8192
    assert merged["engines"]["llama_cpp"]["ctx_size"] == 2048


def test_default_engine_for_format():
    assert default_engine_for_format("safetensors") == "lmdeploy"
    assert default_engine_for_format("gguf") == "llama_cpp"
    assert default_engine_for_format(None) == "llama_cpp"


def test_set_embedding_flag_uses_plural_embeddings_cli_name(monkeypatch):
    class FakeStore:
        def get_active_engine_version(self, engine):
            return {"version": "v1"}

    monkeypatch.setattr("backend.data_store.get_store", lambda: FakeStore())
    monkeypatch.setattr(
        "backend.model_config.get_version_entry",
        lambda store, eng, ver: {
            "sections": [
                {
                    "params": [
                        {
                            "key": "embeddings",
                            "flags": ["--embeddings"],
                            "type": "bool",
                        }
                    ]
                }
            ]
        },
    )
    n = set_embedding_flag(None, model_format="gguf")
    assert n["engines"]["llama_cpp"]["embeddings"] is True
    assert n["engines"]["llama_cpp"]["embedding"] is True


def test_set_embedding_flag_uses_catalog_embedding_key(monkeypatch):
    class FakeStore:
        def get_active_engine_version(self, engine):
            return {"version": "v1"}

    monkeypatch.setattr("backend.data_store.get_store", lambda: FakeStore())
    monkeypatch.setattr(
        "backend.model_config.get_version_entry",
        lambda store, eng, ver: {
            "sections": [
                {
                    "params": [
                        {
                            "key": "embedding",
                            "flags": ["--embedding"],
                            "type": "bool",
                        }
                    ]
                }
            ]
        },
    )
    n = set_embedding_flag(None, model_format="gguf")
    assert n["engine"] == "llama_cpp"
    assert n["engines"]["llama_cpp"]["embedding"] is True


def test_set_embedding_flag_fallback_when_no_catalog_param(monkeypatch):
    class FakeStore:
        def get_active_engine_version(self, engine):
            return {"version": "v1"}

    monkeypatch.setattr("backend.data_store.get_store", lambda: FakeStore())
    monkeypatch.setattr(
        "backend.model_config.get_version_entry",
        lambda store, eng, ver: {"sections": [{"params": [{"key": "ctx_size", "flags": ["--ctx-size"]}]}]},
    )
    n = set_embedding_flag(None, model_format="gguf")
    assert n["engines"]["llama_cpp"]["embedding"] is True


def test_set_embedding_flag_non_embedding_key_sets_alias_and_embedding(monkeypatch):
    class FakeStore:
        def get_active_engine_version(self, engine):
            return {"version": "v1"}

    monkeypatch.setattr("backend.data_store.get_store", lambda: FakeStore())
    monkeypatch.setattr(
        "backend.model_config.get_version_entry",
        lambda store, eng, ver: {
            "sections": [
                {
                    "params": [
                        {"key": "embeddings_only", "flags": ["--embeddings-only"], "type": "bool"}
                    ]
                }
            ]
        },
    )
    n = set_embedding_flag(None, model_format="gguf")
    assert n["engines"]["llama_cpp"]["embeddings_only"] is True
    assert n["engines"]["llama_cpp"]["embedding"] is True


def test_merge_strips_empty_strings():
    existing = normalize_model_config({"engine": "llama_cpp"})
    merged = merge_model_config_put(existing, {"model_alias": ""})
    assert "model_alias" not in merged["engines"]["llama_cpp"]


def test_merge_ignores_nan_float():
    existing = normalize_model_config({"engine": "llama_cpp"})
    merged = merge_model_config_put(existing, {"temperature": float("nan")})
    assert "temperature" not in merged["engines"]["llama_cpp"]
