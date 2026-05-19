"""Data store helpers and isolated DataStore operations."""

from backend.data_store import (
    DataStore,
    collect_config_swap_aliases,
    find_swap_name_conflicts,
    generate_proxy_name,
    normalize_proxy_alias,
    resolve_llama_swap_id,
    resolve_proxy_name,
    resolve_routing_name,
)


def test_generate_proxy_name_basic():
    assert generate_proxy_name("org/model-name") == "org-model-name"


def test_generate_proxy_name_with_quantization():
    assert generate_proxy_name("org/model", "Q4_K_M") == "org-model.q4_k_m"


def test_normalize_proxy_alias():
    assert normalize_proxy_alias("  My Model / Test  ") == "my-model-test"
    assert normalize_proxy_alias(None) == ""
    assert normalize_proxy_alias("a@b#c") == "a-b-c"


def test_resolve_proxy_name_is_stable_even_with_model_alias():
    model = {
        "proxy_name": "org-model.q4_k_m",
        "huggingface_id": "x/y",
        "config": {
            "engine": "llama_cpp",
            "engines": {"llama_cpp": {"model_alias": "my-alias"}},
        },
    }
    assert resolve_proxy_name(model) == "org-model.q4_k_m"
    assert resolve_llama_swap_id(model) == "org-model.q4_k_m"
    assert resolve_routing_name(model) == "my-alias"


def test_collect_config_swap_aliases():
    config = {
        "model_alias": "my-app",
        "swap_aliases": ["extra-alias", "my-app"],
    }
    assert collect_config_swap_aliases(config, "org-model.q4_k_m") == [
        "my-app",
        "extra-alias",
    ]


def test_find_swap_name_conflicts(tmp_path):
    cfg = tmp_path / "config"
    store = DataStore(config_dir=str(cfg))
    store.add_model(
        {
            "id": "m1",
            "proxy_name": "model-a",
            "huggingface_id": "org/a",
            "config": {
                "engine": "llama_cpp",
                "engines": {"llama_cpp": {"model_alias": "shared"}},
            },
        }
    )
    store.add_model(
        {
            "id": "m2",
            "proxy_name": "model-b",
            "huggingface_id": "org/b",
            "config": {"engine": "llama_cpp", "engines": {"llama_cpp": {}}},
        }
    )
    conflicts = find_swap_name_conflicts(
        store,
        "m2",
        {"model_alias": "shared"},
    )
    assert conflicts == ["shared"]


def test_resolve_proxy_name_fallback_generate():
    model = {"huggingface_id": "meta/llama", "quantization": "q8"}
    assert resolve_proxy_name(model) == "meta-llama.q8"


def test_data_store_roundtrip_model(tmp_path):
    cfg = tmp_path / "config"
    store = DataStore(config_dir=str(cfg))
    m = {
        "id": "m1",
        "huggingface_id": "hf/test",
        "name": "Test",
        "config": {},
    }
    store.add_model(m)
    models = store.list_models()
    assert len(models) == 1
    assert store.get_model("m1")["huggingface_id"] == "hf/test"


def test_data_store_engine_version_and_active(tmp_path):
    cfg = tmp_path / "config"
    store = DataStore(config_dir=str(cfg))
    store.add_engine_version(
        "llama_cpp",
        {"version": "v1", "binary_path": "/bin/llama", "type": "source"},
    )
    store.set_active_engine_version("llama_cpp", "v1")
    active = store.get_active_engine_version("llama_cpp")
    assert active["version"] == "v1"
    vers = store.get_engine_versions("llama_cpp")
    assert len(vers) == 1


def test_data_store_build_settings_merge(tmp_path):
    cfg = tmp_path / "config"
    store = DataStore(config_dir=str(cfg))
    store.update_engine_build_settings("llama_cpp", {"CMAKE_ARGS": "-DFOO=1"})
    out = store.update_engine_build_settings("llama_cpp", {"GGML_CUDA": "ON"})
    assert out["CMAKE_ARGS"] == "-DFOO=1"
    assert out["GGML_CUDA"] == "ON"
