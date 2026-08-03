"""llama-swap profiles/selectors routing helpers."""

from __future__ import annotations

import yaml
import pytest

import backend.llama_swap_config as llama_swap_config
import backend.llama_swap_routing as routing
from backend import data_store
from backend.llama_swap_manager import (
    _configs_semantically_equal,
    summarize_llama_swap_yaml_diff,
)


def _install_store(monkeypatch, tmp_path):
    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    return store


def _seed_gguf(store, *, alias=None):
    engines = {"llama_cpp": {}}
    if alias:
        engines["llama_cpp"]["model_alias"] = alias
    return store.add_model(
        {
            "id": "org/model",
            "name": "Model",
            "huggingface_id": "org/model",
            "quantization": "Q4_K_M",
            "format": "gguf",
            "config": {"engine": "llama_cpp", "engines": engines},
        }
    )


def test_data_store_routing_defaults(monkeypatch, tmp_path):
    store = _install_store(monkeypatch, tmp_path)
    assert store.get_llama_swap_routing() == {"profiles": {}, "selectors": {}}
    store.set_llama_swap_routing(
        {"profiles": {"a": {"description": "", "pins": {"x": "y"}}}, "selectors": {}}
    )
    assert store.get_llama_swap_routing()["profiles"]["a"]["pins"]["x"] == "y"


def test_normalize_and_validate_selector_and_profile(monkeypatch, tmp_path):
    store = _install_store(monkeypatch, tmp_path)
    _seed_gguf(store, alias="chat")

    doc = routing.normalize_routing_document(
        {
            "selectors": {
                "Coding Model": {
                    "strategy": "WARM",
                    "targets": ["chat", "org-model.q4_k_m", "chat"],
                    "name": "Coding",
                    "description": "Prefer loaded",
                },
                "spill": {
                    "strategy": "spillover",
                    "targets": ["chat"],
                    "spillover": 3,
                },
            },
            "profiles": {
                "Coding": {
                    "description": "Dev day",
                    "pins": {
                        "llm": "coding-model",
                        "disabled": None,
                    },
                }
            },
        }
    )

    assert set(doc["selectors"]) == {"coding-model", "spill"}
    assert doc["selectors"]["coding-model"]["strategy"] == "warm"
    assert doc["selectors"]["coding-model"]["targets"] == ["chat", "org-model.q4_k_m"]
    assert doc["selectors"]["spill"]["settings"]["spillover"] == 3
    assert doc["profiles"]["coding"]["pins"]["llm"] == "coding-model"
    assert doc["profiles"]["coding"]["pins"]["disabled"] == ""

    assert routing.validate_routing_document(doc, store=store) == []
    saved = routing.save_routing_document(doc, store=store)
    assert store.get_llama_swap_routing()["selectors"]["coding-model"]["name"] == "Coding"
    assert saved["profiles"]["coding"]["description"] == "Dev day"


def test_validate_rejects_bad_strategy_and_selector_collision(monkeypatch, tmp_path):
    store = _install_store(monkeypatch, tmp_path)
    _seed_gguf(store)

    bad = {
        "selectors": {
            "org-model.q4_k_m": {
                "strategy": "nope",
                "targets": ["other-sel"],
            },
            "other-sel": {"strategy": "pin", "targets": ["org-model.q4_k_m"]},
        },
        "profiles": {},
    }
    errors = routing.validate_routing_document(bad, store=store)
    assert any("strategy" in e for e in errors)
    assert any("conflicts with an existing model" in e for e in errors)
    assert any("selector chaining" in e for e in errors)


def test_validate_rejects_normalized_id_collisions_and_empty_profile_pins(
    monkeypatch, tmp_path
):
    store = _install_store(monkeypatch, tmp_path)
    raw = {
        "selectors": {
            "Fast Chat": {"strategy": "pin", "targets": ["a"]},
            "fast-chat": {"strategy": "pin", "targets": ["a"]},
        },
        "profiles": {
            "empty": {"description": "x", "pins": {}},
        },
    }
    errors = routing.validate_routing_document(raw, store=store, raw=raw)
    assert any("both normalize to" in e for e in errors)
    assert any("pins must contain at least one pin" in e for e in errors)


def test_validate_rejects_profile_selector_id_overlap_and_warm_peer(
    monkeypatch, tmp_path
):
    store = _install_store(monkeypatch, tmp_path)
    raw = {
        "selectors": {
            "shared": {
                "strategy": "warm",
                "targets": ["remote/peer-model"],
            }
        },
        "profiles": {
            "shared": {"description": "", "pins": {"llm": "x"}},
        },
    }
    errors = routing.validate_routing_document(raw, store=store, raw=raw)
    assert any("both a profile and a selector" in e for e in errors)
    assert any("local model" in e and "warm" in e for e in errors)


def test_routing_warnings_for_unknown_targets(monkeypatch, tmp_path):
    store = _install_store(monkeypatch, tmp_path)
    _seed_gguf(store)
    warnings = routing.routing_warnings(
        {
            "selectors": {
                "fast": {"strategy": "pin", "targets": ["missing-model"]},
            },
            "profiles": {
                "coding": {"description": "", "pins": {"llm": "also-missing"}},
            },
        },
        store=store,
    )
    assert any("missing-model" in w for w in warnings)
    assert any("also-missing" in w for w in warnings)


def test_routing_for_yaml_omits_empty(monkeypatch, tmp_path):
    store = _install_store(monkeypatch, tmp_path)
    assert routing.routing_for_yaml(store) == {}
    store.set_llama_swap_routing(
        {
            "profiles": {},
            "selectors": {
                "fast": {"strategy": "pin", "targets": ["a"], "name": "Fast"}
            },
        }
    )
    slice_ = routing.routing_for_yaml(store)
    assert "profiles" not in slice_
    assert slice_["selectors"]["fast"]["name"] == "Fast"


def test_save_routing_document_raises(monkeypatch, tmp_path):
    store = _install_store(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="strategy"):
        routing.save_routing_document(
            {
                "profiles": {},
                "selectors": {"x": {"strategy": "nope", "targets": ["a"]}},
            },
            store=store,
        )


def test_generate_llama_swap_config_emits_routing(monkeypatch, tmp_path):
    store = _install_store(monkeypatch, tmp_path)
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")

    store.add_model(
        {
            "id": "org/model",
            "name": "Model",
            "huggingface_id": "org/model",
            "quantization": "Q4_K_M",
            "file_path": str(model_path),
            "format": "gguf",
            "config": {
                "engine": "llama_cpp",
                "engines": {"llama_cpp": {"temperature": 0.2}},
            },
        }
    )
    store.set_llama_swap_routing(
        {
            "profiles": {
                "coding": {
                    "description": "Coding",
                    "pins": {"llm": "fast-chat"},
                }
            },
            "selectors": {
                "fast-chat": {
                    "strategy": "pin",
                    "targets": ["org-model.q4_k_m"],
                    "name": "Fast chat",
                }
            },
        }
    )

    monkeypatch.setattr(
        llama_swap_config,
        "get_active_binary_path_for_engine",
        lambda _store, engine: str(binary) if engine == "llama_cpp" else None,
    )
    monkeypatch.setattr(
        llama_swap_config,
        "resolve_gguf_model_path_for_quant",
        lambda hf_id, quant: str(model_path),
    )
    monkeypatch.setattr(
        llama_swap_config,
        "resolve_llama_server_invocation_paths",
        lambda path: (str(binary), str(tmp_path)),
    )
    monkeypatch.setattr(
        llama_swap_config, "_resolve_cuda_library_path", lambda cwd: "/fake/lib"
    )
    monkeypatch.setattr(
        llama_swap_config, "infer_engine_id_for_binary", lambda path: "llama_cpp"
    )
    monkeypatch.setattr(
        llama_swap_config,
        "_active_engine_param_index",
        lambda engine: {
            "temperature": {"primary_flag": "--temperature", "value_kind": "scalar"}
        },
    )

    yaml_str = llama_swap_config.generate_llama_swap_config(
        {},
        all_models=store.list_models(),
    )
    doc = yaml.safe_load(yaml_str)
    assert "org-model.q4_k_m" in doc["models"]
    assert doc["selectors"]["fast-chat"]["strategy"] == "pin"
    assert doc["profiles"]["coding"]["pins"]["llm"] == "fast-chat"
    keys = list(doc.keys())
    assert keys.index("profiles") < keys.index("models")
    assert keys.index("selectors") < keys.index("models")


def test_summarize_and_semantic_equality_for_routing():
    disk = "models: {}\n"
    desired = """
profiles:
  coding:
    description: x
    pins:
      llm: a
selectors:
  fast:
    strategy: pin
    targets: [a]
models: {}
"""
    lines = summarize_llama_swap_yaml_diff(disk, desired)
    assert any("profile" in line for line in lines)
    assert any("selector" in line for line in lines)
    assert not _configs_semantically_equal(disk, desired)
    assert _configs_semantically_equal(desired, desired)
