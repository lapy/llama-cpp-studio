"""Model config template extract/apply and API helpers."""

import pytest

from backend.data_store import DataStore
from backend.model_config import normalize_model_config
from backend.model_config_templates import (
    apply_template_to_config,
    extract_template_config,
    new_template_record,
)


def test_extract_template_omits_routing_by_default():
    raw = normalize_model_config(
        {
            "engine": "llama_cpp",
            "engines": {
                "llama_cpp": {
                    "temperature": 0.7,
                    "model_alias": "my-app",
                    "swap_aliases": ["extra"],
                },
                "ik_llama": {"temperature": 0.5},
            },
        }
    )
    snap = extract_template_config(raw, include_routing=False, engines_scope="all")
    assert snap["engines"]["llama_cpp"]["temperature"] == 0.7
    assert "model_alias" not in snap["engines"]["llama_cpp"]
    assert "swap_aliases" not in snap["engines"]["llama_cpp"]
    assert snap["engines"]["ik_llama"]["temperature"] == 0.5


def test_extract_template_active_engine_only():
    raw = normalize_model_config(
        {
            "engine": "llama_cpp",
            "engines": {
                "llama_cpp": {"temperature": 0.7},
                "ik_llama": {"temperature": 0.5},
            },
        }
    )
    snap = extract_template_config(raw, engines_scope="active")
    assert list(snap["engines"].keys()) == ["llama_cpp"]


def test_apply_template_active_engine_keeps_target_engine():
    existing = normalize_model_config(
        {
            "engine": "ik_llama",
            "engines": {
                "ik_llama": {"temperature": 0.1, "model_alias": "keep-me"},
                "llama_cpp": {"temperature": 0.9},
            },
        }
    )
    template = normalize_model_config(
        {
            "engine": "llama_cpp",
            "engines": {"llama_cpp": {"temperature": 0.7, "n_gpu_layers": 32}},
        }
    )
    merged = apply_template_to_config(
        existing, template, include_routing=False, apply_engines="active"
    )
    assert merged["engine"] == "ik_llama"
    assert merged["engines"]["ik_llama"]["temperature"] == 0.7
    assert merged["engines"]["ik_llama"]["n_gpu_layers"] == 32
    assert merged["engines"]["ik_llama"]["model_alias"] == "keep-me"
    assert merged["engines"]["llama_cpp"]["temperature"] == 0.9


def test_apply_template_set_engine():
    existing = normalize_model_config({"engine": "ik_llama", "engines": {"ik_llama": {}}})
    template = normalize_model_config(
        {"engine": "llama_cpp", "engines": {"llama_cpp": {"temperature": 0.5}}}
    )
    merged = apply_template_to_config(
        existing, template, apply_engines="set_engine"
    )
    assert merged["engine"] == "llama_cpp"
    assert merged["engines"]["llama_cpp"]["temperature"] == 0.5


def test_new_template_record_requires_name():
    with pytest.raises(ValueError, match="name"):
        new_template_record(
            name="  ",
            config=normalize_model_config(
                {"engine": "llama_cpp", "engines": {"llama_cpp": {"temperature": 1}}}
            ),
        )


def test_data_store_config_templates_roundtrip(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    record = new_template_record(
        name="Reasoning defaults",
        description="test",
        config=normalize_model_config(
            {"engine": "llama_cpp", "engines": {"llama_cpp": {"temperature": 0.8}}}
        ),
    )
    store.add_config_template(record)
    listed = store.list_config_templates()
    assert len(listed) == 1
    assert store.get_config_template(record["id"])["name"] == "Reasoning defaults"
    assert store.delete_config_template(record["id"])
    assert store.list_config_templates() == []
