"""Engine param catalog normalization and registry payload shape."""

import threading
from pathlib import Path

import yaml

from backend.engine_param_catalog import (
    flags_from_entry,
    get_version_entry,
    param_index_from_entry,
    param_mapping_from_entry,
    read_catalog,
    registry_payload_from_entry,
    upsert_version_entry,
)
from backend.studio_engine_fields import studio_sections_for_engine


def test_studio_sections_are_empty_for_parser_driven_ui():
    assert studio_sections_for_engine("llama_cpp") == []
    assert studio_sections_for_engine("ik_llama") == []
    assert studio_sections_for_engine("lmdeploy") == []


def test_param_index_normalizes_primary_negative_flags_and_value_kind():
    entry = {
        "sections": [
            {
                "id": "general",
                "label": "General",
                "params": [
                    {
                        "key": "kv_offload",
                        "label": "KV Offload",
                        "type": "bool",
                        "flags": ["--no-kv-offload", "--kv-offload"],
                    },
                    {
                        "key": "stop",
                        "label": "Stop",
                        "type": "list",
                        "flags": ["--stop"],
                    },
                ],
            }
        ]
    }
    index = param_index_from_entry(entry)
    assert index["kv_offload"]["primary_flag"] == "--kv-offload"
    assert index["kv_offload"]["negative_flag"] == "--no-kv-offload"
    assert index["kv_offload"]["value_kind"] == "flag"
    assert index["stop"]["value_kind"] == "repeatable"
    assert index["stop"]["multiple"] is True


def test_param_mapping_prefers_primary_then_negative_flag():
    entry = {
        "sections": [
            {
                "id": "general",
                "label": "General",
                "params": [
                    {
                        "key": "temperature",
                        "label": "Temperature",
                        "type": "float",
                        "flags": ["--temp", "--temperature"],
                        "primary_flag": "--temperature",
                    },
                    {
                        "key": "kv_offload",
                        "label": "KV Offload",
                        "type": "bool",
                        "flags": ["--kv-offload", "--no-kv-offload"],
                    },
                ],
            }
        ]
    }
    mapping = param_mapping_from_entry(entry)
    assert mapping["temperature"] == ["--temperature", "--temp"]
    assert mapping["kv_offload"] == ["--kv-offload", "--no-kv-offload"]


def test_registry_payload_includes_canonical_metadata_without_duplicates():
    entry = {
        "scanned_at": "2025-01-01T00:00:00Z",
        "scan_error": None,
        "sections": [
            {
                "id": "general",
                "label": "General",
                "params": [
                    {
                        "key": "temperature",
                        "label": "Temperature",
                        "type": "float",
                        "flags": ["--temp", "--temperature"],
                        "primary_flag": "--temperature",
                        "value_kind": "scalar",
                        "scalar_type": "float",
                        "default": 0.8,
                    },
                    {
                        "key": "jinja",
                        "label": "Jinja",
                        "type": "bool",
                        "flags": ["--jinja"],
                    },
                ],
            }
        ],
    }
    data = registry_payload_from_entry("llama_cpp", entry, [], has_active_engine=True)
    assert data["engine"] == "llama_cpp"
    assert data["scan_pending"] is False
    assert len(data["sections"]) == 1
    params = data["sections"][0]["params"]
    assert [p["key"] for p in params] == ["temperature", "jinja"]
    assert params[0]["primary_flag"] == "--temperature"
    assert params[0]["scalar_type"] == "float"
    assert params[0]["default"] == 0.8
    assert params[1]["value_kind"] == "flag"
    assert params[1]["reserved"] is False


def test_registry_payload_scan_pending_without_entry():
    data = registry_payload_from_entry("llama_cpp", None, [], has_active_engine=True)
    assert data["scan_pending"] is True


def test_read_catalog_returns_default_root_for_invalid_store_payload():
    class FakeStore:
        def _read_yaml(self, filename):
            assert filename == "engine_params_catalog.yaml"
            return ["not", "a", "dict"]

    data = read_catalog(FakeStore())
    assert data == {"schema_version": 1, "engines": {}}


def test_upsert_and_get_version_entry_round_trip(tmp_path):
    class FakeStore:
        def __init__(self, root: Path):
            self._config_dir = str(root)
            self._lock = threading.RLock()

        def _read_yaml(self, filename):
            path = Path(self._config_dir) / filename
            if not path.exists():
                return {}
            with open(path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}

        def _write_yaml(self, path, data):
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, sort_keys=False)

    store = FakeStore(tmp_path)
    upsert_version_entry(
        store,
        "llama_cpp",
        "v1",
        {
            "scanned_at": "2026-01-01T00:00:00Z",
            "sections": [{"id": "general", "label": "General", "params": []}],
        },
    )

    entry = get_version_entry(store, "llama_cpp", "v1")
    assert entry["scanned_at"] == "2026-01-01T00:00:00Z"
    assert entry["sections"][0]["id"] == "general"


def test_flags_from_entry_deduplicates_and_ignores_scan_error():
    entry = {
        "sections": [
            {
                "params": [
                    {"flags": ["--threads", "--threads", "-t"]},
                    {"flags": ["--ctx-size"]},
                ]
            }
        ]
    }
    assert flags_from_entry(entry) == ["--threads", "--ctx-size"]
    assert flags_from_entry({"scan_error": "bad"}) == []
