"""Engine param catalog normalization and registry payload shape."""

import threading
from pathlib import Path

import yaml

from backend.engines.params import (
    embedding_mode_config_key_from_entry,
    flags_from_entry,
    get_version_entry,
    param_index_from_entry,
    param_mapping_from_entry,
    read_catalog,
    registry_payload_from_entry,
    scoped_param_index_from_entry,
    get_model_profile_entry,
    upsert_model_profile_entry,
    upsert_version_entry,
)


class _FakeStore:
    """Minimal store surface the catalog writers use (config dir + lock + YAML)."""

    def __init__(self, root):
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


def _fake_store(root):
    return _FakeStore(root)


def test_embedding_mode_key_from_plural_cli_flag():
    entry = {
        "sections": [
            {
                "params": [
                    {
                        "key": "embeddings",
                        "type": "bool",
                        "flags": ["--embeddings"],
                    }
                ]
            }
        ]
    }
    assert embedding_mode_config_key_from_entry(entry) == "embeddings"


def test_embedding_mode_key_from_singular_cli_flag():
    entry = {
        "sections": [
            {
                "params": [
                    {"key": "embedding", "type": "bool", "flags": ["--embedding"]},
                ]
            }
        ]
    }
    assert embedding_mode_config_key_from_entry(entry) == "embedding"


def test_embedding_mode_key_negative_form_still_detects():
    entry = {
        "sections": [
            {
                "params": [
                    {
                        "key": "embeddings",
                        "type": "bool",
                        "flags": ["--no-embeddings", "--embeddings"],
                    }
                ]
            }
        ]
    }
    assert embedding_mode_config_key_from_entry(entry) == "embeddings"


def test_embedding_mode_key_fallback_embedding_prefixed_config_key():
    entry = {
        "sections": [
            {
                "params": [
                    {
                        "key": "embeddings_only",
                        "type": "bool",
                        "flags": ["--embeddings-only"],
                    }
                ]
            }
        ]
    }
    assert embedding_mode_config_key_from_entry(entry) == "embeddings_only"


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


def test_param_index_preserves_csv_enum_multiselect():
    entry = {
        "sections": [
            {
                "id": "speculative_params",
                "label": "Speculative",
                "params": [
                    {
                        "key": "spec_type",
                        "label": "Spec Type",
                        "value_kind": "csv_enum",
                        "type": "multiselect",
                        "multiple": True,
                        "options": [
                            {"value": "none", "label": "none"},
                            {"value": "draft-simple", "label": "draft-simple"},
                        ],
                        "default": ["none"],
                        "flags": ["--spec-type"],
                        "primary_flag": "--spec-type",
                    },
                ],
            }
        ]
    }
    index = param_index_from_entry(entry)
    assert index["spec_type"]["value_kind"] == "csv_enum"
    assert index["spec_type"]["type"] == "multiselect"
    assert index["spec_type"]["multiple"] is True


def test_param_index_preserves_semicolon_enum_multiselect():
    entry = {
        "sections": [
            {
                "id": "sampling_params",
                "label": "Sampling",
                "params": [
                    {
                        "key": "samplers",
                        "label": "Samplers",
                        "value_kind": "semicolon_enum",
                        "type": "multiselect",
                        "multiple": True,
                        "options": [
                            {"value": "penalties", "label": "penalties"},
                            {"value": "dry", "label": "dry"},
                        ],
                        "default": ["penalties", "dry"],
                        "flags": ["--samplers"],
                        "primary_flag": "--samplers",
                    },
                ],
            }
        ]
    }
    index = param_index_from_entry(entry)
    assert index["samplers"]["value_kind"] == "semicolon_enum"
    assert index["samplers"]["type"] == "multiselect"
    assert index["samplers"]["multiple"] is True


def test_param_index_normalizes_csv_repeatable_rows_back_to_scalar():
    entry = {
        "sections": [
            {
                "id": "general",
                "label": "General",
                "params": [
                    {
                        "key": "tensor_split",
                        "label": "Tensor Split",
                        "type": "list",
                        "value_kind": "repeatable",
                        "multiple": True,
                        "description": "fraction of the model to offload to each GPU, comma-separated list of proportions",
                        "flags": ["--tensor-split"],
                    },
                ],
            }
        ]
    }
    index = param_index_from_entry(entry)
    assert index["tensor_split"]["value_kind"] == "scalar"
    assert index["tensor_split"]["multiple"] is False
    assert index["tensor_split"]["type"] == "string"


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
    data = registry_payload_from_entry("llama_cpp", entry, has_active_engine=True)
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
    data = registry_payload_from_entry("llama_cpp", None, has_active_engine=True)
    assert data["scan_pending"] is True


def test_read_catalog_returns_default_root_for_invalid_store_payload():
    class FakeStore:
        def _read_yaml(self, filename):
            assert filename == "engine_params_catalog.yaml"
            return ["not", "a", "dict"]

    data = read_catalog(FakeStore())
    assert data == {"schema_version": 2, "engines": {}}


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

    upsert_model_profile_entry(
        store,
        "audio_cpp",
        "v1",
        "fingerprint",
        {
            "fingerprint": "fingerprint",
            "sections": [
                {
                    "id": "load",
                    "params": [
                        {
                            "key": "voice.path",
                            "scope": "load_option",
                            "transport": "key_value_option",
                        }
                    ],
                }
            ],
        },
    )
    profile = get_model_profile_entry(store, "audio_cpp", "v1", "fingerprint")
    assert profile["fingerprint"] == "fingerprint"
    scoped = scoped_param_index_from_entry(profile)
    assert "voice.path" in scoped["load_option"]


def test_upsert_serves_the_next_read_from_memory(tmp_path, monkeypatch):
    """A write must not force the next reader to re-parse the catalog YAML.

    After an audio.cpp scan the catalog holds megabytes of help text; the UI
    polls status immediately, and that poll used to parse the file on the event
    loop (blocking the API and the proxy for seconds).
    """
    from backend.engines import params

    store = _fake_store(tmp_path)
    params._CATALOG_CACHE.clear()
    upsert_version_entry(store, "audio_cpp", "v9", {"capabilities": {"families": ["demo"]}})

    def _no_parse(*args, **kwargs):
        raise AssertionError("catalog was re-parsed after a write")

    monkeypatch.setattr(params.yaml, "load", _no_parse)
    monkeypatch.setattr(params.yaml, "safe_load", _no_parse)
    entry = get_version_entry(store, "audio_cpp", "v9")
    assert entry["capabilities"]["families"] == ["demo"]


def test_external_catalog_change_still_invalidates_the_cache(tmp_path):
    """Editing the YAML outside the writers must be picked up (mtime+size key)."""
    from backend.engines import params

    store = _fake_store(tmp_path)
    params._CATALOG_CACHE.clear()
    upsert_version_entry(store, "audio_cpp", "v1", {"scanned_at": "first"})
    assert get_version_entry(store, "audio_cpp", "v1")["scanned_at"] == "first"

    path = Path(tmp_path) / params.CATALOG_FILENAME
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["engines"]["audio_cpp"]["versions"]["v1"]["scanned_at"] = "second-and-longer"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    assert get_version_entry(store, "audio_cpp", "v1")["scanned_at"] == "second-and-longer"


def test_profile_upsert_keeps_the_catalog_cache_warm(tmp_path, monkeypatch):
    from backend.engines import params

    store = _fake_store(tmp_path)
    params._CATALOG_CACHE.clear()
    upsert_version_entry(store, "audio_cpp", "v1", {"scanned_at": "x"})
    upsert_model_profile_entry(store, "audio_cpp", "v1", "fp", {"fingerprint": "fp"})

    def _no_parse(*args, **kwargs):
        raise AssertionError("catalog was re-parsed after a profile write")

    monkeypatch.setattr(params.yaml, "load", _no_parse)
    assert get_version_entry(store, "audio_cpp", "v1")["scanned_at"] == "x"
    assert get_model_profile_entry(store, "audio_cpp", "v1", "fp")["fingerprint"] == "fp"


def test_catalog_io_does_not_need_the_store_lock(tmp_path):
    """Catalog reads/writes must not serialize on the store-wide lock.

    Holding that lock across the multi-megabyte catalog parse/dump blocked every
    other request (and the proxy) on the event loop right after an install.
    """

    class LockFreeStore:
        def __init__(self, root):
            self._config_dir = str(root)

    store = LockFreeStore(tmp_path)
    upsert_version_entry(store, "audio_cpp", "v1", {"scanned_at": "lock-free"})
    assert get_version_entry(store, "audio_cpp", "v1")["scanned_at"] == "lock-free"
    upsert_model_profile_entry(store, "audio_cpp", "v1", "fp", {"fingerprint": "fp"})
    assert get_model_profile_entry(store, "audio_cpp", "v1", "fp")["fingerprint"] == "fp"


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
