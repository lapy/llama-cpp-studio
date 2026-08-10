import json

import yaml

from backend.data_store import DataStore
from backend.migrations.migrate_hf_manifests_to_files import (
    MIGRATION_MARKER,
    MIGRATION_VERSION,
    cleanup_legacy_sidecars,
    migrate_document,
)


def _write_manifest(tmp_path, model_format, repo_dir, payload):
    directory = tmp_path / "models" / model_format / repo_dir
    directory.mkdir(parents=True)
    path = directory / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_migrates_gguf_and_safetensors_then_cleans_sidecars(tmp_path):
    gguf_path = _write_manifest(
        tmp_path,
        "gguf",
        "org_gguf",
        [
            {
                "huggingface_id": "org/gguf",
                "model_id": "org--gguf--Q4_K_M",
                "filename": "model-Q4_K_M.gguf",
                "file_size": 123,
                "etag": "etag-g",
                "sha256": "sha-g",
                "max_context_length": 8192,
                "gguf_layer_info": {"layer_count": 33},
                "metadata": {"tokenizer": {"large": "discard"}},
            }
        ],
    )
    st_path = _write_manifest(
        tmp_path,
        "safetensors",
        "org_st",
        {
            "huggingface_id": "org/st",
            "max_context_length": 4096,
            "metadata": {"layer_count": 25, "config": {"large": "discard"}},
            "files": [
                {
                    "filename": "model-00001-of-00002.safetensors",
                    "file_size": 456,
                    "etag": "etag-st",
                }
            ],
        },
    )
    document = {
        "schema_version": 2,
        "models": [
            {
                "id": "org--gguf--Q4_K_M",
                "huggingface_id": "org/gguf",
                "format": "gguf",
                "quantization": "Q4_K_M",
            },
            {
                "id": "org--st",
                "huggingface_id": "org/st",
                "format": "safetensors",
            },
        ],
    }

    migrated, changed, cleanup = migrate_document(document, str(tmp_path))
    assert changed is True
    assert migrated[MIGRATION_MARKER] == MIGRATION_VERSION
    gguf, st = migrated["models"]
    assert gguf["files"] == [
        {
            "filename": "model-Q4_K_M.gguf",
            "role": "weight",
            "size": 123,
            "etag": "etag-g",
            "sha256": "sha-g",
        }
    ]
    assert gguf["max_context_length"] == 8192
    assert gguf["layer_count"] == 33
    assert st["files"][0]["role"] == "shard"
    assert st["files"][0]["size"] == 456
    assert st["max_context_length"] == 4096
    assert st["layer_count"] == 25
    assert "metadata" not in gguf
    assert "metadata" not in st

    cleanup_legacy_sidecars(cleanup, str(tmp_path))
    assert not gguf_path.exists()
    assert not st_path.exists()
    assert not (tmp_path / "models" / "gguf").exists()
    assert not (tmp_path / "models" / "safetensors").exists()

    again, changed_again, cleanup_again = migrate_document(migrated, str(tmp_path))
    assert again == migrated
    assert changed_again is False
    assert cleanup_again == []


def test_does_not_cleanup_until_every_hf_model_has_files(tmp_path):
    path = _write_manifest(
        tmp_path,
        "gguf",
        "org_known",
        [
            {
                "model_id": "org--known--Q4_K_M",
                "filename": "known-Q4_K_M.gguf",
                "file_size": 1,
            }
        ],
    )
    document = {
        "models": [
            {
                "id": "org--known--Q4_K_M",
                "huggingface_id": "org/known",
                "format": "gguf",
                "quantization": "Q4_K_M",
            },
            {
                "id": "org--missing--Q8_0",
                "huggingface_id": "org/missing",
                "format": "gguf",
                "quantization": "Q8_0",
            },
        ]
    }
    migrated, changed, cleanup = migrate_document(document, str(tmp_path))
    assert changed is True
    assert MIGRATION_MARKER not in migrated
    assert cleanup == []
    assert path.exists()


def test_data_store_runs_migration_automatically_on_startup(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "models": [
                    {
                        "id": "org--model--Q4_K_M",
                        "huggingface_id": "org/model",
                        "format": "gguf",
                        "quantization": "Q4_K_M",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path,
        "gguf",
        "org_model",
        [
            {
                "model_id": "org--model--Q4_K_M",
                "filename": "model-Q4_K_M.gguf",
                "file_size": 99,
            }
        ],
    )

    store = DataStore(str(config_dir))
    model = store.get_model("org--model--Q4_K_M")
    assert model["files"][0]["filename"] == "model-Q4_K_M.gguf"
    assert not manifest.exists()
