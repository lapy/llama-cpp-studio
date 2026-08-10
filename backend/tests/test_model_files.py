from backend.model_files import (
    infer_file_role,
    iter_model_files,
    normalize_model_files,
    shard_sort_key,
    upsert_model_file,
)


class Store:
    def __init__(self, model):
        self.model = model

    def get_model(self, model_id):
        return self.model if self.model.get("id") == model_id else None

    def update_model(self, model_id, updates):
        if self.model.get("id") != model_id:
            return None
        self.model.update(updates)
        return self.model


def test_file_roles_and_shard_sorting():
    assert infer_file_role("mmproj-F16.gguf") == "mmproj"
    assert infer_file_role("model-MTP-BF16.gguf") == "mtp"
    assert infer_file_role("MTP/mtp-model-Q8_0.gguf") == "mtp"
    assert infer_file_role("model-DFlash-BF16.gguf") == "dflash"
    entries = normalize_model_files(
        [
            {"filename": "model-00002-of-00002.gguf"},
            {"filename": "model-00001-of-00002.gguf"},
        ]
    )
    assert [row["role"] for row in entries] == ["shard", "shard"]
    assert [row["filename"] for row in sorted(entries, key=shard_sort_key)] == [
        "model-00001-of-00002.gguf",
        "model-00002-of-00002.gguf",
    ]


def test_upsert_preserves_omitted_remote_identity():
    store = Store(
        {
            "id": "org--model--Q4_K_M",
            "files": [
                {
                    "filename": "model-Q4_K_M.gguf",
                    "role": "weight",
                    "size": 100,
                    "etag": "old-etag",
                    "sha256": "old-sha",
                    "downloaded_at": "2025-01-01T00:00:00Z",
                }
            ],
        }
    )
    upsert_model_file(
        store,
        store.model["id"],
        {"filename": "model-Q4_K_M.gguf", "size": 200},
    )
    entry = next(iter_model_files(store.model))
    assert entry["size"] == 200
    assert entry["etag"] == "old-etag"
    assert entry["sha256"] == "old-sha"
    assert entry["downloaded_at"] == "2025-01-01T00:00:00Z"
