"""Tests for HF change detection, model refresh, and companions listing."""

import asyncio

import pytest
from fastapi import BackgroundTasks, HTTPException

import backend.huggingface as hf
import backend.routes.models as models_routes
import backend.services.model_downloads as model_downloads
from backend.tests.test_models_route_bundles import FakeProgressManager, MemoryStore


def test_resolve_repo_path_alias_prefers_mtp_folder():
    repo_files = [
        "gemma-4-31B-it-Q8_0.gguf",
        "MTP/mtp-gemma-4-31B-it-Q8_0.gguf",
        "mmproj-F16.gguf",
    ]
    assert (
        hf._resolve_repo_path_alias("mtp-gemma-4-31B-it-Q8_0.gguf", repo_files)
        == "MTP/mtp-gemma-4-31B-it-Q8_0.gguf"
    )
    assert (
        hf._resolve_repo_path_alias("MTP/mtp-gemma-4-31B-it-Q8_0.gguf", repo_files)
        == "MTP/mtp-gemma-4-31B-it-Q8_0.gguf"
    )


def test_detect_hf_file_changes_resolves_mtp_folder_alias(monkeypatch, tmp_path):
    """Basename-only MTP ledger entries must match files under MTP/ on HF."""
    companion = tmp_path / "mtp-gemma-4-31B-it-Q8_0.gguf"
    companion.write_bytes(b"x" * 10)

    def fake_remote(repo_id, paths):
        out = {}
        for path in paths:
            if path in {
                "mtp-gemma-4-31B-it-Q8_0.gguf",
                "MTP/mtp-gemma-4-31B-it-Q8_0.gguf",
            }:
                out[path] = {
                    "size": 10,
                    "etag": "etag-mtp",
                    "sha256": "sha-mtp",
                    "resolved_path": "MTP/mtp-gemma-4-31B-it-Q8_0.gguf",
                }
                if path == "MTP/mtp-gemma-4-31B-it-Q8_0.gguf":
                    out[path] = {
                        "size": 10,
                        "etag": "etag-mtp",
                        "sha256": "sha-mtp",
                    }
        return out

    monkeypatch.setattr(hf, "get_remote_file_info", fake_remote)
    monkeypatch.setattr(
        hf,
        "resolve_cached_model_path",
        lambda hf_id, filename: str(companion)
        if filename
        in {
            "mtp-gemma-4-31B-it-Q8_0.gguf",
            "MTP/mtp-gemma-4-31B-it-Q8_0.gguf",
        }
        else None,
    )

    result = hf.detect_hf_file_changes(
        "unsloth/gemma-4-31B-it-GGUF",
        ["mtp-gemma-4-31B-it-Q8_0.gguf"],
        {
            "mtp-gemma-4-31B-it-Q8_0.gguf": {
                "file_size": 10,
                "etag": "etag-mtp",
                "sha256": "sha-mtp",
                "file_path": str(companion),
            }
        },
    )
    assert result["removed_remote"] == []
    assert result["changed"] == []
    assert len(result["unchanged"]) == 1
    assert result["unchanged"][0]["filename"] == "MTP/mtp-gemma-4-31B-it-Q8_0.gguf"
    assert result["unchanged"][0]["previous_filename"] == "mtp-gemma-4-31B-it-Q8_0.gguf"


def test_get_remote_file_info_aliases_mtp_basename(monkeypatch):
    class FakePi:
        def __init__(self, path, size):
            self.path = path
            self.size = size
            self.etag = f"etag-{path}"
            self.lfs = {"sha256": f"sha-{path}"}

    calls = {"paths": []}

    def fake_paths_info(repo_id=None, paths=None):
        calls["paths"].append(list(paths or []))
        out = []
        for path in paths or []:
            if path == "MTP/mtp-gemma-4-31B-it-Q8_0.gguf":
                out.append(FakePi(path, 514687104))
        return out

    monkeypatch.setattr(
        hf.hf_api, "get_paths_info", fake_paths_info
    )
    monkeypatch.setattr(
        hf.hf_api,
        "list_repo_files",
        lambda repo_id: [
            "gemma-4-31B-it-Q8_0.gguf",
            "MTP/mtp-gemma-4-31B-it-Q8_0.gguf",
        ],
    )

    info = hf.get_remote_file_info(
        "unsloth/gemma-4-31B-it-GGUF",
        ["mtp-gemma-4-31B-it-Q8_0.gguf"],
    )
    assert "mtp-gemma-4-31B-it-Q8_0.gguf" in info
    assert info["mtp-gemma-4-31B-it-Q8_0.gguf"]["resolved_path"] == (
        "MTP/mtp-gemma-4-31B-it-Q8_0.gguf"
    )
    assert info["mtp-gemma-4-31B-it-Q8_0.gguf"]["size"] == 514687104
    # First call is the basename (miss); second fetches the aliased path.
    assert calls["paths"][0] == ["mtp-gemma-4-31B-it-Q8_0.gguf"]
    assert calls["paths"][1] == ["MTP/mtp-gemma-4-31B-it-Q8_0.gguf"]


def test_collect_model_refresh_plan_reports_mtp_path_correction(monkeypatch):
    monkeypatch.setattr(
        hf,
        "_list_remote_gguf_weight_files",
        lambda hf_id, quant: ["gemma-4-31B-it-Q8_0.gguf"],
    )
    monkeypatch.setattr(
        hf,
        "detect_hf_file_changes",
        lambda hf_id, filenames, local_entries: {
            "changed": [],
            "unchanged": [
                {
                    "filename": "MTP/mtp-gemma-4-31B-it-Q8_0.gguf",
                    "previous_filename": "mtp-gemma-4-31B-it-Q8_0.gguf",
                    "resolved_path": "MTP/mtp-gemma-4-31B-it-Q8_0.gguf",
                    "size": 10,
                    "etag": "e",
                    "sha256": "s",
                    "reason": None,
                }
            ],
            "removed_remote": [],
        },
    )
    plan = hf.collect_model_refresh_plan(
        {
            "id": "unsloth--gemma",
            "huggingface_id": "unsloth/gemma-4-31B-it-GGUF",
            "format": "gguf",
            "quantization": "Q8_0",
            "mtp_filename": "mtp-gemma-4-31B-it-Q8_0.gguf",
            "files": [
                {
                    "filename": "gemma-4-31B-it-Q8_0.gguf",
                    "role": "weight",
                    "size": 1,
                }
            ],
        }
    )
    assert plan["path_corrections"] == [
        {
            "from": "mtp-gemma-4-31B-it-Q8_0.gguf",
            "to": "MTP/mtp-gemma-4-31B-it-Q8_0.gguf",
        }
    ]
    assert plan["removed_remote"] == []


def test_detect_hf_file_changes_size_and_missing(monkeypatch, tmp_path):
    local_file = tmp_path / "model-Q4_K_M.gguf"
    local_file.write_bytes(b"12345")

    monkeypatch.setattr(
        hf,
        "get_remote_file_info",
        lambda repo_id, paths: {
            p: meta
            for p, meta in {
                "model-Q4_K_M.gguf": {
                    "size": 10,
                    "etag": "etag-new",
                    "sha256": "sha-new",
                },
                "mmproj-F16.gguf": {"size": 3, "etag": "etag-mm", "sha256": "sha-mm"},
            }.items()
            if p in paths
        },
    )
    monkeypatch.setattr(hf, "resolve_cached_model_path", lambda *_a, **_k: None)

    result = hf.detect_hf_file_changes(
        "org/model",
        ["model-Q4_K_M.gguf", "mmproj-F16.gguf", "gone.gguf"],
        {
            "model-Q4_K_M.gguf": {
                "file_size": 5,
                "etag": "etag-old",
                "file_path": str(local_file),
            },
            "mmproj-F16.gguf": {
                "file_size": 3,
                "etag": "etag-mm",
                "sha256": "sha-mm",
                "file_path": str(tmp_path / "missing-mmproj.gguf"),
            },
        },
    )
    changed_names = {e["filename"] for e in result["changed"]}
    assert "model-Q4_K_M.gguf" in changed_names
    assert "mmproj-F16.gguf" in changed_names  # missing local
    assert any(e["filename"] == "gone.gguf" for e in result["removed_remote"])


def test_detect_hf_file_changes_unchanged_when_matching(monkeypatch, tmp_path):
    local_file = tmp_path / "weights.gguf"
    local_file.write_bytes(b"abc")
    monkeypatch.setattr(
        hf,
        "get_remote_file_info",
        lambda repo_id, paths: {
            "weights.gguf": {"size": 3, "etag": "etag-1", "sha256": "sha-1"},
        },
    )
    monkeypatch.setattr(
        hf, "resolve_cached_model_path", lambda *_a, **_k: str(local_file)
    )
    result = hf.detect_hf_file_changes(
        "org/model",
        ["weights.gguf"],
        {
            "weights.gguf": {
                "file_size": 3,
                "etag": "etag-1",
                "sha256": "sha-1",
                "file_path": str(local_file),
            }
        },
    )
    assert result["changed"] == []
    assert len(result["unchanged"]) == 1


def test_download_model_passes_force_download(monkeypatch):
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        path = "/tmp/forced.gguf"
        return path

    monkeypatch.setattr(hf, "hf_hub_download", fake_download)
    monkeypatch.setattr(hf, "delete_cached_model_file", lambda *a, **k: True)
    monkeypatch.setattr(hf.os.path, "getsize", lambda _p: 42)

    path, size = asyncio.run(
        hf.download_model("org/model", "model.gguf", force_download=True)
    )
    assert path == "/tmp/forced.gguf"
    assert size == 42
    assert calls["force_download"] is True


def test_refresh_model_route_up_to_date(monkeypatch):
    store = MemoryStore(
        [
            {
                "id": "org--model--Q4_K_M",
                "huggingface_id": "org/model",
                "format": "gguf",
                "quantization": "Q4_K_M",
            }
        ]
    )
    monkeypatch.setattr(models_routes, "get_store", lambda: store)
    monkeypatch.setattr(
        models_routes,
        "collect_model_refresh_plan",
        lambda model: {
            "filenames": ["model-Q4_K_M.gguf"],
            "changed": [],
            "unchanged": [{"filename": "model-Q4_K_M.gguf"}],
            "removed_remote": [],
            "path_corrections": [],
        },
    )
    result = asyncio.run(
        models_routes.refresh_model("org--model--Q4_K_M", BackgroundTasks())
    )
    assert result["updated"] is False
    assert "task_id" not in result


def test_refresh_model_route_heals_mtp_basename_path(monkeypatch):
    store = MemoryStore(
        [
            {
                "id": "org--model--Q8_0",
                "huggingface_id": "org/model",
                "format": "gguf",
                "quantization": "Q8_0",
                "mtp_filename": "mtp-gemma-4-31B-it-Q8_0.gguf",
                "files": [
                    {
                        "filename": "mtp-gemma-4-31B-it-Q8_0.gguf",
                        "role": "mtp",
                        "size": 10,
                    }
                ],
            }
        ]
    )
    monkeypatch.setattr(models_routes, "get_store", lambda: store)
    monkeypatch.setattr(
        models_routes,
        "collect_model_refresh_plan",
        lambda model: {
            "filenames": ["mtp-gemma-4-31B-it-Q8_0.gguf"],
            "changed": [],
            "unchanged": [
                {
                    "filename": "MTP/mtp-gemma-4-31B-it-Q8_0.gguf",
                    "size": 10,
                    "etag": "e",
                    "sha256": "s",
                }
            ],
            "removed_remote": [],
            "path_corrections": [
                {
                    "from": "mtp-gemma-4-31B-it-Q8_0.gguf",
                    "to": "MTP/mtp-gemma-4-31B-it-Q8_0.gguf",
                }
            ],
        },
    )
    result = asyncio.run(
        models_routes.refresh_model("org--model--Q8_0", BackgroundTasks())
    )
    assert result["updated"] is False
    assert result["removed_remote"] == []
    assert store.rows["org--model--Q8_0"]["mtp_filename"] == (
        "MTP/mtp-gemma-4-31B-it-Q8_0.gguf"
    )
    names = {f["filename"] for f in store.rows["org--model--Q8_0"]["files"]}
    assert "MTP/mtp-gemma-4-31B-it-Q8_0.gguf" in names
    assert "mtp-gemma-4-31B-it-Q8_0.gguf" not in names


def test_refresh_model_route_schedules_changed_files(monkeypatch):
    store = MemoryStore(
        [
            {
                "id": "org--model--Q4_K_M",
                "huggingface_id": "org/model",
                "format": "gguf",
                "quantization": "Q4_K_M",
                "mmproj_filename": "mmproj-F16.gguf",
            }
        ]
    )
    pm = FakeProgressManager()
    background = BackgroundTasks()
    scheduled = {}

    monkeypatch.setattr(models_routes, "get_store", lambda: store)
    monkeypatch.setattr(models_routes, "get_progress_manager", lambda: pm)
    monkeypatch.setattr(models_routes.time, "time", lambda: 1000.5)
    monkeypatch.setattr(
        models_routes,
        "collect_model_refresh_plan",
        lambda model: {
            "filenames": ["model-Q4_K_M.gguf", "mmproj-F16.gguf"],
            "changed": [
                {
                    "filename": "model-Q4_K_M.gguf",
                    "size": 10,
                    "etag": "e1",
                    "reason": "size_mismatch",
                },
                {
                    "filename": "mmproj-F16.gguf",
                    "size": 2,
                    "etag": "e2",
                    "reason": "etag_mismatch",
                },
            ],
            "unchanged": [],
            "removed_remote": [],
        },
    )

    async def fake_register(**kwargs):
        scheduled["register"] = kwargs

    monkeypatch.setattr(models_routes, "register_model_refresh_download", fake_register)

    original = dict(model_downloads.active_downloads)
    model_downloads.active_downloads.clear()
    try:
        result = asyncio.run(
            models_routes.refresh_model("org--model--Q4_K_M", background)
        )
        assert result["updated"] is True
        assert result["task_id"].startswith("refresh_")
        assert len(result["files"]) == 2
        assert len(background.tasks) == 1
        assert scheduled["register"]["model_id"] == "org--model--Q4_K_M"
    finally:
        model_downloads.active_downloads.clear()
        model_downloads.active_downloads.update(original)


def test_get_model_companions_route(monkeypatch):
    store = MemoryStore(
        [
            {
                "id": "org--model--Q4_K_M",
                "huggingface_id": "org/model",
                "format": "gguf",
                "mmproj_filename": "mmproj-F16.gguf",
            }
        ]
    )
    monkeypatch.setattr(models_routes, "get_store", lambda: store)
    monkeypatch.setattr(
        models_routes,
        "list_repo_companion_files",
        lambda hf_id: {
            "mmproj_files": [{"filename": "mmproj-F16.gguf", "size": 1}],
            "mtp_files": [{"filename": "MTP/draft.gguf", "size": 2, "label": "Q8_0"}],
            "dflash_files": [],
        },
    )
    result = asyncio.run(models_routes.get_model_companions("org--model--Q4_K_M"))
    assert result["current"]["mmproj_filename"] == "mmproj-F16.gguf"
    assert len(result["mmproj_files"]) == 1
    assert len(result["mtp_files"]) == 1


def test_get_model_companions_rejects_non_gguf(monkeypatch):
    store = MemoryStore(
        [{"id": "st", "huggingface_id": "org/st", "format": "safetensors"}]
    )
    monkeypatch.setattr(models_routes, "get_store", lambda: store)
    with pytest.raises(HTTPException, match="only supported for GGUF"):
        asyncio.run(models_routes.get_model_companions("st"))


def test_list_repo_companion_files_classifies(monkeypatch):
    monkeypatch.setattr(
        hf.hf_api,
        "list_repo_files",
        lambda repo_id: [
            "model-Q4_K_M.gguf",
            "mmproj-F16.gguf",
            "MTP/mtp-Q8_0.gguf",
            "draft-DFlash-BF16.gguf",
        ],
    )
    monkeypatch.setattr(
        hf,
        "get_accurate_file_sizes",
        lambda repo_id, paths: {p: 10 for p in paths},
    )
    result = hf.list_repo_companion_files("org/model")
    assert [f["filename"] for f in result["mmproj_files"]] == ["mmproj-F16.gguf"]
    assert result["mtp_files"][0]["filename"] == "MTP/mtp-Q8_0.gguf"
    assert result["dflash_files"][0]["filename"] == "draft-DFlash-BF16.gguf"


def test_detect_hf_file_changes_companion_size_mismatch(monkeypatch, tmp_path):
    companion = tmp_path / "mmproj-F16.gguf"
    companion.write_bytes(b"old-mmproj")
    monkeypatch.setattr(
        hf,
        "get_remote_file_info",
        lambda repo_id, paths: {
            "mmproj-F16.gguf": {
                "size": 99,
                "etag": "etag-mm-new",
                "sha256": "sha-mm-new",
            }
        },
    )
    result = hf.detect_hf_file_changes(
        "org/model",
        ["mmproj-F16.gguf"],
        {
            "mmproj-F16.gguf": {
                "file_size": len(b"old-mmproj"),
                "etag": "etag-mm-old",
                "sha256": "sha-mm-old",
                "file_path": str(companion),
            }
        },
    )
    assert len(result["changed"]) == 1
    assert result["changed"][0]["filename"] == "mmproj-F16.gguf"
    assert result["changed"][0]["reason"] in {
        "size_mismatch",
        "etag_mismatch",
        "sha_mismatch",
    }


def test_collect_model_refresh_plan_includes_weights_and_companions(
    monkeypatch, tmp_path
):
    weight = tmp_path / "model-Q4_K_M.gguf"
    weight.write_bytes(b"weights")
    companion = tmp_path / "mmproj-F16.gguf"
    companion.write_bytes(b"proj")

    monkeypatch.setattr(
        hf,
        "_list_remote_gguf_weight_files",
        lambda hf_id, quant: ["model-Q4_K_M.gguf", "model-Q4_K_M-00001-of-00002.gguf"],
    )
    monkeypatch.setattr(
        hf,
        "resolve_cached_model_path",
        lambda hf_id, filename: str(companion) if "mmproj" in filename else None,
    )
    monkeypatch.setattr(
        hf,
        "detect_hf_file_changes",
        lambda hf_id, filenames, local_entries: {
            "changed": [
                {
                    "filename": "model-Q4_K_M-00001-of-00002.gguf",
                    "reason": "missing_local",
                }
            ],
            "unchanged": [{"filename": "model-Q4_K_M.gguf"}],
            "removed_remote": [],
        },
    )

    plan = hf.collect_model_refresh_plan(
        {
            "id": "org--model--Q4_K_M",
            "huggingface_id": "org/model",
            "format": "gguf",
            "quantization": "Q4_K_M",
            "mmproj_filename": "mmproj-F16.gguf",
            "mtp_filename": None,
            "dflash_filename": None,
            "files": [
                {
                    "filename": "model-Q4_K_M.gguf",
                    "role": "weight",
                    "size": 7,
                    "etag": "etag-w",
                    "sha256": "sha-w",
                }
            ],
        }
    )
    assert "model-Q4_K_M.gguf" in plan["filenames"]
    assert "model-Q4_K_M-00001-of-00002.gguf" in plan["filenames"]
    assert "mmproj-F16.gguf" in plan["filenames"]
    assert plan["companion_filenames"] == ["mmproj-F16.gguf"]
    assert plan["changed"][0]["filename"] == "model-Q4_K_M-00001-of-00002.gguf"


def test_collect_model_refresh_plan_safetensors_unions_remote(monkeypatch):
    monkeypatch.setattr(
        hf,
        "_list_remote_safetensors_files",
        lambda hf_id: [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
    )
    monkeypatch.setattr(
        hf,
        "detect_hf_file_changes",
        lambda hf_id, filenames, local_entries: {
            "changed": [
                {
                    "filename": "model-00002-of-00002.safetensors",
                    "reason": "missing_local",
                }
            ],
            "unchanged": [{"filename": "model-00001-of-00002.safetensors"}],
            "removed_remote": [],
        },
    )
    plan = hf.collect_model_refresh_plan(
        {
            "id": "org--st",
            "huggingface_id": "org/st",
            "format": "safetensors",
            "files": [
                {
                    "filename": "model-00001-of-00002.safetensors",
                    "role": "shard",
                    "size": 10,
                    "etag": "e1",
                }
            ],
        }
    )
    assert "model-00002-of-00002.safetensors" in plan["filenames"]
    assert plan["companion_filenames"] == []


def test_download_model_with_progress_passes_force_download(monkeypatch):
    calls = {"deleted": [], "hub": {}}

    class FakePm:
        async def send_download_progress(self, **kwargs):
            return None

    def fake_hub(**kwargs):
        calls["hub"] = kwargs
        return "/tmp/forced-progress.gguf"

    monkeypatch.setattr(
        hf,
        "delete_cached_model_file",
        lambda hf_id, filename: calls["deleted"].append((hf_id, filename)) or True,
    )
    monkeypatch.setattr(hf, "hf_hub_download", fake_hub)
    monkeypatch.setattr(hf.os.path, "getsize", lambda _p: 11)
    monkeypatch.setattr(hf.os.path, "realpath", lambda p: p)
    monkeypatch.setattr(hf.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(
        hf,
        "HfApi",
        lambda: type("A", (), {"repo_file_info": lambda *a, **k: None})(),
    )

    # Avoid importing HF_HUB_CACHE path side effects for incomplete blobs.
    import huggingface_hub.constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", "/tmp/hf-hub-cache-test")

    path, size = asyncio.run(
        hf.download_model_with_progress(
            "org/model",
            "model.gguf",
            FakePm(),
            "task-force",
            total_bytes=11,
            force_download=True,
        )
    )
    assert path == "/tmp/forced-progress.gguf"
    assert size == 11
    assert calls["deleted"] == [("org/model", "model.gguf")]
    assert calls["hub"]["force_download"] is True


def test_refresh_model_task_force_downloads_and_writes_store_file_metadata(
    monkeypatch, tmp_path
):
    model_id = "org--model--Q4_K_M"
    store = MemoryStore(
        [
            {
                "id": model_id,
                "huggingface_id": "org/model",
                "format": "gguf",
                "quantization": "Q4_K_M",
                "mmproj_filename": "mmproj-F16.gguf",
            }
        ]
    )
    pm = FakeProgressManager()
    observed = {"downloads": [], "stale": 0}

    async def fake_download(
        hf_id,
        filename,
        proxy,
        task_id,
        size_hint,
        model_format,
        event_hf_id,
        force_download=False,
        **kwargs,
    ):
        observed["downloads"].append(
            {
                "filename": filename,
                "force_download": force_download,
                "size_hint": size_hint,
            }
        )
        return (f"/tmp/{filename}", size_hint or 5)

    monkeypatch.setattr(model_downloads, "get_store", lambda: store)
    monkeypatch.setattr(model_downloads, "download_model_with_progress", fake_download)
    monkeypatch.setattr(
        model_downloads,
        "mark_llama_swap_stale_after_download",
        lambda: observed.__setitem__("stale", observed["stale"] + 1),
    )

    files = [
        {
            "filename": "model-Q4_K_M.gguf",
            "size": 20,
            "etag": "etag-w",
            "sha256": "sha-w",
        },
        {
            "filename": "mmproj-F16.gguf",
            "size": 4,
            "etag": "etag-mm",
            "sha256": "sha-mm",
        },
    ]
    original = dict(model_downloads.active_downloads)
    model_downloads.active_downloads.clear()
    model_downloads.active_downloads["refresh-task"] = {
        "model_id": model_id,
        "model_format": "model-refresh",
    }
    try:
        asyncio.run(
            model_downloads.refresh_model_task(
                model_id,
                files,
                pm,
                "refresh-task",
            )
        )
    finally:
        model_downloads.active_downloads.clear()
        model_downloads.active_downloads.update(original)

    assert [d["filename"] for d in observed["downloads"]] == [
        "model-Q4_K_M.gguf",
        "mmproj-F16.gguf",
    ]
    assert all(d["force_download"] is True for d in observed["downloads"])
    by_name = {entry["filename"]: entry for entry in store.rows[model_id]["files"]}
    assert by_name["model-Q4_K_M.gguf"]["etag"] == "etag-w"
    assert by_name["model-Q4_K_M.gguf"]["sha256"] == "sha-w"
    assert by_name["mmproj-F16.gguf"]["etag"] == "etag-mm"
    assert pm.completed == [("refresh-task", "Model refresh complete")]
    assert pm.broadcasts[-1]["model_format"] == "model-refresh"
    assert pm.broadcasts[-1]["status"] == "completed"
    assert observed["stale"] == 1
    # Companion attachment fields must remain unchanged.
    assert store.rows[model_id]["mmproj_filename"] == "mmproj-F16.gguf"


