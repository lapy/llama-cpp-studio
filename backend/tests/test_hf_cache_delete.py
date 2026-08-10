"""Tests for Hugging Face hub cache deletion helpers."""

import os

import backend.huggingface as hf


def _make_hub_layout(tmp_path, huggingface_id: str, filename: str, content: bytes):
    repo_dir = tmp_path / "hub" / hf._hf_repo_folder_name(huggingface_id)
    blobs_dir = repo_dir / "blobs"
    snap_dir = repo_dir / "snapshots" / "rev123"
    blobs_dir.mkdir(parents=True)
    snap_dir.mkdir(parents=True)

    blob_name = "deadbeef"
    blob_path = blobs_dir / blob_name
    blob_path.write_bytes(content)

    # Nested filenames need parent dirs under the snapshot.
    snapshot_file = snap_dir / filename
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    snapshot_file.symlink_to(os.path.relpath(blob_path, start=snapshot_file.parent))
    return repo_dir, snapshot_file, blob_path


def test_delete_cached_model_file_removes_symlink_and_blob(tmp_path, monkeypatch):
    huggingface_id = "org/model"
    filename = "model-Q4_K_M.gguf"
    repo_dir, snapshot_file, blob_path = _make_hub_layout(
        tmp_path, huggingface_id, filename, b"weights"
    )
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(
        hf,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no hub")),
    )

    assert snapshot_file.exists()
    assert blob_path.exists()
    assert hf.delete_cached_model_file(huggingface_id, filename) is True
    assert not snapshot_file.exists()
    assert not blob_path.exists()
    assert repo_dir.exists()  # repo folder remains until full purge


def test_delete_cached_model_file_uses_explicit_file_path_fallback(
    tmp_path, monkeypatch
):
    huggingface_id = "org/model"
    filename = "weights.safetensors"
    _repo_dir, snapshot_file, blob_path = _make_hub_layout(
        tmp_path, huggingface_id, filename, b"st-weights"
    )
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(
        hf,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no hub")),
    )

    assert (
        hf.delete_cached_model_file(
            huggingface_id,
            "wrong-name.safetensors",
            file_path=str(snapshot_file),
        )
        is True
    )
    assert not os.path.lexists(snapshot_file)
    assert not blob_path.exists()


def test_purge_hf_repo_cache_removes_repo_directory(tmp_path, monkeypatch):
    huggingface_id = "org/model"
    repo_dir, _snapshot_file, _blob_path = _make_hub_layout(
        tmp_path, huggingface_id, "a.gguf", b"x"
    )
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    assert repo_dir.is_dir()
    assert hf.purge_hf_repo_cache(huggingface_id) is True
    assert not repo_dir.exists()
