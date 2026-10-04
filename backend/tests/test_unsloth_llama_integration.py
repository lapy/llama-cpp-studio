"""Contracts for Unsloth llama.cpp prebuilt engine support."""

from pathlib import Path

import pytest

import backend.proxy.llama_swap.config as llama_swap_config
from backend.engines.registry import ENGINE_REGISTRY, inferred_engines_for_artifact_format
from backend.engines.unsloth_llama.installer import (
    CHECKSUM_ASSET_NAME,
    expected_sha256,
    linux_x64_asset_name,
    select_asset_flavor,
    select_release_asset,
    sha256_file,
    verify_sha256,
)
from backend.repo_identity import CANONICAL_REPOSITORY_URLS


def test_unsloth_registry_and_identity():
    spec = ENGINE_REGISTRY["unsloth_llama"]
    assert spec.runtime_kind == "llama_server"
    assert spec.scanner_kind == "llama_help"
    assert spec.install_kind == "prebuilt"
    assert spec.artifact_formats == frozenset({"gguf"})
    assert spec.active_path_fields == ("binary_path",)
    assert "unsloth_llama" in inferred_engines_for_artifact_format("gguf")
    assert CANONICAL_REPOSITORY_URLS["unsloth_llama"] == (
        "https://github.com/unslothai/llama.cpp.git"
    )


@pytest.mark.parametrize(
    ("cuda_major", "flavor"),
    [
        (12, "cuda12-portable"),
        (13, "cuda13-portable"),
        (None, "cpu"),
        (11, "cpu"),
    ],
)
def test_asset_flavor_from_cuda(cuda_major, flavor):
    assert select_asset_flavor(cuda_major) == flavor


def test_select_release_asset_matches_linux_x64_flavor():
    tag = "b11160-mix-a6922cc"
    assets = [
        {"name": "app-b11160-mix-a6922cc-linux-x64-rocm-gfx110X.tar.gz"},
        {"name": linux_x64_asset_name(tag, "cpu")},
        {"name": linux_x64_asset_name(tag, "cuda12-portable")},
        {"name": CHECKSUM_ASSET_NAME},
    ]
    chosen = select_release_asset(assets, tag, "cuda12-portable")
    assert chosen["name"] == "app-b11160-mix-a6922cc-linux-x64-cuda12-portable.tar.gz"
    assert select_release_asset(assets, tag, "cuda13-portable") is None


def test_expected_sha256_reads_name_map_and_file_list():
    name = linux_x64_asset_name("b1", "cpu")
    assert expected_sha256({name: "abc123"}, name) == "abc123"
    assert (
        expected_sha256(
            {"files": [{"name": name, "sha256": "def456"}]},
            name,
        )
        == "def456"
    )
    with pytest.raises(ValueError, match="No sha256"):
        expected_sha256({name: "abc123"}, "other.tar.gz")


def test_verify_sha256_refuses_mismatch(tmp_path):
    archive = tmp_path / "app.tar.gz"
    archive.write_bytes(b"unsloth-mix")
    digest = sha256_file(str(archive))
    verify_sha256(str(archive), digest)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        verify_sha256(str(archive), "0" * 64)


def test_unsloth_preview_uses_llama_server(monkeypatch, tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    model_path = tmp_path / "model.gguf"
    model_path.write_text("gguf", encoding="utf-8")
    monkeypatch.setattr(
        llama_swap_config,
        "resolve_gguf_model_path_for_quant",
        lambda hf_id, quant: str(model_path),
    )
    monkeypatch.setattr(
        llama_swap_config,
        "get_active_binary_path_for_engine",
        lambda store, engine: str(binary) if engine == "unsloth_llama" else None,
    )
    monkeypatch.setattr(
        llama_swap_config,
        "resolve_llama_server_invocation_paths",
        lambda path: (str(path), str(tmp_path)),
    )
    monkeypatch.setattr(
        llama_swap_config,
        "_resolve_cuda_library_path",
        lambda build_dir: "/fake/lib",
    )
    monkeypatch.setattr(llama_swap_config, "_active_engine_param_index", lambda engine: {})
    preview = llama_swap_config.preview_llama_swap_command_for_model(
        {
            "id": "model",
            "huggingface_id": "unsloth/gemma-gguf",
            "quantization": "Q4_K_M",
            "format": "gguf",
            "config": {"engine": "unsloth_llama", "engines": {"unsloth_llama": {}}},
        }
    )
    assert preview["ok"] is True
    assert "${studio_gguf_bin_unsloth_llama}" in preview["cmd"]
    assert "--port ${PORT}" in preview["cmd"]


def test_unsloth_routes_and_status(client, monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    response = client.get("/api/unsloth-llama/status")
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "unsloth_llama"
    assert body["installed"] is False
    response = client.get("/api/engines")
    assert response.status_code == 200
    ids = [row["id"] for row in response.json()["engines"]]
    assert "unsloth_llama" in ids
