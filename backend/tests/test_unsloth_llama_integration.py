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
from backend.engines.unsloth_llama.prebuilt import (
    HostTarget,
    host_target_from_gpu_info,
    is_behind,
    parse_sm,
    resolve_release_plan,
    select_linux_x64_artifact,
)
from backend.repo_identity import CANONICAL_REPOSITORY_URLS

_MIX_TAG = "b11160-mix-a6922cc"


def _artifact(name, *, profile, runtime="", coverage="", sms=(), min_sm=None, max_sm=None, rank=0, kind="linux-cuda"):
    return {
        "asset_name": name,
        "bundle_profile": profile,
        "runtime_line": runtime,
        "coverage_class": coverage,
        "supported_sms": [str(sm) for sm in sms],
        "min_sm": min_sm,
        "max_sm": max_sm,
        "rank": rank,
        "install_kind": kind,
    }


def _cuda12_manifest(tag=_MIX_TAG):
    return {
        "schema_version": 1,
        "upstream_tag": "b11160",
        "release_tag": tag,
        "artifacts": [
            _artifact(
                linux_x64_asset_name(tag, "cuda12-legacy"),
                profile="cuda12-legacy",
                runtime="cuda12",
                coverage="legacy",
                sms=(50, 61),
                min_sm=50,
                max_sm=61,
                rank=5,
            ),
            _artifact(
                linux_x64_asset_name(tag, "cuda12-newer"),
                profile="cuda12-newer",
                runtime="cuda12",
                coverage="newer",
                sms=(86, 89, 90, 100, 120),
                min_sm=86,
                max_sm=120,
                rank=20,
            ),
            _artifact(
                linux_x64_asset_name(tag, "cuda12-older"),
                profile="cuda12-older",
                runtime="cuda12",
                coverage="older",
                sms=(70, 75, 80, 86, 89),
                min_sm=70,
                max_sm=89,
                rank=10,
            ),
            _artifact(
                linux_x64_asset_name(tag, "cuda12-portable"),
                profile="cuda12-portable",
                runtime="cuda12",
                coverage="portable",
                sms=(70, 75, 80, 86, 89, 90, 100, 120),
                min_sm=70,
                max_sm=120,
                rank=30,
            ),
            _artifact(
                linux_x64_asset_name(tag, "cuda13-portable"),
                profile="cuda13-portable",
                runtime="cuda13",
                coverage="portable",
                sms=(75, 80, 86, 89, 90, 100, 120),
                min_sm=75,
                max_sm=120,
                rank=60,
            ),
            _artifact(
                linux_x64_asset_name(tag, "cpu"),
                profile="linux-cpu-x64",
                kind="linux-cpu",
                rank=1000,
            ),
        ],
    }


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("8.6", 86),
        ("8.9", 89),
        ("12.0", 120),
        ("sm_90", 90),
        (89, 89),
        (None, None),
    ],
)
def test_parse_sm(value, expected):
    assert parse_sm(value) == expected


def test_host_target_reads_nvidia_compute_caps():
    host = host_target_from_gpu_info(
        {
            "vendor": "nvidia",
            "device_count": 1,
            "cuda_version": "12.8",
            "gpus": [{"compute_capability": "8.6"}],
        }
    )
    assert host == HostTarget(cuda_major=12, sms=(86,), has_nvidia=True)


@pytest.mark.parametrize(
    ("sms", "cuda_major", "flavor"),
    [
        ((86,), 12, "cuda12-older"),
        ((89,), 12, "cuda12-older"),
        ((120,), 12, "cuda12-newer"),
        ((61,), 13, "cuda12-legacy"),
        ((), 12, "cuda12-portable"),
        ((), None, "linux-cpu-x64"),
    ],
)
def test_manifest_picks_tightest_linux_x64_bundle(sms, cuda_major, flavor):
    host = HostTarget(cuda_major=cuda_major, sms=sms, has_nvidia=bool(sms) or cuda_major in {12, 13})
    picked = select_linux_x64_artifact(_cuda12_manifest(), host)
    assert picked is not None
    artifact, _reason = picked
    assert artifact.bundle_profile == flavor


def test_manifest_refuses_uncovered_nvidia_host():
    host = HostTarget(cuda_major=12, sms=(52,), has_nvidia=True)
    assert select_linux_x64_artifact(_cuda12_manifest(), host) is None


@pytest.mark.parametrize(
    ("installed", "latest", "behind"),
    [
        ("b11160-mix-a6922cc", "b11160-mix-a6922cc", False),
        ("b11160-mix-a6922cc", "b11160", False),
        ("b11160", "b11160-mix-a6922cc", True),
        ("b11160-mix-aaaaaaa", "b11160-mix-bbbbbbb", True),
        ("b11100-mix-aaaaaaa", "b11160-mix-a6922cc", True),
        ("b11160-mix-a6922cc", "b11000-mix-deadbee", False),
        (None, "b11160-mix-a6922cc", False),
    ],
)
def test_mix_freshness(installed, latest, behind):
    assert is_behind(installed, latest) is behind


def test_walk_back_skips_release_without_host_bundle():
    newest = "b11180-mix-ffffff"
    older = _MIX_TAG
    releases = [
        {
            "tag_name": newest,
            "published_at": "2026-10-01T00:00:00Z",
            "assets": [
                {"name": linux_x64_asset_name(newest, "cuda13-portable")},
                {"name": CHECKSUM_ASSET_NAME},
                {"name": "llama-prebuilt-manifest.json"},
            ],
        },
        {
            "tag_name": older,
            "published_at": "2026-09-25T00:00:00Z",
            "assets": [
                {"name": linux_x64_asset_name(older, flavor)}
                for flavor in ("cuda12-older", "cuda12-portable", "cpu")
            ]
            + [{"name": CHECKSUM_ASSET_NAME}, {"name": "llama-prebuilt-manifest.json"}],
        },
    ]
    newest_manifest = {
        "upstream_tag": "b11180",
        "artifacts": [
            _artifact(
                linux_x64_asset_name(newest, "cuda13-portable"),
                profile="cuda13-portable",
                runtime="cuda13",
                coverage="portable",
                sms=(90, 120),
                min_sm=90,
                max_sm=120,
                rank=60,
            )
        ],
    }
    plan = resolve_release_plan(
        releases,
        {newest: newest_manifest, older: _cuda12_manifest(older)},
        HostTarget(cuda_major=12, sms=(86,), has_nvidia=True),
    )
    assert plan.tag == older
    assert plan.flavor == "cuda12-older"
    assert plan.skipped_newer == (newest,)
    assert plan.upstream_tag == "b11160"


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


def test_expected_sha256_reads_unsloth_artifacts_object():
    name = "app-b11160-mix-a6922cc-linux-x64-cuda12-portable.tar.gz"
    manifest = {
        "schema_version": 1,
        "component": "llama.cpp",
        "release_tag": "b11160-mix-a6922cc",
        "artifacts": {
            name: {
                "kind": "linux-cuda-app",
                "sha256": "8bceb71bb24c49bff9affb96b91b56ef6786c60d7724fac34879d73ec11433e3",
            }
        },
    }
    assert expected_sha256(manifest, name) == (
        "8bceb71bb24c49bff9affb96b91b56ef6786c60d7724fac34879d73ec11433e3"
    )


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


def test_unsloth_check_updates_uses_mix_freshness(client, monkeypatch, tmp_path):
    from backend import data_store
    from backend.engines.unsloth_llama.installer import UnslothLlamaInstaller

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    store.add_engine_version(
        "unsloth_llama",
        {
            "version": "b11160-mix-a6922cc",
            "source_ref": "b11160-mix-a6922cc",
            "type": "release",
        },
    )
    store.set_active_engine_version("unsloth_llama", "b11160-mix-a6922cc")

    async def latest(_self):
        return "b11160"

    monkeypatch.setattr(UnslothLlamaInstaller, "latest_published_tag", latest)
    response = client.get("/api/unsloth-llama/check-updates")
    assert response.status_code == 200
    body = response.json()
    assert body["latest_version"] == "b11160"
    assert body["current_version"] == "b11160-mix-a6922cc"
    assert body["update_available"] is False
