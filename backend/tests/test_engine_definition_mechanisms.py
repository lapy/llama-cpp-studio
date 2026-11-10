"""Reusable mechanism contracts: transform assets, fork builds, release assets."""

from __future__ import annotations

import pytest

from backend.engines.definitions.transforms import (
    TransformError,
    apply_transform,
    load_transform,
)
from backend.engines.python.fork_build import ForkBuildError, prepare_fork_build
from backend.engines.python.release_assets import ReleaseAssetError, select_wheel_urls


def _checkout(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True)
    return tmp_path


def test_transform_asset_is_strict_about_declared_fields(tmp_path):
    asset = load_transform("sglang_v100/pyproject")
    assert asset.name == "sglang_v100/pyproject"
    assert asset.prepend_marker is True

    bad = tmp_path / "assets" / "demo" / "broken.yaml"
    bad.parent.mkdir(parents=True)
    bad.write_text(
        "schema_version: 1\nname: demo/broken\nfile: f.txt\nextra: true\noperations: []\n",
        encoding="utf-8",
    )
    with pytest.raises(TransformError, match="unknown fields"):
        load_transform("demo/broken", assets_dir=str(tmp_path / "assets"))


def test_transform_applies_only_where_the_upstream_text_matches(tmp_path):
    checkout = _checkout(tmp_path)
    (checkout / "scripts" / "install_v100.sh").write_text("echo changed\n", encoding="utf-8")
    with pytest.raises(TransformError, match="region markers are missing"):
        apply_transform("sglang_v100/install_v100_studio", str(checkout))

    # Optional assets are skipped, not failed, when the file is absent.
    assert apply_transform("sglang_v100/marlin_builder", str(checkout)) is None


def test_transform_is_idempotent_via_marker(tmp_path):
    checkout = _checkout(tmp_path)
    (checkout / "python").mkdir()
    asset = load_transform("sglang_v100/pyproject")
    (checkout / "python" / "pyproject.toml").write_text(
        "[project]\ndependencies = [\n"
        + "".join("    " + item["search"] + "\n" for item in asset.operations[0]["replacements"])
        + "]\n",
        encoding="utf-8",
    )
    apply_transform("sglang_v100/pyproject", str(checkout))
    once = (checkout / "python" / "pyproject.toml").read_text(encoding="utf-8")
    apply_transform("sglang_v100/pyproject", str(checkout))
    assert (checkout / "python" / "pyproject.toml").read_text(encoding="utf-8") == once


@pytest.mark.asyncio
async def test_prepare_fork_rejects_undeclared_options(tmp_path):
    checkout = _checkout(tmp_path)
    with pytest.raises(ForkBuildError, match="unknown options"):
        await prepare_fork_build(object(), {"magic": True}, checkout=str(checkout), base_env={})


def test_release_asset_groups_are_ordered_and_require_their_wheels():
    release = {
        "tag_name": "v1",
        "assets": [
            {"name": "flash_attn_v100-1.whl", "browser_download_url": "u-flash"},
            {"name": "vllm-1.whl", "browser_download_url": "u-vllm"},
        ],
    }
    tag, urls = select_wheel_urls(
        release,
        [
            {"prefixes": ["1cat_vllm"], "required_prefixes": ["1cat_vllm"]},
            {"prefixes": ["flash_attn_v100", "vllm"], "required_prefixes": ["vllm"]},
        ],
        error="no wheels",
    )
    assert (tag, urls) == ("v1", ["u-flash", "u-vllm"])

    with pytest.raises(ReleaseAssetError, match="no wheels"):
        select_wheel_urls(
            {"tag_name": "v1", "assets": [{"name": "flash_attn_v100-1.whl"}]},
            [{"prefixes": ["flash_attn_v100", "vllm"], "required_prefixes": ["vllm"]}],
            error="no wheels",
        )


def test_definition_load_rejects_missing_transform_asset():
    from types import SimpleNamespace

    from backend.engines.definitions.loader import _validate_transform_assets
    from backend.engines.definitions.schema import DefinitionError

    definition = SimpleNamespace(
        source_path="example.yaml",
        recipes=(
            (
                "source",
                (
                    {
                        "id": "patch",
                        "uses": "vcs.apply_transform",
                        "with": {"transform": "missing/asset"},
                    },
                ),
            ),
        ),
    )
    with pytest.raises(DefinitionError, match="transform asset not found"):
        _validate_transform_assets(definition)


def test_snapshot_retry_resolution_guards_the_definition_digest():
    from backend.engines.install_snapshots import retry_resolved_inputs

    class Store:
        def list_install_snapshots(self, engine_id):
            return [
                {
                    "version": "v1",
                    "definition_digest": "old-digest",
                    "resolved_inputs": {
                        "version": "v1",
                        "source_repo": "https://example.test/frozen.git",
                        "source_ref": "deadbeef",
                    },
                }
            ]

    with pytest.raises(ValueError, match="definition changed"):
        retry_resolved_inputs(
            Store(), "llama_cpp", "v1", current_digest="new-digest", label="llama.cpp"
        )
    inputs = retry_resolved_inputs(
        Store(), "llama_cpp", "v1", current_digest="old-digest", label="llama.cpp"
    )
    assert inputs["source_ref"] == "deadbeef"
    assert inputs["source_repo"] == "https://example.test/frozen.git"


def test_snapshot_read_failure_does_not_silently_disable_the_guard():
    """A corrupt/unreadable ledger must raise, not act like a missing one.

    A genuinely missing ledger is the pre-refactor case and returns {}; a
    ledger that cannot be read would otherwise silently disable the digest
    check that stops a retry from reinterpreting the recorded layout.
    """
    from backend.engines.install_snapshots import install_snapshot_for

    class Store:
        def list_install_snapshots(self, engine_id):
            raise RuntimeError("install_snapshots.yaml is corrupt")

    with pytest.raises(RuntimeError, match="corrupt"):
        install_snapshot_for(Store(), "llama_cpp", "v1")


def test_cancel_after_publish_does_not_fail_the_ready_row(tmp_path):
    """Cancellation during activation must not rewrite a published version."""
    from backend.engines.definitions.loader import load_default_engine_definitions
    from backend.engines.python.native import GenericNativeInstaller

    definition = load_default_engine_definitions().definitions["llama_cpp"]
    failed = []

    class Host(GenericNativeInstaller):
        def _install_failed(self, version_name, error, install_dir):
            failed.append((version_name, error))

    host = Host(
        definition,
        install_root=str(tmp_path / "installs"),
        log_path=str(tmp_path / "install.log"),
    )
    host._published_versions.add("v1")

    assert host._fail_cancelled_install("v1", str(tmp_path / "v1")) is False
    assert failed == []
    assert host._fail_cancelled_install("v2", str(tmp_path / "v2")) is True
    assert failed and failed[0][0] == "v2"

    # A NEW attempt for the previously published version must not inherit the
    # old attempt's marker: cancelling it before publish marks the row failed.
    host._begin_install_attempt("v1")
    assert host._fail_cancelled_install("v1", str(tmp_path / "v1b")) is True
    assert failed[-1][0] == "v1"


@pytest.mark.asyncio
async def test_native_rebuild_refuses_a_changed_definition(monkeypatch):
    from fastapi import HTTPException

    from backend.routes import llama_versions

    class Store:
        def list_install_snapshots(self, engine_id):
            return [{"version": "v1", "definition_digest": "stale-digest"}]

    monkeypatch.setattr(llama_versions, "get_store", lambda: Store())
    with pytest.raises(HTTPException) as raised:
        await llama_versions._schedule_native_rebuild(
            {"version": "v1", "source_ref": "main", "source_repo": "https://example.test/x.git"},
            "llama_cpp",
        )
    assert raised.value.status_code == 400
    assert "definition changed" in str(raised.value.detail)
