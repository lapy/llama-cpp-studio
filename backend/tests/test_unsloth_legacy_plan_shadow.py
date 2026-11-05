"""Phase-3 gates: Unsloth llama.cpp release executes the definition plan.

Unsloth installs PREBUILT llama-server tarballs only: a host-aware release
walk-back (manifest plan, else filename fallback), a pending row that gets
RENAMED once the real tag is known (legacy order: rename before any disk
touch), checksum manifest in many layouts, a path-traversal-guarded r:gz
extract that removes the archive, and a publish with NO swallow — unlike
the Python-family engines, activation/metadata failures fail the install.
These gates pin the executed plan byte-for-byte against that legacy shape.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tarfile
from pathlib import Path

import pytest

from backend.engines.unsloth_llama.installer import UnslothLlamaInstaller
from backend.engines.unsloth_llama.prebuilt import (
    CHECKSUM_ASSET_NAME,
    HostTarget,
    PublishedArtifact,
    ReleasePlan,
)

ENGINE = "unsloth_llama"
ASSET_NAME = "llama-server-cuda12-b9999-linux-x64.tar.gz"
ARCHIVE_URL = f"https://download.test/{ASSET_NAME}"
CHECKSUM_URL = f"https://download.test/{CHECKSUM_ASSET_NAME}"
ARCHIVE_BYTES = b"TARBALL-PAYLOAD"
TAG = "b9999"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tar_bytes(names: tuple[tuple[str, bytes], ...] = (("bundle/bin/llama-server", b"bin\n"),)):
    """Deterministic tar.gz bytes (digest + download must share one buffer)."""
    import io

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for name, data in names:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _make_installer(tmp_path: Path, calls: list):
    installer = UnslothLlamaInstaller(base_dir=str(tmp_path / "unsloth-llama"))
    task_holder: dict = {}
    installer._create_task = lambda coro: task_holder.__setitem__("coro", coro)

    async def fake_progress(pct, message):
        calls.append(("progress", pct, message))

    installer._update_progress_task = fake_progress
    installer._fail_pending = lambda name, error, extra=None, **kw: calls.append(
        ("fail", name, error, dict(extra or {}))
    )
    return installer, task_holder


def _patch_lifecycle(monkeypatch, calls: list):
    import backend.engines.lifecycle as lifecycle
    import backend.engines.python.native as native

    def fake_building(store, engine, payload, task_id=None):
        calls.append(("building", engine, payload.get("version")))

    def fake_ready(store, engine, payload):
        calls.append(("ready", engine, dict(payload)))

    monkeypatch.setattr(lifecycle, "mark_engine_version_building", fake_building)
    monkeypatch.setattr(lifecycle, "mark_engine_version_ready", fake_ready)
    monkeypatch.setattr(native, "mark_engine_version_ready", fake_ready)


def _patch_store(monkeypatch, calls: list, existing=()):
    import backend.data_store

    store_cls = type(backend.data_store.get_store())
    monkeypatch.setattr(store_cls, "get_engine_versions", lambda self, engine: list(existing))

    def fake_delete(self, engine, version):
        calls.append(("row-delete", engine, version))

    monkeypatch.setattr(store_cls, "delete_engine_version", fake_delete)


def _plan(tag=TAG, asset_name=ASSET_NAME):
    return ReleasePlan(
        tag=tag,
        asset_name=asset_name,
        flavor="cuda",
        artifact=PublishedArtifact(
            asset_name=asset_name, runtime_line="cuda12", coverage_class="full"
        ),
        skipped_newer=("b-newer",),
        upstream_tag="b7779",
        reason="best coverage",
    )


def _release(tag=TAG):
    return {
        "tag_name": tag,
        "html_url": f"https://github.test/releases/{tag}",
        "assets": [
            {"name": "ignored.zip", "browser_download_url": "https://download.test/ignored.zip"},
            {"name": ASSET_NAME, "browser_download_url": ARCHIVE_URL},
            {"name": CHECKSUM_ASSET_NAME, "browser_download_url": CHECKSUM_URL},
        ],
    }


def _patch_resolution(installer, calls, plan=None, manifest=None, error=None):
    async def fake_host():
        return HostTarget(cuda_major=12, sms=(90,), has_nvidia=True)

    async def fake_resolve(tag_name, host):
        calls.append(("resolve", tag_name))
        if error is not None:
            raise error
        return _release(plan.tag if plan else TAG), plan or _plan(), manifest

    installer._host_target = fake_host
    installer._resolve_plan = fake_resolve


def _patch_downloads(installer, calls, *, archive_bytes=ARCHIVE_BYTES, sums_digest=None):
    async def fake_download(url, dest):
        calls.append(("download", url, dest))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if url == CHECKSUM_URL:
            manifest = {ASSET_NAME: sums_digest or _digest(archive_bytes)}
            with open(dest, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
        elif url == ARCHIVE_URL:
            with open(dest, "wb") as handle:
                handle.write(archive_bytes)
        else:  # pragma: no cover - guards the recipe wiring
            raise AssertionError(f"unexpected download url {url}")

    installer._download_file = fake_download


async def _drive(installer, task_holder, invoke, calls: list | None = None):
    response = await invoke()
    await task_holder["coro"]
    from backend.operations.supervisor import get_supervisor

    get_supervisor().finish_operation(response["task_id"], "succeeded", "")
    return response


@pytest.mark.asyncio
async def test_release_downloads_verifies_extracts_and_publishes(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    _patch_lifecycle(monkeypatch, calls)
    _patch_store(monkeypatch, calls)

    # Real published manifest + a real tar.gz produced by the download fake.
    published_manifest = {"artifacts": {"files": [{"name": ASSET_NAME, "sha256": "unused"}]}}
    tars = _tar_bytes()
    _patch_downloads(installer, calls, archive_bytes=tars)
    _patch_resolution(installer, calls, manifest=published_manifest)
    _drive_activate(monkeypatch, calls)

    await _drive(installer, task_holder, lambda: installer.install_release(), calls)

    kinds = [call[0] for call in calls]
    assert kinds == [
        "resolve",  # latest tag resolved before the version row is claimed
        "building",  # pending row for the resolved tag
        "progress",  # 8 (recipe fetches the cached release)
        "progress",  # 20
        "download",  # checksum manifest
        "progress",  # 40
        "download",  # archive
        "progress",  # 72
        "progress",  # 82
        "ready",
        "activate",
    ]
    progress = [call[1:] for call in calls if call[0] == "progress"]
    assert progress == [
        (8, "Fetching Unsloth llama.cpp release"),
        (20, "Downloading checksum manifest"),
        (40, f"Downloading {ASSET_NAME}"),
        (72, "Verifying archive checksum"),
        (82, "Extracting llama-server"),
    ]
    urls = [call[1] for call in calls if call[0] == "download"]
    assert urls == [CHECKSUM_URL, ARCHIVE_URL]

    install_dir = tmp_path / "unsloth-llama" / TAG
    archive = install_dir / ASSET_NAME
    assert not archive.exists()  # archive removed after extraction
    binary = install_dir / "bundle" / "bin" / "llama-server"
    assert binary.is_file()
    assert os.access(binary, os.X_OK)  # chmod 0755 by the locate step
    manifest_copy = install_dir / "llama-prebuilt-manifest.json"
    assert json.loads(manifest_copy.read_text(encoding="utf-8")) == published_manifest

    row = next(call for call in calls if call[0] == "ready")[2]
    assert row["version"] == TAG
    assert row["binary_path"] == str(binary)
    assert row["install_dir"] == str(install_dir)
    assert row["source_ref"] == TAG and row["source_ref_type"] == "tag"
    assert row["cuda_version"] == "12"
    config = row["build_config"]
    assert config["tag_name"] == TAG
    assert config["asset_name"] == ASSET_NAME
    assert config["flavor"] == "cuda"
    assert config["coverage_class"] == "full"
    assert config["runtime_line"] == "cuda12"
    assert config["upstream_tag"] == "b7779"
    assert config["selection_reason"] == "best coverage"
    assert config["host_sms"] == [90]
    assert config["skipped_newer"] == ["b-newer"]
    assert config["release_url"] == f"https://github.test/releases/{TAG}"
    assert ("activate", ENGINE, TAG) in calls
    assert installer._last_error is None
    assert not [call for call in calls if call[0] == "fail"]


@pytest.mark.asyncio
async def test_release_keeps_pending_row_when_tag_already_matches(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    _patch_lifecycle(monkeypatch, calls)
    _patch_store(monkeypatch, calls)
    _patch_resolution(installer, calls, plan=_plan(tag="b5"))
    _patch_downloads(installer, calls, archive_bytes=_tar_bytes())
    _drive_activate(monkeypatch, calls)

    await _drive(installer, task_holder, lambda: installer.install_release("b5"), calls)

    assert not [call for call in calls if call[0] == "row-delete"]
    assert next(call for call in calls if call[0] == "ready")[2]["version"] == "b5"
    resolve = next(call for call in calls if call[0] == "resolve")
    assert resolve[1] == "b5"  # tag pinned through to the walk-back


def _drive_activate(monkeypatch, calls):
    import backend.engines.activation as activation
    import backend.engines.python.native as native

    async def fake_activate(engine, version_name, **kw):
        calls.append(("activate", engine, version_name))

    monkeypatch.setattr(activation, "activate_engine_version", fake_activate)
    monkeypatch.setattr(native, "activate_engine_version", fake_activate)


@pytest.mark.asyncio
async def test_release_requires_linux_x64_host(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, _ = _make_installer(tmp_path, calls)
    import backend.engines.unsloth_llama.installer as module

    monkeypatch.setattr(module, "host_is_linux_x64", lambda: False)
    with pytest.raises(RuntimeError, match="Linux x64 only"):
        await installer.install_release()
    assert calls == []


@pytest.mark.asyncio
async def test_release_busy_refuses_second_operation(tmp_path: Path):
    calls: list = []
    installer, _ = _make_installer(tmp_path, calls)
    installer._operation = "install"
    with pytest.raises(RuntimeError, match="already running"):
        await installer.install_release()


@pytest.mark.asyncio
async def test_release_checksum_mismatch_fails_before_extract(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    _patch_lifecycle(monkeypatch, calls)
    _patch_store(monkeypatch, calls)
    _patch_resolution(installer, calls)
    _patch_downloads(installer, calls, sums_digest="0" * 64)
    _drive_activate(monkeypatch, calls)

    await _drive(installer, task_holder, lambda: installer.install_release())

    assert installer._last_error == (
        f"Checksum mismatch for {ASSET_NAME}: expected {'0' * 64}, got {_digest(ARCHIVE_BYTES)}"
    )
    fail = next(call for call in calls if call[0] == "fail")
    assert fail[1] == TAG
    assert "install_dir" in fail[3]
    assert (82, "Extracting llama-server") not in [
        call[1:] for call in calls if call[0] == "progress"
    ]
    assert not (tmp_path / "unsloth-llama" / TAG / "bundle").exists()


@pytest.mark.asyncio
async def test_release_resolution_error_fails_pending_row(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    _patch_lifecycle(monkeypatch, calls)
    _patch_store(monkeypatch, calls)
    error = RuntimeError(f"No Linux x64 bundle in {TAG} covers this host (CUDA 12, SM 90)")
    _patch_resolution(installer, calls, error=error)
    _drive_activate(monkeypatch, calls)

    # Latest-tag resolution now happens before a version row is claimed, so the
    # route rejects the request instead of starting a doomed pending install.
    with pytest.raises(RuntimeError, match="No Linux x64 bundle"):
        await installer.install_release()
    assert not [call for call in calls if call[0] in {"building", "download"}]
    assert not [call for call in calls if call[0] == "row-delete"]


@pytest.mark.asyncio
async def test_release_missing_assets_keep_legacy_error_strings(tmp_path: Path, monkeypatch):
    # No checksum asset in the release.
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    _patch_lifecycle(monkeypatch, calls)
    _patch_store(monkeypatch, calls)

    async def host():
        return HostTarget(cuda_major=12, sms=(90,))

    async def no_checksum(tag_name, host_target):
        release = _release()
        release["assets"] = [a for a in release["assets"] if a["name"] != CHECKSUM_ASSET_NAME]
        return release, _plan(), None

    installer._host_target = host
    installer._resolve_plan = no_checksum
    _drive_activate(monkeypatch, calls)
    await _drive(installer, task_holder, lambda: installer.install_release())
    assert installer._last_error == f"{CHECKSUM_ASSET_NAME} is missing from Unsloth release {TAG}"

    # Planned asset name not present among the release assets.
    calls2: list = []
    installer2, task_holder2 = _make_installer(tmp_path, calls2)
    _patch_lifecycle(monkeypatch, calls2)
    _patch_store(monkeypatch, calls2)
    _patch_resolution(installer2, calls2, plan=_plan(asset_name="ghost.tar.gz"))
    await _drive(
        installer2,
        task_holder2,
        lambda: installer2.install_release(),
    )
    assert installer2._last_error == f"No Linux x64 cuda archive in Unsloth release {TAG}"


@pytest.mark.asyncio
async def test_release_without_llama_server_in_archive_fails(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    _patch_lifecycle(monkeypatch, calls)
    _patch_store(monkeypatch, calls)
    _patch_resolution(installer, calls)
    tars = _tar_bytes((("bundle/README", b"hi"),))
    _patch_downloads(installer, calls, archive_bytes=tars)
    _drive_activate(monkeypatch, calls)

    await _drive(installer, task_holder, lambda: installer.install_release())

    assert installer._last_error == "llama-server was not found in the Unsloth archive"
    # Extraction ran (and removed the archive) before the locate failed.
    assert not (tmp_path / "unsloth-llama" / TAG / ASSET_NAME).exists()
    assert [call[0] for call in calls].count("ready") == 0


@pytest.mark.asyncio
async def test_release_cancellation_fails_pending_and_reraises(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    _patch_lifecycle(monkeypatch, calls)
    _patch_store(monkeypatch, calls)
    _patch_resolution(installer, calls)

    async def cancelling(url, dest):
        raise asyncio.CancelledError()

    installer._download_file = cancelling
    _drive_activate(monkeypatch, calls)

    response = await installer.install_release()
    with pytest.raises(asyncio.CancelledError):
        await task_holder["coro"]
    from backend.operations.supervisor import get_supervisor

    get_supervisor().finish_operation(response["task_id"], "cancelled", "")

    fail = next(call for call in calls if call[0] == "fail")
    assert fail[1] == TAG
    assert fail[2] == "Operation cancelled by user"
    assert "install_dir" in fail[3]
    assert installer._last_error is None or installer._last_error == ("Operation cancelled by user")


def test_release_asset_policy_comes_from_the_definition(tmp_path: Path):
    from backend.engines.definitions.loader import load_default_engine_definitions
    from backend.engines.unsloth_llama.installer import UnslothLlamaInstaller

    policy = dict(load_default_engine_definitions().definitions["unsloth_llama"].release_assets)
    assert policy == {
        "checksum_asset": "llama-prebuilt-sha256.json",
        "manifest_asset": "llama-prebuilt-manifest.json",
        "max_walkback": 8,
        "repository": "unslothai/llama.cpp",
        "repository_url": "https://github.com/unslothai/llama.cpp.git",
    }

    installer = UnslothLlamaInstaller(base_dir=str(tmp_path / "unsloth"))
    assert installer._repository == policy["repository"]
    assert installer._manifest_asset == policy["manifest_asset"]
    assert installer._checksum_asset == policy["checksum_asset"]
    assert installer._max_walkback == policy["max_walkback"]
    assert installer._repository_url == policy["repository_url"]


def test_published_snapshot_records_the_definition_digest(tmp_path: Path, monkeypatch):
    import backend.engines.unsloth_llama.installer as unsloth_installer

    installer = unsloth_installer.UnslothLlamaInstaller(base_dir=str(tmp_path / "unsloth"))
    captured: dict = {}

    class Store:
        def record_install_snapshot(self, engine, **kwargs):
            captured.update(engine=engine, **kwargs)

    monkeypatch.setattr(unsloth_installer, "get_store", lambda: Store())
    installer._record_published_snapshot(
        "b9999",
        {
            "install_type": "release",
            "version": "b9999",
            "release_tag": "b9999",
            "install_dir": str(tmp_path / "unsloth" / "b9999"),
        },
    )
    assert captured["engine"] == "unsloth_llama"
    assert captured["version"] == "b9999"
    assert captured["phase"] == "published"
    assert captured["definition_digest"]
    assert captured["resolved_inputs"]["release_tag"] == "b9999"


def test_sha256_helpers_still_pinned_by_module_functions(
    tmp_path: Path,
):  # verify_sha256 remains importable for existing integration tests; the
    # shared primitive must raise the SAME message on mismatch.
    from backend.engines.unsloth_llama.installer import verify_sha256

    target = tmp_path / "x.bin"
    target.write_bytes(ARCHIVE_BYTES)
    verify_sha256(str(target), _digest(ARCHIVE_BYTES))
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        verify_sha256(str(target), "0" * 64)
