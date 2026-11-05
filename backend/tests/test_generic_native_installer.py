"""Phase-7 host pins: ``GenericNativeInstaller`` (install-dir semantics).

Synthetic ``nativecraft`` definition only — no engine-id literals anywhere.
The recipes are the real audio.cpp shapes (archive download/extract, artifact
locate, CLI materialisation, publish/forget on an install directory) so the
host is proven against the vocabulary it will actually execute when audio is
wired in the next slice.

Pinned here:

* release: the definition's own plan runs end to end, publishes the version
  row, records the published snapshot and activates through the patched hook;
* ``artifact.materialize_cli`` is absolute and idempotent;
* remove: the row is forgotten and the bundle tree removed, with paths outside
  ``install_root`` skipped by the guard;
* a failing handler leaves no row and surfaces its error;
* every flow awaits ``execute_plan`` with the definition's plan, the
  plan-filtered native handlers and the documented context (both the executor
  module and this host's module-bound copies are patched, like the parity
  tests).
"""

from __future__ import annotations

import os
import re
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.data_store import DataStore
from backend.engines.definitions import materialize_plan
from backend.engines.definitions.schema import validate_definition
from backend.engines.python import native as native_mod
from backend.engines.python.native import GenericNativeInstaller

NATIVECRAFT_ID = "nativecraft"
REPO_URL = "https://example.invalid/nativecraft.git"

_HANDLER_FIELDS = {
    "server_binary_path_from": "server",
    "cli_binary_path_from": "cli_final",
}

RECIPES = {
    "release": [
        {
            "id": "download_archive",
            "uses": "archive.download",
            "requires": ["input.archive_url", "input.download_label", "input.archive_path"],
            "with": {
                "url_from": "input.archive_url",
                "path_from": "input.archive_path",
                "progress": 15,
                "progress_message_from": "input.download_label",
            },
        },
        {
            "id": "extract_archive",
            "uses": "archive.extract_tar",
            "requires": ["input.archive_path", "input.install_dir"],
            "with": {
                "archive_from": "input.archive_path",
                "dest_from": "input.install_dir",
            },
        },
        {
            "id": "verify_checksums",
            "uses": "archive.verify_checksums_file",
            "requires": ["input.install_dir"],
            "with": {"root_from": "input.install_dir"},
        },
        {
            "id": "locate_server",
            "uses": "artifact.locate",
            "requires": ["input.install_dir"],
            "with": {
                "root_from": "input.install_dir",
                "filename": "nativecraft_server",
                "label": "server",
                "chmod_or": "111",
                "error": "nativecraft_server was not found in the release archive",
            },
            "produces": ["server"],
        },
        {
            "id": "locate_cli",
            "uses": "artifact.locate",
            "requires": ["input.install_dir"],
            "with": {
                "root_from": "input.install_dir",
                "filename": "nativecraft_cli",
                "label": "cli",
                "chmod_or": "111",
            },
            "produces": ["cli"],
        },
        {
            "id": "materialize_cli",
            "uses": "artifact.materialize_cli",
            "requires": ["input.install_dir", "cli"],
            "with": {"source": "cli", "link_name": "nativecraft_cli"},
            "produces": ["cli_final"],
        },
        {
            "id": "remove_archive",
            "uses": "fs.remove_file",
            "requires": ["input.archive_path"],
            "with": {"path_from": "input.archive_path"},
        },
        {
            "id": "publish_row",
            "uses": "publish.install",
            "requires": ["input.install_dir", "input.version_name", "server", "cli_final"],
            "with": _HANDLER_FIELDS,
        },
    ],
    "source": [
        {
            "id": "ensure_dir",
            "uses": "fs.ensure_dir",
            "requires": ["input.install_dir"],
            "with": {"path_from": "input.install_dir"},
        },
        {
            "id": "publish_row",
            "uses": "publish.install",
            "requires": ["input.install_dir", "input.version_name"],
        },
    ],
    "sync": [
        {
            "id": "ensure_dir",
            "uses": "fs.ensure_dir",
            "requires": ["input.install_dir"],
            "with": {"path_from": "input.install_dir"},
        },
        {
            "id": "publish_row",
            "uses": "publish.install",
            "requires": ["input.install_dir", "input.version_name"],
        },
    ],
    "remove": [
        {
            "id": "prune_bundle",
            "uses": "artifact.remove_bundle",
            "requires": ["input.delete_target", "input.retire_references"],
            "with": {"path_from": "input.delete_target"},
        },
        {
            "id": "forget_row",
            "uses": "publish.forget",
            "requires": ["input.version_name"],
        },
    ],
}

EXPECTED_HANDLERS = {
    "release": ["artifact.materialize_cli", "publish.install"],
    "source": ["publish.install"],
    "sync": ["publish.install"],
    "remove": ["artifact.remove_bundle", "publish.forget"],
}

_BASE_CONTEXT_KEYS = {
    "input.build_dir",
    "input.engine_root",
    "input.install_dir",
    "input.installs_root",
}

EXPECTED_CONTEXT_KEYS = {
    "release": _BASE_CONTEXT_KEYS
    | {
        "input.archive_path",
        "input.archive_url",
        "input.download_label",
        "input.version_name",
        "input.install_type",
        "input.requested_version",
        "input.force_reinstall",
        "input.source_ref",
        "input.source_ref_type",
        "input.auto_activate",
    },
    "source": _BASE_CONTEXT_KEYS
    | {
        "input.version_name",
        "input.install_type",
        "input.repo_url",
        "input.repository_url",
        "input.source_repo",
        "input.branch",
        "input.reuse_mode",
        "input.source_ref",
        "input.source_ref_type",
        "input.auto_activate",
        "input.dist_dir",
    },
    "sync": _BASE_CONTEXT_KEYS
    | {
        "input.version_name",
        "input.install_type",
        "input.repo_url",
        "input.repository_url",
        "input.source_repo",
        "input.branch",
        "input.source_ref",
        "input.source_ref_type",
        "input.auto_activate",
    },
    "remove": _BASE_CONTEXT_KEYS
    | {
        "input.version_name",
        "input.install_type",
        "input.delete_target",
        "input.retire_references",
        "input.registered",
    },
}


def _definition(*, recipes: dict | None = None):
    raw = {
        "schema_version": 2,
        "id": NATIVECRAFT_ID,
        "label": "Nativecraft",
        "repository_source": "Example Org/Nativecraft",
        "install_kind": "cmake",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["server_binary_path"],
        "artifact_formats": ["builtin"],
        "package_kinds": ["builtin"],
        "tasks": ["tts"],
        "input_modalities": ["text"],
        "output_modalities": ["audio"],
        "lifecycle": {
            "binary_name": "nativecraft_server",
            "root_name": "nativecraft",
            "log_name": "nativecraft.log",
            "default_repo_url": REPO_URL,
            "forget_errors": "swallow",
        },
        "recipes": recipes if recipes is not None else RECIPES,
    }
    return validate_definition(raw, source="nativecraft.yaml")


@pytest.fixture()
def store(tmp_path, monkeypatch) -> DataStore:
    instance = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(native_mod, "get_store", lambda: instance)
    return instance


def _installer(tmp_path, *, definition=None) -> GenericNativeInstaller:
    return GenericNativeInstaller(
        definition or _definition(),
        install_root=str(tmp_path / "nativecraft"),
        log_path=str(tmp_path / "nativecraft.log"),
    )


@pytest.fixture()
def activations(monkeypatch) -> list:
    calls: list = []

    async def fake_activate(engine_id, version_name, *, covered_by=None, row=None):
        calls.append(
            {
                "engine_id": engine_id,
                "version": version_name,
                "covered_by": covered_by,
                "row": dict(row or {}),
            }
        )

    monkeypatch.setattr(native_mod, "activate_engine_version", fake_activate)
    return calls


def _make_release_archive(path) -> None:
    """A real tarball with the two binaries the release recipe locates."""
    release_root = path.parent / "staging" / "nativecraft-2.1.0"
    (release_root / "bin").mkdir(parents=True)
    for name in ("nativecraft_server", "nativecraft_cli"):
        binary = release_root / "bin" / name
        binary.write_text(f"#!/bin/sh\necho {name}\n")
        os.chmod(binary, 0o755)
    with tarfile.open(path, "w:gz") as archive:
        archive.add(release_root, arcname="nativecraft-2.1.0")


def _capture_download(installer, archive_path) -> list:
    downloads: list = []

    async def fake_download(url: str, dest: str) -> None:
        downloads.append((url, dest))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(archive_path, dest)

    installer._download_file = fake_download  # type: ignore[method-assign]
    return downloads


def _release_extra_context(archive_path) -> dict:
    return {
        "input.archive_url": "https://example.invalid/nativecraft-2.1.0.tar.gz",
        "input.download_label": "Downloading nativecraft-2.1.0.tar.gz",
        "input.archive_path": str(archive_path),
    }


def _assert_cli_entry(entry, source) -> None:
    """The entry points at (symlink) or mirrors (copy fallback) its source."""
    assert os.path.isabs(str(entry))
    assert os.path.isfile(entry)
    assert os.access(entry, os.X_OK)
    if os.path.islink(entry):
        assert os.path.realpath(entry) == os.path.realpath(str(source))
    else:
        assert Path(entry).read_bytes() == Path(source).read_bytes()
    if os.name == "posix":
        # The lane runs on Linux, where the absolute symlink itself is pinned.
        assert os.path.islink(entry)


def _seed_active_install(store, tmp_path, *, version: str = "1.0.0", install_type: str = "release"):
    install_dir = tmp_path / "nativecraft" / version
    binary = install_dir / "bin" / "nativecraft_server"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    row = {
        "version": version,
        "install_type": install_type,
        "install_dir": str(install_dir),
        "installed_at": "2026-11-03T00:00:00Z",
        "server_binary_path": str(binary),
    }
    store.add_engine_version(NATIVECRAFT_ID, row)
    store.set_active_engine_version(NATIVECRAFT_ID, version)
    return row, install_dir


# --- release: the definition's plan, row, snapshot, activation -------------------


@pytest.mark.asyncio
async def test_release_flow_publishes_the_row_snapshot_and_activation(tmp_path, store, activations):
    installer = _installer(tmp_path)
    source_archive = tmp_path / "dist" / "nativecraft-2.1.0.tar.gz"
    source_archive.parent.mkdir()
    _make_release_archive(source_archive)
    downloads = _capture_download(installer, source_archive)
    download_path = tmp_path / "downloads" / "nativecraft-2.1.0.tar.gz"

    result = await installer.install_release(
        version="2.1.0",
        extra_context=_release_extra_context(download_path),
    )

    install_dir = tmp_path / "nativecraft" / "2.1.0"
    assert result["message"] == "Nativecraft installed"
    assert result["task_id"].startswith("install_nativecraft_install_")
    assert len(downloads) == 1
    assert downloads[0][0] == "https://example.invalid/nativecraft-2.1.0.tar.gz"
    assert downloads[0][1] == str(download_path)

    rows = store.get_engine_versions(NATIVECRAFT_ID)
    assert len(rows) == 1
    row = rows[0]
    assert row["version"] == "2.1.0"
    assert row["install_type"] == row["type"] == "release"
    assert row["install_dir"] == str(install_dir)
    assert row["installed_at"]
    assert row["build_status"] == "ready"
    assert row["source_ref"] == "2.1.0"
    assert row["source_ref_type"] == "release"
    assert row["server_binary_path"].endswith("nativecraft_server")
    assert os.path.isfile(row["server_binary_path"])

    cli_entry = install_dir / "nativecraft_cli"
    assert row["cli_binary_path"] == str(cli_entry)
    _assert_cli_entry(cli_entry, install_dir / "nativecraft-2.1.0" / "bin" / "nativecraft_cli")
    # The recipe's remove_archive step deleted the downloaded tarball.
    assert not download_path.exists()
    # The staged archive itself is untouched (a download, not a source tree).
    assert source_archive.exists()

    snapshots = store.list_install_snapshots(NATIVECRAFT_ID)
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["version"] == "2.1.0"
    assert snapshot["phase"] == "published"
    assert snapshot["definition_digest"] == installer.definition.definition_digest
    assert snapshot["resolved_inputs"]["install_type"] == "release"
    assert snapshot["resolved_inputs"]["install_dir"] == str(install_dir)

    assert len(activations) == 1
    assert activations[0]["engine_id"] == NATIVECRAFT_ID
    assert activations[0]["version"] == "2.1.0"
    assert activations[0]["covered_by"] == result["task_id"]
    assert activations[0]["row"]["install_dir"] == str(install_dir)
    # The synchronous core is terminal by the time the response is returned.
    assert installer.status()["operation"] is None


@pytest.mark.asyncio
async def test_status_reports_the_active_install_and_its_binary(tmp_path, store):
    installer = _installer(tmp_path)
    row, install_dir = _seed_active_install(store, tmp_path)

    status = installer.status()

    assert status["installed"] is True
    assert status["version"] == "1.0.0"
    assert status["install_dir"] == str(install_dir)
    assert status["binary_path"] == os.path.abspath(row["server_binary_path"])
    assert status["install_type"] == "release"
    assert status["operation"] is None
    assert status["log_path"] == str(tmp_path / "nativecraft.log")


# --- artifact.materialize_cli ----------------------------------------------------


@pytest.mark.asyncio
async def test_materialize_cli_is_absolute_and_idempotent(tmp_path, store):
    installer = _installer(tmp_path)
    install_dir = tmp_path / "nativecraft" / "1.0.0"
    payload = install_dir / "payload"
    payload.mkdir(parents=True)
    cli = payload / "nativecraft_cli"
    cli.write_text("#!/bin/sh\n")
    os.chmod(cli, 0o755)

    step = SimpleNamespace(with_={"source": "cli", "link_name": "nativecraft_cli"})
    context = {"input.install_dir": str(install_dir), "cli": str(cli)}

    first = await installer._materialize_cli(step, {}, context)
    second = await installer._materialize_cli(step, {}, context)

    assert first["cli_created"] is True
    assert second["cli_created"] is False
    assert first["cli_final"] == second["cli_final"]
    assert first["cli_final"] == os.path.join(str(install_dir), "nativecraft_cli")
    _assert_cli_entry(first["cli_final"], cli)

    # A stale SYMLINK entry is replaced, never silently kept.
    stale_target = payload / "nativecraft_cli.next"
    stale_target.write_text("#!/bin/sh\necho next\n")
    os.chmod(stale_target, 0o755)
    entry = Path(first["cli_final"])
    if os.path.islink(entry):
        os.unlink(entry)
        os.symlink(str(stale_target), str(entry))
        replaced = await installer._materialize_cli(step, {}, context)
        assert replaced["cli_created"] is True
        assert os.path.realpath(replaced["cli_final"]) == os.path.realpath(cli)


# --- remove: forget the row, remove the bundle, guard the root -------------------


@pytest.mark.asyncio
async def test_remove_forgets_the_row_and_removes_the_bundle(tmp_path, store):
    installer = _installer(tmp_path)
    _row, install_dir = _seed_active_install(store, tmp_path)

    result = await installer.remove()

    assert result["message"] == "Nativecraft removed"
    assert result["task_id"].startswith("install_nativecraft_remove_")
    assert store.get_engine_versions(NATIVECRAFT_ID) == []
    assert store.get_active_engine_version(NATIVECRAFT_ID) is None
    assert not install_dir.exists()

    with pytest.raises(RuntimeError, match="no active installation to remove"):
        await installer.remove()


@pytest.mark.asyncio
async def test_remove_bundle_skips_paths_outside_the_install_root(tmp_path, store):
    installer = _installer(tmp_path)
    outside = tmp_path / "elsewhere" / "bundle"
    outside.mkdir(parents=True)
    (outside / "keep.txt").write_text("keep")

    step = SimpleNamespace(with_={})
    result = await installer._remove_bundle(step, {}, {"input.install_dir": str(outside)})
    assert result == {"removed": False}
    assert (outside / "keep.txt").exists()

    # The install root itself is not a bundle either.
    result = await installer._remove_bundle(step, {}, {"input.install_dir": installer.install_root})
    assert result == {"removed": False}
    assert os.path.isdir(installer.install_root)


@pytest.mark.asyncio
async def test_install_dir_outside_the_install_root_is_refused(tmp_path, store):
    installer = _installer(tmp_path)
    outside = tmp_path / "elsewhere"

    with pytest.raises(ValueError, match="outside the install root"):
        await installer.install_release(version="1.0.0", reuse_dir=str(outside))
    assert not outside.exists()


@pytest.mark.asyncio
async def test_long_version_names_keep_their_uniqueness_suffix(tmp_path, store, activations):
    """A 64-char version must not collapse onto the existing install dir.

    Regression: slugging AFTER the uniqueness check truncated ``<base>-<stamp>``
    back to ``<base>``, so the second install reused the first row and dir.
    """
    definition = _definition(
        recipes={
            "release": [
                {
                    "id": "publish_row",
                    "uses": "publish.install",
                    "requires": ["input.install_dir", "input.version_name"],
                }
            ]
        }
    )
    installer = _installer(tmp_path, definition=definition)
    requested = "v" * 64

    await installer.install_release(version=requested)
    await installer.install_release(version=requested)

    rows = store.get_engine_versions(NATIVECRAFT_ID)
    assert len(rows) == 2
    first, second = rows[0]["version"], rows[1]["version"]
    assert first == requested
    assert second != first
    assert re.fullmatch(r".+-\d{10}", second), second
    assert len(second) <= 64

    first_dir, second_dir = rows[0]["install_dir"], rows[1]["install_dir"]
    assert first_dir != second_dir
    assert os.path.dirname(first_dir) == os.path.dirname(second_dir) == installer.install_root
    assert os.path.isdir(first_dir) and os.path.isdir(second_dir)

    # A third pass still lands on a fresh name (no suffix reuse).
    await installer.install_release(version=requested)
    third = store.get_engine_versions(NATIVECRAFT_ID)[2]["version"]
    assert third not in {first, second}
    assert third.startswith("v" * 10)
    assert len(third) <= 64


# --- failure surfaces, and a failed install publishes nothing --------------------


@pytest.mark.asyncio
async def test_handler_failure_leaves_no_row_and_surfaces_the_error(tmp_path, store, activations):
    definition = _definition(
        recipes={
            "release": [
                {
                    "id": "materialize_cli",
                    "uses": "artifact.materialize_cli",
                    "requires": ["input.install_dir"],
                    "with": {"source": "cli", "link_name": "nativecraft_cli"},
                }
            ]
        }
    )
    installer = _installer(tmp_path, definition=definition)

    with pytest.raises(RuntimeError, match="no CLI binary"):
        await installer.install_release(version="9.9.9")

    assert store.get_engine_versions(NATIVECRAFT_ID) == []
    assert store.list_install_snapshots(NATIVECRAFT_ID) == []
    assert activations == []
    status = installer.status()
    assert status["operation"] is None
    assert "no CLI binary" in (status["last_error"] or "")


# --- source + sync publish source provenance -------------------------------------


@pytest.mark.asyncio
async def test_source_flow_publishes_source_provenance(tmp_path, store, activations):
    installer = _installer(tmp_path)

    result = await installer.install_from_source(REPO_URL, "feature/x")

    assert result["message"] == "Nativecraft install from source started"
    assert result["task_id"].startswith("install_nativecraft_install_source_")
    rows = store.get_engine_versions(NATIVECRAFT_ID)
    assert len(rows) == 1
    row = rows[0]
    assert row["version"] == "source-feature-x"
    assert row["install_type"] == row["type"] == "source"
    assert row["source_repo"] == REPO_URL
    assert row["source_branch"] == row["source_ref"] == "feature/x"
    assert row["source_ref_type"] == "branch"
    assert row["install_dir"] == str(tmp_path / "nativecraft" / "source-feature-x")
    assert activations[0]["version"] == "source-feature-x"


@pytest.mark.asyncio
async def test_retry_resolves_frozen_snapshot_inputs(tmp_path, store, activations):
    installer = _installer(tmp_path)
    await installer.install_from_source(REPO_URL, "feature/x")
    row = store.get_engine_versions(NATIVECRAFT_ID)[0]
    captured = {}

    async def fake_source(repo_url, branch, *, reuse_dir=None, existing_version=None):
        captured.update(
            repo_url=repo_url,
            branch=branch,
            reuse_dir=reuse_dir,
            existing_version=existing_version,
        )
        return {"task_id": "retry"}

    installer.install_from_source = fake_source
    # The row omits source identity: the frozen snapshot must supply it.
    retry_row = {
        "version": row["version"],
        "install_type": "source",
        "install_dir": row["install_dir"],
    }
    await installer.retry_existing_install(retry_row)
    assert captured == {
        "repo_url": REPO_URL,
        "branch": "feature/x",
        "reuse_dir": row["install_dir"],
        "existing_version": row["version"],
    }


@pytest.mark.asyncio
async def test_retry_refuses_a_changed_definition(tmp_path, store, activations):
    installer = _installer(tmp_path)
    store.record_install_snapshot(
        NATIVECRAFT_ID,
        definition_digest="stale-digest",
        resolved_inputs={"version": "source-main"},
        version="source-main",
    )
    with pytest.raises(ValueError, match="definition changed"):
        await installer.retry_existing_install(
            {
                "version": "source-main",
                "install_type": "source",
                "install_dir": str(tmp_path / "nativecraft" / "source-main"),
            }
        )


@pytest.mark.asyncio
async def test_sync_republishes_the_recorded_version(tmp_path, store, activations):
    installer = _installer(tmp_path)
    row, _install_dir = _seed_active_install(store, tmp_path, version="source-main")

    result = await installer.sync_source_version(
        {
            "version": "source-main",
            "install_type": "source",
            "install_dir": str(tmp_path / "nativecraft" / "source-main"),
            "source_branch": "main",
            "source_repo": REPO_URL,
        }
    )

    assert result["task_id"].startswith("install_nativecraft_sync_source_")
    updated = store.get_engine_versions(NATIVECRAFT_ID)[0]
    assert updated["version"] == "source-main"
    assert updated["install_type"] == "source"
    assert activations[-1]["version"] == "source-main"

    # Syncing an INACTIVE install republishes its row but never switches the
    # engine (legacy audio.cpp semantics).
    store.add_engine_version(
        NATIVECRAFT_ID,
        {
            "version": "source-dev",
            "install_type": "source",
            "install_dir": str(tmp_path / "nativecraft" / "source-dev"),
        },
    )
    activations.clear()
    await installer.sync_source_version(
        {
            "version": "source-dev",
            "install_type": "source",
            "install_dir": str(tmp_path / "nativecraft" / "source-dev"),
            "source_branch": "dev",
            "source_repo": REPO_URL,
        }
    )
    assert activations == []

    with pytest.raises(RuntimeError, match="Only Nativecraft source installs can be synced"):
        await installer.sync_source_version({**row, "install_type": "release"})


# --- the flows execute the definition's plan, filtered handlers and context ------


class _CompletedExecution:
    status = "completed"
    failed_step = None


def _step_signature(plan) -> list:
    steps = getattr(plan, "steps", plan)
    return [(str(step.id), str(step.primitive)) for step in steps]


def _capture_plan(monkeypatch) -> list:
    calls: list = []

    def fake_make_engine_runner(installer, handlers):
        # The runner is built BEFORE execute_plan is awaited, so the call row
        # is opened here and filled in by the executor stand-in.
        calls.append({"handlers": sorted(handlers)})
        return lambda *args, **kwargs: None

    async def fake_execute_plan(plan, *, run_primitive, context):
        calls[-1]["plan"] = _step_signature(plan)
        calls[-1]["context"] = sorted(context)
        return _CompletedExecution()

    # The executor module AND this host's module-bound copies (parity-test
    # convention): patching only one side would leave a real run behind.
    monkeypatch.setattr("backend.engines.definitions.executor.execute_plan", fake_execute_plan)
    monkeypatch.setattr("backend.engines.definitions.make_engine_runner", fake_make_engine_runner)
    monkeypatch.setattr(native_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(native_mod, "make_engine_runner", fake_make_engine_runner)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("flow", ["release", "source", "sync", "remove"])
async def test_flows_execute_the_definition_plan_with_filtered_handlers(
    tmp_path, store, activations, monkeypatch, flow
):
    installer = _installer(tmp_path)
    calls = _capture_plan(monkeypatch)
    archive_path = tmp_path / "nativecraft-2.1.0.tar.gz"

    if flow == "release":
        await installer.install_release(
            version="2.1.0",
            extra_context={
                "input.archive_url": "https://example.invalid/nativecraft-2.1.0.tar.gz",
                "input.download_label": "Downloading nativecraft-2.1.0.tar.gz",
                "input.archive_path": str(archive_path),
            },
        )
    elif flow == "source":
        await installer.install_from_source(
            REPO_URL, "main", extra_context={"input.dist_dir": str(tmp_path / "dist")}
        )
    elif flow == "sync":
        _seed_active_install(store, tmp_path, version="source-main")
        await installer.sync_source_version(
            {
                "version": "source-main",
                "install_type": "source",
                "install_dir": str(tmp_path / "nativecraft" / "source-main"),
                "source_branch": "main",
                "source_repo": REPO_URL,
            }
        )
    else:
        _seed_active_install(store, tmp_path)
        await installer.remove()

    assert len(calls) == 1, calls
    expected = materialize_plan(installer.definition, flow)
    assert calls[0]["plan"] == _step_signature(expected)
    assert calls[0]["handlers"] == EXPECTED_HANDLERS[flow]
    assert calls[0]["context"] == sorted(EXPECTED_CONTEXT_KEYS[flow])


# --- download contract -----------------------------------------------------------


@pytest.mark.asyncio
async def test_download_refuses_non_http_urls(tmp_path, store):
    installer = _installer(tmp_path)
    with pytest.raises(native_mod.PrimitiveExecutionError, match="non-HTTP"):
        await installer._download_file("file:///etc/passwd", str(tmp_path / "payload.bin"))
    assert not (tmp_path / "payload.bin").exists()


# --- externally owned operations (route task plumbing) ---------------------------


@pytest.mark.asyncio
async def test_execute_operation_binds_the_route_task_and_install_dir(tmp_path, store, monkeypatch):
    """`execute_operation` runs a recipe without owning the operation.

    The caller (a route with its own task/admission bookkeeping) supplies the
    task id and install dir; progress from shared primitives must reach that
    task, no operation may be started/finished, and the bindings must be
    restored afterwards.
    """
    installer = _installer(tmp_path)
    captured: dict = {}

    def fake_make_engine_runner(instance, handlers):
        captured["task_id"] = instance.progress_task_id
        captured["install_dir"] = instance.install_dir
        captured["handlers"] = sorted(handlers)
        return lambda *args, **kwargs: None

    async def fake_execute_plan(plan, *, run_primitive, context):
        captured["plan"] = _step_signature(plan)
        captured["context"] = sorted(context)
        return _CompletedExecution()

    monkeypatch.setattr(native_mod, "make_engine_runner", fake_make_engine_runner)
    monkeypatch.setattr(native_mod, "execute_plan", fake_execute_plan)

    install_dir = tmp_path / "nativecraft" / "9.9.9"
    execution = await installer.execute_operation(
        "release",
        task_id="route-task-1",
        install_dir=str(install_dir),
        context={"input.install_dir": str(install_dir), "input.version_name": "9.9.9"},
    )

    assert execution.status == "completed"
    assert captured["task_id"] == "route-task-1"
    assert captured["install_dir"] == str(install_dir)
    assert captured["handlers"] == ["artifact.materialize_cli", "publish.install"]
    assert captured["context"] == ["input.install_dir", "input.version_name"]
    assert os.path.isdir(install_dir)
    # Nothing was started/finished, and the host state is restored.
    assert installer.status()["operation"] is None
    assert installer.progress_task_id is None
    assert installer.install_dir is None
    assert installer.build_dir is None
