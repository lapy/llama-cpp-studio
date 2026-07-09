"""Prepared-bundle install/import transactions and cancellation."""

from types import SimpleNamespace

import pytest

from backend.services.audio_model_installer import (
    AudioModelInstaller,
    _source_kind_method,
)
from backend.task_cancel_registry import (
    TaskCancelledError,
    register_task_cancel,
    request_task_cancel,
    unregister_task_cancel,
)


class _Store:
    def __init__(self, active):
        self.active = active
        self.models = {}

    def get_active_engine_version(self, engine):
        return self.active if engine == "audio_cpp" else None

    def get_model(self, model_id):
        return self.models.get(model_id)

    def add_model(self, model):
        self.models[model["id"]] = model
        return model


class _Progress:
    def __init__(self):
        self.updates = []

    def update_task(self, task_id, **payload):
        self.updates.append((task_id, payload))


def _installer(tmp_path, monkeypatch):
    cli = tmp_path / "audiocpp_cli"
    cli.write_text("#!/bin/sh\n", encoding="utf-8")
    manager_script = tmp_path / "model_manager.py"
    manager_script.write_text("", encoding="utf-8")
    active = {
        "version": "v1",
        "source_commit": "abc",
        "cli_binary_path": str(cli),
        "model_manager_path": str(manager_script),
        "build_config": {"backend": "cpu"},
    }
    store = _Store(active)
    installer = AudioModelInstaller(store)
    installer.manager = SimpleNamespace(
        models_dir=str(tmp_path / "managed"),
        tools_dir=str(tmp_path / "tools"),
    )
    installer.pm = _Progress()
    monkeypatch.setattr(
        "backend.services.audio_model_installer.scan_audio_cpp_model_profile",
        lambda *args, **kwargs: {},
    )
    return installer, store


def test_install_method_contract_covers_direct_composite_and_converter():
    assert _source_kind_method({"kind": "huggingface_snapshot"}) == "direct"
    assert _source_kind_method({"kind": "composite_snapshot"}) == "composite"
    assert _source_kind_method({"kind": "composite"}) == "converter"
    assert _source_kind_method({"kind": "utility"}) == "converter"
    assert _source_kind_method({"kind": "external"}) == "unavailable"


@pytest.mark.asyncio
async def test_local_import_validates_then_atomically_promotes_bundle(
    tmp_path, monkeypatch
):
    installer, store = _installer(tmp_path, monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "weights.bin").write_bytes(b"model")

    async def inspect(*args, **kwargs):
        return {
            "family": "demo_tts",
            "task_names": ["tts"],
            "tasks": [{"task": "tts", "modes": ["offline"]}],
            "capabilities": {"streaming": False},
        }

    monkeypatch.setattr(installer, "_inspect", inspect)

    record = await installer.import_local_bundle(
        "import-1", str(source), package_id="demo"
    )

    final_bundle = tmp_path / "managed" / "demo"
    assert record["id"] == "audio-cpp--demo"
    assert record["compatible_engines"] == ["audio_cpp"]
    assert record["artifact"]["package_kind"] == "prepared_bundle"
    assert record["artifact"]["path"] == str(final_bundle)
    assert (final_bundle / "weights.bin").read_bytes() == b"model"
    assert (final_bundle / ".studio-manifest.json").is_file()
    assert store.get_model(record["id"]) is record
    assert not list((tmp_path / "managed" / ".staging").glob("import-*"))


@pytest.mark.asyncio
async def test_failed_inspection_removes_staging_and_promoted_paths(
    tmp_path, monkeypatch
):
    installer, _ = _installer(tmp_path, monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"model")

    async def fail_inspection(*args, **kwargs):
        raise RuntimeError("invalid package")

    monkeypatch.setattr(installer, "_inspect", fail_inspection)

    with pytest.raises(RuntimeError, match="invalid package"):
        await installer.import_local_bundle(
            "import-fail", str(source), package_id="broken"
        )

    assert not (tmp_path / "managed" / "broken").exists()
    assert not list((tmp_path / "managed" / ".staging").glob("import-*"))


@pytest.mark.asyncio
async def test_local_copy_rejects_symlinks(tmp_path, monkeypatch):
    installer, _ = _installer(tmp_path, monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (source / "escape").symlink_to(target, target_is_directory=True)
    destination = tmp_path / "destination"

    with pytest.raises(ValueError, match="unsupported directory symlink"):
        await installer._copy_local_bundle("copy-symlink", str(source), str(destination))


@pytest.mark.asyncio
async def test_local_copy_honors_cancellation_before_writing(tmp_path, monkeypatch):
    installer, _ = _installer(tmp_path, monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"model")
    destination = tmp_path / "destination"
    task_id = "cancel-copy"
    register_task_cancel(task_id)
    request_task_cancel(task_id)
    try:
        with pytest.raises(TaskCancelledError, match="cancelled"):
            await installer._copy_local_bundle(
                task_id, str(source), str(destination)
            )
    finally:
        unregister_task_cancel(task_id)

    assert not (destination / "weights.bin").exists()


@pytest.mark.asyncio
async def test_existing_model_record_blocks_import_before_copy(tmp_path, monkeypatch):
    installer, store = _installer(tmp_path, monkeypatch)
    store.models["audio-cpp--demo"] = {"id": "audio-cpp--demo"}
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"model")

    with pytest.raises(FileExistsError, match="already exists"):
        await installer.import_local_bundle(
            "duplicate", str(source), package_id="demo"
        )

    assert not (tmp_path / "managed").exists()

