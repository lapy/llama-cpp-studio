"""Prepared-bundle install/import transactions and cancellation."""

import os
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
        self.events = []

    def update_task(self, task_id, **payload):
        self.updates.append((task_id, payload))

    def emit(self, event, payload):
        self.events.append((event, payload))


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


def test_resolve_installed_model_path_publishes_nested_gguf_as_model_gguf(tmp_path):
    root = tmp_path / "ACE-Step1.5-GGUF"
    nested = root / "turbo"
    nested.mkdir(parents=True)
    gguf = nested / "ace-step-1.5-turbo-bf16.gguf"
    gguf.write_bytes(b"GGUF")
    # No package.files hint — discovery must find the unique nested weight.
    package = {"id": "ace_step_turbo_bf16", "format": "gguf", "files": []}
    resolved = AudioModelInstaller._resolve_installed_model_path(package, str(root))
    assert resolved == str(root.resolve())
    link = root / "model.gguf"
    assert link.is_file()
    assert link.resolve() == gguf.resolve()


def test_resolve_installed_model_path_uses_declared_gguf_when_ambiguous(tmp_path):
    root = tmp_path / "ACE-Step1.5-GGUF"
    (root / "turbo").mkdir(parents=True)
    (root / "base").mkdir(parents=True)
    turbo = root / "turbo" / "ace-step-1.5-turbo-bf16.gguf"
    base = root / "base" / "ace-step-1.5-base-bf16.gguf"
    turbo.write_bytes(b"GGUF")
    base.write_bytes(b"GGUF")
    package = {
        "id": "ace_step_turbo_bf16",
        "format": "gguf",
        "strip_prefix": "ACE-Step1.5-GGUF",
        "files": ["ACE-Step1.5-GGUF/turbo/ace-step-1.5-turbo-bf16.gguf"],
    }
    resolved = AudioModelInstaller._resolve_installed_model_path(package, str(root))
    assert resolved == str(root.resolve())
    assert (root / "model.gguf").resolve() == turbo.resolve()


def test_select_package_gguf_errors_when_ambiguous_without_declaration(tmp_path):
    root = tmp_path / "multi"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    (root / "a" / "one.gguf").write_bytes(b"GGUF")
    (root / "b" / "two.gguf").write_bytes(b"GGUF")
    package = {"id": "multi_gguf", "format": "gguf", "files": []}
    with pytest.raises(RuntimeError, match="Ambiguous GGUF package"):
        AudioModelInstaller._select_package_gguf(package, str(root))


def test_resolve_installed_model_path_keeps_directory_without_gguf(tmp_path):
    root = tmp_path / "Ace-Step1.5"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    package = {"id": "ace_step", "files": []}
    assert AudioModelInstaller._resolve_installed_model_path(package, str(root)) == str(
        root.resolve()
    )


def test_resolve_installed_model_path_keeps_top_level_gguf_directory(tmp_path):
    root = tmp_path / "Qwen3-ASR-0.6B-GGUF"
    root.mkdir()
    gguf = root / "qwen3-asr-0.6b-q8_0.gguf"
    gguf.write_bytes(b"GGUF")
    package = {"id": "qwen3_asr_q8", "format": "gguf", "files": []}
    resolved = AudioModelInstaller._resolve_installed_model_path(package, str(root))
    assert resolved == str(root.resolve())
    # Convenience model.gguf link is fine; directory remains the runtime path.
    assert (root / "model.gguf").is_file() or gguf.is_file()


def test_family_from_bundle_reads_model_type(tmp_path):
    bundle = tmp_path / "Qwen3-ASR-0.6B"
    bundle.mkdir()
    (bundle / "config.json").write_text(
        '{"model_type": "qwen3_asr", "architectures": ["Qwen3ASRForConditionalGeneration"]}',
        encoding="utf-8",
    )
    assert AudioModelInstaller._family_from_bundle(str(bundle)) == "qwen3_asr"


def test_resolve_inspect_family_prefers_hint_then_bundle(tmp_path, monkeypatch):
    installer, _ = _installer(tmp_path, monkeypatch)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text(
        '{"model_type": "qwen3_asr"}', encoding="utf-8"
    )
    assert (
        installer._resolve_inspect_family(
            package={"id": "Qwen3-ASR-0.6B"},
            family_hint="nemotron_asr",
            model_path=str(bundle),
        )
        == "nemotron_asr"
    )
    assert (
        installer._resolve_inspect_family(
            package={"id": "Qwen3-ASR-0.6B"},
            family_hint=None,
            model_path=str(bundle),
        )
        == "qwen3_asr"
    )


@pytest.mark.asyncio
async def test_install_package_passes_resolved_family_to_inspect(tmp_path, monkeypatch):
    installer, store = _installer(tmp_path, monkeypatch)
    package = {
        "id": "Qwen3-ASR-0.6B",
        "display_name": "Qwen3 ASR",
        "target_directory": ".",
        "installable": True,
        "install_kind": "direct",
        "family": "qwen3_asr",
        "required_files": ["config.json"],
        "source": {"kind": "huggingface_snapshot", "repo_id": "demo/qwen"},
    }
    monkeypatch.setattr(installer, "package_metadata", lambda *a, **k: package)

    async def fake_download(task_id, pkg, staging_root, active):
        model_dir = os.path.join(staging_root, "model")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
            handle.write('{"model_type": "qwen3_asr"}')
        return model_dir

    seen = {}

    async def fake_inspect(task_id, active, model_path, family):
        seen["family"] = family
        seen["model_path"] = model_path
        return {
            "family": "qwen3_asr",
            "task_names": ["asr"],
            "tasks": [{"task": "asr", "modes": ["offline"]}],
            "capabilities": {},
        }

    monkeypatch.setattr(installer, "_download_direct", fake_download)
    monkeypatch.setattr(installer, "_inspect", fake_inspect)

    record = await installer.install_package("install-qwen", "Qwen3-ASR-0.6B")
    assert seen["family"] == "qwen3_asr"
    assert record["family"] == "qwen3_asr"


@pytest.mark.asyncio
async def test_pocket_tts_install_does_not_seed_missing_session_voice(
    tmp_path, monkeypatch
):
    installer, store = _installer(tmp_path, monkeypatch)
    package = {
        "id": "pocket-tts",
        "display_name": "PocketTTS",
        "target_directory": ".",
        "installable": True,
        "install_kind": "direct",
        "family": "pocket_tts",
        "required_files": ["config.json"],
        "source": {"kind": "huggingface_snapshot", "repo_id": "kyutai/pocket-tts"},
    }
    monkeypatch.setattr(installer, "package_metadata", lambda *a, **k: package)

    async def fake_download(task_id, pkg, staging_root, active):
        model_dir = os.path.join(staging_root, "model")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        return model_dir

    async def fake_inspect(*_a, **_k):
        return {
            "family": "pocket_tts",
            "task_names": ["tts"],
            "tasks": [{"task": "tts", "modes": ["offline"]}],
            "capabilities": {},
        }

    monkeypatch.setattr(installer, "_download_direct", fake_download)
    monkeypatch.setattr(installer, "_inspect", fake_inspect)

    record = await installer.install_package("install-pocket", "pocket-tts")
    audio = record["config"]["engines"]["audio_cpp"]
    assert "voice_presets" not in audio
    assert "default_voice_preset" not in audio


@pytest.mark.asyncio
async def test_pocket_tts_install_seeds_discovered_session_voice(tmp_path, monkeypatch):
    installer, store = _installer(tmp_path, monkeypatch)
    package = {
        "id": "pocket-tts",
        "display_name": "PocketTTS",
        "target_directory": ".",
        "installable": True,
        "install_kind": "direct",
        "family": "pocket_tts",
        "required_files": ["config.json"],
        "source": {"kind": "huggingface_snapshot", "repo_id": "kyutai/pocket-tts"},
    }
    monkeypatch.setattr(installer, "package_metadata", lambda *a, **k: package)

    async def fake_download(task_id, pkg, staging_root, active):
        model_dir = os.path.join(staging_root, "model")
        embeddings = os.path.join(model_dir, "embeddings")
        os.makedirs(embeddings, exist_ok=True)
        with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        with open(os.path.join(embeddings, "cosette.safetensors"), "wb") as handle:
            handle.write(b"x")
        with open(os.path.join(embeddings, "alba.safetensors"), "wb") as handle:
            handle.write(b"x")
        return model_dir

    async def fake_inspect(*_a, **_k):
        return {
            "family": "pocket_tts",
            "task_names": ["tts"],
            "tasks": [{"task": "tts", "modes": ["offline"]}],
            "capabilities": {},
        }

    monkeypatch.setattr(installer, "_download_direct", fake_download)
    monkeypatch.setattr(installer, "_inspect", fake_inspect)

    record = await installer.install_package("install-pocket-voices", "pocket-tts")
    audio = record["config"]["engines"]["audio_cpp"]
    assert audio["default_voice_preset"] == "alba"
    assert audio["voice_presets"]["alba"]["voice_id"] == "alba"
    assert record["manifest"]["inspection"]["packaged_voices"] == ["alba", "cosette"]


@pytest.mark.asyncio
async def test_pocket_tts_install_seeds_only_discovered_voice(tmp_path, monkeypatch):
    installer, _store = _installer(tmp_path, monkeypatch)
    package = {
        "id": "pocket-tts",
        "display_name": "PocketTTS",
        "target_directory": ".",
        "installable": True,
        "install_kind": "direct",
        "family": "pocket_tts",
        "required_files": ["config.json"],
        "source": {"kind": "huggingface_snapshot", "repo_id": "kyutai/pocket-tts"},
    }
    monkeypatch.setattr(installer, "package_metadata", lambda *a, **k: package)

    async def fake_download(task_id, pkg, staging_root, active):
        model_dir = os.path.join(staging_root, "model")
        embeddings = os.path.join(model_dir, "embeddings")
        os.makedirs(embeddings, exist_ok=True)
        with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        with open(os.path.join(embeddings, "cosette.safetensors"), "wb") as handle:
            handle.write(b"x")
        return model_dir

    async def fake_inspect(*_a, **_k):
        return {
            "family": "pocket_tts",
            "task_names": ["tts"],
            "tasks": [{"task": "tts", "modes": ["offline"]}],
            "capabilities": {},
        }

    monkeypatch.setattr(installer, "_download_direct", fake_download)
    monkeypatch.setattr(installer, "_inspect", fake_inspect)

    record = await installer.install_package("install-pocket-cosette", "pocket-tts")
    audio = record["config"]["engines"]["audio_cpp"]
    assert audio["default_voice_preset"] == "cosette"
    assert audio["voice_presets"]["cosette"]["voice_id"] == "cosette"


def test_install_method_contract_covers_direct_composite_and_converter():
    assert _source_kind_method({"kind": "huggingface_snapshot"}) == "direct"
    assert _source_kind_method({"kind": "composite_snapshot"}) == "composite"
    assert _source_kind_method({"kind": "composite"}) == "converter"
    assert _source_kind_method({"kind": "utility"}) == "converter"
    assert _source_kind_method({"kind": "bundled_asset"}) == "bundled"
    assert _source_kind_method({"kind": "external"}) == "unavailable"
    # Post-processed SnapshotSource packages must use model_manager, not direct HF.
    assert (
        _source_kind_method(
            {
                "id": "voxcpm2",
                "install_kind": "composite",
                "source": {"kind": "huggingface_snapshot", "repo_id": "OpenBMB/VoxCPM2"},
            }
        )
        == "composite"
    )
    assert (
        _source_kind_method(
            {
                "id": "vibevoice_asr",
                "install_kind": "composite",
                "source": {"kind": "composite_snapshot"},
            }
        )
        == "composite"
    )
    assert (
        _source_kind_method(
            {
                "id": "citrinet_asr",
                "install_kind": "composite",
                "source": {"kind": "composite", "operation_kind": "nemo_archive"},
            }
        )
        == "converter"
    )


@pytest.mark.asyncio
async def test_bundled_package_install_copies_asset(tmp_path, monkeypatch):
    installer, store = _installer(tmp_path, monkeypatch)
    asset = tmp_path / "assets" / "silero_vad"
    asset.mkdir(parents=True)
    (asset / "silero_vad_16k.safetensors").write_bytes(b"weights")
    package = {
        "id": "silero_vad",
        "display_name": "Silero VAD",
        "target_directory": "silero_vad",
        "installable": True,
        "install_kind": "bundled",
        "family": "silero_vad",
        "required_files": ["silero_vad_16k.safetensors"],
        "source": {"kind": "bundled_asset", "path": str(asset)},
    }
    monkeypatch.setattr(installer, "package_metadata", lambda *a, **k: package)

    async def fake_inspect(task_id, active, model_path, family):
        return {
            "family": "silero_vad",
            "task_names": ["vad"],
            "tasks": [{"task": "vad", "modes": ["offline"]}],
            "capabilities": {},
        }

    monkeypatch.setattr(installer, "_inspect", fake_inspect)
    record = await installer.install_package("install-silero", "silero_vad")
    assert record["family"] == "silero_vad"
    assert record["manifest"]["install_method"] == "bundled"
    assert os.path.isfile(
        os.path.join(record["bundle_path"], "silero_vad", "silero_vad_16k.safetensors")
    )


@pytest.mark.asyncio
async def test_v2_direct_package_install_uses_model_manager_v2(tmp_path, monkeypatch):
    installer, store = _installer(tmp_path, monkeypatch)
    package = {
        "id": "qwen3_tts_q8",
        "display_name": "Qwen3 TTS Q8",
        "target_directory": "qwen3_tts",
        "installable": True,
        "install_kind": "snapshot",
        "manager_backend": "v2",
        "required_files": ["config.json"],
        "source": {
            "kind": "huggingface_snapshot",
            "repo_id": "audio-cpp/audio.cpp-gguf",
        },
    }
    monkeypatch.setattr(installer, "package_metadata", lambda *_a, **_k: package)
    manager_calls = []
    direct_calls = []

    async def fake_manager(task_id, pkg, staging_root, active, options, **kwargs):
        manager_calls.append((pkg["id"], kwargs.get("require_helper_venv")))
        target = os.path.join(staging_root, pkg["target_directory"])
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "config.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        return target

    async def fake_direct(*_a, **_k):
        direct_calls.append(True)
        raise AssertionError("v2 packages must use model_manager_v2 install")

    async def fake_inspect(*_a, **_k):
        return {
            "family": "qwen3_tts",
            "task_names": ["tts"],
            "tasks": [{"task": "tts", "modes": ["offline"]}],
            "capabilities": {},
        }

    monkeypatch.setattr(installer, "_install_with_manager", fake_manager)
    monkeypatch.setattr(installer, "_download_direct", fake_direct)
    monkeypatch.setattr(installer, "_inspect", fake_inspect)
    monkeypatch.setattr(
        "backend.llama_swap_manager.mark_swap_config_stale",
        lambda: None,
    )

    record = await installer.install_package("task-v2", "qwen3_tts_q8")
    assert manager_calls == [("qwen3_tts_q8", False)]
    assert direct_calls == []
    assert record["id"] == "audio-cpp--qwen3_tts_q8"
    assert store.get_model(record["id"]) is record


@pytest.mark.asyncio
async def test_composite_package_install_uses_model_manager(tmp_path, monkeypatch):
    installer, store = _installer(tmp_path, monkeypatch)
    package = {
        "id": "vibevoice_asr",
        "display_name": "VibeVoice ASR",
        "target_directory": "VibeVoice-ASR",
        "installable": True,
        "install_kind": "composite",
        "required_files": ["config.json"],
        "source": {"kind": "composite_snapshot", "placements": []},
    }
    monkeypatch.setattr(installer, "package_metadata", lambda *_a, **_k: package)
    manager_calls = []
    direct_calls = []

    async def fake_manager(task_id, pkg, staging_root, active, options):
        manager_calls.append(pkg["id"])
        target = os.path.join(staging_root, pkg["target_directory"])
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "config.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        return target

    async def fake_direct(*_a, **_k):
        direct_calls.append(True)
        raise AssertionError("composite packages must not use direct HF download")

    async def fake_inspect(*_a, **_k):
        return {
            "family": "vibevoice_asr",
            "task_names": ["asr"],
            "tasks": [{"task": "asr", "modes": ["offline"]}],
            "capabilities": {},
        }

    monkeypatch.setattr(installer, "_install_with_manager", fake_manager)
    monkeypatch.setattr(installer, "_download_direct", fake_direct)
    monkeypatch.setattr(installer, "_inspect", fake_inspect)
    monkeypatch.setattr(
        "backend.llama_swap_manager.mark_swap_config_stale",
        lambda: None,
    )

    record = await installer.install_package("task-1", "vibevoice_asr")
    assert manager_calls == ["vibevoice_asr"]
    assert direct_calls == []
    assert record["id"] == "audio-cpp--vibevoice_asr"
    assert store.get_model(record["id"]) is record


@pytest.mark.asyncio
async def test_local_import_validates_then_atomically_promotes_bundle(
    tmp_path, monkeypatch
):
    installer, store = _installer(tmp_path, monkeypatch)
    stale_calls = []
    monkeypatch.setattr(
        "backend.llama_swap_manager.mark_swap_config_stale",
        lambda: stale_calls.append(True),
    )
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
    assert stale_calls == [True]


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
