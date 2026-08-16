"""Transactional prepared-bundle installs and local imports for audio.cpp."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from huggingface_hub import HfApi

from backend.audio_cpp_artifact import build_artifact_descriptor
from backend.audio_cpp_voices import attach_packaged_voices
from backend.audio_tts_profiles import family_requires_session_voice
from backend.audio_voice_presets import seed_session_voice_from_ids
from backend.audio_cpp_inspect import (
    audio_cpp_inspect_env,
    build_audio_cpp_inspect_argv,
    inspect_command_variants,
    select_inspect_payload,
)
from backend.audio_cpp_manager import get_audio_cpp_manager
from backend.data_store import generate_proxy_name, get_store
from backend.engine_param_scanner import scan_audio_cpp_model_profile
from backend.feature_flags import audio_cpp_enabled
from backend.huggingface import (
    download_model_with_progress,
    get_huggingface_token,
)
from backend.logging_config import get_logger
from backend.model_catalog.audio_cpp_provider import (
    AudioCppCatalogProvider,
    resolve_studio_install_method,
)
from backend.model_catalog.base import modalities_for_tasks
from backend.model_config import normalize_model_config
from backend.progress_manager import get_progress_manager
from backend.task_cancel_registry import (
    TaskCancelledError,
    is_task_cancel_requested,
    register_task_cancel,
    unregister_task_cancel,
)
from backend.utils.fs_ops import robust_rmtree


logger = get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_id(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    return re.sub(r"-{2,}", "-", result).strip("-._")[:96] or "audio-model"


def _directory_size_and_files(root: str) -> tuple[int, List[dict]]:
    total = 0
    files: List[dict] = []
    base = os.path.realpath(root)
    for directory, _, filenames in os.walk(base):
        for filename in sorted(filenames):
            path = os.path.join(directory, filename)
            if os.path.islink(path) and not os.path.exists(path):
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            relative = os.path.relpath(path, base)
            total += size
            files.append({"path": relative, "size": size})
    return total, files


def _copy_or_link(source: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    real_source = os.path.realpath(source)
    try:
        os.link(real_source, destination)
    except OSError:
        shutil.copy2(real_source, destination)


def _source_kind_method(package_or_source: dict) -> str:
    """Resolve Studio install method from a package dict or a bare source dict."""
    payload = package_or_source if isinstance(package_or_source, dict) else {}
    if "source" in payload or "id" in payload or "install_kind" in payload:
        return resolve_studio_install_method(payload)
    return resolve_studio_install_method({"source": payload})


class _BundleProgressAdapter:
    def __init__(
        self,
        task_id: str,
        package_id: str,
        base_bytes: int,
        total_bytes: int,
        file_index: int,
        file_count: int,
    ):
        self.task_id = task_id
        self.package_id = package_id
        self.base_bytes = base_bytes
        self.total_bytes = total_bytes
        self.file_index = file_index
        self.file_count = file_count
        self.pm = get_progress_manager()

    async def send_download_progress(self, **payload):
        current = int(payload.get("bytes_downloaded") or 0)
        aggregate = self.base_bytes + current
        progress = (
            min(84, int(5 + (aggregate / self.total_bytes) * 79))
            if self.total_bytes
            else 10
        )
        message = payload.get("message") or "Downloading audio package"
        self.pm.update_task(
            self.task_id,
            progress=progress,
            message=message,
            metadata_update={
                "stage": "download",
                "files_completed": self.file_index,
                "files_total": self.file_count,
                "bytes_downloaded": aggregate,
                "total_bytes": self.total_bytes,
            },
        )
        self.pm.emit(
            "download_progress",
            {
                **payload,
                "task_id": self.task_id,
                "model_format": "audio_cpp_bundle",
                "huggingface_id": self.package_id,
                "bytes_downloaded": aggregate,
                "total_bytes": self.total_bytes,
                "progress": progress,
                "files_completed": self.file_index,
                "files_total": self.file_count,
            },
        )


class AudioModelInstaller:
    def __init__(self, store=None):
        self.store = store or get_store()
        self.manager = get_audio_cpp_manager()
        self.pm = get_progress_manager()
        self._active_processes: Dict[str, asyncio.subprocess.Process] = {}
        self._helper_lock = asyncio.Lock()

    def _active_version(self) -> dict:
        if not audio_cpp_enabled():
            raise RuntimeError(
                "The audio.cpp integration is disabled by AUDIO_CPP_ENABLED"
            )
        active = self.store.get_active_engine_version("audio_cpp")
        if not active:
            raise RuntimeError("Install and activate audio.cpp first")
        cli_path = str(active.get("cli_binary_path") or "")
        if not cli_path or not os.path.isfile(cli_path):
            raise RuntimeError("Active audio.cpp version is missing cli_binary_path")
        from backend.audio_cpp_model_managers import resolve_model_manager_path

        manager_path = resolve_model_manager_path(version_row=active)
        if not manager_path:
            raise RuntimeError(
                "Active audio.cpp version is missing model_manager_v2.py "
                "and a legacy model_manager*.py"
            )
        # Keep row usable for callers that still read model_manager_path.
        if not active.get("model_manager_path") or not os.path.isfile(
            str(active.get("model_manager_path") or "")
        ):
            active = {**active, "model_manager_path": manager_path}
        return active

    def _packages(self, active: dict) -> List[dict]:
        provider = AudioCppCatalogProvider(self.store)
        return provider._manager_packages(active)

    def package_metadata(self, package_id: str, active: Optional[dict] = None) -> dict:
        active = active or self._active_version()
        package = next(
            (
                item
                for item in self._packages(active)
                if str(item.get("id")) == str(package_id)
            ),
            None,
        )
        if not package:
            raise ValueError(f"Unknown audio.cpp package: {package_id}")
        return package

    @staticmethod
    def _family_from_bundle(model_path: str) -> Optional[str]:
        """Best-effort family from prepared-bundle config.json (e.g. qwen3_asr)."""
        config_path = os.path.join(str(model_path or ""), "config.json")
        if not os.path.isfile(config_path):
            return None
        try:
            with open(config_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        for key in ("model_type", "family"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
        architectures = payload.get("architectures")
        if not isinstance(architectures, list) or not architectures:
            return None
        arch = str(architectures[0] or "").strip()
        if not arch:
            return None
        # Qwen3ASRForConditionalGeneration → qwen3_asr
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", arch).lower()
        for suffix in (
            "_for_conditional_generation",
            "_for_causal_lm",
            "_model",
        ):
            if snake.endswith(suffix):
                snake = snake[: -len(suffix)]
                break
        return snake or None

    def _resolve_inspect_family(
        self,
        *,
        package: Optional[dict],
        family_hint: Optional[str],
        model_path: str,
    ) -> Optional[str]:
        package = package if isinstance(package, dict) else {}
        candidates = [
            family_hint,
            package.get("family"),
            (package.get("source") or {}).get("family")
            if isinstance(package.get("source"), dict)
            else None,
            self._family_from_bundle(model_path),
        ]
        package_id = str(package.get("id") or "").strip()
        if package_id:
            try:
                inferred = AudioCppCatalogProvider(self.store)._infer_metadata(package_id)
                candidates.append((inferred or {}).get("family"))
            except Exception:
                pass
        for raw in candidates:
            value = str(raw or "").strip()
            if value:
                return value
        return None

    async def _run_process(
        self,
        task_id: str,
        argv: List[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        stage: str,
        start_progress: int,
        check: bool = True,
    ) -> List[str]:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=(os.name != "nt"),
        )
        self._active_processes[task_id] = process
        lines: List[str] = []
        try:
            assert process.stdout is not None
            while True:
                if is_task_cancel_requested(task_id):
                    await self._terminate_process(process)
                    raise TaskCancelledError("Audio model installation cancelled")
                try:
                    chunk = await asyncio.wait_for(process.stdout.readline(), 0.25)
                except asyncio.TimeoutError:
                    if process.returncode is not None:
                        break
                    continue
                if not chunk:
                    break
                line = chunk.decode("utf-8", errors="replace").rstrip()
                if line:
                    lines.append(line)
                    self.pm.update_task(
                        task_id,
                        progress=start_progress,
                        message=line,
                        metadata_update={
                            "stage": stage,
                            "log_lines": lines[-100:],
                        },
                    )
                    self.pm.emit(
                        "task_log",
                        {
                            "task_id": task_id,
                            "line": line,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
            returncode = await process.wait()
            if check and returncode != 0:
                detail = "\n".join(lines[-40:])
                raise RuntimeError(
                    f"{stage} failed with exit code {returncode}"
                    + (f":\n{detail}" if detail else "")
                )
            return lines
        finally:
            self._active_processes.pop(task_id, None)

    @staticmethod
    async def _terminate_process(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except (ProcessLookupError, PermissionError):
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except (ProcessLookupError, PermissionError):
                pass
            await process.wait()

    async def ensure_helper_environment(self, task_id: str, active: dict) -> str:
        async with self._helper_lock:
            existing = str(active.get("helper_venv_path") or "")
            python_name = "python.exe" if os.name == "nt" else "python"
            python_subdir = "Scripts" if os.name == "nt" else "bin"
            if existing and os.path.isfile(
                os.path.join(existing, python_subdir, python_name)
            ):
                return existing

            venv_path = os.path.join(self.manager.tools_dir, "model-manager-venv")
            python_path = os.path.join(venv_path, python_subdir, python_name)
            ready_marker = os.path.join(venv_path, ".studio-ready")
            if os.path.isfile(python_path) and os.path.isfile(ready_marker):
                self.store.update_engine_version(
                    "audio_cpp",
                    str(active.get("version")),
                    {
                        "helper_venv_path": venv_path,
                        "helper_environment_status": "ready",
                    },
                )
                return venv_path
            self.pm.update_task(
                task_id,
                progress=5,
                message="Preparing isolated audio.cpp model-manager environment",
                metadata_update={"stage": "helper_environment"},
            )
            if not os.path.isfile(python_path):
                if os.path.isdir(venv_path):
                    robust_rmtree(venv_path)
                await self._run_process(
                    task_id,
                    [sys.executable, "-m", "venv", venv_path],
                    stage="helper_environment",
                    start_progress=6,
                )
            pip_path = os.path.join(
                venv_path,
                python_subdir,
                "pip.exe" if os.name == "nt" else "pip",
            )
            await self._run_process(
                task_id,
                [pip_path, "install", "--upgrade", "pip"],
                stage="helper_environment",
                start_progress=8,
            )
            await self._run_process(
                task_id,
                [
                    pip_path,
                    "install",
                    "torch",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ],
                stage="helper_environment",
                start_progress=10,
            )
            await self._run_process(
                task_id,
                [pip_path, "install", "safetensors", "pyyaml"],
                stage="helper_environment",
                start_progress=12,
            )
            Path(ready_marker).write_text(_utcnow() + "\n", encoding="utf-8")
            self.store.update_engine_version(
                "audio_cpp",
                str(active.get("version")),
                {
                    "helper_venv_path": venv_path,
                    "helper_environment_status": "ready",
                    "helper_environment_updated_at": _utcnow(),
                },
            )
            return venv_path

    async def _install_bundled_asset(
        self,
        task_id: str,
        package: dict,
        staging_root: str,
        active: dict,
    ) -> str:
        """Copy a source-tree asset (e.g. silero_vad) into the models staging dir."""
        source = package.get("source") if isinstance(package.get("source"), dict) else {}
        asset_path = str(source.get("path") or "").strip()
        if not asset_path or not os.path.isdir(asset_path):
            relative = str(source.get("relative_path") or "").strip()
            source_root = str(active.get("source_path") or "").strip()
            if relative and source_root:
                asset_path = os.path.join(source_root, relative)
        if not asset_path or not os.path.isdir(asset_path):
            raise FileNotFoundError(
                f"Bundled asset for '{package.get('id')}' was not found under the "
                "active audio.cpp source tree."
            )
        target_name = str(package.get("target_directory") or package.get("id") or "model")
        destination = os.path.join(staging_root, target_name)
        self.pm.update_task(
            task_id,
            progress=10,
            message=f"Copying bundled asset {package.get('id')}",
            metadata_update={"stage": "bundled_copy", "install_method": "bundled"},
        )
        await self._copy_local_bundle(task_id, asset_path, destination)
        return destination

    async def _copy_local_bundle(
        self, task_id: str, source_root: str, destination_root: str
    ) -> None:
        if is_task_cancel_requested(task_id):
            raise TaskCancelledError("Audio model import cancelled")
        files: List[tuple[str, str, int]] = []
        total_bytes = 0
        for directory, dirnames, filenames in os.walk(source_root):
            for dirname in list(dirnames):
                path = os.path.join(directory, dirname)
                if os.path.islink(path):
                    raise ValueError(
                        f"Local bundle contains an unsupported directory symlink: {path}"
                    )
            relative_directory = os.path.relpath(directory, source_root)
            target_directory = (
                destination_root
                if relative_directory == "."
                else os.path.join(destination_root, relative_directory)
            )
            os.makedirs(target_directory, exist_ok=True)
            for filename in filenames:
                source = os.path.join(directory, filename)
                if os.path.islink(source):
                    raise ValueError(
                        f"Local bundle contains an unsupported file symlink: {source}"
                    )
                size = os.path.getsize(source)
                relative = os.path.relpath(source, source_root)
                files.append((source, os.path.join(destination_root, relative), size))
                total_bytes += size

        copied = 0
        try:
            for source, destination, size in files:
                if is_task_cancel_requested(task_id):
                    raise TaskCancelledError("Audio model import cancelled")
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                with open(source, "rb") as input_handle, open(
                    destination, "xb"
                ) as output_handle:
                    while True:
                        if is_task_cancel_requested(task_id):
                            raise TaskCancelledError("Audio model import cancelled")
                        chunk = input_handle.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output_handle.write(chunk)
                        copied += len(chunk)
                        progress = (
                            min(82, int(10 + (copied / total_bytes) * 72))
                            if total_bytes
                            else 82
                        )
                        self.pm.update_task(
                            task_id,
                            progress=progress,
                            message=f"Copying {os.path.basename(source)}",
                            metadata_update={
                                "stage": "copy",
                                "bytes_copied": copied,
                                "total_bytes": total_bytes,
                            },
                        )
                        await asyncio.sleep(0)
                try:
                    shutil.copystat(source, destination)
                except OSError:
                    pass
                copied += max(0, size - os.path.getsize(destination))
        except TaskCancelledError:
            if os.path.isdir(destination_root):
                robust_rmtree(destination_root)
            raise

    async def _download_direct(
        self,
        task_id: str,
        package: dict,
        staging_root: str,
        active: dict,
    ) -> str:
        source = package.get("source") or {}
        repo_id = str(source.get("repo_id") or "")
        revision = str(source.get("revision") or "main")
        if not repo_id:
            raise RuntimeError("Direct package source has no Hugging Face repo_id")
        include = [str(item) for item in source.get("include_prefixes") or []]
        exclude = [str(item) for item in source.get("exclude_prefixes") or []]
        token = get_huggingface_token()
        api = HfApi(token=token or None)
        info = await asyncio.to_thread(
            api.repo_info,
            repo_id=repo_id,
            revision=revision,
            files_metadata=True,
        )
        entries = []
        for sibling in getattr(info, "siblings", []) or []:
            filename = getattr(sibling, "rfilename", None)
            if not filename:
                continue
            if include and not any(filename.startswith(prefix) for prefix in include):
                continue
            if any(filename.startswith(prefix) for prefix in exclude):
                continue
            entries.append((filename, int(getattr(sibling, "size", 0) or 0)))
        if not entries:
            raise RuntimeError(f"No installable files found in {repo_id}@{revision}")

        target_directory = str(package.get("target_directory") or package["id"])
        package_root = os.path.join(staging_root, target_directory)
        os.makedirs(package_root, exist_ok=True)
        total_bytes = sum(size for _, size in entries)
        downloaded = 0
        for index, (filename, size) in enumerate(entries):
            if is_task_cancel_requested(task_id):
                raise TaskCancelledError("Audio model installation cancelled")
            adapter = _BundleProgressAdapter(
                task_id,
                str(package["id"]),
                downloaded,
                total_bytes,
                index,
                len(entries),
            )
            cache_path, actual_size = await download_model_with_progress(
                repo_id,
                filename,
                adapter,
                task_id,
                total_bytes=size,
                model_format="audio_cpp_bundle",
                huggingface_id_for_progress=str(package["id"]),
                revision=revision,
                token=token or None,
            )
            _copy_or_link(cache_path, os.path.join(package_root, filename))
            downloaded += actual_size
        return package_root

    async def _install_with_manager(
        self,
        task_id: str,
        package: dict,
        staging_root: str,
        active: dict,
        options: dict,
        *,
        manager_path: Optional[str] = None,
        require_helper_venv: bool = True,
    ) -> str:
        from backend.audio_cpp_model_managers import (
            manager_script_kind,
            resolve_model_manager_legacy_path,
            resolve_model_manager_path,
            resolve_model_manager_v2_path,
        )

        backend = str(package.get("manager_backend") or "").strip().lower()
        if not manager_path:
            if backend == "v2":
                manager_path = resolve_model_manager_v2_path(version_row=active)
            elif backend == "legacy" or _source_kind_method(package) in {
                "composite",
                "converter",
            }:
                manager_path = resolve_model_manager_legacy_path(version_row=active)
            else:
                manager_path = resolve_model_manager_path(version_row=active)
        manager_path = str(manager_path or "")
        if not manager_path or not os.path.isfile(manager_path):
            raise RuntimeError(
                "No usable audio.cpp model manager script found for this package"
            )

        kind = manager_script_kind(manager_path)
        if require_helper_venv and kind != "v2":
            venv = await self.ensure_helper_environment(task_id, active)
            python_path = os.path.join(
                venv,
                "Scripts" if os.name == "nt" else "bin",
                "python.exe" if os.name == "nt" else "python",
            )
        else:
            # v2 only needs stdlib + network; prefer helper python when present.
            from backend.model_catalog.audio_cpp_provider import _manager_python

            python_path = _manager_python(active)

        # Managers resolve assets / model_specs from the audio.cpp repo root
        # (parent of tools/), not from the tools/ directory itself.
        manager_cwd = os.path.dirname(os.path.dirname(manager_path))
        if not os.path.isdir(manager_cwd):
            manager_cwd = os.path.dirname(manager_path)
        argv = [
            python_path,
            manager_path,
            "install",
            str(package["id"]),
            "--models-root",
            staging_root,
        ]
        argument_map = {
            "source_file": "--source-file",
            "source_dir": "--source-dir",
            "output_file": "--output-file",
            "variant": "--variant",
        }
        if kind != "v2":
            for key, flag in argument_map.items():
                value = options.get(key)
                if value not in (None, ""):
                    if key in {"source_file", "source_dir"} and not os.path.exists(
                        str(value)
                    ):
                        raise ValueError(f"{key} does not exist: {value}")
                    argv.extend([flag, str(value)])
        env = os.environ.copy()
        token = get_huggingface_token()
        if token:
            env["HF_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token
        env["PYTHONUNBUFFERED"] = "1"
        script_name = os.path.basename(manager_path)
        self.pm.update_task(
            task_id,
            progress=44,
            message=f"Running {script_name} install {package['id']}",
            metadata_update={
                "stage": "download_convert",
                "install_method": _source_kind_method(package),
                "uses_model_manager": kind != "v2",
                "manager_backend": kind,
                "manager_script": script_name,
            },
        )
        try:
            await self._run_process(
                task_id,
                argv,
                cwd=manager_cwd,
                env=env,
                stage="download_convert",
                start_progress=45,
            )
        except RuntimeError as exc:
            if kind == "v2":
                hint = (
                    "Spec-backed installs use model_manager_v2.py. Gated Hugging Face "
                    "repos need a valid HF token with accepted access."
                )
            else:
                hint = (
                    "Assemble/convert packages require the audio.cpp model_manager helper "
                    "(Torch + safetensors). Gated Hugging Face repos also need a valid HF token."
                )
            raise RuntimeError(f"{exc} ({hint})") from exc
        target = os.path.join(
            staging_root,
            str(package.get("target_directory") or package["id"]),
        )
        if not os.path.exists(target):
            raise RuntimeError(
                f"{script_name} completed but target directory was not created: {target}"
            )
        return target

    async def _inspect(
        self,
        task_id: str,
        active: dict,
        model_path: str,
        family: Optional[str],
    ) -> dict:
        cli_path = str(active["cli_binary_path"])
        from backend.engine_param_scanner import (
            _audio_cpp_model_spec_override,
            _audio_cpp_workdir,
        )

        workdir = _audio_cpp_workdir(active, cli_path)
        spec_override = _audio_cpp_model_spec_override(active, cli_path)
        base_argv = build_audio_cpp_inspect_argv(
            cli_path,
            model_path,
            family=family,
            model_spec_override=spec_override,
        )
        json_argv, text_argv = inspect_command_variants(base_argv)
        env = {
            **os.environ,
            **audio_cpp_inspect_env(
                cli_path,
                source_path=str(active.get("source_path") or ""),
            ),
        }
        json_lines = await self._run_process(
            task_id,
            json_argv,
            cwd=workdir,
            env=env,
            stage="inspect",
            start_progress=90,
            check=False,
        )
        text_lines: List[str] = []
        _, inspection = select_inspect_payload("\n".join(json_lines), "")
        if not inspection.get("family") or not inspection.get("task_names"):
            text_lines = await self._run_process(
                task_id,
                text_argv,
                cwd=workdir,
                env=env,
                stage="inspect",
                start_progress=92,
            )
            _, inspection = select_inspect_payload(
                "\n".join(json_lines),
                "\n".join(text_lines),
            )
        if not inspection.get("family") or not inspection.get("task_names"):
            raise RuntimeError(
                "audio.cpp inspection did not identify a valid family and task"
            )
        return inspection

    @staticmethod
    def _check_required_files(package: dict, model_path: str) -> None:
        missing = [
            relative
            for relative in package.get("required_files") or []
            if not os.path.exists(os.path.join(model_path, str(relative)))
        ]
        if missing:
            raise RuntimeError(f"Installed package is missing required files: {missing}")

    @staticmethod
    def _strip_package_prefix(remote_path: str, strip_prefix: str) -> str:
        remote = str(remote_path or "").replace("\\", "/").lstrip("/")
        prefix = str(strip_prefix or "").replace("\\", "/").strip("/")
        if not prefix:
            return remote
        if remote == prefix:
            return ""
        head = prefix + "/"
        if remote.startswith(head):
            return remote[len(head) :]
        return remote

    @staticmethod
    def _is_under_root(root: str, path: str) -> bool:
        try:
            return os.path.commonpath([root, path]) == root
        except ValueError:
            return False

    @classmethod
    def _declared_gguf_paths(cls, package: dict, package_root: str) -> List[str]:
        """GGUF paths declared by package.files (after strip_prefix), if present on disk."""
        root = os.path.realpath(package_root)
        strip_prefix = str(package.get("strip_prefix") or "")
        found: List[str] = []
        for remote in package.get("files") or []:
            remote_s = str(remote or "")
            if not remote_s.lower().endswith(".gguf"):
                continue
            relative = cls._strip_package_prefix(remote_s, strip_prefix)
            if not relative:
                continue
            candidate = os.path.realpath(os.path.join(root, relative))
            if cls._is_under_root(root, candidate) and os.path.isfile(candidate):
                found.append(candidate)
        return list(dict.fromkeys(found))

    @classmethod
    def _discover_gguf_files(
        cls,
        package_root: str,
        *,
        max_depth: int = 4,
    ) -> List[str]:
        """Find ``*.gguf`` files under a package root (bounded recursive walk)."""
        root = os.path.realpath(package_root)
        if not os.path.isdir(root):
            return []

        found: List[str] = []
        root_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            # Keep the walk shallow and skip cache / VCS noise.
            depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
            if depth >= max_depth:
                dirnames[:] = []
            dirnames[:] = [
                name
                for name in dirnames
                if name not in {".git", ".cache", "__pycache__", ".staging"}
                and not name.startswith(".")
            ]
            for name in filenames:
                if not name.lower().endswith(".gguf"):
                    continue
                path = os.path.realpath(os.path.join(dirpath, name))
                if cls._is_under_root(root, path) and os.path.isfile(path):
                    found.append(path)
        found.sort()
        return found

    @classmethod
    def _select_package_gguf(cls, package: dict, package_root: str) -> Optional[str]:
        """Choose the package's primary GGUF weight, or None for non-GGUF layouts.

        Selection mirrors audio.cpp directory discovery where possible, then falls
        back to package declarations and a unique nested GGUF. Ambiguous GGUF
        packages raise instead of guessing.
        """
        root = os.path.realpath(package_root)
        if not os.path.isdir(root):
            return None

        named = os.path.join(root, "model.gguf")
        if os.path.isfile(named):
            return os.path.realpath(named)

        top_level = [
            os.path.realpath(os.path.join(root, name))
            for name in sorted(os.listdir(root))
            if name.lower().endswith(".gguf")
            and os.path.isfile(os.path.join(root, name))
        ]
        if len(top_level) == 1:
            return top_level[0]

        declared = cls._declared_gguf_paths(package, root)
        if len(declared) == 1:
            return declared[0]

        discovered = cls._discover_gguf_files(root)
        # Ignore a root model.gguf symlink target double-count by uniquing.
        discovered = list(dict.fromkeys(discovered))
        if len(discovered) == 1:
            return discovered[0]

        if len(declared) > 1:
            raise RuntimeError(
                f"Ambiguous GGUF package under {root}: package.files lists "
                f"{len(declared)} GGUF weights. Install a single-weight package "
                "or point audiocpp at one file explicitly."
            )

        expects_gguf = (
            str(package.get("format") or "").lower() == "gguf"
            or any(
                str(item or "").lower().endswith(".gguf")
                for item in (package.get("files") or [])
            )
        )
        if expects_gguf and len(discovered) > 1:
            preview = ", ".join(
                os.path.relpath(path, root) for path in discovered[:6]
            )
            raise RuntimeError(
                f"Ambiguous GGUF package under {root}: found {len(discovered)} "
                f".gguf files ({preview}). audio.cpp only auto-detects a single "
                "root GGUF; fix the package layout or declare exactly one "
                "package.files GGUF entry."
            )
        return None

    @classmethod
    def _ensure_root_model_gguf_link(cls, package_root: str, gguf_path: str) -> bool:
        """Expose a nested GGUF as ``model.gguf`` so audio.cpp directory discovery works.

        Returns True when the package root already has, or successfully gains, a
        root-level ``model.gguf`` usable by ``find_directory_gguf``.
        """
        root = os.path.realpath(package_root)
        target = os.path.realpath(gguf_path)
        if not cls._is_under_root(root, target) or not os.path.isfile(target):
            return False

        link_path = os.path.join(root, "model.gguf")
        if os.path.isfile(link_path):
            return os.path.realpath(link_path) == target

        if os.path.lexists(link_path):
            # Broken / wrong symlink left behind — replace it.
            try:
                os.unlink(link_path)
            except OSError:
                return False

        if os.path.dirname(target) == root:
            # Unique top-level GGUF already discoverable; optional convenience link.
            rel = os.path.basename(target)
        else:
            rel = os.path.relpath(target, root)

        try:
            os.symlink(rel, link_path)
            return os.path.isfile(link_path)
        except OSError:
            # Symlinks unavailable (some Windows/container FS setups): hardlink, then copy.
            try:
                os.link(target, link_path)
                return os.path.isfile(link_path)
            except OSError:
                try:
                    shutil.copy2(target, link_path)
                    return os.path.isfile(link_path)
                except OSError:
                    return False

    @classmethod
    def _resolve_installed_model_path(cls, package: dict, package_root: str) -> str:
        """Return the path Studio should inspect / persist for an installed package.

        For GGUF packages this normalizes nested layouts (e.g. ``turbo/*.gguf``) to
        something audio.cpp can load:
        1. select the primary GGUF (root / declared / unique nested)
        2. publish ``model.gguf`` at the package root when needed
        3. prefer the package directory once discovery works; otherwise the GGUF file
        """
        root = os.path.realpath(package_root)
        if not os.path.isdir(root):
            return package_root

        try:
            gguf = cls._select_package_gguf(package, root)
        except RuntimeError:
            raise
        if not gguf:
            return root

        if cls._ensure_root_model_gguf_link(root, gguf):
            return root
        return gguf

    def _model_record(
        self,
        package: dict,
        final_bundle: str,
        model_path: str,
        inspection: dict,
        method: str,
        active: dict,
    ) -> dict:
        total_size, files = _directory_size_and_files(final_bundle)
        tasks = inspection.get("task_names") or []
        inputs, outputs = modalities_for_tasks(tasks)
        package_id = str(package["id"])
        manifest = {
            "schema_version": 1,
            "package_id": package_id,
            "installed_at": _utcnow(),
            "install_method": method,
            "engine_version": active.get("version"),
            "engine_commit": active.get("source_commit"),
            "source": package.get("source") or {},
            "files": files,
            "size": total_size,
            "family": inspection.get("family"),
            "tasks": inspection.get("tasks") or [],
            "modes": {
                row.get("task"): row.get("modes") or []
                for row in inspection.get("tasks") or []
            },
            "inspection": inspection,
        }
        fingerprint_payload = {
            "package_id": package_id,
            "engine_commit": active.get("source_commit"),
            "files": files,
            "inspection": inspection,
        }
        manifest["capability_fingerprint"] = hashlib.sha256(
            json.dumps(
                fingerprint_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        with open(
            os.path.join(final_bundle, ".studio-manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")

        model_id = f"audio-cpp--{_safe_id(package_id)}"
        primary_task = tasks[0] if tasks else None
        modes = next(
            (
                row.get("modes") or []
                for row in inspection.get("tasks") or []
                if row.get("task") == primary_task
            ),
            [],
        )
        family = inspection.get("family")
        voices = attach_packaged_voices(
            inspection,
            model_path,
            family,
            source_path=str(active.get("source_path") or "") or None,
        )
        audio_engine = {
            "family": family,
            "task": primary_task,
            "mode": "offline" if "offline" in modes else (modes[0] if modes else None),
            "backend": (
                (active.get("build_config") or {}).get("backend")
                or "cpu"
            ),
            "device": 0,
            "threads": max(1, os.cpu_count() or 1),
            "lazy_load": False,
            "load_options": {},
            "session_options": {},
        }
        if family_requires_session_voice(family):
            seed_session_voice_from_ids(audio_engine, voices)
        config = normalize_model_config(
            {
                "engine": "audio_cpp",
                "engines": {"audio_cpp": audio_engine},
            }
        )
        artifact = build_artifact_descriptor(
            bundle_path=final_bundle,
            runtime_path=model_path,
            size=total_size,
        )
        preferred_endpoint = str(
            inspection.get("preferred_api_endpoint") or ""
        ).strip() or None
        return {
            "id": model_id,
            "name": package.get("display_name") or package_id,
            "display_name": package.get("display_name") or package_id,
            "base_model_name": package.get("display_name") or package_id,
            "proxy_name": generate_proxy_name(f"audio-cpp/{package_id}"),
            "source": {
                "provider": "audio_cpp",
                "id": package_id,
                "package": package.get("source") or {},
                "engine_commit": active.get("source_commit"),
            },
            "format": "mixed",
            "artifact": artifact,
            "local_path": artifact["runtime_path"],
            "bundle_path": final_bundle,
            "family": inspection.get("family"),
            "tasks": tasks,
            "task": primary_task,
            "modes": manifest["modes"],
            "input_modalities": inputs,
            "output_modalities": outputs,
            "capabilities": inspection.get("capabilities") or {},
            "preferred_api_endpoint": preferred_endpoint,
            "compatible_engines": ["audio_cpp"],
            "manifest": manifest,
            "file_size": total_size,
            "downloaded_at": _utcnow(),
            "config": config,
        }

    async def install_package(
        self,
        task_id: str,
        package_id: str,
        *,
        options: Optional[dict] = None,
    ) -> dict:
        options = dict(options or {})
        active = self._active_version()
        package = self.package_metadata(package_id, active)
        method = _source_kind_method(package)
        if method == "unavailable" or not package.get("installable", True):
            raise ValueError(f"Package '{package_id}' is not installable")
        source = package.get("source") if isinstance(package.get("source"), dict) else {}
        if str(source.get("kind") or "") == "utility" and not (
            options.get("source_file") or options.get("source_dir")
        ):
            raise ValueError(
                f"Package '{package_id}' requires source_file or source_dir for conversion"
            )

        models_root = os.path.realpath(self.manager.models_dir)
        safe_package_id = _safe_id(package_id)
        record_id = f"audio-cpp--{safe_package_id}"
        if self.store.get_model(record_id):
            raise FileExistsError(f"Model record '{record_id}' already exists")
        final_bundle = os.path.realpath(os.path.join(models_root, safe_package_id))
        if os.path.commonpath([models_root, final_bundle]) != models_root:
            raise ValueError("Invalid package destination")
        if os.path.exists(final_bundle):
            raise FileExistsError(f"Audio package '{package_id}' is already installed")

        staging_parent = os.path.join(models_root, ".staging")
        os.makedirs(staging_parent, exist_ok=True)
        staging_root = os.path.join(staging_parent, f"{_safe_id(package_id)}-{uuid.uuid4().hex}")
        os.makedirs(staging_root, exist_ok=False)
        register_task_cancel(task_id)
        stored_model = False
        try:
            self.pm.update_task(
                task_id,
                progress=2,
                message=f"Resolving {package_id}",
                metadata_update={"stage": "resolve", "install_method": method},
            )
            if method == "direct":
                if str(package.get("manager_backend") or "").strip().lower() == "v2":
                    staged_model_path = await self._install_with_manager(
                        task_id,
                        package,
                        staging_root,
                        active,
                        options,
                        require_helper_venv=False,
                    )
                else:
                    staged_model_path = await self._download_direct(
                        task_id, package, staging_root, active
                    )
            elif method == "bundled":
                staged_model_path = await self._install_bundled_asset(
                    task_id, package, staging_root, active
                )
            else:
                staged_model_path = await self._install_with_manager(
                    task_id, package, staging_root, active, options
                )
            self._check_required_files(package, staged_model_path)
            runtime_model_path = self._resolve_installed_model_path(
                package, staged_model_path
            )
            try:
                inspection = await self._inspect(
                    task_id,
                    active,
                    runtime_model_path,
                    self._resolve_inspect_family(
                        package=package,
                        family_hint=str(options.get("family") or "") or None,
                        model_path=runtime_model_path,
                    ),
                )
            except RuntimeError as exc:
                detail = str(exc)
                if "missing model package file" in detail and "safetensors" in detail:
                    raise RuntimeError(
                        f"{detail} Hint: audio.cpp fell back to the safetensors source, "
                        "usually because no discoverable package-root GGUF was present. "
                        "Studio tries to publish nested *.gguf weights as model.gguf; "
                        "if this persists, the upstream package layout is incomplete."
                    ) from exc
                raise
            relative_model_path = os.path.relpath(runtime_model_path, staging_root)
            self.pm.update_task(
                task_id,
                progress=95,
                message="Promoting validated audio package",
                metadata_update={"stage": "promote"},
            )
            os.replace(staging_root, final_bundle)
            promoted_model_path = os.path.join(final_bundle, relative_model_path)
            record = self._model_record(
                package,
                final_bundle,
                promoted_model_path,
                inspection,
                method,
                active,
            )
            existing = self.store.get_model(record["id"])
            if existing:
                raise FileExistsError(f"Model record '{record['id']}' already exists")
            stored = self.store.add_model(record)
            stored_model = True
            await asyncio.to_thread(
                scan_audio_cpp_model_profile,
                self.store,
                active,
                stored,
                force=True,
            )
            try:
                from backend.llama_swap_manager import mark_swap_config_stale

                mark_swap_config_stale()
            except Exception:
                pass
            return stored
        except BaseException:
            if os.path.isdir(staging_root):
                robust_rmtree(staging_root)
            if os.path.isdir(final_bundle) and not stored_model:
                robust_rmtree(final_bundle)
            raise
        finally:
            unregister_task_cancel(task_id)

    async def import_local_bundle(
        self,
        task_id: str,
        source_path: str,
        *,
        package_id: Optional[str] = None,
        family: Optional[str] = None,
    ) -> dict:
        active = self._active_version()
        source_real = os.path.realpath(str(source_path or ""))
        if not os.path.isdir(source_real):
            raise ValueError("source_path must be an existing directory")
        package_id = _safe_id(package_id or os.path.basename(source_real))
        models_root = os.path.realpath(self.manager.models_dir)
        final_bundle = os.path.realpath(os.path.join(models_root, package_id))
        record_id = f"audio-cpp--{package_id}"
        if self.store.get_model(record_id):
            raise FileExistsError(f"Model record '{record_id}' already exists")
        if os.path.exists(final_bundle):
            raise FileExistsError(f"Audio package '{package_id}' is already installed")
        staging_parent = os.path.join(models_root, ".staging")
        os.makedirs(staging_parent, exist_ok=True)
        staging_root = os.path.join(staging_parent, f"import-{uuid.uuid4().hex}")
        if os.path.commonpath([source_real, staging_root]) == source_real:
            raise ValueError(
                "source_path cannot contain the managed audio.cpp staging directory"
            )
        register_task_cancel(task_id)
        stored_model = False
        try:
            self.pm.update_task(
                task_id,
                progress=10,
                message="Copying local audio bundle",
                metadata_update={"stage": "copy"},
            )
            os.makedirs(staging_root, exist_ok=False)
            await self._copy_local_bundle(task_id, source_real, staging_root)
            if is_task_cancel_requested(task_id):
                raise TaskCancelledError("Audio model import cancelled")
            package = {
                "id": package_id,
                "display_name": package_id,
                "target_directory": ".",
                "description": "Locally imported audio.cpp bundle",
                "source": {"kind": "local_import", "path": source_real},
                "required_files": [],
                "format": "gguf"
                if any(
                    name.lower().endswith(".gguf")
                    for _, _, names in os.walk(staging_root)
                    for name in names
                )
                else "mixed",
            }
            runtime_model_path = self._resolve_installed_model_path(
                package, staging_root
            )
            inspection = await self._inspect(
                task_id,
                active,
                runtime_model_path,
                self._resolve_inspect_family(
                    package=package,
                    family_hint=family,
                    model_path=runtime_model_path,
                ),
            )
            relative_model_path = os.path.relpath(runtime_model_path, staging_root)
            os.replace(staging_root, final_bundle)
            promoted_model_path = os.path.join(final_bundle, relative_model_path)
            record = self._model_record(
                package,
                final_bundle,
                promoted_model_path,
                inspection,
                "local_import",
                active,
            )
            stored = self.store.add_model(record)
            stored_model = True
            await asyncio.to_thread(
                scan_audio_cpp_model_profile,
                self.store,
                active,
                stored,
                force=True,
            )
            try:
                from backend.llama_swap_manager import mark_swap_config_stale

                mark_swap_config_stale()
            except Exception:
                pass
            return stored
        except BaseException:
            if os.path.isdir(staging_root):
                robust_rmtree(staging_root)
            if os.path.isdir(final_bundle) and not stored_model:
                robust_rmtree(final_bundle)
            raise
        finally:
            unregister_task_cancel(task_id)

    async def cancel(self, task_id: str) -> bool:
        from backend.task_cancel_registry import request_task_cancel

        requested = request_task_cancel(task_id)
        process = self._active_processes.get(task_id)
        if process and process.returncode is None:
            await self._terminate_process(process)
            requested = True
        return requested


_installer: Optional[AudioModelInstaller] = None


def get_audio_model_installer() -> AudioModelInstaller:
    global _installer
    if _installer is None:
        _installer = AudioModelInstaller()
    return _installer
