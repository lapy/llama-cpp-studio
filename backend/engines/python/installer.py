"""Shared versioned venv installer used by SGLang, vLLM, LMDeploy, and 1Cat-vLLM."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from asyncio.subprocess import PIPE, STDOUT
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from backend.data_store import get_store
from backend.logging_config import get_logger
from backend.operations.cancellable import CancellableOperationManager
from backend.operations.progress import get_progress_manager
from backend.paths import studio_data_dir


logger = get_logger(__name__)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def unique_version_name(store: Any, engine_id: str, base: str) -> str:
    existing = {
        str(row.get("version"))
        for row in store.get_engine_versions(engine_id)
        if row.get("version")
    }
    if base not in existing:
        return base
    stamp = int(time.time())
    for suffix in range(10000):
        candidate = f"{base}-{stamp}" if suffix == 0 else f"{base}-{stamp}-{suffix}"
        if candidate not in existing:
            return candidate
    return f"{base}-{stamp}-x"


class PythonVenvInstaller(CancellableOperationManager, ABC):
    """Owns versioned directories, a venv, pip, git, and engine-version rows."""

    engine_id: str = ""
    label: str = ""
    distribution_names: Tuple[str, ...] = ()

    def __init__(
        self,
        *,
        engine_id: str,
        label: str,
        root_name: str,
        log_name: str,
        log_path: Optional[str] = None,
        base_dir: Optional[str] = None,
        state_path: Optional[str] = None,
        state_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.engine_id = engine_id
        self.label = label
        self.MANAGER_NAME = engine_id
        if not self.distribution_names:
            self.distribution_names = (engine_id,)

        data_root = studio_data_dir()
        self._root_dir = os.path.abspath(
            base_dir or os.path.join(data_root, root_name)
        )
        self._base_dir = self._root_dir
        self._venv_path = os.path.join(self._base_dir, "venv")
        self._log_path = os.path.abspath(
            log_path or os.path.join(data_root, "logs", log_name)
        )
        self._state_path: Optional[str] = None
        if state_path or state_name:
            self._state_path = os.path.abspath(
                state_path
                or os.path.join(data_root, "config", str(state_name))
            )
        self._ensure_directories()
        self._unpack_heartbeat_seconds = 4.0
        self._active_build_window = None

    @property
    def install_root(self) -> str:
        return self._root_dir

    @property
    def log_path(self) -> str:
        return self._log_path

    @property
    def operation_descriptions(self) -> Dict[str, str]:
        return {
            "install": f"Install {self.label}",
            "install_source": f"Install {self.label} from Source",
            "sync_source": f"Sync {self.label} Source",
            "remove": f"Remove {self.label}",
        }

    def _ensure_directories(self) -> None:
        os.makedirs(self._base_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        if self._state_path:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)

    def _venv_bin(self, executable: str) -> str:
        if os.name == "nt":
            executable = (
                executable
                if executable.lower().endswith(".exe")
                else f"{executable}.exe"
            )
            return os.path.join(self._venv_path, "Scripts", executable)
        return os.path.join(self._venv_path, "bin", executable)

    def _venv_python(self) -> str:
        return self._venv_bin("python")

    def _prepare_versioned_paths(self, label: str = "") -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        version_dir = f"{timestamp}-{label}" if label else timestamp
        self._base_dir = os.path.join(self._root_dir, version_dir)
        self._venv_path = os.path.join(self._base_dir, "venv")
        self._ensure_directories()
        return version_dir

    def _bind_install_dir(self, reuse_dir: Optional[str], label: str) -> str:
        if reuse_dir:
            self._base_dir = os.path.abspath(reuse_dir)
            self._venv_path = os.path.join(self._base_dir, "venv")
            self._ensure_directories()
            return os.path.basename(self._base_dir)
        return self._prepare_versioned_paths(label)

    def _register_pending_version(self, version_name: str, extra: Dict[str, Any]) -> None:
        from backend.engines.lifecycle import mark_engine_version_building

        mark_engine_version_building(
            get_store(),
            self.engine_id,
            {
                "version": version_name,
                "venv_path": self._venv_path,
                "install_dir": self._base_dir,
                "installed_at": utcnow(),
                **(extra or {}),
            },
            task_id=self._progress_task_id,
        )

    def _fail_pending_version(
        self,
        version_name: str,
        error: str,
        extra: Optional[Dict[str, Any]] = None,
        *,
        cancelled: bool = False,
    ) -> None:
        from backend.engines.lifecycle import mark_engine_version_failed

        mark_engine_version_failed(
            get_store(),
            self.engine_id,
            version_name,
            error=str(error),
            cancelled=cancelled,
            extra={
                "venv_path": self._venv_path,
                "install_dir": self._base_dir,
                **(extra or {}),
            },
        )

    def _ready_pending_version(self, pending_version: str, meta: Dict[str, Any]) -> str:
        from backend.engines.lifecycle import mark_engine_version_ready

        store = get_store()
        final_name = str(meta.get("version") or pending_version)
        payload = {
            **meta,
            "version": final_name,
            "venv_path": self._venv_path,
            "install_dir": self._base_dir,
        }
        if final_name != pending_version:
            store.delete_engine_version(self.engine_id, pending_version)
        mark_engine_version_ready(store, self.engine_id, payload)
        return final_name

    def _ensure_venv(self) -> None:
        if os.path.exists(self._venv_python()):
            return
        os.makedirs(self._base_dir, exist_ok=True)
        try:
            subprocess.run([sys.executable, "-m", "venv", self._venv_path], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to create {self.label} virtual environment: {exc}"
            ) from exc

    def _load_state(self) -> Dict[str, Any]:
        if not self._state_path or not os.path.exists(self._state_path):
            return {}
        try:
            with open(self._state_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load %s manager state: %s", self.label, exc)
            return {}

    def _save_state(self, state: Dict[str, Any]) -> None:
        if not self._state_path:
            return
        tmp_path = f"{self._state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
        os.replace(tmp_path, self._state_path)

    def _update_installed_state(
        self, installed: bool, version: Optional[str]
    ) -> None:
        state = self._load_state()
        state["installed_version"] = version if installed else None
        state["installed_at"] = utcnow() if installed else None
        state["venv_path"] = self._venv_path
        if not installed:
            state["removed_at"] = utcnow()
        self._save_state(state)

    def _refresh_state_from_environment(self) -> None:
        state = self._load_state()
        version = self._detect_installed_version()
        state["installed_version"] = version
        state["venv_path"] = self._venv_path
        if version is None:
            state["removed_at"] = utcnow()
        self._save_state(state)

    def _stateful_status(
        self,
        *,
        resolve_binary_path: Callable[[], Optional[str]],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        active = get_store().get_active_engine_version(self.engine_id)
        saved_venv = self._venv_path
        try:
            if active and active.get("venv_path"):
                self._venv_path = str(active["venv_path"])
            version = self._detect_installed_version()
            binary_path = resolve_binary_path()
            state = self._load_state()
            payload: Dict[str, Any] = {
                "installed": version is not None and binary_path is not None,
                "version": version,
                "binary_path": binary_path,
                "venv_path": (
                    (active.get("venv_path") if active else None)
                    or state.get("venv_path")
                    or self._venv_path
                ),
                "installed_at": (active.get("installed_at") if active else None)
                or state.get("installed_at"),
                "removed_at": state.get("removed_at"),
                "operation": self._operation,
                "operation_started_at": self._operation_started_at,
                "progress_task_id": self._progress_task_id,
                "last_error": self._last_error,
                "log_path": self._log_path,
                "install_type": active.get("install_type") if active else None,
                "source_repo": active.get("source_repo") if active else None,
                "source_branch": active.get("source_branch") if active else None,
            }
            payload.update(extra or {})
            return payload
        finally:
            self._venv_path = saved_venv

    def _detect_installed_version(self, venv_path: Optional[str] = None) -> Optional[str]:
        venv = os.path.abspath(venv_path or self._venv_path)
        python_bin = os.path.join(
            venv,
            "Scripts" if os.name == "nt" else "bin",
            "python.exe" if os.name == "nt" else "python",
        )
        if not os.path.isfile(python_bin) and not os.path.exists(python_bin):
            return None
        names = ", ".join(repr(name) for name in self.distribution_names)
        script = (
            "import sys\n"
            "from importlib import metadata\n"
            f"for dist in ({names},):\n"
            "    try:\n"
            "        print(metadata.version(dist))\n"
            "        break\n"
            "    except metadata.PackageNotFoundError:\n"
            "        continue\n"
            "else:\n"
            "    sys.exit(1)\n"
        )
        try:
            output = subprocess.check_output(
                [python_bin, "-c", script],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return output or None
        except (OSError, subprocess.CalledProcessError):
            return None

    @staticmethod
    def _prepend_path(env: Dict[str, str], directory: str) -> None:
        directory = os.path.abspath(directory)
        current = env.get("PATH", "")
        parts = [part for part in current.split(os.pathsep) if part]
        if directory not in parts:
            env["PATH"] = os.pathsep.join([directory, *parts]) if current else directory

    def _source_install_operation(self) -> bool:
        return self._operation in {"install_source", "sync_source"}

    def _install_progress_tracker(self):
        from backend.build_progress import (
            PipInstallProgressTracker,
            SourceInstallProgressTracker,
        )

        task_id = self._progress_task_id
        source = self._source_install_operation()
        current = getattr(self, "_pip_progress", None)
        if source:
            tracker_matches = isinstance(current, SourceInstallProgressTracker)
        else:
            tracker_matches = isinstance(current, PipInstallProgressTracker)
        if getattr(self, "_pip_progress_task", None) != task_id or not tracker_matches:
            self._pip_progress = (
                SourceInstallProgressTracker() if source else PipInstallProgressTracker()
            )
            self._pip_progress_task = task_id
        return self._pip_progress

    async def _broadcast_log_line(self, line: str, *, record: bool = True) -> None:
        try:
            from backend.build_progress import (
                is_compiler_progress_label,
                progress_from_install_log,
            )

            if not self._progress_task_id:
                if record:
                    await self._append_task_log(line)
                return
            existing = get_progress_manager().get_task(self._progress_task_id) or {}
            log_count = int((existing.get("metadata") or {}).get("log_count", 0))
            if record:
                log_count += 1
                await self._append_task_log(line)
            tracker = self._install_progress_tracker()
            current_progress = float(existing.get("progress") or 0)
            if self._source_install_operation():
                progress, label = tracker.observe(
                    line,
                    current_progress=current_progress,
                    log_count=log_count,
                    window=getattr(self, "_active_build_window", None),
                )
            else:
                progress, label = progress_from_install_log(
                    line,
                    current_progress=current_progress,
                    log_count=log_count,
                    tracker=tracker,
                )
            if not record:
                return
            if is_compiler_progress_label(label):
                message = f"{line} {label}".strip()
                if len(line) > 120:
                    message = f"Building… {label}"
            elif label:
                message = label
            else:
                message = line.strip()[:180]
            await self._update_progress_task(
                progress,
                message,
                metadata_update={"log_count": log_count, "stage": tracker.phase},
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed to broadcast %s log line: %s", self.label, exc)

    async def _emit_install_line(self, text: str, log_file, *, force: bool = False) -> None:
        """Record a pip/build line, throttling in-progress download bars."""
        from backend.build_progress import is_partial_pip_download

        partial = is_partial_pip_download(text) and not force
        record = True
        if partial:
            now = time.monotonic()
            last = float(getattr(self, "_pip_partial_at", 0.0) or 0.0)
            record = now - last >= 0.25
            if record:
                self._pip_partial_at = now
        if record:
            log_file.write(text + "\n")
        await self._broadcast_log_line(text, record=record)

    async def _run_logged(
        self,
        argv: Sequence[str],
        operation: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        append: bool = True,
    ) -> int:
        mode = "a" if append else "w"
        argv_list = list(argv)
        previous_window = getattr(self, "_active_build_window", None)
        restore_window = False
        if self._source_install_operation() and argv_list:
            executable = os.path.basename(str(argv_list[0]))
            if executable in {"git", "git.exe"}:
                self._active_build_window = (1, 8)
                restore_window = True
        header = f"[{utcnow()}] {self.label} {operation}: {' '.join(argv_list)}\n"
        with open(self._log_path, mode, encoding="utf-8") as log_file:
            log_file.write(header)
        await self._broadcast_log_line(f"$ {' '.join(argv_list)}")

        process = await asyncio.create_subprocess_exec(
            *argv_list,
            stdout=PIPE,
            stderr=STDOUT,
            cwd=cwd,
            env=env,
        )
        self._track_process(process)

        async def _stream_output() -> None:
            if process.stdout is None:
                return
            pending = b""
            read_task: Optional[asyncio.Task] = None
            heartbeat = float(getattr(self, "_unpack_heartbeat_seconds", 0) or 0)
            try:
                with open(self._log_path, "a", encoding="utf-8", buffering=1) as log_file:
                    while True:
                        if read_task is None:
                            read_task = asyncio.create_task(process.stdout.read(65536))
                        if heartbeat > 0:
                            done, _pending_tasks = await asyncio.wait(
                                {read_task}, timeout=heartbeat
                            )
                            if not done:
                                await self._emit_unpack_heartbeat()
                                continue
                        else:
                            await read_task
                        chunk = read_task.result()
                        read_task = None
                        if not chunk:
                            break
                        pending += chunk
                        while True:
                            breaks = [
                                index
                                for index in (
                                    pending.find(b"\n"),
                                    pending.find(b"\r"),
                                )
                                if index >= 0
                            ]
                            if not breaks:
                                break
                            cut = min(breaks)
                            raw = pending[:cut]
                            pending = pending[cut + 1 :]
                            text = raw.decode("utf-8", errors="replace").strip()
                            if not text:
                                continue
                            await self._emit_install_line(text, log_file)
                    tail = pending.decode("utf-8", errors="replace").strip()
                    if tail:
                        await self._emit_install_line(tail, log_file, force=True)
            finally:
                if read_task is not None and not read_task.done():
                    read_task.cancel()

        await asyncio.gather(process.wait(), _stream_output())
        self._clear_active_process()
        code = process.returncode or 0
        if code == 0 and self._source_install_operation():
            tracker = self._install_progress_tracker()
            if getattr(tracker, "phase", "") == "compile" and hasattr(
                tracker, "complete_compile"
            ):
                tracker.complete_compile()
                await self._update_progress_task(
                    tracker.progress,
                    tracker.message or "Compile step finished",
                    metadata_update={"stage": tracker.phase},
                )
        if restore_window:
            self._active_build_window = previous_window
        return code

    async def _emit_unpack_heartbeat(self) -> None:
        """Advance the bar during a quiet pip unpack or source compile."""
        if not self._progress_task_id:
            return
        tracker = self._install_progress_tracker()
        if hasattr(tracker, "note_idle"):
            before = tracker.progress
            progress, message = tracker.note_idle()
            if progress <= before:
                return
            await self._update_progress_task(
                progress,
                message,
                metadata_update={"stage": tracker.phase},
            )
            return
        if tracker.phase != "unpack":
            return
        before = tracker.progress
        progress, message = tracker.note_unpack_wait()
        if progress <= before:
            return
        await self._update_progress_task(
            progress,
            message,
            metadata_update={"stage": tracker.phase},
        )

    async def _run_pip(
        self,
        args: Sequence[str],
        operation: str,
        ensure_venv: bool = True,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        append: bool = True,
    ) -> int:
        if ensure_venv:
            self._ensure_venv()
        python_exe = self._venv_python()
        if not os.path.exists(python_exe):
            raise RuntimeError(
                f"{self.label} virtual environment is missing; cannot run pip."
            )
        return await self._run_logged(
            [python_exe, "-m", "pip", *args],
            operation,
            cwd=cwd,
            env=env,
            append=append,
        )

    async def _git_head(self, clone_dir: str) -> Optional[str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "HEAD",
                stdout=PIPE,
                stderr=STDOUT,
                cwd=clone_dir,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return None
            return stdout.decode("utf-8", errors="replace").strip() or None
        except Exception as exc:
            logger.debug("Could not read %s source HEAD: %s", self.label, exc)
            return None

    async def _sync_git_checkout(self, clone_dir: str, branch: str) -> None:
        branch = str(branch or "").strip()
        if not branch:
            raise RuntimeError("A source branch is required for sync")
        if not os.path.isdir(os.path.join(clone_dir, ".git")):
            raise RuntimeError(f"Source checkout not found: {clone_dir}")

        code = await self._run_logged(
            ["git", "fetch", "--prune", "origin", branch],
            "sync_source",
            cwd=clone_dir,
            append=False,
        )
        if code != 0:
            raise RuntimeError(f"git fetch failed with code {code}")

        code = await self._run_logged(
            ["git", "checkout", "-B", branch, "FETCH_HEAD"],
            "sync_source",
            cwd=clone_dir,
        )
        if code != 0:
            await self._broadcast_log_line(
                "Checkout had local conflicts; cleaning untracked source files while keeping build caches."
            )
            clean_code = await self._run_logged(
                [
                    "git",
                    "clean",
                    "-fd",
                    "-e",
                    "build/",
                    "-e",
                    "build",
                    "-e",
                    "dist/",
                    "-e",
                    "dist",
                    "-e",
                    ".cache/",
                    "-e",
                    ".cache",
                ],
                "sync_source",
                cwd=clone_dir,
            )
            if clean_code != 0:
                raise RuntimeError(f"git clean failed with code {clean_code}")
            code = await self._run_logged(
                ["git", "checkout", "-B", branch, "FETCH_HEAD"],
                "sync_source",
                cwd=clone_dir,
            )
            if code != 0:
                raise RuntimeError(f"git checkout failed with code {code}")

        code = await self._run_logged(
            ["git", "reset", "--hard", "FETCH_HEAD"],
            "sync_source",
            cwd=clone_dir,
        )
        if code != 0:
            raise RuntimeError(f"git reset failed with code {code}")

    async def _start_operation(
        self, operation: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        description = self.operation_descriptions.get(operation, f"Install {self.label}")
        extra = {"engine": self.engine_id, **(metadata or {})}
        self._active_build_window = None
        return await self._begin_operation(operation, description, extra)

    def _raise_if_busy(self) -> None:
        if self._operation:
            raise RuntimeError(f"Another {self.label} operation is already running")

    def _on_task_error(self, exc: Exception) -> None:
        logger.error("%s manager task error: %s", self.label, exc)

    def read_log_tail(self, max_bytes: int = 8192) -> str:
        if not os.path.exists(self._log_path):
            return ""
        with open(self._log_path, "rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - max_bytes))
            data = log_file.read().decode("utf-8", errors="replace")
            if size > max_bytes:
                data = data.split("\n", 1)[-1]
            return data.strip()

    async def sync_source(self, version_entry: Dict[str, Any]) -> Dict[str, Any]:
        return await self.sync_source_version(version_entry)

    @abstractmethod
    async def sync_source_version(
        self, version_entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError
