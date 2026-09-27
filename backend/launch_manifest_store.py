"""Immutable launch generations, atomic pointers, gates, and receipts.

Generations live outside every llama-swap watched directory. A published
pointer names one revision; that generation is never edited in place.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from backend.data_store import studio_data_dir

_REVISION_RE_TEXT = r"^[0-9a-f]{64}$"
_DIR_MODE = 0o700
_FILE_MODE = 0o600


class ManifestStoreError(ValueError):
    pass


@dataclass
class PublishedPointer:
    revision: str
    previous_revision: Optional[str]


def runtime_root() -> str:
    path = os.path.join(studio_data_dir(), "runtime", "models")
    os.makedirs(path, mode=_DIR_MODE, exist_ok=True)
    return path


def safe_model_key(model_id: str) -> str:
    if not str(model_id or "").strip():
        raise ManifestStoreError("model id is required")
    return hashlib.sha256(str(model_id).encode("utf-8")).hexdigest()


def model_runtime_dir(model_id: str) -> str:
    return os.path.join(runtime_root(), safe_model_key(model_id))


def _validate_revision(revision: str) -> str:
    text = str(revision or "")
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ManifestStoreError("revision must be a lowercase SHA-256 hex digest")
    return text


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: str, payload: str) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, mode=_DIR_MODE, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=parent)
    try:
        os.chmod(tmp, _FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        os.chmod(path, _FILE_MODE)
        _fsync_dir(parent)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _reject_escape(root: str, path: str) -> str:
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(path)
    if real_path != real_root and not real_path.startswith(real_root + os.sep):
        raise ManifestStoreError(f"path escapes the generation root: {path}")
    return real_path


class LaunchManifestStore:
    def __init__(self, root: Optional[str] = None):
        self.root = root or runtime_root()
        os.makedirs(self.root, mode=_DIR_MODE, exist_ok=True)

    def model_dir(self, model_id: str) -> str:
        return os.path.join(self.root, safe_model_key(model_id))

    def generation_dir(self, model_id: str, revision: str) -> str:
        revision = _validate_revision(revision)
        path = os.path.join(self.model_dir(model_id), "generations", revision)
        return _reject_escape(self.model_dir(model_id), path) if os.path.exists(path) else path

    def stage(self, model_id: str, document: Dict[str, Any]) -> str:
        """Write a complete generation. Existing revisions are immutable."""
        revision = _validate_revision(str(document.get("revision") or ""))
        body = dict(document)
        inline = body.pop("artifacts_inline", {}) or {}
        dest = os.path.join(self.model_dir(model_id), "generations", revision)
        stored = {key: value for key, value in body.items()}
        if os.path.exists(dest):
            existing = self.read_manifest(model_id, revision)
            if _canonical(existing) != _canonical(stored):
                raise ManifestStoreError(
                    f"generation {revision} already exists with different content"
                )
            return revision
        parent = os.path.join(self.model_dir(model_id), "generations")
        os.makedirs(parent, mode=_DIR_MODE, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix=".stage-", dir=parent)
        try:
            os.chmod(tmp, _DIR_MODE)
            artifacts = os.path.join(tmp, "artifacts")
            os.makedirs(artifacts, mode=_DIR_MODE, exist_ok=True)
            for name, payload in inline.items():
                _write_artifact(artifacts, name, payload)
            _atomic_write(
                os.path.join(tmp, "manifest.json"),
                json.dumps(stored, sort_keys=True, indent=2) + "\n",
            )
            os.replace(tmp, dest)
            os.chmod(dest, _DIR_MODE)
            _fsync_dir(parent)
        except Exception:
            if os.path.isdir(tmp):
                import shutil

                shutil.rmtree(tmp, ignore_errors=True)
            raise
        return revision

    def publish(self, model_id: str, revision: str) -> PublishedPointer:
        revision = _validate_revision(revision)
        manifest_path = os.path.join(
            self.model_dir(model_id), "generations", revision, "manifest.json"
        )
        if not os.path.isfile(manifest_path):
            raise ManifestStoreError(f"cannot publish missing revision {revision}")
        _reject_escape(self.model_dir(model_id), manifest_path)
        current = self.read_pointer(model_id)
        previous = current.revision if current else None
        if previous == revision:
            previous = current.previous_revision if current else None
        pointer = {
            "schema_version": 1,
            "model_id": model_id,
            "revision": revision,
            "previous_revision": previous,
        }
        _atomic_write(
            os.path.join(self.model_dir(model_id), "active.json"),
            json.dumps(pointer, sort_keys=True, indent=2) + "\n",
        )
        return PublishedPointer(revision=revision, previous_revision=previous)

    def read_pointer(self, model_id: str) -> Optional[PublishedPointer]:
        path = os.path.join(self.model_dir(model_id), "active.json")
        if not os.path.isfile(path):
            return None
        _reject_escape(self.model_dir(model_id), path)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestStoreError(f"active pointer is unreadable: {exc}") from exc
        revision = _validate_revision(str(payload.get("revision") or ""))
        previous = payload.get("previous_revision")
        if previous:
            previous = _validate_revision(str(previous))
        return PublishedPointer(revision=revision, previous_revision=previous)

    def read_manifest(self, model_id: str, revision: str) -> Dict[str, Any]:
        revision = _validate_revision(revision)
        path = os.path.join(
            self.model_dir(model_id), "generations", revision, "manifest.json"
        )
        _reject_escape(self.model_dir(model_id), path)
        if not os.path.isfile(path):
            raise ManifestStoreError(f"manifest {revision} is missing")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ManifestStoreError("manifest must be a JSON object")
        if str(payload.get("revision") or "") != revision:
            raise ManifestStoreError("manifest revision does not match its directory")
        return payload

    def write_receipt(
        self,
        model_id: str,
        *,
        revision: str,
        launch_id: str,
        pid: int,
        start_ticks: str,
    ) -> str:
        revision = _validate_revision(revision)
        if not launch_id or "/" in launch_id or "\\" in launch_id or ".." in launch_id:
            raise ManifestStoreError("launch id is invalid")
        payload = {
            "model_id": model_id,
            "revision": revision,
            "launch_id": launch_id,
            "pid": int(pid),
            "start_ticks": str(start_ticks),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = os.path.join(self.model_dir(model_id), "launches", f"{launch_id}.json")
        _atomic_write(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")
        return path

    def read_receipts(self, model_id: str) -> List[Dict[str, Any]]:
        directory = os.path.join(self.model_dir(model_id), "launches")
        if not os.path.isdir(directory):
            return []
        rows = []
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(directory, name)
            try:
                _reject_escape(self.model_dir(model_id), path)
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError, ManifestStoreError):
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def retained_revisions(self, model_id: str, extra: Optional[Iterable[str]] = None) -> Set[str]:
        keep: Set[str] = set()
        pointer = None
        try:
            pointer = self.read_pointer(model_id)
        except ManifestStoreError:
            pointer = None
        if pointer:
            keep.add(pointer.revision)
            if pointer.previous_revision:
                keep.add(pointer.previous_revision)
        for revision in extra or []:
            if revision:
                keep.add(_validate_revision(revision))
        return keep

    def garbage_collect(self, model_id: str, extra: Optional[Iterable[str]] = None) -> List[str]:
        import shutil

        keep = self.retained_revisions(model_id, extra)
        directory = os.path.join(self.model_dir(model_id), "generations")
        removed: List[str] = []
        if not os.path.isdir(directory):
            return removed
        for name in os.listdir(directory):
            if name in keep or name.startswith("."):
                continue
            try:
                _validate_revision(name)
            except ManifestStoreError:
                continue
            path = os.path.join(directory, name)
            _reject_escape(self.model_dir(model_id), path)
            shutil.rmtree(path)
            removed.append(name)
        return removed

    def referenced_external_paths(self) -> List[str]:
        found: List[str] = []
        if not os.path.isdir(self.root):
            return found
        for key in os.listdir(self.root):
            generations = os.path.join(self.root, key, "generations")
            if not os.path.isdir(generations):
                continue
            for revision in os.listdir(generations):
                manifest_path = os.path.join(generations, revision, "manifest.json")
                if not os.path.isfile(manifest_path):
                    continue
                try:
                    with open(manifest_path, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    continue
                executable = str(payload.get("executable") or "")
                if executable:
                    found.append(executable)
                cwd = payload.get("cwd")
                if isinstance(cwd, str) and cwd:
                    found.append(cwd)
                for item in payload.get("argv") or []:
                    if isinstance(item, str) and os.path.isabs(item):
                        found.append(item)
                for identity in (payload.get("file_identities") or {}).values():
                    if isinstance(identity, dict) and identity.get("path"):
                        found.append(str(identity["path"]))
        return found

    def acquire_gate(self, model_id: str, *, exclusive: bool, timeout: float = 30.0):
        import fcntl

        directory = self.model_dir(model_id)
        os.makedirs(directory, mode=_DIR_MODE, exist_ok=True)
        path = os.path.join(directory, "launch.lock")
        fd = os.open(path, os.O_RDWR | os.O_CREAT, _FILE_MODE)
        flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, flag | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise TimeoutError(
                        f"timed out waiting for the launch gate of {model_id}"
                    )
                time.sleep(0.02)

    def release_gate(self, fd: int) -> None:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def deletion_block_reason(path: str) -> Optional[str]:
    """Return a reason when ``path`` is still required by a retained generation."""
    if not path:
        return None
    try:
        target = os.path.realpath(path)
    except OSError:
        return None
    try:
        store = LaunchManifestStore()
    except Exception:
        return None
    for ref in store.referenced_external_paths():
        try:
            real = os.path.realpath(ref)
        except OSError:
            continue
        if real == target or real.startswith(target + os.sep) or target.startswith(real + os.sep):
            return (
                f"Refusing to delete {path}; it is referenced by a published "
                "or retained launch generation"
            )
    if os.path.isdir(store.root):
        for key in os.listdir(store.root):
            model_root = os.path.join(store.root, key)
            generations = os.path.join(model_root, "generations")
            if not os.path.isdir(generations):
                continue
            try:
                pointer = None
                active = os.path.join(model_root, "active.json")
                if os.path.isfile(active):
                    with open(active, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    pointer = str(payload.get("revision") or "")
                    previous = str(payload.get("previous_revision") or "")
                else:
                    previous = ""
            except (OSError, json.JSONDecodeError):
                continue
            for revision in (pointer, previous):
                if not revision:
                    continue
                retained = os.path.join(generations, revision)
                try:
                    real = os.path.realpath(retained)
                except OSError:
                    continue
                if real == target or real.startswith(target + os.sep) or target.startswith(real + os.sep):
                    return (
                        f"Refusing to delete {path}; a retained launch generation "
                        "still depends on it"
                    )
    return None


def new_launch_id() -> str:
    return uuid.uuid4().hex


def _canonical(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_artifact(directory: str, name: str, payload: Any) -> None:
    if not name or name != os.path.basename(name) or name in {".", ".."}:
        raise ManifestStoreError(f"invalid artifact name {name!r}")
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    _atomic_write(os.path.join(directory, name), text)
