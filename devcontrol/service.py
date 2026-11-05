"""Development control service shared by the JSON CLI and MCP transport."""

from __future__ import annotations

import json
import tarfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .checks import run_checks
from .compose import Compose
from .config import Config, worktree_name_is_valid
from .errors import BusyError, ConfigError, DevControlError, GuardError, NotFoundError
from .gitops import GitRepo, Revision
from .http_client import request_json, request_text, task_snapshot
from .journal import (
    CheckpointStore,
    Journal,
    compact_records,
    new_operation_id,
    operation_record,
    utc_now,
)
from .locks import FileLock, atomic_write_json, atomic_write_text
from .logs import read_logs
from .redact import redact_obj, redact_text


class DevControl:
    def __init__(self, config: Config, *, actor: str = ""):
        self.config = config
        self.actor = actor or config.actor
        self.repo = GitRepo(config.repo)
        if config.runtime == "local":
            from .local_runtime import LocalRuntime

            self.compose = LocalRuntime(config)
        else:
            self.compose = Compose(config)
        self.journal = Journal(config.journal_path)
        self.checkpoints = CheckpointStore(config.checkpoints_path)
        self.guard = _Guard(config)

    # ------------------------------------------------------------------ API
    def inspect_instance(self) -> dict[str, Any]:
        return self._execute(
            "inspect_instance",
            lambda op_id, revision: self._inspect_payload(),
            exclusive=False,
        )

    def read_logs(self, **kwargs: Any) -> dict[str, Any]:
        return self._execute(
            "read_logs",
            lambda op_id, revision: read_logs(self.compose, **kwargs),
            request={key: value for key, value in kwargs.items() if key != "token"},
            exclusive=False,
        )

    def apply_patch(
        self,
        patch_text: str,
        *,
        worktree: str = "",
        create_checkpoint: bool = True,
        check_only: bool = False,
        label: str = "",
    ) -> dict[str, Any]:
        target = self._worktree_name(worktree)
        payload = {
            "bytes": len(patch_text.encode("utf-8")),
            "check_only": check_only,
            "create_checkpoint": create_checkpoint,
        }

        def action(op_id: str, revision: Revision) -> dict[str, Any]:
            path = self._worktree_path(target)
            repo = GitRepo(path)
            repo.ensure()
            # Validate before creating a checkpoint; a rejected patch should not
            # leave a new checkpoint behind.
            checked_files = repo.check_unified_patch(patch_text)
            if check_only:
                return {"checked": True, "files": checked_files, **payload}
            checkpoint = None
            if create_checkpoint:
                checkpoint = repo.checkpoint(
                    worktree=target, label=label or "before apply_patch", operation_id=op_id
                )
                self.checkpoints.add(checkpoint)
            repo.apply_unified_patch(patch_text)
            return {
                "checked": False,
                "files": checked_files,
                "checkpoint_id": checkpoint.get("id") if checkpoint else None,
                **payload,
            }

        mutating = not check_only
        return self._execute(
            "apply_patch",
            action,
            worktree=target,
            request=payload,
            exclusive=mutating and target == self.config.active,
            lock_worktree=mutating,
        )

    def checkpoint_changes(self, *, worktree: str = "", label: str = "") -> dict[str, Any]:
        target = self._worktree_name(worktree)

        def action(op_id: str, revision: Revision) -> dict[str, Any]:
            repo = GitRepo(self._worktree_path(target))
            checkpoint = repo.checkpoint(worktree=target, label=label, operation_id=op_id)
            self.checkpoints.add(checkpoint)
            return {"checkpoint": checkpoint}

        return self._execute(
            "checkpoint_changes",
            action,
            worktree=target,
            request={"label": label},
            exclusive=target == self.config.active,
            lock_worktree=True,
        )

    def rollback_changes(
        self,
        *,
        worktree: str = "",
        checkpoint: str = "",
    ) -> dict[str, Any]:
        target = self._worktree_name(worktree)

        def action(op_id: str, revision: Revision) -> dict[str, Any]:
            item = self.checkpoints.get(checkpoint, worktree=target) if checkpoint else None
            if item is None and not checkpoint:
                item = self.checkpoints.latest(target)
            if item is None:
                raise NotFoundError(
                    f"no checkpoint is recorded for worktree {target}; run checkpoint-changes first"
                )
            if item.get("worktree") != target:
                raise GuardError("checkpoint belongs to a different worktree")
            checkpoint_id = str(item.get("id") or "")
            repo = GitRepo(self._worktree_path(target))
            commit = repo.checkpoint_commit(checkpoint_id)
            repo.restore_to_commit(commit)
            return {
                "checkpoint": item,
                "restored_commit": commit,
                "head_unchanged": revision.commit,
            }

        return self._execute(
            "rollback_changes",
            action,
            worktree=target,
            request={"checkpoint": checkpoint},
            exclusive=target == self.config.active,
            lock_worktree=True,
        )

    def run_checks(
        self,
        *,
        checks: list[str],
        worktree: str = "",
        tests: Optional[list[str]] = None,
        frontend_targets: Optional[list[str]] = None,
        timeout_seconds: Optional[float] = None,
        fail_fast: bool = True,
    ) -> dict[str, Any]:
        target = self._worktree_name(worktree)

        def action(op_id: str, revision: Revision) -> dict[str, Any]:
            if target != self.config.active:
                raise GuardError(
                    f"worktree {target!r} is not active; activate it before running checks"
                )
            if not self.compose.any_running():
                raise GuardError("development stack is not running; run 'devcontrol stack up'")
            return run_checks(
                self.compose,
                names=checks,
                repo=self._worktree_path(target),
                tests=tests,
                frontend_targets=frontend_targets,
                timeout_seconds=timeout_seconds,
                fail_fast=fail_fast,
            )

        return self._execute(
            "run_checks",
            action,
            worktree=target,
            request={
                "checks": checks,
                "tests": tests or [],
                "frontend_targets": frontend_targets or [],
                "timeout_seconds": timeout_seconds,
                "fail_fast": fail_fast,
            },
            exclusive=target == self.config.active,
        )

    def reload_service(
        self, *, service: str = "backend", wait_seconds: float = 90.0
    ) -> dict[str, Any]:
        def action(op_id: str, revision: Revision) -> dict[str, Any]:
            if service not in {"backend", "frontend", "all"}:
                raise GuardError("service must be backend, frontend, or all")
            if not self.compose.any_running():
                raise GuardError("development stack is not running; run 'devcontrol stack up'")
            services = ["backend", "frontend"] if service == "all" else [service]
            restarted = []
            for name in services:
                if not self.compose.running(name):
                    raise GuardError(f"service is not running: {name}")
                self.compose.restart(name)
                restarted.append(name)
            health = self._wait_health(services, wait_seconds)
            return {"services": restarted, "health": health, "ok": bool(health.get("ok"))}

        return self._execute(
            "reload_service",
            action,
            worktree=self.config.active,
            request={"service": service},
            exclusive=True,
        )

    def capture_diagnostics(self, *, reason: str = "", note: str = "") -> dict[str, Any]:
        def action(op_id: str, revision: Revision) -> dict[str, Any]:
            return self._capture(op_id, revision, reason=reason, note=note)

        return self._execute(
            "capture_diagnostics",
            action,
            worktree=self.config.active,
            request={"reason": reason, "note": note},
            exclusive=False,
        )

    # Worktrees ---------------------------------------------------------
    def list_worktrees(self) -> dict[str, Any]:
        return self._execute(
            "list_worktrees",
            lambda op_id, revision: self._list_worktrees(),
            exclusive=False,
        )

    def create_worktree(self, *, name: str, base: str = "HEAD") -> dict[str, Any]:
        if not worktree_name_is_valid(name) or name == "main":
            raise GuardError(f"invalid worktree name: {name!r}")

        def action(op_id: str, revision: Revision) -> dict[str, Any]:
            path = self.config.worktrees / name
            created = self.repo.add_worktree(path, name=name, base=base)
            checkpoint = GitRepo(path).checkpoint(
                worktree=name, label=f"created from {base}", operation_id=op_id
            )
            self.checkpoints.add(checkpoint)
            return {"worktree": created, "checkpoint": checkpoint}

        return self._execute(
            "create_worktree",
            action,
            worktree="main",
            request={"name": name, "base": base},
            exclusive=False,
            lock_worktree=False,
        )

    def activate_worktree(self, *, name: str, start: bool = True) -> dict[str, Any]:
        if not worktree_name_is_valid(name):
            raise GuardError(f"invalid worktree name: {name!r}")

        def action(op_id: str, revision: Revision) -> dict[str, Any]:
            target = self._worktree_path(name)
            if name != "main" and not target.is_dir():
                raise NotFoundError(f"worktree does not exist: {name}")
            GitRepo(target).ensure()
            previous = self.config.active
            was_running = self.compose.any_running()
            self.compose.ensure_ready()
            self.config.active = name
            self.config.save()
            if hasattr(self.compose, "set_active"):
                self.compose.set_active(name)
            try:
                if start:
                    if was_running:
                        self.compose.run_init()
                        self.compose.up(
                            services=["backend", "frontend"],
                            force_recreate=True,
                            no_deps=True,
                        )
                    else:
                        self.compose.up()
                health = (
                    self._wait_health(["backend", "frontend"], 120.0) if start else {"ok": False}
                )
            except Exception:
                self.config.active = previous
                self.config.save()
                if hasattr(self.compose, "set_active"):
                    try:
                        self.compose.set_active(previous)
                    except Exception:
                        pass
                if was_running:
                    try:
                        self.compose.up(
                            services=["backend", "frontend"],
                            force_recreate=True,
                            no_deps=True,
                        )
                    except Exception:
                        pass
                raise
            active_revision = self._safe_revision(name)
            payload = {
                "previous": previous,
                "active": name,
                "worktree": name,
                "health": health,
                "ok": bool(health.get("ok")) if start else True,
            }
            if active_revision is not None:
                payload["revision"] = active_revision.label
                payload["revision_after"] = active_revision.label
                payload["revision_detail"] = active_revision.to_dict()
            return payload

        return self._execute(
            "activate_worktree",
            action,
            worktree="main",
            request={"name": name, "start": start},
            exclusive=True,
        )

    def remove_worktree(self, *, name: str, force: bool = False) -> dict[str, Any]:
        if not worktree_name_is_valid(name) or name == "main":
            raise GuardError("the main checkout cannot be removed as a worktree")
        if name == self.config.active:
            raise GuardError("cannot remove the active worktree; activate another checkout first")

        def action(op_id: str, revision: Revision) -> dict[str, Any]:
            path = self._worktree_path(name)
            if not path.exists():
                self.repo.prune_worktrees()
                return {
                    "removed": False,
                    "name": name,
                    "detail": "worktree directory did not exist",
                }
            self.repo.remove_worktree(path, force=force)
            return {"removed": True, "name": name, "path": str(path)}

        return self._execute(
            "remove_worktree",
            action,
            worktree="main",
            request={"name": name, "force": force},
            exclusive=True,
        )

    # Stack -------------------------------------------------------------
    def stack_build(self, *, no_cache: bool = False) -> dict[str, Any]:
        def action(op_id: str, revision: Revision) -> dict[str, Any]:
            return self.compose.build(no_cache=no_cache)

        return self._execute(
            "stack_build",
            action,
            worktree=self.config.active,
            request={"no_cache": no_cache},
            exclusive=True,
        )

    def stack_up(
        self,
        *,
        gpu: Optional[str] = None,
        build: bool = False,
    ) -> dict[str, Any]:
        def action(op_id: str, revision: Revision) -> dict[str, Any]:
            previous_gpu = self.config.gpu
            if gpu is not None:
                self.config.gpu = gpu
                self.config.validate()
                self.config.save()
            try:
                if build:
                    self.compose.build()
                self.compose.ensure_ready()
                self.compose.up(build=False)
                health = self._wait_health(["backend", "frontend"], 120.0)
            except Exception:
                if gpu is not None:
                    self.config.gpu = previous_gpu
                    self.config.save()
                raise
            return {
                "gpu": self.config.gpu,
                "health": health,
                "stack": self.compose.summary(),
                "ok": bool(health.get("ok")),
            }

        return self._execute(
            "stack_up",
            action,
            worktree=self.config.active,
            request={"gpu": gpu, "build": build},
            exclusive=True,
        )

    def stack_down(self, *, volumes: bool = False, confirmed: bool = False) -> dict[str, Any]:
        if volumes and not confirmed:
            raise GuardError("removing the node_modules volume requires confirmation")

        def action(op_id: str, revision: Revision) -> dict[str, Any]:
            result = self.compose.down(volumes=volumes)
            return {"stack": self.compose.summary(), "output_tail": result.tail(4000)}

        return self._execute(
            "stack_down",
            action,
            worktree=self.config.active,
            request={"volumes": volumes},
            exclusive=True,
        )

    def stack_status(self) -> dict[str, Any]:
        return self._execute(
            "stack_status",
            lambda op_id, revision: {
                "stack": self.compose.summary(),
                "health": self._quick_health(),
            },
            exclusive=False,
        )

    def journal_tail(self, *, limit: int = 50) -> dict[str, Any]:
        return self._execute(
            "journal",
            lambda op_id, revision: {
                "operations": compact_records(self.journal.read(limit), limit),
            },
            exclusive=False,
        )

    # --------------------------------------------------------------- internals
    def _worktree_name(self, requested: str) -> str:
        name = (requested or "").strip() or self.config.active
        if not worktree_name_is_valid(name):
            raise GuardError(f"invalid worktree name: {name!r}")
        return name

    def _worktree_path(self, name: str) -> Path:
        if name == "main":
            return self.config.repo
        path = (self.config.worktrees / name).resolve()
        if not path.is_relative_to(self.config.worktrees.resolve()):
            raise GuardError(f"worktree escapes the configured root: {name}")
        return path

    def _execute(
        self,
        action: str,
        fn: Callable[[str, Revision], dict[str, Any]],
        *,
        worktree: str = "",
        request: Optional[dict[str, Any]] = None,
        exclusive: bool = False,
        lock_worktree: bool = False,
    ) -> dict[str, Any]:
        operation_id = new_operation_id()
        started_at = utc_now()
        target = self._worktree_name(worktree)
        before: Optional[Revision] = None
        after: Optional[Revision] = None
        status = "succeeded"
        error_payload: Optional[dict[str, Any]] = None
        result: dict[str, Any] = {}
        instance_locked = False
        worktree_lock: Optional[FileLock] = None
        try:
            if target == self.config.active:
                before = self._safe_revision(target)
            if exclusive:
                self.guard.acquire(
                    operation_id=operation_id,
                    action=action,
                    worktree=target,
                    actor=self.actor,
                )
                instance_locked = True
            if lock_worktree:
                worktree_lock = FileLock(
                    self.config.locks_dir / f"worktree-{target}.lock", blocking=False
                )
                try:
                    worktree_lock.acquire()
                except BusyError:
                    raise BusyError(
                        f"another operation is editing worktree {target}",
                        holder={"worktree": target},
                    ) from None
            if before is None:
                before = self._safe_revision(target)
            result = fn(operation_id, before) or {}
            if result.get("ok") is False:
                status = "succeeded"
            after = self._safe_revision(target)
        except DevControlError as exc:
            status = "failed"
            after = self._safe_revision(target)
            error_payload = {
                "code": exc.code,
                "message": redact_text(exc.message),
                "detail": redact_obj(exc.detail or {}),
                "exit_code": exc.exit_code,
            }
        except Exception as exc:  # keep the operation id visible for diagnosis
            status = "failed"
            after = self._safe_revision(target)
            error_payload = {
                "code": "INTERNAL_ERROR",
                "message": redact_text(f"{type(exc).__name__}: {exc}"),
                "detail": {},
                "exit_code": 1,
            }
        finally:
            if worktree_lock is not None:
                worktree_lock.release()
            if instance_locked:
                self.guard.release()
            revision_after = result.get("revision_after") or (
                after.label if after else (before.label if before else None)
            )
            revision_before = before.label if before else None
            record = operation_record(
                operation_id=operation_id,
                action=action,
                actor=self.actor,
                worktree=result.get("worktree") or target,
                revision_before=revision_before,
                revision_after=revision_after,
                status=status,
                request=redact_obj(request or {}),
                result=redact_obj(result),
                error=error_payload,
                started_at=started_at,
            )
            try:
                self.journal.append(record)
            except OSError:
                pass
        envelope: dict[str, Any] = {
            "ok": status == "succeeded" and result.get("ok", True) is not False,
            "operation_id": operation_id,
            "action": action,
            "worktree": target,
            "revision": (after.label if after else (before.label if before else None)),
            "revision_before": before.label if before else None,
            "revision_after": after.label if after else None,
            "revision_detail": (after or before).to_dict() if (after or before) else None,
            "status": status,
        }
        envelope.update(result)
        if error_payload is not None:
            envelope["error"] = error_payload
        return envelope

    def _safe_revision(self, name: str) -> Optional[Revision]:
        try:
            return self._safe_revision_path(self._worktree_path(name))
        except Exception:
            return None

    @staticmethod
    def _safe_revision_path(path: Path) -> Optional[Revision]:
        try:
            repo = GitRepo(path)
            repo.ensure()
            return repo.revision()
        except Exception:
            return None

    def _list_worktrees(self) -> dict[str, Any]:
        rows = []
        known = {str(item.get("path")): item for item in self.repo.worktrees()}
        worktrees_root = self.config.worktrees.resolve()
        for path_text, item in known.items():
            path = Path(path_text)
            resolved = path.resolve()
            if resolved == self.config.repo.resolve():
                name = "main"
            elif resolved.is_relative_to(worktrees_root):
                name = path.name
            else:
                # The control plane owns main plus the configured worktree
                # root only. Other git worktrees (review or harness scratch
                # trees) are not agent slots.
                continue
            revision = self._safe_revision_path(path)
            latest = self.checkpoints.latest(name)
            rows.append(
                {
                    "name": name,
                    "path": path_text,
                    "branch": item.get("branch") or "HEAD",
                    "commit": item.get("commit"),
                    "active": name == self.config.active,
                    "revision": revision.label if revision else None,
                    "checkpoint": latest.get("id") if latest else None,
                }
            )
        if not any(row["name"] == "main" for row in rows):
            revision = self._safe_revision("main")
            rows.insert(
                0,
                {
                    "name": "main",
                    "path": str(self.config.repo),
                    "branch": revision.branch if revision else "",
                    "commit": revision.commit if revision else None,
                    "active": self.config.active == "main",
                    "revision": revision.label if revision else None,
                    "checkpoint": (self.checkpoints.latest("main") or {}).get("id"),
                },
            )
        return {"worktrees": sorted(rows, key=lambda row: (row["name"] != "main", row["name"]))}

    def _inspect_payload(self) -> dict[str, Any]:
        revision = self._safe_revision(self.config.active)
        if revision is None:
            raise ConfigError(
                f"active worktree cannot be read: {self._worktree_path(self.config.active)}"
            )
        base = f"http://127.0.0.1:{self.config.backend_port}"
        live = request_json(base + "/api/live", timeout=3.0)
        ready = request_json(base + "/api/ready", timeout=3.0)
        status = request_json(base + "/api/status", token=self.config.api_token, timeout=4.0)
        gpu_info = request_json(base + "/api/gpu-info", token=self.config.api_token, timeout=6.0)
        tasks = task_snapshot(base, token=self.config.api_token, timeout=3.0)
        stack: dict[str, Any]
        try:
            stack = self.compose.summary()
        except Exception as exc:
            stack = {"error": redact_text(str(exc)), "running": False, "services": {}}
        status_data = status.data if isinstance(status.data, dict) else {}
        running = status_data.get("running_instances") if isinstance(status_data, dict) else []
        models = []
        for item in running or []:
            if not isinstance(item, dict):
                continue
            models.append(
                {
                    "model_id": item.get("model_id") or item.get("proxy_model_name"),
                    "engine": item.get("runtime_type"),
                    "port": item.get("port"),
                }
            )
        active_tasks = [
            self._task_summary(item)
            for item in tasks
            if str(item.get("status") or "").lower() in {"running", "queued", "cancelling"}
        ]
        recent_tasks = [self._task_summary(item) for item in tasks[-5:]]
        gpu = self._gpu_payload(gpu_info)
        health = {
            "live": live.ok,
            "ready": ready.ok,
            "live_status": live.status,
            "ready_status": ready.status,
            "ready_error": redact_text(ready.error) if not ready.ok else "",
            "status_authorized": status.ok,
            "status_error": redact_text(status.error) if not status.ok else "",
            "overall": "healthy" if live.ok and ready.ok and status.ok else "degraded",
        }
        if not live.ok:
            health["overall"] = "down"
        return {
            "revision": revision.label,
            "revision_detail": revision.to_dict(),
            "active_worktree": self.config.active,
            "checkout": str(self._worktree_path(self.config.active)),
            "health": health,
            "stack": stack,
            "models": models,
            "tasks": {"active": active_tasks, "recent": recent_tasks, "retained": len(tasks)},
            "gpu": gpu,
            "proxy": status_data.get("proxy_status") if isinstance(status_data, dict) else None,
            "ports": {
                "backend": self.config.backend_port,
                "frontend": self.config.frontend_port,
                "proxy": self.config.proxy_port,
            },
        }

    @staticmethod
    def _task_summary(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": item.get("task_id"),
            "type": item.get("type"),
            "status": item.get("status"),
            "description": item.get("description"),
            "progress": item.get("progress"),
        }

    def _gpu_payload(self, gpu_info: Any) -> dict[str, Any]:
        requested = self.config.gpu
        devices: list[dict[str, Any]] = []
        source = "api"
        if requested:
            result = self.compose.exec(
                "backend",
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                timeout=8,
                user=f"{self.config.uid}:{self.config.gid}",
            )
            if result.ok:
                source = "nvidia-smi"
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 5:
                        devices.append(
                            {
                                "index": parts[0],
                                "name": parts[1],
                                "utilization_percent": parts[2],
                                "memory_used_mib": parts[3],
                                "memory_total_mib": parts[4],
                            }
                        )
        if not devices:
            payload = gpu_info.data if getattr(gpu_info, "data", None) else {}
            if isinstance(payload, dict):
                for item in payload.get("gpus") or []:
                    if isinstance(item, dict):
                        devices.append(item)
            source = "api" if devices else "unavailable"
        return {"requested": requested, "source": source, "devices": devices}

    def _quick_health(self) -> dict[str, Any]:
        base = f"http://127.0.0.1:{self.config.backend_port}"
        live = request_json(base + "/api/live", timeout=2.0)
        frontend = request_text(f"http://127.0.0.1:{self.config.frontend_port}/", timeout=2.0)
        return {
            "ok": live.ok and frontend.ok,
            "backend_live": live.ok,
            "backend_status": live.status,
            "frontend": frontend.ok,
            "frontend_status": frontend.status,
        }

    def _wait_health(self, services: list[str], timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + max(5.0, timeout)
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            live = request_json(
                f"http://127.0.0.1:{self.config.backend_port}/api/live", timeout=2.0
            )
            ready = request_json(
                f"http://127.0.0.1:{self.config.backend_port}/api/ready", timeout=2.0
            )
            frontend_ok = True
            if "frontend" in services:
                frontend = request_text(
                    f"http://127.0.0.1:{self.config.frontend_port}/", timeout=2.0
                )
                frontend_ok = frontend.ok
            last = {
                "ok": live.ok and ready.ok and frontend_ok,
                "backend_live": live.ok,
                "backend_ready": ready.ok,
                "frontend": frontend_ok,
            }
            if last["ok"]:
                return last
            time.sleep(0.75)
        last.setdefault("ok", False)
        last["timeout_seconds"] = timeout
        return last

    def _capture(
        self,
        operation_id: str,
        revision: Revision,
        *,
        reason: str,
        note: str,
    ) -> dict[str, Any]:
        root = self.config.diagnostics_dir / operation_id
        root.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []

        def write(name: str, text: str, *, limit: int = 2_000_000) -> None:
            body = redact_text(text or "")
            if len(body) > limit:
                body = body[:limit] + "\n[truncated by devcontrol]\n"
            path = root / name
            atomic_write_text(path, body, mode=0o600)
            files.append({"name": name, "bytes": path.stat().st_size})

        write("revision.json", json.dumps(revision.to_dict(), indent=2))
        try:
            inspect_payload = self._inspect_payload()
            write("inspect.json", json.dumps(redact_obj(inspect_payload), indent=2, default=str))
        except Exception as exc:
            inspect_payload = {"error": redact_text(str(exc))}
            write("inspect.json", json.dumps(inspect_payload, indent=2))
        try:
            write("compose-ps.json", json.dumps(self.compose.ps(), indent=2, default=str))
        except Exception as exc:
            write("compose-ps.json", json.dumps({"error": redact_text(str(exc))}))
        app_logs = read_logs(
            self.compose,
            source="app",
            service="all",
            tail=500,
            max_bytes=256_000,
        )
        write("logs-app.txt", "\n".join(app_logs.get("lines") or []))
        build_logs = read_logs(
            self.compose,
            source="build",
            tail=500,
            max_bytes=256_000,
        )
        write("logs-build.txt", "\n".join(build_logs.get("lines") or []))
        if isinstance(inspect_payload, dict):
            for model in (inspect_payload.get("models") or [])[:3]:
                if not isinstance(model, dict):
                    continue
                model_id = str(model.get("model_id") or "").strip()
                if not model_id:
                    continue
                try:
                    model_logs = read_logs(
                        self.compose,
                        source="model",
                        model=model_id,
                        tail=300,
                        max_bytes=128_000,
                    )
                    safe = "".join(
                        char if char.isalnum() or char in "._-" else "_" for char in model_id
                    )
                    write(f"logs-model-{safe}.txt", "\n".join(model_logs.get("lines") or []))
                except Exception:
                    continue
        active_repo = GitRepo(self._worktree_path(self.config.active))
        write("git-diff.patch", active_repo.diff(max_bytes=1_000_000))
        write("untracked.txt", "\n".join(active_repo.untracked()))
        write(
            "journal-tail.jsonl",
            "\n".join(json.dumps(record, default=str) for record in self.journal.read(50)),
        )
        base = f"http://127.0.0.1:{self.config.backend_port}"
        bundle = request_json(
            base + "/api/diagnostics/bundle", token=self.config.api_token, timeout=10.0
        )
        if bundle.ok and bundle.text:
            write("server-diagnostics.json", bundle.text)
        if self.config.gpu:
            smi = self.compose.exec(
                "backend",
                ["nvidia-smi"],
                timeout=10,
                user=f"{self.config.uid}:{self.config.gid}",
            )
            write("nvidia-smi.txt", smi.stdout + "\n" + smi.stderr)
        manifest = {
            "schema": 1,
            "operation_id": operation_id,
            "reason": reason,
            "note": note,
            "created_at": utc_now(),
            "worktree": self.config.active,
            "revision": revision.to_dict(),
            "files": files,
        }
        write("manifest.json", json.dumps(manifest, indent=2))
        archive = root.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            for item in sorted(root.iterdir()):
                if item.is_file():
                    tar.add(item, arcname=f"{operation_id}/{item.name}")
        return {
            "path": str(root),
            "archive": str(archive),
            "files": files,
            "bytes": archive.stat().st_size if archive.exists() else 0,
            "reason": reason,
            "note": note,
        }


class _Guard:
    """Lazily constructed instance guard to keep import-time state minimal."""

    def __init__(self, config: Config):
        from .locks import InstanceGuard

        self._guard = InstanceGuard(config.state_home)

    def acquire(self, **kwargs: Any) -> None:
        self._guard.acquire(**kwargs)

    def release(self) -> None:
        self._guard.release()
