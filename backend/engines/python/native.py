"""Data-driven installer host for NATIVE (install-dir) engines.

Phase 7 of the YAML campaign: audio.cpp and the native llama family do not
install into a Python venv, so ``GenericVenvInstaller`` does not apply. This
host keeps the SAME flow shape (definition -> ``materialize_plan`` ->
``execute_plan``) but binds an INSTALL DIRECTORY (with a ``build`` child)
instead of a venv, and supplies the four primitives the venv host never
needed (R55's audit):

- ``publish.install``: write the engine-version row (install type, source
  fields, install dir, timestamp), record the published install snapshot and
  activate the version;
- ``publish.forget``: delete the row, honouring ``lifecycle.forget_errors``
  (swallow by default, raise when the definition declares ``raise``);
- ``artifact.materialize_cli``: create the install's CLI entry from the
  recipe's ``source``/``link_name`` inputs (absolute link, copy fallback,
  idempotent);
- ``artifact.remove_bundle``: remove an installed bundle tree, guarded so a
  recipe can never delete anything outside the engine's install root.

Everything else is the executor's shared vocabulary (archive/artifact/fs);
nothing is added to the primitive vocabulary, and no engine-ID knowledge lives
here — identity, names and policy come from the ``EngineDefinition``.

Synchronous core by design for source builds: each flow awaits ``execute_plan``
and returns the runtime's started-response shape ``{"message", "task_id"}``.
Release hosts whose route returns before a long download use
``start_release_install``, which schedules the same recipe under host-owned
operation bookkeeping; terminal publication/activation remain shared.
``retry_existing_install`` resolves the snapshot before retrying.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

import httpx

from backend.data_store import get_store
from backend.engines.activation import activate_engine_version
from backend.engines.definitions import make_engine_runner, materialize_plan
from backend.engines.definitions.executor import PrimitiveExecutionError, execute_plan
from backend.engines.definitions.schema import EngineDefinition
from backend.engines.lifecycle import mark_engine_version_ready
from backend.engines.native_support import native_binary_candidates
from backend.engines.python.installer import utcnow
from backend.logging_config import get_logger
from backend.operations.action_recovery import engine_installation_key
from backend.operations.cancellable import CancellableOperationManager
from backend.paths import studio_data_dir
from backend.proxy.control import mark_runtime_changed
from backend.utils.fs_ops import robust_rmtree


logger = get_logger(__name__)

DEFAULT_INSTALL_TYPE = "source"
VERSION_SLUG_LIMIT = 64


def _safe_slug(value: str, *, limit: int = VERSION_SLUG_LIMIT) -> str:
    """Filesystem-safe version/directory name (audio.cpp's historical slug).

    Idempotent: the result never ends with a separator, so normalizing an
    already-normalized name is a no-op (see ``_unique_version_name``).
    """
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    slug = re.sub(r"-{2,}", "-", slug).strip("-._")
    return (slug or "install")[:limit].rstrip("-._") or "install"


def _unique_version_name(store: Any, engine_id: str, base: str) -> str:
    """Normalized version name, unique for this engine, suffix never truncated.

    Normalization runs FIRST: the version name is what the install directory
    and the version row are named after. Slugging AFTER the uniqueness check
    would drop a collision suffix from a long name (64 chars + ``-<stamp>``
    truncated back to the 64-char stem) and resolve to an EXISTING install
    directory. The suffix is appended to the normalized stem with room
    reserved for it.
    """
    normalized = _safe_slug(base)
    existing = {
        str(row.get("version"))
        for row in store.get_engine_versions(engine_id)
        if row.get("version")
    }
    if normalized not in existing:
        return normalized
    stamp = int(time.time())
    for slot in range(10000):
        suffix = f"-{stamp}" if slot == 0 else f"-{stamp}-{slot}"
        stem = _safe_slug(normalized, limit=VERSION_SLUG_LIMIT - len(suffix))
        candidate = f"{stem}{suffix}"
        if candidate not in existing:
            return candidate
    return f"{_safe_slug(normalized, limit=VERSION_SLUG_LIMIT - 2)}-x"


def _within(root: str, path: str) -> bool:
    """True when ``path`` is strictly inside ``root`` (never ``root`` itself)."""
    if not root or not path:
        return False
    root_real = os.path.realpath(root)
    path_real = os.path.realpath(path)
    if path_real == root_real:
        return False
    try:
        return os.path.commonpath([root_real, path_real]) == root_real
    except ValueError:  # different drives (Windows): not comparable
        return False


def cleanup_partial_build(checkout_path: str) -> None:
    """Remove an incomplete checkout. A finished install is not this directory."""
    path = os.path.abspath(str(checkout_path))
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def _cancellation_check(cancel_event) -> Callable[[], bool]:
    return lambda: cancel_event is not None and cancel_event.is_set()


async def run_checked_command(
    run_command: Callable[..., Awaitable[Any]],
    *args: Any,
    cancelled: Callable[[], bool],
    checkout_path: str,
    cleanup_on_failure: bool = True,
    **kwargs: Any,
) -> Any:
    """Run one build command. Cancellation and failure both remove a partial checkout."""
    from backend.task_cancel_registry import TaskCancelledError

    if cancelled():
        cleanup_partial_build(checkout_path)
        raise TaskCancelledError("Build cancelled by user")
    result = await run_command(*args, **kwargs)
    if cleanup_on_failure and getattr(result, "returncode", 0) != 0:
        cleanup_partial_build(checkout_path)
        detail = getattr(result, "stderr", b"") or b""
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        raise RuntimeError(detail or "build command failed")
    return result


async def checked_streaming_command(
    host: Any,
    args: List[str],
    *,
    cancel_event: Any,
    checkout_path: str,
    cleanup_on_failure: bool = True,
    **kwargs: Any,
) -> Any:
    """``run_checked_command`` over the host's streaming runner."""
    return await run_checked_command(
        host._run_command_streaming,
        args,
        cancelled=_cancellation_check(cancel_event),
        checkout_path=checkout_path,
        cleanup_on_failure=cleanup_on_failure,
        cancel_event=cancel_event,
        **kwargs,
    )


async def checked_command(
    host: Any,
    *args: str,
    cancel_event: Any,
    checkout_path: str,
    cleanup_on_failure: bool = True,
    **kwargs: Any,
) -> Any:
    """``run_checked_command`` over the host's plain runner."""
    return await run_checked_command(
        host._run_command,
        *args,
        cancelled=_cancellation_check(cancel_event),
        checkout_path=checkout_path,
        cleanup_on_failure=cleanup_on_failure,
        **kwargs,
    )


class GenericNativeInstaller(CancellableOperationManager):
    """One install-dir (native) installer shaped entirely by an engine definition."""

    def __init__(
        self,
        definition: EngineDefinition,
        *,
        install_root: Optional[str] = None,
        log_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        lifecycle = dict(definition.lifecycle)
        self._definition = definition
        self.engine_id = definition.id
        self.label = definition.label
        self.MANAGER_NAME = definition.id
        # Versions whose publish step already marked the row ready/activated.
        # A cancellation delivered during activation must not turn that row
        # back into "failed" (the files and the active pointer are committed).
        self._published_versions: set[str] = set()
        data_root = studio_data_dir()
        self._root_dir = os.path.abspath(
            install_root
            or os.path.join(data_root, str(lifecycle.get("root_name") or definition.id))
        )
        self._log_path = os.path.abspath(
            log_path
            or os.path.join(
                data_root,
                "logs",
                str(lifecycle.get("log_name") or f"{definition.id}_install.log"),
            )
        )
        self._install_dir: Optional[str] = None
        self._build_dir: Optional[str] = None
        self._operation_store: Any = None
        self._ensure_directories()

    def _get_store(self):
        """Bookkeeping store: one bound by the caller's operation wins."""
        return self._operation_store or get_store()

    # --- Identity / layout ---------------------------------------------------------

    @property
    def definition(self) -> EngineDefinition:
        return self._definition

    @property
    def install_root(self) -> str:
        return self._root_dir

    @property
    def log_path(self) -> str:
        return self._log_path

    @property
    def install_dir(self) -> Optional[str]:
        """The installation the current flow targets (``None`` before binding)."""
        return self._install_dir

    @property
    def build_dir(self) -> Optional[str]:
        return self._build_dir

    @property
    def operation_descriptions(self) -> Dict[str, str]:
        return {
            "install": f"Install {self.label}",
            "install_source": f"Install {self.label} from Source",
            "sync_source": f"Sync {self.label} Source",
            "remove": f"Remove {self.label}",
        }

    def _ensure_directories(self) -> None:
        os.makedirs(self._root_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)

    def _lifecycle(self) -> Dict[str, Any]:
        return dict(getattr(self._definition, "lifecycle", {}) or {})

    def _default_repo_url(self) -> str:
        return str(self._lifecycle().get("default_repo_url") or "")

    def _require_repo_url(self, repo_url: Optional[str]) -> str:
        repo = str(repo_url or "").strip() or self._default_repo_url()
        if not repo:
            raise RuntimeError(
                f"{self.label} has no source repository; set default_repo_url "
                "in the engine definition or pass repo_url"
            )
        return repo

    def _raise_if_busy(self) -> None:
        if self._operation:
            raise RuntimeError(f"Another {self.label} operation is already running")

    def _claim_install_dir(
        self,
        *,
        requested_version: Optional[str],
        existing_version: Optional[str],
        reuse_dir: Optional[str],
        label: str,
    ) -> tuple[str, str]:
        """Resolve (version name, install dir) and bind them to this flow.

        Normalization happens exactly once, in the VERSION NAME; the install
        dir is derived from that name (or from ``reuse_dir``), never re-slugged
        afterwards, so a unique name can never fall back onto an existing dir.
        """
        store = self._get_store()
        if existing_version:
            version_name = str(existing_version)
            dir_name = _safe_slug(version_name)
        elif reuse_dir:
            version_name = _safe_slug(os.path.basename(os.path.abspath(reuse_dir)))
            dir_name = version_name
        else:
            version_name = _unique_version_name(
                store, self.engine_id, str(requested_version or "").strip() or label
            )
            dir_name = version_name
        install_dir = (
            os.path.abspath(reuse_dir) if reuse_dir else os.path.join(self._root_dir, dir_name)
        )
        if not _within(self._root_dir, install_dir):
            raise ValueError(
                f"{self.label} install directory {install_dir} is outside the "
                f"install root {self._root_dir}"
            )
        self._install_dir = install_dir
        self._build_dir = os.path.join(install_dir, "build")
        os.makedirs(install_dir, exist_ok=True)
        return version_name, install_dir

    def _bind_existing_install(self, install_dir: str, *, create: bool = True) -> str:
        target = os.path.abspath(install_dir)
        if not _within(self._root_dir, target):
            raise ValueError(
                f"{self.label} install directory {target} is outside the "
                f"install root {self._root_dir}"
            )
        self._install_dir = target
        self._build_dir = os.path.join(target, "build")
        if create:
            os.makedirs(target, exist_ok=True)
        return target

    def _context(self, **values: Any) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "input.engine_root": self._root_dir,
            "input.installs_root": self._root_dir,
            "input.install_dir": self._install_dir or "",
            "input.build_dir": self._build_dir or "",
        }
        context.update(values)
        return context

    async def _start_operation(
        self, operation: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        description = self.operation_descriptions.get(operation, f"Install {self.label}")
        return await self._begin_operation(
            operation, description, {"engine": self.engine_id, **(metadata or {})}
        )

    # --- Recipe execution ----------------------------------------------------------

    def _recipe(self, name: str):
        if name not in dict(self._definition.recipes):
            raise RuntimeError(f"{self.label} definition has no '{name}' recipe")
        return materialize_plan(self._definition, name)

    def _native_handlers(self) -> Dict[str, Any]:
        return {
            "artifact.materialize_cli": self._materialize_cli,
            "artifact.remove_bundle": self._remove_bundle,
            "cmake.configure": self._cmake_configure,
            "cmake.build": self._cmake_build,
            "publish.install": self._publish_install,
            "publish.record_update": self._publish_record_update,
            "publish.forget": self._publish_forget,
            "smoke.test": self._smoke_test,
        }

    def _extra_plan_handlers(self) -> Dict[str, Any]:
        """Engine-specific primitives for this recipe family.

        Specialised engines override this to REPLACE a core handler (e.g.
        audio.cpp's release archive may need a companion download for its CLI)
        while flow control, publication and bookkeeping stay inherited.
        Extras win over the core map, deliberately: the override is the
        engine's declared difference, not a fallback.
        """
        return {}

    def _plan_handler_map(self) -> Dict[str, Any]:
        return {**self._native_handlers(), **self._extra_plan_handlers()}

    @staticmethod
    def _plan_primitives(plan) -> set:
        return {str(getattr(step, "primitive", "")) for step in getattr(plan, "steps", plan)}

    def _plan_handlers(self, plan, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Override only the primitives THIS recipe uses.

        The handler surface stays equal to what the definition asks for, so a
        primitive a recipe drops cannot silently shadow the executor.
        """
        used = self._plan_primitives(plan)
        return {name: handler for name, handler in overrides.items() if name in used}

    async def _run_recipe(self, name: str, context: Mapping[str, Any]):
        plan = self._recipe(name)
        return await execute_plan(
            plan,
            run_primitive=make_engine_runner(
                self, self._plan_handlers(plan, self._plan_handler_map())
            ),
            context=context,
        )

    async def execute_operation(
        self,
        recipe: str,
        *,
        task_id: str,
        install_dir: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        store: Any = None,
        create_install_dir: bool = True,
    ):
        """Run one recipe under an EXTERNALLY owned operation.

        The caller keeps its admission, background task and terminal
        bookkeeping (progress completion/failure, notifications, failed-row
        marking); this only binds that task id for progress emitted by shared
        primitives, the install dir, and (optionally) the caller's store for
        bookkeeping — all restored afterwards. No operation is started or
        finished here — used by engines whose route still owns the task
        plumbing (audio.cpp's prebuilt install and source build, the shared
        version-removal route). ``create_install_dir=False`` lets a source
        build create the version directory itself (its workspace publisher
        refuses to replace an existing one silently).
        """
        previous_task = self._progress_task_id
        previous_install = self._install_dir
        previous_build = self._build_dir
        previous_store = self._operation_store
        self._progress_task_id = task_id or None
        self._operation_store = store
        try:
            if install_dir:
                self._bind_existing_install(install_dir, create=create_install_dir)
            return await self._run_recipe(recipe, dict(context or {}))
        finally:
            self._progress_task_id = previous_task
            self._install_dir = previous_install
            self._build_dir = previous_build
            self._operation_store = previous_store

    def _raise_if_cancelled(self) -> None:
        from backend.task_cancel_registry import TaskCancelledError, is_task_cancel_requested

        if self._progress_task_id and is_task_cancel_requested(self._progress_task_id):
            raise TaskCancelledError(f"{self.label} build cancelled")

    async def _git_head(self, clone_dir: str) -> Optional[str]:
        """HEAD commit of a source checkout (the `vcs.capture_commit` hook)."""
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=clone_dir,
            )
            stdout, _ = await process.communicate()
            if process.returncode != 0:
                return None
            return stdout.decode("utf-8", errors="replace").strip() or None
        except Exception:  # noqa: BLE001 — a missing checkout must not fail publication
            logger.debug("Could not read %s source HEAD", self.label, exc_info=True)
            return None

    def _require_completed(self, execution, what: str) -> None:
        if execution.status == "completed":
            return
        failure = execution.failed_step
        raise RuntimeError(
            failure.error if failure is not None and failure.error else f"{what} did not complete"
        )

    async def _run_flow(
        self,
        operation: str,
        message: str,
        metadata: Dict[str, Any],
        runner,
    ) -> Dict[str, Any]:
        """Start the operation, await the synchronous core, finish it, respond.

        The response shape is the routes' started contract; because the core is
        synchronous the operation has already reached its terminal state.
        """
        task_id = await self._start_operation(operation, metadata)
        try:
            await runner()
        except asyncio.CancelledError:
            self._last_error = "Operation cancelled by user"
            await self._finish_operation(False, self._last_error, cancelled=True)
            raise
        except Exception as exc:
            self._last_error = str(exc)
            await self._finish_operation(False, str(exc))
            raise
        await self._finish_operation(True, message)
        return {"message": message, "task_id": task_id}

    # --- Handlers (R55: the four primitives the venv host never needed) -------------

    def _row_extras(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        """Engine-specific published-row fields (plan-derived provenance).

        Merged into the row by `publish.install` after the generic fields and
        the recipe's `<field>_from` declarations.
        """
        return {}

    async def _publish_install(self, step, requirements, context) -> Dict[str, Any]:
        """`publish.install`: authoritative install-dir version row + activation.

        Recipe-declared path fields use the shared ``<field>_from: <produced>``
        convention (e.g. ``server_binary_path_from: server``), so an engine's
        active-path fields reach the row without engine code here.
        """
        version_name = str(context.get("input.version_name") or "").strip()
        if not version_name:
            raise PrimitiveExecutionError("publish.install requires input.version_name")
        install_dir = str(context.get("input.install_dir") or self._install_dir or "")
        install_type = (
            str(context.get("input.install_type") or DEFAULT_INSTALL_TYPE).strip().lower()
        )
        row: Dict[str, Any] = {
            "version": version_name,
            "type": install_type,
            "install_type": install_type,
            "install_dir": install_dir,
            "installed_at": utcnow(),
            "repository_source": self._definition.repository_source,
        }
        for field, key in (
            ("source_repo", "input.source_repo"),
            ("source_ref", "input.source_ref"),
            ("source_ref_type", "input.source_ref_type"),
        ):
            value = str(context.get(key) or "").strip()
            if value:
                row[field] = value
        if row.get("source_ref_type") == "branch":
            branch = str(context.get("input.branch") or "").strip()
            if branch:
                row["source_branch"] = branch
        # Installed-path row fields are declared as `<field>_from: <produced>`:
        # audio's server/cli binaries are active-path fields, and the `_from`
        # convention is the one every recipe primitive already uses.
        for key, source_key in step.with_.items():
            name = str(key)
            if not name.endswith("_from"):
                continue
            value = context.get(str(source_key))
            if value:
                row[name[: -len("_from")]] = value
        # Engine-declared row fields last: plan-derived provenance (audio's
        # cuda_version/build_config) that cannot be expressed as a `with` scalar.
        row.update(self._row_extras(context))
        mark_engine_version_ready(self._get_store(), self.engine_id, row)
        self._published_versions.add(version_name)
        self._record_published_snapshot(
            version_name,
            {
                "install_type": install_type,
                "install_dir": install_dir,
                "version": version_name,
                "requested_version": str(context.get("input.requested_version") or ""),
                "source_repo": row.get("source_repo"),
                "source_ref": row.get("source_ref"),
            },
        )
        if context.get("input.auto_activate", True):
            await activate_engine_version(
                self.engine_id,
                version_name,
                covered_by=self.progress_task_id,
                row=row,
            )
        return {}

    async def _publish_forget(self, step, requirements, context) -> Dict[str, Any]:
        """`publish.forget`: delete the version row under the declared policy.

        ``input.registered`` (removal flows set it) gates the store call, so a
        synthetic/orphan row whose row was never persisted is left untouched.
        """
        version_name = str(context.get("input.version_name") or "").strip()
        registered = bool(context.get("input.registered", True))
        policy = str(self._lifecycle().get("forget_errors") or "swallow").strip().lower()
        if version_name and registered:
            try:
                store = self._get_store()
                store.delete_engine_version(self.engine_id, version_name)
                delete_snapshots = getattr(store, "delete_install_snapshots", None)
                if callable(delete_snapshots):
                    delete_snapshots(self.engine_id, version_name)
            except Exception as exc:  # noqa: BLE001 — the policy decides
                # Default keeps the historical swallow-everything behavior;
                # `forget_errors: raise` is the strict engine policy.
                if policy == "raise":
                    raise
                logger.debug("Failed to delete %s engine version metadata: %s", self.label, exc)
        mark_runtime_changed()
        return {}

    async def _publish_record_update(self, step, requirements, context) -> Dict[str, Any]:
        """`publish.record_update`: merge a REBUILT row without re-dating it.

        Unlike `publish.install` this updates an existing installation in place:
        ``installed_at`` is preserved, ``updated_at`` moves, and activation stays
        with the caller (a sync re-activates under its own policy).
        """
        version_name = str(context.get("input.version_name") or "").strip()
        if not version_name:
            raise PrimitiveExecutionError("publish.record_update requires input.version_name")
        row: Dict[str, Any] = {"version": version_name, "updated_at": utcnow()}
        for field, key in (
            ("source_repo", "input.source_repo"),
            ("source_ref", "input.source_ref"),
            ("source_ref_type", "input.source_ref_type"),
        ):
            value = str(context.get(key) or "").strip()
            if value:
                row[field] = value
        if row.get("source_ref_type") == "branch":
            branch = str(context.get("input.branch") or "").strip()
            if branch:
                row["source_branch"] = branch
        for key, source_key in step.with_.items():
            name = str(key)
            if not name.endswith("_from"):
                continue
            value = context.get(str(source_key))
            if value:
                row[name[: -len("_from")]] = value
        row.update(self._row_extras(context))
        if self._get_store().update_engine_version(self.engine_id, version_name, row) is None:
            raise PrimitiveExecutionError(f"Version '{version_name}' disappeared during the update")
        mark_runtime_changed()
        return {}

    def _publish_source_version(
        self,
        mode: str,
        row: Mapping[str, Any],
        *,
        missing_error: str = "Version '{version}' disappeared during the update",
    ) -> None:
        """Write a SOURCE-build version row in ``install`` or ``update`` mode.

        ``install`` upserts the row and marks it ready (a new installation);
        ``update`` merges a rebuilt row in place: ``installed_at`` is preserved,
        ``updated_at`` moves, and a row that vanished is an error. The caller
        owns the provenance fields (labels, resolved commit, build config),
        activation policy and install snapshots — these are the source-build
        semantics every native source host would otherwise re-implement.
        """
        version_name = str(row.get("version") or "").strip()
        if not version_name:
            raise PrimitiveExecutionError("a source version row requires a version name")
        payload = dict(row)
        if str(mode or "").strip().lower() == "update":
            payload.pop("installed_at", None)
            payload["updated_at"] = utcnow()
            if (
                self._get_store().update_engine_version(self.engine_id, version_name, payload)
                is None
            ):
                raise PrimitiveExecutionError(missing_error.format(version=version_name))
            return
        payload.setdefault("installed_at", utcnow())
        mark_engine_version_ready(self._get_store(), self.engine_id, payload)

    async def _materialize_cli(self, step, requirements, context) -> Dict[str, Any]:
        """`artifact.materialize_cli`: absolute, idempotent CLI entry in the install dir."""
        source_key = str(step.with_.get("source") or "cli")
        source = str(context.get(source_key) or "").strip()
        if not source or not os.path.isfile(source):
            raise PrimitiveExecutionError(
                f"artifact.materialize_cli found no CLI binary at {source or source_key!r}"
            )
        install_dir = str(context.get("input.install_dir") or self._install_dir or "")
        if not install_dir:
            raise PrimitiveExecutionError("artifact.materialize_cli requires input.install_dir")
        link_name = os.path.basename(str(step.with_.get("link_name") or "")) or os.path.basename(
            source
        )
        destination = os.path.abspath(os.path.join(install_dir, link_name))
        created = self._materialize_entry(source, destination)
        return {"cli_final": destination, "cli_created": created}

    @staticmethod
    def _materialize_entry(source: str, destination: str) -> bool:
        """Create a CLI entry; True when created, False when already current.

        Symlink when the platform allows it (the recorded path is absolute),
        executable copy otherwise. An existing regular file is current — the
        install dir is versioned, so it cannot belong to another version — while
        a symlink pointing somewhere else IS replaced.
        """
        source = os.path.abspath(source)
        destination = os.path.abspath(destination)
        if os.path.islink(destination):
            if os.path.realpath(destination) == os.path.realpath(source):
                return False
            os.unlink(destination)
        elif os.path.lexists(destination):
            return False
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        try:
            os.symlink(source, destination)
        except (AttributeError, NotImplementedError, OSError):
            shutil.copy2(source, destination)
            os.chmod(destination, os.stat(destination).st_mode | 0o111)
        return True

    async def _remove_bundle(self, step, requirements, context) -> Dict[str, Any]:
        """`artifact.remove_bundle`: delete one installed bundle, root-guarded."""
        key = str(step.with_.get("path_from") or "input.install_dir")
        target = str(context.get(key) or context.get("input.install_dir") or "").strip()
        if not target:
            raise PrimitiveExecutionError("artifact.remove_bundle requires a bundle path")
        if not _within(self._root_dir, target):
            # Deletion containment: a recipe can never remove files outside the
            # engine's install root, and the root itself is not a bundle.
            logger.warning(
                "Refusing to remove %s bundle outside %s: %s",
                self.label,
                self._root_dir,
                target,
            )
            return {"removed": False}
        retire_references = bool(context.get("input.retire_references"))
        if os.path.isdir(target):
            robust_rmtree(target, retire_references=retire_references)
        elif os.path.isfile(target) or os.path.islink(target):
            os.remove(target)
        else:
            return {"removed": False}
        return {"removed": True}

    # --- Build commands (shared streaming runner) -----------------------------------

    @staticmethod
    def _declared_argv(step, context: Mapping[str, Any]) -> List[str]:
        """Static ``argv`` list, or the list named by ``argv_from``."""
        declared = step.with_.get("argv")
        if declared:
            return [str(token) for token in declared]
        key = str(step.with_.get("argv_from") or "")
        value = context.get(key) if key else None
        if isinstance(value, (list, tuple)):
            return [str(token) for token in value]
        return [str(value)] if value else []

    @staticmethod
    def _declared_cwd(step, context: Mapping[str, Any]) -> Optional[str]:
        name = str(step.with_.get("cwd_from") or "")
        value = str(context.get(name) or "") if name else ""
        return value or None

    @staticmethod
    def _declared_env(step, context: Mapping[str, Any]) -> Optional[Dict[str, str]]:
        name = str(step.with_.get("env_from") or "")
        if not name:
            return None
        value = context.get(name)
        return dict(value) if isinstance(value, Mapping) else None

    async def _run_cmake_step(self, step, context: Mapping[str, Any], *, default_stage: str):
        stage = str(step.with_.get("stage") or default_stage)
        argv = self._declared_argv(step, context)
        if not argv:
            raise PrimitiveExecutionError(f"{stage} requires an argv (argv or argv_from)")
        try:
            await self._run_build_command(
                argv,
                stage=stage,
                cwd=self._declared_cwd(step, context),
                env=self._declared_env(step, context),
                progress=int(step.with_.get("progress") or 0),
                base_message=str(step.with_.get("progress_message") or ""),
            )
        except RuntimeError as exc:
            raise PrimitiveExecutionError(str(exc)) from exc
        return {}

    async def _cmake_configure(self, step, requirements, context) -> Dict[str, Any]:
        """`cmake.configure`: run the recipe-declared configure argv."""
        return await self._run_cmake_step(step, context, default_stage="configure")

    async def _cmake_build(self, step, requirements, context) -> Dict[str, Any]:
        """`cmake.build`: run the recipe-declared build argv."""
        return await self._run_cmake_step(step, context, default_stage="build")

    async def _smoke_test(self, step, requirements, context) -> Dict[str, Any]:
        """`smoke.test`: run a produced/declared command and require output."""
        argv = self._declared_argv(step, context)
        if not argv:
            raise PrimitiveExecutionError("smoke.test requires an argv (argv or argv_from)")
        try:
            output = await self._capture_command(
                argv,
                cwd=self._declared_cwd(step, context),
                env=self._declared_env(step, context),
            )
        except RuntimeError as exc:
            raise PrimitiveExecutionError(str(exc)) from exc
        if step.with_.get("require_output", True) and not output.strip():
            raise PrimitiveExecutionError(
                str(step.with_.get("error") or f"{argv[0]} returned empty output")
            )
        return {}

    async def _run_build_command(
        self,
        argv: Sequence[str],
        *,
        stage: str,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        progress: int = 0,
        base_message: str = "",
    ) -> List[str]:
        """Stream one build command through the shared native runner."""
        from backend.engines.native_build import run_streaming_command
        from backend.task_cancel_registry import is_task_cancel_requested

        task_id = self._progress_task_id
        return await run_streaming_command(
            argv,
            stage=stage,
            progress=progress,
            base_message=base_message or stage,
            emit=self._emit_build_progress,
            is_cancelled=lambda: bool(task_id) and is_task_cancel_requested(task_id),
            terminate=self._terminate_active_process,
            on_spawn=self._track_process,
            on_finish=self._clear_active_process,
            cancel_message=f"{self.label} build cancelled",
            cwd=cwd,
            env=env,
            command_label=self.label,
        )

    async def _emit_build_progress(
        self, stage: str, progress: int, message: str, new_lines: List[str], all_lines: List[str]
    ) -> None:
        from backend.engines.native_build import emit_build_progress
        from backend.operations.progress import get_progress_manager

        emit_build_progress(
            get_progress_manager(),
            self._progress_task_id,
            stage,
            progress,
            message,
            new_lines=new_lines,
            all_lines=all_lines,
        )

    async def _terminate_active_process(self) -> None:
        process = self._active_process
        if not process or process.returncode is not None:
            return
        from backend.operations.cancel import terminate_process_tree

        if process.pid:
            await asyncio.to_thread(terminate_process_tree, process.pid)
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()

    @staticmethod
    async def _capture_command(
        argv: Sequence[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> str:
        process = await asyncio.create_subprocess_exec(
            *[str(a) for a in argv],
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await process.communicate()
        text = (stdout or b"").decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            raise RuntimeError(text or f"{argv[0]} exited {process.returncode}")
        return text

    def _record_snapshot(self, version_name: str, resolved: Dict[str, Any], *, phase: str) -> None:
        try:
            digest = str(getattr(self._definition, "definition_digest", "") or "")
            if not digest:
                return
            self._get_store().record_install_snapshot(
                self.engine_id,
                definition_digest=digest,
                recipe_version=f"schema-v{self._definition.schema_version}",
                resolved_inputs={
                    key: value for key, value in resolved.items() if value is not None
                },
                version=version_name,
                phase=phase,
            )
        except Exception:  # noqa: BLE001 — bookkeeping must not fail installs
            logger.debug("install snapshot %s failed for %s", phase, self.engine_id, exc_info=True)

    def _record_published_snapshot(self, version_name: str, resolved: Dict[str, Any]) -> None:
        """Record the authoritative snapshot once the version exists."""
        self._record_snapshot(version_name, resolved, phase="published")

    def _record_pending_snapshot(self, version_name: str, resolved: Dict[str, Any]) -> None:
        """Associate the pending row with the definition before execution.

        A failed install must still refuse a retry after the definition changed;
        writing only on publication leaves failed attempts unguarded.
        """
        self._record_snapshot(version_name, resolved, phase="scheduled")

    # --- Public interface (EngineInstaller surface used by routes) -------------------

    def _register_pending_install(self, version_name: str, install_dir: str, task_id: str) -> None:
        """Hook for hosts whose UI needs a pending row before publication."""

    def _install_failed(self, version_name: str, error: str, install_dir: str) -> None:
        """Hook for hosts that keep a pending row to mark failed/cancelled."""

    def _fail_cancelled_install(self, version_name: str, install_dir: str) -> bool:
        """Mark a cancelled install failed only when publish never committed.

        The publish step writes the row, the snapshot and (for auto-activating
        recipes) the active pointer. A cancellation delivered while activation
        is still awaiting must not rewrite that committed row to failed.
        """
        if version_name in self._published_versions:
            return False
        self._install_failed(version_name, "Operation cancelled by user", install_dir)
        return True

    def _begin_install_attempt(self, version_name: str) -> None:
        """Clear a previous attempt's publication marker for this version.

        Hosts are long-lived singletons; without this, a failed reinstall of a
        version that was published once would inherit the old marker and skip
        failure marking on cancellation.
        """
        self._published_versions.discard(version_name)

    async def start_release_install(
        self,
        version: Optional[str] = None,
        *,
        version_resolver: Optional[Callable[[], Awaitable[Optional[str]]]] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start the ``release`` recipe in the background and return at once.

        Release hosts whose routes return a started response (Unsloth) use this
        instead of the synchronous ``install_release``. Publication, snapshots,
        activation and terminal bookkeeping stay in the shared recipe/host.
        """
        async with self._lock:
            self._raise_if_busy()
            if not version and version_resolver is not None:
                version = await version_resolver()
            version_name, install_dir = self._claim_install_dir(
                requested_version=version,
                existing_version=None,
                reuse_dir=None,
                label="release",
            )
            # Publication marks are per attempt, not per version name: a failed
            # reinstall of a previously published version must not inherit the
            # earlier attempt's marker.
            self._begin_install_attempt(version_name)
            self._record_pending_snapshot(
                version_name,
                {
                    "install_type": "release",
                    "requested_version": str(version or ""),
                    "release_tag": str(version or ""),
                    "install_dir": install_dir,
                },
            )
            context = self._context(
                **{
                    **(dict(extra_context or {})),
                    "input.version_name": version_name,
                    "input.install_type": "release",
                    "input.requested_version": str(version or ""),
                    "input.source_ref": str(version or version_name),
                    "input.source_ref_type": "release",
                    "input.auto_activate": True,
                }
            )
            task_id = await self._start_operation(
                "install",
                {
                    "version_name": version_name,
                    "install_dir": install_dir,
                    "install_type": "release",
                    "resource_key": engine_installation_key(self.engine_id, install_dir),
                },
            )
            self._register_pending_install(version_name, install_dir, task_id)

            async def runner() -> None:
                try:
                    execution = await self._run_recipe("release", context)
                    self._require_completed(execution, f"{self.label} installation")
                except asyncio.CancelledError:
                    self._last_error = "Operation cancelled by user"
                    self._fail_cancelled_install(version_name, install_dir)
                    await self._finish_operation(False, self._last_error, cancelled=True)
                    raise
                except Exception as exc:
                    self._last_error = str(exc)
                    self._install_failed(version_name, str(exc), install_dir)
                    await self._finish_operation(False, str(exc))
                    return
                await self._finish_operation(True, f"Installed {self.label} {version_name}")

            self._create_task(runner())
            return self._started_response(f"{self.label} installation started")

    async def install_release(
        self,
        version: Optional[str] = None,
        force_reinstall: bool = False,
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the ``release`` recipe into a new (or reused) install dir.

        ``extra_context`` is where an engine supplies resolution outputs the
        host cannot invent (a release asset plan/URL). It is merged FIRST, so
        it can add context keys but never override the host's own bindings.
        """
        async with self._lock:
            self._raise_if_busy()
            version_name, install_dir = self._claim_install_dir(
                requested_version=version,
                existing_version=existing_version,
                reuse_dir=reuse_dir,
                label="release",
            )
            context = self._context(
                **{
                    **(dict(extra_context or {})),
                    "input.version_name": version_name,
                    "input.install_type": "release",
                    "input.requested_version": str(version or ""),
                    "input.force_reinstall": bool(force_reinstall),
                    "input.source_ref": str(version or version_name),
                    "input.source_ref_type": "release",
                    "input.auto_activate": True,
                }
            )

            async def run() -> None:
                execution = await self._run_recipe("release", context)
                self._require_completed(execution, f"{self.label} installation")

            return await self._run_flow(
                "install",
                f"{self.label} installed",
                {
                    "version_name": version_name,
                    "install_dir": install_dir,
                    "install_type": "release",
                    "resource_key": engine_installation_key(self.engine_id, install_dir),
                },
                run,
            )

    async def install_from_source(
        self,
        repo_url: Optional[str] = None,
        branch: str = "main",
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        repo_url = self._require_repo_url(repo_url)
        async with self._lock:
            self._raise_if_busy()
            version_name, install_dir = self._claim_install_dir(
                requested_version=None,
                existing_version=existing_version,
                reuse_dir=reuse_dir,
                label=f"source-{_safe_slug(branch)}",
            )
            context = self._context(
                **{
                    **(dict(extra_context or {})),
                    "input.version_name": version_name,
                    "input.install_type": "source",
                    "input.repo_url": repo_url,
                    "input.repository_url": repo_url,
                    "input.source_repo": repo_url,
                    "input.branch": branch,
                    "input.reuse_mode": bool(existing_version or reuse_dir),
                    "input.source_ref": branch,
                    "input.source_ref_type": "branch",
                    "input.auto_activate": True,
                }
            )

            async def run() -> None:
                execution = await self._run_recipe("source", context)
                self._require_completed(execution, f"{self.label} source install")

            return await self._run_flow(
                "install_source",
                f"{self.label} install from source started",
                {
                    "version_name": version_name,
                    "install_dir": install_dir,
                    "install_type": "source",
                    "resource_key": engine_installation_key(self.engine_id, install_dir),
                },
                run,
            )

    def _install_snapshot_for(self, version_name: str) -> Dict[str, Any]:
        """Snapshot recorded for THIS installation, or {} (shared selection)."""
        from backend.engines.install_snapshots import install_snapshot_for

        return install_snapshot_for(self._get_store(), self.engine_id, version_name)

    async def retry_existing_install(self, version_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Retry through the snapshot ledger, refusing a changed definition."""
        version_entry = version_entry or {}
        version_name = str(version_entry.get("version") or "").strip()
        install_dir = str(version_entry.get("install_dir") or "").strip()
        if not version_name or not install_dir:
            raise ValueError(f"This {self.label} version does not have enough metadata to retry")
        from backend.engines.install_snapshots import retry_resolved_inputs

        frozen = retry_resolved_inputs(
            self._get_store(),
            self.engine_id,
            version_name,
            current_digest=str(getattr(self._definition, "definition_digest", "") or ""),
            label=self.label,
        )
        kind = (
            str(version_entry.get("install_type") or version_entry.get("type") or "")
            .strip()
            .lower()
        )
        if kind in {"source", "fork", "patched", "local"}:
            repo = str(
                version_entry.get("source_repo")
                or frozen.get("source_repo")
                or self._default_repo_url()
            ).strip()
            ref = str(
                version_entry.get("source_branch")
                or version_entry.get("source_ref")
                or frozen.get("source_ref")
                or frozen.get("branch")
                or "main"
            ).strip()
            return await self.install_from_source(
                repo, ref, reuse_dir=install_dir, existing_version=version_name
            )
        frozen_version = str(frozen.get("version") or "")
        if frozen_version == version_name:
            frozen_version = ""
        version = (
            version_entry.get("package_version")
            or version_entry.get("release_tag")
            or frozen.get("requested_version")
            or frozen_version
        )
        return await self.install_release(
            version=str(version) if version else None,
            force_reinstall=True,
            reuse_dir=install_dir,
            existing_version=version_name,
        )

    async def sync_source_version(self, version_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Rebuild an existing source install from its recorded branch."""
        version_entry = version_entry or {}
        version_name = str(version_entry.get("version") or "").strip()
        install_dir = str(version_entry.get("install_dir") or "").strip()
        branch = str(
            version_entry.get("source_branch") or version_entry.get("source_ref") or ""
        ).strip()
        kind = (
            str(version_entry.get("install_type") or version_entry.get("type") or "")
            .strip()
            .lower()
        )
        if kind not in {"source", "fork"}:
            raise RuntimeError(f"Only {self.label} source installs can be synced")
        if not branch:
            raise RuntimeError(f"{self.label} source install is missing source_branch")
        if not version_name or not install_dir:
            raise RuntimeError(f"{self.label} source install metadata is incomplete")
        repo_url = str(version_entry.get("source_repo") or "").strip() or self._default_repo_url()
        active = self._get_store().get_active_engine_version(self.engine_id)
        # Legacy engine semantics: a sync re-activates only the version that was
        # already active; syncing an inactive install must not switch the engine.
        auto_activate = not active or str(active.get("version")) == version_name

        async with self._lock:
            self._raise_if_busy()
            bound = self._bind_existing_install(install_dir)
            context = self._context(
                **{
                    "input.version_name": version_name,
                    "input.install_type": "source",
                    "input.repo_url": repo_url,
                    "input.repository_url": repo_url,
                    "input.source_repo": repo_url,
                    "input.branch": branch,
                    "input.source_ref": branch,
                    "input.source_ref_type": "branch",
                    "input.auto_activate": bool(auto_activate),
                }
            )

            async def run() -> None:
                execution = await self._run_recipe("sync", context)
                self._require_completed(execution, f"{self.label} source sync")

            return await self._run_flow(
                "sync_source",
                f"{self.label} source sync started",
                {
                    "version_name": version_name,
                    "install_dir": bound,
                    "install_type": "source",
                    "resource_key": engine_installation_key(self.engine_id, bound),
                },
                run,
            )

    async def sync_source(self, version_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Protocol alias for :meth:`sync_source_version`."""
        return await self.sync_source_version(version_entry)

    async def remove(self, retire_references: bool = False) -> Dict[str, Any]:
        """Remove the ACTIVE installation: its bundle tree and its version row."""
        store = self._get_store()
        active = store.get_active_engine_version(self.engine_id)
        version_name = str((active or {}).get("version") or "").strip()
        install_dir = str((active or {}).get("install_dir") or self._install_dir or "").strip()
        if not install_dir:
            raise RuntimeError(f"{self.label} has no active installation to remove")

        async with self._lock:
            self._raise_if_busy()
            bound = self._bind_existing_install(install_dir, create=False)
            context = self._context(
                **{
                    "input.version_name": version_name,
                    "input.install_type": str(
                        (active or {}).get("install_type") or (active or {}).get("type") or ""
                    )
                    or DEFAULT_INSTALL_TYPE,
                    "input.delete_target": bound,
                    "input.retire_references": bool(retire_references),
                    "input.registered": bool(active),
                }
            )

            async def run() -> None:
                execution = await self._run_recipe("remove", context)
                self._require_completed(execution, f"{self.label} removal")

            result = await self._run_flow(
                "remove",
                f"{self.label} removed",
                {
                    "version_name": version_name,
                    "install_dir": bound,
                    "resource_key": engine_installation_key(self.engine_id, bound),
                },
                run,
            )
            self._install_dir = None
            self._build_dir = None
            return result

    # --- Introspection -------------------------------------------------------------

    def _resolve_binary_path(self, row: Mapping[str, Any], install_dir: str) -> Optional[str]:
        for key in ("binary_path", "server_binary_path", "cli_binary_path"):
            value = str(row.get(key) or "").strip()
            if value and os.path.isfile(value):
                return os.path.abspath(value)
        name = str(self._lifecycle().get("binary_name") or self.engine_id)
        for root in (os.path.join(install_dir, "build") if install_dir else "", install_dir):
            if not root:
                continue
            for candidate in native_binary_candidates(root, name):
                if os.path.isfile(candidate):
                    return os.path.abspath(candidate)
        return None

    def status(self) -> Dict[str, Any]:
        active = dict(self._get_store().get_active_engine_version(self.engine_id) or {})
        install_dir = str(active.get("install_dir") or self._install_dir or "")
        return {
            "installed": bool(active),
            "version": active.get("version"),
            "binary_path": self._resolve_binary_path(active, install_dir),
            "install_dir": install_dir or None,
            "installed_at": active.get("installed_at"),
            "operation": self._operation,
            "operation_started_at": self._operation_started_at,
            "progress_task_id": self._progress_task_id,
            "last_error": self._last_error,
            "log_path": self._log_path,
            "install_type": active.get("install_type") or active.get("type"),
            "source_repo": active.get("source_repo"),
            "source_branch": active.get("source_branch") or active.get("source_ref"),
        }

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

    async def _download_file(self, url: str, dest: str) -> None:
        """Stream one artifact to ``dest`` (HTTP(S) only, shared archive contract)."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise PrimitiveExecutionError(f"Refusing download from a non-HTTP URL: {url}")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(dest, "wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)
