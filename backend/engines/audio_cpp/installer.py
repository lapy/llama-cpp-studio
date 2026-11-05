"""audio.cpp on the definition-driven native host (Phase 7).

The ``audio_cpp`` definition owns the release (``prebuilt``), removal and
source-building recipes; this module supplies ONLY what is genuinely
audio-specific, delegating to the existing helpers so there is no second
implementation:

- ``_download_file``: the streaming release downloader with its HTTP(S) guard
  (``prebuilt.stream_release_download``), which ``archive.download`` uses;
- ``artifact.materialize_cli``: the Ubuntu CUDA archive ships the server only,
  so a missing CLI is fetched from the SAME release's CPU companion archive and
  copied next to the server (``prebuilt.materialize_cli``, progress 55);
- ``artifact.remove_bundle``: the manager-tracked version-bundle removal
  (row-resolved, builds-root guarded, launch hold released);
- source steps: managed ``BuildWorkspace`` or a plain recursive clone
  (``vcs.workspace_checkout``), the manager's configure/build argv+env
  (``cmake.configure``/``cmake.build``), workspace-layout publication
  (``vcs.seal_checkout``) and the built binaries' ``--help`` gate
  (``smoke.test``);
- ``_row_extras``: prebuilt provenance from the resolved ``PrebuiltPlan``, or
  the source-build row fields (labels, source path/commit, config, manager
  paths).

The route keeps its task/admission/notification plumbing and calls
``GenericNativeInstaller.execute_operation``; nothing here schedules tasks.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from backend.engines.definitions import load_default_engine_definitions
from backend.engines.definitions.executor import PrimitiveExecutionError
from backend.engines.definitions.schema import EngineDefinition
from backend.engines.python.native import GenericNativeInstaller
from backend.paths import studio_data_dir


ENGINE_ID = "audio_cpp"


def audio_cpp_definition() -> EngineDefinition:
    definition = load_default_engine_definitions().get(ENGINE_ID)
    if definition is None:
        raise RuntimeError("audio.cpp engine definition is unavailable")
    return definition


class AudioCppInstaller(GenericNativeInstaller):
    """audio.cpp's native host: definition-driven flows + its prebuilt deltas."""

    def __init__(
        self,
        definition: Optional[EngineDefinition] = None,
        *,
        install_root: Optional[str] = None,
        log_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            definition or audio_cpp_definition(),
            install_root=install_root or os.path.join(studio_data_dir(), "audio-cpp", "builds"),
            log_path=log_path,
        )
        self._active_workspace: Any = None

    @property
    def active_workspace(self) -> Any:
        """The managed build workspace held by a running source build."""
        return self._active_workspace

    def _release_active_workspace(self) -> None:
        workspace = self._active_workspace
        self._active_workspace = None
        if workspace is not None:
            from backend.engines.build_workspace import release_held

            release_held(workspace)

    async def _download_file(self, url: str, dest: str) -> None:
        # Late import: the prebuilt module owns the streaming downloader and
        # its non-HTTP guard (pinned by the prebuilt contract tests).
        from backend.engines.audio_cpp import prebuilt

        await prebuilt.stream_release_download(url, dest)

    def _extra_plan_handlers(self) -> Dict[str, Any]:
        return {
            "artifact.materialize_cli": self._materialize_cli_from_companion,
            "artifact.remove_bundle": self._remove_bundle,
            # Source/sync deltas: the definition owns the step order, the
            # manager's existing helpers own the argv/env/toolchain details.
            "cmake.build": self._build_source_tree,
            "cmake.configure": self._configure_source,
            "host.inspect": self._inspect_sync_target,
            "smoke.test": self._validate_binary,
            "vcs.seal_checkout": self._seal_source,
            "vcs.sync": self._sync_checkout,
            "vcs.workspace_checkout": self._checkout_source,
        }

    async def _inspect_sync_target(self, step, requirements, context) -> Dict[str, Any]:
        """`host.inspect` for a sync: validate the row's checkout and build dir."""
        from backend.engines.audio_cpp.manager import get_audio_cpp_manager

        manager = get_audio_cpp_manager()
        entry = context.get("input.version_entry")
        entry = entry if isinstance(entry, Mapping) else {}
        version_name = str(entry.get("version") or context.get("input.version_name") or "").strip()
        raw_source = str(entry.get("source_path") or "").strip()
        if not version_name or not raw_source:
            raise PrimitiveExecutionError("audio.cpp source install metadata is incomplete")
        source_dir = os.path.abspath(raw_source)
        if not os.path.isdir(os.path.join(source_dir, ".git")):
            raise PrimitiveExecutionError(f"Existing source checkout not found: {source_dir}")
        if not manager._allowed_sync_source_dir(source_dir, entry):
            raise PrimitiveExecutionError(
                "Refusing to sync audio.cpp files outside the builds root"
            )
        build_dir = manager._resolve_sync_build_dir(entry, source_dir)
        return {"checkout_work": source_dir, "build_work": build_dir}

    async def _sync_checkout(self, step, requirements, context) -> Dict[str, Any]:
        """`vcs.sync`: fetch/reset/submodule the recorded checkout."""
        from backend.engines.audio_cpp.manager import sync_git_checkout

        source_dir = str(context.get("checkout_work") or "")
        branch = str(context.get("input.branch") or "")
        if not source_dir:
            raise PrimitiveExecutionError("vcs.sync requires a checkout")

        async def run(argv, *, stage, progress):
            await self._run_build_command(
                argv,
                stage=stage,
                cwd=source_dir,
                progress=progress,
                base_message="Syncing audio.cpp",
            )

        async def emit(stage, progress, message):
            await self._update_progress_task(progress, message)

        try:
            await sync_git_checkout(
                source_dir=source_dir,
                branch=branch,
                run=run,
                emit=emit,
                raise_if_cancelled=self._raise_if_cancelled,
            )
        except ValueError as exc:
            raise PrimitiveExecutionError(str(exc)) from exc
        return {}

    # --- Source build (recipe steps) -------------------------------------------------

    @staticmethod
    def _build_config(context: Mapping[str, Any]):
        from backend.engines.audio_cpp.manager import AudioCppBuildConfig

        raw = context.get("input.build_config")
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        allowed = set(AudioCppBuildConfig.__dataclass_fields__)
        return AudioCppBuildConfig(**{k: v for k, v in raw.items() if k in allowed}).normalized()

    async def _checkout_source(self, step, requirements, context) -> Dict[str, Any]:
        """`vcs.workspace_checkout`: managed workspace or a plain recursive clone."""
        from dataclasses import asdict

        from backend.build_progress import cmake_stage_start
        from backend.engines.audio_cpp.manager import AUDIO_CPP_DEFAULT_REF
        from backend.engines.build_workspace import BuildWorkspace
        from backend.git_https import git_argv

        version_dir = str(context.get("input.install_dir") or "")
        repo_url = str(context.get("input.repository_url") or "")
        ref = str(context.get("input.source_ref") or "") or AUDIO_CPP_DEFAULT_REF
        if not version_dir or not repo_url:
            raise PrimitiveExecutionError(
                "vcs.workspace_checkout requires input.install_dir and input.repository_url"
            )
        config = self._build_config(context)
        self._raise_if_cancelled()
        if bool(context.get("input.replace_existing")) and os.path.isdir(version_dir):
            from backend.utils.fs_ops import robust_rmtree

            robust_rmtree(version_dir)
        if bool(context.get("input.use_workspace", True)):
            workspace = BuildWorkspace.open(ENGINE_ID, repo_url, asdict(config))
            workspace.acquire()
            self._active_workspace = workspace
            await self._update_progress_task(
                cmake_stage_start("clone"), "Updating the audio.cpp build workspace"
            )
            await asyncio.to_thread(workspace.sync_git, repo_url, ref)
            self._raise_if_cancelled()
            return {
                "checkout_work": workspace.checkout_dir,
                "build_work": workspace.build_dir,
            }

        source_dir = os.path.join(version_dir, "source")
        build_dir = os.path.join(version_dir, "build")
        os.makedirs(version_dir, exist_ok=True)
        await self._update_progress_task(cmake_stage_start("clone"), "Cloning audio.cpp")
        await self._run_build_command(
            git_argv("clone", "--recursive", repo_url, source_dir),
            stage="clone",
            cwd=version_dir,
            progress=cmake_stage_start("clone"),
            base_message="Cloning audio.cpp",
        )
        self._raise_if_cancelled()
        await self._update_progress_task(cmake_stage_start("checkout"), f"Checking out {ref}")
        await self._run_build_command(
            ["git", "checkout", ref],
            stage="checkout",
            cwd=source_dir,
            progress=cmake_stage_start("checkout"),
            base_message=f"Checking out {ref}",
        )
        await self._run_build_command(
            git_argv("submodule", "update", "--init", "--recursive"),
            stage="checkout",
            cwd=source_dir,
            progress=14,
            base_message="Checking out audio.cpp",
        )
        return {"checkout_work": source_dir, "build_work": build_dir}

    async def _configure_source(self, step, requirements, context) -> Dict[str, Any]:
        """`cmake.configure`: the manager's configure argv + build environment."""
        from backend.build_progress import cmake_stage_start
        from backend.engines.audio_cpp.manager import get_audio_cpp_manager

        manager = get_audio_cpp_manager()
        config = self._build_config(context)
        checkout = str(context.get("checkout_work") or "")
        build_dir = str(context.get("build_work") or "")
        await self._update_progress_task(
            cmake_stage_start("configure"), f"Configuring {config.backend} build"
        )
        argv, env = await manager._prepare_configure(checkout, build_dir, config)
        try:
            await self._run_build_command(
                argv,
                stage="configure",
                cwd=checkout,
                env=env,
                progress=cmake_stage_start("configure"),
                base_message="Configuring audio.cpp",
            )
        except RuntimeError as exc:
            raise PrimitiveExecutionError(str(exc)) from exc
        return {"build_env": env}

    async def _build_source_tree(self, step, requirements, context) -> Dict[str, Any]:
        """`cmake.build`: the manager's build argv against the configure environment."""
        from backend.build_progress import cmake_stage_start
        from backend.engines.audio_cpp.manager import get_audio_cpp_manager

        manager = get_audio_cpp_manager()
        config = self._build_config(context)
        checkout = str(context.get("checkout_work") or "")
        build_dir = str(context.get("build_work") or "")
        env = context.get("build_env")
        env = dict(env) if isinstance(env, Mapping) else None
        await self._update_progress_task(cmake_stage_start("build"), "Building audio.cpp")
        try:
            await self._run_build_command(
                manager._build_argv(build_dir, config),
                stage="build",
                cwd=checkout,
                env=env,
                progress=cmake_stage_start("build"),
                base_message="Building audio.cpp",
            )
        except RuntimeError as exc:
            raise PrimitiveExecutionError(str(exc)) from exc
        return {}

    async def _seal_source(self, step, requirements, context) -> Dict[str, Any]:
        """`vcs.seal_checkout`: publish the workspace layout into the version dir."""
        checkout = str(context.get("checkout_work") or "")
        build_dir = str(context.get("build_work") or "")
        server = str(context.get("server_built") or "")
        cli = str(context.get("cli_built") or "")
        workspace = self._active_workspace
        if workspace is None:
            return {
                "checkout": checkout,
                "build_dir": build_dir,
                "server": server,
                "cli": cli,
            }
        version_dir = str(context.get("input.install_dir") or "")
        required = [os.path.relpath(path, workspace.path) for path in (server, cli) if path]
        published = await asyncio.to_thread(
            workspace.publish_layout,
            {"source": checkout, "build": build_dir},
            version_dir,
            required,
        )
        return {
            "checkout": os.path.join(published, "source"),
            "build_dir": os.path.join(published, "build"),
            "server": os.path.join(published, os.path.relpath(server, workspace.path)),
            "cli": os.path.join(published, os.path.relpath(cli, workspace.path)),
        }

    async def _validate_binary(self, step, requirements, context) -> Dict[str, Any]:
        """`smoke.test`: the built binary must answer ``--help`` with output."""
        from backend.build_progress import cmake_stage_start

        key = str(step.with_.get("binary") or "server")
        binary = str(context.get(key) or "")
        if not binary:
            raise PrimitiveExecutionError(f"smoke.test requires produced binary {key!r}")
        await self._update_progress_task(
            cmake_stage_start("validate"), "Validating audio.cpp binaries"
        )
        try:
            output = await self._capture_command([binary, "--help"], cwd=os.path.dirname(binary))
        except RuntimeError as exc:
            raise PrimitiveExecutionError(str(exc)) from exc
        if not output:
            raise PrimitiveExecutionError("audio.cpp binaries returned empty help output")
        return {}

    async def _remove_bundle(self, step, requirements, context) -> Dict[str, Any]:
        """`artifact.remove_bundle` for audio's manager-tracked version bundles.

        Moved verbatim from ``AudioCppManager.delete_version_files``: resolve the
        version directory from the row (``source_path`` -> ``install_dir`` ->
        ``<builds>/<version>``), refuse anything outside the builds root, walk up
        to the version directory (never the root itself), then remove it with the
        launch hold released. Refusals are ``ValueError`` so the removal route
        keeps mapping them to HTTP 400.
        """
        from backend.utils.fs_ops import robust_rmtree

        entry = context.get("input.version_entry")
        entry = entry if isinstance(entry, Mapping) else {}
        raw_source = str(entry.get("source_path") or "").strip()
        raw_install = str(entry.get("install_dir") or "").strip()
        version_name = str(entry.get("version") or context.get("input.version_name") or "").strip()
        if raw_source:
            source_path = os.path.abspath(raw_source)
        elif raw_install:
            source_path = os.path.abspath(raw_install)
        elif version_name:
            source_path = os.path.abspath(os.path.join(self.install_root, version_name))
        else:
            return {}
        build_root = os.path.realpath(self.install_root)
        source_real = os.path.realpath(source_path)
        if os.path.commonpath([source_real, build_root]) != build_root:
            raise ValueError("Refusing to delete audio.cpp files outside the builds root")
        version_dir = Path(source_real)
        while version_dir.parent != Path(build_root) and version_dir != Path(build_root):
            version_dir = version_dir.parent
        if version_dir == Path(build_root):
            raise ValueError("Could not resolve audio.cpp version directory")
        delete_target = str(context.get("input.delete_target") or "").strip()
        if delete_target and os.path.realpath(delete_target) == str(version_dir):
            # The recipe's tree step already removed this exact target; the
            # bundle step only owns rows whose recorded layout differs
            # (source_path / manager-tracked dirs).
            return {}
        if os.path.exists(version_dir):
            robust_rmtree(
                str(version_dir),
                retire_references=bool(context.get("input.retire_references")),
            )
        return {}

    async def _materialize_cli_from_companion(self, step, requirements, context) -> Dict[str, Any]:
        """`artifact.materialize_cli` for audio's prebuilt archives."""
        from backend.engines.audio_cpp import prebuilt

        plan = context.get("input.plan")
        binaries = {
            "server_binary_path": str(context.get("server") or ""),
            "cli_binary_path": str(context.get("cli") or ""),
        }
        if not binaries["cli_binary_path"] and plan is not None:
            await self._update_progress_task(
                55, f"Downloading audiocpp_cli from {plan.cli_asset_name}"
            )
        cli_path = await prebuilt.materialize_cli(
            binaries, plan, str(context.get("input.install_dir") or "")
        )
        return {"cli_final": cli_path}

    def _row_extras(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        """Engine-declared row fields: prebuilt provenance or source-build facts."""
        plan = context.get("input.plan")
        if plan is not None:
            source_ref = str(context.get("input.source_ref") or "")
            return {
                "is_fork": False,
                "source_branch": source_ref,
                "cuda_version": plan.host_cuda,
                "build_config": {
                    "prebuilt": True,
                    "release_tag": source_ref,
                    "backend": plan.backend,
                    "asset_name": plan.asset_name,
                    "cli_asset_name": plan.cli_asset_name,
                    "package_cuda": plan.package_cuda,
                    "host_cuda": plan.host_cuda,
                    "selection_reason": plan.reason,
                    "architectures": list(plan.architectures),
                },
            }
        labels = context.get("input.type_labels")
        if not isinstance(labels, Mapping):
            return {}
        # Source builds: labels selected by the route (release tag vs branch/ref),
        # the published checkout, manager-tracked paths and the resolved config.
        checkout = str(context.get("checkout") or "")
        ref = str(context.get("input.source_ref") or "")
        ref_type = str(context.get("input.source_ref_type") or "")
        raw_config = context.get("input.build_config")
        build_config = dict(raw_config) if isinstance(raw_config, Mapping) else {}
        if ref_type == "release":
            build_config = {**build_config, "prebuilt": False, "release_tag": ref}
        extras: Dict[str, Any] = {
            "type": labels.get("type"),
            "is_fork": bool(labels.get("is_fork")),
            "source_branch": ref if ref_type in {"branch", "release"} else None,
            "source_path": checkout or None,
            "source_commit": str(context.get("source_commit") or "") or None,
            "build_config": build_config,
        }
        if checkout:
            from backend.engines.audio_cpp.model_managers import manager_paths_for_source

            extras.update(manager_paths_for_source(checkout))
        return {key: value for key, value in extras.items() if value is not None}
