"""Data-driven venv installer for AUTHORITATIVE (YAML-only) engines.

YAML campaign S3: the reusable "pip wheel persona" installer. NO engine
knowledge lives here — identity, label, package names, update source and
source-repo defaults all come from an ``EngineDefinition``'s ``lifecycle``
section, and the install flows execute the definition's own recipes
(``release`` / ``source`` / ``sync`` / ``remove``) through the SAME shared
primitive executor the cut-over Python engines use. New engines onboard
through YAML with zero installer classes.

Boundary (deliberate): this installer implements the venv+pip surface only —
venv.create / pip.install_package / pip.capture_metadata / pip.uninstall /
vcs.* / fs.remove_tree plus the generic publish handlers. Primitives that
have no generic implementation (cmake builds, host inspection, publish of
native artifacts…) simply make a recipe that uses them fail loudly with
``plan uses unimplemented primitive`` — the executor never silently
no-steps. The nine generated engines keep their dedicated Python installer
classes untouched; this class serves YAML-onboarded engines only.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import sys
from typing import Any, Dict, Optional

from backend.data_store import get_store
from backend.engines.activation import activate_engine_version
from backend.engines.definitions import (
    load_default_engine_definitions,
    make_engine_runner,
    materialize_plan,
)
from backend.engines.definitions.executor import (
    PrimitiveExecutionError,
    execute_plan,
)
from backend.engines.definitions.schema import EngineDefinition
from backend.engines.python.installer import PythonVenvInstaller, unique_version_name, utcnow
from backend.logging_config import get_logger
from backend.proxy.control import mark_runtime_changed


logger = get_logger(__name__)


def definition_for(engine_id: str) -> Optional[EngineDefinition]:
    return load_default_engine_definitions().definitions.get(engine_id)


# One installer instance per definition-owned engine: installs hold task state
# and cancellation needs the same instance the route started. Reset in tests
# that swap the definition directory.
_DEFINITION_INSTALLERS: Dict[str, "GenericVenvInstaller"] = {}


def get_definition_installer(engine_id: str) -> "GenericVenvInstaller":
    """Singleton installer for a definition-owned engine."""
    engine_id = str(engine_id or "")
    installer = _DEFINITION_INSTALLERS.get(engine_id)
    if installer is None:
        definition = definition_for(engine_id)
        if definition is None:
            raise RuntimeError(f"engine definition is unavailable for {engine_id!r}")
        installer = GenericVenvInstaller(definition)
        _DEFINITION_INSTALLERS[engine_id] = installer
    return installer


def reset_definition_installers() -> None:
    _DEFINITION_INSTALLERS.clear()


def _source_extra(engine_id: str, repo_url: str, branch: str) -> Dict[str, Any]:
    """Pending-version metadata for a source install (shared by handlers)."""
    from backend.repo_identity import source_build_type_labels_for_engine

    labels = source_build_type_labels_for_engine(engine_id, repo_url)
    return {
        "type": labels["type"],
        "install_type": labels["install_type"],
        "is_fork": labels["is_fork"],
        "source_repo": repo_url,
        "source_branch": branch,
    }


class GenericVenvInstaller(PythonVenvInstaller):
    """One venv+pip installer shaped entirely by an engine definition."""

    def __init__(
        self,
        definition: EngineDefinition,
        *,
        log_path: Optional[str] = None,
        base_dir: Optional[str] = None,
        state_path: Optional[str] = None,
    ) -> None:
        lifecycle = dict(definition.lifecycle)
        self._definition = definition
        self.distribution_names = tuple(lifecycle.get("distribution_names") or (definition.id,))
        self._binary_name = str(lifecycle.get("binary_name") or definition.id)
        self._default_repo_url = str(lifecycle.get("default_repo_url") or "")
        self._default_branch = str(lifecycle.get("default_branch") or "main")
        state_name = lifecycle.get("state_name")
        super().__init__(
            engine_id=definition.id,
            label=definition.label,
            root_name=str(lifecycle.get("root_name") or definition.id),
            log_name=str(lifecycle.get("log_name") or f"{definition.id}_install.log"),
            log_path=log_path,
            base_dir=base_dir,
            state_path=state_path,
            state_name=None if state_path else (str(state_name) if state_name else None),
        )
        # Routes/UI parse the legacy manager token for some engines.
        if lifecycle.get("manager_name"):
            self.MANAGER_NAME = str(lifecycle["manager_name"])

    @property
    def definition(self) -> EngineDefinition:
        return self._definition

    def _binary_override(self) -> Optional[str]:
        """Resolve the definition-declared binary override, if any.

        ``lifecycle.binary_override_env`` names an environment variable that
        may carry the engine binary (a path or a PATH-resolvable name). The
        mechanism lives here once; a definition only selects the variable name.
        An override that names nothing resolvable falls through to the venv
        binary, preserving the historic behaviour of the engines that used it.
        """
        env_name = str(self._lifecycle().get("binary_override_env") or "").strip()
        if not env_name:
            return None
        override = os.getenv(env_name)
        if not override:
            return None
        override_path = os.path.abspath(os.path.expanduser(override))
        if os.path.exists(override_path):
            return override_path
        return shutil.which(override)

    def _resolve_binary_path(self) -> Optional[str]:
        override = self._binary_override()
        if override:
            return override
        candidate = self._venv_bin(self._binary_name)
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)
        return shutil.which(self._binary_name)

    def _recipe(self, name: str):
        plan = materialize_plan(self._definition, name)
        if plan is None:
            raise RuntimeError(f"{self.label} definition has no '{name}' recipe")
        return plan

    def _record_snapshot(self, version_name: str, resolved: Dict[str, Any], *, phase: str) -> None:
        """Record a version-associated snapshot (pending or published).

        A pending row is written before execution so a FAILED install is still
        associated with the definition that attempted it; the published row is
        written once the installation exists. Both select by version, never by
        "latest for the engine".
        """
        try:
            digest = str(getattr(self.definition, "definition_digest", "") or "")
            if not digest:
                return
            get_store().record_install_snapshot(
                self.engine_id,
                definition_digest=digest,
                recipe_version=f"schema-v{self.definition.schema_version}",
                resolved_inputs=resolved,
                version=version_name,
                phase=phase,
            )
        except Exception:  # noqa: BLE001 — bookkeeping must not fail installs
            logger.debug("install snapshot %s failed for %s", phase, self.engine_id, exc_info=True)

    def _record_published_snapshot(self, version_name: str, resolved: Dict[str, Any]) -> None:
        self._record_snapshot(version_name, resolved, phase="published")

    def _record_pending_snapshot(self, version_name: str, resolved: Dict[str, Any]) -> None:
        self._record_snapshot(version_name, resolved, phase="scheduled")

    def _merge_pending_snapshot(self, resolved: Dict[str, Any]) -> None:
        """Attach a late-resolved tag to the active install's pending row."""
        version_name = str(getattr(self, "_pending_resolution_version", "") or "")
        if not version_name:
            return
        try:
            store = get_store()
            update = getattr(store, "update_install_snapshot", None)
            if callable(update):
                update(self.engine_id, version=version_name, resolved_inputs=resolved)
        except Exception:  # noqa: BLE001 — bookkeeping must not fail installs
            logger.debug(
                "pending install snapshot merge failed for %s", self.engine_id, exc_info=True
            )

    def _install_settings(self) -> Dict[str, Any]:
        try:
            return dict(get_store().get_engine_build_settings(self.engine_id) or {})
        except Exception:  # noqa: BLE001 — optional compatibility data
            return {}

    @staticmethod
    def _setting_enabled(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

    async def _apply_compat_install(self) -> None:
        """Install a definition-declared compatibility wheel set.

        LMDeploy's CUDA-13 default wheel has no sm70 kernels. The selected
        V100/sm70 profile declares a CUDA-12 torch spec; install it into the
        same venv after the normal release/source install. No engine-id branch
        lives here: the packages and index come from ``settings_defaults``.
        """
        settings = self._install_settings()
        if not self._setting_enabled(settings.get("compat_sm70")):
            return
        packages = shlex.split(str(settings.get("compat_torch_spec") or ""))
        if not packages:
            return
        argv = ["install", "--upgrade", *packages]
        index_url = str(settings.get("compat_torch_index_url") or "").strip()
        if index_url:
            argv.extend(["--index-url", index_url])
        code = await self._run_pip(argv, "compatibility")
        if code != 0:
            raise RuntimeError(
                f"{self.label} compatibility PyTorch install failed with status {code}"
            )

    def _install_snapshot_for(self, version_name: str) -> Dict[str, Any]:
        """Snapshot recorded for THIS installation, or {} (shared selection)."""
        from backend.engines.install_snapshots import install_snapshot_for

        return install_snapshot_for(get_store(), self.engine_id, version_name)

    # --- Definition-declared installer policy (Phase-3 typed contracts) --------------

    def _lifecycle(self) -> Dict[str, Any]:
        return dict(getattr(self.definition, "lifecycle", {}) or {})

    def _python_requirement(self) -> tuple[str, str]:
        lifecycle = self._lifecycle()
        return str(lifecycle.get("python_min") or ""), str(lifecycle.get("python_max") or "")

    def _ensure_venv(self) -> None:
        """Enforce a definition-declared interpreter requirement before venv.create."""
        lifecycle = self._lifecycle()
        exact = str(lifecycle.get("python_exact") or "")
        if exact:
            expected = exact.split(".")
            running = sys.version_info[:2]
            wanted = (int(expected[0]), int(expected[1]) if len(expected) > 1 else 0)
            if running != wanted:
                raise RuntimeError(
                    f"{self.label} requires Python {exact} "
                    f"(Studio is running {running[0]}.{running[1]})"
                )
            super()._ensure_venv()
            return

        minimum, maximum = self._python_requirement()

        def _version(text: str) -> tuple[int, int]:
            parts = text.split(".")
            return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)

        running = sys.version_info[:2]
        too_old = bool(minimum) and running < _version(minimum)
        too_new = bool(maximum) and running > _version(maximum)
        if too_old or too_new:
            span = f"{minimum}–{maximum}" if minimum and maximum else (minimum or maximum)
            raise RuntimeError(
                f"{self.label} requires Python {span}; "
                f"Studio is running Python {sys.version_info.major}.{sys.version_info.minor}"
            )
        super()._ensure_venv()

    def _runtime_metadata(self) -> Dict[str, Any]:
        """Install-record fields for the declared build environment."""
        if self._lifecycle().get("record_runtime_metadata") is False:
            return {}
        metadata: Dict[str, Any] = {}
        if str(self._lifecycle().get("build_env") or "") == "cuda":
            from backend.cuda_installer import get_cuda_installer

            installer = get_cuda_installer()
            cuda = installer.status() if hasattr(installer, "status") else {}
            metadata = {
                "cuda_version": cuda.get("version"),
                "cuda_path": cuda.get("cuda_path"),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            }
        metadata.update(dict(self._lifecycle().get("runtime_metadata") or {}))
        return metadata

    def _compose_build_env(self) -> Dict[str, str]:
        lifecycle = self._lifecycle()
        env = dict(os.environ)
        if str(lifecycle.get("build_env") or "") == "cuda":
            from backend.cuda_installer import get_cuda_installer

            env.update(get_cuda_installer().get_cuda_env())
            fallback = str(lifecycle.get("cuda_home_fallback") or "")
            if fallback and not env.get("CUDA_HOME"):
                env["CUDA_HOME"] = fallback
                self._prepend_path(env, os.path.join(fallback, "bin"))
                env["LD_LIBRARY_PATH"] = (
                    f"{os.path.join(fallback, 'lib64')}:{env.get('LD_LIBRARY_PATH', '')}"
                )
        env["VIRTUAL_ENV"] = self._venv_path
        declared = dict(lifecycle.get("build_env_vars") or {})
        # Host-derived nvcc parallelism for every engine whose build reads
        # NVCC_THREADS (SGLang's kernel builds, torch extensions, 1Cat): the
        # service environment must not decide it, and a definition that wants a
        # fixed value still pins it through build_env_vars.
        if "NVCC_THREADS" not in declared:
            from backend.engines.python.fork_build import host_nvcc_threads

            env["NVCC_THREADS"] = str(
                host_nvcc_threads(
                    declared.get("TORCH_CUDA_ARCH_LIST") or env.get("TORCH_CUDA_ARCH_LIST")
                )
            )
        for name, value in declared.items():
            env.setdefault(str(name), str(value))
        if lifecycle.get("build_rust"):
            from backend.engines.python.rust_toolchain import compose_rust_env

            compose_rust_env(self, env, self._managed_tools_dir("rust"))
        # The venv must stay ahead of cargo/rustup shims in PATH.
        self._prepend_path(env, os.path.dirname(self._venv_python()))
        return env

    def _managed_tools_dir(self, name: str) -> str:
        """Persistent tools prefix beside the engine install root."""
        return os.path.join(os.path.dirname(self._root_dir), "tools", name)

    async def _compose_env(self, step, requirements, context):
        """`env.compose` primitive: build env + runtime metadata for the recipe.

        Produces the names recipes declare (`build_env`, `runtime_meta`) and
        remembers the metadata so failure bookkeeping can include it, exactly
        like the legacy per-engine flows did.
        """
        build_env = self._compose_build_env()
        runtime = self._runtime_metadata()
        self._composed_runtime = dict(runtime)
        return {"build_env": build_env, "runtime_meta": runtime}

    def _finalize_handler(
        self,
        *,
        existing_version: Optional[str],
        pending_version: str,
        extra: Optional[Dict[str, Any]] = None,
        branch: str = "",
        include_commit: bool = False,
    ):
        """`publish.finalize` primitive: validated, reuse-aware publication.

        Verifies the installed package actually imports, composes the version
        name (commit-tagged for source builds), records the version row and
        activates it. Shared by any definition whose recipe ends in
        ``publish.finalize``.
        """

        async def finalize(step, requirements, context):
            from backend.engines.definitions.executor import PrimitiveExecutionError

            commit = str(requirements.get("source_commit") or "") if include_commit else ""
            await self._update_progress_task(98, f"Validating {self.label} installation")
            detected = self._detect_installed_version()
            if not detected:
                raise PrimitiveExecutionError(
                    f"Installed {self.label} package could not be imported"
                )
            store = get_store()
            if include_commit:
                base = f"{detected or branch}-{commit[:8]}".rstrip("-")
            else:
                base = detected or pending_version
            meta: Dict[str, Any] = {
                **dict(extra or {}),
                **self._release_meta(requirements),
                "version": existing_version or base or pending_version,
                "reuse_existing": bool(existing_version),
                **self._runtime_metadata(),
            }
            if include_commit:
                meta["source_commit"] = commit
            final_base = str(meta.get("version") or detected or pending_version)
            final_version = (
                pending_version
                if meta.get("reuse_existing")
                else unique_version_name(store, self.engine_id, final_base)
            )
            meta = {
                **meta,
                "version": final_version,
                "package_version": detected,
                "installed_at": utcnow(),
                "venv_path": self._venv_path,
                "install_dir": self._base_dir,
            }
            meta.pop("reuse_existing", None)
            final_version = self._ready_pending_version(pending_version, meta)
            self._record_published_snapshot(
                final_version,
                {
                    key: meta.get(key)
                    for key in (
                        "install_type",
                        "package_version",
                        "release_tag",
                        "venv_path",
                    )
                    if meta.get(key) is not None
                },
            )
            await activate_engine_version(
                self.engine_id,
                final_version,
                covered_by=self.progress_task_id,
                row=meta,
            )
            return {}

        return finalize

    def _release_meta(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Engine-specific release metadata merged into the published row.

        GitHub-wheel engines keep the selected release tag here so retries and
        provenance do not have to re-derive it from the package version.
        """
        install_type = str(self._lifecycle().get("release_install_type") or "")
        if not install_type:
            return {}
        tag = str(requirements.get("release_tag") or "").strip()
        return {"install_type": install_type, "release_tag": tag} if tag else {}

    def _source_dist_dir(self) -> str:
        """Wheel-family recipes build into this directory ('' = not used)."""
        subdir = str(self._lifecycle().get("source_dist_dir") or "")
        return os.path.join(self._base_dir, subdir) if subdir else ""

    def _extra_plan_handlers(self) -> Dict[str, Any]:
        """Handlers for the reusable fork/wheel primitives every definition may use.

        Filesystem/pip primitives are shared already; only genuinely new
        mechanisms are registered here.
        """
        return {
            "vcs.apply_transform": self._apply_transform_handler,
            "build.prepare_fork": self._prepare_fork_handler,
            "artifact.build_wheels": self._handler_build_wheels,
            "toolchain.ensure": self._handler_toolchain,
            "prereqs.resolve_release_assets": self._handler_release_assets,
        }

    async def _handler_build_wheels(self, step, requirements, context) -> Dict[str, Any]:
        options = dict(getattr(step, "with_", {}) or {})
        checkout = str(requirements.get("checkout") or "")
        build_env = dict(requirements.get("build_env") or {})
        dist_dir = str(requirements.get("input.dist_dir") or "")
        python_exe = self._venv_python()
        empty_error = str(options.get("empty_error") or "Source build produced no wheels")

        async def build(cwd: str, *, window, error: str) -> int:
            self._active_build_window = tuple(window)
            code = await self._run_logged(
                [
                    python_exe,
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    dist_dir,
                ],
                "build_wheels",
                cwd=cwd,
                env=build_env,
            )
            if code != 0:
                raise RuntimeError(error.format(code=code))
            return code

        for sub in options.get("sub_builds") or ():
            subdir = os.path.join(checkout, str(sub.get("path") or ""))
            if not os.path.isdir(subdir):
                continue
            await build(
                subdir,
                window=sub.get("progress_window") or (10, 30),
                error=str(sub.get("error") or "wheel build failed ({code})"),
            )
        try:
            await build(
                checkout,
                window=options.get("main_progress_window") or (36, 92),
                error=str(options.get("main_error") or "wheel build failed ({code})"),
            )
        except RuntimeError as exc:
            if options.get("rust_hint"):
                from backend.engines.python.rust_toolchain import discover_rust_bin_dir

                if not discover_rust_bin_dir(build_env, self._managed_tools_dir("rust")):
                    raise RuntimeError(
                        f"{exc}; Rust toolchain (cargo/rustc) was not found"
                    ) from exc
            raise
        wheels = [
            os.path.join(dist_dir, name)
            for name in sorted(os.listdir(dist_dir))
            if name.endswith(".whl")
        ]
        if not wheels:
            raise RuntimeError(empty_error)
        return {"wheels": wheels}

    async def _handler_toolchain(self, step, requirements, context) -> Dict[str, Any]:
        options = dict(getattr(step, "with_", {}) or {})
        kind = str(options.get("kind") or "rust")
        if kind != "rust":
            raise RuntimeError(f"unsupported toolchain kind {kind!r}")
        from backend.engines.python.rust_toolchain import ensure_rust

        env = dict(requirements.get("build_env") or {})
        self._active_build_window = tuple(options.get("progress_window") or (4, 16))
        await ensure_rust(
            self,
            env,
            self._managed_tools_dir("rust"),
            installer_url=str(options.get("installer_url") or "https://sh.rustup.rs"),
        )
        output_key = str(options.get("output_key") or "build_env")
        return {output_key: env}

    async def _handler_release_assets(self, step, requirements, context) -> Dict[str, Any]:
        options = dict(getattr(step, "with_", {}) or {})
        repository = str(options.get("repository") or "")
        if not repository:
            raise RuntimeError("prereqs.resolve_release_assets requires with.repository")
        from backend.engines.python.release_assets import (
            fetch_github_release,
            select_wheel_urls,
        )

        version = requirements.get("input.requested_version")
        release = await fetch_github_release(repository, version)
        tag, urls = select_wheel_urls(
            release,
            [dict(group) for group in (options.get("wheel_groups") or ())],
            error=str(
                options.get("error") or "release does not contain an installable wheel asset"
            ),
        )
        # The pending row was written before this resolution. Persist the
        # concrete tag now so a failed install retries THIS release, not a
        # fresh "latest" lookup.
        self._merge_pending_snapshot(
            {
                "install_type": "pip",
                "package_version": str(tag or ""),
                "release_tag": str(tag or ""),
            }
        )
        return {"release_tag": tag, "wheel_urls": urls}

    async def _apply_transform_handler(self, step, requirements, context) -> Dict[str, Any]:
        """`vcs.apply_transform`: versioned, applicability-checked file rewrite."""
        from backend.engines.definitions.transforms import apply_transform

        name = str(dict(getattr(step, "with_", {}) or {}).get("transform") or "")
        if not name:
            raise RuntimeError("vcs.apply_transform requires with.transform")
        checkout = str(requirements.get("checkout") or "")
        if not checkout:
            raise RuntimeError("vcs.apply_transform requires a checkout")
        path = apply_transform(name, checkout)
        return {str(output): path for output in (getattr(step, "produces", ()) or ())}

    async def _prepare_fork_handler(self, step, requirements, context) -> Dict[str, Any]:
        """`build.prepare_fork`: declared builder environment for fork builds."""
        from backend.engines.python.fork_build import prepare_fork_build

        checkout = str(requirements.get("checkout") or "")
        if not checkout:
            raise RuntimeError("build.prepare_fork requires a checkout")
        env = await prepare_fork_build(
            self,
            dict(getattr(step, "with_", {}) or {}),
            checkout=checkout,
            base_env=dict(requirements.get("rust_env") or requirements.get("build_env") or {}),
        )
        return {"install_env": env}

    @staticmethod
    def _plan_primitives(plan) -> set:
        return {str(getattr(step, "primitive", "")) for step in getattr(plan, "steps", plan)}

    def _plan_handlers(self, plan, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Override only the primitives THIS recipe uses.

        Shadowing unrelated primitives would make an engine's handler surface
        wider than its definition and hide drift; filtering keeps it equal to
        what the recipe asks for.
        """
        used = self._plan_primitives(plan)
        return {name: handler for name, handler in overrides.items() if name in used}

    def _require_repo_url(self, repo_url: Optional[str]) -> str:
        repo = str(repo_url or "").strip() or self._default_repo_url
        if not repo:
            raise RuntimeError(
                f"{self.label} has no source repository; set default_repo_url "
                "in the engine definition or pass repo_url"
            )
        return repo

    # --- Public interface (EngineInstaller protocol) ---------------------------------

    async def install_release(
        self,
        version: Optional[str] = None,
        force_reinstall: bool = False,
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        if bool(self._lifecycle().get("release_from_source")):
            # The engine has no pip release; a release install is its source
            # install (repo/branch are the declared lifecycle defaults).
            return await self.install_from_source(
                str(self._lifecycle().get("default_repo_url") or "").strip(),
                str(self._lifecycle().get("default_branch") or "main"),
                reuse_dir=reuse_dir,
                existing_version=existing_version,
            )
        async with self._lock:
            if self._operation:
                raise RuntimeError(f"Another {self.label} operation is already running")
            await self._start_operation("install")
            dir_name = self._bind_install_dir(reuse_dir, "pip")
            pending_version = existing_version or dir_name
            self._register_pending_version(
                pending_version,
                {"install_type": "pip", "package_version": version},
            )
            # Associate the pending row with the definition NOW: a failed
            # install must still refuse a retry after the definition changed.
            self._record_pending_snapshot(
                pending_version,
                {
                    "install_type": "pip",
                    "requested_version": str(version or ""),
                    "package_version": str(version or ""),
                    "venv_path": self._venv_path,
                    "install_dir": self._base_dir,
                },
            )
            self._pending_resolution_version = pending_version

            async def _runner():
                try:
                    execution = await self._execute_release_plan(
                        requested_version=version,
                        pending_version=pending_version,
                        existing_version=existing_version,
                        force_reinstall=force_reinstall,
                    )
                    if execution.status != "completed":
                        failure = execution.failed_step
                        raise RuntimeError(
                            failure.error
                            if failure is not None and failure.error
                            else f"{self.label} installation did not complete"
                        )
                    await self._apply_compat_install()
                    await self._finish_operation(True, f"{self.label} installed")
                except Exception as exc:
                    self._last_error = str(exc)
                    self._fail_pending_version(pending_version, str(exc), {"install_type": "pip"})
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))
                finally:
                    self._pending_resolution_version = ""

            self._create_task(_runner())
            return self._started_response(f"{self.label} installation started")

    async def _execute_release_plan(
        self,
        *,
        requested_version: Optional[str],
        pending_version: str,
        existing_version: Optional[str],
        force_reinstall: bool,
    ):
        plan = self._recipe("release")

        async def publish_release(step, requirements, context):
            from fastapi import HTTPException

            detected_version = (requirements.get("metadata") or {}).get("version")
            try:
                store = get_store()
                if existing_version:
                    version_name = pending_version
                else:
                    base = detected_version or pending_version
                    version_name = (
                        pending_version
                        if base == pending_version
                        else unique_version_name(store, self.engine_id, base)
                    )
                meta: Dict[str, Any] = {
                    "version": version_name,
                    "install_type": "pip",
                    "package_version": detected_version,
                    "venv_path": self._venv_path,
                    "install_dir": self._base_dir,
                    "installed_at": utcnow(),
                }
                self._ready_pending_version(pending_version, meta)
                self._record_published_snapshot(
                    version_name,
                    {
                        "install_type": "pip",
                        "version": version_name,
                        "requested_version": requested_version,
                        "package_version": detected_version,
                        "venv_path": self._venv_path,
                        "install_dir": self._base_dir,
                    },
                )
                await activate_engine_version(
                    self.engine_id,
                    version_name,
                    covered_by=self.progress_task_id,
                    row=meta,
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.debug(f"Failed to persist {self.label} engine metadata: {exc}")
            return {}

        return await execute_plan(
            plan,
            run_primitive=make_engine_runner(
                self,
                self._plan_handlers(
                    plan,
                    {
                        **self._extra_plan_handlers(),
                        "publish.install": publish_release,
                        "publish.finalize": self._finalize_handler(
                            existing_version=existing_version,
                            pending_version=pending_version,
                            extra={
                                "install_type": "pip",
                                "package_version": requested_version,
                            },
                        ),
                    },
                ),
            ),
            context={
                "input.engine_root": self._base_dir,
                "input.requested_version": str(requested_version or ""),
                "input.force_reinstall": bool(force_reinstall),
            },
        )

    def _source_started_message(self) -> str:
        """Start-response wording, optionally declared by the definition."""
        template = str(self._lifecycle().get("source_started_message") or "")
        if template:
            return template.replace("{label}", self.label)
        return f"{self.label} install from source started"

    async def install_from_source(
        self,
        repo_url: Optional[str] = None,
        branch: str = "main",
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        repo_url = self._require_repo_url(repo_url)
        async with self._lock:
            if self._operation:
                raise RuntimeError(f"Another {self.label} operation is already running")
            await self._start_operation("install_source")
            dir_name = self._bind_install_dir(reuse_dir, "source")
            pending_version = existing_version or dir_name
            from backend.repo_identity import source_build_type_labels_for_engine

            type_labels = source_build_type_labels_for_engine(self.engine_id, repo_url)
            self._register_pending_version(
                pending_version,
                {
                    "type": type_labels["type"],
                    "install_type": type_labels["install_type"],
                    "is_fork": type_labels["is_fork"],
                    "source_repo": repo_url,
                    "source_branch": branch,
                },
            )
            self._record_pending_snapshot(
                pending_version,
                {
                    "install_type": type_labels["install_type"],
                    "requested_version": str(branch or ""),
                    "source_repo": repo_url,
                    "source_branch": branch,
                    "install_dir": self._base_dir,
                },
            )
            clone_dir = os.path.join(self._base_dir, "source")
            reuse = bool(existing_version or reuse_dir)
            workspace_holder: Dict[str, Any] = {}

            async def _runner():
                try:
                    execution = await self._execute_source_plan(
                        repo_url=repo_url,
                        branch=branch,
                        reuse=reuse,
                        clone_dir=clone_dir,
                        pending_version=pending_version,
                        existing_version=existing_version,
                        workspace_holder=workspace_holder,
                    )
                    if execution.status != "completed":
                        failure = execution.failed_step
                        raise RuntimeError(
                            failure.error
                            if failure is not None and failure.error
                            else f"{self.label} source install did not complete"
                        )
                    await self._apply_compat_install()
                    await self._finish_operation(True, f"Installed from {branch}")
                except Exception as exc:
                    self._last_error = str(exc)
                    self._fail_pending_version(
                        pending_version,
                        str(exc),
                        {
                            "source_repo": repo_url,
                            "source_branch": branch,
                            "install_type": "source",
                            **getattr(self, "_composed_runtime", {}),
                        },
                    )
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))
                finally:
                    from backend.engines.build_workspace import release_held

                    release_held(workspace_holder.get("workspace"))

            self._create_task(_runner())
            return self._started_response(
                self._source_started_message(),
                repo=repo_url,
                branch=branch,
            )

    async def _execute_source_plan(
        self,
        *,
        repo_url: str,
        branch: str,
        reuse: bool,
        clone_dir: str,
        pending_version: str,
        existing_version: Optional[str],
        workspace_holder: Dict[str, Any],
    ):
        plan = self._recipe("source")

        async def workspace_checkout(step, requirements, context):
            from backend.engines.build_workspace import BuildWorkspace

            # Legacy source flows bracketed the checkout with these two
            # progress stages; the shared host keeps them identical so the
            # install panel does not regress on migration.
            await self._update_progress_task(3, "Cloning source")
            if requirements.get("input.reuse_mode"):
                if os.path.exists(clone_dir):
                    shutil.rmtree(clone_dir)
                os.makedirs(clone_dir, exist_ok=True)
                proc = await asyncio.create_subprocess_exec(
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    branch,
                    repo_url,
                    clone_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                await proc.wait()
                if proc.returncode != 0:
                    raise PrimitiveExecutionError(
                        f"git clone failed with code {proc.returncode}",
                        exit_code=proc.returncode,
                    )
                await self._update_progress_task(8, "Source checkout ready")
                return {"checkout": clone_dir}
            workspace = BuildWorkspace.open(self.engine_id, repo_url, {"kind": "source"})
            workspace_holder["workspace"] = workspace
            await asyncio.to_thread(workspace.acquire)
            await asyncio.to_thread(workspace.sync_git, repo_url, branch)
            await self._update_progress_task(8, "Source checkout ready")
            # Publish the workspace too: sealing may then use the executor's
            # default `vcs.seal_checkout` (which resolves it from the plan's
            # workspace_from), and the explicit handler below stays identical
            # to that default.
            return {"checkout": workspace.checkout_dir, "workspace": workspace}

        async def seal_checkout(step, requirements, context):
            """Same semantics as the executor default (see definitions/primitives).

            Destination honours the step's `destination`, the checkout is
            rebound to the sealed path, and `checkout_sealed` is kept for
            recipes that label it that way.
            """
            workspace = context.get(
                str(step.with_.get("workspace_from") or "workspace")
            ) or workspace_holder.get("workspace")
            if workspace is None:
                return {}
            destination = os.path.join(
                self._base_dir, str(step.with_.get("destination") or "source")
            )
            # Sealing copies the checkout and rewrites the virtualenv; on a
            # multi-GB venv it takes minutes and used to look frozen at the
            # builder's last line. Say what it is doing.
            await self._update_progress_task(98, "Sealing the installed source")
            sealed = await asyncio.to_thread(
                workspace.seal_installed_source, destination, self._venv_path
            )
            await self._update_progress_task(98, "Sealed; publishing the version")
            return {"checkout": sealed, "checkout_sealed": sealed}

        async def publish_source(step, requirements, context):
            from fastapi import HTTPException
            from backend.repo_identity import source_build_type_labels_for_engine

            detected_version = (requirements.get("metadata") or {}).get("version")
            try:
                store = get_store()
                if existing_version:
                    version_name = pending_version
                else:
                    base_version = detected_version or branch or pending_version
                    version_name = unique_version_name(
                        store, self.engine_id, f"{base_version}-{utcnow()}"
                    )
                type_labels = source_build_type_labels_for_engine(self.engine_id, repo_url)
                meta: Dict[str, Any] = {
                    "version": version_name,
                    "type": type_labels["type"],
                    "install_type": type_labels["install_type"],
                    "is_fork": type_labels["is_fork"],
                    "source_repo": repo_url,
                    "source_branch": branch,
                    "package_version": detected_version,
                    "venv_path": self._venv_path,
                    "install_dir": self._base_dir,
                    "installed_at": utcnow(),
                }
                self._ready_pending_version(pending_version, meta)
                self._record_published_snapshot(
                    version_name,
                    {
                        "install_type": "source",
                        "version": version_name,
                        "requested_version": str(branch or ""),
                        "package_version": detected_version,
                        "source_repo": repo_url,
                        "source_branch": branch,
                        "venv_path": self._venv_path,
                        "install_dir": self._base_dir,
                    },
                )
                await activate_engine_version(
                    self.engine_id,
                    version_name,
                    covered_by=self.progress_task_id,
                    row=meta,
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.debug(f"Failed to persist {self.label} engine metadata (source): {exc}")
            return {}

        return await execute_plan(
            plan,
            run_primitive=make_engine_runner(
                self,
                self._plan_handlers(
                    plan,
                    {
                        **self._extra_plan_handlers(),
                        "vcs.workspace_checkout": workspace_checkout,
                        "vcs.seal_checkout": seal_checkout,
                        "publish.install": publish_source,
                        "publish.finalize": self._finalize_handler(
                            existing_version=existing_version,
                            pending_version=pending_version,
                            extra=_source_extra(self.engine_id, repo_url, branch),
                            branch=branch,
                            include_commit=True,
                        ),
                        "env.compose": self._compose_env,
                    },
                ),
            ),
            context={
                "input.engine_root": self._base_dir,
                "input.repo_url": repo_url,
                "input.branch": branch,
                "input.reuse_mode": reuse,
                "input.dist_dir": self._source_dist_dir(),
            },
        )

    async def retry_existing_install(self, version_entry: Dict[str, Any]) -> Dict[str, Any]:
        version_entry = version_entry or {}
        version_name = str(version_entry.get("version") or "").strip()
        install_dir = str(version_entry.get("install_dir") or "").strip()
        venv_path = str(version_entry.get("venv_path") or "").strip()
        if not install_dir and venv_path:
            install_dir = os.path.dirname(os.path.abspath(venv_path))
        if not version_name or not install_dir:
            raise ValueError(f"This {self.label} version does not have enough metadata to retry")
        # Snapshot-backed retry: the install-snapshot ledger froze the
        # definition digest and the resolved inputs for THIS installation.
        # A changed definition means retrying would reinterpret the recorded
        # layout — refuse and ask for a reinstall.
        from backend.engines.install_snapshots import resolve_snapshot_inputs

        resolved_inputs = resolve_snapshot_inputs(
            self._install_snapshot_for(version_name),
            version_name=version_name,
            current_digest=str(getattr(self.definition, "definition_digest", "") or ""),
            label=self.label,
        )
        kind = (
            str(version_entry.get("install_type") or version_entry.get("type") or "")
            .strip()
            .lower()
        )
        if kind in {"source", "fork", "patched", "local"}:
            repo = str(version_entry.get("source_repo") or "").strip() or self._default_repo_url
            branch = str(
                version_entry.get("source_branch")
                or version_entry.get("source_ref")
                or resolved_inputs.get("branch")
                or resolved_inputs.get("source_branch")
                or "main"
            ).strip()
            return await self.install_from_source(
                repo, branch, reuse_dir=install_dir, existing_version=version_name
            )
        # Prefer the snapshot's frozen resolution over live re-resolution.
        # ``resolved_inputs["version"]`` historically meant the REQUESTED
        # package version; pending snapshots use the installation row name in
        # that key, which must never be passed to pip as a package version.
        snapshot_version = str(resolved_inputs.get("version") or "")
        if snapshot_version == version_name:
            snapshot_version = ""
        pip_version = (
            resolved_inputs.get("release_tag")
            or version_entry.get("release_tag")
            or resolved_inputs.get("package_version")
            or resolved_inputs.get("requested_version")
            or snapshot_version
            or version_entry.get("package_version")
        )
        return await self.install_release(
            version=str(pip_version) if pip_version else None,
            force_reinstall=True,
            reuse_dir=install_dir,
            existing_version=version_name,
        )

    async def sync_source_version(self, version_entry: Dict[str, Any]) -> Dict[str, Any]:
        version_entry = version_entry or {}
        branch = str(version_entry.get("source_branch") or "").strip()
        version_name = str(version_entry.get("version") or "").strip()
        venv_path = str(version_entry.get("venv_path") or "").strip()
        install_dir = str(version_entry.get("install_dir") or "").strip()
        kind = (
            str(version_entry.get("install_type") or version_entry.get("type") or "")
            .strip()
            .lower()
        )
        if kind not in {"source", "fork"}:
            raise RuntimeError(f"Only {self.label} source installs can be synced")
        if not branch:
            raise RuntimeError(f"{self.label} source install is missing source_branch")
        # Legacy engines rebound the install root from the version row; rows
        # without an explicit venv_path still sync from their install_dir.
        if not venv_path and install_dir:
            venv_path = os.path.join(install_dir, "venv")
        if not version_name or not venv_path:
            raise RuntimeError(f"{self.label} source install metadata is incomplete")

        async with self._lock:
            if self._operation:
                raise RuntimeError(f"Another {self.label} operation is already running")

            self._venv_path = os.path.abspath(venv_path)
            self._base_dir = os.path.dirname(self._venv_path)
            self._ensure_directories()
            clone_dir = os.path.join(self._base_dir, "source")

            await self._start_operation(
                "sync_source",
                {"version": version_name, "branch": branch, "sync": True},
            )

            async def _runner():
                try:
                    execution = await self._execute_sync_plan(
                        branch=branch,
                        clone_dir=clone_dir,
                        version_name=version_name,
                        source_repo=version_entry.get("source_repo"),
                        row=version_entry,
                    )
                    if execution.status != "completed":
                        failure = execution.failed_step
                        raise RuntimeError(
                            failure.error
                            if failure is not None and failure.error
                            else f"{self.label} source sync did not complete"
                        )
                    await self._finish_operation(True, f"Synced from {branch}")
                except Exception as exc:
                    self._last_error = str(exc)
                    self._fail_pending_version(
                        version_name,
                        str(exc),
                        {
                            **dict(version_entry),
                            **getattr(self, "_composed_runtime", {}),
                        },
                    )
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))

            self._create_task(_runner())
            return self._started_response(
                f"{self.label} source sync started",
                version=version_name,
                branch=branch,
            )

    async def _execute_sync_plan(
        self,
        *,
        branch: str,
        clone_dir: str,
        version_name: str,
        source_repo: Optional[str],
        row: Optional[Dict[str, Any]] = None,
    ):
        plan = self._recipe("sync")

        async def sync_checkout(step, requirements, context):
            await self._sync_git_checkout(clone_dir, branch)
            return {"checkout": clone_dir}

        async def record_update(step, requirements, context):
            try:
                store = get_store()
                updated = store.update_engine_version(
                    self.engine_id,
                    version_name,
                    {
                        "source_commit": await self._git_head(clone_dir),
                        "source_branch": branch,
                        "source_repo": source_repo,
                        "venv_path": self._venv_path,
                        "updated_at": utcnow(),
                    },
                )
                if updated:
                    try:
                        from backend.engines.scan.scanner import scan_engine_version

                        scan_engine_version(store, self.engine_id, updated)
                    except Exception as scan_e:
                        logger.warning(f"{self.label} param scan after source sync: %s", scan_e)
                mark_runtime_changed()
            except Exception as exc:
                logger.debug(f"Failed to update {self.label} metadata after sync: {exc}")
            return {}

        return await execute_plan(
            plan,
            run_primitive=make_engine_runner(
                self,
                self._plan_handlers(
                    plan,
                    {
                        **self._extra_plan_handlers(),
                        "vcs.sync": sync_checkout,
                        "publish.record_update": record_update,
                        "publish.finalize": self._finalize_handler(
                            existing_version=version_name,
                            pending_version=version_name,
                            # The sync row is the source of truth for provenance
                            # fields (repo/branch/commit) — merge it like the
                            # legacy engines did.
                            extra=dict(row or {}),
                            branch=branch,
                            include_commit=True,
                        ),
                        "env.compose": self._compose_env,
                    },
                ),
            ),
            context={
                "input.engine_root": self._base_dir,
                "input.branch": branch,
                "input.version_name": version_name,
                "input.source_repo": source_repo,
            },
        )

    async def remove(self, retire_references: bool = False) -> Dict[str, Any]:
        from backend.utils.fs_ops import release_launch_hold

        async with self._lock:
            if self._operation:
                raise RuntimeError(f"Another {self.label} operation is already running")
            store = get_store()
            active = store.get_active_engine_version(self.engine_id)
            venv_path = active.get("venv_path") if active else self._venv_path
            # Remove the installation the ACTIVE row points at (legacy engines
            # derived this from the row, not from the current process binding).
            install_dir = str((active or {}).get("install_dir") or self._base_dir)
            # The launch reference is held on the venv when present; rows that
            # only record an install_dir (legacy source installs) held it there.
            hold_path = str(venv_path or install_dir or "")
            if hold_path and os.path.exists(hold_path):
                release_launch_hold(hold_path, retire_references=retire_references)
            await self._start_operation("remove")

            async def _runner():
                try:
                    execution = await self._execute_remove_plan(
                        venv_path=venv_path or "",
                        install_dir=install_dir,
                        python_exists=os.path.exists(self._venv_python()),
                        retire_references=retire_references,
                        version_name=str((active or {}).get("version") or ""),
                        store=store,
                    )
                    if execution.status != "completed":
                        failure = execution.failed_step
                        raise RuntimeError(
                            failure.error
                            if failure is not None and failure.error
                            else f"{self.label} removal did not complete"
                        )
                    self._forget_remaining_versions(store)
                    await self._finish_operation(True, f"{self.label} removed")
                except Exception as exc:
                    self._last_error = str(exc)
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))

            self._create_task(_runner())
            return self._started_response(f"{self.label} removal started")

    def _forget_remaining_versions(self, store: Any) -> None:
        """Clear failed/cancelled/orphan rows after engine-wide removal.

        The remove recipe is version-scoped by contract (the legacy plan pins
        its step sequence and only deletes the active version). Engine-wide
        removal still has to clear the other associations, or
        ``/api/llama-versions`` keeps reporting stale rows after the engine
        says it is gone.
        """
        try:
            versions = list(store.get_engine_versions(self.engine_id) or [])
            delete_snapshots = getattr(store, "delete_install_snapshots", None)
            seen: set[str] = set()
            for row in versions:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("version") or "")
                if not name or name in seen:
                    continue
                seen.add(name)
                store.delete_engine_version(self.engine_id, name)
                if callable(delete_snapshots):
                    delete_snapshots(self.engine_id, name)
            # A version-scoped remove can leave an empty version directory;
            # the scanner then re-reports it as a broken orphan row. Remove only
            # EMPTY directories, never a tree that may still hold data.
            root = os.path.realpath(getattr(self, "_root_dir", "") or "")
            if root and os.path.isdir(root):
                for entry in os.scandir(root):
                    try:
                        if entry.is_dir(follow_symlinks=False) and not os.listdir(entry.path):
                            os.rmdir(entry.path)
                    except OSError:
                        logger.debug("Could not prune empty %s", entry.path, exc_info=True)
        except Exception:  # noqa: BLE001 — cleanup must not undo a completed removal
            logger.debug(
                "Failed to clear remaining %s engine versions", self.engine_id, exc_info=True
            )

    async def _execute_remove_plan(
        self,
        *,
        venv_path: str,
        install_dir: str = "",
        python_exists: bool,
        retire_references: bool,
        version_name: str,
        store: Any,
    ):
        plan = self._recipe("remove")
        forget_policy = str(self._lifecycle().get("forget_errors") or "swallow").strip().lower()

        async def forget_install(step, requirements, context):
            if version_name:
                try:
                    store.delete_engine_version(self.engine_id, version_name)
                    delete_snapshots = getattr(store, "delete_install_snapshots", None)
                    if callable(delete_snapshots):
                        delete_snapshots(self.engine_id, version_name)
                except Exception as exc:
                    # Default keeps the historical swallow-everything behavior;
                    # `forget_errors: raise` is the strict engine policy.
                    if forget_policy == "raise":
                        raise
                    logger.debug(f"Failed to delete {self.label} engine version metadata: {exc}")
            self._update_installed_state(False, None)
            mark_runtime_changed()
            return {}

        return await execute_plan(
            plan,
            run_primitive=make_engine_runner(
                self,
                self._plan_handlers(
                    plan,
                    {**self._extra_plan_handlers(), "publish.forget": forget_install},
                ),
            ),
            context={
                "input.engine_root": self._base_dir,
                # Both context vocabularies are supplied: recipes declare which
                # they need (venv-family remove uses venv_path; the wheel/
                # install-dir family uses install_dir + installs_root).
                "input.install_dir": install_dir or self._base_dir,
                "input.installs_root": getattr(self, "_root_dir", self._base_dir),
                "input.venv_path": venv_path,
                "input.venv_python_exists": python_exists,
                "input.retire_references": bool(retire_references),
                "input.version_name": version_name,
            },
        )

    # --- Introspection ---------------------------------------------------------------

    async def _update_progress_task(
        self,
        progress: float,
        message: str = "",
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish monotonic integer milestones when log streaming is declared."""
        if not self._lifecycle().get("log"):
            return await super()._update_progress_task(
                progress, message, metadata_update=metadata_update
            )
        if self._progress_task_id:
            from backend.operations.progress import get_progress_manager

            task = get_progress_manager().get_task(self._progress_task_id) or {}
            progress = max(float(task.get("progress") or 0), float(progress))
        await super()._update_progress_task(
            round(progress), message, metadata_update=metadata_update
        )

    async def _run_logged(
        self,
        argv,
        operation: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        append: bool = True,
    ) -> int:
        """Streaming build log when the definition declares ``lifecycle.log``.

        Without a declaration the base installer's log runner is used, so
        engines that never opted in keep their established output behavior.
        """
        log_spec = dict(self._lifecycle().get("log") or {})
        if not log_spec:
            return await super()._run_logged(argv, operation, cwd=cwd, env=env, append=append)
        from backend.engines.python.log_streaming import run_streamed_log

        if not hasattr(self, "_stream_progress"):
            self._stream_progress = 0
        heartbeat = float(
            getattr(self, "_log_heartbeat_seconds", None) or log_spec.get("heartbeat_seconds") or 0
        )
        return await run_streamed_log(
            self,
            argv,
            operation,
            heartbeat_seconds=heartbeat,
            milestones=[dict(item) for item in (log_spec.get("progress_milestones") or ())],
            cwd=cwd,
            env=env,
            append=append,
        )

    def status(self) -> Dict[str, Any]:
        spec = dict(self._lifecycle().get("status") or {})
        if not spec:
            return self._stateful_status(resolve_binary_path=self._resolve_binary_path)
        return self._declared_status(spec)

    def _declared_status(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Status payload projected from the definition's ``status`` block.

        Mirrors the historic per-engine overrides: the venv-interpreter binary,
        an optional ``engine`` key, row-derived fields, constants, and the
        choice of whether ``removed_at`` appears.
        """
        active = get_store().get_active_engine_version(self.engine_id)
        saved_venv = self._venv_path
        try:
            if active and active.get("venv_path"):
                self._venv_path = str(active["venv_path"])
            version = self._detect_installed_version()
            if str(spec.get("binary") or "binary_name") == "venv_python":
                venv = os.path.abspath(self._venv_path)
                python_bin = os.path.join(
                    venv,
                    "Scripts" if os.name == "nt" else "bin",
                    "python.exe" if os.name == "nt" else "python",
                )
                binary_path = python_bin if os.path.isfile(python_bin) else None
                installed = bool(version and os.path.isfile(python_bin))
            else:
                binary_path = self._resolve_binary_path()
                installed = version is not None and binary_path is not None
            row = active or {}
            payload: Dict[str, Any] = {}
            if bool(spec.get("include_engine")):
                payload["engine"] = self.engine_id
            payload.update(
                {
                    "installed": installed,
                    "version": version,
                    "binary_path": binary_path,
                    "venv_path": row.get("venv_path") or self._venv_path,
                    "installed_at": row.get("installed_at"),
                }
            )
            if bool(spec.get("include_removed_at", True)):
                payload["removed_at"] = self._load_state().get("removed_at")
            payload.update(
                {
                    "operation": self._operation,
                    "operation_started_at": self._operation_started_at,
                    "progress_task_id": self._progress_task_id,
                    "last_error": self._last_error,
                    "log_path": self._log_path,
                    "install_type": row.get("install_type"),
                    "source_repo": row.get("source_repo"),
                    "source_branch": row.get("source_branch"),
                }
            )
            for field in spec.get("row_fields") or ():
                payload[str(field)] = row.get(str(field))
            for key, value in dict(spec.get("constants") or {}).items():
                payload[str(key)] = value
            return payload
        finally:
            self._venv_path = saved_venv
