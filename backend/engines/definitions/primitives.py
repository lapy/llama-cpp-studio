"""Shared primitive implementations (Phase 3) — engine-agnostic steps.

These are the recipe primitives whose behavior does not vary per engine:
venv creation, pip install/uninstall invocations, version detection, and
tree removal. They are built against the :class:`PythonVenvInstaller` surface
(``_run_pip``/``_ensure_venv``/``_detect_installed_version``/...), so migrated
engines execute identical process-level behavior to legacy.

Everything engine-specific — publish metadata shapes, version-name policies,
per-engine error strings that live OUTSIDE the pip boundary — stays in the
engine's own primitive handlers, which delegate here for the shared names.
Pinned behavior tests (test_lmdeploy_legacy_plan_shadow.py,
test_vllm_release_plan_shadow.py) are the preservation gate.

Recipe ``with`` keys consumed here:

- ``package``/``package_from`` (``subpath`` joins a constant child, e.g.
  SGLang's ``python/`` monorepo package; a produced LIST — like 1Cat-vLLM's
  wheel URLs — extends argv), ``packages`` (static target list), ``stage``,
  ``version_from``, ``error`` (a ``{code}``/``{path}`` template)
- ``editable`` (adds ``-v -e``), ``extra_args`` (fixed flags after
  ``--upgrade``, e.g. SGLang's ``--pre``), ``append`` (tri-state: omitted
  keys keep ``_run_pip``'s own defaults), ``cwd_from``, ``upgrade`` (set
  false where legacy omitted ``--upgrade``), ``check_exit`` (set false
  where legacy ignored non-zero pip exits), ``packages`` for
  ``pip.uninstall``
- env composition: ``env: ccache`` (os.environ + ccache) or ``env_from``
  (base env produced by an earlier step, e.g. vLLM's CUDA build env) with
  ``env_ccache: true`` layering ccache on top
- ``path_from``/``retire_references_from``/``require_parent_from`` for
  ``fs.remove_tree``; ``path_from`` (+ optional ``subpath``) for
  ``fs.assert_dir``; ``path_from`` for ``fs.ensure_dir`` and
  ``fs.prune_files`` (``suffix`` selects the files to delete)
- ``env_from`` + ``path_from`` for ``env.compose_ccache`` (layer ccache once,
  keyed at the checkout); ``cwd_from``/``env_from``/``extra_index_url``/
  ``requirement_files`` (candidate relative paths, existing ones installed
  in order) for ``pip.install_requirements`` (error template may use
  ``{code}`` and ``{rel}``)
- ``destination``/``workspace_from`` for ``vcs.seal_checkout``
- ``interpreter``/``script_from``/``cwd_from``/``env_from`` for
  ``exec.run_logged`` (append stays tri-state)
- ``url_from``/``path_from`` for ``archive.download`` (delegates to the
  installer's streaming downloader), ``path_from``/``expected_from`` for
  ``archive.verify_sha256``, ``root_from`` for ``archive.verify_checksums_file``
  (a SHA256SUMS file found INSIDE the extracted tree), ``archive_from``/
  ``dest_from``/``delete_after`` for ``archive.extract_tar``
  (path-traversal-guarded r:gz), and ``root_from``/``filename``/``chmod``/
  ``chmod_or``/``required``/``label``/``error`` for ``artifact.locate``
  (``label`` renames the produced key for plans locating several files).
  Archive steps accept ``progress`` (+ ``progress_message`` or
  ``progress_message_from`` naming a produced label) because Unsloth's
  legacy flow announced each network stage.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Mapping, Optional

from .executor import PrimitiveExecutionError
from .plan import PlannedStep

SHARED_PRIMITIVES = frozenset(
    {
        "prereqs.resolve_version",
        "venv.create",
        "pip.install_package",
        "pip.uninstall",
        "pip.capture_metadata",
        "fs.remove_tree",
        "fs.remove_file",
        "fs.assert_dir",
        "fs.ensure_dir",
        "fs.prune_files",
        "env.compose_ccache",
        "pip.install_requirements",
        "vcs.capture_commit",
        "vcs.seal_checkout",
        "exec.run_logged",
        "archive.download",
        "archive.verify_sha256",
        "archive.verify_checksums_file",
        "archive.extract_tar",
        "artifact.locate",
    }
)

_DEFAULT_PIP_ERROR = "pip exited with status {code}"


def _compose_build_env(
    requirements: Mapping[str, Any],
    with_: Mapping[str, Any],
    cwd: Optional[str],
) -> Dict[str, str]:
    """Compose a build environment for a step.

    Base is a produced env (``env_from``, e.g. vLLM's CUDA env or SGLang
    V100's fork environment) or a fresh copy of ``os.environ``; ccache
    (keyed at the step cwd) layers on top when requested. Mirrors the
    legacy call sites value-for-value.
    """
    if with_.get("env_from"):
        base_env: Dict[str, str] = dict(requirements.get(str(with_["env_from"])) or {})
    else:
        base_env = os.environ.copy()
    if str(with_.get("env") or "") == "ccache" or with_.get("env_ccache"):
        from backend.engines.build_workspace import (
            ccache_base_dir,
            ccache_environment,
        )

        base_env.update(
            ccache_environment(
                ccache_base_dir(cwd or os.getcwd()),
                launchers=True,
                cuda=True,
            )
        )
    return base_env


def _sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_tar(archive, dest: str) -> None:
    # Path-traversal guard (Unsloth's legacy helper, byte-for-byte).
    dest_real = os.path.realpath(dest)
    for member in archive.getmembers():
        target = os.path.realpath(os.path.join(dest, member.name))
        if target != dest_real and not target.startswith(dest_real + os.sep):
            raise PrimitiveExecutionError(f"Refusing to extract unsafe path {member.name}")
    archive.extractall(dest)


async def _emit_progress(installer, step, requirements) -> None:
    progress = step.with_.get("progress")
    if progress is None:
        return
    if step.with_.get("progress_message_from"):
        message = str(requirements.get(str(step.with_["progress_message_from"])) or "")
    else:
        message = str(step.with_.get("progress_message") or "")
    await installer._update_progress_task(int(progress), message)


async def run_shared_primitive(
    installer: Any,
    primitive: str,
    step: PlannedStep,
    requirements: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Execute a shared primitive; only names in SHARED_PRIMITIVES may arrive."""
    if primitive == "prereqs.resolve_version":
        # Legacy semantics preserved: the caller (settings pip_version or a
        # retry entry) selects the version; empty string = latest spec.
        return {"version": str(context.get("input.requested_version") or "")}

    if primitive == "venv.create":
        installer._ensure_venv()  # engine overrides (e.g. vLLM's Python gate) apply
        return {"venv": installer._venv_path}

    if primitive == "pip.install_package":
        with_ = step.with_
        package = str(with_.get("package") or "")
        stage = str(with_.get("stage") or "install")
        error_template = str(with_.get("error") or _DEFAULT_PIP_ERROR)
        args = ["install"]
        if with_.get("editable"):
            if with_.get("package_from"):
                package = str(requirements.get(str(with_["package_from"])) or ".")
                if with_.get("subpath"):
                    package = os.path.join(package, str(with_["subpath"]))
            args += ["-v", "-e", package]
        else:
            # ``upgrade`` defaults True (every pre-1Cat recipe passed
            # --upgrade); 1Cat-vLLM's wheel install did NOT.
            if with_.get("upgrade", True):
                args.append("--upgrade")
            for extra in with_.get("extra_args") or ():
                args.append(str(extra))
            if requirements.get("input.force_reinstall"):
                args.append("--force-reinstall")
            # Targets: a static ``packages`` list, a produced value (single
            # string or a LIST of wheel URLs), or the single package spec.
            static = with_.get("packages")
            pkg_from = with_.get("package_from")
            if static:
                args.extend(str(item) for item in static)
            elif pkg_from:
                value = requirements.get(str(pkg_from))
                if isinstance(value, (list, tuple)):
                    args.extend(str(item) for item in value)
                else:
                    args.append(str(value or ""))
            else:
                pinned = str(requirements.get(str(with_.get("version_from") or "version")) or "")
                args.append(f"{package}=={pinned}" if pinned else package)
        kwargs: Dict[str, Any] = {}
        # Tri-state append: an absent with_ key must NOT override _run_pip's
        # default (True); passing False would change log overwrite behavior.
        if "append" in with_:
            kwargs["append"] = bool(with_["append"])
        if with_.get("cwd_from"):
            checkout = requirements.get(str(with_["cwd_from"]))
            if checkout:
                kwargs["cwd"] = str(checkout)
        env_kind = str(with_.get("env") or "")
        needs_ccache = env_kind == "ccache" or bool(with_.get("env_ccache"))
        if env_kind == "ccache" or with_.get("env_from") or with_.get("env_ccache"):
            kwargs["env"] = _compose_build_env(
                requirements, with_, str(kwargs.get("cwd") or "") or None
            )
        code = await installer._run_pip(args, stage, **kwargs)
        if code != 0 and with_.get("check_exit", True):
            # 1Cat-vLLM's release bootstrap pip ignored the exit code
            # outright; every other legacy call site failed on non-zero.
            raise PrimitiveExecutionError(error_template.format(code=code), exit_code=code)
        return {}

    if primitive == "pip.uninstall":
        with_ = step.with_
        packages = [str(p) for p in (with_.get("packages") or [])] or [
            str(with_.get("package") or "")
        ]
        stage = str(with_.get("stage") or "remove")
        kwargs: Dict[str, Any] = {"ensure_venv": False}
        if "append" in with_:
            kwargs["append"] = bool(with_["append"])
        code = await installer._run_pip(["uninstall", "-y", *packages], stage, **kwargs)
        if code != 0:
            raise PrimitiveExecutionError(
                str(with_.get("error") or _DEFAULT_PIP_ERROR).format(code=code),
                exit_code=code,
            )
        return {}

    if primitive == "pip.capture_metadata":
        detected = installer._detect_installed_version()
        installer._update_installed_state(True, detected)
        return {"metadata": {"version": detected}}

    if primitive == "fs.remove_tree":
        from backend.utils.fs_ops import robust_rmtree

        path = str(requirements.get(str(step.with_.get("path_from") or "path")) or "")
        guarded = step.with_.get("require_parent_from")
        if path and os.path.exists(path):
            # vLLM only removes directories that live directly under the
            # engine root; a mismatch skips removal entirely (legacy guard).
            if guarded:
                expected_parent = str(
                    context.get(str(guarded)) or requirements.get(str(guarded)) or ""
                )
                if not expected_parent or os.path.dirname(path) != expected_parent:
                    return {}
            robust_rmtree(
                path,
                retire_references=bool(
                    requirements.get(str(step.with_.get("retire_references_from") or ""))
                ),
            )
        return {}

    if primitive == "fs.assert_dir":
        path = str(requirements.get(str(step.with_.get("path_from") or "path")) or "")
        if path and step.with_.get("subpath"):
            path = os.path.join(path, str(step.with_["subpath"]))
        if not path or not os.path.isdir(path):
            template = str(step.with_.get("error") or "{path} is not a directory")
            raise PrimitiveExecutionError(template.format(code=0, path=path))
        return {}

    if primitive == "vcs.capture_commit":
        path = str(requirements.get(str(step.with_.get("path_from") or "checkout")) or "")
        commit = await installer._git_head(path)
        return {"source_commit": commit}

    if primitive == "vcs.seal_checkout":
        # Reuse-mode installs hold no managed workspace; sealing is a no-op
        # there (legacy skipped it via a None check). The sealed path
        # rebinds `checkout` for later steps, as vLLM's git-head did.
        workspace = context.get(str(step.with_.get("workspace_from") or "workspace"))
        if workspace is None:
            return {}
        destination = os.path.join(
            installer._base_dir, str(step.with_.get("destination") or "source")
        )
        sealed = await asyncio.to_thread(
            workspace.seal_installed_source, destination, installer._venv_path
        )
        return {"checkout": sealed}

    if primitive == "exec.run_logged":
        # Runs a produced script through the installer's logged executor
        # (SGLang V100's builder script today; any "run this checked-in
        # script with env X" flow tomorrow). argv order, stage, cwd, env
        # layering and the tri-state append mirror the legacy call site.
        with_ = step.with_
        script = str(requirements.get(str(with_.get("script_from") or "script")) or "")
        argv = [str(with_.get("interpreter") or "bash"), script]
        stage = str(with_.get("stage") or "install_source")
        kwargs: Dict[str, Any] = {}
        if "append" in with_:
            kwargs["append"] = bool(with_["append"])
        cwd: Optional[str] = None
        if with_.get("cwd_from"):
            cwd = str(requirements.get(str(with_["cwd_from"])) or "") or None
            if cwd:
                kwargs["cwd"] = cwd
        if with_.get("env_from") or with_.get("env_ccache") or str(with_.get("env") or ""):
            kwargs["env"] = _compose_build_env(requirements, with_, cwd)
        code = await installer._run_logged(argv, stage, **kwargs)
        if code != 0:
            raise PrimitiveExecutionError(
                str(with_.get("error") or "command exited with status {code}").format(code=code),
                exit_code=code,
            )
        return {}

    if primitive == "fs.ensure_dir":
        path = str(requirements.get(str(step.with_.get("path_from") or "path")) or "")
        if path:
            os.makedirs(path, exist_ok=True)
        return {}

    if primitive == "fs.remove_file":
        path = str(requirements.get(str(step.with_.get("path_from") or "path")) or "")
        if path and os.path.isfile(path):
            # Native version removal holds a launch guard on the server
            # binary itself: release_hold_from names a requirement whose
            # boolean feeds release_launch_hold (imported at call time,
            # like robust_rmtree, so tests can patch the seam), and
            # "optional" swallows removal errors the legacy fallback
            # swallowed. Both keys stay unset for plain archive cleanups.
            hold_key = str(step.with_.get("release_hold_from") or "")
            if hold_key:
                from backend.utils.fs_ops import release_launch_hold

                release_launch_hold(path, retire_references=bool(requirements.get(hold_key)))
            try:
                os.remove(path)
            except OSError:
                if not step.with_.get("optional"):
                    raise
        return {}

    if primitive == "fs.prune_files":
        # 1Cat-vLLM's sync wipes stale wheels before rebuilding; removal
        # errors were swallowed by the legacy loop.
        path = str(requirements.get(str(step.with_.get("path_from") or "path")) or "")
        suffix = str(step.with_.get("suffix") or "")
        if path and suffix and os.path.isdir(path):
            for name in os.listdir(path):
                if name.endswith(suffix):
                    try:
                        os.remove(os.path.join(path, name))
                    except OSError:
                        pass
        return {}

    if primitive == "env.compose_ccache":
        # Layers ccache (keyed at a checkout path) onto a produced build
        # environment exactly once, BEFORE any step consumes it — 1Cat's
        # legacy code mutated the env once at this point, so the layered
        # value must not depend on later steps' cwds.
        with_ = step.with_
        base = dict(requirements.get(str(with_.get("env_from") or "build_env")) or {})
        path = str(requirements.get(str(with_.get("path_from") or "checkout")) or "")
        from backend.engines.build_workspace import (
            ccache_base_dir,
            ccache_environment,
        )

        base.update(
            ccache_environment(
                ccache_base_dir(path or os.getcwd()),
                launchers=True,
                cuda=True,
            )
        )
        return {"build_env": base}

    if primitive == "pip.install_requirements":
        # Installs each EXISTING candidate requirements file (first-existing
        # order preserved from the recipe's candidate list), one pip call per
        # file, with the legacy per-file error string.
        with_ = step.with_
        clone = str(requirements.get(str(with_.get("cwd_from") or "checkout")) or "")
        index = str(with_.get("extra_index_url") or "")
        stage = str(with_.get("stage") or "install_source")
        error_template = str(with_.get("error") or "pip install -r {rel} failed ({code})")
        env = (
            dict(requirements.get(str(with_["env_from"])) or {}) if with_.get("env_from") else None
        )
        for rel in with_.get("requirement_files") or ():
            req_path = os.path.join(clone, str(rel))
            if not os.path.isfile(req_path):
                continue
            args = ["install"]
            if index:
                args += ["--extra-index-url", index]
            args += ["-r", req_path]
            code = await installer._run_pip(args, stage, cwd=clone, env=env)
            if code != 0:
                raise PrimitiveExecutionError(
                    error_template.format(code=code, rel=os.path.relpath(req_path, clone)),
                    exit_code=code,
                )
        return {}

    if primitive == "archive.download":
        await _emit_progress(installer, step, requirements)
        url = str(requirements.get(str(step.with_.get("url_from") or "url")) or "")
        dest = str(requirements.get(str(step.with_.get("path_from") or "path")) or "")
        await installer._download_file(url, dest)
        return {}

    if primitive == "archive.verify_sha256":
        await _emit_progress(installer, step, requirements)
        path = str(requirements.get(str(step.with_.get("path_from") or "path")) or "")
        want = (
            str(requirements.get(str(step.with_.get("expected_from") or "expected_sha256")) or "")
            .strip()
            .lower()
        )
        actual = _sha256_file(path)
        if actual.lower() != want:
            raise PrimitiveExecutionError(
                f"Checksum mismatch for {os.path.basename(path)}: expected {want}, got {actual}"
            )
        return {}

    if primitive == "archive.extract_tar":
        import tarfile

        await _emit_progress(installer, step, requirements)
        archive_path = str(requirements.get(str(step.with_.get("archive_from") or "archive")) or "")
        dest = str(requirements.get(str(step.with_.get("dest_from") or "path")) or "")
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract_tar(archive, dest)
        if step.with_.get("delete_after") and os.path.exists(archive_path):
            os.remove(archive_path)
        return {}

    if primitive == "archive.verify_checksums_file":
        # audio.cpp style: a SHA256SUMS file shipped INSIDE the archive.
        # Missing file = nothing to verify; missing entries/files skip.
        root = str(requirements.get(str(step.with_.get("root_from") or "path")) or "")
        checksum_path = ""
        for dirpath, _, files in os.walk(root):
            if "SHA256SUMS" in files:
                checksum_path = os.path.join(dirpath, "SHA256SUMS")
                break
        if not checksum_path:
            return {}
        base = os.path.dirname(checksum_path)
        with open(checksum_path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                digest, _, filename = text.partition(" ")
                filename = filename.strip().lstrip("*")
                if not digest or not filename:
                    continue
                target = os.path.join(base, filename)
                if not os.path.isfile(target):
                    continue
                if _sha256_file(target).lower() != digest.lower():
                    raise PrimitiveExecutionError(f"Checksum mismatch for {filename}")
        return {}

    if primitive == "artifact.locate":
        root = str(requirements.get(str(step.with_.get("root_from") or "path")) or "")
        filename = str(step.with_.get("filename") or "")
        # `label` renames the output so one plan can locate several artifacts.
        label = str(step.with_.get("label") or "binary")
        found = None
        for dirpath, _, filenames in os.walk(root):
            if filename in filenames:
                found = os.path.join(dirpath, filename)
                break
        if not found:
            if not step.with_.get("required", True):
                return {label: ""}
            template = str(step.with_.get("error") or f"{filename} was not found under {root}")
            raise PrimitiveExecutionError(template.format(code=0, path=found or root))
        if step.with_.get("chmod_or"):
            os.chmod(found, os.stat(found).st_mode | int(str(step.with_["chmod_or"]), 8))
        elif step.with_.get("chmod"):
            os.chmod(found, int(str(step.with_["chmod"]), 8))
        return {label: found}

    raise PrimitiveExecutionError(f"primitive {primitive!r} is not engine-agnostic")


def make_engine_runner(
    installer: Any,
    engine_primitives: Mapping[str, Any],
) -> "object":
    """Compose shared + engine primitives into a runner for execute_plan.

    Engine-local handlers win for their own names; everything in
    SHARED_PRIMITIVES routes here. Raises for names neither side claims, so a
    recipe can never silently no-step.
    """

    async def run_primitive(
        primitive: Optional[str],
        step: PlannedStep,
        requirements: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        handler = engine_primitives.get(primitive or "")
        if handler is not None:
            return dict(await handler(step, requirements, context) or {})
        if primitive in SHARED_PRIMITIVES:
            return await run_shared_primitive(
                installer, primitive or "", step, requirements, context
            )
        raise PrimitiveExecutionError(
            f"{installer.engine_id} plan uses unimplemented primitive {primitive!r}"
        )

    return run_primitive
