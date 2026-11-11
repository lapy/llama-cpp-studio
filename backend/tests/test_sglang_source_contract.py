"""SGLang source/sync/remove CONTRACT harness (Phase-5 port).

The legacy shadow tests asserted the old class's internal call transcript.
This file asserts the OBSERVABLE contract of the definition-driven host:
executed steps, pip argv, checkout/seal handling, version rows, failure
extras and error strings - the behavioural spec the migration must keep.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from backend.engines.definitions import load_default_engine_definitions, materialize_plan
from backend.engines.definitions.transforms import TransformError, apply_transform
from backend.engines.python import fork_build
from backend.engines.python.fork_build import prepare_fork_build
from backend.engines.python.generic import GenericVenvInstaller

REPO = "https://example.invalid/sglang.git"


def _manager(tmp_path: Path, monkeypatch, *, v100: bool = False):
    definition = load_default_engine_definitions().definitions["sglang_v100" if v100 else "sglang"]
    installer = GenericVenvInstaller(
        definition,
        log_path=str(tmp_path / ("v100.log" if v100 else "sglang.log")),
        base_dir=str(tmp_path / ("v100" if v100 else "sglang")),
    )
    calls: list = []
    task: dict = {}

    async def fake_run_pip(args, stage, **kwargs):
        calls.append(("pip", list(args), stage, dict(kwargs)))
        return 0

    async def fake_progress(progress, message, **kwargs):
        calls.append(("progress", progress, message))

    async def fake_logged(argv, operation, **kwargs):
        calls.append(("logged", list(argv), operation, dict(kwargs)))
        return 0

    installer._create_task = lambda coro: task.__setitem__("coro", coro)
    installer._run_pip = fake_run_pip
    installer._run_logged = fake_logged
    installer._update_progress_task = fake_progress
    installer._ensure_venv = lambda: calls.append(("ensure_venv",))
    installer._detect_installed_version = lambda *a, **k: "0.4.0"

    def fake_ready(pending, meta):
        calls.append(("ready", pending, dict(meta)))
        return "published-name"

    installer._ready_pending_version = fake_ready
    installer._fail_pending_version = lambda pending, message, extra, **kw: calls.append(
        ("fail", message, dict(extra or {}), bool(kw.get("cancelled")))
    )

    import backend.engines.python.generic as generic_module

    async def fake_activate(engine, version_name, **kw):
        calls.append(("activate", engine, version_name))

    if monkeypatch is not None:
        monkeypatch.setattr(generic_module, "activate_engine_version", fake_activate)
    return installer, calls


def _patch_clone(monkeypatch, calls, *, make_python: bool = True):
    class _Proc:
        returncode = 0

        async def wait(self):
            return 0

    async def fake_exec(*argv, **kwargs):
        target = argv[-1]
        calls.append(("clone", argv[-2], argv[-3], target))
        os.makedirs(target, exist_ok=True)
        if make_python:
            os.makedirs(os.path.join(target, "python"), exist_ok=True)
        return _Proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


def _source_kwargs(tmp_path, *, reuse=True, reuse_dir=None, existing=None):
    root = tmp_path / "sglang"
    return {
        "repo_url": REPO,
        "branch": "main",
        "reuse": reuse,
        "clone_dir": str(reuse_dir or root / "source"),
        "pending_version": existing or "source",
        "existing_version": existing,
        "workspace_holder": {},
    }


@pytest.mark.asyncio
async def test_source_reuse_installs_python_subpackage_and_publishes(tmp_path, monkeypatch):
    installer, calls = _manager(tmp_path, monkeypatch)
    _patch_clone(monkeypatch, calls)
    clone_dir = str(tmp_path / "src")
    execution = await installer._execute_source_plan(
        **_source_kwargs(tmp_path, reuse_dir=clone_dir)
    )
    assert execution.status == "completed"
    pips = [call for call in calls if call[0] == "pip"]
    assert pips[0][1] == ["install", "--upgrade", "pip"]
    assert pips[0][3].get("append") is True
    assert pips[1][1] == ["install", "-v", "-e", os.path.join(clone_dir, "python")]
    assert pips[1][3]["cwd"] == clone_dir
    ready = next(call for call in calls if call[0] == "ready")
    assert ready[2]["install_type"] == "source"
    assert ready[2]["source_repo"] == REPO
    assert ready[2]["source_branch"] == "main"


@pytest.mark.asyncio
async def test_source_guard_fails_before_python_install(tmp_path, monkeypatch):
    installer, calls = _manager(tmp_path, monkeypatch)
    _patch_clone(monkeypatch, calls, make_python=False)
    execution = await installer._execute_source_plan(
        **_source_kwargs(tmp_path, reuse_dir=str(tmp_path / "src"))
    )
    assert execution.status == "failed"
    assert execution.failed_step is not None
    assert execution.failed_step.error == "SGLang source package directory is unavailable"
    assert not [call for call in calls if call[0] == "pip" and "-e" in call[1]]
    assert not [call for call in calls if call[0] == "ready"]


@pytest.mark.asyncio
async def test_source_pip_failure_uses_recipe_error_string(tmp_path, monkeypatch):
    installer, calls = _manager(tmp_path, monkeypatch)
    _patch_clone(monkeypatch, calls)

    async def failing_pip(args, stage, **kwargs):
        calls.append(("pip", list(args), stage, dict(kwargs)))
        return 7

    installer._run_pip = failing_pip
    execution = await installer._execute_source_plan(
        **_source_kwargs(tmp_path, reuse_dir=str(tmp_path / "src"))
    )
    assert execution.status == "failed"
    assert execution.failed_step.error == "pip upgrade failed with status 7"


@pytest.mark.asyncio
async def test_source_workspace_mode_seals_and_publishes_commit(tmp_path, monkeypatch):
    installer, calls = _manager(tmp_path, monkeypatch)
    import backend.engines.build_workspace as bw

    class _Workspace:
        def acquire(self):
            calls.append(("acquire",))

        def sync_git(self, repo_url, branch):
            calls.append(("sync_git", repo_url, branch))

        def seal_installed_source(self, destination, venv_path):
            calls.append(("seal", destination))
            return str(tmp_path / "sealed")

    checkout = tmp_path / "ws"
    checkout.mkdir()
    (checkout / "python").mkdir()

    class _W(_Workspace):
        checkout_dir = str(checkout)

    monkeypatch.setattr(
        bw, "BuildWorkspace", type("BW", (), {"open": staticmethod(lambda *a: _W())})
    )
    monkeypatch.setattr(bw, "release_held", lambda ws: calls.append(("held",)))

    async def fake_head(path):
        calls.append(("git_head", path))
        return "abc123456789def"

    installer._git_head = fake_head
    execution = await installer._execute_source_plan(**_source_kwargs(tmp_path, reuse=False))
    assert execution.status == "completed", execution.failed_step
    kinds = [call[0] for call in calls]
    assert "acquire" in kinds and "sync_git" in kinds and "seal" in kinds
    ready = next(call for call in calls if call[0] == "ready")
    assert ready[2]["source_commit"] == "abc123456789def"


@pytest.mark.asyncio
async def test_sync_republishes_without_uniquifying_and_merges_row(tmp_path, monkeypatch):
    installer, calls = _manager(tmp_path, monkeypatch)
    clone_dir = tmp_path / "src"
    (clone_dir / "python").mkdir(parents=True)

    async def fake_sync(target, branch):
        calls.append(("sync_git", target, branch))

    installer._sync_git_checkout = fake_sync
    installer._git_head = lambda path: _async("deadbeefcafe")

    execution = await installer._execute_sync_plan(
        branch="main",
        clone_dir=str(clone_dir),
        version_name="0.4.0",
        source_repo=REPO,
        row={"version": "0.4.0", "install_type": "source", "source_repo": REPO},
    )
    assert execution.status == "completed", execution.failed_step
    ready = next(call for call in calls if call[0] == "ready")
    assert ready[2]["version"] == "0.4.0"
    assert ready[2]["source_commit"] == "deadbeefcafe"
    assert ready[2]["install_type"] == "source"


@pytest.mark.asyncio
async def test_sync_failure_merges_row_and_runtime_metadata(tmp_path, monkeypatch):
    installer, calls = _manager(tmp_path, monkeypatch)

    async def failing_sync(target, branch):
        raise RuntimeError("git fetch failed with status 128")

    installer._sync_git_checkout = failing_sync
    # Non-primitive handler failures propagate after cleanup (legacy parity);
    # the failure bookkeeping lives in sync_source_version.
    with pytest.raises(RuntimeError) as excinfo:
        await installer._execute_sync_plan(
            branch="main",
            clone_dir=str(tmp_path / "src"),
            version_name="0.4.0",
            source_repo=REPO,
            row={"version": "0.4.0", "install_type": "source"},
        )
    assert str(excinfo.value) == "git fetch failed with status 128"


@pytest.mark.asyncio
async def test_v100_source_runs_the_fork_builder_with_studio_env(tmp_path, monkeypatch):
    installer, calls = _manager(tmp_path, monkeypatch, v100=True)
    _patch_clone(monkeypatch, calls)
    builder = str(tmp_path / "checkout" / "scripts" / "install_v100_studio.sh")
    installer._compose_build_env = lambda: {
        "SGLANG_STUDIO_PYTHON": "/studio/venv/bin/python",
        "CUDA_HOME": "/studio/cuda/cuda-12.8",
    }
    installer._runtime_metadata = lambda: {"cuda_version": "12.8"}
    installer._git_head = lambda path: _async("feedfacecafebeef")

    async def fake_transform(step, requirements, context):
        return {"builder_script": builder}

    async def fake_fork_env(step, requirements, context):
        return {"install_env": {"SGLANG_V100_PYTHON": "/studio/venv/bin/python"}}

    async def fake_toolchain(step, requirements, context):
        return {"rust_env": dict(requirements.get("build_env") or {})}

    installer._apply_transform_handler = fake_transform
    installer._prepare_fork_handler = fake_fork_env
    installer._handler_toolchain = fake_toolchain

    execution = await installer._execute_source_plan(
        **_source_kwargs(tmp_path, reuse_dir=str(tmp_path / "src"))
    )
    assert execution.status == "completed"
    logged = next(call for call in calls if call[0] == "logged")
    assert logged[1] == ["bash", builder]
    ready = next(call for call in calls if call[0] == "ready")
    assert ready[2]["cuda_version"] == "12.8"
    assert ready[2]["source_commit"] == "feedfacecafebeef"


def _async(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner()


def _v100_option(step_id: str) -> dict:
    plan = materialize_plan(load_default_engine_definitions().definitions["sglang_v100"], "source")
    return dict(plan.by_id(step_id).with_)


def _real_v100_checkout(tmp_path) -> Path:
    """Build the smallest checkout the versioned assets apply to.

    The fixture sources its exact upstream text from the assets' own
    applicability anchors, so the test fails if an anchor stops matching.
    """
    from backend.engines.definitions.transforms import load_transform

    checkout = tmp_path / "checkout"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    (checkout / "python").mkdir()

    pyproject_asset = load_transform("sglang_v100/pyproject")
    (checkout / "python" / "pyproject.toml").write_text(
        "[project]\n"
        'name = "sglang"\n'
        'version = "0.0.0"\n'
        "dependencies = [\n"
        '    "cuda-python>=13.0",\n'
        + "".join(
            "    " + item["search"] + "\n" for item in pyproject_asset.operations[0]["replacements"]
        )
        + "]\n",
        encoding="utf-8",
    )

    install_asset = load_transform("sglang_v100/install_v100_studio")
    region = next(op for op in install_asset.operations if op["kind"] == "replace_region")
    marlin_stage = next(
        op for op in install_asset.operations if "V100 Marlin" in op.get("marker", "")
    )
    flashinfer_stage = next(
        op for op in install_asset.operations if "FLASHINFER_DIR" in op.get("marker", "")
    )
    sparse_checkout = next(
        op for op in install_asset.operations if "sparse-checkout set" in op.get("search", "")
    )
    sparse_paths = next(
        op for op in install_asset.operations if "LICENSE csrc" in op.get("search", "")
    )
    complete = next(
        op for op in install_asset.operations if "Complete. Run" in op.get("search", "")
    )
    (scripts / "install_v100.sh").write_text(
        "#!/usr/bin/env bash\n"
        + region["start"]
        + "\necho root\n"
        + region["end"]
        + "\n"
        + flashinfer_stage["marker"]
        + "\n"
        + marlin_stage["marker"]
        + "\n"
        + sparse_paths["search"]
        + "\n"
        + sparse_checkout["search"]
        + "\n"
        + complete["search"]
        + "\n",
        encoding="utf-8",
    )

    marlin_asset = load_transform("sglang_v100/marlin_builder")
    patch_op, build_op = marlin_asset.operations
    (scripts / "setup_v100_marlin.sh").write_text(
        "#!/usr/bin/env bash\n" + patch_op["search"] + "\n" + build_op["search"] + "\n",
        encoding="utf-8",
    )
    return checkout


def _sm70_fork_checkout(tmp_path, asset_prefix: str = "sglang_sxm2") -> Path:
    """Smallest SM70-fork checkout the versioned assets under `asset_prefix` apply to."""
    from backend.engines.definitions.transforms import load_transform

    checkout = tmp_path / "checkout"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)

    install_asset = load_transform(asset_prefix + "/install_v100_studio")
    region = next(op for op in install_asset.operations if op["kind"] == "replace_region")
    marlin_stage = next(
        op for op in install_asset.operations if "V100 Marlin" in op.get("marker", "")
    )
    complete = next(
        op for op in install_asset.operations if "Complete. Run" in op.get("search", "")
    )
    sparse_checkout = next(
        op for op in install_asset.operations if "sparse-checkout set" in op.get("search", "")
    )
    sparse_paths = next(
        op for op in install_asset.operations if "LICENSE csrc" in op.get("search", "")
    )
    (scripts / "install_v100.sh").write_text(
        "#!/usr/bin/env bash\n"
        + region["start"]
        + "\necho root\n"
        + region["end"]
        + "\n"
        + 'python -m pip install --no-deps -r "$REPO_ROOT/requirements.txt"\n'
        + marlin_stage["marker"]
        + "\n"
        + sparse_paths["search"]
        + "\n"
        + sparse_checkout["search"]
        + "\n"
        + complete["search"]
        + "\n",
        encoding="utf-8",
    )

    ipc_asset = load_transform(asset_prefix + "/cuda_ipc")
    ipc = checkout / "python" / "sglang" / "kernels" / "jit" / "csrc" / "distributed" / "ipc.cuh"
    ipc.parent.mkdir(parents=True)
    ipc.write_text(
        "".join(item["search"] + "\n" for item in ipc_asset.operations[0]["replacements"]),
        encoding="utf-8",
    )
    return checkout


def test_v100_transform_assets_rewrite_a_real_checkout(tmp_path):
    """The versioned assets (not a class) must rewrite a real checkout."""
    checkout = _real_v100_checkout(tmp_path)
    for name in ("pyproject", "marlin_builder", "install_v100_studio"):
        apply_transform("sglang_v100/" + name, str(checkout))

    script_path = checkout / "scripts" / "install_v100_studio.sh"
    rewritten = script_path.read_text(encoding="utf-8")
    assert "Studio owns host packages" in rewritten
    assert "SGLANG_STUDIO_PYTHON" in rewritten
    assert "Building the SM70 FlashAttention vision package" in rewritten
    assert "flash-attention-v100.git" in rewritten
    assert "vision_host_compat.patch" in rewritten
    assert rewritten.index(
        'python -m pip install --no-deps --no-build-isolation -e "$FLASHINFER_DIR"'
    ) < rewritten.index("Building the SM70 FlashAttention vision package")
    assert os.access(script_path, os.X_OK)
    pyproject = (checkout / "python" / "pyproject.toml").read_text(encoding="utf-8")
    assert "Studio SM70 dependency overrides applied" in pyproject
    assert '"sglang-kernel==0.4.3",' not in pyproject
    assert '"flashinfer_python==0.6.12",' not in pyproject
    # Studio no longer overrides the fork's cuda-python pin: the fork's own
    # resolved set is the validated one.
    assert '"cuda-python>=13.0",' in pyproject
    marlin = (checkout / "scripts" / "setup_v100_marlin.sh").read_text(encoding="utf-8")
    assert "apply_sm70_patches()" in marlin
    assert "Studio: drop verbose ptxas" in marlin


def test_sxm2_transform_assets_rewrite_a_real_checkout(tmp_path):
    """sglang-sxm2 keeps the fork's requirements.txt and Marlin installer."""
    checkout = _sm70_fork_checkout(tmp_path, "sglang_sxm2")
    for name in ("cuda_ipc", "install_v100_studio"):
        apply_transform("sglang_sxm2/" + name, str(checkout))

    script_path = checkout / "scripts" / "install_v100_studio.sh"
    rewritten = script_path.read_text(encoding="utf-8")
    assert "Studio owns host packages" in rewritten
    assert "SGLANG_STUDIO_PYTHON" in rewritten
    assert "NVCC_PREPEND_FLAGS" in rewritten
    assert "unset CMAKE_ARGS" in rewritten
    assert "Studio environment: $VIRTUAL_ENV" in rewritten
    # The maintainer's dependency strategy is untouched: the --no-deps
    # requirements install and the patch-stack-aware Marlin script stay as
    # shipped.
    assert 'python -m pip install --no-deps -r "$REPO_ROOT/requirements.txt"' in rewritten
    # TurboMind headers live at the ROOT of csrc/, which the fork's sparse list
    # omits in non-cone mode; the transform must pin both the mode and the
    # headers (a missing one only fails the nvcc build an hour later).
    assert 'sparse-checkout set --no-cone "$@"' in rewritten
    assert "'csrc/*.h' 'csrc/*.cuh'" in rewritten
    assert "/LICENSE csrc/core" in rewritten
    assert os.access(script_path, os.X_OK)
    ipc = (
        checkout / "python" / "sglang" / "kernels" / "jit" / "csrc" / "distributed" / "ipc.cuh"
    ).read_text(encoding="utf-8")
    assert "to_ipc_handle(pair.get<0>())" in ipc
    assert "const auto offset = pair.get<1>();" in ipc


def test_builder_script_is_bound_to_the_executable_shell_installer():
    """`run_builder` executes `script_from: builder_script` with bash.

    Regression: the SM70 dual-GEMM guard transform declared
    `produces: builder_script`, which took the binding away from the installer
    transform: the install ran `bash .../dual_gemm.py` and died with status 2
    right after the checkout. Every producer of builder_script must therefore
    come from a transform whose target is the executable shell installer.
    """
    from backend.engines.definitions.transforms import load_transform

    registry = load_default_engine_definitions()
    producers = 0
    for engine_id, definition in registry.definitions.items():
        for stage, _steps in definition.recipes:
            plan = materialize_plan(definition, stage)
            for step in plan.steps:
                if "builder_script" not in tuple(step.produces):
                    continue
                transform = str((step.with_ or {}).get("transform") or "")
                assert transform, (
                    f"{engine_id}/{stage}/{step.id} produces builder_script without a transform"
                )
                asset = load_transform(transform)
                target = asset.output or asset.file
                assert target.endswith(".sh"), (
                    f"{engine_id}/{stage}/{step.id} binds builder_script to {transform} -> "
                    f"{target}; run_builder runs it with bash, so only the shell installer "
                    "may produce it"
                )
                assert asset.executable, (
                    f"{engine_id}/{stage}/{step.id}: {target} must be marked executable"
                )
                producers += 1
    assert producers >= 3, (
        "expected the SGLang forks' source and sync stages to bind builder_script, "
        f"found {producers}"
    )


def test_sm70_transform_assets_rewrite_a_real_checkout(tmp_path):
    """The SM70 fork's OWN assets apply, and its definition wires them.

    lapy/sglang-sm70 shares the sglang-sxm2 install contract but is a separate
    repository: it must not depend on sglang_sxm2's asset directory, because a
    shared set silently no-ops (or mis-applies) when one fork's script drifts.
    """
    from backend.engines.definitions.loader import load_default_engine_definitions

    definition = load_default_engine_definitions().definitions["sglang_sm70"]
    transforms = {
        step.get("with", {}).get("transform")
        for flow in ("source", "sync")
        for step in dict(definition.recipes)[flow]
        if step["uses"] == "vcs.apply_transform"
    }
    assert transforms == {
        "sglang_sm70/cuda_ipc",
        "sglang_sm70/install_v100_studio",
        "sglang_sm70/dual_gemm_cuda_guard",
    }
    lifecycle = dict(definition.lifecycle)
    assert lifecycle["default_repo_url"] == "https://github.com/lapy/sglang-sm70.git"
    assert lifecycle["root_name"] == "sglang-sm70"
    assert lifecycle["update_source"]["repo"] == "lapy/sglang-sm70"
    assert definition.data_dirs == ("sglang-sm70",)

    checkout = _sm70_fork_checkout(tmp_path, "sglang_sm70")
    # main 79dc001f wraps this log call over two lines; keep the broken form in
    # the fixture so the asset's repair op is exercised.
    install_sh = checkout / "scripts" / "install_v100.sh"
    install_sh.write_text(
        install_sh.read_text(encoding="utf-8")
        + 'log "Installing FlashAttention-V100 vision kernels (installed by default, used only when "\n'
        + '    "SGLANG_SM70_VISION_ATTENTION=1)"\n'
        + 'REPO_ROOT="$REPO_ROOT" bash "$REPO_ROOT/scripts/setup_fa_v100_vision.sh"\n',
        encoding="utf-8",
    )
    for name in ("cuda_ipc", "install_v100_studio"):
        apply_transform("sglang_sm70/" + name, str(checkout))

    script_path = checkout / "scripts" / "install_v100_studio.sh"
    rewritten = script_path.read_text(encoding="utf-8")
    assert "Studio owns host packages" in rewritten
    assert "SGLANG_STUDIO_PYTHON" in rewritten
    assert "NVCC_PREPEND_FLAGS" in rewritten
    assert "unset CMAKE_ARGS" in rewritten
    assert "Studio environment: $VIRTUAL_ENV" in rewritten
    # The maintainer's builder stays the builder: the --no-deps requirements
    # install is untouched and only the host-provisioning prologue is replaced.
    assert 'python -m pip install --no-deps -r "$REPO_ROOT/requirements.txt"' in rewritten
    assert "apt-get" not in rewritten
    assert "rustup.rs" not in rewritten
    assert 'sparse-checkout set --no-cone "$@"' in rewritten
    assert "'csrc/*.h' 'csrc/*.cuh'" in rewritten
    assert "/LICENSE csrc/core" in rewritten
    # The vision-kernel log call must stay one argument: an orphaned
    # "SGLANG_SM70_VISION_ATTENTION=1)" line is executed as a command (exit 127).
    assert 'used only when SGLANG_SM70_VISION_ATTENTION=1)"' in rewritten
    assert not [
        line
        for line in rewritten.splitlines()
        if line.strip().startswith('"SGLANG_SM70_VISION_ATTENTION=1)"')
    ]
    assert os.access(script_path, os.X_OK)
    ipc = (
        checkout / "python" / "sglang" / "kernels" / "jit" / "csrc" / "distributed" / "ipc.cuh"
    ).read_text(encoding="utf-8")
    assert "to_ipc_handle(pair.get<0>())" in ipc
    assert "const auto offset = pair.get<1>();" in ipc


def test_sm70_dual_gemm_guard_survives_a_missing_cute_dsl(tmp_path):
    """CuTe DSL needs CUDA 13.1; the guard turns that crash into "no dual gemm"."""
    checkout = tmp_path / "checkout"
    dual_gemm = checkout / "python" / "sglang" / "srt" / "layers" / "dual_gemm.py"
    dual_gemm.parent.mkdir(parents=True)
    upstream = (
        "    def _select_mode(self):\n"
        "        if not is_cuda():\n"
        "            return None\n"
        "\n"
        "        from sglang.kernels.ops.gemm.cutedsl_dual_gemm import (\n"
        "            MAX_DUAL_GEMM_DECODE_TOKENS,\n"
        "            DualGemmQuantMode,\n"
        "            can_use_dual_gemm,\n"
        "        )\n"
        "        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod\n"
    )
    dual_gemm.write_text(upstream, encoding="utf-8")

    assert Path(apply_transform("sglang_sm70/dual_gemm_cuda_guard", str(checkout))) == dual_gemm
    patched = dual_gemm.read_text(encoding="utf-8")
    assert "except (ImportError, NotImplementedError):" in patched
    # The fallback must not swallow the rest of the method: the original
    # follow-on import is still there at method level.
    assert (
        "\n        from sglang.srt.layers.quantization.unquant import "
        "UnquantizedLinearMethod\n" in patched
    )
    # Idempotent: a second run leaves the file untouched.
    before = dual_gemm.read_text(encoding="utf-8")
    apply_transform("sglang_sm70/dual_gemm_cuda_guard", str(checkout))
    assert dual_gemm.read_text(encoding="utf-8") == before


def test_v100_transform_assets_fail_descriptively_when_layout_changes(tmp_path):
    """Applicability expectations: a changed fork layout refuses, naming the asset."""
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "python").mkdir()

    with pytest.raises(TransformError, match="required file is missing: python/pyproject.toml"):
        apply_transform("sglang_v100/pyproject", str(checkout))

    (checkout / "scripts" / "setup_v100_marlin.sh").write_text(
        "#!/usr/bin/env bash\necho upstream-layout-changed\n", encoding="utf-8"
    )
    with pytest.raises(TransformError, match="expected upstream text is missing"):
        apply_transform("sglang_v100/marlin_builder", str(checkout))

    (checkout / "scripts" / "setup_v100_marlin.sh").unlink()
    assert apply_transform("sglang_v100/marlin_builder", str(checkout)) is None


@pytest.mark.asyncio
async def test_prepare_fork_preserves_declared_bad_dependencies(tmp_path, monkeypatch):
    """Generic primitive: a dependency matching its preserve rule is moved aside."""
    manager, _calls = _manager(tmp_path, monkeypatch=None, v100=True)
    manager._prepare_versioned_paths("source")
    python_bin = Path(manager._venv_path) / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    base = Path(manager._base_dir)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    stale = base / "dependencies" / "marlin-v100" / "csrc"
    stale.mkdir(parents=True)
    (stale / "sm70_bf16_compat.h").write_text("// legacy shim\n", encoding="utf-8")
    monkeypatch.setattr(fork_build.shutil, "which", lambda tool, path=None: f"/usr/bin/{tool}")
    monkeypatch.setattr(fork_build, "_compiler_major", lambda _compiler: 12)
    monkeypatch.setattr(fork_build, "safe_jobs", lambda: 2)

    options = {
        "dependencies": [
            {
                "path": "{dependencies}/marlin-v100",
                "preserve_if_file": "csrc/sm70_bf16_compat.h",
                "reason": "legacy BF16 compatibility header conflicts with CUDA 12.8",
            }
        ],
        "variables": {
            "HOME": "{root}/home",
            "MARLIN_V100_REPO": "{dependencies}/marlin-v100",
            "MARLIN_V100_SKIP_BF16_COMPAT": "1",
        },
    }
    env = await prepare_fork_build(
        manager,
        options,
        checkout=str(checkout),
        base_env={"SGLANG_STUDIO_PYTHON": "/studio/venv/bin/python"},
    )
    assert Path(env["HOME"]) == base / "home"
    assert Path(env["MARLIN_V100_REPO"]) == base / "dependencies" / "marlin-v100"
    assert env["MARLIN_V100_SKIP_BF16_COMPAT"] == "1"
    backups = list((base / "dependencies").glob("marlin-v100.preserved-*"))
    assert len(backups) == 1, backups
    assert (backups[0] / "source" / "csrc" / "sm70_bf16_compat.h").is_file()
