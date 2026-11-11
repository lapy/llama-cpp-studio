"""Reusable mechanism contracts: transform assets, fork builds, release assets."""

from __future__ import annotations

import os

import pytest

from backend.engines.definitions.transforms import (
    TransformError,
    apply_transform,
    load_transform,
)
from backend.engines.python.fork_build import ForkBuildError, prepare_fork_build
from backend.engines.python.release_assets import ReleaseAssetError, select_wheel_urls


def _checkout(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True)
    return tmp_path


def test_transform_asset_is_strict_about_declared_fields(tmp_path):
    asset = load_transform("sglang_v100/pyproject")
    assert asset.name == "sglang_v100/pyproject"
    assert asset.prepend_marker is True

    bad = tmp_path / "assets" / "demo" / "broken.yaml"
    bad.parent.mkdir(parents=True)
    bad.write_text(
        "schema_version: 1\nname: demo/broken\nfile: f.txt\nextra: true\noperations: []\n",
        encoding="utf-8",
    )
    with pytest.raises(TransformError, match="unknown fields"):
        load_transform("demo/broken", assets_dir=str(tmp_path / "assets"))


def test_transform_applies_only_where_the_upstream_text_matches(tmp_path):
    checkout = _checkout(tmp_path)
    (checkout / "scripts" / "install_v100.sh").write_text("echo changed\n", encoding="utf-8")
    with pytest.raises(TransformError, match="region markers are missing"):
        apply_transform("sglang_v100/install_v100_studio", str(checkout))

    # Optional assets are skipped, not failed, when the file is absent.
    assert apply_transform("sglang_v100/marlin_builder", str(checkout)) is None


def test_transform_is_idempotent_via_marker(tmp_path):
    checkout = _checkout(tmp_path)
    (checkout / "python").mkdir()
    asset = load_transform("sglang_v100/pyproject")
    (checkout / "python" / "pyproject.toml").write_text(
        "[project]\ndependencies = [\n"
        + "".join("    " + item["search"] + "\n" for item in asset.operations[0]["replacements"])
        + "]\n",
        encoding="utf-8",
    )
    apply_transform("sglang_v100/pyproject", str(checkout))
    once = (checkout / "python" / "pyproject.toml").read_text(encoding="utf-8")
    apply_transform("sglang_v100/pyproject", str(checkout))
    assert (checkout / "python" / "pyproject.toml").read_text(encoding="utf-8") == once


@pytest.mark.asyncio
async def test_prepare_fork_rejects_undeclared_options(tmp_path):
    checkout = _checkout(tmp_path)
    with pytest.raises(ForkBuildError, match="unknown options"):
        await prepare_fork_build(object(), {"magic": True}, checkout=str(checkout), base_env={})


class _ForkHost:
    """Minimal host surface prepare_fork_build reads."""

    def __init__(self, base_dir, venv_path):
        self._base_dir = str(base_dir)
        self._venv_path = str(venv_path)

    def _ensure_venv(self):
        self._venv_path = self._venv_path or "venv"

    def _venv_python(self):
        return os.path.join(self._venv_path, "bin", "python")


@pytest.mark.asyncio
async def test_prepare_fork_normalizes_an_inherited_ccache_dir(tmp_path, monkeypatch):
    """A fork build that enables ccache ITSELF must not inherit a foreign cache.

    The SGLang SM70 fork's sglang-kernel CMakeLists turns ccache on whenever the
    binary is on PATH (no Studio opt-in), and the service environment can carry a
    CCACHE_DIR owned by another user - the standalone image does exactly that, and
    every cache miss then fails with "ccache: error: Permission denied".
    """
    from backend.engines import build_workspace

    cache = tmp_path / "studio-ccache"
    monkeypatch.setattr(build_workspace, "ccache_dir", lambda: str(cache))
    monkeypatch.setattr(build_workspace.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("CCACHE_DIR", "/app/data/ccache")  # foreign, unwritable
    monkeypatch.setenv("CCACHE_MAXSIZE", "5G")

    checkout = _checkout(tmp_path)
    host = _ForkHost(tmp_path / "install", tmp_path / "install" / "venv")
    env = await prepare_fork_build(host, {}, checkout=str(checkout), base_env={})

    # The location is Studio's writable cache, not the inherited one.
    assert env["CCACHE_DIR"] == str(cache)
    assert env["CCACHE_MAXSIZE"] == "50G"
    # Normalization must not opt the engine into ccache launchers by itself.
    assert "CMAKE_CUDA_COMPILER_LAUNCHER" not in env
    assert "CMAKE_CXX_COMPILER_LAUNCHER" not in env


@pytest.mark.asyncio
async def test_prepare_fork_scales_parallelism_from_the_host(tmp_path, monkeypatch):
    """File jobs fill the host; nvcc threads only split a multi-arch compile."""
    from backend.engines.python import fork_build

    checkout = _checkout(tmp_path)
    host = _ForkHost(tmp_path / "install", tmp_path / "install" / "venv")
    monkeypatch.setattr(fork_build, "safe_jobs", lambda: 32)

    concurrency = {
        "memory_aware": True,
        "env_vars": ["MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL"],
    }

    async def build(arch_list, options=None, base=None):
        env = {"TORCH_CUDA_ARCH_LIST": arch_list} if arch_list else {}
        return await prepare_fork_build(
            host, options if options is not None else {"concurrency": concurrency},
            checkout=str(checkout), base_env={**env, **(base or {})},
        )

    single = await build("7.0")
    assert single["NVCC_THREADS"] == "1"
    assert single["MAX_JOBS"] == "32"
    assert single["CMAKE_BUILD_PARALLEL_LEVEL"] == "32"

    multi = await build("7.0;7.5;8.0")
    assert multi["NVCC_THREADS"] == "3"
    # jobs * threads stays inside the host budget.
    assert multi["MAX_JOBS"] == "10"

    explicit = await build(
        "7.0;7.5;8.0",
        options={"concurrency": {**concurrency, "nvcc_threads": 1}},
    )
    assert explicit["NVCC_THREADS"] == "1"
    assert explicit["MAX_JOBS"] == "32"

    auto = await build(
        "7.0;7.5",
        options={"concurrency": {**concurrency, "nvcc_threads": "auto"}},
    )
    assert auto["NVCC_THREADS"] == "2"

    # No concurrency block: parallelism is still host-derived, not left to the
    # fork script's own default.
    bare = await build("7.0", options={})
    assert bare["NVCC_THREADS"] == "1"
    assert "MAX_JOBS" not in bare


class _VenvBuildHost:
    """Minimal host surface ``GenericVenvInstaller._compose_build_env`` reads."""

    _venv_path = "/tmp/studio-venv"

    def __init__(self, lifecycle):
        self._lifecycle_data = lifecycle

    def _lifecycle(self):
        return self._lifecycle_data

    def _venv_python(self):
        return os.path.join(self._venv_path, "bin", "python")

    def _managed_tools_dir(self, name):
        return os.path.join("/tmp/studio-tools", name)

    def _prepend_path(self, env, *parts):
        existing = env.get("PATH", "")
        env["PATH"] = os.pathsep.join([part for part in [*parts, existing] if part])


def test_venv_build_env_derives_nvcc_threads(monkeypatch):
    """Every venv engine that reads NVCC_THREADS gets a host-derived value."""
    import backend.cuda_installer as cuda_installer
    from backend.engines.python import fork_build, generic

    class FakeCudaInstaller:
        def get_cuda_env(self, version=None):
            return {}

    monkeypatch.setattr(cuda_installer, "get_cuda_installer", lambda: FakeCudaInstaller())
    monkeypatch.setattr(fork_build, "safe_jobs", lambda: 32)
    monkeypatch.delenv("NVCC_THREADS", raising=False)

    def compose(build_env_vars):
        lifecycle = {"build_env": "cuda", "build_env_vars": build_env_vars}
        return generic.GenericVenvInstaller._compose_build_env(_VenvBuildHost(lifecycle))

    assert compose({"TORCH_CUDA_ARCH_LIST": "7.0"})["NVCC_THREADS"] == "1"
    assert compose({"TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0"})["NVCC_THREADS"] == "3"
    # A definition may still pin it, and the pin wins over the host default.
    pinned = compose({"TORCH_CUDA_ARCH_LIST": "7.0;7.5", "NVCC_THREADS": "1"})
    assert pinned["NVCC_THREADS"] == "1"


def test_release_asset_groups_are_ordered_and_require_their_wheels():
    release = {
        "tag_name": "v1",
        "assets": [
            {"name": "flash_attn_v100-1.whl", "browser_download_url": "u-flash"},
            {"name": "vllm-1.whl", "browser_download_url": "u-vllm"},
        ],
    }
    tag, urls = select_wheel_urls(
        release,
        [
            {"prefixes": ["1cat_vllm"], "required_prefixes": ["1cat_vllm"]},
            {"prefixes": ["flash_attn_v100", "vllm"], "required_prefixes": ["vllm"]},
        ],
        error="no wheels",
    )
    assert (tag, urls) == ("v1", ["u-flash", "u-vllm"])

    with pytest.raises(ReleaseAssetError, match="no wheels"):
        select_wheel_urls(
            {"tag_name": "v1", "assets": [{"name": "flash_attn_v100-1.whl"}]},
            [{"prefixes": ["flash_attn_v100", "vllm"], "required_prefixes": ["vllm"]}],
            error="no wheels",
        )


def test_definition_load_rejects_missing_transform_asset():
    from types import SimpleNamespace

    from backend.engines.definitions.loader import _validate_transform_assets
    from backend.engines.definitions.schema import DefinitionError

    definition = SimpleNamespace(
        source_path="example.yaml",
        recipes=(
            (
                "source",
                (
                    {
                        "id": "patch",
                        "uses": "vcs.apply_transform",
                        "with": {"transform": "missing/asset"},
                    },
                ),
            ),
        ),
    )
    with pytest.raises(DefinitionError, match="transform asset not found"):
        _validate_transform_assets(definition)


def test_snapshot_retry_resolution_guards_the_definition_digest():
    from backend.engines.install_snapshots import retry_resolved_inputs

    class Store:
        def list_install_snapshots(self, engine_id):
            return [
                {
                    "version": "v1",
                    "definition_digest": "old-digest",
                    "resolved_inputs": {
                        "version": "v1",
                        "source_repo": "https://example.test/frozen.git",
                        "source_ref": "deadbeef",
                    },
                }
            ]

    with pytest.raises(ValueError, match="definition changed"):
        retry_resolved_inputs(
            Store(), "llama_cpp", "v1", current_digest="new-digest", label="llama.cpp"
        )
    inputs = retry_resolved_inputs(
        Store(), "llama_cpp", "v1", current_digest="old-digest", label="llama.cpp"
    )
    assert inputs["source_ref"] == "deadbeef"
    assert inputs["source_repo"] == "https://example.test/frozen.git"


def test_snapshot_read_failure_does_not_silently_disable_the_guard():
    """A corrupt/unreadable ledger must raise, not act like a missing one.

    A genuinely missing ledger is the pre-refactor case and returns {}; a
    ledger that cannot be read would otherwise silently disable the digest
    check that stops a retry from reinterpreting the recorded layout.
    """
    from backend.engines.install_snapshots import install_snapshot_for

    class Store:
        def list_install_snapshots(self, engine_id):
            raise RuntimeError("install_snapshots.yaml is corrupt")

    with pytest.raises(RuntimeError, match="corrupt"):
        install_snapshot_for(Store(), "llama_cpp", "v1")


def test_cancel_after_publish_does_not_fail_the_ready_row(tmp_path):
    """Cancellation during activation must not rewrite a published version."""
    from backend.engines.definitions.loader import load_default_engine_definitions
    from backend.engines.python.native import GenericNativeInstaller

    definition = load_default_engine_definitions().definitions["llama_cpp"]
    failed = []

    class Host(GenericNativeInstaller):
        def _install_failed(self, version_name, error, install_dir):
            failed.append((version_name, error))

    host = Host(
        definition,
        install_root=str(tmp_path / "installs"),
        log_path=str(tmp_path / "install.log"),
    )
    host._published_versions.add("v1")

    assert host._fail_cancelled_install("v1", str(tmp_path / "v1")) is False
    assert failed == []
    assert host._fail_cancelled_install("v2", str(tmp_path / "v2")) is True
    assert failed and failed[0][0] == "v2"

    # A NEW attempt for the previously published version must not inherit the
    # old attempt's marker: cancelling it before publish marks the row failed.
    host._begin_install_attempt("v1")
    assert host._fail_cancelled_install("v1", str(tmp_path / "v1b")) is True
    assert failed[-1][0] == "v1"


@pytest.mark.asyncio
async def test_native_rebuild_refuses_a_changed_definition(monkeypatch):
    from fastapi import HTTPException

    from backend.routes import llama_versions

    class Store:
        def list_install_snapshots(self, engine_id):
            return [{"version": "v1", "definition_digest": "stale-digest"}]

    monkeypatch.setattr(llama_versions, "get_store", lambda: Store())
    with pytest.raises(HTTPException) as raised:
        await llama_versions._schedule_native_rebuild(
            {"version": "v1", "source_ref": "main", "source_repo": "https://example.test/x.git"},
            "llama_cpp",
        )
    assert raised.value.status_code == 400
    assert "definition changed" in str(raised.value.detail)
