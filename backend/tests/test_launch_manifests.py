"""Launch manifest compiler, store, launcher, and selective apply."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from backend.engines.registry import VALID_ENGINE_IDS
from backend.proxy.manifests import LaunchManifestStore, ManifestStoreError
from backend.models.config import merge_model_config_put
from backend.proxy.launch_spec import (
    PORT_PLACEHOLDER,
    LaunchCompileError,
    LaunchSpec,
    ProxyModelSpec,
    compile_model_runtime,
    compiler_engine_ids,
    launch_revision,
    manifest_document,
    project_stable_proxy_block,
    redact_env_map,
)
from backend.services.model_runtime_apply import (
    ApplyRejected,
    apply_model,
    build_apply_plan,
    reconcile_journals,
)


def test_launch_manifests_default_on(monkeypatch):
    monkeypatch.delenv("LAUNCH_MANIFESTS_ENABLED", raising=False)
    from backend.feature_flags import launch_manifests_enabled

    assert launch_manifests_enabled() is True


def test_compiler_covers_every_registered_engine():
    assert compiler_engine_ids() == set(VALID_ENGINE_IDS)


def test_empty_environment_values_survive_config_merge():
    merged = merge_model_config_put(
        {"engine": "llama_cpp", "engines": {"llama_cpp": {"swap_env": {"KEEP": "1"}}}},
        {
            "swap_env": {"EMPTY": "", "KEEP": "1"},
            "swap_env_unset": [],
            "gpu_devices": ["GPU-2", "GPU-0"],
        },
    )
    section = merged["engines"]["llama_cpp"]
    assert section["swap_env"]["EMPTY"] == ""
    assert section["swap_env"]["KEEP"] == "1"
    assert section["swap_env_unset"] == []
    assert section["gpu_devices"] == ["GPU-2", "GPU-0"]


def test_revision_preserves_argument_order_and_ignores_port_value(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("bin", encoding="utf-8")
    spec = LaunchSpec(
        model_id="model-a",
        engine_id="llama_cpp",
        engine_install_id="install",
        executable=str(binary),
        argv=["--stop", "", "--stop", "-x", "--port", dict(PORT_PLACEHOLDER)],
        cwd=str(tmp_path),
        env=__import__(
            "backend.proxy.launch_spec", fromlist=["EnvSpec"]
        ).EnvSpec(set={"A": "1"}, unset=["B"]),
        file_identities={"executable": {"path": str(binary), "present": True, "size": 3, "mtime_ns": 1}},
    )
    again = LaunchSpec(
        model_id="model-a",
        engine_id="llama_cpp",
        engine_install_id="install",
        executable=str(binary),
        argv=["--stop", "-x", "--stop", "", "--port", dict(PORT_PLACEHOLDER)],
        cwd=str(tmp_path),
        env=spec.env,
        file_identities=spec.file_identities,
    )
    assert launch_revision(spec) != launch_revision(again)
    assert "--port" in spec.argv
    assert spec.argv[-1] == {"runtime": "port"}


def test_secret_values_are_redacted():
    assert redact_env_map({"HF_TOKEN": "sekret", "FOO": "bar"}) == {
        "HF_TOKEN": "***",
        "FOO": "bar",
    }


def _install_tree(tmp_path: Path, engine: str) -> dict:
    if engine in {"llama_cpp", "ik_llama", "unsloth_llama"}:
        binary = tmp_path / engine / "llama-server"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o755)
        model = tmp_path / "model.gguf"
        model.write_text("gguf", encoding="utf-8")
        return {"binary": binary, "model": model, "cwd": binary.parent}
    if engine == "lmdeploy":
        binary = tmp_path / "lmdeploy-venv" / "bin" / "lmdeploy"
    else:
        binary = tmp_path / engine / "bin" / "python"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    toolkit = tmp_path / "cuda"
    (toolkit / "bin").mkdir(parents=True, exist_ok=True)
    return {"binary": binary, "cwd": binary.parent.parent, "toolkit": toolkit}


def _patch_resolvers(monkeypatch, tmp_path: Path):
    trees = {engine: _install_tree(tmp_path, engine) for engine in VALID_ENGINE_IDS}

    class Store:
        def get_active_engine_version(self, engine):
            tree = trees[engine]
            return {
                "version": f"{engine}-1",
                "source_commit": "abc123",
                "binary_path": str(tree["binary"]),
                "venv_path": str(tree["binary"].parents[1]),
                "server_binary_path": str(tree["binary"]),
                "build_config": {"backend": "cpu"},
            }

    monkeypatch.setattr("backend.data_store.get_store", lambda: Store())
    monkeypatch.setattr(
        "backend.engines.llama_cpp.resolve.get_active_binary_path_for_engine",
        lambda store, engine: str(trees[engine]["binary"]),
    )
    monkeypatch.setattr(
        "backend.proxy.llama_swap.config.get_active_binary_path_for_engine",
        lambda store, engine: str(trees[engine]["binary"]),
    )
    monkeypatch.setattr(
        "backend.proxy.llama_swap.config.resolve_llama_server_invocation_paths",
        lambda path: (str(path), str(Path(path).parent)),
    )
    monkeypatch.setattr(
        "backend.engines.llama_cpp.server_exec.resolve_llama_server_invocation_paths",
        lambda path: (str(path), str(Path(path).parent)),
    )
    monkeypatch.setattr(
        "backend.proxy.llama_swap.config._resolve_cuda_library_path",
        lambda build_dir: "/opt/engine/lib",
    )
    monkeypatch.setattr(
        "backend.proxy.llama_swap.config._active_engine_param_index",
        lambda engine: {
            "ctx_size": {"primary_flag": "--ctx-size", "value_kind": "scalar"},
            "stop": {"primary_flag": "--stop", "value_kind": "repeatable"},
        },
    )
    monkeypatch.setattr(
        "backend.proxy.llama_swap.config._resolve_lmdeploy_bin",
        lambda: str(trees["lmdeploy"]["binary"]),
    )
    monkeypatch.setattr(
        "backend.proxy.llama_swap.config._resolve_onecat_vllm_bin",
        lambda: str(trees["1cat_vllm"]["binary"]),
    )
    monkeypatch.setattr(
        "backend.proxy.llama_swap.config._resolve_sglang_bin",
        lambda engine: str(trees[engine]["binary"]),
    )
    monkeypatch.setattr(
        "backend.proxy.llama_swap.config._resolve_sglang_cuda_env",
        lambda engine: {
            "CUDA_HOME": str(trees["sglang_v100"]["toolkit"]),
            "PATH": str(trees["sglang_v100"]["toolkit"] / "bin"),
            "LD_LIBRARY_PATH": "/opt/cuda/lib64",
        },
    )
    monkeypatch.setattr(
        "backend.proxy.llama_swap.config.resolve_gguf_model_path",
        lambda model: str(trees["llama_cpp"]["model"]),
    )
    monkeypatch.setattr(
        "backend.models.hub.resolve_gguf_model_path",
        lambda model: str(trees["llama_cpp"]["model"]),
    )

    def audio_runtime(store, model, config, stable_id):
        binary = trees["audio_cpp"]["binary"]
        return {
            "cmd_argv": [str(binary), "--config", "${studio_audio_config}", "--port", "${PORT}"],
            "cmd_cwd": str(binary.parent),
            "env": ["FOO=bar"],
            "macros": {"studio_audio_config": "/outside/server.json"},
            "sidecar": {"models": [{"id": stable_id}]},
            "use_model_name": stable_id,
        }

    monkeypatch.setattr(
        "backend.engines.audio_cpp.runtime.build_audio_cpp_runtime",
        audio_runtime,
    )
    monkeypatch.setattr(
        "backend.services.model_metadata.get_startup_gpu_list",
        lambda: {
            "vendor": "nvidia",
            "cpu_only_mode": False,
            "gpus": [
                {"index": 0, "uuid": "GPU-0", "name": "A"},
                {"index": 1, "uuid": "GPU-2", "name": "B"},
            ],
        },
    )
    return trees


def _model(engine: str, **section):
    return {
        "id": f"catalog-{engine}",
        "proxy_name": f"model-{engine}",
        "huggingface_id": f"org/{engine}",
        "quantization": "Q4_K_M",
        "files": ["model.gguf"],
        "config": {"engine": engine, "engines": {engine: section}},
    }


@pytest.mark.parametrize("engine", sorted(VALID_ENGINE_IDS))
def test_each_engine_compiles_structured_argv(monkeypatch, tmp_path, engine):
    _patch_resolvers(monkeypatch, tmp_path)
    section = {"ctx_size": 2048, "swap_env": {"FOO": "bar baz", "EMPTY": ""}}
    if engine == "sglang_v100":
        section["swap_env"] = {"FOO": "bar baz"}
    compiled = compile_model_runtime(_model(engine, **section))
    assert compiled.launch.engine_id == engine
    assert compiled.launch.executable
    assert PORT_PLACEHOLDER in compiled.launch.argv
    assert compiled.launch.env.set["FOO"] == "bar baz"
    assert "$HOME" not in compiled.engine_command
    if engine == "audio_cpp":
        assert {"artifact": "server.json"} in compiled.launch.argv
        assert compiled.launch.artifacts["server.json"]["models"][0]["id"] == f"model-{engine}"
    if engine in {"llama_cpp", "ik_llama", "unsloth_llama"}:
        assert compiled.launch.argv[:2] == ["--model", str(tmp_path / "model.gguf")]
        assert "--ctx-size" in compiled.launch.argv


def test_environment_unset_empty_and_shell_literals(monkeypatch, tmp_path):
    _patch_resolvers(monkeypatch, tmp_path)
    monkeypatch.setenv("DEMO_INHERITED", "from-parent")
    compiled = compile_model_runtime(
        _model(
            "llama_cpp",
            swap_env={"EMPTY": "", "QUOTED": "a b $HOME 'x';"},
            swap_env_unset=["DEMO_INHERITED"],
        )
    )
    assert compiled.launch.env.set["EMPTY"] == ""
    assert compiled.launch.env.set["QUOTED"] == "a b $HOME 'x';"
    assert "DEMO_INHERITED" not in compiled.launch.env.set
    assert "DEMO_INHERITED" in compiled.launch.env.unset


def test_conflicting_set_and_unset_is_rejected(monkeypatch, tmp_path):
    _patch_resolvers(monkeypatch, tmp_path)
    with pytest.raises(LaunchCompileError, match="both set and unset"):
        compile_model_runtime(
            _model(
                "llama_cpp",
                swap_env={"FOO": "1"},
                swap_env_unset=["FOO"],
            )
        )


def test_inherit_keeps_user_env_and_does_not_pin_every_gpu(monkeypatch, tmp_path):
    _patch_resolvers(monkeypatch, tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    compiled = compile_model_runtime(
        _model(
            "llama_cpp",
            gpu_mode="inherit",
            swap_env={"FOO": "bar", "EMPTY": ""},
            swap_env_unset=["REMOVED_VAR"],
        )
    )
    assert compiled.launch.env.set["FOO"] == "bar"
    assert compiled.launch.env.set["EMPTY"] == ""
    assert "REMOVED_VAR" not in compiled.launch.env.set
    assert "REMOVED_VAR" in compiled.launch.env.unset
    assert "CUDA_VISIBLE_DEVICES" not in compiled.launch.env.set

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    narrowed = compile_model_runtime(
        _model("llama_cpp", gpu_mode="inherit", swap_env={"FOO": "kept"})
    )
    assert narrowed.launch.env.set["FOO"] == "kept"
    assert narrowed.launch.env.set["CUDA_VISIBLE_DEVICES"] == "0"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "all")
    unrestricted = compile_model_runtime(
        _model("llama_cpp", gpu_mode="inherit", swap_env={"FOO": "kept"})
    )
    assert unrestricted.launch.env.set["FOO"] == "kept"
    assert "CUDA_VISIBLE_DEVICES" not in unrestricted.launch.env.set


def test_gpu_order_is_preserved_and_cpu_mode_is_rejected_for_vllm(monkeypatch, tmp_path):
    _patch_resolvers(monkeypatch, tmp_path)
    compiled = compile_model_runtime(
        _model(
            "llama_cpp",
            gpu_mode="selected",
            gpu_devices=["GPU-2", "GPU-0"],
        )
    )
    assert compiled.launch.env.set["CUDA_VISIBLE_DEVICES"] == "GPU-2,GPU-0"
    with pytest.raises(LaunchCompileError, match="does not support CPU"):
        compile_model_runtime(_model("vllm", gpu_mode="cpu"))
    with pytest.raises(LaunchCompileError, match="not visible"):
        compile_model_runtime(
            _model("llama_cpp", gpu_mode="selected", gpu_devices=["GPU-missing"])
        )


def test_repeated_flags_and_owned_port_flag(monkeypatch, tmp_path):
    _patch_resolvers(monkeypatch, tmp_path)
    compiled = compile_model_runtime(
        _model("llama_cpp", stop=["END", "END"])
    )
    argv = compiled.launch.argv
    assert argv.count("--stop") == 2
    with pytest.raises(LaunchCompileError, match="Studio-owned"):
        compile_model_runtime(_model("llama_cpp", custom_args="--port 1"))


def test_store_is_atomic_and_retains_rollback_generation(tmp_path):
    store = LaunchManifestStore(str(tmp_path / "models"))
    first = _document("model-a", "one")
    second = _document("model-a", "two")
    third = _document("model-a", "three")
    store.stage("model-a", first)
    store.publish("model-a", first["revision"])
    store.stage("model-a", second)
    pointer = store.publish("model-a", second["revision"])
    assert pointer.previous_revision == first["revision"]
    store.stage("model-a", third)
    removed = store.garbage_collect("model-a")
    assert third["revision"] in removed
    assert (tmp_path / "models").exists()
    assert store.read_manifest("model-a", first["revision"])["executable"] == "/bin/one"
    with pytest.raises(ManifestStoreError):
        store.stage("model-a", {**second, "executable": "/bin/other"})
    with pytest.raises(ManifestStoreError):
        store.read_manifest("model-a", "../etc/passwd")


def test_deletion_of_referenced_engine_is_refused(tmp_path, monkeypatch):
    binary = tmp_path / "install" / "llama-server"
    binary.parent.mkdir()
    binary.write_text("bin", encoding="utf-8")
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "data"))
    store = LaunchManifestStore()
    document = _document("model-a", "bin", executable=str(binary))
    store.stage("model-a", document)
    store.publish("model-a", document["revision"])
    from backend.utils.fs_ops import robust_rmtree

    with pytest.raises(PermissionError, match="referenced"):
        robust_rmtree(str(binary.parent))


def test_launcher_execs_engine_without_parent_environment(tmp_path):
    engine = tmp_path / "engine.py"
    outfile = tmp_path / "seen.json"
    engine.write_text(
        textwrap.dedent(
            """
            import json, os, sys
            json.dump({"argv": sys.argv, "cwd": os.getcwd(), "env": dict(os.environ)}, open(sys.argv[-1], "w"))
            """
        ),
        encoding="utf-8",
    )
    store = LaunchManifestStore(str(tmp_path / "models"))
    document = _document(
        "model-a",
        "engine",
        executable=sys.executable,
        argv=[str(engine), "--port", {"runtime": "port"}, str(outfile)],
        env={"set": {"FOO": "bar baz", "EMPTY": ""}, "unset": ["CUDA_VISIBLE_DEVICES"]},
        cwd=str(tmp_path),
    )
    store.stage("model-a", document)
    store.publish("model-a", document["revision"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "parent"
    env["FOO"] = "parent"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "proxy" / "launcher.py"),
            "--manifest-root",
            store.model_dir("model-a"),
            "--port",
            "4321",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    seen = json.loads(outfile.read_text(encoding="utf-8"))
    assert seen["argv"][-3:] == ["--port", "4321", str(outfile)]
    assert seen["env"].get("FOO") == "bar baz"
    assert seen["env"].get("EMPTY") == ""
    assert "CUDA_VISIBLE_DEVICES" not in seen["env"]
    assert seen["cwd"] == str(tmp_path)
    receipts = store.read_receipts("model-a")
    assert receipts and receipts[0]["revision"] == document["revision"]


def test_launcher_rejects_malformed_pointer_and_lock_timeout(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "active.json").write_text("{", encoding="utf-8")
    launcher = str(Path(__file__).resolve().parents[1] / "proxy" / "launcher.py")
    bad = subprocess.run(
        [sys.executable, launcher, "--manifest-root", str(root), "--port", "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 3
    store = LaunchManifestStore(str(tmp_path / "models"))
    document = _document("model-a", "x", executable=sys.executable, argv=["--port", {"runtime": "port"}])
    store.stage("model-a", document)
    store.publish("model-a", document["revision"])
    fd = store.acquire_gate("model-a", exclusive=True)
    try:
        blocked = subprocess.run(
            [
                sys.executable,
                launcher,
                "--manifest-root",
                store.model_dir("model-a"),
                "--port",
                "1",
                "--gate-timeout",
                "0.1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        store.release_gate(fd)
    assert blocked.returncode == 4


def test_stable_proxy_block_ignores_launch_env(monkeypatch):
    monkeypatch.setenv("LAUNCH_MANIFESTS_ENABLED", "1")
    from backend.proxy.llama_swap.config import _llama_swap_yaml_model_block_for_config

    model = {"id": "catalog", "proxy_name": "model-a"}
    first = _llama_swap_yaml_model_block_for_config(
        cmd="legacy-a",
        env_list=["CUDA_VISIBLE_DEVICES=0"],
        model_id="model-a",
        config={"engine": "llama_cpp"},
        model=model,
    )
    second = _llama_swap_yaml_model_block_for_config(
        cmd="legacy-b",
        env_list=["CUDA_VISIBLE_DEVICES=1"],
        model_id="model-a",
        config={"engine": "llama_cpp"},
        model=model,
    )
    assert first == second
    assert "env" not in first
    assert first["capabilities"] == {"disableAuto": True}
    assert "launcher.py" in first["cmd"]
    assert "${PORT}" in first["cmd"]


def test_preflight_allows_yaml_routes_removed_from_the_catalog(monkeypatch):
    from backend.services.model_runtime_apply import preflight_deployment

    kept = _compiled("kept")
    monkeypatch.setattr(
        "backend.services.model_runtime_apply.compile_model_runtime",
        lambda model: kept,
    )
    disk = """
models:
  kept:
    cmd: old
    proxy: http://127.0.0.1:${PORT}
  removed-model:
    cmd: old
    proxy: http://127.0.0.1:${PORT}
"""
    compiled = preflight_deployment([{"id": "catalog", "proxy_name": "kept"}], disk)
    assert [item.proxy.model_id for item in compiled] == ["kept"]


def test_preflight_rejects_a_published_catalog_model_that_cannot_compile(monkeypatch):
    from backend.services.model_runtime_apply import PreflightError, preflight_deployment

    def boom(_model):
        raise LaunchCompileError("llama-server binary not found")

    monkeypatch.setattr(
        "backend.services.model_runtime_apply.compile_model_runtime",
        boom,
    )
    disk = """
models:
  kept:
    cmd: old
    proxy: http://127.0.0.1:${PORT}
"""
    with pytest.raises(PreflightError, match="llama-server binary not found") as exc:
        preflight_deployment([{"id": "catalog", "proxy_name": "kept"}], disk)
    assert exc.value.status == 409
    assert exc.value.detail["failures"] == ["kept: llama-server binary not found"]


def test_preflight_ignores_catalog_models_that_are_not_published(monkeypatch):
    from backend.services.model_runtime_apply import preflight_deployment

    def boom(_model):
        raise LaunchCompileError("no binary")

    monkeypatch.setattr(
        "backend.services.model_runtime_apply.compile_model_runtime",
        boom,
    )
    assert preflight_deployment([{"proxy_name": "never-deployed"}], "models: {}\n") == []


def test_plan_classifies_proxy_edits_as_global_and_launch_edits_as_selective(tmp_path, monkeypatch):
    monkeypatch.setenv("LAUNCH_MANIFESTS_ENABLED", "1")
    store = LaunchManifestStore(str(tmp_path / "models"))
    compiled = _compiled("model-a")
    monkeypatch.setattr(
        "backend.services.model_runtime_apply.compile_model_runtime",
        lambda model: compiled,
    )
    disk = _proxy_yaml(compiled)
    plan = build_apply_plan(
        [_catalog_model()],
        disk_yaml=disk,
        states={"model-a": "running"},
        store=store,
    )
    assert plan["models"][0]["action"] == "restart_now"
    assert plan["requires_proxy_reload"] is False
    alias_disk = disk.replace("proxy: http://127.0.0.1:${PORT}", "proxy: http://127.0.0.1:${PORT}\n    aliases:\n      - other")
    global_plan = build_apply_plan(
        [_catalog_model()],
        disk_yaml=alias_disk,
        states={"model-a": "running"},
        store=store,
    )
    assert global_plan["requires_proxy_reload"] is True
    assert global_plan["models"][0]["action"] == "global_proxy"


def test_selective_apply_restarts_only_the_selected_model(tmp_path, monkeypatch):
    store = LaunchManifestStore(str(tmp_path / "models"))
    old = _compiled("model-a", marker="old")
    new = _compiled("model-a", marker="new")
    store.stage(old.proxy.model_id, manifest_document(old.launch, old.revision))
    store.publish(old.proxy.model_id, old.revision)
    store.write_receipt(
        "model-a",
        revision=old.revision,
        launch_id="live",
        pid=os.getpid(),
        start_ticks=_ticks(os.getpid()),
    )
    monkeypatch.setattr(
        "backend.services.model_runtime_apply.compile_model_runtime",
        lambda model: new,
    )
    gateway = _Gateway(store, ready_revision=new.revision, state="running")
    result = asyncio.run(
        apply_model(
            _catalog_model(),
            mode="restart_now",
            expected_desired_revision=new.revision,
            expected_published_revision=old.revision,
            check_published_revision=True,
            idempotency_key="apply-1",
            gateway=gateway,
            disk_yaml=_proxy_yaml(new),
            store=store,
        )
    )
    assert result["status"] == "succeeded"
    assert gateway.unloads == ["model-a"]
    assert gateway.loads == ["model-a"]
    assert not hasattr(gateway, "unload_all") or gateway.unload_all == []
    assert store.read_pointer("model-a").revision == new.revision
    again = asyncio.run(
        apply_model(
            _catalog_model(),
            mode="restart_now",
            expected_desired_revision=new.revision,
            expected_published_revision=old.revision,
            check_published_revision=True,
            idempotency_key="apply-1",
            gateway=gateway,
            disk_yaml=_proxy_yaml(new),
            store=store,
        )
    )
    assert again["operation_id"] == result["operation_id"]
    assert gateway.unloads == ["model-a"]


def test_stopped_model_is_published_without_start(tmp_path, monkeypatch):
    store = LaunchManifestStore(str(tmp_path / "models"))
    compiled = _compiled("model-a")
    monkeypatch.setattr(
        "backend.services.model_runtime_apply.compile_model_runtime",
        lambda model: compiled,
    )
    gateway = _Gateway(store, ready_revision=compiled.revision)
    result = asyncio.run(
        apply_model(
            _catalog_model(),
            mode="restart_now",
            expected_desired_revision=compiled.revision,
            expected_published_revision=None,
            idempotency_key="stopped-1",
            gateway=gateway,
            disk_yaml=_proxy_yaml(compiled),
            store=store,
        )
    )
    assert result["status"] == "succeeded"
    assert "left stopped" in result["message"]
    assert gateway.loads == []
    assert gateway.unloads == []


def test_stale_plan_does_not_restart(tmp_path, monkeypatch):
    store = LaunchManifestStore(str(tmp_path / "models"))
    compiled = _compiled("model-a")
    monkeypatch.setattr(
        "backend.services.model_runtime_apply.compile_model_runtime",
        lambda model: compiled,
    )
    gateway = _Gateway(store, ready_revision=compiled.revision)
    with pytest.raises(ApplyRejected) as caught:
        asyncio.run(
            apply_model(
                _catalog_model(),
                mode="restart_now",
                expected_desired_revision="0" * 64,
                expected_published_revision=None,
                idempotency_key="stale",
                gateway=gateway,
                disk_yaml=_proxy_yaml(compiled),
                store=store,
            )
        )
    assert caught.value.detail["error"] == "revision_conflict"
    assert gateway.unloads == []


def test_failed_readiness_restores_verified_revision(tmp_path, monkeypatch):
    store = LaunchManifestStore(str(tmp_path / "models"))
    old = _compiled("model-a", marker="old")
    new = _compiled("model-a", marker="new")
    store.stage(old.proxy.model_id, manifest_document(old.launch, old.revision))
    store.publish(old.proxy.model_id, old.revision)
    store.write_receipt(
        "model-a",
        revision=old.revision,
        launch_id="live",
        pid=os.getpid(),
        start_ticks=_ticks(os.getpid()),
    )
    monkeypatch.setattr(
        "backend.services.model_runtime_apply.compile_model_runtime",
        lambda model: new,
    )
    gateway = _Gateway(store, ready_revision=None, state="running")
    gateway.startup_timeout = 0.05
    result = asyncio.run(
        apply_model(
            _catalog_model(),
            mode="restart_now",
            expected_desired_revision=new.revision,
            expected_published_revision=old.revision,
            check_published_revision=True,
            idempotency_key="rollback",
            gateway=gateway,
            disk_yaml=_proxy_yaml(new),
            store=store,
        )
    )
    assert result["status"] == "failed"
    assert result["rollback"]["published_revision"] == old.revision
    assert store.read_pointer("model-a").revision == old.revision
    assert gateway.unloads == ["model-a", "model-a"]


def test_interrupted_journal_is_not_replayed(tmp_path):
    store = LaunchManifestStore(str(tmp_path / "models"))
    ops = tmp_path / "operations"
    ops.mkdir()
    (ops / f"{'a' * 32}.json").write_text(
        json.dumps(
            {
                "operation_id": "a" * 32,
                "model_id": "model-a",
                "phase": "stopping",
                "desired_revision": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    assert reconcile_journals(store) == 1
    journal = json.loads((ops / f"{'a' * 32}.json").read_text(encoding="utf-8"))
    assert journal["phase"] == "interrupted"
    assert "not replayed" in journal["message"]


class _Gateway:
    def __init__(self, store, ready_revision, state="stopped"):
        self.store = store
        self.ready_revision = ready_revision
        self._state = state
        self.unloads = []
        self.loads = []
        self.unload_all = []
        self.startup_timeout = 1.0

    async def state(self, model_id):
        return self._state

    async def unload(self, model_id):
        self.unloads.append(model_id)

    async def load(self, model_id):
        self.loads.append(model_id)
        if self.ready_revision:
            self.store.write_receipt(
                model_id,
                revision=self.ready_revision,
                launch_id=f"load-{len(self.loads)}",
                pid=os.getpid(),
                start_ticks=_ticks(os.getpid()),
            )


def _ticks(pid: int) -> str:
    with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
        stat = handle.read()
    fields = stat[stat.rfind(")") + 2 :].split()
    return fields[19]


def _document(model_id, marker, **overrides):
    from backend.proxy.launch_spec import EnvSpec

    raw_env = overrides.get("env") or {}
    spec = LaunchSpec(
        model_id=model_id,
        engine_id="llama_cpp",
        engine_install_id=marker,
        executable=overrides.get("executable", f"/bin/{marker}"),
        argv=overrides.get("argv", ["--model", f"/models/{marker}.gguf", "--port", dict(PORT_PLACEHOLDER)]),
        cwd=overrides.get("cwd"),
        env=EnvSpec(
            set=dict(raw_env.get("set") or {"FOO": marker}),
            unset=list(raw_env.get("unset") or []),
        ),
    )
    return manifest_document(spec, launch_revision(spec))


def _compiled(model_id: str, marker: str = "marker"):
    from backend.proxy.launch_spec import CompiledModel, EnvSpec

    launch = LaunchSpec(
        model_id=model_id,
        engine_id="llama_cpp",
        engine_install_id=marker,
        executable=f"/bin/{marker}",
        argv=["--model", f"/models/{marker}.gguf", "--port", dict(PORT_PLACEHOLDER)],
        cwd=None,
        env=EnvSpec(set={"FOO": marker}, unset=[]),
    )
    proxy = ProxyModelSpec(
        model_id=model_id,
        catalog_id="catalog-a",
        engine_id="llama_cpp",
        aliases=[],
        filters=None,
        use_model_name=None,
    )
    revision = launch_revision(launch)
    block = project_stable_proxy_block({"cmd": "ignored"}, model_id)
    return CompiledModel(
        launch=launch,
        proxy=proxy,
        revision=revision,
        engine_command="engine",
        launcher_command=block["cmd"],
    )


def _catalog_model():
    return {
        "id": "catalog-a",
        "proxy_name": "model-a",
        "huggingface_id": "org/model",
        "config": {"engine": "llama_cpp", "engines": {"llama_cpp": {}}},
    }


def _proxy_yaml(compiled) -> str:
    block = project_stable_proxy_block({"cmd": "ignored"}, compiled.proxy.model_id)
    lines = [
        "models:",
        f"  {compiled.proxy.model_id}:",
        f"    cmd: {json.dumps(block['cmd'])}",
        "    proxy: http://127.0.0.1:${PORT}",
    ]
    capabilities = block.get("capabilities") or {}
    if capabilities:
        lines.append("    capabilities:")
        for key, value in capabilities.items():
            if value is True:
                rendered = "true"
            elif value is False:
                rendered = "false"
            else:
                rendered = json.dumps(value)
            lines.append(f"      {key}: {rendered}")
    return "\n".join(lines) + "\n"
