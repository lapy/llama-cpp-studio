"""Integration tests: every task-start surface returns a real ProgressManager task_id.

Surfaces covered:
  Builds:     POST /api/llama-versions/build-source (llama.cpp, ik_llama.cpp)
              POST /api/llama-versions/versions/sync (llama.cpp, lmdeploy, 1cat-vllm)
  Installs:   POST /api/lmdeploy/install, /install-source, /remove
              POST /api/1cat-vllm/install, /install-source, /remove
              POST /api/llama-versions/cuda-install, /cuda-uninstall
  Downloads:  POST /api/models/download (gguf, safetensors)
              POST /api/models/safetensors/download-bundle
              POST /api/models/gguf/download-bundle
              POST /api/models/{id}/projector (when download needed)
  Contract:   legacy SSE events reuse the same task_id; unified /api/tasks/cancel removed
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Optional

import pytest
from fastapi import BackgroundTasks

import backend.progress_manager as pm_mod
import backend.routes.models as models_routes
import backend.services.model_downloads as model_downloads
from backend.cuda_installer import CUDAInstaller
from backend.lmdeploy_manager import LMDeployManager
from backend.onecat_vllm_manager import OneCatVllmManager

# --- Task ID contract (kept inline — only used by these tests) ----------------

FORBIDDEN_SYNTHETIC_TASK_IDS = frozenset(
    {
        "cuda_operation",
        "lmdeploy_operation",
        "onecat_vllm_operation",
    }
)
FORBIDDEN_SYNTHETIC_PATTERN = re.compile(
    r"^(cuda|lmdeploy|onecat_vllm|download|build)_operation$"
)
INSTALL_TASK_ID = re.compile(r"^install_(cuda|lmdeploy|onecat_vllm)_[a-z0-9_]+_\d+$")
BUILD_TASK_ID = re.compile(r"^build(?:_sync)?_[a-zA-Z0-9_.-]+_\d+$")
DOWNLOAD_TASK_ID = re.compile(r"^download_[a-zA-Z0-9_.-]+_\d+$")


def assert_not_synthetic(task_id: str) -> None:
    assert task_id, "task_id must be non-empty"
    assert task_id not in FORBIDDEN_SYNTHETIC_TASK_IDS
    assert not FORBIDDEN_SYNTHETIC_PATTERN.match(task_id)


def assert_install_task_id(task_id: str, manager: str) -> None:
    assert_not_synthetic(task_id)
    match = INSTALL_TASK_ID.match(task_id)
    assert match, f"install task_id has unexpected shape: {task_id!r}"
    assert match.group(1) == manager


def assert_build_task_id(task_id: str) -> None:
    assert_not_synthetic(task_id)
    assert BUILD_TASK_ID.match(task_id), f"build task_id has unexpected shape: {task_id!r}"


def assert_download_task_id(task_id: str) -> None:
    assert_not_synthetic(task_id)
    assert DOWNLOAD_TASK_ID.match(task_id), f"download task_id has unexpected shape: {task_id!r}"


def assert_task_in_progress_manager(
    pm: Any,
    task_id: str,
    *,
    expected_type: str,
    expected_manager: Optional[str] = None,
) -> dict:
    assert_not_synthetic(task_id)
    task = pm.get_task(task_id)
    assert task is not None, f"task {task_id!r} missing from ProgressManager"
    assert task["task_id"] == task_id
    assert task["type"] == expected_type
    if expected_manager is not None:
        assert (task.get("metadata") or {}).get("manager") == expected_manager
    return task


# --- Fixtures -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_progress_manager():
    pm_mod._progress_manager = pm_mod.ProgressManager()
    yield pm_mod.get_progress_manager()
    pm_mod._progress_manager = None


@pytest.fixture(autouse=True)
def reset_manager_singletons():
    import backend.cuda_installer as cuda_mod
    import backend.lmdeploy_manager as lm_mod
    import backend.onecat_vllm_manager as oc_mod

    cuda_mod._installer_instance = None
    lm_mod._manager_instance = None
    oc_mod._manager_instance = None
    yield
    cuda_mod._installer_instance = None
    lm_mod._manager_instance = None
    oc_mod._manager_instance = None


@pytest.fixture(autouse=True)
def clear_cancel_registry():
    from backend import task_cancel_registry as reg

    reg._events.clear()
    yield
    reg._events.clear()


def _install_temp_store(monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    return store


def _prevent_background_task(manager):
    """Schedule a hung asyncio task so cancel endpoints can succeed in tests."""

    def _patched(coro):
        coro.close()

        async def _hang():
            await asyncio.Event().wait()

        loop = asyncio.get_running_loop()
        task = loop.create_task(_hang())
        manager._current_task = task

    manager._create_task = _patched  # type: ignore[method-assign]
    return manager


def _patch_cuda_installer(monkeypatch, tmp_path, manager):
    import backend.cuda_installer as cuda_mod
    from backend.routes import llama_versions as llama_routes

    monkeypatch.setattr(cuda_mod, "get_cuda_installer", lambda: manager)
    monkeypatch.setattr(llama_routes, "get_cuda_installer", lambda: manager)


def _assert_started_task(body: dict, *, task_type: str, manager: Optional[str] = None) -> str:
    task_id = body["task_id"]
    if task_type == "install":
        assert_install_task_id(task_id, manager)
    elif task_type == "build":
        assert_build_task_id(task_id)
    elif task_type == "download":
        assert_download_task_id(task_id)
    assert_task_in_progress_manager(
        pm_mod.get_progress_manager(),
        task_id,
        expected_type=task_type,
        expected_manager=manager,
    )
    return task_id


# --- Build tasks --------------------------------------------------------------


@pytest.mark.parametrize(
    "repository_source",
    ["llama.cpp", "ik_llama.cpp"],
)
def test_build_source_returns_registered_task_id(
    client, monkeypatch, tmp_path, repository_source
):
    _install_temp_store(monkeypatch, tmp_path)

    from backend.routes import llama_versions as llama_routes

    monkeypatch.setattr(llama_routes.asyncio, "create_task", lambda coro: coro.close())
    monkeypatch.setattr(llama_routes.time, "time", lambda: 1_700_000_000)

    r = client.post(
        "/api/llama-versions/build-source",
        json={"commit_sha": "main", "repository_source": repository_source},
    )
    assert r.status_code == 200
    _assert_started_task(r.json(), task_type="build")


def test_sync_llama_branch_returns_registered_task_id(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.add_engine_version(
        "llama_cpp",
        {
            "version": "source-main",
            "type": "source",
            "binary_path": str(tmp_path / "llama-server"),
            "source_ref": "main",
            "source_ref_type": "branch",
            "source_repo": "https://example.test/llama.cpp.git",
            "build_config": {"enable_cuda": False, "build_type": "Release"},
            "repository_source": "llama.cpp",
        },
    )

    from backend.routes import llama_versions as llama_routes

    monkeypatch.setattr(llama_routes.asyncio, "create_task", lambda coro: coro.close())
    monkeypatch.setattr(llama_routes.time, "time", lambda: 1_700_000_000)

    r = client.post(
        "/api/llama-versions/versions/sync",
        json={"version_id": "llama_cpp:source-main"},
    )
    assert r.status_code == 200
    task_id = _assert_started_task(r.json(), task_type="build")
    assert task_id.startswith("build_sync_")


@pytest.mark.parametrize(
    "engine,version_id,manager_name,route_module,getter_name",
    [
        (
            "lmdeploy",
            "lmdeploy:source-main",
            "lmdeploy",
            "lmdeploy_versions",
            "get_lmdeploy_manager",
        ),
        (
            "1cat_vllm",
            "1cat_vllm:source-main",
            "onecat_vllm",
            "onecat_vllm_versions",
            "get_onecat_vllm_manager",
        ),
    ],
)
def test_sync_engine_branch_returns_registered_task_id(
    client,
    monkeypatch,
    tmp_path,
    engine,
    version_id,
    manager_name,
    route_module,
    getter_name,
):
    store = _install_temp_store(monkeypatch, tmp_path)
    venv_path = str(tmp_path / engine / "venv")
    store.add_engine_version(
        engine,
        {
            "version": "source-main",
            "type": "source",
            "install_type": "source",
            "source_ref": "main",
            "source_ref_type": "branch",
            "source_branch": "main",
            "source_repo": f"https://example.test/{engine}.git",
            "venv_path": venv_path,
        },
    )

    if manager_name == "lmdeploy":
        manager = _prevent_background_task(
            LMDeployManager(
                log_path=str(tmp_path / f"{engine}.log"),
                state_path=str(tmp_path / f"{engine}_state.json"),
                base_dir=str(tmp_path / engine),
            )
        )
    else:
        manager = _prevent_background_task(
            OneCatVllmManager(
                log_path=str(tmp_path / f"{engine}.log"),
                state_path=str(tmp_path / f"{engine}_state.json"),
                base_dir=str(tmp_path / engine),
            )
        )

    routes = __import__(f"backend.routes.{route_module}", fromlist=[getter_name])
    monkeypatch.setattr(routes, getter_name, lambda: manager)

    r = client.post(
        "/api/llama-versions/versions/sync",
        json={"version_id": version_id},
    )
    assert r.status_code == 200
    _assert_started_task(r.json(), task_type="install", manager=manager_name)


# --- Install manager tasks (API + direct cancel round-trip) -------------------


@pytest.mark.parametrize(
    "route,manager_name,manager_cls",
    [
        ("/api/lmdeploy/install", "lmdeploy", LMDeployManager),
        ("/api/1cat-vllm/install", "onecat_vllm", OneCatVllmManager),
    ],
)
def test_install_release_api_returns_real_task_id(
    client, monkeypatch, tmp_path, route, manager_name, manager_cls
):
    routes = __import__(
        f"backend.routes.{'lmdeploy' if manager_name == 'lmdeploy' else 'onecat_vllm'}_versions",
        fromlist=["get_lmdeploy_manager", "get_onecat_vllm_manager"],
    )
    getter = (
        routes.get_lmdeploy_manager
        if manager_name == "lmdeploy"
        else routes.get_onecat_vllm_manager
    )
    manager = _prevent_background_task(
        manager_cls(
            log_path=str(tmp_path / f"{manager_name}.log"),
            state_path=str(tmp_path / f"{manager_name}_state.json"),
            base_dir=str(tmp_path / manager_name),
        )
    )
    monkeypatch.setattr(routes, getter.__name__, lambda: manager)

    r = client.post(route, json={})
    assert r.status_code == 200
    _assert_started_task(r.json(), task_type="install", manager=manager_name)


@pytest.mark.parametrize(
    "route,manager_name,manager_cls",
    [
        ("/api/lmdeploy/install-source", "lmdeploy", LMDeployManager),
        ("/api/1cat-vllm/install-source", "onecat_vllm", OneCatVllmManager),
    ],
)
def test_install_from_source_api_returns_real_task_id(
    client, monkeypatch, tmp_path, route, manager_name, manager_cls
):
    routes = __import__(
        f"backend.routes.{'lmdeploy' if manager_name == 'lmdeploy' else 'onecat_vllm'}_versions",
        fromlist=["get_lmdeploy_manager", "get_onecat_vllm_manager"],
    )
    getter = (
        routes.get_lmdeploy_manager
        if manager_name == "lmdeploy"
        else routes.get_onecat_vllm_manager
    )
    manager = _prevent_background_task(
        manager_cls(
            log_path=str(tmp_path / f"{manager_name}.log"),
            state_path=str(tmp_path / f"{manager_name}_state.json"),
            base_dir=str(tmp_path / manager_name),
        )
    )
    monkeypatch.setattr(routes, getter.__name__, lambda: manager)

    r = client.post(route, json={"branch": "main"})
    assert r.status_code == 200
    _assert_started_task(r.json(), task_type="install", manager=manager_name)


@pytest.mark.parametrize(
    "route,manager_name,manager_cls",
    [
        ("/api/lmdeploy/remove", "lmdeploy", LMDeployManager),
        ("/api/1cat-vllm/remove", "onecat_vllm", OneCatVllmManager),
    ],
)
def test_remove_api_returns_real_task_id(
    client, monkeypatch, tmp_path, route, manager_name, manager_cls
):
    routes = __import__(
        f"backend.routes.{'lmdeploy' if manager_name == 'lmdeploy' else 'onecat_vllm'}_versions",
        fromlist=["get_lmdeploy_manager", "get_onecat_vllm_manager"],
    )
    getter = (
        routes.get_lmdeploy_manager
        if manager_name == "lmdeploy"
        else routes.get_onecat_vllm_manager
    )
    manager = _prevent_background_task(
        manager_cls(
            log_path=str(tmp_path / f"{manager_name}.log"),
            state_path=str(tmp_path / f"{manager_name}_state.json"),
            base_dir=str(tmp_path / manager_name),
        )
    )
    monkeypatch.setattr(routes, getter.__name__, lambda: manager)

    r = client.post(route, json={})
    assert r.status_code == 200
    _assert_started_task(r.json(), task_type="install", manager=manager_name)


@pytest.mark.asyncio
@pytest.mark.parametrize("manager_cls,manager_name,start_method,start_kwargs", [
    (LMDeployManager, "lmdeploy", "install_release", {}),
    (LMDeployManager, "lmdeploy", "install_from_source", {"branch": "main"}),
    (
        LMDeployManager,
        "lmdeploy",
        "sync_source_version",
        {
            "version_entry": {
                "version": "source-main",
                "type": "source",
                "install_type": "source",
                "source_branch": "main",
                "source_repo": "https://example.test/lmdeploy.git",
                "venv_path": "/tmp/lmdeploy-venv",
            }
        },
    ),
    (LMDeployManager, "lmdeploy", "remove", {}),
    (OneCatVllmManager, "onecat_vllm", "install_release", {}),
    (OneCatVllmManager, "onecat_vllm", "install_from_source", {"branch": "main"}),
    (
        OneCatVllmManager,
        "onecat_vllm",
        "sync_source_version",
        {
            "version_entry": {
                "version": "source-main",
                "type": "source",
                "install_type": "source",
                "source_branch": "main",
                "source_repo": "https://example.test/1cat.git",
                "venv_path": "/tmp/onecat-venv",
            }
        },
    ),
    (OneCatVllmManager, "onecat_vllm", "remove", {}),
])
async def test_install_manager_operations_return_real_task_ids(
    tmp_path, manager_cls, manager_name, start_method, start_kwargs
):
    manager = _prevent_background_task(
        manager_cls(
            log_path=str(tmp_path / f"{manager_name}.log"),
            state_path=str(tmp_path / f"{manager_name}_state.json"),
            base_dir=str(tmp_path / manager_name),
        )
    )
    if start_method == "sync_source_version":
        venv = tmp_path / manager_name / "venv"
        venv.mkdir(parents=True)
        start_kwargs["version_entry"]["venv_path"] = str(venv)

    body = await getattr(manager, start_method)(**start_kwargs)
    task_id = _assert_started_task(body, task_type="install", manager=manager_name)
    cancel = manager.cancel_task(task_id)
    assert cancel["ok"] is True
    assert cancel["task_id"] == task_id


def test_cuda_install_api_returns_real_task_id(client, monkeypatch, tmp_path):
    manager = _prevent_background_task(
        CUDAInstaller(
            log_path=str(tmp_path / "cuda.log"),
            state_path=str(tmp_path / "cuda_state.json"),
        )
    )
    async def fake_fetch(_version):
        return "https://example.test/cuda.run"

    monkeypatch.setattr(manager, "_fetch_download_url", fake_fetch)
    _patch_cuda_installer(monkeypatch, tmp_path, manager)

    r = client.post(
        "/api/llama-versions/cuda-install",
        json={"version": "12.6", "install_cudnn": False, "install_tensorrt": False},
    )
    assert r.status_code == 200
    _assert_started_task(r.json(), task_type="install", manager="cuda")


def test_cuda_uninstall_api_returns_real_task_id(client, monkeypatch, tmp_path):
    install_path = tmp_path / "cuda-12.6"
    install_path.mkdir()
    manager = _prevent_background_task(
        CUDAInstaller(
            log_path=str(tmp_path / "cuda.log"),
            state_path=str(tmp_path / "cuda_state.json"),
        )
    )
    manager._load_state = lambda: {  # type: ignore[method-assign]
        "installations": {"12.6": {"path": str(install_path)}},
        "installed_version": "12.6",
    }
    _patch_cuda_installer(monkeypatch, tmp_path, manager)

    r = client.post("/api/llama-versions/cuda-uninstall", json={"version": "12.6"})
    assert r.status_code == 200
    _assert_started_task(r.json(), task_type="install", manager="cuda")


# --- Download tasks -----------------------------------------------------------


@pytest.mark.parametrize(
    "payload,expected_prefix",
    [
        (
            {
                "huggingface_id": "org/model",
                "filename": "model-Q4_K_M.gguf",
                "model_format": "gguf",
            },
            "download_gguf_",
        ),
        (
            {
                "huggingface_id": "org/model",
                "filename": "model.safetensors",
                "model_format": "safetensors",
            },
            "download_safetensors_",
        ),
    ],
)
def test_single_file_download_returns_real_task_id(monkeypatch, payload, expected_prefix):
    pm = pm_mod.get_progress_manager()
    background_tasks = BackgroundTasks()
    original_downloads = dict(model_downloads.active_downloads)
    model_downloads.active_downloads.clear()
    monkeypatch.setattr(models_routes.time, "time", lambda: 1_700_000_000.123)

    try:
        result = asyncio.run(
            models_routes.download_huggingface_model(payload, background_tasks)
        )
        task_id = _assert_started_task(result, task_type="download")
        assert task_id.startswith(expected_prefix)
        assert task_id in model_downloads.active_downloads

        cancel = asyncio.run(models_routes.cancel_download({"task_id": task_id}))
        assert cancel["ok"] is True
        assert cancel["task_id"] == task_id
    finally:
        model_downloads.active_downloads.clear()
        model_downloads.active_downloads.update(original_downloads)


def test_safetensors_bundle_download_returns_real_task_id(monkeypatch):
    background_tasks = BackgroundTasks()
    original_downloads = dict(model_downloads.active_downloads)
    model_downloads.active_downloads.clear()
    monkeypatch.setattr(models_routes.time, "time", lambda: 1_700_000_000)

    try:
        result = asyncio.run(
            models_routes.download_safetensors_bundle(
                models_routes.SafetensorsBundleRequest(
                    huggingface_id="org/repo",
                    files=[{"filename": "model.safetensors", "size": 12}],
                ),
                background_tasks,
            )
        )
        task_id = _assert_started_task(result, task_type="download")
        assert task_id.startswith("download_safetensors_bundle_")
        assert task_id in model_downloads.active_downloads
    finally:
        model_downloads.active_downloads.clear()
        model_downloads.active_downloads.update(original_downloads)


def test_gguf_bundle_download_returns_real_task_id(monkeypatch):
    background_tasks = BackgroundTasks()
    original_downloads = dict(model_downloads.active_downloads)
    model_downloads.active_downloads.clear()
    monkeypatch.setattr(models_routes.time, "time", lambda: 1_700_000_000)

    try:
        result = asyncio.run(
            models_routes.download_gguf_bundle(
                {
                    "huggingface_id": "org/repo",
                    "quantization": "Q4_K_M",
                    "files": [{"filename": "model-Q4_K_M.gguf", "size": 100}],
                },
                background_tasks,
            )
        )
        task_id = _assert_started_task(result, task_type="download")
        assert task_id.startswith("download_gguf_bundle_")
        assert task_id in model_downloads.active_downloads
    finally:
        model_downloads.active_downloads.clear()
        model_downloads.active_downloads.update(original_downloads)


def test_projector_download_returns_real_task_id(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.add_model(
        {
            "id": "org--model--Q4_K_M",
            "name": "Test",
            "huggingface_id": "org/model",
            "quantization": "Q4_K_M",
            "format": "gguf",
            "config": {},
        }
    )
    monkeypatch.setattr(models_routes.time, "time", lambda: 1_700_000_000)

    r = client.post(
        "/api/models/org--model--Q4_K_M/projector",
        json={"mmproj_filename": "mmproj-F16.gguf"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("task_id"), body
    _assert_started_task(body, task_type="download")
    assert body["task_id"].startswith("download_projector_")


# --- Legacy + removed unified API ---------------------------------------------


def test_legacy_status_events_carry_same_task_id(client, monkeypatch, tmp_path):
    from backend.routes import lmdeploy_versions as routes

    manager = _prevent_background_task(
        LMDeployManager(
            log_path=str(tmp_path / "lmdeploy.log"),
            state_path=str(tmp_path / "lmdeploy_state.json"),
            base_dir=str(tmp_path / "lmdeploy"),
        )
    )
    broadcasts = []

    async def capture_broadcast(body):
        broadcasts.append(body)

    monkeypatch.setattr(pm_mod.get_progress_manager(), "broadcast", capture_broadcast)
    monkeypatch.setattr(routes, "get_lmdeploy_manager", lambda: manager)

    r = client.post("/api/lmdeploy/install", json={})
    task_id = r.json()["task_id"]
    assert_not_synthetic(task_id)

    status_events = [b for b in broadcasts if b.get("type") == "lmdeploy_install_status"]
    assert status_events, "expected lmdeploy_install_status legacy broadcast"
    assert status_events[0].get("task_id") == task_id


def test_unified_tasks_cancel_route_removed(client):
    r = client.post("/api/tasks/cancel", json={"task_id": "anything"})
    assert r.status_code in (404, 405)

    schema = client.get("/openapi.json").json()
    assert "/api/tasks/cancel" not in schema.get("paths", {})
