"""Activate one ready engine version.

Every engine uses this function: the versions API, a build that auto-activates,
and an installer that makes the new version current. The caller passes
``covered_by`` when this process already holds the engine, so activation does
not take a second lock.
"""

from __future__ import annotations

import asyncio
import os
from typing import Mapping, Optional

from fastapi import HTTPException

from backend.data_store import get_store
from backend.engines.lifecycle import BUILD_STATUS_READY, normalize_engine_version_status
from backend.engines.registry import get_engine_spec, required_active_path_fields
from backend.logging_config import get_logger
from backend.operations.exclusive import exclusive_action, exclusive_http_error

logger = get_logger(__name__)

_LLAMA_SERVER_ENGINES = frozenset({"llama_cpp", "unsloth_llama"})
_PYTHON_SERVER_ENGINES = frozenset(
    {"lmdeploy", "1cat_vllm", "sglang", "sglang_v100", "vllm"}
)


def resolve_installed_binary(binary_path: str) -> str:
    """Absolute path for a stored engine binary, including paths relative to /app."""
    if not binary_path:
        return ""
    if os.path.isabs(binary_path):
        return binary_path
    if os.path.exists("/app/data"):
        return os.path.normpath(os.path.join("/app", binary_path))
    cwd = os.getcwd()
    resolved = os.path.normpath(os.path.join(cwd, binary_path))
    if os.path.exists(resolved):
        return resolved
    parent = os.path.dirname(cwd)
    return os.path.normpath(os.path.join(parent, binary_path))


def _abs_venv(venv: str) -> str:
    if not venv:
        return ""
    if os.path.isabs(venv):
        return venv
    if os.path.exists("/app/data"):
        return os.path.normpath(os.path.join("/app", venv))
    return os.path.normpath(os.path.join(os.getcwd(), venv))


def _venv_executable(row: dict, name: str) -> str:
    venv = _abs_venv(str((row or {}).get("venv_path") or ""))
    if not venv:
        return ""
    sub = "Scripts" if os.name == "nt" else "bin"
    if os.name == "nt" and not name.endswith(".exe"):
        name = name + ".exe"
    return os.path.join(venv, sub, name)


def missing_runtime_files(engine: str, row: dict) -> list:
    """Required files that are absent for this engine's active-path fields."""
    spec = get_engine_spec(engine)
    if spec is None or not isinstance(row, dict):
        return ["version"]
    missing = []
    for field in required_active_path_fields(engine, row):
        if field == "binary_path":
            path = resolve_installed_binary(str(row.get(field) or ""))
        elif field == "venv_path" and engine == "lmdeploy":
            path = _venv_executable(row, "lmdeploy")
        elif field == "venv_path":
            path = _venv_executable(row, "python")
        else:
            path = str(row.get(field) or "").strip()
        if not path or not os.path.isfile(path):
            missing.append(field)
    return missing


def _version_row(store, engine: str, version: str) -> Optional[dict]:
    return next(
        (
            item
            for item in store.get_engine_versions(engine)
            if str(item.get("version")) == str(version)
        ),
        None,
    )


async def _scan_when_catalog_missing(store, engine: str, version: str, row: dict) -> None:
    from backend.engines.params import get_version_entry

    if get_version_entry(store, engine, version) is not None:
        return
    try:
        from backend.engines.scan.scanner import scan_engine_version

        await asyncio.to_thread(scan_engine_version, store, engine, row)
    except Exception as exc:
        logger.warning(
            "CLI param scan after activating %s:%s: %s", engine, version, exc
        )


async def _start_llama_server_proxy(engine: str) -> None:
    try:
        from backend.proxy.llama_swap.manager import get_llama_swap_manager

        manager = get_llama_swap_manager()
        await manager._ensure_correct_binary_path()
        try:
            await manager.start_proxy()
        except Exception as exc:
            logger.warning(
                "Failed to start llama-swap after version activation: %s", exc
            )
    except Exception as exc:
        logger.error("Failed to start llama-swap after activation: %s", exc)


async def _start_python_server_proxy(engine: str) -> None:
    try:
        from backend.proxy.llama_swap.manager import get_llama_swap_manager

        manager = get_llama_swap_manager()
        await manager.sync_running_models()
        try:
            await manager.start_proxy()
        except Exception as exc:
            logger.warning(
                "Failed to start llama-swap after %s activation: %s", engine, exc
            )
    except Exception as exc:
        logger.error("Failed after %s activation: %s", engine, exc)


async def _finish_activation(store, engine: str, version: str, row: dict) -> dict:
    from backend.proxy.llama_swap.manager import mark_swap_config_stale

    if engine == "audio_cpp":
        from backend.engines.audio_cpp.activation import finish_audio_activation

        return await finish_audio_activation(store, row)
    await _scan_when_catalog_missing(store, engine, version, row)
    if engine in _LLAMA_SERVER_ENGINES:
        await _start_llama_server_proxy(engine)
    elif engine in _PYTHON_SERVER_ENGINES:
        await _start_python_server_proxy(engine)
    mark_swap_config_stale()
    logger.info("Activated %s version: %s", engine, version)
    return {"message": f"Activated {engine} version {version}"}


async def activate_engine_version(
    engine: str,
    version: str,
    *,
    payload: Optional[Mapping] = None,
    covered_by: Optional[str] = None,
    row: Optional[dict] = None,
) -> dict:
    """Make ``version`` the active install for ``engine``.

    ``covered_by`` is the operation id already holding this engine, such as
    the build or install that just produced the version. A request from the
    API leaves it empty and takes the activation lock itself.
    """
    engine = str(engine or "").strip()
    version = str(version or "").strip()
    spec = get_engine_spec(engine)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown engine {engine}")
    store = get_store()
    if row is None:
        row = _version_row(store, engine, version)
    if not row:
        raise HTTPException(status_code=404, detail=f"{spec.label} version not found")
    status = normalize_engine_version_status(engine, row)
    if status != BUILD_STATUS_READY:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot activate a {status} engine version",
        )
    missing = missing_runtime_files(engine, row)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"{spec.label} version is missing: {', '.join(missing)}",
        )
    try:
        async with exclusive_action(
            "activate",
            f"engine:{engine}",
            detail={"engine": engine, "version": version},
            payload=payload,
            covered_by=covered_by,
        ):
            store.set_active_engine_version(engine, version)
            result = await _finish_activation(store, engine, version, row)
    except Exception as exc:
        mapped = exclusive_http_error(exc)
        if mapped is not None:
            raise mapped from exc
        raise
    result.setdefault("message", f"Activated {engine} version {version}")
    result.setdefault("engine", engine)
    result.setdefault("version", version)
    return result
