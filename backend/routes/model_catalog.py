"""Normalized multi-provider model discovery API."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from backend.engine_registry import VALID_ENGINE_IDS
from backend.model_catalog import ModelCatalogService
from backend.progress_manager import get_progress_manager
from backend.services.audio_model_installer import get_audio_model_installer
from backend.task_cancel_registry import TaskCancelledError


router = APIRouter()


def _filters(payload: Dict[str, Any]) -> dict:
    keys = (
        "engine",
        "task",
        "input_modality",
        "output_modality",
        "feature",
        "provider",
        "source",
        "package_kind",
        "install_method",
        "release_status",
        "language",
        "artifact_format",
        "format",
    )
    filters = {
        key: payload.get(key)
        for key in keys
        if payload.get(key) not in (None, "")
    }
    if isinstance(payload.get("gated"), bool):
        filters["gated"] = payload["gated"]
    engine = filters.get("engine")
    if engine and engine not in VALID_ENGINE_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown engine '{engine}'. Expected one of {sorted(VALID_ENGINE_IDS)}",
        )
    return filters


@router.post("/search")
async def search_catalog(payload: dict = Body(default_factory=dict)):
    payload = payload or {}
    return await ModelCatalogService().search(
        query=str(payload.get("query") or ""),
        filters=_filters(payload),
        page=int(payload.get("page") or 1),
        page_size=int(payload.get("page_size") or payload.get("limit") or 20),
        force_refresh=bool(payload.get("force_refresh")),
    )


@router.get("/search")
async def search_catalog_get(
    query: str = "",
    engine: Optional[str] = None,
    task: Optional[str] = None,
    input_modality: Optional[str] = None,
    output_modality: Optional[str] = None,
    feature: Optional[str] = None,
    provider: Optional[str] = None,
    package_kind: Optional[str] = None,
    install_method: Optional[str] = None,
    release_status: Optional[str] = None,
    language: Optional[str] = None,
    artifact_format: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    raw = {
        "engine": engine,
        "task": task,
        "input_modality": input_modality,
        "output_modality": output_modality,
        "feature": feature,
        "provider": provider,
        "package_kind": package_kind,
        "install_method": install_method,
        "release_status": release_status,
        "language": language,
        "artifact_format": artifact_format,
    }
    return await ModelCatalogService().search(
        query=query,
        filters=_filters(raw),
        page=page,
        page_size=page_size,
    )


async def _run_audio_install(task_id: str, package_id: str, options: dict) -> None:
    installer = get_audio_model_installer()
    pm = get_progress_manager()
    try:
        model = await installer.install_package(
            task_id,
            package_id,
            options=options,
        )
        pm.update_task(
            task_id,
            metadata_update={"model_id": model.get("id"), "stage": "complete"},
        )
        pm.complete_task(task_id, f"Installed {model.get('display_name') or package_id}")
        pm.emit("models_changed", {"action": "installed", "model_id": model.get("id")})
    except TaskCancelledError:
        pm.fail_task(task_id, "Audio package installation cancelled")
    except Exception as exc:
        pm.fail_task(task_id, str(exc))


@router.post("/install")
async def install_catalog_item(payload: dict = Body(default_factory=dict)):
    payload = payload or {}
    provider = str(payload.get("provider") or "")
    if provider != "audio_cpp":
        raise HTTPException(
            status_code=400,
            detail="Catalog installs are currently available for audio.cpp packages; use the existing model download flow for Hugging Face results.",
        )
    package_id = str(
        payload.get("provider_item_id")
        or payload.get("variant_id")
        or payload.get("package_id")
        or ""
    ).strip()
    if not package_id:
        raise HTTPException(status_code=400, detail="provider_item_id is required")
    installer = get_audio_model_installer()
    try:
        package = installer.package_metadata(package_id)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    source = package.get("source") or {}
    options = {
        key: payload.get(key)
        for key in ("source_file", "source_dir", "output_file", "variant", "family")
        if payload.get(key) not in (None, "")
    }
    source_kind = str(source.get("kind") or "")
    if source_kind == "utility" and not (
        options.get("source_file") or options.get("source_dir")
    ):
        raise HTTPException(
            status_code=422,
            detail="This converter package requires source_file or source_dir.",
        )

    pm = get_progress_manager()
    task_id = pm.create_task(
        "audio_model_install",
        f"Install audio.cpp package {package_id}",
        metadata={
            "package_id": package_id,
            "provider": "audio_cpp",
            "stage": "queued",
        },
    )
    asyncio.create_task(_run_audio_install(task_id, package_id, options))
    return {
        "success": True,
        "task_id": task_id,
        "message": f"Installation of {package_id} started",
    }


async def _run_audio_import(
    task_id: str,
    source_path: str,
    package_id: Optional[str],
    family: Optional[str],
) -> None:
    installer = get_audio_model_installer()
    pm = get_progress_manager()
    try:
        model = await installer.import_local_bundle(
            task_id,
            source_path,
            package_id=package_id,
            family=family,
        )
        pm.update_task(
            task_id,
            metadata_update={"model_id": model.get("id"), "stage": "complete"},
        )
        pm.complete_task(task_id, f"Imported {model.get('display_name')}")
        pm.emit("models_changed", {"action": "imported", "model_id": model.get("id")})
    except TaskCancelledError:
        pm.fail_task(task_id, "Audio bundle import cancelled")
    except Exception as exc:
        pm.fail_task(task_id, str(exc))


@router.post("/import")
async def import_audio_bundle(payload: dict = Body(default_factory=dict)):
    payload = payload or {}
    source_path = str(payload.get("source_path") or "").strip()
    if not source_path:
        raise HTTPException(status_code=400, detail="source_path is required")
    installer = get_audio_model_installer()
    try:
        installer._active_version()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pm = get_progress_manager()
    task_id = pm.create_task(
        "audio_model_import",
        f"Import local audio.cpp bundle {source_path}",
        metadata={"source_path": source_path, "stage": "queued"},
    )
    asyncio.create_task(
        _run_audio_import(
            task_id,
            source_path,
            str(payload.get("package_id") or "").strip() or None,
            str(payload.get("family") or "").strip() or None,
        )
    )
    return {
        "success": True,
        "task_id": task_id,
        "message": "Local audio.cpp bundle import started",
    }


@router.post("/tasks/{task_id}/cancel")
async def cancel_audio_install(task_id: str):
    task = get_progress_manager().get_task(task_id)
    if not task or task.get("type") not in {
        "audio_model_install",
        "audio_model_import",
    }:
        raise HTTPException(status_code=404, detail="Audio install task not found")
    if task.get("status") != "running":
        return {"success": False, "message": "Task is no longer running"}
    cancelled = await get_audio_model_installer().cancel(task_id)
    return {"success": cancelled, "task_id": task_id}


@router.get("/tasks/{task_id}")
async def get_audio_install_task(task_id: str):
    task = get_progress_manager().get_task(task_id)
    if not task or task.get("type") not in {
        "audio_model_install",
        "audio_model_import",
    }:
        raise HTTPException(status_code=404, detail="Audio install task not found")
    return task

