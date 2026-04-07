from fastapi import APIRouter, Body, HTTPException, BackgroundTasks, Query
from fastapi.responses import Response
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import json
import os
import time
import asyncio

from backend.data_store import get_store, resolve_proxy_name
from backend.model_config import (
    config_api_response,
    effective_model_config_from_raw,
    merge_model_config_put,
    normalize_model_config,
)
from backend.progress_manager import get_progress_manager
from backend.huggingface import (
    search_models,
    set_huggingface_token,
    get_huggingface_token,
    get_model_details,
    extract_quantization,
    list_grouped_safetensors_downloads,
    get_safetensors_manifest_entries,
    get_accurate_file_sizes,
    resolve_cached_model_path,
    get_gguf_limits_from_manifest,
    get_safetensors_limits_from_manifest,
    purge_gguf_store_model,
    purge_safetensors_repo_completely,
    delete_cached_model_file,
)
from backend.logging_config import get_logger
import backend.llama_swap_config as llama_swap_config
from backend.services.model_downloads import (
    ActiveDownloadConflict,
    download_gguf_bundle_task,
    download_model_projector_task,
    download_model_task,
    download_safetensors_bundle_task,
    register_gguf_bundle_download,
    register_gguf_projector_download,
    register_safetensors_bundle_download,
    register_single_model_download,
)
from backend.services.model_metadata import (
    model_is_embedding,
)

logger = get_logger(__name__)


def _passthrough_llama_swap_response(response) -> Response:
    headers = {}
    content_type = response.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=headers,
    )


def _mark_llama_swap_stale() -> None:
    try:
        from backend.llama_swap_manager import mark_swap_config_stale

        mark_swap_config_stale()
    except Exception as exc:
        logger.debug("mark_swap_config_stale: %s", exc)


router = APIRouter()


def _is_mmproj_filename(filename: Optional[str]) -> bool:
    name = (filename or "").strip().lower()
    return bool(name) and "mmproj" in name and name.endswith(".gguf")


def _get_model_or_404(store, model_id: str) -> dict:
    """Return model dict from store or raise 404. Accepts str model_id (YAML id)."""
    if model_id is None:
        raise HTTPException(status_code=404, detail="Model not found")
    model_id = str(model_id)
    model = store.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


def _other_models_share_mmproj(
    store,
    huggingface_id: str,
    mmproj_filename: str,
    exclude_model_id: str,
) -> bool:
    for m in store.list_models():
        if m.get("id") == exclude_model_id:
            continue
        if m.get("huggingface_id") != huggingface_id:
            continue
        if m.get("mmproj_filename") == mmproj_filename:
            return True
    return False


async def _remove_model_from_disk_and_manifests(store, model: dict) -> None:
    """Delete HF cache / on-disk files and per-repo manifests before dropping the store row."""
    fmt = (model.get("format") or model.get("model_format") or "gguf").lower()
    hf_id = model.get("huggingface_id")
    mid = model.get("id")
    if fmt == "safetensors" and hf_id:
        purge_safetensors_repo_completely(hf_id)
    elif fmt == "gguf" and hf_id:
        purge_gguf_store_model(hf_id, mid, model.get("quantization"))
        mmproj = model.get("mmproj_filename")
        if mmproj and not _other_models_share_mmproj(store, hf_id, mmproj, mid):
            delete_cached_model_file(hf_id, mmproj)


def _coerce_model_config(config_value: Optional[Any]) -> Dict[str, Any]:
    """Effective flat model config (per active engine)."""
    return effective_model_config_from_raw(config_value)


def _get_safetensors_model(store, model_id: str) -> dict:
    model = _get_model_or_404(store, model_id)
    model_format = (model.get("model_format") or model.get("format") or "gguf").lower()
    if model_format != "safetensors":
        raise HTTPException(
            status_code=400, detail="Model is not a safetensors download"
        )
    # Safetensors models are treated as repo-level entities; concrete file paths
    # are tracked in the safetensors manifest, not on the model record itself.
    return dict(model)


def _load_manifest_entry_for_model(model: dict) -> Dict[str, Any]:
    """Load unified manifest for a safetensors model (repo-level, not per-file)."""
    manifest = get_safetensors_manifest_entries(model.get("huggingface_id"))
    if not manifest:
        raise HTTPException(status_code=404, detail="Safetensors manifest not found")
    return manifest


PROMPT_RESERVED_TOKENS = 8192


def _apply_prompt_reservation(value: Optional[int]) -> Optional[int]:
    if value and value > PROMPT_RESERVED_TOKENS:
        adjusted = value - PROMPT_RESERVED_TOKENS
        return adjusted if adjusted >= 1024 else max(adjusted, 1024)
    return value


def _normalize_hf_overrides(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="hf_overrides must be valid JSON"
            ) from exc
    if isinstance(value, dict):

        def _sanitize(obj: Any) -> Any:
            if isinstance(obj, dict):
                result = {}
                for key, nested in obj.items():
                    if not isinstance(key, str) or not key.strip():
                        raise HTTPException(
                            status_code=400,
                            detail="hf_overrides keys must be non-empty strings",
                        )
                    result[key.strip()] = _sanitize(nested)
                return result
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            raise HTTPException(
                status_code=400,
                detail="hf_overrides values must be scalars or nested objects",
            )

        sanitized = _sanitize(value)
        return sanitized
    raise HTTPException(
        status_code=400, detail="hf_overrides must be an object or JSON string"
    )


class SafetensorsBundleRequest(BaseModel):
    huggingface_id: str
    model_id: Optional[int] = None
    files: List[Dict[str, Any]]


@router.get("/param-registry")
async def get_param_registry_endpoint(
    engine: str = "llama_cpp",
    dynamic: bool = Query(
        True,
        description="When true, auto-scan the active engine binary once if no catalog entry exists yet.",
    ),
):
    """Return param definitions from ``engine_params_catalog.yaml`` plus studio-only fields."""
    from backend.engine_param_catalog import (
        get_version_entry,
        registry_payload_from_entry,
    )
    from backend.engine_param_scanner import scan_engine_version
    from backend.studio_engine_fields import studio_sections_for_engine

    store = get_store()
    if engine not in ("llama_cpp", "ik_llama", "lmdeploy"):
        return registry_payload_from_entry(engine, None, [], has_active_engine=False)

    studio = studio_sections_for_engine(engine)
    active = store.get_active_engine_version(engine)
    has_active = bool(
        active
        and (
            active.get("binary_path")
            or (engine == "lmdeploy" and active.get("venv_path"))
        )
    )
    entry = None
    if active and active.get("version"):
        entry = get_version_entry(store, engine, active["version"])
    if dynamic and has_active and active and entry is None:
        await asyncio.to_thread(scan_engine_version, store, engine, active)
        entry = get_version_entry(store, engine, active["version"])

    return registry_payload_from_entry(
        engine, entry, studio, has_active_engine=has_active
    )


@router.get("")
@router.get("/")
async def list_models():
    """List all managed models grouped by base model"""
    from backend.llama_swap_client import LlamaSwapClient

    store = get_store()
    # Include all stored models (GGUF and safetensors). GGUF entries appear as
    # individual quantizations; safetensors entries appear as a single logical
    # quantization per repo with format "safetensors".
    models = list(store.list_models())
    try:
        running_data = await LlamaSwapClient().get_running_models()
        running_list = running_data.get("running") or []
        proxy_state_by_name: Dict[str, str] = {}
        for item in running_list:
            name = item.get("model")
            if not name:
                continue
            st = (item.get("state") or "").lower()
            if st in ("running", "ready", "loading"):
                proxy_state_by_name[name] = st
        running_names = set(proxy_state_by_name.keys())
    except Exception:
        running_names = set()
        proxy_state_by_name = {}

    grouped_models = {}
    for model in models:
        hf_id = model.get("huggingface_id") or ""
        base_name = model.get("base_model_name") or (
            hf_id.split("/")[-1] if hf_id else model.get("display_name") or "unknown"
        )
        proxy_name = resolve_proxy_name(model)
        is_active = proxy_name in running_names
        raw_state = (
            proxy_state_by_name.get(proxy_name) if proxy_name in running_names else None
        )
        if raw_state == "loading":
            run_state = "loading"
        elif raw_state in ("running", "ready"):
            run_state = "running"
        else:
            run_state = None
        is_embedding = model_is_embedding(model)
        key = f"{hf_id}_{base_name}"
        if key not in grouped_models:
            author = (
                hf_id.split("/")[0] if isinstance(hf_id, str) and "/" in hf_id else ""
            )
            grouped_models[key] = {
                "base_model_name": base_name,
                "huggingface_id": hf_id,
                "model_type": model.get("model_type"),
                "author": author,
                "pipeline_tag": model.get("pipeline_tag"),
                "is_embedding_model": is_embedding,
                "quantizations": [],
            }
        else:
            if model.get("pipeline_tag") and not grouped_models[key].get(
                "pipeline_tag"
            ):
                grouped_models[key]["pipeline_tag"] = model.get("pipeline_tag")
            if is_embedding and not grouped_models[key].get("is_embedding_model"):
                grouped_models[key]["is_embedding_model"] = True

        file_size = model.get("file_size") or 0

        grouped_models[key]["quantizations"].append(
            {
                "id": model.get("id"),
                "name": model.get("display_name") or model.get("name"),
                # No filename persisted for GGUF models; a model is a single logical
                # entity per (huggingface_id, quantization).
                "file_size": file_size,
                "quantization": model.get("quantization"),
                "format": model.get("format") or model.get("model_format") or "gguf",
                "downloaded_at": model.get("downloaded_at"),
                "is_active": is_active,
                "status": raw_state,
                "run_state": run_state,
                "has_config": bool(model.get("config")),
                "mmproj_filename": model.get("mmproj_filename"),
                "huggingface_id": hf_id,
                "base_model_name": base_name,
                "model_type": model.get("model_type"),
                "config": _coerce_model_config(model.get("config")),
                "proxy_name": proxy_name,
                "pipeline_tag": model.get("pipeline_tag"),
                "is_embedding_model": is_embedding,
            }
        )

    result = []
    for group in grouped_models.values():
        group["quantizations"].sort(key=lambda x: x.get("file_size") or 0)
        result.append(group)
    result.sort(key=lambda x: x.get("base_model_name") or "")
    return result


@router.post("/search")
async def search_huggingface_models(request: dict):
    """Search HuggingFace for GGUF models"""
    try:
        query = request.get("query")
        limit = request.get("limit", 20)
        model_format = (request.get("model_format") or "gguf").lower()

        if not query:
            raise HTTPException(status_code=400, detail="query parameter is required")
        if model_format not in ("gguf", "safetensors"):
            raise HTTPException(
                status_code=400,
                detail="model_format must be either 'gguf' or 'safetensors'",
            )

        results = await search_models(query, limit, model_format=model_format)
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/{model_id:path}/file-sizes")
async def get_search_file_sizes(
    model_id: str,
    filenames: str = Query(
        ..., description="Comma-separated list of file paths in the repo"
    ),
):
    """Get accurate file sizes for specific files in a repo via HuggingFace API."""
    file_list = [f.strip() for f in filenames.split(",") if f.strip()]
    if not file_list:
        raise HTTPException(status_code=400, detail="At least one filename is required")
    sizes = get_accurate_file_sizes(model_id, file_list)
    return {"sizes": sizes}


@router.get("/safetensors")
async def list_safetensors_models():
    """List safetensors downloads stored locally."""
    try:
        return list_grouped_safetensors_downloads()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/safetensors")
async def delete_safetensors_model(request: dict):
    """Delete entire safetensors model (all files for the repo)."""
    try:
        huggingface_id = request.get("huggingface_id")
        if not huggingface_id:
            raise HTTPException(status_code=400, detail="huggingface_id is required")

        store = get_store()
        model_id = huggingface_id.replace("/", "--")
        target_model = store.get_model(model_id)
        if (
            not target_model
            or (target_model.get("format") or target_model.get("model_format"))
            != "safetensors"
        ):
            raise HTTPException(status_code=404, detail="Safetensors model not found")

        # LMDeploy runtime is now managed via llama-swap; safetensors models
        # are served through the same generic start/stop flow, so we don't
        # need to special-case LMDeploy here.

        manifest = get_safetensors_manifest_entries(huggingface_id)
        if not manifest or not manifest.get("files"):
            raise HTTPException(status_code=404, detail="Safetensors model not found")

        from backend.llama_swap_client import LlamaSwapClient

        proxy_name = resolve_proxy_name(target_model)
        try:
            running_data = await LlamaSwapClient().get_running_models()
            running_list = running_data.get("running") or []
            running_names = {
                item.get("model")
                for item in running_list
                if item.get("state") in ("running", "ready", "loading")
            }
        except Exception:
            running_names = set()
        if proxy_name in running_names:
            try:
                from backend.llama_swap_manager import get_llama_swap_manager

                await get_llama_swap_manager().unregister_model(proxy_name)
            except Exception as e:
                logger.warning(f"Failed to stop model {proxy_name}: {e}")

        purge_safetensors_repo_completely(huggingface_id)
        store.delete_model(model_id)
        _mark_llama_swap_stale()
        return {"message": f"Safetensors model {huggingface_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download")
async def download_huggingface_model(request: dict, background_tasks: BackgroundTasks):
    """Download model from HuggingFace"""
    try:
        huggingface_id = request.get("huggingface_id")
        filename = request.get("filename")
        total_bytes = request.get(
            "total_bytes", 0
        )  # Get total size from search results
        model_format = (request.get("model_format") or "gguf").lower()
        pipeline_tag = request.get("pipeline_tag")

        if not huggingface_id or not filename:
            raise HTTPException(
                status_code=400, detail="huggingface_id and filename are required"
            )
        if model_format not in ("gguf", "safetensors"):
            raise HTTPException(
                status_code=400,
                detail="model_format must be either 'gguf' or 'safetensors'",
            )
        if model_format == "gguf" and not filename.endswith(".gguf"):
            raise HTTPException(
                status_code=400,
                detail="filename must end with .gguf for GGUF downloads",
            )
        if model_format == "safetensors" and not filename.endswith(".safetensors"):
            raise HTTPException(
                status_code=400,
                detail="filename must end with .safetensors for Safetensors downloads",
            )

        store = get_store()
        is_mmproj_download = model_format == "gguf" and "mmproj" in filename.lower()
        # Check if this specific quantization already exists
        if model_format == "gguf" and not is_mmproj_download:
            quantization = extract_quantization(filename)
            model_id = f"{huggingface_id.replace('/', '--')}--{quantization}"
            if store.get_model(model_id):
                raise HTTPException(
                    status_code=400, detail="This quantization is already downloaded"
                )

        # Extract quantization for better task_id (use same function as search results)
        quantization = (
            os.path.splitext(os.path.basename(filename))[0]
            if is_mmproj_download
            else extract_quantization(filename)
            if model_format == "gguf"
            else os.path.splitext(filename)[0]
        )

        # Generate unique task ID with quantization and milliseconds
        task_id = f"download_{model_format}_{huggingface_id.replace('/', '_')}_{quantization}_{int(time.time() * 1000)}"

        try:
            await register_single_model_download(
                task_id=task_id,
                huggingface_id=huggingface_id,
                filename=filename,
                quantization=quantization,
                model_format=model_format,
            )
        except ActiveDownloadConflict as exc:
            raise HTTPException(status_code=409, detail=exc.detail) from exc

        # Start download in background with progress_manager for SSE
        pm = get_progress_manager()
        pm.create_task(
            "download",
            f"Download {filename}",
            {"huggingface_id": huggingface_id, "filename": filename},
            task_id=task_id,
        )
        background_tasks.add_task(
            download_model_task,
            huggingface_id,
            filename,
            pm,
            task_id,
            total_bytes,
            model_format,
            pipeline_tag,
        )

        return {
            "message": "Download started",
            "huggingface_id": huggingface_id,
            "task_id": task_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/safetensors/download-bundle")
async def download_safetensors_bundle(
    request: SafetensorsBundleRequest, background_tasks: BackgroundTasks
):
    huggingface_id = request.huggingface_id
    files = request.files or []

    if not huggingface_id:
        raise HTTPException(status_code=400, detail="huggingface_id is required")
    if not files:
        raise HTTPException(status_code=400, detail="Repository file list is required")

    sanitized_files = []
    declared_total = 0
    for file in files:
        filename = file.get("filename")
        if not filename:
            continue
        size = max(int(file.get("size") or 0), 0)
        declared_total += size
        sanitized_files.append({"filename": filename, "size": size})

    task_id = f"download_safetensors_bundle_{huggingface_id.replace('/', '_')}_{int(time.time() * 1000)}"

    try:
        await register_safetensors_bundle_download(
            task_id=task_id, huggingface_id=huggingface_id
        )
    except ActiveDownloadConflict as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc

    pm = get_progress_manager()
    pm.create_task(
        "download",
        f"Safetensors bundle {huggingface_id}",
        {"huggingface_id": huggingface_id},
        task_id=task_id,
    )
    background_tasks.add_task(
        download_safetensors_bundle_task,
        huggingface_id,
        sanitized_files,
        pm,
        task_id,
        declared_total,
    )

    return {
        "message": "Safetensors bundle download started",
        "huggingface_id": huggingface_id,
        "task_id": task_id,
    }


@router.post("/gguf/download-bundle")
async def download_gguf_bundle(
    request: dict,
    background_tasks: BackgroundTasks,
):
    huggingface_id = request.get("huggingface_id")
    quantization = request.get("quantization")
    files = request.get("files") or []
    pipeline_tag = request.get("pipeline_tag")
    projector_filename = (request.get("mmproj_filename") or "").strip()
    projector_size = max(int(request.get("mmproj_size") or 0), 0)

    if not huggingface_id:
        raise HTTPException(status_code=400, detail="huggingface_id is required")
    if not quantization:
        raise HTTPException(status_code=400, detail="quantization is required")
    if not files:
        raise HTTPException(status_code=400, detail="Repository file list is required")
    if projector_filename and not _is_mmproj_filename(projector_filename):
        raise HTTPException(status_code=400, detail="Invalid projector filename")

    sanitized_files = []
    declared_total = 0
    for file in files:
        filename = file.get("filename")
        if not filename:
            continue
        size = max(int(file.get("size") or 0), 0)
        declared_total += size
        sanitized_files.append({"filename": filename, "size": size})

    if not sanitized_files:
        raise HTTPException(status_code=400, detail="No valid files to download")

    projector_payload = None
    if projector_filename:
        declared_total += projector_size
        projector_payload = {"filename": projector_filename, "size": projector_size}

    task_id = f"download_gguf_bundle_{huggingface_id.replace('/', '_')}_{quantization}_{int(time.time() * 1000)}"

    try:
        await register_gguf_bundle_download(
            task_id=task_id,
            huggingface_id=huggingface_id,
            quantization=quantization,
        )
    except ActiveDownloadConflict as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc

    pm = get_progress_manager()
    pm.create_task(
        "download",
        f"GGUF bundle {huggingface_id} ({quantization})",
        {"huggingface_id": huggingface_id, "quantization": quantization},
        task_id=task_id,
    )
    background_tasks.add_task(
        download_gguf_bundle_task,
        huggingface_id,
        quantization,
        sanitized_files,
        pm,
        task_id,
        declared_total,
        pipeline_tag,
        projector_payload,
    )

    return {
        "message": "GGUF bundle download started",
        "huggingface_id": huggingface_id,
        "quantization": quantization,
        "task_id": task_id,
    }


@router.get("/huggingface-token")
async def get_huggingface_token_status():
    """Get HuggingFace API token status"""
    token = get_huggingface_token()
    env_token = os.getenv("HUGGINGFACE_API_KEY")

    return {
        "has_token": bool(token),
        "token_preview": f"{token[:8]}..." if token else None,
        "from_environment": bool(env_token),
        "environment_set": bool(env_token),
    }


@router.post("/huggingface-token")
async def set_huggingface_token_endpoint(request: dict):
    """Set HuggingFace API token"""
    try:
        # Check if token is set via environment variable
        env_token = os.getenv("HUGGINGFACE_API_KEY")
        if env_token:
            return {
                "message": "Token is set via environment variable and cannot be overridden via UI",
                "has_token": True,
                "from_environment": True,
            }

        token = request.get("token", "").strip()

        if not token:
            set_huggingface_token("")
            return {"message": "HuggingFace token cleared", "has_token": False}

        # Validate token format (basic check)
        if len(token) < 10:
            raise HTTPException(status_code=400, detail="Invalid token format")

        set_huggingface_token(token)
        return {"message": "HuggingFace token set successfully", "has_token": True}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id:path}/projector")
async def update_model_projector(
    model_id: str,
    request: dict,
    background_tasks: BackgroundTasks,
):
    store = get_store()
    model = _get_model_or_404(store, model_id)
    if (model.get("format") or model.get("model_format")) != "gguf":
        raise HTTPException(
            status_code=400, detail="Projectors are only supported for GGUF models"
        )

    mmproj_filename = (request.get("mmproj_filename") or "").strip() or None
    total_bytes = max(int(request.get("total_bytes") or 0), 0)

    if mmproj_filename and not _is_mmproj_filename(mmproj_filename):
        raise HTTPException(status_code=400, detail="Invalid projector filename")

    current_projector = model.get("mmproj_filename")
    if mmproj_filename == current_projector:
        return {"message": "Projector already selected", "applied": True}

    if not mmproj_filename:
        store.update_model(model_id, {"mmproj_filename": None})
        _mark_llama_swap_stale()
        return {"message": "Projector cleared", "applied": True}

    huggingface_id = model.get("huggingface_id")
    cached_path = resolve_cached_model_path(huggingface_id, mmproj_filename)
    if cached_path and os.path.exists(cached_path):
        store.update_model(model_id, {"mmproj_filename": mmproj_filename})
        _mark_llama_swap_stale()
        return {"message": "Projector applied", "applied": True}

    task_id = (
        f"download_projector_{model_id.replace('/', '_')}_{int(time.time() * 1000)}"
    )
    try:
        await register_gguf_projector_download(
            task_id=task_id,
            huggingface_id=huggingface_id,
            model_id=model_id,
            mmproj_filename=mmproj_filename,
        )
    except ActiveDownloadConflict as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc

    pm = get_progress_manager()
    pm.create_task(
        "download",
        f"Projector {mmproj_filename}",
        {
            "huggingface_id": huggingface_id,
            "filename": mmproj_filename,
            "model_id": model_id,
        },
        task_id=task_id,
    )
    background_tasks.add_task(
        download_model_projector_task,
        model_id,
        mmproj_filename,
        pm,
        task_id,
        total_bytes,
    )
    return {
        "message": "Projector download started",
        "task_id": task_id,
        "applied": False,
    }


@router.get("/{model_id:path}/limits")
async def get_model_limits(model_id: str):
    """
    Return model limits in an engine-agnostic way. For GGUF models with a manifest
    entry, uses the GGUF manifest; for safetensors, uses the safetensors manifest.
    Otherwise falls back to the Hugging Face model card (config.json / model info).
    """
    store = get_store()
    model = _get_model_or_404(store, model_id)
    hf_id = model.get("huggingface_id")
    if not hf_id:
        return {"max_context_length": None, "layer_count": None}

    max_ctx = None
    layer_count = None
    fmt = model.get("format") or model.get("model_format") or "gguf"
    quant = model.get("quantization")

    if fmt == "gguf" and quant:
        max_ctx, layer_count = get_gguf_limits_from_manifest(hf_id, quant)
    elif fmt == "safetensors":
        max_ctx, layer_count = get_safetensors_limits_from_manifest(hf_id)

    if max_ctx is None or layer_count is None:
        try:
            details = await get_model_details(hf_id)
            config = details.get("config") or {}
            if max_ctx is None:
                hf_max = details.get("model_max_length") or config.get(
                    "max_position_embeddings"
                )
                if isinstance(hf_max, (int, float)) and hf_max > 0:
                    max_ctx = int(hf_max)
            if layer_count is None:
                for key in ("num_hidden_layers", "n_layer", "num_layers"):
                    val = config.get(key)
                    if isinstance(val, (int, float)) and val > 0:
                        layer_count = (
                            int(val) + 1
                        )  # + output head for n_gpu_layers hint
                        break
        except Exception:
            pass
    return {"max_context_length": max_ctx, "layer_count": layer_count}


@router.get("/{model_id:path}/config")
async def get_model_config(model_id: str):
    """Get model's llama.cpp configuration"""
    store = get_store()
    model = _get_model_or_404(store, model_id)
    return config_api_response(normalize_model_config(model.get("config")))


@router.put("/{model_id:path}/config")
async def update_model_config(model_id: str, config: dict):
    """Update model's llama.cpp configuration"""
    store = get_store()
    model = _get_model_or_404(store, model_id)
    merged = merge_model_config_put(model.get("config"), config)
    store.update_model(model_id, {"config": merged})
    _mark_llama_swap_stale()
    return config_api_response(merged)


@router.get("/{model_id:path}/saved-llama-swap-cmd")
async def get_saved_llama_swap_cmd(model_id: str):
    """
    Return the llama-swap ``cmd`` for this model using **stored** DB config only.
    Cheap to poll: no request-body merge (unlike POST preview).
    """
    store = get_store()
    model = _get_model_or_404(store, model_id)
    return llama_swap_config.preview_llama_swap_command_for_model({**model})


@router.post("/{model_id:path}/preview-llama-swap-cmd")
async def preview_llama_swap_cmd(
    model_id: str, body: dict = Body(default_factory=dict)
):
    """
    Return the llama-swap ``cmd`` string that would be generated for this model,
    using the same merge rules as PUT /config. Body: ``{ "engine", "engines" }`` (optional).
    """
    store = get_store()
    model = _get_model_or_404(store, model_id)
    merged = merge_model_config_put(model.get("config"), body or {})
    preview_model = {**model, "config": merged}
    return llama_swap_config.preview_llama_swap_command_for_model(preview_model)


@router.post("/{model_id:path}/start")
async def start_model(model_id: str):
    """Pass through model start to llama-swap."""
    from backend.llama_swap_client import LlamaSwapClient

    store = get_store()
    model = _get_model_or_404(store, model_id)
    proxy_model_name = resolve_proxy_name(model)
    response = await LlamaSwapClient().start_model_passthrough(proxy_model_name)
    return _passthrough_llama_swap_response(response)


@router.post("/{model_id:path}/stop")
async def stop_model(model_id: str):
    """Pass through model stop to llama-swap."""
    from backend.llama_swap_client import LlamaSwapClient

    store = get_store()
    model = _get_model_or_404(store, model_id)
    proxy_name = resolve_proxy_name(model)
    response = await LlamaSwapClient().stop_model_passthrough(proxy_name)
    return _passthrough_llama_swap_response(response)


@router.post("/quantization-sizes")
async def get_quantization_sizes(request: dict):
    """Get actual file sizes for quantizations from HuggingFace API"""
    try:
        huggingface_id = request.get("huggingface_id")
        quantizations = request.get("quantizations", {})

        if not huggingface_id or not quantizations:
            raise HTTPException(
                status_code=400, detail="huggingface_id and quantizations are required"
            )
        # Use centralized Hugging Face service helper
        from backend.huggingface import get_quantization_sizes_from_hf

        updated_quantizations = await get_quantization_sizes_from_hf(
            huggingface_id, quantizations
        )

        # Fallback: for any remaining without size, try HTTP HEAD
        if updated_quantizations is None:
            updated_quantizations = {}

        missing = [q for q in quantizations.keys() if q not in updated_quantizations]
        if missing:
            import requests

            for quant_name in missing:
                quant_data = quantizations.get(quant_name) or {}
                filename = quant_data.get("filename")
                if not filename:
                    continue
                url = f"https://huggingface.co/{huggingface_id}/resolve/main/{filename}"
                try:
                    response = requests.head(url, timeout=10)
                    if response.status_code == 200:
                        content_length = response.headers.get("content-length")
                        if content_length:
                            actual_size = int(content_length)
                            updated_quantizations[quant_name] = {
                                "filename": filename,
                                "size": actual_size,
                                "size_mb": round(actual_size / (1024 * 1024), 2),
                            }
                except Exception:
                    continue

        return {"quantizations": updated_quantizations}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeleteGroupRequest(BaseModel):
    huggingface_id: str


@router.post("/delete-group")
async def delete_model_group(request: DeleteGroupRequest):
    """Delete all quantizations of a model group"""
    from backend.llama_swap_client import LlamaSwapClient

    huggingface_id = request.huggingface_id
    store = get_store()
    models = [
        m for m in store.list_models() if m.get("huggingface_id") == huggingface_id
    ]
    if not models:
        raise HTTPException(status_code=404, detail="Model group not found")

    try:
        running_data = await LlamaSwapClient().get_running_models()
        running_list = running_data.get("running") or []
        running_names = {
            item.get("model")
            for item in running_list
            if item.get("state") in ("running", "ready", "loading")
        }
    except Exception:
        running_names = set()

    deleted_count = 0
    for model in models:
        proxy_name = resolve_proxy_name(model)
        if proxy_name in running_names:
            try:
                from backend.llama_swap_manager import get_llama_swap_manager

                await get_llama_swap_manager().unregister_model(proxy_name)
            except Exception as e:
                logger.warning(f"Failed to stop model {proxy_name}: {e}")

        await _remove_model_from_disk_and_manifests(store, model)
        store.delete_model(model.get("id"))
        deleted_count += 1

    _mark_llama_swap_stale()
    return {"message": f"Deleted {deleted_count} quantizations"}


@router.delete("/{model_id:path}")
async def delete_model(model_id: str):
    """Delete individual model quantization and its files"""
    from backend.llama_swap_client import LlamaSwapClient

    store = get_store()
    model = _get_model_or_404(store, model_id)
    proxy_name = resolve_proxy_name(model)

    try:
        running_data = await LlamaSwapClient().get_running_models()
        running_list = running_data.get("running") or []
        running_names = {
            item.get("model")
            for item in running_list
            if item.get("state") in ("running", "ready", "loading")
        }
    except Exception:
        running_names = set()
    if proxy_name in running_names:
        try:
            from backend.llama_swap_manager import get_llama_swap_manager

            await get_llama_swap_manager().unregister_model(proxy_name)
        except Exception as e:
            logger.warning(f"Failed to stop model {proxy_name}: {e}")

    await _remove_model_from_disk_and_manifests(store, model)
    store.delete_model(model_id)
    _mark_llama_swap_stale()
    return {"message": "Model quantization deleted"}
