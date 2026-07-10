from fastapi import APIRouter, Body, HTTPException, BackgroundTasks, Query, UploadFile, File
from fastapi.responses import Response
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import json
import os
import time
import asyncio
import hashlib

from backend.data_store import (
    find_swap_name_conflicts,
    get_store,
    resolve_llama_swap_id,
    resolve_proxy_name,
    resolve_routing_name,
)
from backend.engine_registry import VALID_ENGINE_IDS, active_engine_row_is_runnable
from backend.model_config import (
    config_api_response,
    effective_model_config,
    effective_model_config_from_raw,
    merge_model_config_put,
    normalize_model_config,
)
from backend.model_config_templates import (
    apply_template_to_config,
    new_template_record,
)
from backend.progress_manager import get_progress_manager
from backend.huggingface import (
    search_models,
    set_huggingface_token,
    get_huggingface_token,
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
from backend.download_task_manager import DownloadTaskManager
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
    artifact = model.get("artifact") if isinstance(model.get("artifact"), dict) else {}
    source = model.get("source") if isinstance(model.get("source"), dict) else {}
    package_kind = str(artifact.get("package_kind") or model.get("package_kind") or "")
    compatible = set(model.get("compatible_engines") or [])
    if package_kind == "prepared_bundle" and (
        "audio_cpp" in compatible
        or str(source.get("provider") or "") == "audio_cpp"
    ):
        from backend.audio_cpp_manager import get_audio_cpp_manager
        from backend.utils.fs_ops import robust_rmtree

        managed_root = os.path.realpath(get_audio_cpp_manager().models_dir)
        bundle_path = os.path.realpath(
            str(
                artifact.get("bundle_path")
                or model.get("bundle_path")
                or artifact.get("path")
                or model.get("local_path")
                or ""
            )
        )
        if (
            not bundle_path
            or bundle_path == managed_root
            or os.path.commonpath([managed_root, bundle_path]) != managed_root
        ):
            raise HTTPException(
                status_code=400,
                detail="Refusing to delete an audio bundle outside managed storage",
            )
        if os.path.isdir(bundle_path):
            robust_rmtree(bundle_path)
    elif fmt == "safetensors" and hf_id:
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


_param_registry_cache: Dict[str, tuple] = {}
_param_registry_locks: Dict[str, asyncio.Lock] = {}
_PARAM_REGISTRY_CACHE_TTL = 30.0


def _param_registry_cache_key(
    store, engine: str, model_id: Optional[str] = None
) -> str:
    active = (
        store.get_active_engine_version(engine)
        if engine in VALID_ENGINE_IDS
        else None
    )
    version = (active or {}).get("version") or ""
    catalog_mtime = 0.0
    config_dir = getattr(store, "_config_dir", None)
    if config_dir:
        try:
            catalog_mtime = os.path.getmtime(
                os.path.join(config_dir, "engine_params_catalog.yaml")
            )
        except OSError:
            pass
    model_fingerprint = ""
    if engine == "audio_cpp" and model_id and active:
        model = store.get_model(str(model_id))
        if model:
            try:
                from backend.engine_param_scanner import (
                    audio_cpp_model_profile_fingerprint,
                )

                model_fingerprint = audio_cpp_model_profile_fingerprint(active, model)
            except Exception:
                model_fingerprint = str(model_id)
    return f"{engine}:{version}:{model_fingerprint}:{catalog_mtime}"


def _build_param_registry_payload(
    store, engine: str, model_id: Optional[str] = None, force_rescan: bool = False
) -> dict:
    from backend.engine_param_catalog import (
        get_version_entry,
        registry_payload_from_entry,
    )
    from backend.studio_engine_fields import studio_sections_for_engine

    if engine not in VALID_ENGINE_IDS:
        return registry_payload_from_entry(engine, None, [], has_active_engine=False)

    studio = studio_sections_for_engine(engine)
    active = store.get_active_engine_version(engine)
    has_active = active_engine_row_is_runnable(engine, active)
    entry = None
    profile = None
    warnings: List[str] = []
    if active and active.get("version"):
        entry = get_version_entry(store, engine, active["version"])
        if engine == "audio_cpp" and model_id:
            model = store.get_model(str(model_id))
            if not model:
                warnings.append("Model context was not found.")
            else:
                from backend.engine_param_scanner import scan_audio_cpp_model_profile

                profile = scan_audio_cpp_model_profile(
                    store, active, model, force=force_rescan
                )
                if profile.get("scan_error"):
                    warnings.append(str(profile["scan_error"]))

                config = normalize_model_config(model.get("config"))
                audio_config = ((config.get("engines") or {}).get("audio_cpp") or {})
                known: Dict[str, set] = {}
                for section in (profile or {}).get("sections") or []:
                    for param in section.get("params") or []:
                        known.setdefault(str(param.get("scope") or "process"), set()).add(
                            str(param.get("key") or "")
                        )
                for scope, config_key in (
                    ("load_option", "load_options"),
                    ("session_option", "session_options"),
                    ("request_option", "request_options"),
                ):
                    saved = audio_config.get(config_key)
                    if not isinstance(saved, dict):
                        continue
                    unknown = sorted(set(saved) - known.get(scope, set()))
                    if unknown:
                        warnings.append(
                            f"Preserved unknown {config_key}: {', '.join(unknown)}"
                        )

    payload = registry_payload_from_entry(
        engine,
        entry,
        studio,
        has_active_engine=has_active,
        profile=profile,
        compatibility_warnings=warnings,
    )
    if engine == "audio_cpp" and model_id:
        model = store.get_model(str(model_id))
        if model:
            config = normalize_model_config(model.get("config"))
            audio_config = ((config.get("engines") or {}).get("audio_cpp") or {})
            family = str(audio_config.get("family") or model.get("family") or "")
            task = str(audio_config.get("task") or "")
            from backend.audio_task_profiles import (
                api_endpoint_for,
                api_example_hint_for,
                is_profiled_task,
                request_defaults_key_for,
                request_field_groups_for,
                task_profile_for,
            )

            if is_profiled_task(task, family):
                payload["task_profile"] = task_profile_for(task, family)
                payload["request_field_groups"] = request_field_groups_for(task, family)
                payload["request_defaults_key"] = request_defaults_key_for(task, family)
                payload["api_endpoint"] = api_endpoint_for(task, family)
                payload["api_example_hint"] = api_example_hint_for(task, family)
    return payload


@router.get("/param-registry")
async def get_param_registry_endpoint(
    engine: str = "llama_cpp",
    model_id: Optional[str] = None,
    rescan: bool = False,
):
    """Return param definitions from ``engine_params_catalog.yaml`` plus studio-only fields (read-only)."""
    store = get_store()
    cache_key = _param_registry_cache_key(store, engine, model_id)
    now = time.monotonic()
    cached = _param_registry_cache.get(cache_key)
    if not rescan and cached and now - cached[1] < _PARAM_REGISTRY_CACHE_TTL:
        return cached[0]

    lock = _param_registry_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        now = time.monotonic()
        cached = _param_registry_cache.get(cache_key)
        if not rescan and cached and now - cached[1] < _PARAM_REGISTRY_CACHE_TTL:
            return cached[0]
        payload = await asyncio.to_thread(
            _build_param_registry_payload, store, engine, model_id, rescan
        )
    _param_registry_cache[cache_key] = (payload, now)
    return payload


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
        source = model.get("source") if isinstance(model.get("source"), dict) else {}
        hf_id = model.get("huggingface_id") or source.get("id") or ""
        base_name = model.get("base_model_name") or (
            hf_id.split("/")[-1] if hf_id else model.get("display_name") or "unknown"
        )
        llama_swap_id = resolve_llama_swap_id(model)
        proxy_name = llama_swap_id
        is_active = llama_swap_id in running_names
        raw_state = (
            proxy_state_by_name.get(llama_swap_id)
            if llama_swap_id in running_names
            else None
        )
        if raw_state == "loading":
            run_state = "loading"
        elif raw_state in ("running", "ready"):
            run_state = "running"
        else:
            run_state = None
        is_embedding = model_is_embedding(model)
        source_provider = source.get("provider") or (
            "huggingface" if model.get("huggingface_id") else "local"
        )
        key = f"{source_provider}_{hf_id}_{base_name}"
        if key not in grouped_models:
            author = (
                hf_id.split("/")[0] if isinstance(hf_id, str) and "/" in hf_id else ""
            )
            grouped_models[key] = {
                "base_model_name": base_name,
                "huggingface_id": hf_id,
                "source": source,
                "model_type": model.get("model_type"),
                "family": model.get("family"),
                "tasks": list(model.get("tasks") or []),
                "input_modalities": list(model.get("input_modalities") or []),
                "output_modalities": list(model.get("output_modalities") or []),
                "capabilities": model.get("capabilities") or {},
                "compatible_engines": list(model.get("compatible_engines") or []),
                "package_kind": (model.get("artifact") or {}).get("package_kind")
                if isinstance(model.get("artifact"), dict)
                else None,
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
                "format": model.get("format")
                or model.get("model_format")
                or (
                    (model.get("artifact") or {}).get("format")
                    if isinstance(model.get("artifact"), dict)
                    else None
                )
                or "gguf",
                "artifact": model.get("artifact") or {},
                "source": source,
                "family": model.get("family"),
                "tasks": list(model.get("tasks") or []),
                "input_modalities": list(model.get("input_modalities") or []),
                "output_modalities": list(model.get("output_modalities") or []),
                "capabilities": model.get("capabilities") or {},
                "compatible_engines": list(model.get("compatible_engines") or []),
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
                "llama_swap_id": llama_swap_id,
                "routing_name": resolve_routing_name(model),
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


@router.post("/downloads/cancel")
async def cancel_download(payload: dict = Body(...)):
    """Request cancellation of an in-flight model download."""
    task_id = (payload or {}).get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    return DownloadTaskManager.cancel(str(task_id))


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


_saved_cmd_cache: Dict[str, tuple] = {}
_SAVED_CMD_CACHE_TTL = 60.0


def _safe_mtime(path: Optional[str]) -> float:
    if not path:
        return 0.0
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _saved_cmd_cache_key(store, model: Dict[str, Any]) -> str:
    config_dir = getattr(store, "_config_dir", "")
    payload = {
        "config_dir": os.path.abspath(config_dir) if config_dir else "",
        "model": model,
        "engines_mtime": _safe_mtime(
            os.path.join(config_dir, "engines.yaml") if config_dir else None
        ),
        "catalog_mtime": _safe_mtime(
            os.path.join(config_dir, "engine_params_catalog.yaml")
            if config_dir
            else None
        ),
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _local_model_limits(model: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """
    Local runtime limits for UI hints.

    Best-effort and deliberately offline: GGUF limits come from metadata captured
    from the local GGUF file into the manifest; safetensors limits come from
    config metadata captured at download time. If metadata is missing, return
    nulls rather than doing hidden network work on page load.
    """
    stored_ctx = model.get("max_context_length") or model.get("context_length")
    stored_layers = model.get("layer_count")
    if isinstance(stored_ctx, (int, float)) and stored_ctx > 0 and isinstance(
        stored_layers, (int, float)
    ) and stored_layers > 0:
        return {
            "max_context_length": int(stored_ctx),
            "layer_count": int(stored_layers),
        }

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

    return {"max_context_length": max_ctx, "layer_count": layer_count}


def _model_config_response(model: Dict[str, Any]) -> Dict[str, Any]:
    payload = config_api_response(normalize_model_config(model.get("config")))
    payload["runtime_limits"] = _local_model_limits(model)
    return payload


def _validate_model_runtime_config(store, model: dict, normalized: dict) -> None:
    if effective_model_config(normalized).get("engine") != "audio_cpp":
        return
    from backend.audio_model_config import validate_audio_model_config

    try:
        validate_audio_model_config(store, model, normalized)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{model_id:path}/limits")
async def get_model_limits(model_id: str):
    store = get_store()
    model = _get_model_or_404(store, model_id)
    return _local_model_limits(model)


@router.get("/{model_id:path}/config")
async def get_model_config(model_id: str):
    """Get model's llama.cpp configuration"""
    store = get_store()
    model = _get_model_or_404(store, model_id)
    return _model_config_response(model)


def _audio_model_bundle_root_or_400(model: dict) -> str:
    from backend.reference_audio import get_audio_model_bundle_root

    compatible = set(model.get("compatible_engines") or [])
    fmt = str(model.get("format") or model.get("model_format") or "").lower()
    if "audio_cpp" not in compatible and fmt != "audio_cpp":
        raise HTTPException(
            status_code=400,
            detail="Reference audio is only supported for audio.cpp models",
        )
    return get_audio_model_bundle_root(model)


def _reference_audio_usage(model: dict) -> Dict[str, List[str]]:
    from backend.reference_audio import find_config_references

    effective = effective_model_config(model.get("config") or {})
    usage: Dict[str, List[str]] = {}
    for entry in _list_reference_audio_entries(model):
        path = entry["path"]
        refs = find_config_references(effective, path)
        if refs:
            usage[path] = refs
    return usage


def _list_reference_audio_entries(model: dict) -> List[Dict[str, Any]]:
    from backend.reference_audio import list_reference_audio

    bundle_root = _audio_model_bundle_root_or_400(model)
    return list_reference_audio(bundle_root, storage_key=model.get("id"))


@router.get("/{model_id:path}/reference-audio")
async def list_model_reference_audio(model_id: str):
    """List WAV reference clips stored under the per-model data refs/ directory."""
    store = get_store()
    model = _get_model_or_404(store, model_id)
    entries = _list_reference_audio_entries(model)
    usage = _reference_audio_usage(model)
    for entry in entries:
        entry["used_by"] = usage.get(entry["path"], [])
    return {"items": entries}


@router.post("/{model_id:path}/reference-audio")
async def upload_model_reference_audio(
    model_id: str,
    file: UploadFile = File(...),
):
    """Upload a reference WAV into the per-model data refs/ directory."""
    store = get_store()
    model = _get_model_or_404(store, model_id)
    bundle_root = _audio_model_bundle_root_or_400(model)
    content = await file.read()
    from backend.reference_audio import save_reference_audio

    entry = save_reference_audio(
        bundle_root,
        filename=file.filename or "reference.wav",
        content=content,
        storage_key=model.get("id"),
    )
    _mark_llama_swap_stale()
    return entry


@router.delete("/{model_id:path}/reference-audio/{filename}")
async def delete_model_reference_audio(model_id: str, filename: str):
    """Delete a reference WAV from the per-model data refs/ directory."""
    store = get_store()
    model = _get_model_or_404(store, model_id)
    bundle_root = _audio_model_bundle_root_or_400(model)
    effective = effective_model_config(model.get("config") or {})
    from backend.reference_audio import delete_reference_audio

    delete_reference_audio(
        bundle_root,
        filename=filename,
        storage_key=model.get("id"),
        effective_config=effective,
    )
    _mark_llama_swap_stale()
    return {"ok": True}


@router.put("/{model_id:path}/config")
async def update_model_config(model_id: str, config: dict):
    """Update model's llama.cpp configuration"""
    store = get_store()
    model = _get_model_or_404(store, model_id)
    merged = merge_model_config_put(model.get("config"), config)
    _validate_model_runtime_config(store, model, merged)
    eff = effective_model_config(merged)
    conflicts = find_swap_name_conflicts(store, model_id, eff)
    if conflicts:
        names = ", ".join(sorted(set(conflicts)))
        raise HTTPException(
            status_code=400,
            detail=(
                f"Routing name or alias already used by another model: {names}. "
                "Each llama-swap id and alias must be unique across the catalog."
            ),
        )
    updated_model = store.update_model(model_id, {"config": merged}) or {
        **model,
        "config": merged,
    }
    _mark_llama_swap_stale()
    return _model_config_response(updated_model)


class SaveConfigTemplateBody(BaseModel):
    name: str
    description: str = ""
    include_routing: bool = False
    engines_scope: str = "all"
    use_saved: bool = False
    config: Optional[Dict[str, Any]] = None


class ApplyConfigTemplateBody(BaseModel):
    template_id: str
    include_routing: Optional[bool] = None
    apply_engines: str = "active"
    persist: bool = False


@router.post("/{model_id:path}/config/save-template")
async def save_model_config_as_template(model_id: str, body: SaveConfigTemplateBody):
    """Snapshot this model's configuration as a reusable template."""
    store = get_store()
    model = _get_model_or_404(store, model_id)
    if body.use_saved or not body.config:
        raw = model.get("config")
    else:
        raw = merge_model_config_put(model.get("config"), body.config)
    try:
        record = new_template_record(
            name=body.name,
            description=body.description,
            config=normalize_model_config(raw),
            source_model_id=model_id,
            include_routing=body.include_routing,
            engines_scope=body.engines_scope
            if body.engines_scope in ("all", "active")
            else "all",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    store.add_config_template(record)
    return record


@router.post("/{model_id:path}/config/apply-template")
async def apply_model_config_template(model_id: str, body: ApplyConfigTemplateBody):
    """Apply a stored template to this model (form preview or persist)."""
    store = get_store()
    model = _get_model_or_404(store, model_id)
    template = store.get_config_template(body.template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    include_routing = (
        body.include_routing
        if body.include_routing is not None
        else bool(template.get("include_routing"))
    )
    apply_engines = body.apply_engines
    if apply_engines not in ("active", "all", "set_engine"):
        raise HTTPException(
            status_code=400,
            detail="apply_engines must be active, all, or set_engine",
        )
    merged = apply_template_to_config(
        model.get("config"),
        template.get("config") or {},
        include_routing=include_routing,
        apply_engines=apply_engines,
    )
    if body.persist:
        _validate_model_runtime_config(store, model, merged)
        eff = effective_model_config(merged)
        conflicts = find_swap_name_conflicts(store, model_id, eff)
        if conflicts:
            names = ", ".join(sorted(set(conflicts)))
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Routing name or alias already used by another model: {names}. "
                    "Each llama-swap id and alias must be unique across the catalog."
                ),
            )
        store.update_model(model_id, {"config": merged})
        _mark_llama_swap_stale()
    return {
        "config": config_api_response(merged),
        "persisted": bool(body.persist),
        "template_id": body.template_id,
        "template_name": template.get("name"),
    }


@router.get("/{model_id:path}/saved-llama-swap-cmd")
async def get_saved_llama_swap_cmd(model_id: str):
    """
    Return the llama-swap ``cmd`` for this model using **stored** DB config only.
    Cheap to poll: no request-body merge (unlike POST preview).
    """
    store = get_store()
    model = _get_model_or_404(store, model_id)
    cache_key = _saved_cmd_cache_key(store, model)
    now = time.monotonic()
    cached = _saved_cmd_cache.get(cache_key)
    if cached and now - cached[1] < _SAVED_CMD_CACHE_TTL:
        return cached[0]

    payload = await llama_swap_config.preview_llama_swap_command_async({**model})
    _saved_cmd_cache[cache_key] = (payload, now)
    return payload


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
    _validate_model_runtime_config(store, model, merged)
    preview_model = {**model, "config": merged}
    return await llama_swap_config.preview_llama_swap_command_async(preview_model)


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
