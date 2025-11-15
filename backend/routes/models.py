from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
import json
import os
import time
import asyncio
from datetime import datetime

from backend.database import get_db, Model, RunningInstance, generate_proxy_name, LlamaVersion
from backend.huggingface import (
    search_models,
    download_model,
    download_model_with_websocket_progress,
    set_huggingface_token,
    get_huggingface_token,
    get_model_details,
    _extract_quantization,
    clear_search_cache,
    get_safetensors_metadata_summary,
    list_safetensors_downloads,
    delete_safetensors_download,
    record_safetensors_download,
    get_default_lmdeploy_config,
    get_safetensors_manifest_entry,
    update_lmdeploy_config,
)
from backend.smart_auto import SmartAutoConfig
from backend.smart_auto.model_metadata import get_model_metadata
from backend.smart_auto.architecture_config import normalize_architecture, detect_architecture_from_name
from backend.gpu_detector import get_gpu_info
from backend.gguf_reader import get_model_layer_info
from backend.presets import get_architecture_and_presets
from backend.llama_swap_config import get_supported_flags
from backend.logging_config import get_logger
from backend.lmdeploy_manager import get_lmdeploy_manager
from backend.lmdeploy_installer import get_lmdeploy_installer
import psutil

router = APIRouter()
logger = get_logger(__name__)

# Lightweight cache for GPU info to avoid repeated NVML calls during rapid estimate requests
_gpu_info_cache: Dict[str, Any] = {"data": None, "timestamp": 0.0}
GPU_INFO_CACHE_TTL = 2.0  # seconds


def _coerce_model_config(config_value: Optional[Any]) -> Dict[str, Any]:
    """Return a dict regardless of whether config is stored as dict or JSON string."""
    if not config_value:
        return {}
    if isinstance(config_value, dict):
        return config_value
    if isinstance(config_value, str):
        try:
            return json.loads(config_value)
        except json.JSONDecodeError:
            logger.warning("Failed to parse model config JSON; returning empty config")
            return {}
    return {}


def _refresh_model_metadata_from_file(model: Model, db: Session) -> Dict[str, Any]:
    """
    Re-read GGUF metadata from disk and update the model record similar to the refresh endpoint.
    Returns metadata details for downstream consumers.
    """
    if not model.file_path or not os.path.exists(model.file_path):
        raise FileNotFoundError("Model file not found on disk")
    
    layer_info = get_model_layer_info(model.file_path)
    if not layer_info:
        raise ValueError("Failed to read model metadata from file")
    
    raw_architecture = layer_info.get("architecture", "")
    normalized_architecture = normalize_architecture(raw_architecture)
    if not normalized_architecture or normalized_architecture == "unknown":
        normalized_architecture = detect_architecture_from_name(model.name or model.huggingface_id or "")
    
    update_fields = {}
    if normalized_architecture and normalized_architecture != "unknown" and normalized_architecture != model.model_type:
        update_fields["model_type"] = normalized_architecture
    
    file_size = os.path.getsize(model.file_path)
    if file_size != model.file_size:
        update_fields["file_size"] = file_size
    
    if update_fields:
        for key, value in update_fields.items():
            setattr(model, key, value)
        db.commit()
        db.refresh(model)
    
    return {
        "updated_fields": update_fields,
        "metadata": {
            "architecture": normalized_architecture,
            "layer_count": layer_info.get("layer_count", 0),
            "context_length": layer_info.get("context_length", 0),
            "vocab_size": layer_info.get("vocab_size", 0),
            "embedding_length": layer_info.get("embedding_length", 0),
            "attention_head_count": layer_info.get("attention_head_count", 0),
            "attention_head_count_kv": layer_info.get("attention_head_count_kv", 0),
            "block_count": layer_info.get("block_count", 0),
            "is_moe": layer_info.get("is_moe", False),
            "expert_count": layer_info.get("expert_count", 0),
            "experts_used_count": layer_info.get("experts_used_count", 0),
        }
    }


async def _collect_safetensors_runtime_metadata(
    huggingface_id: str,
    filename: str
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[int]]:
    """
    Gather repository metadata and safetensors tensor summaries for manifest/config defaults.
    """
    metadata: Dict[str, Any] = {}
    tensor_summary: Dict[str, Any] = {}
    max_context_length: Optional[int] = None
    
    try:
        details = await get_model_details(huggingface_id)
        config_data = details.get("config", {}) if isinstance(details, dict) else {}
        
        context_from_card = details.get("context_length")
        context_from_config = config_data.get("max_position_embeddings")
        max_context_length = context_from_card or context_from_config
        
        metadata = {
            "architecture": details.get("architecture"),
            "base_model": details.get("base_model"),
            "pipeline_tag": details.get("pipeline_tag"),
            "parameters": details.get("parameters"),
            "context_length": context_from_card,
            "config": config_data,
            "language": details.get("language"),
            "license": details.get("license"),
        }
        if max_context_length:
            metadata["max_context_length"] = max_context_length
    except Exception as exc:
        logger.warning(f"Failed to collect model details for {huggingface_id}: {exc}")
    
    try:
        safetensors_meta = await get_safetensors_metadata_summary(huggingface_id)
        if safetensors_meta:
            matching_file = next(
                (entry for entry in safetensors_meta.get("files", []) if entry.get("filename") == filename),
                None
            )
            if matching_file:
                tensor_summary = {
                    "tensor_count": matching_file.get("tensor_count"),
                    "dtype_counts": matching_file.get("dtype_counts"),
                }
    except Exception as exc:
        logger.warning(f"Failed to collect safetensors metadata for {huggingface_id}/{filename}: {exc}")
    
    return metadata or {}, tensor_summary or {}, max_context_length


async def _save_safetensors_download(
    db: Session,
    huggingface_id: str,
    filename: str,
    file_path: str,
    file_size: int
) -> Model:
    safetensors_metadata, tensor_summary, max_context = await _collect_safetensors_runtime_metadata(
        huggingface_id,
        filename
    )
    model_record = Model(
        name=filename.replace(".safetensors", ""),
        huggingface_id=huggingface_id,
        base_model_name=extract_base_model_name(filename),
        file_path=file_path,
        file_size=file_size,
        quantization=os.path.splitext(filename)[0],
        model_type=extract_model_type(filename),
        downloaded_at=datetime.utcnow(),
        model_format="safetensors"
    )
    db.add(model_record)
    db.commit()
    db.refresh(model_record)

    lmdeploy_config = get_default_lmdeploy_config(max_context)
    record_safetensors_download(
        huggingface_id=huggingface_id,
        filename=filename,
        file_path=file_path,
        file_size=file_size,
        metadata=safetensors_metadata,
        tensor_summary=tensor_summary,
        lmdeploy_config=lmdeploy_config,
        model_id=model_record.id
    )
    logger.info(f"Safetensors download recorded for {huggingface_id}/{filename} (model_id={model_record.id})")
    return model_record


def _get_safetensors_model(model_id: int, db: Session) -> Model:
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model_format = (model.model_format or "gguf").lower()
    if model_format != "safetensors":
        raise HTTPException(status_code=400, detail="Model is not a safetensors download")
    if not model.file_path or not os.path.exists(model.file_path):
        raise HTTPException(status_code=400, detail="Model file not found on disk")
    return model


def _load_manifest_entry_for_model(model: Model) -> Dict[str, Any]:
    filename = os.path.basename(model.file_path)
    manifest_entry = get_safetensors_manifest_entry(model.huggingface_id, filename)
    if not manifest_entry:
        raise HTTPException(status_code=404, detail="Safetensors manifest entry not found")
    return manifest_entry


def _validate_lmdeploy_config(
    new_config: Optional[Dict[str, Any]],
    manifest_entry: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge and validate LMDeploy configuration.
    """
    if new_config is not None and not isinstance(new_config, dict):
        raise HTTPException(status_code=400, detail="Config payload must be an object")
    
    metadata = manifest_entry.get("metadata") or {}
    max_context = manifest_entry.get("max_context_length") or metadata.get("max_context_length")
    stored_config = (manifest_entry.get("lmdeploy") or {}).get("config")
    baseline = stored_config or get_default_lmdeploy_config(max_context)
    merged = dict(baseline)
    if new_config:
        merged.update(new_config)
    
    def _as_int(key: str, minimum: int = 1) -> int:
        value = merged.get(key, minimum)
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{key} must be an integer")
        if value < minimum:
            value = minimum
        return value
    
    def _as_float(key: str, minimum: float, maximum: float) -> float:
        value = merged.get(key, minimum)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{key} must be a number")
        if value < minimum:
            value = minimum
        if value > maximum:
            value = maximum
        return value
    
    context_length = _as_int("context_length", minimum=1024)
    if max_context and context_length > max_context:
        context_length = max_context
    merged["context_length"] = context_length
    
    merged["tensor_parallel"] = _as_int("tensor_parallel", minimum=1)
    merged["max_batch_size"] = _as_int("max_batch_size", minimum=1)
    merged["max_batch_tokens"] = max(
        context_length,
        _as_int("max_batch_tokens", minimum=context_length)
    )
    
    merged["temperature"] = _as_float("temperature", 0.0, 2.0)
    merged["top_p"] = _as_float("top_p", 0.0, 1.0)
    merged["top_k"] = _as_int("top_k", minimum=1)
    merged["kv_cache_percent"] = _as_float("kv_cache_percent", 0.0, 100.0)
    
    tensor_split = merged.get("tensor_split") or []
    if isinstance(tensor_split, str):
        tensor_split = [part.strip() for part in tensor_split.split(",") if part.strip()]
    if tensor_split:
        cleaned_split = []
        for part in tensor_split:
            try:
                cleaned_split.append(float(part))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="tensor_split values must be numbers")
        merged["tensor_split"] = cleaned_split
    else:
        merged["tensor_split"] = []
    
    # Boolean/style cleanups
    merged["use_streaming"] = bool(merged.get("use_streaming", True))
    additional_args = merged.get("additional_args")
    if additional_args is None:
        merged["additional_args"] = ""
    elif not isinstance(additional_args, str):
        raise HTTPException(status_code=400, detail="additional_args must be a string")
    
    return merged


class BundleProgressProxy:
    """Proxy websocket manager that converts per-file progress into bundle-level updates."""

    def __init__(
        self,
        base_manager,
        master_task_id: str,
        bytes_completed: int,
        total_bytes: int,
        file_index: int,
        total_files: int,
        current_filename: str
    ):
        self._manager = base_manager
        self.master_task_id = master_task_id
        self.base_bytes = bytes_completed
        self.total_bytes = total_bytes or 0
        self.file_index = file_index
        self.total_files = total_files
        self.current_filename = current_filename
        self.completed_files = file_index

    @property
    def active_connections(self):
        return getattr(self._manager, "active_connections", [])

    async def send_download_progress(
        self,
        task_id: str,
        progress: int,
        message: str = "",
        bytes_downloaded: int = 0,
        total_bytes: int = 0,
        speed_mbps: float = 0,
        eta_seconds: int = 0,
        filename: str = "",
        model_format: str = "gguf"
    ):
        aggregate_downloaded = self.base_bytes + bytes_downloaded
        bundle_total = self.total_bytes if self.total_bytes > 0 else max(self.base_bytes + total_bytes, 1)

        aggregate_progress = int((aggregate_downloaded / bundle_total) * 100) if bundle_total else progress
        files_completed = self.file_index
        if progress >= 100:
            files_completed = min(self.file_index + 1, self.total_files)

        await self._manager.send_download_progress(
            task_id=self.master_task_id,
            progress=aggregate_progress,
            message=message or f"Downloading {self.current_filename}",
            bytes_downloaded=aggregate_downloaded,
            total_bytes=bundle_total,
            speed_mbps=speed_mbps,
            eta_seconds=eta_seconds,
            filename=self.current_filename,
            model_format="safetensors-bundle"
        )

    async def send_notification(self, *args, **kwargs):
        if hasattr(self._manager, "send_notification"):
            return await self._manager.send_notification(*args, **kwargs)

    async def broadcast(self, message: dict):
        if hasattr(self._manager, "broadcast"):
            await self._manager.broadcast(message)


async def get_cached_gpu_info() -> Dict[str, Any]:
    """Return cached GPU info when available to reduce NVML overhead."""
    now = time.monotonic()
    cached = _gpu_info_cache["data"]
    if cached is not None and now - _gpu_info_cache["timestamp"] < GPU_INFO_CACHE_TTL:
        return cached
    
    data = await get_gpu_info()
    _gpu_info_cache["data"] = data
    _gpu_info_cache["timestamp"] = now
    return data

# Global download tracking to prevent duplicates and track active downloads
active_downloads = {}  # {task_id: {"huggingface_id": str, "filename": str, "quantization": str}}
download_lock = asyncio.Lock()

class EstimationRequest(BaseModel):
    model_id: int
    config: dict
    usage_mode: Optional[str] = "single_user"


class SafetensorsBundleRequest(BaseModel):
    huggingface_id: str
    model_id: Optional[int] = None
    files: List[Dict[str, Any]]


@router.get("")
@router.get("/")
async def list_models(db: Session = Depends(get_db)):
    """List all managed models grouped by base model"""
    # Sync is_active status before returning models
    from backend.database import sync_model_active_status
    sync_model_active_status(db)
    
    models = db.query(Model).filter(
        or_(Model.model_format.is_(None), Model.model_format == "gguf")
    ).all()
    
    # Group models by huggingface_id and base_model_name
    grouped_models = {}
    for model in models:
        key = f"{model.huggingface_id}_{model.base_model_name}"
        if key not in grouped_models:
            # derive author/owner from huggingface_id
            hf_id = model.huggingface_id or ""
            author = hf_id.split('/')[0] if isinstance(hf_id, str) and '/' in hf_id else ""
            grouped_models[key] = {
                "base_model_name": model.base_model_name,
                "huggingface_id": model.huggingface_id,
                "model_type": model.model_type,
                "author": author,
                "quantizations": []
            }
        
        grouped_models[key]["quantizations"].append({
            "id": model.id,
            "name": model.name,
            "file_path": model.file_path,
            "file_size": model.file_size,
            "quantization": model.quantization,
            "downloaded_at": model.downloaded_at,
            "is_active": model.is_active,
            "has_config": bool(model.config),
            "huggingface_id": model.huggingface_id,
            "base_model_name": model.base_model_name,
            "model_type": model.model_type,
            "config": _coerce_model_config(model.config),
            "proxy_name": model.proxy_name
        })
    
    # Convert to list and sort quantizations by file size (smallest first)
    result = []
    for group in grouped_models.values():
        group["quantizations"].sort(key=lambda x: x["file_size"] or 0)
        result.append(group)
    
    # Sort groups by base model name
    result.sort(key=lambda x: x["base_model_name"])
    
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
            raise HTTPException(status_code=400, detail="model_format must be either 'gguf' or 'safetensors'")
        
        results = await search_models(query, limit, model_format=model_format)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/clear-cache")
async def clear_search_cache_endpoint():
    """Clear the search cache to force fresh results"""
    try:
        clear_search_cache()
        return {"message": "Search cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/{model_id}/details")
async def get_model_details_endpoint(model_id: str):
    """Get detailed model information including config and architecture"""
    try:
        details = await get_model_details(model_id)
        return details
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/safetensors/{model_id:path}/metadata")
async def get_safetensors_metadata_endpoint(model_id: str, filename: Optional[str] = None):
    """Fetch safetensors metadata on demand for a HuggingFace repo and include local entry details when available."""
    try:
        metadata = await get_safetensors_metadata_summary(model_id)
        if filename:
            safe_filename = os.path.basename(filename)
            entries = list_safetensors_downloads()
            local_entry = next(
                (entry for entry in entries if entry.get("huggingface_id") == model_id and entry.get("filename") == safe_filename),
                None
            )
            if local_entry:
                metadata["local_entry"] = local_entry
                metadata["max_context_length"] = local_entry.get("max_context_length") or metadata.get("max_context_length")
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/safetensors")
async def list_safetensors_models():
    """List safetensors downloads stored locally."""
    try:
        return list_safetensors_downloads()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/safetensors")
async def delete_safetensors_model(request: dict, db: Session = Depends(get_db)):
    try:
        huggingface_id = request.get("huggingface_id")
        filename = request.get("filename")
        if not huggingface_id or not filename:
            raise HTTPException(status_code=400, detail="huggingface_id and filename are required")
        # Prevent deletion while runtime is active
        active_instance = db.query(RunningInstance).filter(RunningInstance.runtime_type == "lmdeploy").first()
        target_model = db.query(Model).filter(
            Model.huggingface_id == huggingface_id,
            Model.model_format == "safetensors",
            Model.name == filename.replace(".safetensors", "")
        ).first()
        if active_instance and target_model and active_instance.model_id == target_model.id:
            raise HTTPException(status_code=400, detail="Cannot delete a model currently served by LMDeploy")
        
        deleted_entry = delete_safetensors_download(huggingface_id, filename)
        if not deleted_entry:
            raise HTTPException(status_code=404, detail="Safetensors file not found")
        
        model_id = deleted_entry.get("model_id") if isinstance(deleted_entry, dict) else None
        model_record = None
        if model_id:
            model_record = db.query(Model).filter(Model.id == model_id).first()
        if not model_record:
            model_record = target_model or db.query(Model).filter(
                Model.huggingface_id == huggingface_id,
                Model.model_format == "safetensors",
                Model.name == filename.replace(".safetensors", "")
            ).first()
        if model_record:
            db.delete(model_record)
            db.commit()
        
        return {"message": "Safetensors model deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/safetensors/{model_id}/lmdeploy/config")
async def get_lmdeploy_config_endpoint(model_id: int, db: Session = Depends(get_db)):
    """Return stored LMDeploy config and metadata for a safetensors model."""
    model = _get_safetensors_model(model_id, db)
    manifest_entry = _load_manifest_entry_for_model(model)
    metadata = manifest_entry.get("metadata") or {}
    tensor_summary = manifest_entry.get("tensor_summary") or {}
    max_context = manifest_entry.get("max_context_length") or metadata.get("max_context_length")
    config = (manifest_entry.get("lmdeploy") or {}).get("config") or get_default_lmdeploy_config(max_context)
    manager_status = get_lmdeploy_manager().status()
    installer_status = get_lmdeploy_installer().status()
    return {
        "config": config,
        "metadata": metadata,
        "tensor_summary": tensor_summary,
        "max_context_length": max_context,
        "manager": manager_status,
        "installer": installer_status,
    }


@router.put("/safetensors/{model_id}/lmdeploy/config")
async def update_lmdeploy_config_endpoint(
    model_id: int,
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Persist LMDeploy configuration changes for a safetensors model."""
    model = _get_safetensors_model(model_id, db)
    manifest_entry = _load_manifest_entry_for_model(model)
    validated_config = _validate_lmdeploy_config(request, manifest_entry)
    updated_entry = update_lmdeploy_config(
        model.huggingface_id,
        os.path.basename(model.file_path),
        validated_config
    )
    return {
        "config": updated_entry.get("lmdeploy", {}).get("config", validated_config),
        "updated_at": updated_entry.get("lmdeploy", {}).get("updated_at")
    }


@router.get("/safetensors/lmdeploy/status")
async def get_lmdeploy_status(db: Session = Depends(get_db)):
    """Return LMDeploy runtime status and running instance info."""
    installer = get_lmdeploy_installer()
    installer_status = installer.status()
    if not installer_status.get("installed"):
        raise HTTPException(
            status_code=400,
            detail="LMDeploy is not installed. Install it from the LMDeploy page before starting a runtime.",
        )
    if installer_status.get("operation"):
        raise HTTPException(
            status_code=409,
            detail="An LMDeploy install/remove operation is still running. Try again once it finishes.",
        )

    manager = get_lmdeploy_manager()
    installer = get_lmdeploy_installer()
    running_instance = db.query(RunningInstance).filter(RunningInstance.runtime_type == "lmdeploy").first()
    instance_payload = None
    if running_instance:
        instance_payload = {
            "model_id": running_instance.model_id,
            "started_at": running_instance.started_at.isoformat() if running_instance.started_at else None,
            "config": json.loads(running_instance.config) if running_instance.config else {},
        }
    return {
        "manager": manager.status(),
        "installer": installer.status(),
        "running_instance": instance_payload
    }


@router.post("/safetensors/{model_id}/lmdeploy/start")
async def start_lmdeploy_runtime(
    model_id: int,
    request: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """Start LMDeploy runtime for a safetensors model."""
    model = _get_safetensors_model(model_id, db)
    manifest_entry = _load_manifest_entry_for_model(model)
    requested_config = (request or {}).get("config") if isinstance(request, dict) else None
    validated_config = _validate_lmdeploy_config(requested_config, manifest_entry)
    
    existing_instance = db.query(RunningInstance).filter(RunningInstance.runtime_type == "lmdeploy").first()
    if existing_instance:
        if existing_instance.model_id == model.id:
            raise HTTPException(status_code=400, detail="LMDeploy is already running for this model")
        raise HTTPException(status_code=400, detail="Another safetensors model is already running via LMDeploy")
    
    manager = get_lmdeploy_manager()
    status = manager.status()
    current_instance = status.get("current_instance") or {}
    if status.get("running") and current_instance.get("model_id") not in (None, model.id):
        raise HTTPException(status_code=400, detail="LMDeploy runtime is already serving another model")
    
    filename = os.path.basename(model.file_path)
    update_lmdeploy_config(model.huggingface_id, filename, validated_config)
    
    from backend.main import websocket_manager
    await websocket_manager.send_model_status_update(
        model_id=model.id,
        status="starting",
        details={"runtime": "lmdeploy", "message": f"Starting LMDeploy for {model.name}"}
    )
    
    try:
        runtime_status = await manager.start(
            {
                "model_id": model.id,
                "huggingface_id": model.huggingface_id,
                "filename": filename,
                "file_path": model.file_path,
                "model_dir": os.path.dirname(model.file_path),
            },
            validated_config,
        )
    except Exception as exc:
        await websocket_manager.send_model_status_update(
            model_id=model.id,
            status="error",
            details={"runtime": "lmdeploy", "message": str(exc)}
        )
        raise HTTPException(status_code=500, detail=str(exc))
    
    running_instance = RunningInstance(
        model_id=model.id,
        llama_version="lmdeploy",
        proxy_model_name=f"lmdeploy::{model.id}",
        started_at=datetime.utcnow(),
        config=json.dumps({"lmdeploy": validated_config}),
        runtime_type="lmdeploy",
    )
    db.add(running_instance)
    model.is_active = True
    db.commit()
    
    from backend.unified_monitor import unified_monitor
    await unified_monitor._collect_and_send_unified_data()
    await websocket_manager.send_model_status_update(
        model_id=model.id,
        status="running",
        details={"runtime": "lmdeploy", "message": "LMDeploy is ready"}
    )
    
    return {
        "manager": runtime_status,
        "config": validated_config
    }


@router.post("/safetensors/{model_id}/lmdeploy/stop")
async def stop_lmdeploy_runtime(model_id: int, db: Session = Depends(get_db)):
    """Stop the LMDeploy runtime if it is running."""
    running_instance = db.query(RunningInstance).filter(RunningInstance.runtime_type == "lmdeploy").first()
    if not running_instance:
        raise HTTPException(status_code=404, detail="No LMDeploy runtime is active")
    if running_instance.model_id != model_id:
        raise HTTPException(status_code=400, detail="A different model is currently running in LMDeploy")
    
    manager = get_lmdeploy_manager()
    try:
        await manager.stop()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    
    db.delete(running_instance)
    model = db.query(Model).filter(Model.id == model_id).first()
    if model:
        model.is_active = False
    db.commit()
    
    from backend.unified_monitor import unified_monitor
    await unified_monitor._collect_and_send_unified_data()
    from backend.main import websocket_manager
    await websocket_manager.send_model_status_update(
        model_id=model_id,
        status="stopped",
        details={"runtime": "lmdeploy", "message": "LMDeploy runtime stopped"}
    )
    
    return {"message": "LMDeploy runtime stopped"}


@router.post("/download")
async def download_huggingface_model(
    request: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Download model from HuggingFace"""
    try:
        huggingface_id = request.get("huggingface_id")
        filename = request.get("filename")
        total_bytes = request.get("total_bytes", 0)  # Get total size from search results
        model_format = (request.get("model_format") or "gguf").lower()
        
        if not huggingface_id or not filename:
            raise HTTPException(status_code=400, detail="huggingface_id and filename are required")
        if model_format not in ("gguf", "safetensors"):
            raise HTTPException(status_code=400, detail="model_format must be either 'gguf' or 'safetensors'")
        if model_format == "gguf" and not filename.endswith(".gguf"):
            raise HTTPException(status_code=400, detail="filename must end with .gguf for GGUF downloads")
        if model_format == "safetensors" and not filename.endswith(".safetensors"):
            raise HTTPException(status_code=400, detail="filename must end with .safetensors for Safetensors downloads")
        
        # Check if this specific quantization already exists in database
        if model_format == "gguf":
            existing = db.query(Model).filter(
                Model.huggingface_id == huggingface_id,
                Model.name == filename.replace(".gguf", "")
            ).first()
            if existing:
                raise HTTPException(status_code=400, detail="This quantization is already downloaded")

        # Extract quantization for better task_id (use same function as search results)
        quantization = _extract_quantization(filename) if model_format == "gguf" else os.path.splitext(filename)[0]

        # Generate unique task ID with quantization and milliseconds
        task_id = f"download_{model_format}_{huggingface_id.replace('/', '_')}_{quantization}_{int(time.time() * 1000)}"

        # Check if this specific file is already being downloaded
        async with download_lock:
            is_downloading = any(
                d["huggingface_id"] == huggingface_id and d["filename"] == filename and d.get("model_format", model_format) == model_format
                for d in active_downloads.values()
            )
            if is_downloading:
                raise HTTPException(status_code=409, detail="This quantization is already being downloaded")
            
            # Register this download as active
            active_downloads[task_id] = {
                "huggingface_id": huggingface_id,
                "filename": filename,
                "quantization": quantization,
                "model_format": model_format
            }

        # Get websocket manager from main app
        from backend.main import websocket_manager

        # Start download in background (REMOVE db parameter, pass task_id)
        background_tasks.add_task(
            download_model_task,
            huggingface_id,
            filename,
            websocket_manager,
            task_id,
            total_bytes,
            model_format
        )
        
        return {"message": "Download started", "huggingface_id": huggingface_id, "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/huggingface-token")
async def get_huggingface_token_status():
    """Get HuggingFace API token status"""
    token = get_huggingface_token()
    env_token = os.getenv('HUGGINGFACE_API_KEY')
    
    return {
        "has_token": bool(token),
        "token_preview": f"{token[:8]}..." if token else None,
        "from_environment": bool(env_token),
        "environment_set": bool(env_token)
    }


@router.post("/huggingface-token")
async def set_huggingface_token_endpoint(request: dict):
    """Set HuggingFace API token"""
    try:
        # Check if token is set via environment variable
        env_token = os.getenv('HUGGINGFACE_API_KEY')
        if env_token:
            return {
                "message": "Token is set via environment variable and cannot be overridden via UI",
                "has_token": True,
                "from_environment": True
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def download_model_task(
    huggingface_id: str,
    filename: str,
    websocket_manager=None,
    task_id: str = None,
    total_bytes: int = 0,
    model_format: str = "gguf"
):
    """Background task to download model with WebSocket progress"""
    from backend.database import SessionLocal
    db = SessionLocal()
    
    try:
        model_record = None
        metadata_result = None

        if websocket_manager and task_id:
            file_path, file_size = await download_model_with_websocket_progress(
                huggingface_id, filename, websocket_manager, task_id, total_bytes, model_format
            )
        else:
            file_path, file_size = await download_model(huggingface_id, filename, model_format)
        
        if model_format == "gguf":
            quantization = _extract_quantization(filename)
            model_record = Model(
                name=filename.replace(".gguf", ""),
                huggingface_id=huggingface_id,
                base_model_name=extract_base_model_name(filename),
                file_path=file_path,
                file_size=file_size,
                quantization=quantization,
                model_type=extract_model_type(filename),
                proxy_name=generate_proxy_name(huggingface_id, quantization),
                downloaded_at=datetime.utcnow()
            )
            db.add(model_record)
            db.commit()
            db.refresh(model_record)

            try:
                metadata_result = _refresh_model_metadata_from_file(model_record, db)
            except FileNotFoundError:
                logger.warning(f"Model file missing during metadata refresh for {model_record.id}")
            except Exception as meta_exc:
                logger.warning(f"Failed to refresh metadata for model {model_record.id}: {meta_exc}")
        else:
            model_record = await _save_safetensors_download(
                db,
                huggingface_id,
                filename,
                file_path,
                file_size
            )
        
        # Send download complete WebSocket event (NEW)
        if websocket_manager:
            payload = {
                "type": "download_complete",
                "huggingface_id": huggingface_id,
                "filename": filename,
                "model_format": model_format,
                "quantization": model_record.quantization if model_record else None,
                "model_id": model_record.id if model_record else None,
                "base_model_name": model_record.base_model_name if model_record else None,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata_result["metadata"] if metadata_result else None,
                "updated_fields": metadata_result["updated_fields"] if isinstance(metadata_result, dict) else {},
                "file_size": file_size,
                "file_path": file_path
            }
            await websocket_manager.broadcast({
                **payload
            })
            
            await websocket_manager.send_notification(
                title="Download Complete",
                message=f"Successfully downloaded {filename} ({model_format})",
                type="success"
            )
        
    except Exception as e:
        if websocket_manager:
            await websocket_manager.send_notification(
                title="Download Failed",
                message=f"Failed to download {filename}: {str(e)}",
                type="error"
            )
    finally:
        # Cleanup: remove from active downloads and close session
        if task_id:
            async with download_lock:
                active_downloads.pop(task_id, None)
        db.close()


async def download_safetensors_bundle_task(
    huggingface_id: str,
    files: List[Dict[str, Any]],
    websocket_manager,
    task_id: str,
    total_bundle_bytes: int = 0
):
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        total_files = len(files)
        bytes_completed = 0
        aggregate_total = total_bundle_bytes or sum(max(f.get("size") or 0, 0) for f in files)
        aggregate_total = aggregate_total or None

        for index, file_info in enumerate(files):
            filename = file_info["filename"]
            size_hint = max(file_info.get("size") or 0, 0)
            proxy = BundleProgressProxy(
                websocket_manager,
                task_id,
                bytes_completed,
                aggregate_total or 0,
                index,
                total_files,
                filename
            )

            file_path, file_size = await download_model_with_websocket_progress(
                huggingface_id,
                filename,
                proxy,
                task_id,
                size_hint,
                "safetensors"
            )

            if filename.endswith(".safetensors"):
                try:
                    await _save_safetensors_download(
                        db,
                        huggingface_id,
                        filename,
                        file_path,
                        file_size
                    )
                except Exception as exc:
                    logger.error(f"Failed to record safetensors download for {filename}: {exc}")

            bytes_completed += file_size

        final_total = aggregate_total or bytes_completed
        await websocket_manager.send_download_progress(
            task_id=task_id,
            progress=100,
            message=f"Safetensors bundle downloaded ({total_files} files)",
            bytes_downloaded=final_total,
            total_bytes=final_total,
            speed_mbps=0,
            eta_seconds=0,
            filename=files[-1]["filename"] if files else "",
            model_format="safetensors-bundle"
        )

        await websocket_manager.broadcast({
            "type": "download_complete",
            "huggingface_id": huggingface_id,
            "model_format": "safetensors-bundle",
            "filenames": [f["filename"] for f in files],
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as exc:
        logger.error(f"Safetensors bundle download failed: {exc}")
        if websocket_manager:
            await websocket_manager.send_notification(
                "error",
                "Download Failed",
                f"Safetensors bundle failed: {str(exc)}",
                task_id
            )
        await websocket_manager.broadcast({
            "type": "download_complete",
            "huggingface_id": huggingface_id,
            "model_format": "safetensors_bundle",
            "filenames": [f["filename"] for f in files],
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(exc)
        })
    else:
        await websocket_manager.broadcast({
            "type": "download_complete",
            "huggingface_id": huggingface_id,
            "model_format": "safetensors_bundle",
            "filenames": [f["filename"] for f in files],
            "timestamp": datetime.utcnow().isoformat()
        })

    finally:
        if task_id:
            async with download_lock:
                active_downloads.pop(task_id, None)
        db.close()


@router.post("/safetensors/download-bundle")
async def download_safetensors_bundle(
    request: SafetensorsBundleRequest,
    background_tasks: BackgroundTasks
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

    async with download_lock:
        is_downloading = any(
            d["huggingface_id"] == huggingface_id and d.get("model_format") == "safetensors_bundle"
            for d in active_downloads.values()
        )
        if is_downloading:
            raise HTTPException(status_code=409, detail="Safetensors bundle is already being downloaded")
        active_downloads[task_id] = {
            "huggingface_id": huggingface_id,
            "filename": "bundle",
            "quantization": "safetensors_bundle",
            "model_format": "safetensors_bundle"
        }

    from backend.main import websocket_manager
    background_tasks.add_task(
        download_safetensors_bundle_task,
        huggingface_id,
        sanitized_files,
        websocket_manager,
        task_id,
        declared_total
    )

    return {
        "message": "Safetensors bundle download started",
        "huggingface_id": huggingface_id,
        "task_id": task_id
    }


# Removed duplicate extract_quantization; use `_extract_quantization` from backend.huggingface


def extract_model_type(filename: str) -> str:
    """Extract model type from filename"""
    filename_lower = filename.lower()
    if "llama" in filename_lower:
        return "llama"
    elif "mistral" in filename_lower:
        return "mistral"
    elif "codellama" in filename_lower:
        return "codellama"
    elif "gemma" in filename_lower:
        return "gemma"
    return "unknown"


def extract_base_model_name(filename: str) -> str:
    """Extract base model name from filename by removing quantization"""
    import re
    
    # Remove file extension
    name = filename.replace('.gguf', '').replace('.safetensors', '')
    
    # Remove quantization patterns
    quantization_patterns = [
        r'IQ\d+_[A-Z]+',  # IQ1_S, IQ2_M, etc.
        r'Q\d+_K_[A-Z]+',  # Q4_K_M, Q8_0, etc.
        r'Q\d+_[A-Z]+',   # Q4_0, Q5_1, etc.
        r'Q\d+[K_]?[A-Z]*',  # Q2_K, Q6_K, etc.
        r'Q\d+',  # Q4, Q8, etc.
    ]
    
    for pattern in quantization_patterns:
        name = re.sub(pattern, '', name)
    
    # Clean up any trailing underscores or dots
    name = name.rstrip('._')
    
    return name if name else filename


@router.get("/{model_id}/config")
async def get_model_config(model_id: int, db: Session = Depends(get_db)):
    """Get model's llama.cpp configuration"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return _coerce_model_config(model.config)


@router.put("/{model_id}/config")
async def update_model_config(
    model_id: int,
    config: dict,
    db: Session = Depends(get_db)
):
    """Update model's llama.cpp configuration"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model.config = config
    db.commit()
    
    # Regenerate llama-swap configuration to reflect the updated model config
    try:
        from backend.llama_swap_manager import get_llama_swap_manager
        llama_swap_manager = get_llama_swap_manager()
        await llama_swap_manager.regenerate_config_with_active_version()
        logger.info(f"Regenerated llama-swap config after updating model {model.name} configuration")
    except Exception as e:
        logger.warning(f"Failed to regenerate llama-swap config after model config update: {e}")
    
    return {"message": "Configuration updated"}


@router.post("/{model_id}/auto-config")
async def generate_auto_config(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Generate optimal configuration using Smart-Auto"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        gpu_info = await get_gpu_info()
        smart_auto = SmartAutoConfig()
        config = await smart_auto.generate_config(model, gpu_info)
        
        # Save the generated config
        model.config = config
        db.commit()
        
        # Regenerate llama-swap configuration to reflect the updated model config
        try:
            from backend.llama_swap_manager import get_llama_swap_manager
            llama_swap_manager = get_llama_swap_manager()
            await llama_swap_manager.regenerate_config_with_active_version()
            logger.info(f"Regenerated llama-swap config after auto-config for model {model.name}")
        except Exception as e:
            logger.warning(f"Failed to regenerate llama-swap config after auto-config: {e}")
        
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/smart-auto")
async def generate_smart_auto_config(
    model_id: int,
    preset: Optional[str] = None,
    usage_mode: str = "single_user",
    speed_quality: Optional[int] = None,
    use_case: Optional[str] = None,
    debug: Optional[bool] = False,
    db: Session = Depends(get_db)
):
    """
    Generate smart auto configuration with optional preset tuning, speed/quality balance, and use case.
    
    preset: Optional preset name (coding, conversational, long_context) to use as tuning parameters
    usage_mode: 'single_user' (sequential, peak KV cache) or 'multi_user' (server, typical usage)
    speed_quality: Speed/quality balance (0-100), where 0 = max speed, 100 = max quality. Default: 50 (balanced)
    use_case: Optional use case ('chat', 'code', 'creative', 'analysis') for targeted optimization
    """
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        gpu_info = await get_gpu_info()
        smart_auto = SmartAutoConfig()
        debug_map = {} if debug else None
        
        # Validate usage_mode
        if usage_mode not in ["single_user", "multi_user"]:
            usage_mode = "single_user"  # Default to single_user if invalid
        
        # Validate and normalize speed_quality (0-100, default 50)
        if speed_quality is not None:
            speed_quality = max(0, min(100, int(speed_quality)))
        else:
            speed_quality = 50
        
        # Validate use_case
        if use_case is not None and use_case not in ["chat", "code", "creative", "analysis"]:
            use_case = None  # Invalid use case, ignore it
        
        # If preset is provided, pass it to generate_config for tuning
        # Also pass speed_quality and use_case for wizard-based configuration
        config = await smart_auto.generate_config(
            model, gpu_info, 
            preset=preset, 
            usage_mode=usage_mode,
            speed_quality=speed_quality,
            use_case=use_case,
            debug=debug_map
        )
        
        if debug_map is not None:
            return {"config": config, "debug": debug_map}
        return config
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/start")
async def start_model(model_id: int, db: Session = Depends(get_db)):
    """Start model via llama-swap"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check if already running
    existing = db.query(RunningInstance).filter(RunningInstance.model_id == model_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model already running")
    
    try:
        from backend.llama_swap_manager import get_llama_swap_manager
        llama_swap_manager = get_llama_swap_manager()
        
        # Send starting notification
        from backend.main import websocket_manager
        await websocket_manager.send_model_status_update(
            model_id=model_id, status="starting",
            details={"message": f"Starting {model.name}"}
        )
        
        # Get model configuration
        config = _coerce_model_config(model.config)
        
        # Register the model with llama-swap (in memory only)
        try:
            proxy_model_name = await llama_swap_manager.register_model(model, config)
            logger.info(f"Model {model.name} registered with llama-swap as {proxy_model_name}")
                
        except ValueError as e:
            if "already registered" in str(e):
                logger.info(f"Model {model.name} already registered with llama-swap")
                # Use the stored proxy name from the database
                if not model.proxy_name:
                    raise ValueError(f"Model '{model.name}' does not have a proxy_name set")
                proxy_model_name = model.proxy_name
            else:
                raise e
        
        # Trigger model startup by making a test API request
        # This ensures llama-swap actually starts the model process
        try:
            import httpx
            import asyncio
            
            # Wait a moment for llama-swap to process the request
            await asyncio.sleep(2)
            
            async with httpx.AsyncClient() as client:
                test_request = {
                    "model": proxy_model_name,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1
                }
                # Large models can take a while to load, use a longer timeout
                timeout = 120.0  # 2 minutes for very large models
                response = await client.post(
                    "http://localhost:2000/v1/chat/completions",
                    json=test_request,
                    timeout=timeout
                )
                if response.status_code == 200:
                    logger.info(f"Model {proxy_model_name} started successfully via API trigger")
                else:
                    logger.warning(f"Model {proxy_model_name} API trigger returned status {response.status_code}")
                    # Try to get error details
                    try:
                        error_text = response.text
                        logger.warning(f"Error details: {error_text}")
                    except:
                        pass
        except Exception as e:
            import traceback
            logger.warning(f"Failed to trigger model startup via API: {e}")
            logger.debug(f"API trigger error details:\n{traceback.format_exc()}")
            # Continue anyway - the model might still work
        
        # Save to database
        running_instance = RunningInstance(
            model_id=model_id,
            llama_version=config.get("llama_version", "default"),
            proxy_model_name=proxy_model_name,
            started_at=datetime.utcnow(),
            config=json.dumps(config),
            runtime_type="llama_cpp",
        )
        db.add(running_instance)
        model.is_active = True
        db.commit()
        
        # Send success notification via unified monitoring
        from backend.unified_monitor import unified_monitor
        await unified_monitor._collect_and_send_unified_data()
        
        return {
            "model_id": model_id,
            "proxy_model_name": proxy_model_name,
            "port": 2000,
            "api_endpoint": f"http://localhost:2000/v1/chat/completions"
        }
        
    except Exception as e:
        await websocket_manager.send_model_status_update(
            model_id=model_id, status="error",
            details={"message": f"Failed to start: {str(e)}"}
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/stop")
async def stop_model(model_id: int, db: Session = Depends(get_db)):
    """Stop model via llama-swap"""
    running_instance = db.query(RunningInstance).filter(
        RunningInstance.model_id == model_id
    ).first()
    if not running_instance:
        raise HTTPException(status_code=404, detail="No running instance found")
    
    try:
        from backend.llama_swap_manager import get_llama_swap_manager
        from backend.main import websocket_manager
        llama_swap_manager = get_llama_swap_manager()
        
        # Unregister from llama-swap (it stops the process)
        if running_instance.proxy_model_name:
            logger.info(f"Calling unregister_model with proxy_model_name: {running_instance.proxy_model_name}")
            await llama_swap_manager.unregister_model(running_instance.proxy_model_name)
            logger.info("unregister_model call completed")
        
        # Update database
        db.delete(running_instance)
        model = db.query(Model).filter(Model.id == model_id).first()
        if model:
            model.is_active = False
        db.commit()
        
        # Send success notification via unified monitoring
        from backend.unified_monitor import unified_monitor
        await unified_monitor._collect_and_send_unified_data()
        
        return {"message": "Model stopped"}
        
    except Exception as e:
        await websocket_manager.send_model_status_update(
            model_id=model_id, status="error",
            details={"message": f"Failed to stop: {str(e)}"}
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vram-estimate")
async def estimate_vram_usage(
    request: EstimationRequest,
    db: Session = Depends(get_db)
):
    """Estimate VRAM usage for given configuration"""
    model = db.query(Model).filter(Model.id == request.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        gpu_info = await get_cached_gpu_info()
        smart_auto = SmartAutoConfig()
        usage_mode = request.usage_mode if request.usage_mode in ["single_user", "multi_user"] else "single_user"
        metadata = get_model_metadata(model)
        vram_estimate = smart_auto.estimate_vram_usage(
            model,
            request.config,
            gpu_info,
            usage_mode=usage_mode,
            metadata=metadata,
        )
        
        return vram_estimate
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ram-estimate")
async def estimate_ram_usage(
    request: EstimationRequest,
    db: Session = Depends(get_db)
):
    """Estimate RAM usage for given configuration"""
    try:
        model = db.query(Model).filter(Model.id == request.model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        smart_auto = SmartAutoConfig()
        usage_mode = request.usage_mode if request.usage_mode in ["single_user", "multi_user"] else "single_user"
        metadata = get_model_metadata(model)
        ram_estimate = smart_auto.estimate_ram_usage(
            model,
            request.config,
            usage_mode=usage_mode,
            metadata=metadata,
        )
        
        return ram_estimate
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quantization-sizes")
async def get_quantization_sizes(request: dict):
    """Get actual file sizes for quantizations from HuggingFace API"""
    try:
        huggingface_id = request.get("huggingface_id")
        quantizations = request.get("quantizations", {})
        
        if not huggingface_id or not quantizations:
            raise HTTPException(status_code=400, detail="huggingface_id and quantizations are required")
        # Use centralized Hugging Face service helper
        from backend.huggingface import get_quantization_sizes_from_hf
        updated_quantizations = await get_quantization_sizes_from_hf(huggingface_id, quantizations)

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
                        content_length = response.headers.get('content-length')
                        if content_length:
                            actual_size = int(content_length)
                            updated_quantizations[quant_name] = {
                                "filename": filename,
                                "size": actual_size,
                                "size_mb": round(actual_size / (1024 * 1024), 2)
                            }
                except Exception:
                    continue
        
        return {"quantizations": updated_quantizations}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel

class DeleteGroupRequest(BaseModel):
    huggingface_id: str

@router.post("/delete-group")
async def delete_model_group(
    request: DeleteGroupRequest,
    db: Session = Depends(get_db)
):
    """Delete all quantizations of a model group"""
    huggingface_id = request.huggingface_id
    models = db.query(Model).filter(Model.huggingface_id == huggingface_id).all()
    if not models:
        raise HTTPException(status_code=404, detail="Model group not found")
    
    deleted_count = 0
    for model in models:
        # Stop if running
        running_instance = db.query(RunningInstance).filter(RunningInstance.model_id == model.id).first()
        if running_instance:
            # Stop via llama-swap
            try:
                from backend.llama_swap_manager import get_llama_swap_manager
                llama_swap_manager = get_llama_swap_manager()
                if running_instance.proxy_model_name:
                    await llama_swap_manager.unregister_model(running_instance.proxy_model_name)
            except Exception as e:
                logger.warning(f"Failed to stop model {running_instance.proxy_model_name}: {e}")
            db.delete(running_instance)
        
        # Delete file
        if model.file_path and os.path.exists(model.file_path):
            os.remove(model.file_path)
        
        # Delete from database
        db.delete(model)
        deleted_count += 1
    
    db.commit()
    
    return {"message": f"Deleted {deleted_count} quantizations"}


@router.delete("/{model_id}")
async def delete_model(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Delete individual model quantization and its files"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Stop if running
    running_instance = db.query(RunningInstance).filter(RunningInstance.model_id == model_id).first()
    if running_instance:
        # Stop via llama-swap
        try:
            from backend.llama_swap_manager import get_llama_swap_manager
            llama_swap_manager = get_llama_swap_manager()
            if running_instance.proxy_model_name:
                await llama_swap_manager.unregister_model(running_instance.proxy_model_name)
        except Exception as e:
            logger.warning(f"Failed to stop model {running_instance.proxy_model_name}: {e}")
        db.delete(running_instance)
    
    # Delete file
    if model.file_path and os.path.exists(model.file_path):
        os.remove(model.file_path)
    
    # Delete from database
    db.delete(model)
    db.commit()
    
    return {"message": "Model quantization deleted"}


@router.get("/{model_id}/layer-info")
async def get_model_layer_info_endpoint(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Get model layer information from GGUF metadata"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model.file_path or not os.path.exists(model.file_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    try:
        layer_info = get_model_layer_info(model.file_path)
        if layer_info:
            return {
                "layer_count": layer_info["layer_count"],
                "architecture": layer_info["architecture"],
                "context_length": layer_info["context_length"],
                "vocab_size": layer_info["vocab_size"],
                "embedding_length": layer_info["embedding_length"],
                "attention_head_count": layer_info["attention_head_count"],
                "attention_head_count_kv": layer_info["attention_head_count_kv"],
                "block_count": layer_info["block_count"],
                "is_moe": layer_info.get("is_moe", False),
                "expert_count": layer_info.get("expert_count", 0),
                "experts_used_count": layer_info.get("experts_used_count", 0)
            }
        else:
            # Fallback to default values if metadata reading fails
            return {
                "layer_count": 32,
                "architecture": "unknown",
                "context_length": 0,
                "vocab_size": 0,
                "embedding_length": 0,
                "attention_head_count": 0,
                "attention_head_count_kv": 0,
                "block_count": 0,
                "is_moe": False,
                "expert_count": 0,
                "experts_used_count": 0
            }
    except Exception as e:
        logger.error(f"Failed to get layer info for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read model metadata: {str(e)}")


@router.get("/{model_id}/recommendations")
async def get_model_recommendations_endpoint(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Get configuration recommendations for a model based on its architecture"""
    from backend.smart_auto.recommendations import get_model_recommendations
    
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model.file_path or not os.path.exists(model.file_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    try:
        # Get layer info from GGUF metadata
        layer_info = get_model_layer_info(model.file_path)
        if not layer_info:
            # Fallback to basic defaults
            layer_info = {
                "layer_count": 32,
                "architecture": "unknown",
                "context_length": 0,
                "attention_head_count": 0,
                "embedding_length": 0
            }
        
        # Get recommendations using smart_auto with balanced preset
        recommendations = await get_model_recommendations(
            model_layer_info=layer_info,
            model_name=model.name or model.huggingface_id or "",
            file_path=model.file_path
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to get recommendations for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.get("/{model_id}/architecture-presets")
async def get_architecture_presets_endpoint(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Get architecture-specific presets for a model"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    architecture, presets = get_architecture_and_presets(model)
    return {
        "architecture": architecture,
        "presets": presets,
        "available_presets": list(presets.keys())
    }


@router.post("/{model_id}/regenerate-info")
async def regenerate_model_info_endpoint(
    model_id: int,
    db: Session = Depends(get_db)
):
    """
    Regenerate model information from GGUF metadata and update the database.
    This will re-read the model file and update architecture, layer count, and other metadata.
    """
    model = db.query(Model).filter(Model.id == model_id).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        metadata = _refresh_model_metadata_from_file(model, db)
        return {
            "success": True,
            "model_id": model_id,
            "updated_fields": metadata["updated_fields"],
            "metadata": metadata["metadata"]
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to regenerate model info for model {model_id}: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to regenerate model info: {str(e)}")


@router.get("/supported-flags")
async def get_supported_flags_endpoint(db: Session = Depends(get_db)):
    """Get the list of supported flags for the active llama-server binary"""
    try:
        # Get the active llama-cpp version
        active_version = db.query(LlamaVersion).filter(LlamaVersion.is_active == True).first()
        
        if not active_version or not active_version.binary_path:
            return {
                "supported_flags": [],
                "binary_path": None,
                "error": "No active llama-cpp version found"
            }
        
        binary_path = active_version.binary_path
        
        # Convert to absolute path if needed
        if not os.path.isabs(binary_path):
            binary_path = os.path.join("/app", binary_path.lstrip("/"))
        
        # Get supported flags
        supported_flags = get_supported_flags(binary_path)
        
        # Map config keys to their flags for easier frontend use
        param_mapping = {
            "typical_p": ["--typical"],
            "min_p": ["--min-p"],
            "tfs_z": [],  # Flag not supported in this version
            "presence_penalty": ["--presence-penalty"],
            "frequency_penalty": ["--frequency-penalty"],
            "json_schema": ["--json-schema"],
            "cache_type_v": ["--cache-type-v"],
        }
        
        # Build a map of config keys to whether they're supported
        supported_config_keys = {}
        for config_key, flag_options in param_mapping.items():
            # Empty list means flag is not supported
            if not flag_options:
                supported_config_keys[config_key] = False
            else:
                supported_config_keys[config_key] = any(flag in supported_flags for flag in flag_options)
        
        return {
            "supported_flags": list(supported_flags),
            "supported_config_keys": supported_config_keys,
            "binary_path": binary_path
        }
        
    except Exception as e:
        logger.error(f"Failed to get supported flags: {e}")
        return {
            "supported_flags": [],
            "supported_config_keys": {},
            "binary_path": None,
            "error": str(e)
        }
