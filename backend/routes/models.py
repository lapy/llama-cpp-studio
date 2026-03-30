from fastapi import APIRouter, Body, HTTPException, BackgroundTasks, Query
from fastapi.responses import Response
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
import json
import os
import time
import asyncio
import re
from datetime import datetime

from backend.data_store import get_store, generate_proxy_name, resolve_proxy_name
from backend.model_config import (
    config_api_response,
    effective_model_config_from_raw,
    merge_model_config_put,
    normalize_model_config,
    set_embedding_flag,
)
from backend.progress_manager import get_progress_manager
from backend.huggingface import (
    search_models,
    download_model,
    download_model_with_progress,
    set_huggingface_token,
    get_huggingface_token,
    get_model_details,
    _extract_quantization,
    get_safetensors_metadata_summary,
    list_safetensors_downloads,
    record_safetensors_download,
    list_grouped_safetensors_downloads,
    create_gguf_manifest_entry,
    get_safetensors_manifest_entries,
    MAX_ROPE_SCALING_FACTOR,
    get_model_disk_size,
    get_accurate_file_sizes,
    resolve_cached_model_path,
    get_gguf_limits_from_manifest,
    get_safetensors_limits_from_manifest,
    purge_gguf_store_model,
    purge_safetensors_repo_completely,
    delete_cached_model_file,
)
from backend.gpu_detector import get_gpu_info
from backend.gguf_reader import get_model_layer_info
from backend.logging_config import get_logger

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

# Common embedding indicators for automatic detection
EMBEDDING_PIPELINE_TAGS = {
    "text-embedding",
    "feature-extraction",
    "sentence-similarity",
}
EMBEDDING_KEYWORDS = [
    "embedding",
    "embed-",
    "text-embedding",
    "feature-extraction",
    "nomic",
    "gte-",
    "e5-",
    "bge-",
    "snowflake-arctic-embed",
    "minilm",
]


def _is_mmproj_filename(filename: Optional[str]) -> bool:
    name = (filename or "").strip().lower()
    return bool(name) and "mmproj" in name and name.endswith(".gguf")


# Lightweight cache for GPU info to avoid repeated NVML calls during rapid estimate requests
_gpu_info_cache: Dict[str, Any] = {"data": None, "timestamp": 0.0}
GPU_INFO_CACHE_TTL = 2.0  # seconds


def _looks_like_embedding_model(
    pipeline_tag: Optional[str], *name_parts: Optional[str]
) -> bool:
    """Detect embedding-capable models based on pipeline metadata or name heuristics."""
    pipeline = (pipeline_tag or "").lower()
    if pipeline in EMBEDDING_PIPELINE_TAGS:
        return True

    combined = " ".join(part for part in name_parts if part).lower()
    if not combined:
        return False
    return any(keyword in combined for keyword in EMBEDDING_KEYWORDS)


def _model_is_embedding(model: dict) -> bool:
    """Determine if a stored model should run in embedding mode."""
    config = effective_model_config_from_raw(model.get("config"))
    if config.get("embedding"):
        return True
    return _looks_like_embedding_model(
        model.get("pipeline_tag"),
        model.get("huggingface_id"),
        model.get("display_name") or model.get("name"),
        model.get("base_model_name"),
    )


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


def normalize_architecture(raw_architecture: str) -> str:
    """Normalize GGUF architecture string (stub after smart_auto removal)."""
    if not raw_architecture or not isinstance(raw_architecture, str):
        return "unknown"
    return raw_architecture.strip() or "unknown"


def detect_architecture_from_name(name: str) -> str:
    """Infer architecture from model name (stub after smart_auto removal)."""
    if not name or not isinstance(name, str):
        return "unknown"
    name_lower = name.lower()
    if "llama" in name_lower:
        return "llama"
    if "qwen" in name_lower:
        return "qwen2"
    if "mistral" in name_lower:
        return "mistral"
    if "phi" in name_lower:
        return "phi-2"
    return "unknown"

def _coerce_model_config(config_value: Optional[Any]) -> Dict[str, Any]:
    """Effective flat model config (per active engine)."""
    return effective_model_config_from_raw(config_value)


def _refresh_gguf_model_metadata(model: dict, store, gguf_path: str) -> Dict[str, Any]:
    """
    Re-read GGUF metadata from a concrete on-disk file (HF cache path) and update the store.
    """
    normalized_path = gguf_path.replace("\\", "/")
    if not normalized_path or not os.path.exists(normalized_path):
        raise FileNotFoundError("Model file not found on disk")

    layer_info = get_model_layer_info(normalized_path)
    if not layer_info:
        raise ValueError("Failed to read model metadata from file")

    raw_architecture = layer_info.get("architecture", "")
    normalized_architecture = normalize_architecture(raw_architecture)
    if not normalized_architecture or normalized_architecture == "unknown":
        normalized_architecture = detect_architecture_from_name(
            model.get("display_name") or model.get("name") or model.get("huggingface_id") or ""
        )

    update_fields = {}
    if (
        normalized_architecture
        and normalized_architecture != "unknown"
        and normalized_architecture != model.get("model_type")
    ):
        update_fields["model_type"] = normalized_architecture

    if update_fields:
        store.update_model(model["id"], update_fields)

    return {
        "updated_fields": update_fields,
        "metadata": {
            "architecture": normalized_architecture,
            "layer_count": layer_info.get("layer_count", 0),
            "context_length": layer_info.get("context_length", 0),
            "parameter_count": layer_info.get("parameter_count"),
            "vocab_size": layer_info.get("vocab_size", 0),
            "embedding_length": layer_info.get("embedding_length", 0),
            "attention_head_count": layer_info.get("attention_head_count", 0),
            "attention_head_count_kv": layer_info.get("attention_head_count_kv", 0),
            "block_count": layer_info.get("block_count", 0),
            "is_moe": layer_info.get("is_moe", False),
            "expert_count": layer_info.get("expert_count", 0),
            "experts_used_count": layer_info.get("experts_used_count", 0),
        },
    }


async def _collect_safetensors_runtime_metadata(
    huggingface_id: str, filename: str
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[int]]:
    """
    Gather repository metadata and safetensors tensor summaries for manifest/config defaults.
    """
    metadata: Dict[str, Any] = {}
    tensor_summary: Dict[str, Any] = {}
    max_context_length: Optional[int] = None

    def _coerce_positive_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
        if isinstance(value, str):
            cleaned = value.replace(",", "").strip()
            match = re.search(r"\d+", cleaned)
            if match:
                try:
                    candidate = int(match.group())
                    return candidate if candidate > 0 else None
                except ValueError:
                    return None
        return None

    def _coerce_positive_float(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace(",", "").strip()
            try:
                candidate = float(cleaned)
                return candidate if candidate > 0 else None
            except ValueError:
                return None
        return None

    try:
        details = await get_model_details(huggingface_id)
        config_data = details.get("config", {}) if isinstance(details, dict) else {}

        # Only use model_max_length and max_position_embeddings - no fishing for other values
        config_sources = config_data if isinstance(config_data, dict) else {}

        # Extract only the two specific fields we care about
        model_max_length = _coerce_positive_int(details.get("model_max_length"))
        max_position_embeddings = _coerce_positive_int(
            config_sources.get("max_position_embeddings")
        )

        # Use model_max_length if available, otherwise max_position_embeddings
        max_context_length = model_max_length or max_position_embeddings

        metadata = {
            "architecture": details.get("architecture"),
            "base_model": details.get("base_model"),
            "pipeline_tag": details.get("pipeline_tag"),
            "parameters": details.get("parameters"),
            "model_max_length": model_max_length,  # Store explicitly
            "config": config_data,  # Contains max_position_embeddings
            "language": details.get("language"),
            "license": details.get("license"),
        }
        if max_context_length:
            metadata["max_context_length"] = max_context_length

        # Fetch tokenizer_config.json to get model_max_length for RoPE scaling clamp
        try:
            from backend.huggingface import _get_tokenizer_config

            tokenizer_config = _get_tokenizer_config(huggingface_id)
            if tokenizer_config:
                if "tokenizer_config" not in metadata:
                    metadata["tokenizer_config"] = tokenizer_config
                # Extract model_max_length (used to clamp RoPE scaling)
                tokenizer_max = None
                for key in ("model_max_length", "max_len", "max_length"):
                    candidate = _coerce_positive_int(tokenizer_config.get(key))
                    if candidate:
                        tokenizer_max = candidate
                        break
                if tokenizer_max:
                    metadata["model_max_length"] = tokenizer_max
        except Exception as exc:
            logger.debug(
                f"Failed to fetch tokenizer_config for {huggingface_id}: {exc}"
            )
    except Exception as exc:
        logger.warning(f"Failed to collect model details for {huggingface_id}: {exc}")

    try:
        safetensors_meta = await get_safetensors_metadata_summary(huggingface_id)
        if safetensors_meta:
            matching_file = next(
                (
                    entry
                    for entry in safetensors_meta.get("files", [])
                    if entry.get("filename") == filename
                ),
                None,
            )
            if matching_file:
                tensor_summary = {
                    "tensor_count": matching_file.get("tensor_count"),
                    "dtype_counts": matching_file.get("dtype_counts"),
                }
    except Exception as exc:
        logger.warning(
            f"Failed to collect safetensors metadata for {huggingface_id}/{filename}: {exc}"
        )

    return metadata or {}, tensor_summary or {}, max_context_length


async def _save_safetensors_download(
    store,
    huggingface_id: str,
    filename: str,
    file_path: str,
    file_size: int,
    pipeline_tag: Optional[str] = None,
) -> dict:
    """
    Persist safetensors download information using a single logical model entry per repo.
    Returns the model dict with "id" (string, YAML model id).
    """
    safetensors_metadata, tensor_summary, max_context = (
        await _collect_safetensors_runtime_metadata(huggingface_id, filename)
    )
    detected_pipeline = pipeline_tag or safetensors_metadata.get("pipeline_tag")
    is_embedding_like = _looks_like_embedding_model(
        detected_pipeline, huggingface_id, filename
    )
    model_id = huggingface_id.replace("/", "--")
    model_record = store.get_model(model_id)

    if not model_record:
        from datetime import timezone as _tz
        # Safetensors-backed models are treated as a single logical entity per
        # Hugging Face repo. Derive base name and type from the repo id, not the
        # shard filename.
        repo_name = huggingface_id.split("/")[-1] if isinstance(huggingface_id, str) else ""
        base_model_name = repo_name or extract_base_model_name(filename)
        model_type = extract_model_type(huggingface_id or repo_name or filename)
        model_record = {
            "id": model_id,
            "huggingface_id": huggingface_id,
            "display_name": base_model_name,
            "base_model_name": base_model_name,
            "file_size": file_size,
            "model_type": model_type,
            "downloaded_at": datetime.now(_tz.utc).isoformat(),
            "format": "safetensors",
            "pipeline_tag": detected_pipeline,
            "config": (
                set_embedding_flag({}, model_format="safetensors") if is_embedding_like else {}
            ),
        }
        store.add_model(model_record)
    else:
        updates = {}
        if not model_record.get("pipeline_tag") and detected_pipeline:
            updates["pipeline_tag"] = detected_pipeline
        if is_embedding_like and not effective_model_config_from_raw(
            model_record.get("config")
        ).get("embedding"):
            updates["config"] = set_embedding_flag(
                model_record.get("config"), model_format="safetensors"
            )
        if updates:
            store.update_model(model_id, updates)
        model_record = store.get_model(model_id) or model_record

    record_safetensors_download(
        huggingface_id=huggingface_id,
        filename=filename,
        file_path=file_path,
        file_size=file_size,
        metadata=safetensors_metadata,
        tensor_summary=tensor_summary,
        model_id=model_record.get("id"),
    )
    # Re-aggregate total safetensors repo size AFTER recording this file into the manifest.
    # This prevents `file_size` from lagging behind during multi-file/safetensors bundle downloads.
    try:
        from backend.huggingface import list_safetensors_downloads

        manifests = list_safetensors_downloads()
        total_size = 0
        for manifest in manifests:
            if manifest.get("huggingface_id") == huggingface_id:
                total_size = sum((f.get("file_size") or 0) for f in manifest.get("files", []))
                break

        if total_size and total_size != (model_record.get("file_size") or 0):
            store.update_model(model_id, {"file_size": total_size})
            model_record = store.get_model(model_id) or model_record
    except Exception as exc:
        logger.warning(f"Failed to aggregate safetensors file sizes for {huggingface_id}: {exc}")

    logger.info(f"Safetensors download recorded for {huggingface_id}/{filename} (model_id={model_record.get('id')})")
    return model_record


def _get_safetensors_model(store, model_id: str) -> dict:
    model = _get_model_or_404(store, model_id)
    model_format = (model.get("model_format") or model.get("format") or "gguf").lower()
    if model_format != "safetensors":
        raise HTTPException(status_code=400, detail="Model is not a safetensors download")
    # Safetensors models are treated as repo-level entities; concrete file paths
    # are tracked in the safetensors manifest, not on the model record itself.
    return dict(model)


def _load_manifest_entry_for_model(model: dict) -> Dict[str, Any]:
    """Load unified manifest for a safetensors model (repo-level, not per-file)."""
    manifest = get_safetensors_manifest_entries(model.get("huggingface_id"))
    if not manifest:
        raise HTTPException(status_code=404, detail="Safetensors manifest not found")
    return manifest


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = int(value)
        # Sanity check: cap at reasonable maximum (1 billion tokens)
        # This prevents corrupted metadata from causing display issues
        MAX_REASONABLE_VALUE = 1_000_000_000
        if value > MAX_REASONABLE_VALUE:
            logger.warning(
                f"Unreasonably large value detected: {value}, capping at {MAX_REASONABLE_VALUE}"
            )
            return None
        return value if value > 0 else None
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            candidate = int(cleaned)
            # Sanity check: cap at reasonable maximum
            MAX_REASONABLE_VALUE = 1_000_000_000
            if candidate > MAX_REASONABLE_VALUE:
                logger.warning(
                    f"Unreasonably large value detected: {candidate}, capping at {MAX_REASONABLE_VALUE}"
                )
                return None
            return candidate if candidate > 0 else None
        except ValueError:
            return None
    return None


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

class BundleProgressProxy:
    """Proxy progress manager that converts per-file progress into bundle-level updates."""

    def __init__(
        self,
        base_manager,
        master_task_id: str,
        bytes_completed: int,
        total_bytes: int,
        file_index: int,
        total_files: int,
        current_filename: str,
        huggingface_id: str = None,
        bundle_format: str = "safetensors-bundle",
    ):
        self._manager = base_manager
        self.master_task_id = master_task_id
        self.base_bytes = bytes_completed
        self.total_bytes = total_bytes or 0
        self.file_index = file_index
        self.total_files = total_files
        self.current_filename = current_filename
        self.completed_files = file_index
        self.huggingface_id = huggingface_id
        self.bundle_format = bundle_format

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
        model_format: str = "gguf",
        huggingface_id: str = None,
        **kwargs,
    ):
        aggregate_downloaded = self.base_bytes + bytes_downloaded

        # If we know the bundle total (from size hints), compute true aggregate progress.
        # Otherwise, fall back to per-file progress reported by the underlying downloader.
        if self.total_bytes > 0:
            bundle_total = self.total_bytes
            aggregate_progress = int((aggregate_downloaded / bundle_total) * 100)
        else:
            bundle_total = aggregate_downloaded or 0
            aggregate_progress = progress
        # files_completed should be file_index + 1 (1-based counting)
        # When progress < 100: currently downloading file (file_index + 1)
        # When progress >= 100: file is complete, so we've completed file_index + 1 files
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
            model_format=self.bundle_format,
            files_completed=files_completed,
            files_total=self.total_files,
            current_filename=self.current_filename,
            huggingface_id=self.huggingface_id,
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
active_downloads = (
    {}
)  # {task_id: {"huggingface_id": str, "filename": str, "quantization": str}}
download_lock = asyncio.Lock()


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
    from backend.engine_param_catalog import get_version_entry, registry_payload_from_entry
    from backend.engine_param_scanner import scan_engine_version
    from backend.studio_engine_fields import studio_sections_for_engine

    store = get_store()
    if engine not in ("llama_cpp", "ik_llama", "lmdeploy"):
        return registry_payload_from_entry(
            engine, None, [], has_active_engine=False
        )

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
        base_name = model.get("base_model_name") or (hf_id.split("/")[-1] if hf_id else model.get("display_name") or "unknown")
        proxy_name = resolve_proxy_name(model)
        is_active = proxy_name in running_names
        raw_state = proxy_state_by_name.get(proxy_name) if proxy_name in running_names else None
        if raw_state == "loading":
            run_state = "loading"
        elif raw_state in ("running", "ready"):
            run_state = "running"
        else:
            run_state = None
        is_embedding = _model_is_embedding(model)
        key = f"{hf_id}_{base_name}"
        if key not in grouped_models:
            author = hf_id.split("/")[0] if isinstance(hf_id, str) and "/" in hf_id else ""
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
            if model.get("pipeline_tag") and not grouped_models[key].get("pipeline_tag"):
                grouped_models[key]["pipeline_tag"] = model.get("pipeline_tag")
            if is_embedding and not grouped_models[key].get("is_embedding_model"):
                grouped_models[key]["is_embedding_model"] = True

        file_size = model.get("file_size") or 0

        grouped_models[key]["quantizations"].append({
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
        })

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
    filenames: str = Query(..., description="Comma-separated list of file paths in the repo"),
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
        if not target_model or (target_model.get("format") or target_model.get("model_format")) != "safetensors":
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
async def download_huggingface_model(
    request: dict, background_tasks: BackgroundTasks
):
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
            quantization = _extract_quantization(filename)
            model_id = f"{huggingface_id.replace('/', '--')}--{quantization}"
            if store.get_model(model_id):
                raise HTTPException(
                    status_code=400, detail="This quantization is already downloaded"
                )

        # Extract quantization for better task_id (use same function as search results)
        quantization = (
            os.path.splitext(os.path.basename(filename))[0]
            if is_mmproj_download
            else _extract_quantization(filename)
            if model_format == "gguf"
            else os.path.splitext(filename)[0]
        )

        # Generate unique task ID with quantization and milliseconds
        task_id = f"download_{model_format}_{huggingface_id.replace('/', '_')}_{quantization}_{int(time.time() * 1000)}"

        # Check if this specific file is already being downloaded
        async with download_lock:
            is_downloading = any(
                d["huggingface_id"] == huggingface_id
                and d["filename"] == filename
                and d.get("model_format", model_format) == model_format
                for d in active_downloads.values()
            )
            if is_downloading:
                raise HTTPException(
                    status_code=409,
                    detail="This quantization is already being downloaded",
                )

            # Register this download as active
            active_downloads[task_id] = {
                "huggingface_id": huggingface_id,
                "filename": filename,
                "quantization": quantization,
                "model_format": model_format,
            }

        # Start download in background with progress_manager for SSE
        pm = get_progress_manager()
        pm.create_task("download", f"Download {filename}", {"huggingface_id": huggingface_id, "filename": filename}, task_id=task_id)
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


async def download_model_task(
    huggingface_id: str,
    filename: str,
    progress_manager=None,
    task_id: str = None,
    total_bytes: int = 0,
    model_format: str = "gguf",
    pipeline_tag: Optional[str] = None,
):
    """Background task to download model with SSE progress"""
    store = get_store()

    try:
        model_record = None
        metadata_result = None
        is_mmproj_download = model_format == "gguf" and "mmproj" in filename.lower()

        if progress_manager and task_id:
            file_path, file_size = await download_model_with_progress(
                huggingface_id,
                filename,
                progress_manager,
                task_id,
                total_bytes,
                model_format,
                huggingface_id,
            )
        else:
            file_path, file_size = await download_model(
                huggingface_id, filename, model_format
            )

        if model_format == "gguf" and not is_mmproj_download:
            model_record, metadata_result = await _record_gguf_download_post_fetch(
                store,
                huggingface_id,
                filename,
                file_path,
                file_size,
                pipeline_tag=pipeline_tag,
                aggregate_size=True,
            )
        elif model_format == "gguf":
            logger.info("Downloaded standalone mmproj file for %s: %s", huggingface_id, filename)
        else:
            model_record = await _save_safetensors_download(
                store,
                huggingface_id,
                filename,
                file_path,
                file_size,
                pipeline_tag=pipeline_tag,
            )

        # Send download complete via SSE
        if progress_manager and task_id:
            progress_manager.complete_task(task_id, f"Downloaded {filename}")
            payload = {
                "type": "download_complete",
                "status": "completed",
                "huggingface_id": huggingface_id,
                "filename": filename,
                "model_format": model_format,
                "quantization": model_record.get("quantization") if model_record else None,
                "model_id": model_record.get("id") if model_record else None,
                "base_model_name": (
                    model_record.get("base_model_name") if model_record else None
                ),
                "pipeline_tag": (
                    model_record.get("pipeline_tag") if model_record else pipeline_tag
                ),
                "is_embedding_model": (
                    _model_is_embedding(model_record) if model_record else False
                ),
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata_result["metadata"] if metadata_result else None,
                "updated_fields": (
                    metadata_result["updated_fields"]
                    if isinstance(metadata_result, dict)
                    else {}
                ),
                "file_size": file_size,
                "file_path": file_path,
            }
            await progress_manager.broadcast({**payload})
            await progress_manager.send_notification(
                title="Download Complete",
                message=f"Successfully downloaded {filename} ({model_format})",
                type="success",
            )
        _mark_llama_swap_stale()

    except Exception as e:
        if progress_manager and task_id:
            progress_manager.fail_task(task_id, str(e))
            await progress_manager.send_notification(
                title="Download Failed",
                message=f"Failed to download {filename}: {str(e)}",
                type="error",
            )
    finally:
        if task_id:
            async with download_lock:
                active_downloads.pop(task_id, None)


async def _record_gguf_download_post_fetch(
    store,
    huggingface_id: str,
    filename: str,
    file_path: str,
    file_size: int,
    pipeline_tag: Optional[str] = None,
    aggregate_size: bool = True,
) -> Tuple[dict, Optional[Dict[str, Any]]]:
    """
    Shared helper to create GGUF model entries and manifest after a file has been downloaded.
    Returns (model_record dict, metadata_result).
    """
    quantization = _extract_quantization(filename)
    # Derive the base model name from the Hugging Face repo id instead of any
    # specific filename. For typical repos like "unsloth/Qwen3.5-0.8B-GGUF",
    # this yields "Qwen3.5-0.8B".
    repo_name = huggingface_id.split("/")[-1] if isinstance(huggingface_id, str) else ""
    base_model_name = repo_name
    if repo_name.endswith("-GGUF"):
        base_model_name = repo_name[: -len("-GGUF")]
    detected_pipeline = pipeline_tag
    is_embedding_like = _looks_like_embedding_model(
        detected_pipeline,
        huggingface_id,
        filename,
        base_model_name,
    )
    if not detected_pipeline and is_embedding_like:
        detected_pipeline = "text-embedding"
    metadata_result: Optional[Dict[str, Any]] = None

    model_id = f"{huggingface_id.replace('/', '--')}--{quantization}"
    model_record = store.get_model(model_id)

    if not model_record:
        from datetime import timezone as _tz
        # New GGUF records do not persist any per-file name. The model is a single
        # logical entity identified by (huggingface_id, quantization).
        model_record = {
            "id": model_id,
            "huggingface_id": huggingface_id,
            "display_name": f"{base_model_name}-{quantization}",
            "base_model_name": base_model_name,
            "file_size": file_size if aggregate_size else 0,
            "quantization": quantization,
            "model_type": extract_model_type(filename),
            "proxy_name": generate_proxy_name(huggingface_id, quantization),
            # Persist only the canonical "format" field (older rows may still have model_format).
            "format": "gguf",
            "downloaded_at": datetime.now(_tz.utc).isoformat(),
            "pipeline_tag": detected_pipeline,
            "config": (
                set_embedding_flag({}, model_format="gguf") if is_embedding_like else {}
            ),
        }
        store.add_model(model_record)
    else:
        updates = {}
        if aggregate_size and file_size and file_size > 0:
            current_size = model_record.get("file_size") or 0
            updates["file_size"] = current_size + file_size
        if not model_record.get("pipeline_tag") and detected_pipeline:
            updates["pipeline_tag"] = detected_pipeline
        if is_embedding_like:
            if not effective_model_config_from_raw(model_record.get("config")).get(
                "embedding"
            ):
                updates["config"] = set_embedding_flag(
                    model_record.get("config"), model_format="gguf"
                )
        if updates:
            store.update_model(model_id, updates)
        model_record = store.get_model(model_id) or model_record

    manifest_entry = None
    try:
        manifest_entry = await create_gguf_manifest_entry(
            model_record.get("huggingface_id"),
            file_path,
            file_size,
            model_id=model_record.get("id"),
        )
    except Exception as manifest_exc:
        logger.warning(f"Failed to record GGUF manifest entry for {filename}: {manifest_exc}")

    metadata_result = None
    try:
        metadata_result = _refresh_gguf_model_metadata(model_record, store, file_path)
    except FileNotFoundError:
        logger.warning(
            "Model file missing during metadata refresh for %s",
            model_record.get("id"),
        )
    except Exception as meta_exc:
        logger.warning(
            "Failed to refresh metadata for model %s: %s",
            model_record.get("id"),
            meta_exc,
        )

    return model_record, metadata_result


async def download_safetensors_bundle_task(
    huggingface_id: str,
    files: List[Dict[str, Any]],
    progress_manager,
    task_id: str,
    total_bundle_bytes: int = 0,
):
    store = get_store()
    try:
        total_files = len(files)
        bytes_completed = 0
        aggregate_total = total_bundle_bytes or sum(
            max(f.get("size") or 0, 0) for f in files
        )
        aggregate_total = aggregate_total or None

        for index, file_info in enumerate(files):
            filename = file_info["filename"]
            size_hint = max(file_info.get("size") or 0, 0)
            proxy = BundleProgressProxy(
                progress_manager,
                task_id,
                bytes_completed,
                aggregate_total or 0,
                index,
                total_files,
                filename,
                huggingface_id,
                "safetensors-bundle",
            )

            file_path, file_size = await download_model_with_progress(
                huggingface_id,
                filename,
                proxy,
                task_id,
                size_hint,
                "safetensors",
                huggingface_id,
            )

            if filename.endswith(".safetensors"):
                try:
                    await _save_safetensors_download(
                        store, huggingface_id, filename, file_path, file_size
                    )
                except Exception as exc:
                    logger.error(
                        f"Failed to record safetensors download for {filename}: {exc}"
                    )

            bytes_completed += file_size

        final_total = aggregate_total or bytes_completed
        await progress_manager.send_download_progress(
            task_id=task_id,
            progress=100,
            message=f"Safetensors bundle downloaded ({total_files} files)",
            bytes_downloaded=final_total,
            total_bytes=final_total,
            speed_mbps=0,
            eta_seconds=0,
            filename=files[-1]["filename"] if files else "",
            model_format="safetensors-bundle",
            files_completed=total_files,
            files_total=total_files,
            current_filename=files[-1]["filename"] if files else "",
            huggingface_id=huggingface_id,
        )
        if progress_manager:
            progress_manager.complete_task(task_id, "Safetensors bundle downloaded")
        await progress_manager.broadcast(
            {
                "type": "download_complete",
                "status": "completed",
                "huggingface_id": huggingface_id,
                "model_format": "safetensors-bundle",
                "filenames": [f["filename"] for f in files],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        _mark_llama_swap_stale()
    except Exception as exc:
        logger.error(f"Safetensors bundle download failed: {exc}")
        if progress_manager:
            await progress_manager.send_notification(
                "error",
                "Download Failed",
                f"Safetensors bundle failed: {str(exc)}",
                task_id,
            )
            progress_manager.fail_task(task_id, str(exc))
        await progress_manager.broadcast(
            {
                "type": "download_complete",
                "status": "failed",
                "huggingface_id": huggingface_id,
                "model_format": "safetensors-bundle",
                "filenames": [f["filename"] for f in files],
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(exc),
            }
        )

    finally:
        if task_id:
            async with download_lock:
                active_downloads.pop(task_id, None)


async def download_gguf_bundle_task(
    huggingface_id: str,
    quantization: str,
    files: List[Dict[str, Any]],
    progress_manager,
    task_id: str,
    total_bundle_bytes: int = 0,
    pipeline_tag: Optional[str] = None,
    projector: Optional[Dict[str, Any]] = None,
):
    store = get_store()
    try:
        total_files = len(files) + (1 if projector and projector.get("filename") else 0)
        bytes_completed = 0
        aggregate_total = total_bundle_bytes or sum(
            max(f.get("size") or 0, 0) for f in files
        )
        aggregate_total = aggregate_total or None

        # Track the total on-disk size of all GGUF shards for this quantization only
        # (projector size is stored separately on the model record).
        bundle_model_bytes = 0

        for index, file_info in enumerate(files):
            filename = file_info["filename"]
            size_hint = max(file_info.get("size") or 0, 0)
            proxy = BundleProgressProxy(
                progress_manager,
                task_id,
                bytes_completed,
                aggregate_total or 0,
                index,
                total_files,
                filename,
                huggingface_id,
                "gguf-bundle",
            )

            file_path, file_size = await download_model_with_progress(
                huggingface_id,
                filename,
                proxy,
                task_id,
                size_hint,
                "gguf",
                huggingface_id,
            )

            try:
                # For bundles, record manifest/metadata per shard but do not
                # increment the model's stored file_size here. We will set the
                # final aggregated size once at the end of the bundle download.
                await _record_gguf_download_post_fetch(
                    store,
                    huggingface_id,
                    filename,
                    file_path,
                    file_size,
                    pipeline_tag=pipeline_tag,
                    aggregate_size=False,
                )
            except Exception as exc:
                logger.error(f"Failed to record GGUF download for {filename}: {exc}")

            bytes_completed += file_size
            bundle_model_bytes += file_size

        model_id = f"{huggingface_id.replace('/', '--')}--{quantization}"
        model_record = store.get_model(model_id)

        projector_filename = (projector or {}).get("filename")
        if projector_filename and model_record:
            projector_size_hint = max(int((projector or {}).get("size") or 0), 0)
            cached_projector = resolve_cached_model_path(huggingface_id, projector_filename)
            if cached_projector and os.path.exists(cached_projector):
                try:
                    bytes_completed += os.path.getsize(cached_projector)
                except OSError:
                    bytes_completed += projector_size_hint
            else:
                proxy = BundleProgressProxy(
                    progress_manager,
                    task_id,
                    bytes_completed,
                    aggregate_total or 0,
                    len(files),
                    total_files,
                    projector_filename,
                    huggingface_id,
                    "gguf-bundle",
                )
                _, projector_file_size = await download_model_with_progress(
                    huggingface_id,
                    projector_filename,
                    proxy,
                    task_id,
                    projector_size_hint,
                    "gguf",
                    huggingface_id,
                )
                bytes_completed += projector_file_size

            store.update_model(model_id, {"mmproj_filename": projector_filename})

        # Persist the aggregated GGUF shard size on the model record once,
        # after all shards have been downloaded.
        if model_record and bundle_model_bytes > 0:
            try:
                store.update_model(model_id, {"file_size": bundle_model_bytes})
                model_record = store.get_model(model_id) or model_record
            except Exception as size_exc:
                logger.warning(
                    f"Failed to update aggregated GGUF size for {model_id}: {size_exc}"
                )

        final_total = aggregate_total or bytes_completed
        await progress_manager.send_download_progress(
            task_id=task_id,
            progress=100,
            message=f"GGUF bundle downloaded ({total_files} files)",
            bytes_downloaded=final_total,
            total_bytes=final_total,
            speed_mbps=0,
            eta_seconds=0,
            filename=files[-1]["filename"] if files else "",
            model_format="gguf-bundle",
            files_completed=total_files,
            files_total=total_files,
            current_filename=files[-1]["filename"] if files else "",
            huggingface_id=huggingface_id,
        )
        if progress_manager:
            progress_manager.complete_task(task_id, "GGUF bundle downloaded")
        await progress_manager.broadcast(
            {
                "type": "download_complete",
                "status": "completed",
                "huggingface_id": huggingface_id,
                "model_format": "gguf-bundle",
                "quantization": quantization,
                "filenames": [f["filename"] for f in files],
                "mmproj_filename": projector_filename,
                "model_id": model_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        _mark_llama_swap_stale()
    except Exception as exc:
        logger.error(f"GGUF bundle download failed: {exc}")
        if progress_manager:
            await progress_manager.send_notification(
                "error",
                "Download Failed",
                f"GGUF bundle failed: {str(exc)}",
                task_id,
            )
            progress_manager.fail_task(task_id, str(exc))
        await progress_manager.broadcast(
            {
                "type": "download_complete",
                "status": "failed",
                "huggingface_id": huggingface_id,
                "model_format": "gguf-bundle",
                "quantization": quantization,
                "filenames": [f["filename"] for f in files],
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(exc),
            }
        )
    finally:
        if task_id:
            async with download_lock:
                active_downloads.pop(task_id, None)


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

    async with download_lock:
        is_downloading = any(
            d["huggingface_id"] == huggingface_id
            and d.get("model_format") == "safetensors-bundle"
            for d in active_downloads.values()
        )
        if is_downloading:
            raise HTTPException(
                status_code=409, detail="Safetensors bundle is already being downloaded"
            )
        active_downloads[task_id] = {
            "huggingface_id": huggingface_id,
            "filename": "bundle",
            "quantization": "safetensors-bundle",
            "model_format": "safetensors-bundle",
        }

    pm = get_progress_manager()
    pm.create_task("download", f"Safetensors bundle {huggingface_id}", {"huggingface_id": huggingface_id}, task_id=task_id)
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

    async with download_lock:
        is_downloading = any(
            d["huggingface_id"] == huggingface_id
            and d.get("model_format") == "gguf-bundle"
            and d.get("quantization") == quantization
            for d in active_downloads.values()
        )
        if is_downloading:
            raise HTTPException(
                status_code=409, detail="GGUF bundle is already being downloaded"
            )
        active_downloads[task_id] = {
            "huggingface_id": huggingface_id,
            "filename": quantization,
            "quantization": quantization,
            "model_format": "gguf-bundle",
        }

    pm = get_progress_manager()
    pm.create_task("download", f"GGUF bundle {huggingface_id} ({quantization})", {"huggingface_id": huggingface_id, "quantization": quantization}, task_id=task_id)
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


# Removed duplicate extract_quantization; use `_extract_quantization` from backend.huggingface


async def download_model_projector_task(
    model_id: str,
    mmproj_filename: str,
    progress_manager,
    task_id: str,
    total_bytes: int = 0,
):
    store = get_store()
    try:
        model = store.get_model(model_id)
        if not model:
            raise RuntimeError("Model no longer exists")

        huggingface_id = model.get("huggingface_id")
        if not huggingface_id:
            raise RuntimeError("Model is missing huggingface_id")

        cached_path = resolve_cached_model_path(huggingface_id, mmproj_filename)
        if cached_path and os.path.exists(cached_path):
            file_path = cached_path
            try:
                file_size = os.path.getsize(cached_path)
            except OSError:
                file_size = max(int(total_bytes or 0), 0)
        else:
            file_path, file_size = await download_model_with_progress(
                huggingface_id,
                mmproj_filename,
                progress_manager,
                task_id,
                total_bytes,
                "gguf",
                huggingface_id,
            )

        store.update_model(model_id, {"mmproj_filename": mmproj_filename})

        if progress_manager:
            progress_manager.complete_task(task_id, f"Applied projector {mmproj_filename}")
            await progress_manager.broadcast(
                {
                    "type": "download_complete",
                    "status": "completed",
                    "huggingface_id": huggingface_id,
                    "model_format": "gguf-projector",
                    "model_id": model_id,
                    "filename": mmproj_filename,
                    "mmproj_filename": mmproj_filename,
                    "file_size": file_size,
                    "file_path": file_path,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            await progress_manager.send_notification(
                title="Projector Ready",
                message=f"Applied projector {mmproj_filename}",
                type="success",
            )
        _mark_llama_swap_stale()
    except Exception as exc:
        if progress_manager:
            progress_manager.fail_task(task_id, str(exc))
            await progress_manager.send_notification(
                title="Projector Update Failed",
                message=str(exc),
                type="error",
            )
    finally:
        if task_id:
            async with download_lock:
                active_downloads.pop(task_id, None)


@router.post("/{model_id:path}/projector")
async def update_model_projector(
    model_id: str,
    request: dict,
    background_tasks: BackgroundTasks,
):
    store = get_store()
    model = _get_model_or_404(store, model_id)
    if (model.get("format") or model.get("model_format")) != "gguf":
        raise HTTPException(status_code=400, detail="Projectors are only supported for GGUF models")

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

    task_id = f"download_projector_{model_id.replace('/', '_')}_{int(time.time() * 1000)}"
    async with download_lock:
        is_downloading = any(
            d.get("model_id") == model_id
            and d.get("filename") == mmproj_filename
            and d.get("model_format") == "gguf-projector"
            for d in active_downloads.values()
        )
        if is_downloading:
            raise HTTPException(status_code=409, detail="This projector is already being applied")
        active_downloads[task_id] = {
            "huggingface_id": huggingface_id,
            "model_id": model_id,
            "filename": mmproj_filename,
            "model_format": "gguf-projector",
        }

    pm = get_progress_manager()
    pm.create_task(
        "download",
        f"Projector {mmproj_filename}",
        {"huggingface_id": huggingface_id, "filename": mmproj_filename, "model_id": model_id},
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
    # Heuristic: treat any Qwen-family filename as "qwen" unless a more
    # specific architecture is provided by GGUF metadata later.
    elif "qwen" in filename_lower:
        return "qwen"
    return "unknown"


def extract_base_model_name(filename: str) -> str:
    """Extract base model name from filename by removing quantization"""
    import re

    # Remove file extension
    name = filename.replace(".gguf", "").replace(".safetensors", "")

    # Remove quantization patterns
    quantization_patterns = [
        r"IQ\d+_[A-Z]+",  # IQ1_S, IQ2_M, etc.
        r"Q\d+_K_[A-Z]+",  # Q4_K_M, Q8_0, etc.
        r"Q\d+_[A-Z]+",  # Q4_0, Q5_1, etc.
        r"Q\d+[K_]?[A-Z]*",  # Q2_K, Q6_K, etc.
        r"Q\d+",  # Q4, Q8, etc.
    ]

    for pattern in quantization_patterns:
        name = re.sub(pattern, "", name)

    # Clean up any trailing underscores or dots
    name = name.rstrip("._")

    return name if name else filename


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
                hf_max = details.get("model_max_length") or config.get("max_position_embeddings")
                if isinstance(hf_max, (int, float)) and hf_max > 0:
                    max_ctx = int(hf_max)
            if layer_count is None:
                for key in ("num_hidden_layers", "n_layer", "num_layers"):
                    val = config.get(key)
                    if isinstance(val, (int, float)) and val > 0:
                        layer_count = int(val) + 1  # + output head for n_gpu_layers hint
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
    from backend.llama_swap_config import preview_llama_swap_command_for_model

    store = get_store()
    model = _get_model_or_404(store, model_id)
    return preview_llama_swap_command_for_model({**model})


@router.post("/{model_id:path}/preview-llama-swap-cmd")
async def preview_llama_swap_cmd(model_id: str, body: dict = Body(default_factory=dict)):
    """
    Return the llama-swap ``cmd`` string that would be generated for this model,
    using the same merge rules as PUT /config. Body: ``{ "engine", "engines" }`` (optional).
    """
    from backend.llama_swap_config import preview_llama_swap_command_for_model

    store = get_store()
    model = _get_model_or_404(store, model_id)
    merged = merge_model_config_put(model.get("config"), body or {})
    preview_model = {**model, "config": merged}
    return preview_llama_swap_command_for_model(preview_model)


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
    models = [m for m in store.list_models() if m.get("huggingface_id") == huggingface_id]
    if not models:
        raise HTTPException(status_code=404, detail="Model group not found")

    try:
        running_data = await LlamaSwapClient().get_running_models()
        running_list = running_data.get("running") or []
        running_names = {item.get("model") for item in running_list if item.get("state") in ("running", "ready", "loading")}
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
        running_names = {item.get("model") for item in running_list if item.get("state") in ("running", "ready", "loading")}
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
