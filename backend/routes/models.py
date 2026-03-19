from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
import json
import os
import time
import asyncio
import re
from datetime import datetime

from backend.data_store import get_store, generate_proxy_name, resolve_proxy_name
from backend.progress_manager import get_progress_manager
from backend.huggingface import (
    search_models,
    download_model,
    download_model_with_progress,
    set_huggingface_token,
    get_huggingface_token,
    get_model_details,
    _extract_quantization,
    clear_search_cache,
    get_safetensors_metadata_summary,
    list_safetensors_downloads,
    delete_safetensors_download,
    record_safetensors_download,
    list_grouped_safetensors_downloads,
    create_gguf_manifest_entry,
    get_safetensors_manifest_entries,
    save_safetensors_manifest_entries,
    MAX_ROPE_SCALING_FACTOR,
    get_model_disk_size,
    get_accurate_file_sizes,
    resolve_cached_model_path,
    get_gguf_limits_from_manifest,
    get_safetensors_limits_from_manifest,
)
from backend.gpu_detector import get_gpu_info
from backend.gguf_reader import get_model_layer_info
from backend.logging_config import get_logger

logger = get_logger(__name__)
from backend.llama_swap_config import get_supported_flags
import psutil

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


async def _regenerate_llama_swap_config(reason: str):
    try:
        from backend.llama_swap_manager import get_llama_swap_manager

        llama_swap_manager = get_llama_swap_manager()
        await llama_swap_manager.regenerate_config_with_active_version()
        logger.info("Regenerated llama-swap config after %s", reason)
    except Exception as exc:
        logger.warning("Failed to regenerate llama-swap config after %s: %s", reason, exc)

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
    config = _coerce_model_config(model.get("config"))
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


def _get_actual_file_size(file_path: Optional[str]) -> Optional[int]:
    """Return actual file size in bytes from disk, or None if not available."""
    if not file_path:
        return None
    # For new HF-backed models we do not store paths; this helper is only used for
    # legacy/local models that still reference concrete filesystem locations.
    path = file_path.replace("\\", "/")
    if not path or not os.path.exists(path):
        return None
    try:
        real = os.path.realpath(path)
        return os.path.getsize(real if os.path.exists(real) else path)
    except OSError:
        return None


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



def _derive_hf_defaults(metadata: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    max_ctx = metadata.get("max_context_length")
    if isinstance(max_ctx, int) and max_ctx > 0:
        defaults["ctx_size"] = max_ctx

    generation_cfg = metadata.get("generation_config") or {}
    if isinstance(generation_cfg, dict):

        def _assign_numeric(src_key: str, dest_keys):
            value = generation_cfg.get(src_key)
            if isinstance(value, (int, float)):
                for dest_key in dest_keys:
                    defaults.setdefault(dest_key, value)

        _assign_numeric("temperature", ("temp", "temperature"))
        _assign_numeric("top_p", ("top_p",))
        _assign_numeric("top_k", ("top_k",))
        _assign_numeric("typical_p", ("typical_p",))
        _assign_numeric("min_p", ("min_p",))
        _assign_numeric("repetition_penalty", ("repeat_penalty",))
        _assign_numeric("presence_penalty", ("presence_penalty",))
        _assign_numeric("frequency_penalty", ("frequency_penalty",))
        _assign_numeric("seed", ("seed",))

        gen_ctx = (
            generation_cfg.get("max_length")
            or generation_cfg.get("max_position_embeddings")
            or generation_cfg.get("max_tokens")
        )
        if isinstance(gen_ctx, int) and gen_ctx > 0 and "ctx_size" not in defaults:
            defaults["ctx_size"] = gen_ctx

    return defaults


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


def _refresh_model_metadata_from_file(model: dict, store) -> Dict[str, Any]:
    """
    Re-read GGUF metadata from disk and update the model record.
    Returns metadata details for downstream consumers.
    """
    # Only supported for legacy/local models that still carry a concrete file_path.
    file_path = model.get("file_path")
    if not file_path:
        raise FileNotFoundError("Model file not found on disk")
    normalized_path = file_path.replace("\\", "/")
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
            "config": {"embedding": True} if is_embedding_like else {},
        }
        store.add_model(model_record)
    else:
        updates = {}
        if not model_record.get("pipeline_tag") and detected_pipeline:
            updates["pipeline_tag"] = detected_pipeline
        if is_embedding_like and not _coerce_model_config(model_record.get("config")).get("embedding"):
            cfg = _coerce_model_config(model_record.get("config"))
            cfg["embedding"] = True
            updates["config"] = cfg
        try:
            from backend.huggingface import list_safetensors_downloads
            manifests = list_safetensors_downloads()
            total_size = 0
            for manifest in manifests:
                if manifest.get("huggingface_id") == huggingface_id:
                    total_size = sum((f.get("file_size") or 0) for f in manifest.get("files", []))
                    break
            if total_size and total_size != (model_record.get("file_size") or 0):
                updates["file_size"] = total_size
        except Exception as exc:
            logger.warning(f"Failed to aggregate safetensors file sizes for {huggingface_id}: {exc}")
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
        huggingface_id: str = None,  # accepted for compatibility, stored on instance
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
async def get_param_registry_endpoint(engine: str = "llama_cpp"):
    """Return param definitions (basic + advanced) for config forms."""
    from backend.param_registry import get_param_registry
    return get_param_registry(engine)


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
        running_names = {item.get("model") for item in running_list if item.get("state") in ("running", "ready")}
    except Exception:
        running_names = set()

    grouped_models = {}
    for model in models:
        hf_id = model.get("huggingface_id") or ""
        base_name = model.get("base_model_name") or (hf_id.split("/")[-1] if hf_id else model.get("display_name") or "unknown")
        proxy_name = resolve_proxy_name(model)
        is_active = proxy_name in running_names
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

        # Resolve actual disk size:
        # - For HF-backed GGUF models (identified by huggingface_id + quantization),
        #   trust the aggregated file_size stored on the model record.
        # - For legacy/local models, fall back to resolving a concrete file_path.
        if (model.get("format") or model.get("model_format") or "gguf") == "gguf" and model.get("huggingface_id") and model.get("quantization"):
            file_size = model.get("file_size") or 0
        else:
            legacy_path = model.get("file_path")
            file_size = _get_actual_file_size(legacy_path) or model.get("file_size") or 0

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


@router.get("/search/{model_id}/details")
async def get_model_details_endpoint(model_id: str):
    """Get detailed model information including config and architecture"""
    try:
        details = await get_model_details(model_id)
        return details
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/safetensors/{model_id:path}/metadata")
async def get_safetensors_metadata_endpoint(model_id: str):
    """Fetch safetensors metadata on demand for a HuggingFace repo and include unified manifest details when available."""
    try:
        metadata = await get_safetensors_metadata_summary(model_id)
        # Get unified manifest for local entry details
        from backend.huggingface import get_safetensors_manifest_entries

        local_manifest = get_safetensors_manifest_entries(model_id)
        if local_manifest:
            metadata["local_manifest"] = local_manifest
            metadata["max_context_length"] = local_manifest.get(
                "max_context_length"
            ) or metadata.get("max_context_length")
        return metadata
    except RuntimeError as e:
        # Handle case where safetensors metadata is not supported
        logger.warning(f"Safetensors metadata not available for {model_id}: {e}")
        return {
            "repo_id": model_id,
            "total_files": 0,
            "total_tensors": 0,
            "dtype_totals": {},
            "files": [],
            "error": str(e),
            "cached_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(
            f"Error fetching safetensors metadata for {model_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch safetensors metadata: {str(e)}"
        )


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

        from backend.huggingface import (
            get_safetensors_manifest_entries,
            delete_safetensors_download,
        )
        manifest = get_safetensors_manifest_entries(huggingface_id)
        if not manifest or not manifest.get("files"):
            raise HTTPException(status_code=404, detail="Safetensors model not found")

        for file_entry in manifest.get("files", []):
            entry_filename = file_entry.get("filename")
            if entry_filename:
                delete_safetensors_download(huggingface_id, entry_filename)

        store.delete_model(model_id)
        return {"message": f"Safetensors model {huggingface_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/safetensors/reload-from-disk")
async def reload_safetensors_from_disk():
    """Reset all safetensors store entries and reload them from file storage."""
    try:
        from backend.huggingface import (
            SAFETENSORS_DIR,
            record_safetensors_download,
        )

        store = get_store()
        safetensors_models = [
            m for m in store.list_models()
            if (m.get("format") or m.get("model_format")) == "safetensors"
        ]
        deleted_count = len(safetensors_models)
        for model in safetensors_models:
            store.delete_model(model.get("id"))
        logger.info(f"Deleted {deleted_count} safetensors model entries from store")

        # Delete all existing manifest files to regenerate from HuggingFace with defaults
        from backend.huggingface import _get_manifest_path

        deleted_manifests = 0
        if os.path.exists(SAFETENSORS_DIR):
            for repo_dir in os.scandir(SAFETENSORS_DIR):
                if not repo_dir.is_dir():
                    continue
                repo_name = repo_dir.name
                huggingface_id = repo_name.replace("_", "/")
                manifest_path = _get_manifest_path("safetensors", huggingface_id)
                if os.path.exists(manifest_path):
                    try:
                        os.remove(manifest_path)
                        deleted_manifests += 1
                    except Exception as exc:
                        logger.warning(
                            f"Failed to delete manifest {manifest_path}: {exc}"
                        )
        logger.info(f"Deleted {deleted_manifests} safetensors manifest files")

        # Scan file storage and rebuild entries
        if not os.path.exists(SAFETENSORS_DIR):
            return {
                "message": "No safetensors directory found",
                "reloaded": 0,
                "deleted": deleted_count,
                "deleted_manifests": deleted_manifests,
            }

        reloaded_count = 0
        errors = []

        # Scan each repo directory
        for repo_dir in os.scandir(SAFETENSORS_DIR):
            if not repo_dir.is_dir():
                continue

            # Extract huggingface_id from directory name
            repo_name = repo_dir.name
            huggingface_id = repo_name.replace("_", "/")

            # Find all .safetensors files in this directory
            safetensors_files = []
            for file_entry in os.scandir(repo_dir.path):
                if file_entry.is_file() and file_entry.name.endswith(".safetensors"):
                    safetensors_files.append(
                        {
                            "filename": file_entry.name,
                            "file_path": file_entry.path,
                            "file_size": file_entry.stat().st_size,
                        }
                    )

            if not safetensors_files:
                continue

            # Process each file to rebuild store entries (one model per repo via _save_safetensors_download)
            for file_info in safetensors_files:
                try:
                    filename = file_info["filename"]
                    file_path = file_info["file_path"]
                    file_size = file_info["file_size"]
                    await _save_safetensors_download(
                        store,
                        huggingface_id,
                        filename,
                        file_path,
                        file_size,
                    )
                except Exception as exc:
                    error_msg = f"Failed to reload {huggingface_id}/{file_info.get('filename', 'unknown')}: {exc}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
            reloaded_count += 1

        result = {
            "message": f"Reloaded {reloaded_count} safetensors models from disk",
            "reloaded": reloaded_count,
            "deleted": deleted_count,
            "deleted_manifests": deleted_manifests,
        }
        if errors:
            result["errors"] = errors
            result["error_count"] = len(errors)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reload safetensors from disk: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/safetensors/{model_id:path}/metadata/regenerate")
async def regenerate_safetensors_metadata_endpoint(model_id: str):
    """Refresh safetensors metadata/manifest entries without redownloading files."""
    store = get_store()
    model = _get_safetensors_model(store, model_id)
    huggingface_id = model.get("huggingface_id")
    manifest = get_safetensors_manifest_entries(huggingface_id)
    if not manifest or not manifest.get("files"):
        raise HTTPException(
            status_code=404,
            detail="No safetensors manifest entries found for this model",
        )

    # Collect metadata for all files in the unified model
    # Note: We iterate over files to collect file-level tensor summaries,
    # but the model is treated as a unified entity (one model per repo)
    unified_metadata = {}
    max_context = 0
    files = manifest.get("files", [])

    for file_entry in files:
        filename = file_entry.get("filename")
        try:
            metadata, tensor_summary, context_len = (
                await _collect_safetensors_runtime_metadata(huggingface_id, filename)
            )
        except Exception as exc:
            logger.warning(
                f"Failed to regenerate metadata for {huggingface_id}/{filename}: {exc}"
            )
            metadata, tensor_summary, context_len = (
                manifest.get("metadata") or {},
                file_entry.get("tensor_summary") or {},
                manifest.get("max_context_length"),
            )

        # Update file-level tensor summary (file-level data for unified model)
        if tensor_summary:
            file_entry["tensor_summary"] = tensor_summary

        # Aggregate repo-level metadata (use first successful metadata)
        # All files share the same unified metadata
        if metadata and not unified_metadata:
            unified_metadata = metadata

        # Resolve context length
        resolved_context = context_len or metadata.get("max_context_length")
        if resolved_context:
            max_context = max(max_context, resolved_context)

    # Update unified manifest
    if unified_metadata:
        manifest["metadata"] = unified_metadata
    if max_context:
        manifest["max_context_length"] = max_context

    save_safetensors_manifest_entries(huggingface_id, manifest)
    return {
        "message": f"Metadata regenerated for {huggingface_id}",
        "max_context_length": max_context,
        "files": files,
    }


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
            # Persist only the canonical "format" field. "model_format" is still
            # read for backward compatibility but no longer written for new records.
            "format": "gguf",
            "downloaded_at": datetime.now(_tz.utc).isoformat(),
            "pipeline_tag": detected_pipeline,
            "config": {"embedding": True} if is_embedding_like else {},
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
            current_config = _coerce_model_config(model_record.get("config"))
            if not current_config.get("embedding"):
                current_config["embedding"] = True
                updates["config"] = current_config
        if updates:
            store.update_model(model_id, updates)
        model_record = store.get_model(model_id) or model_record

    metadata_result = None
    try:
        metadata_result = _refresh_model_metadata_from_file(model_record, store)
    except FileNotFoundError:
        logger.warning(f"Model file missing during metadata refresh for {model_record.get('id')}")
    except Exception as meta_exc:
        logger.warning(f"Failed to refresh metadata for model {model_record.get('id')}: {meta_exc}")

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
                "huggingface_id": huggingface_id,
                "model_format": "safetensors-bundle",
                "filenames": [f["filename"] for f in files],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
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
                "huggingface_id": huggingface_id,
                "model_format": "safetensors_bundle",
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
                "huggingface_id": huggingface_id,
                "model_format": "gguf-bundle",
                "quantization": quantization,
                "filenames": [f["filename"] for f in files],
                "mmproj_filename": projector_filename,
                "model_id": model_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
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
            and d.get("model_format") == "safetensors_bundle"
            for d in active_downloads.values()
        )
        if is_downloading:
            raise HTTPException(
                status_code=409, detail="Safetensors bundle is already being downloaded"
            )
        active_downloads[task_id] = {
            "huggingface_id": huggingface_id,
            "filename": "bundle",
            "quantization": "safetensors_bundle",
            "model_format": "safetensors_bundle",
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
        await _regenerate_llama_swap_config(f"projector update for {model_id}")

        if progress_manager:
            progress_manager.complete_task(task_id, f"Applied projector {mmproj_filename}")
            await progress_manager.broadcast(
                {
                    "type": "download_complete",
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
        await _regenerate_llama_swap_config(f"projector cleared for {model_id}")
        return {"message": "Projector cleared", "applied": True}

    huggingface_id = model.get("huggingface_id")
    cached_path = resolve_cached_model_path(huggingface_id, mmproj_filename)
    if cached_path and os.path.exists(cached_path):
        store.update_model(model_id, {"mmproj_filename": mmproj_filename})
        await _regenerate_llama_swap_config(f"projector update for {model_id}")
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
                        layer_count = int(val)
                        break
        except Exception:
            pass
    return {"max_context_length": max_ctx, "layer_count": layer_count}


@router.get("/{model_id:path}/config")
async def get_model_config(model_id: str):
    """Get model's llama.cpp configuration"""
    store = get_store()
    model = _get_model_or_404(store, model_id)
    return _coerce_model_config(model.get("config"))


@router.put("/{model_id:path}/config")
async def update_model_config(model_id: str, config: dict):
    """Update model's llama.cpp configuration"""
    store = get_store()
    model = _get_model_or_404(store, model_id)
    store.update_model(model_id, {"config": config})

    try:
        from backend.llama_swap_manager import get_llama_swap_manager
        llama_swap_manager = get_llama_swap_manager()
        await llama_swap_manager.regenerate_config_with_active_version()
        logger.info(
            f"Regenerated llama-swap config after updating model {model.get('display_name') or model.get('name')} configuration"
        )
    except Exception as e:
        logger.warning(f"Failed to regenerate llama-swap config after model config update: {e}")

    return {"message": "Configuration updated"}


@router.post("/{model_id:path}/start")
async def start_model(model_id: str):
    """Start model via llama-swap"""
    from backend.llama_swap_client import LlamaSwapClient

    store = get_store()
    model = _get_model_or_404(store, model_id)
    proxy_model_name = resolve_proxy_name(model)

    try:
        running_data = await LlamaSwapClient().get_running_models()
        running_list = running_data.get("running") or []
        running_names = {item.get("model") for item in running_list if item.get("state") in ("running", "ready")}
    except Exception:
        running_names = set()
    if proxy_model_name in running_names:
        raise HTTPException(status_code=400, detail="Model already running")

    try:
        await get_progress_manager().send_model_status_update(
            model_id=model_id,
            status="starting",
            details={"message": f"Starting {model.get('display_name') or model.get('name')}"},
        )
    except Exception:
        pass

    config = _coerce_model_config(model.get("config"))
    if _model_is_embedding(model) and not config.get("embedding"):
        config["embedding"] = True
        store.update_model(model_id, {"config": config})

    try:
        from backend.llama_swap_manager import get_llama_swap_manager
        llama_swap_manager = get_llama_swap_manager()
        await llama_swap_manager.regenerate_config_with_active_version()
        model_with_proxy = {**(model or {}), "proxy_name": proxy_model_name}
        await llama_swap_manager.register_model(model_with_proxy, config)
        client = LlamaSwapClient()
        client.mark_model_loading(proxy_model_name)
        await client.load_model(proxy_model_name)
    except Exception as e:
        try:
            await get_progress_manager().send_model_status_update(
                model_id=model_id,
                status="error",
                details={"message": f"Failed to start: {str(e)}"},
            )
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))

    try:
        get_progress_manager().emit("model_event", {"event": "ready", "proxy_name": proxy_model_name, "model_id": model_id, "model_name": model.get("display_name") or model.get("name")})
    except Exception:
        pass

    return {
        "model_id": model_id,
        "proxy_model_name": proxy_model_name,
        "port": 2000,
        "api_endpoint": "http://localhost:2000/v1/chat/completions",
    }


@router.post("/{model_id:path}/stop")
async def stop_model(model_id: str):
    """Stop model via llama-swap"""
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
    if proxy_name not in running_names:
        raise HTTPException(status_code=404, detail="No running instance found")

    try:
        from backend.llama_swap_manager import get_llama_swap_manager
        llama_swap_manager = get_llama_swap_manager()
        logger.info(f"Calling unregister_model with proxy_model_name: {proxy_name}")
        await llama_swap_manager.unregister_model(proxy_name)
        try:
            get_progress_manager().emit("model_event", {"event": "stopped", "proxy_name": proxy_name, "model_id": model_id})
        except Exception:
            pass
        return {"message": "Model stopped"}
    except Exception as e:
        try:
            await get_progress_manager().send_model_status_update(
                model_id=model_id,
                status="error",
                details={"message": f"Failed to stop: {str(e)}"},
            )
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel


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

        store.delete_model(model.get("id"))
        deleted_count += 1

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

    store.delete_model(model_id)
    return {"message": "Model quantization deleted"}


@router.post("/{model_id:path}/regenerate-info")
async def regenerate_model_info_endpoint(model_id: str):
    """
    Regenerate model information from GGUF metadata and update the store.
    """
    store = get_store()
    model = _get_model_or_404(store, model_id)

    try:
        metadata = _refresh_model_metadata_from_file(model, store)
        return {
            "success": True,
            "model_id": model_id,
            "updated_fields": metadata["updated_fields"],
            "metadata": metadata["metadata"],
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to regenerate model info for model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to regenerate model info: {str(e)}")


@router.get("/supported-flags")
async def get_supported_flags_endpoint():
    """Get the list of supported flags for the active llama-server binary"""
    try:
        store = get_store()
        active_version = store.get_active_engine_version("llama_cpp")
        if not active_version:
            active_version = store.get_active_engine_version("ik_llama")

        if not active_version or not active_version.get("binary_path"):
            return {
                "supported_flags": [],
                "binary_path": None,
                "error": "No active llama-cpp version found",
            }

        binary_path = active_version.get("binary_path")

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
                supported_config_keys[config_key] = any(
                    flag in supported_flags for flag in flag_options
                )

        return {
            "supported_flags": list(supported_flags),
            "supported_config_keys": supported_config_keys,
            "binary_path": binary_path,
        }

    except Exception as e:
        logger.error(f"Failed to get supported flags: {e}")
        return {
            "supported_flags": [],
            "supported_config_keys": {},
            "binary_path": None,
            "error": str(e),
        }
