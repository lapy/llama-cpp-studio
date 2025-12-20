from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
import json
import os
import time
import asyncio
import re
import httpx
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
    update_lmdeploy_config,
    list_grouped_safetensors_downloads,
    create_gguf_manifest_entry,
    get_gguf_manifest_entry,
    get_safetensors_manifest_entries,
    save_safetensors_manifest_entries,
    DEFAULT_LMDEPLOY_CONTEXT,
    MAX_LMDEPLOY_CONTEXT,
    MAX_ROPE_SCALING_FACTOR,
)
from backend.smart_auto import SmartAutoConfig
from backend.smart_auto.model_metadata import get_model_metadata
from backend.smart_auto.architecture_config import normalize_architecture, detect_architecture_from_name
from backend.gpu_detector import get_gpu_info
from backend.gguf_reader import get_model_layer_info
from backend.presets import get_architecture_and_presets
from backend.logging_config import get_logger

logger = get_logger(__name__)
from backend.llama_swap_config import get_supported_flags
from backend.logging_config import get_logger
from backend.lmdeploy_manager import get_lmdeploy_manager
from backend.lmdeploy_installer import get_lmdeploy_installer
import psutil

router = APIRouter()
logger = get_logger(__name__)

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

# Lightweight cache for GPU info to avoid repeated NVML calls during rapid estimate requests
_gpu_info_cache: Dict[str, Any] = {"data": None, "timestamp": 0.0}
GPU_INFO_CACHE_TTL = 2.0  # seconds


def _looks_like_embedding_model(
    pipeline_tag: Optional[str],
    *name_parts: Optional[str]
) -> bool:
    """Detect embedding-capable models based on pipeline metadata or name heuristics."""
    pipeline = (pipeline_tag or "").lower()
    if pipeline in EMBEDDING_PIPELINE_TAGS:
        return True
    
    combined = " ".join(part for part in name_parts if part).lower()
    if not combined:
        return False
    return any(keyword in combined for keyword in EMBEDDING_KEYWORDS)


def _model_is_embedding(model: Model) -> bool:
    """Determine if a stored model should run in embedding mode."""
    config = _coerce_model_config(model.config)
    if config.get("embedding"):
        return True
    return _looks_like_embedding_model(
        model.pipeline_tag,
        model.huggingface_id,
        model.name,
        model.base_model_name,
    )


def _normalize_model_path(file_path: Optional[str]) -> Optional[str]:
    if not file_path:
        return None
    normalized = file_path.replace("\\", "/")
    normalized = os.path.normpath(normalized)
    return normalized


def _extract_filename(file_path: Optional[str]) -> str:
    if not file_path:
        return ""
    normalized = file_path.replace("\\", "/")
    parts = normalized.split("/")
    return parts[-1] if parts else normalized


def _cleanup_model_folder_if_no_quantizations(
    db: Session,
    huggingface_id: Optional[str],
    model_format: Optional[str],
) -> None:
    """
    If there are no remaining quantizations for a given Hugging Face repo and format,
    delete the corresponding local model folder (e.g. data/models/gguf/<repo_safe>).
    """
    if not huggingface_id or not model_format:
        return

    model_format = (model_format or "").lower()
    if model_format not in ("gguf", "safetensors"):
        return

    # Check for remaining models of this repo/format, excluding any pending deletions
    remaining = db.query(Model).filter(
        Model.huggingface_id == huggingface_id,
        Model.model_format == model_format,
    ).count()
    if remaining > 0:
        return

    safe_repo = (huggingface_id or "unknown").replace("/", "_") or "unknown"
    base_dir = os.path.join("data", "models", model_format)
    repo_dir = os.path.join(base_dir, safe_repo)

    if os.path.isdir(repo_dir):
        try:
            if not os.listdir(repo_dir):
                os.rmdir(repo_dir)
                logger.info(f"Removed empty model folder: {repo_dir}")
        except Exception as exc:
            logger.warning(f"Failed to remove model folder {repo_dir}: {exc}")


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

        gen_ctx = generation_cfg.get("max_length") or generation_cfg.get("max_position_embeddings") or generation_cfg.get("max_tokens")
        if isinstance(gen_ctx, int) and gen_ctx > 0 and "ctx_size" not in defaults:
            defaults["ctx_size"] = gen_ctx

    return defaults


def _apply_hf_defaults_to_model(model: Model, metadata: Dict[str, Any], db: Session):
    if not metadata:
        return
    defaults = _derive_hf_defaults(metadata)
    if not defaults:
        return
    config = _coerce_model_config(model.config)
    changed = False
    for key, value in defaults.items():
        if value is None:
            continue
        existing = config.get(key)
        if existing in (None, "", 0):
            config[key] = value
            changed = True
    if changed:
        model.config = config
        db.commit()
        db.refresh(model)


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
    normalized_path = _normalize_model_path(model.file_path)
    if not normalized_path or not os.path.exists(normalized_path):
        raise FileNotFoundError("Model file not found on disk")
    
    layer_info = get_model_layer_info(normalized_path)
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
            "parameter_count": layer_info.get("parameter_count"),  # Formatted as "32B", "36B", etc.
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
        max_position_embeddings = _coerce_positive_int(config_sources.get("max_position_embeddings"))
        
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
            logger.debug(f"Failed to fetch tokenizer_config for {huggingface_id}: {exc}")
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
    file_size: int,
    pipeline_tag: Optional[str] = None
) -> Model:
    """
    Persist safetensors download information using a single logical Model row per repo.

    Historically we created one Model row per .safetensors file. This caused
    multi‑file repositories to appear as multiple independent models. The new
    behavior is:
      * Exactly one Model row per Hugging Face repo (huggingface_id) with
        model_format == "safetensors".
      * All individual .safetensors files for that repo are tracked in the
        safetensors manifest and share the same model_id.
      * The logical Model.file_size reflects the aggregate size of all files.
    """
    safetensors_metadata, tensor_summary, max_context = await _collect_safetensors_runtime_metadata(
        huggingface_id,
        filename
    )
    # Determine / reuse logical Model for this Hugging Face repo
    detected_pipeline = pipeline_tag or safetensors_metadata.get("pipeline_tag")
    is_embedding_like = _looks_like_embedding_model(
        detected_pipeline,
        huggingface_id,
        filename
    )

    # Try to find an existing logical model for this repo
    model_record = db.query(Model).filter(
        Model.huggingface_id == huggingface_id,
        Model.model_format == "safetensors"
    ).first()

    if not model_record:
        # Create a single logical model entry for the whole repo
        model_record = Model(
            name=filename.replace(".safetensors", ""),
            huggingface_id=huggingface_id,
            base_model_name=extract_base_model_name(filename),
            file_path=file_path,
            file_size=file_size,
            quantization=os.path.splitext(filename)[0],
            model_type=extract_model_type(filename),
            downloaded_at=datetime.utcnow(),
            model_format="safetensors",
            pipeline_tag=detected_pipeline
        )
        if is_embedding_like:
            model_record.config = {"embedding": True}
        db.add(model_record)
        db.commit()
        db.refresh(model_record)
    else:
        # Update existing logical model with any missing metadata and aggregate size
        updated = False
        if not model_record.pipeline_tag and detected_pipeline:
            model_record.pipeline_tag = detected_pipeline
            updated = True
        if is_embedding_like and not (model_record.config or {}).get("embedding"):
            # Ensure embedding flag is propagated
            current_config = _coerce_model_config(model_record.config)
            current_config["embedding"] = True
            model_record.config = current_config
            updated = True
        # Aggregate size across all files for this repo by summing manifest entries.
        # This avoids double‑counting if a file is redownloaded.
        try:
            from backend.huggingface import list_safetensors_downloads
            manifests = list_safetensors_downloads()
            total_size = 0
            for manifest in manifests:
                if manifest.get("huggingface_id") == huggingface_id:
                    total_size = sum(
                        (f.get("file_size") or 0)
                        for f in manifest.get("files", [])
                    )
                    break
            if total_size and total_size != (model_record.file_size or 0):
                model_record.file_size = total_size
                updated = True
        except Exception as exc:
            logger.warning(f"Failed to aggregate safetensors file sizes for {huggingface_id}: {exc}")
        if updated:
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
    normalized_path = _normalize_model_path(model.file_path)
    if not normalized_path or not os.path.exists(normalized_path):
        raise HTTPException(status_code=400, detail="Model file not found on disk")
    model.file_path = normalized_path
    return model


def _load_manifest_entry_for_model(model: Model) -> Dict[str, Any]:
    """Load unified manifest for a safetensors model (repo-level, not per-file)."""
    manifest = get_safetensors_manifest_entries(model.huggingface_id)
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
            logger.warning(f"Unreasonably large value detected: {value}, capping at {MAX_REASONABLE_VALUE}")
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
                logger.warning(f"Unreasonably large value detected: {candidate}, capping at {MAX_REASONABLE_VALUE}")
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

def _resolve_context_limit(manifest_entry: Dict[str, Any]) -> int:
    """
    Determine the maximum base context length allowed for UI clamping.
    
    For models where max_position_embeddings includes reserved prompt tokens
    (e.g., Qwen3: 40960 = 32768 output + 8192 prompt), we apply the reservation
    here for UI display/clamping purposes only. The full value is stored and
    passed to LMDeploy, which expects the full positional capacity.
    """
    metadata = manifest_entry.get("metadata") or {}
    config_data = metadata.get("config", {}) if isinstance(metadata.get("config"), dict) else {}
    
    candidates = [
        manifest_entry.get("max_context_length"),
        metadata.get("max_context_length"),
        metadata.get("context_length"),
    ]
    
    resolved_value = None
    for candidate in candidates:
        value = _coerce_positive_int(candidate)
        if value:
            resolved_value = value
            break
    
    if not resolved_value:
        return DEFAULT_LMDEPLOY_CONTEXT
    
    # Apply prompt reservation for UI clamping only (models handle this internally)
    config_max = _coerce_positive_int(config_data.get("max_position_embeddings"))
    if config_max and resolved_value == config_max:
        adjusted = _apply_prompt_reservation(config_max)
        if adjusted:
            resolved_value = adjusted
    
    return max(1024, min(resolved_value, MAX_LMDEPLOY_CONTEXT))


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
            raise HTTPException(status_code=400, detail="hf_overrides must be valid JSON") from exc
    if isinstance(value, dict):
        def _sanitize(obj: Any) -> Any:
            if isinstance(obj, dict):
                result = {}
                for key, nested in obj.items():
                    if not isinstance(key, str) or not key.strip():
                        raise HTTPException(status_code=400, detail="hf_overrides keys must be non-empty strings")
                    result[key.strip()] = _sanitize(nested)
                return result
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            raise HTTPException(status_code=400, detail="hf_overrides values must be scalars or nested objects")
        sanitized = _sanitize(value)
        return sanitized
    raise HTTPException(status_code=400, detail="hf_overrides must be an object or JSON string")


def _validate_lmdeploy_config(
    new_config: Optional[Dict[str, Any]],
    manifest_entry: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge and validate LMDeploy configuration.
    """
    if new_config is not None and not isinstance(new_config, dict):
        raise HTTPException(status_code=400, detail="Config payload must be an object")
    
    base_context_limit = _resolve_context_limit(manifest_entry)
    stored_config = (manifest_entry.get("lmdeploy") or {}).get("config")
    baseline = stored_config or get_default_lmdeploy_config(base_context_limit)
    merged = dict(baseline)
    if new_config:
        merged.update(new_config)
    
    def _as_int(key: str, minimum: int = 1, maximum: Optional[int] = None) -> int:
        value = merged.get(key, minimum)
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{key} must be an integer")
        if value < minimum:
            value = minimum
        if maximum is not None and value > maximum:
            value = maximum
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
    
    legacy_keys = {
        "context_length": "session_len",
        "max_batch_tokens": "max_prefill_token_num",
    }
    for legacy, target in legacy_keys.items():
        if legacy in merged and target not in merged:
            merged[target] = merged[legacy]

    session_len = _as_int("session_len", minimum=1024, maximum=base_context_limit)

    raw_scaling_mode = str(
        merged.get("rope_scaling_mode")
        or merged.get("rope_scaling_type")
        or "disabled"
    ).lower()
    if raw_scaling_mode in {"", "none", "disabled"}:
        scaling_mode = "disabled"
    else:
        scaling_mode = raw_scaling_mode

    scaling_factor_value = merged.get("rope_scaling_factor", 1.0)
    try:
        scaling_factor = float(scaling_factor_value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="rope_scaling_factor must be a number")
    if scaling_factor < 1.0:
        scaling_factor = 1.0
    if scaling_factor > MAX_ROPE_SCALING_FACTOR:
        scaling_factor = MAX_ROPE_SCALING_FACTOR

    if scaling_mode == "disabled" or scaling_factor <= 1.0:
        scaling_mode = "disabled"
        scaling_factor = 1.0
    else:
        # Scaling only makes sense when we know the base context; otherwise reject it.
        if not base_context_limit:
            raise HTTPException(
                status_code=400,
                detail="RoPE scaling cannot be enabled without a known base context length"
            )
        
        # Check if model_max_length > max_position_embeddings (means rope scaling can achieve model_max_length)
        metadata = manifest_entry.get("metadata") or {}
        config_data = metadata.get("config", {}) if isinstance(metadata.get("config"), dict) else {}
        model_max_length = _coerce_positive_int(metadata.get("model_max_length"))
        max_position_embeddings = _coerce_positive_int(config_data.get("max_position_embeddings"))
        
        if model_max_length and max_position_embeddings and model_max_length > max_position_embeddings:
            # Adapt base context to model_max_length / 4 for scaling
            # This allows 4x scaling to reach model_max_length
            adapted_base = int(model_max_length / 4)
            if adapted_base >= 1024:
                session_len = adapted_base
            else:
                # If adapted base is too small, use base context limit
                session_len = base_context_limit
        else:
            # Use base context limit (max_position_embeddings is used for clamping, not for scaling decisions)
            session_len = base_context_limit

    effective_session_len = session_len
    if scaling_mode != "disabled":
        effective_session_len = int(session_len * scaling_factor)
        # Clamp to model_max_length if available, otherwise max_position_embeddings
        metadata = manifest_entry.get("metadata") or {}
        config_data = metadata.get("config", {}) if isinstance(metadata.get("config"), dict) else {}
        model_max_length = _coerce_positive_int(metadata.get("model_max_length"))
        max_position_embeddings = _coerce_positive_int(config_data.get("max_position_embeddings"))
        if model_max_length:
            effective_session_len = min(effective_session_len, model_max_length)
        elif max_position_embeddings:
            effective_session_len = min(effective_session_len, max_position_embeddings)
        # Also clamp to LMDeploy's maximum
        effective_session_len = max(session_len, min(effective_session_len, MAX_LMDEPLOY_CONTEXT))

    merged["session_len"] = session_len
    merged["effective_session_len"] = effective_session_len
    merged["rope_scaling_mode"] = scaling_mode
    merged["rope_scaling_factor"] = scaling_factor

    max_context_token_num = _as_int(
        "max_context_token_num",
        minimum=session_len,
        maximum=base_context_limit,
    )
    merged["max_context_token_num"] = max(max_context_token_num, session_len)

    max_prefill_token_num = _as_int(
        "max_prefill_token_num",
        minimum=session_len,
        maximum=base_context_limit,
    )
    merged["max_prefill_token_num"] = max(max_prefill_token_num, session_len)

    merged["tensor_parallel"] = _as_int("tensor_parallel", minimum=1)
    merged["max_batch_size"] = _as_int("max_batch_size", minimum=1)
    
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

    # Build hf_overrides from individual fields or use provided hf_overrides
    hf_overrides_dict = _normalize_hf_overrides(merged.get("hf_overrides"))
    
    # If scaling is enabled and model_max_length > max_position_embeddings,
    # automatically set original_max_position_embeddings in HF overrides
    if scaling_mode != "disabled":
        metadata = manifest_entry.get("metadata") or {}
        config_data = metadata.get("config", {}) if isinstance(metadata.get("config"), dict) else {}
        model_max_length = _coerce_positive_int(metadata.get("model_max_length"))
        max_position_embeddings = _coerce_positive_int(config_data.get("max_position_embeddings"))
        
        if model_max_length and max_position_embeddings and model_max_length > max_position_embeddings:
            # Set original_max_position_embeddings to adapted base (model_max_length / 4)
            adapted_base = int(model_max_length / 4)
            if adapted_base >= 1024:
                hf_overrides_dict.setdefault("rope_scaling", {})
                hf_overrides_dict["rope_scaling"]["original_max_position_embeddings"] = adapted_base
                # Also set rope_type if not already set and scaling mode is yarn
                if scaling_mode == "yarn" and "rope_type" not in hf_overrides_dict["rope_scaling"]:
                    hf_overrides_dict["rope_scaling"]["rope_type"] = "yarn"
                # Set factor if not already set
                if "factor" not in hf_overrides_dict["rope_scaling"]:
                    hf_overrides_dict["rope_scaling"]["factor"] = scaling_factor
        elif max_position_embeddings and max_position_embeddings >= 1024:
            # Fallback: use max_position_embeddings directly
            hf_overrides_dict.setdefault("rope_scaling", {})
            hf_overrides_dict["rope_scaling"]["original_max_position_embeddings"] = max_position_embeddings
            # Also set rope_type if not already set and scaling mode is yarn
            if scaling_mode == "yarn" and "rope_type" not in hf_overrides_dict["rope_scaling"]:
                hf_overrides_dict["rope_scaling"]["rope_type"] = "yarn"
            # Set factor if not already set
            if "factor" not in hf_overrides_dict["rope_scaling"]:
                hf_overrides_dict["rope_scaling"]["factor"] = scaling_factor
    
    merged["hf_overrides"] = hf_overrides_dict
    
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
            huggingface_id=self.huggingface_id
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
        is_embedding = _model_is_embedding(model)
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
                "pipeline_tag": model.pipeline_tag,
                "is_embedding_model": is_embedding,
                "quantizations": []
            }
        else:
            if model.pipeline_tag and not grouped_models[key].get("pipeline_tag"):
                grouped_models[key]["pipeline_tag"] = model.pipeline_tag
            if is_embedding and not grouped_models[key].get("is_embedding_model"):
                grouped_models[key]["is_embedding_model"] = True
        
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
            "proxy_name": model.proxy_name,
            "pipeline_tag": model.pipeline_tag,
            "is_embedding_model": is_embedding
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
async def get_safetensors_metadata_endpoint(model_id: str):
    """Fetch safetensors metadata on demand for a HuggingFace repo and include unified manifest details when available."""
    try:
        metadata = await get_safetensors_metadata_summary(model_id)
        # Get unified manifest for local entry details
        from backend.huggingface import get_safetensors_manifest_entries
        local_manifest = get_safetensors_manifest_entries(model_id)
        if local_manifest:
            metadata["local_manifest"] = local_manifest
            metadata["max_context_length"] = local_manifest.get("max_context_length") or metadata.get("max_context_length")
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
            "cached_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching safetensors metadata for {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch safetensors metadata: {str(e)}")


@router.get("/safetensors")
async def list_safetensors_models():
    """List safetensors downloads stored locally."""
    try:
        return list_grouped_safetensors_downloads()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/safetensors")
async def delete_safetensors_model(request: dict, db: Session = Depends(get_db)):
    """Delete entire safetensors model (all files for the repo)."""
    try:
        huggingface_id = request.get("huggingface_id")
        if not huggingface_id:
            raise HTTPException(status_code=400, detail="huggingface_id is required")

        # Prevent deletion while runtime is active for this logical model
        active_instance = db.query(RunningInstance).filter(
            RunningInstance.runtime_type == "lmdeploy"
        ).first()
        target_model = db.query(Model).filter(
            Model.huggingface_id == huggingface_id,
            Model.model_format == "safetensors",
        ).first()
        if active_instance and target_model and active_instance.model_id == target_model.id:
            raise HTTPException(status_code=400, detail="Cannot delete a model currently served by LMDeploy")

        # Get unified manifest and delete all files
        from backend.huggingface import get_safetensors_manifest_entries, delete_safetensors_download

        manifest = get_safetensors_manifest_entries(huggingface_id)
        if not manifest or not manifest.get("files"):
            raise HTTPException(status_code=404, detail="Safetensors model not found")

        # Delete all files in the unified manifest
        for file_entry in manifest.get("files", []):
            entry_filename = file_entry.get("filename")
            if entry_filename:
                delete_safetensors_download(huggingface_id, entry_filename)

        # Delete the single logical Model row
        if target_model:
            db.delete(target_model)
            db.commit()

        return {"message": f"Safetensors model {huggingface_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/safetensors/reload-from-disk")
async def reload_safetensors_from_disk(db: Session = Depends(get_db)):
    """Reset all safetensors database entries and reload them from file storage."""
    try:
        from backend.huggingface import SAFETENSORS_DIR, record_safetensors_download, get_default_lmdeploy_config
        
        # Prevent reload while runtime is active
        active_instance = db.query(RunningInstance).filter(
            RunningInstance.runtime_type == "lmdeploy"
        ).first()
        if active_instance:
            raise HTTPException(
                status_code=400,
                detail="Cannot reload safetensors models while LMDeploy runtime is active. Please stop the runtime first."
            )
        
        # Delete all existing safetensors Model entries
        safetensors_models = db.query(Model).filter(Model.model_format == "safetensors").all()
        deleted_count = len(safetensors_models)
        for model in safetensors_models:
            db.delete(model)
        db.commit()
        logger.info(f"Deleted {deleted_count} safetensors model entries from database")
        
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
                        logger.warning(f"Failed to delete manifest {manifest_path}: {exc}")
        logger.info(f"Deleted {deleted_manifests} safetensors manifest files")
        
        # Scan file storage and rebuild entries
        if not os.path.exists(SAFETENSORS_DIR):
            return {
                "message": "No safetensors directory found",
                "reloaded": 0,
                "deleted": deleted_count,
                "deleted_manifests": deleted_manifests
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
                    safetensors_files.append({
                        "filename": file_entry.name,
                        "file_path": file_entry.path,
                        "file_size": file_entry.stat().st_size
                    })
            
            if not safetensors_files:
                continue
            
            # Process each file to rebuild database entries
            model_record = None
            for file_info in safetensors_files:
                try:
                    filename = file_info["filename"]
                    file_path = file_info["file_path"]
                    file_size = file_info["file_size"]
                    
                    # Collect metadata (this will also create/update the manifest)
                    safetensors_metadata, tensor_summary, max_context = await _collect_safetensors_runtime_metadata(
                        huggingface_id,
                        filename
                    )
                    
                    # Get or create model record (one per repo)
                    if not model_record:
                        detected_pipeline = safetensors_metadata.get("pipeline_tag")
                        is_embedding_like = _looks_like_embedding_model(
                            detected_pipeline,
                            huggingface_id,
                            filename
                        )
                        
                        model_record = db.query(Model).filter(
                            Model.huggingface_id == huggingface_id,
                            Model.model_format == "safetensors"
                        ).first()
                        
                        if not model_record:
                            model_record = Model(
                                name=filename.replace(".safetensors", ""),
                                huggingface_id=huggingface_id,
                                base_model_name=extract_base_model_name(filename),
                                file_path=file_path,  # Use first file's path
                                file_size=0,  # Will be aggregated below
                                quantization=os.path.splitext(filename)[0],
                                model_type=extract_model_type(filename),
                                downloaded_at=datetime.utcnow(),
                                model_format="safetensors",
                                pipeline_tag=detected_pipeline
                            )
                            if is_embedding_like:
                                model_record.config = {"embedding": True}
                            db.add(model_record)
                            db.commit()
                            db.refresh(model_record)
                    
                    # Record file in manifest
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
                    
                except Exception as exc:
                    error_msg = f"Failed to reload {huggingface_id}/{file_info.get('filename', 'unknown')}: {exc}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
            
            # Update model record with aggregated size
            if model_record:
                try:
                    from backend.huggingface import list_safetensors_downloads
                    manifests = list_safetensors_downloads()
                    total_size = 0
                    for manifest in manifests:
                        if manifest.get("huggingface_id") == huggingface_id:
                            total_size = sum(
                                (f.get("file_size") or 0)
                                for f in manifest.get("files", [])
                            )
                            break
                    if total_size:
                        model_record.file_size = total_size
                        db.commit()
                        db.refresh(model_record)
                except Exception as exc:
                    logger.warning(f"Failed to update aggregate size for {huggingface_id}: {exc}")
                
                reloaded_count += 1
        
        result = {
            "message": f"Reloaded {reloaded_count} safetensors models from disk",
            "reloaded": reloaded_count,
            "deleted": deleted_count,
            "deleted_manifests": deleted_manifests
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
        validated_config
    )
    return {
        "config": updated_entry.get("lmdeploy", {}).get("config", validated_config),
        "updated_at": updated_entry.get("lmdeploy", {}).get("updated_at")
    }


@router.post("/safetensors/{model_id}/metadata/regenerate")
async def regenerate_safetensors_metadata_endpoint(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Refresh safetensors metadata/manifest entries without redownloading files."""
    model = _get_safetensors_model(model_id, db)
    huggingface_id = model.huggingface_id
    manifest = get_safetensors_manifest_entries(huggingface_id)
    if not manifest or not manifest.get("files"):
        raise HTTPException(status_code=404, detail="No safetensors manifest entries found for this model")

    # Collect metadata for all files in the unified model
    # Note: We iterate over files to collect file-level tensor summaries,
    # but the model is treated as a unified entity (one model per repo)
    unified_metadata = {}
    max_context = 0
    files = manifest.get("files", [])
    
    for file_entry in files:
        filename = file_entry.get("filename")
        try:
            metadata, tensor_summary, context_len = await _collect_safetensors_runtime_metadata(
                huggingface_id,
                filename
            )
        except Exception as exc:
            logger.warning(f"Failed to regenerate metadata for {huggingface_id}/{filename}: {exc}")
            metadata, tensor_summary, context_len = manifest.get("metadata") or {}, file_entry.get("tensor_summary") or {}, manifest.get("max_context_length")

        # Update file-level tensor summary (file-level data for unified model)
        if tensor_summary:
            file_entry["tensor_summary"] = tensor_summary
        
        # Aggregate repo-level metadata (use first successful metadata)
        # All files share the same unified metadata
        if metadata and not unified_metadata:
            unified_metadata = metadata

        # Resolve context length
        resolved_context = context_len or metadata.get("max_context_length")
        rope_override = None
        lmdeploy_config = (manifest.get("lmdeploy") or {}).get("config") if isinstance(manifest.get("lmdeploy"), dict) else {}
        if isinstance(lmdeploy_config, dict):
            hf_overrides = lmdeploy_config.get("hf_overrides")
            if isinstance(hf_overrides, dict):
                rope_scaling = hf_overrides.get("rope_scaling")
                if isinstance(rope_scaling, dict):
                    rope_override = (
                        rope_scaling.get("original_max_position_embeddings")
                        or rope_scaling.get("original_max_position_embedding")
                    )
        if rope_override:
            try:
                resolved_context = int(rope_override)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid rope override for %s during metadata regen",
                    huggingface_id,
                )
        if resolved_context:
            max_context = max(max_context, resolved_context)

    # Update unified manifest
    if unified_metadata:
        manifest["metadata"] = unified_metadata
    if max_context:
        manifest["max_context_length"] = max_context
    
    manifest.setdefault("lmdeploy", {})
    manifest["lmdeploy"].setdefault(
        "config",
        get_default_lmdeploy_config(manifest.get("max_context_length"))
    )

    save_safetensors_manifest_entries(huggingface_id, manifest)
    return {
        "message": f"Metadata regenerated for {huggingface_id}",
        "max_context_length": max_context,
        "files": files
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
    manager_status = manager.status()
    
    # Only return running_instance if LMDeploy is actually running
    instance_payload = None
    if manager_status.get("running"):
        running_instance = db.query(RunningInstance).filter(RunningInstance.runtime_type == "lmdeploy").first()
        if running_instance:
            instance_payload = {
                "model_id": running_instance.model_id,
                "started_at": running_instance.started_at.isoformat() if running_instance.started_at else None,
                "config": json.loads(running_instance.config) if running_instance.config else {},
            }
    else:
        # Clean up stale RunningInstance records if LMDeploy is not running
        stale_instances = db.query(RunningInstance).filter(RunningInstance.runtime_type == "lmdeploy").all()
        if stale_instances:
            for instance in stale_instances:
                model = db.query(Model).filter(Model.id == instance.model_id).first()
                if model:
                    model.is_active = False
                db.delete(instance)
            db.commit()
    
    return {
        "manager": manager_status,
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
    
    update_lmdeploy_config(model.huggingface_id, validated_config)
    
    from backend.main import websocket_manager
    await websocket_manager.send_model_status_update(
        model_id=model.id,
        status="starting",
        details={"runtime": "lmdeploy", "message": f"Starting LMDeploy for {model.name}"}
    )
    
    try:
        # Derive a human-friendly model name for LMDeploy (used by --model-name).
        # For unified safetensors models, use the Hugging Face repo id.
        display_name = model.huggingface_id or model.base_model_name or model.name
        # For unified manifests, use the model directory (contains all files)
        model_dir = os.path.dirname(model.file_path)
        runtime_status = await manager.start(
            {
                "model_id": model.id,
                "huggingface_id": model.huggingface_id,
                "file_path": model.file_path,
                "model_dir": model_dir,
                "model_name": display_name,
                "display_name": display_name,
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
        pipeline_tag = request.get("pipeline_tag")
        
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
            model_format,
            pipeline_tag
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
    model_format: str = "gguf",
    pipeline_tag: Optional[str] = None
):
    """Background task to download model with WebSocket progress"""
    from backend.database import SessionLocal
    db = SessionLocal()
    
    try:
        model_record = None
        metadata_result = None

        if websocket_manager and task_id:
            file_path, file_size = await download_model_with_websocket_progress(
                huggingface_id, filename, websocket_manager, task_id, total_bytes, model_format, huggingface_id
            )
        else:
            file_path, file_size = await download_model(huggingface_id, filename, model_format)
        
        if model_format == "gguf":
            model_record, metadata_result = await _record_gguf_download_post_fetch(
                db,
                huggingface_id,
                filename,
                file_path,
                file_size,
                pipeline_tag=pipeline_tag,
            )
        else:
            model_record = await _save_safetensors_download(
                db,
                huggingface_id,
                filename,
                file_path,
                file_size,
                pipeline_tag=pipeline_tag
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
                "pipeline_tag": model_record.pipeline_tag if model_record else pipeline_tag,
                "is_embedding_model": _model_is_embedding(model_record) if model_record else False,
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


async def _record_gguf_download_post_fetch(
    db: Session,
    huggingface_id: str,
    filename: str,
    file_path: str,
    file_size: int,
    pipeline_tag: Optional[str] = None,
) -> Tuple[Model, Optional[Dict[str, Any]]]:
    """
    Shared helper to create GGUF Model rows and manifest entries after a file has been downloaded.
    Returns (model_record, metadata_result).
    """
    quantization = _extract_quantization(filename)
    base_model_name = extract_base_model_name(filename)
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

    # Reuse a single logical Model row per (huggingface_id, quantization) to avoid
    # creating one entry per GGUF shard. Additional shards for the same quantization
    # simply update size/metadata and are tracked in the GGUF manifest.
    model_record = db.query(Model).filter(
        Model.huggingface_id == huggingface_id,
        Model.quantization == quantization,
        Model.model_format == "gguf",
    ).first()

    if not model_record:
        model_record = Model(
            name=filename.replace(".gguf", ""),
            huggingface_id=huggingface_id,
            base_model_name=base_model_name,
            file_path=file_path,
            file_size=file_size,
            quantization=quantization,
            model_type=extract_model_type(filename),
            proxy_name=generate_proxy_name(huggingface_id, quantization),
            model_format="gguf",
            downloaded_at=datetime.utcnow(),
            pipeline_tag=detected_pipeline,
        )
        if is_embedding_like:
            model_record.config = {"embedding": True}
        db.add(model_record)
        db.commit()
        db.refresh(model_record)
    else:
        updated = False
        # Keep first file_path as canonical; just update aggregate size.
        if file_size and file_size > 0:
            current_size = model_record.file_size or 0
            model_record.file_size = current_size + file_size
            updated = True
        if not model_record.pipeline_tag and detected_pipeline:
            model_record.pipeline_tag = detected_pipeline
            updated = True
        if is_embedding_like:
            current_config = _coerce_model_config(model_record.config)
            if not current_config.get("embedding"):
                current_config["embedding"] = True
                model_record.config = current_config
                updated = True
        if updated:
            db.commit()
            db.refresh(model_record)

    try:
        metadata_result = _refresh_model_metadata_from_file(model_record, db)
    except FileNotFoundError:
        logger.warning(f"Model file missing during metadata refresh for {model_record.id}")
    except Exception as meta_exc:
        logger.warning(f"Failed to refresh metadata for model {model_record.id}: {meta_exc}")

    manifest_entry = None
    try:
        manifest_entry = await create_gguf_manifest_entry(
            model_record.huggingface_id,
            file_path,
            file_size,
            model_id=model_record.id,
        )
    except Exception as manifest_exc:
        logger.warning(f"Failed to record GGUF manifest entry for {filename}: {manifest_exc}")
    if manifest_entry:
        metadata_for_defaults = manifest_entry.get("metadata") or {}
        try:
            _apply_hf_defaults_to_model(model_record, metadata_for_defaults, db)
        except Exception as default_exc:
            logger.warning(f"Failed to apply HF defaults for model {model_record.id}: {default_exc}")

    return model_record, metadata_result


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
                filename,
                huggingface_id,
                "safetensors-bundle",
            )

            file_path, file_size = await download_model_with_websocket_progress(
                huggingface_id,
                filename,
                proxy,
                task_id,
                size_hint,
                "safetensors",
                huggingface_id
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
            model_format="safetensors-bundle",
            files_completed=total_files,
            files_total=total_files,
            current_filename=files[-1]["filename"] if files else "",
            huggingface_id=huggingface_id
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


async def download_gguf_bundle_task(
    huggingface_id: str,
    quantization: str,
    files: List[Dict[str, Any]],
    websocket_manager,
    task_id: str,
    total_bundle_bytes: int = 0,
    pipeline_tag: Optional[str] = None,
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
                filename,
                huggingface_id,
                "gguf-bundle",
            )

            file_path, file_size = await download_model_with_websocket_progress(
                huggingface_id,
                filename,
                proxy,
                task_id,
                size_hint,
                "gguf",
                huggingface_id,
            )

            # Reuse the standard GGUF recording path to keep DB and manifest consistent
            try:
                await _record_gguf_download_post_fetch(
                    db,
                    huggingface_id,
                    filename,
                    file_path,
                    file_size,
                    pipeline_tag=pipeline_tag,
                )
            except Exception as exc:
                logger.error(f"Failed to record GGUF download for {filename}: {exc}")

            bytes_completed += file_size

        final_total = aggregate_total or bytes_completed
        await websocket_manager.send_download_progress(
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

        await websocket_manager.broadcast(
            {
                "type": "download_complete",
                "huggingface_id": huggingface_id,
                "model_format": "gguf-bundle",
                "quantization": quantization,
                "filenames": [f["filename"] for f in files],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    except Exception as exc:
        logger.error(f"GGUF bundle download failed: {exc}")
        if websocket_manager:
            await websocket_manager.send_notification(
                "error",
                "Download Failed",
                f"GGUF bundle failed: {str(exc)}",
                task_id,
            )
        await websocket_manager.broadcast(
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


@router.post("/gguf/download-bundle")
async def download_gguf_bundle(
    request: dict,
    background_tasks: BackgroundTasks,
):
    huggingface_id = request.get("huggingface_id")
    quantization = request.get("quantization")
    files = request.get("files") or []
    pipeline_tag = request.get("pipeline_tag")

    if not huggingface_id:
        raise HTTPException(status_code=400, detail="huggingface_id is required")
    if not quantization:
        raise HTTPException(status_code=400, detail="quantization is required")
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

    if not sanitized_files:
        raise HTTPException(status_code=400, detail="No valid files to download")

    task_id = f"download_gguf_bundle_{huggingface_id.replace('/', '_')}_{quantization}_{int(time.time() * 1000)}"

    async with download_lock:
        is_downloading = any(
            d["huggingface_id"] == huggingface_id
            and d.get("model_format") == "gguf-bundle"
            and d.get("quantization") == quantization
            for d in active_downloads.values()
        )
        if is_downloading:
            raise HTTPException(status_code=409, detail="GGUF bundle is already being downloaded")
        active_downloads[task_id] = {
            "huggingface_id": huggingface_id,
            "filename": quantization,
            "quantization": quantization,
            "model_format": "gguf-bundle",
        }

    from backend.main import websocket_manager
    background_tasks.add_task(
        download_gguf_bundle_task,
        huggingface_id,
        quantization,
        sanitized_files,
        websocket_manager,
        task_id,
        declared_total,
        pipeline_tag,
    )

    return {
        "message": "GGUF bundle download started",
        "huggingface_id": huggingface_id,
        "quantization": quantization,
        "task_id": task_id,
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
        from backend.unified_monitor import unified_monitor
        from backend.main import websocket_manager
        await websocket_manager.send_model_status_update(
            model_id=model_id, status="starting",
            details={"message": f"Starting {model.name}"}
        )
        
        # Get proxy name from database (config already contains this model)
        if not model.proxy_name:
            raise ValueError(f"Model '{model.name}' does not have a proxy_name set")
        proxy_model_name = model.proxy_name
        
        # Mark model as loading immediately so UI shows loading state
        unified_monitor.mark_model_loading(proxy_model_name)
        
        # Get model configuration (for database record, not config file)
        config = _coerce_model_config(model.config)
        if _looks_like_embedding_model(
            model.pipeline_tag,
            model.huggingface_id,
            model.name,
            model.base_model_name
        ) and not config.get("embedding"):
            config["embedding"] = True
            model.config = config
            db.commit()
        
        # NOTE: We do NOT modify the llama-swap config here.
        # The config already contains all models (written on startup/config edit).
        # We just need to trigger llama-swap to load this specific model.
        
        # Trigger model loading in the background
        # This makes a request to llama-swap which starts loading the model.
        # We don't wait for it to complete - the UnifiedMonitor polls /running
        # to detect when the model transitions from "loading" to "running".
        # 
        # The /running endpoint is safe to poll (never returns 503).
        # Only /v1/chat/completions returns 503 during loading, but we
        # fire-and-forget here so we don't block on it.
        
        async def trigger_model_load():
            """Background task to trigger model loading"""
            try:
                async with httpx.AsyncClient() as client:
                    # Make a minimal request to trigger loading
                    # With sendLoadingState: true, this will stream progress
                    test_request = {
                        "model": proxy_model_name,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1
                    }
                    # Long timeout since loading can take a while
                    await client.post(
                        "http://localhost:2000/v1/chat/completions",
                        json=test_request,
                        timeout=300.0  # 5 minutes for large models
                    )
                    logger.info(f"Model {proxy_model_name} load trigger completed")
            except Exception as e:
                # Expected during loading (might timeout or get 503)
                # The /running polling will detect actual state
                logger.debug(f"Model load trigger finished (may have timed out): {e}")
        
        # Start loading in background - don't wait
        asyncio.create_task(trigger_model_load())
        logger.info(f"Model {proxy_model_name} loading triggered (background)")
        
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
        
        # Broadcast loading event (model is loading in background)
        # The "ready" event will be broadcast by UnifiedMonitor when it
        # detects the model state change from "loading" to "running" via /running polling
        await unified_monitor.broadcast_model_event("loading", proxy_model_name, {
            "model_id": model_id,
            "model_name": model.name
        })
        await unified_monitor.trigger_status_update()
        
        return {
            "model_id": model_id,
            "proxy_model_name": proxy_model_name,
            "port": 2000,
            "api_endpoint": f"http://localhost:2000/v1/chat/completions"
        }
        
    except Exception as e:
        # Clear loading state on error
        if model.proxy_name:
            unified_monitor.mark_model_stopped(model.proxy_name)
        
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
        from backend.unified_monitor import unified_monitor
        llama_swap_manager = get_llama_swap_manager()
        
        proxy_name = running_instance.proxy_model_name
        
        # Clear loading state if model was still loading
        if proxy_name:
            unified_monitor.mark_model_stopped(proxy_name)
        
        # Unregister from llama-swap (it stops the process)
        if proxy_name:
            logger.info(f"Calling unregister_model with proxy_model_name: {proxy_name}")
            await llama_swap_manager.unregister_model(proxy_name)
            logger.info("unregister_model call completed")
        
        # Update database
        db.delete(running_instance)
        model = db.query(Model).filter(Model.id == model_id).first()
        if model:
            model.is_active = False
        db.commit()
        
        # Broadcast stopped event immediately (event-driven, no polling)
        if proxy_name:
            await unified_monitor.broadcast_model_event("stopped", proxy_name, {
                "model_id": model_id
            })
        await unified_monitor.trigger_status_update()
        
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
        normalized_path = _normalize_model_path(model.file_path)
        if normalized_path and os.path.exists(normalized_path):
            os.remove(normalized_path)
        
        # Delete from database
        db.delete(model)
        deleted_count += 1
    
    db.commit()

    # If this was a GGUF group and no models remain, clean up the repo folder
    remaining_gguf = db.query(Model).filter(
        Model.huggingface_id == huggingface_id,
        Model.model_format == "gguf"
    ).count()
    if remaining_gguf == 0:
        _cleanup_model_folder_if_no_quantizations(db, huggingface_id, "gguf")
    
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
    
    huggingface_id = model.huggingface_id
    model_format = (model.model_format or "gguf").lower()

    # Delete file
    normalized_path = _normalize_model_path(model.file_path)
    if normalized_path and os.path.exists(normalized_path):
        os.remove(normalized_path)
    
    # Delete from database
    db.delete(model)
    db.commit()

    # If this was the last quantization for this repo/format, remove its folder
    _cleanup_model_folder_if_no_quantizations(db, huggingface_id, model_format)
    
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
    
    layer_info = None
    normalized_path = _normalize_model_path(model.file_path)
    if normalized_path and os.path.exists(normalized_path):
        try:
            layer_info = get_model_layer_info(normalized_path)
        except Exception as e:
            logger.error(f"Failed to get layer info for model {model_id}: {e}")
    if layer_info:
        return {
            "layer_count": layer_info["layer_count"],
            "architecture": layer_info["architecture"],
            "context_length": layer_info["context_length"],
            "parameter_count": layer_info.get("parameter_count"),  # Formatted as "32B", "36B", etc.
            "vocab_size": layer_info["vocab_size"],
            "embedding_length": layer_info["embedding_length"],
            "attention_head_count": layer_info["attention_head_count"],
            "attention_head_count_kv": layer_info["attention_head_count_kv"],
            "block_count": layer_info["block_count"],
            "is_moe": layer_info.get("is_moe", False),
            "expert_count": layer_info.get("expert_count", 0),
            "experts_used_count": layer_info.get("experts_used_count", 0)
        }
    # Fallback to default values if metadata unavailable
    logger.warning(
        f"Using default layer info fallback (32 layers) for model_id={model_id}; "
        "GGUF metadata could not be read or did not provide layer information."
    )
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
    
    normalized_path = _normalize_model_path(model.file_path)
    file_path = normalized_path if normalized_path and os.path.exists(normalized_path) else None
    
    try:
        # Get layer info from GGUF metadata (if available)
        layer_info = get_model_layer_info(file_path) if file_path else None
    except Exception as e:
        logger.error(f"Failed to get layer info for recommendations (model {model_id}): {e}")
        layer_info = None

    if not layer_info:
        layer_info = {
            "layer_count": 32,
            "architecture": "unknown",
            "context_length": 0,
            "attention_head_count": 0,
            "embedding_length": 0
        }
    
    try:
        recommendations = await get_model_recommendations(
            model_layer_info=layer_info,
            model_name=model.name or model.huggingface_id or "",
            file_path=file_path
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


@router.get("/{model_id}/hf-metadata")
async def get_model_hf_metadata(
    model_id: int,
    db: Session = Depends(get_db)
):
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    metadata_entry = None
    if (model.model_format or "gguf").lower() == "safetensors":
        metadata_entry = _load_manifest_entry_for_model(model)
    else:
        filename = _extract_filename(model.file_path)
        if not filename:
            raise HTTPException(status_code=400, detail="Model file path is not set")
        metadata_entry = get_gguf_manifest_entry(model.huggingface_id, filename)

    if not metadata_entry:
        raise HTTPException(status_code=404, detail="Metadata not found for model")

    metadata = metadata_entry.get("metadata") or {}
    defaults = _derive_hf_defaults(metadata)

    return {
        "metadata": metadata,
        "gguf_layer_info": metadata_entry.get("gguf_layer_info"),
        "max_context_length": metadata_entry.get("max_context_length"),
        "hf_defaults": defaults
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
