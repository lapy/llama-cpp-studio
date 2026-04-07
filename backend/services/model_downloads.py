"""Model download orchestration (background tasks, locks, bundle progress)."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from backend.data_store import generate_proxy_name, get_store
from backend.huggingface import (
    create_gguf_manifest_entry,
    download_model,
    download_model_with_progress,
    extract_quantization,
    get_model_details,
    get_safetensors_metadata_summary,
    get_tokenizer_config,
    list_safetensors_downloads,
    record_safetensors_download,
    resolve_cached_model_path,
)
from backend.logging_config import get_logger
from backend.model_config import (
    effective_model_config_from_raw,
    set_embedding_flag,
)
from backend.services import model_metadata as mm
from backend.utils.coercion import coerce_positive_int

logger = get_logger(__name__)


def mark_llama_swap_stale_after_download() -> None:
    try:
        from backend.llama_swap_manager import mark_swap_config_stale

        mark_swap_config_stale()
    except Exception as exc:
        logger.debug("mark_swap_config_stale: %s", exc)


# Global download tracking to prevent duplicates and track active downloads
active_downloads: Dict[str, Dict[str, Any]] = {}
download_lock = asyncio.Lock()


class ActiveDownloadConflict(Exception):
    """Raised when an equivalent download is already in progress (HTTP 409)."""

    __slots__ = ("detail",)

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


async def register_single_model_download(
    *,
    task_id: str,
    huggingface_id: str,
    filename: str,
    quantization: str,
    model_format: str,
) -> None:
    async with download_lock:
        if any(
            d["huggingface_id"] == huggingface_id
            and d["filename"] == filename
            and d.get("model_format", model_format) == model_format
            for d in active_downloads.values()
        ):
            raise ActiveDownloadConflict(
                "This quantization is already being downloaded"
            )
        active_downloads[task_id] = {
            "huggingface_id": huggingface_id,
            "filename": filename,
            "quantization": quantization,
            "model_format": model_format,
        }


async def register_safetensors_bundle_download(
    *, task_id: str, huggingface_id: str
) -> None:
    async with download_lock:
        if any(
            d["huggingface_id"] == huggingface_id
            and d.get("model_format") == "safetensors-bundle"
            for d in active_downloads.values()
        ):
            raise ActiveDownloadConflict(
                "Safetensors bundle is already being downloaded"
            )
        active_downloads[task_id] = {
            "huggingface_id": huggingface_id,
            "filename": "bundle",
            "quantization": "safetensors-bundle",
            "model_format": "safetensors-bundle",
        }


async def register_gguf_bundle_download(
    *, task_id: str, huggingface_id: str, quantization: str
) -> None:
    async with download_lock:
        if any(
            d["huggingface_id"] == huggingface_id
            and d.get("model_format") == "gguf-bundle"
            and d.get("quantization") == quantization
            for d in active_downloads.values()
        ):
            raise ActiveDownloadConflict("GGUF bundle is already being downloaded")
        active_downloads[task_id] = {
            "huggingface_id": huggingface_id,
            "filename": quantization,
            "quantization": quantization,
            "model_format": "gguf-bundle",
        }


async def register_gguf_projector_download(
    *,
    task_id: str,
    huggingface_id: str,
    model_id: str,
    mmproj_filename: str,
) -> None:
    async with download_lock:
        if any(
            d.get("model_id") == model_id
            and d.get("filename") == mmproj_filename
            and d.get("model_format") == "gguf-projector"
            for d in active_downloads.values()
        ):
            raise ActiveDownloadConflict("This projector is already being applied")
        active_downloads[task_id] = {
            "huggingface_id": huggingface_id,
            "model_id": model_id,
            "filename": mmproj_filename,
            "model_format": "gguf-projector",
        }


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
        if self.total_bytes > 0:
            bundle_total = self.total_bytes
            aggregate_progress = int((aggregate_downloaded / bundle_total) * 100)
        else:
            bundle_total = aggregate_downloaded or 0
            aggregate_progress = progress
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


async def collect_safetensors_runtime_metadata(
    huggingface_id: str, filename: str
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
        config_sources = config_data if isinstance(config_data, dict) else {}

        model_max_length = coerce_positive_int(details.get("model_max_length"))
        max_position_embeddings = coerce_positive_int(
            config_sources.get("max_position_embeddings")
        )
        max_context_length = model_max_length or max_position_embeddings

        metadata = {
            "architecture": details.get("architecture"),
            "base_model": details.get("base_model"),
            "pipeline_tag": details.get("pipeline_tag"),
            "parameters": details.get("parameters"),
            "model_max_length": model_max_length,
            "config": config_data,
            "language": details.get("language"),
            "license": details.get("license"),
        }
        if max_context_length:
            metadata["max_context_length"] = max_context_length

        try:
            tokenizer_config = get_tokenizer_config(huggingface_id)
            if tokenizer_config:
                if "tokenizer_config" not in metadata:
                    metadata["tokenizer_config"] = tokenizer_config
                tokenizer_max = None
                for key in ("model_max_length", "max_len", "max_length"):
                    candidate = coerce_positive_int(tokenizer_config.get(key))
                    if candidate:
                        tokenizer_max = candidate
                        break
                if tokenizer_max:
                    metadata["model_max_length"] = tokenizer_max
        except Exception as exc:
            logger.debug(
                "Failed to fetch tokenizer_config for %s: %s", huggingface_id, exc
            )
    except Exception as exc:
        logger.warning(
            "Failed to collect model details for %s: %s", huggingface_id, exc
        )

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
            "Failed to collect safetensors metadata for %s/%s: %s",
            huggingface_id,
            filename,
            exc,
        )

    return metadata or {}, tensor_summary or {}, max_context_length


async def save_safetensors_download(
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
    (
        safetensors_metadata,
        tensor_summary,
        _max_context,
    ) = await collect_safetensors_runtime_metadata(huggingface_id, filename)
    detected_pipeline = pipeline_tag or safetensors_metadata.get("pipeline_tag")
    is_embedding_like = mm.looks_like_embedding_model(
        detected_pipeline, huggingface_id, filename
    )
    model_id = huggingface_id.replace("/", "--")
    model_record = store.get_model(model_id)

    if not model_record:
        from datetime import timezone as _tz

        repo_name = (
            huggingface_id.split("/")[-1] if isinstance(huggingface_id, str) else ""
        )
        base_model_name = repo_name or mm.extract_base_model_name(filename)
        model_type = mm.extract_model_type(huggingface_id or repo_name or filename)
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
                set_embedding_flag({}, model_format="safetensors", store=store)
                if is_embedding_like
                else {}
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
                model_record.get("config"), model_format="safetensors", store=store
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
    try:
        manifests = list_safetensors_downloads()
        total_size = 0
        for manifest in manifests:
            if manifest.get("huggingface_id") == huggingface_id:
                total_size = sum(
                    (f.get("file_size") or 0) for f in manifest.get("files", [])
                )
                break

        if total_size and total_size != (model_record.get("file_size") or 0):
            store.update_model(model_id, {"file_size": total_size})
            model_record = store.get_model(model_id) or model_record
    except Exception as exc:
        logger.warning(
            "Failed to aggregate safetensors file sizes for %s: %s", huggingface_id, exc
        )

    logger.info(
        "Safetensors download recorded for %s/%s (model_id=%s)",
        huggingface_id,
        filename,
        model_record.get("id"),
    )
    return model_record


async def record_gguf_download_post_fetch(
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
    quantization = extract_quantization(filename)
    repo_name = huggingface_id.split("/")[-1] if isinstance(huggingface_id, str) else ""
    base_model_name = repo_name
    if repo_name.endswith("-GGUF"):
        base_model_name = repo_name[: -len("-GGUF")]
    detected_pipeline = pipeline_tag
    is_embedding_like = mm.looks_like_embedding_model(
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

        model_record = {
            "id": model_id,
            "huggingface_id": huggingface_id,
            "display_name": f"{base_model_name}-{quantization}",
            "base_model_name": base_model_name,
            "file_size": file_size if aggregate_size else 0,
            "quantization": quantization,
            "model_type": mm.extract_model_type(filename),
            "proxy_name": generate_proxy_name(huggingface_id, quantization),
            "format": "gguf",
            "downloaded_at": datetime.now(_tz.utc).isoformat(),
            "pipeline_tag": detected_pipeline,
            "config": (
                set_embedding_flag({}, model_format="gguf", store=store)
                if is_embedding_like
                else {}
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
                    model_record.get("config"), model_format="gguf", store=store
                )
        if updates:
            store.update_model(model_id, updates)
        model_record = store.get_model(model_id) or model_record

    try:
        await create_gguf_manifest_entry(
            model_record.get("huggingface_id"),
            file_path,
            file_size,
            model_id=model_record.get("id"),
        )
    except Exception as manifest_exc:
        logger.warning(
            "Failed to record GGUF manifest entry for %s: %s", filename, manifest_exc
        )

    metadata_result = None
    try:
        metadata_result = mm.refresh_gguf_model_metadata(model_record, store, file_path)
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
            model_record, metadata_result = await record_gguf_download_post_fetch(
                store,
                huggingface_id,
                filename,
                file_path,
                file_size,
                pipeline_tag=pipeline_tag,
                aggregate_size=True,
            )
        elif model_format == "gguf":
            logger.info(
                "Downloaded standalone mmproj file for %s: %s", huggingface_id, filename
            )
        else:
            model_record = await save_safetensors_download(
                store,
                huggingface_id,
                filename,
                file_path,
                file_size,
                pipeline_tag=pipeline_tag,
            )

        if progress_manager and task_id:
            progress_manager.complete_task(task_id, f"Downloaded {filename}")
            payload = {
                "type": "download_complete",
                "status": "completed",
                "huggingface_id": huggingface_id,
                "filename": filename,
                "model_format": model_format,
                "quantization": model_record.get("quantization")
                if model_record
                else None,
                "model_id": model_record.get("id") if model_record else None,
                "base_model_name": (
                    model_record.get("base_model_name") if model_record else None
                ),
                "pipeline_tag": (
                    model_record.get("pipeline_tag") if model_record else pipeline_tag
                ),
                "is_embedding_model": (
                    mm.model_is_embedding(model_record) if model_record else False
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
        mark_llama_swap_stale_after_download()

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
                    await save_safetensors_download(
                        store, huggingface_id, filename, file_path, file_size
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to record safetensors download for %s: %s",
                        filename,
                        exc,
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
        mark_llama_swap_stale_after_download()
    except Exception as exc:
        logger.error("Safetensors bundle download failed: %s", exc)
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
                await record_gguf_download_post_fetch(
                    store,
                    huggingface_id,
                    filename,
                    file_path,
                    file_size,
                    pipeline_tag=pipeline_tag,
                    aggregate_size=False,
                )
            except Exception as exc:
                logger.error("Failed to record GGUF download for %s: %s", filename, exc)

            bytes_completed += file_size
            bundle_model_bytes += file_size

        model_id = f"{huggingface_id.replace('/', '--')}--{quantization}"
        model_record = store.get_model(model_id)

        projector_filename = (projector or {}).get("filename")
        if projector_filename and model_record:
            projector_size_hint = max(int((projector or {}).get("size") or 0), 0)
            cached_projector = resolve_cached_model_path(
                huggingface_id, projector_filename
            )
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

        if model_record and bundle_model_bytes > 0:
            try:
                store.update_model(model_id, {"file_size": bundle_model_bytes})
                model_record = store.get_model(model_id) or model_record
            except Exception as size_exc:
                logger.warning(
                    "Failed to update aggregated GGUF size for %s: %s",
                    model_id,
                    size_exc,
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
        mark_llama_swap_stale_after_download()
    except Exception as exc:
        logger.error("GGUF bundle download failed: %s", exc)
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
            progress_manager.complete_task(
                task_id, f"Applied projector {mmproj_filename}"
            )
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
        mark_llama_swap_stale_after_download()
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


# Back-compat aliases for tests and routes that patch underscore names
_collect_safetensors_runtime_metadata = collect_safetensors_runtime_metadata
_save_safetensors_download = save_safetensors_download
_record_gguf_download_post_fetch = record_gguf_download_post_fetch
