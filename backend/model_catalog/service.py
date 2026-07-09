"""Provider aggregation, filtering, pagination, and cache keys."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Tuple

from backend.data_store import get_store
from backend.model_catalog.audio_cpp_provider import AudioCppCatalogProvider
from backend.model_catalog.base import item_matches_filters, unique_strings
from backend.model_catalog.huggingface_provider import HuggingFaceCatalogProvider


class ModelCatalogService:
    _cache: Dict[str, Tuple[float, dict]] = {}
    cache_ttl = 300.0

    def __init__(self, store=None):
        self.store = store or get_store()

    def _version_token(self) -> dict:
        active_audio = self.store.get_active_engine_version("audio_cpp") or {}
        config_dir = getattr(self.store, "_config_dir", "")
        catalog_mtime = 0
        if config_dir:
            try:
                catalog_mtime = os.stat(
                    os.path.join(config_dir, "engine_params_catalog.yaml")
                ).st_mtime_ns
            except OSError:
                pass
        return {
            "audio_cpp": active_audio.get("source_commit")
            or active_audio.get("version"),
            "engine_catalog_mtime": catalog_mtime,
        }

    def _cache_key(self, query: str, filters: dict, page: int, page_size: int) -> str:
        payload = {
            "query": str(query or "").strip().lower(),
            "filters": filters,
            "page": page,
            "page_size": page_size,
            "versions": self._version_token(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _provider_ids(filters: dict) -> List[str]:
        requested = str(filters.get("provider") or filters.get("source") or "")
        if requested in {"huggingface", "audio_cpp"}:
            return [requested]
        engine = str(filters.get("engine") or "")
        if engine == "audio_cpp":
            return ["audio_cpp"]
        if engine in {"llama_cpp", "ik_llama", "lmdeploy", "1cat_vllm"}:
            return ["huggingface"]
        return ["audio_cpp", "huggingface"]

    @staticmethod
    def _facets(items: List[dict]) -> dict:
        return {
            "engines": unique_strings(
                engine
                for item in items
                for engine in item.get("compatible_engines") or []
            ),
            "tasks": unique_strings(
                task for item in items for task in item.get("tasks") or []
            ),
            "input_modalities": unique_strings(
                modality
                for item in items
                for modality in item.get("input_modalities") or []
            ),
            "output_modalities": unique_strings(
                modality
                for item in items
                for modality in item.get("output_modalities") or []
            ),
            "features": unique_strings(
                feature for item in items for feature in item.get("features") or []
            ),
            "providers": unique_strings(item.get("provider") for item in items),
            "package_kinds": unique_strings(
                item.get("package_kind") for item in items
            ),
            "install_methods": unique_strings(
                variant.get("method")
                for item in items
                for variant in item.get("install_variants") or []
            ),
            "languages": unique_strings(
                language for item in items for language in item.get("languages") or []
            ),
        }

    async def search(
        self,
        *,
        query: str = "",
        filters: dict | None = None,
        page: int = 1,
        page_size: int = 20,
        force_refresh: bool = False,
    ) -> dict:
        filters = dict(filters or {})
        page = max(1, int(page or 1))
        page_size = min(100, max(1, int(page_size or 20)))
        cache_key = self._cache_key(query, filters, page, page_size)
        cached = self._cache.get(cache_key)
        if (
            not force_refresh
            and cached
            and time.monotonic() - cached[0] < self.cache_ttl
        ):
            return cached[1]

        provider_ids = self._provider_ids(filters)
        providers: Dict[str, Any] = {}
        if "audio_cpp" in provider_ids:
            providers["audio_cpp"] = AudioCppCatalogProvider(self.store)
        if "huggingface" in provider_ids:
            providers["huggingface"] = HuggingFaceCatalogProvider()

        requested_limit = min(100, max(page * page_size, page_size))

        async def _run(provider_id: str, provider: Any):
            try:
                return provider_id, await provider.search(
                    query, requested_limit, filters
                ), None
            except Exception as exc:
                return provider_id, [], str(exc)

        results = await asyncio.gather(
            *(_run(provider_id, provider) for provider_id, provider in providers.items())
        )
        all_items: List[dict] = []
        provider_status: Dict[str, dict] = {}
        for provider_id, items, error in results:
            provider = providers[provider_id]
            status = dict(getattr(provider, "status", {}) or {})
            if error:
                status.update({"available": False, "reason": error})
            elif "available" not in status:
                status["available"] = True
            provider_status[provider_id] = status
            all_items.extend(
                item for item in items if item_matches_filters(item, filters)
            )

        engine_filter = str(filters.get("engine") or "")
        all_items.sort(
            key=lambda item: (
                0
                if engine_filter
                and engine_filter in (item.get("compatible_engines") or [])
                else 1,
                0 if item.get("unavailable_reason") is None else 1,
                -int((item.get("metadata") or {}).get("downloads") or 0),
                str(item.get("display_name") or "").lower(),
                str(item.get("id") or ""),
            )
        )
        total = len(all_items)
        start = (page - 1) * page_size
        page_items = all_items[start : start + page_size]
        payload = {
            "schema_version": 1,
            "items": page_items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": start + page_size < total,
            "facets": self._facets(all_items),
            "provider_status": provider_status,
            "cache_key": cache_key,
            "filters": filters,
        }
        self._cache[cache_key] = (time.monotonic(), payload)
        return payload

