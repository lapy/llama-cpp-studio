"""Provider aggregation, filtering, pagination, and cache keys."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

from backend.data_store import get_store
from backend.engines.registry import GGUF_ENGINE_IDS, HF_SNAPSHOT_ENGINE_IDS
from backend.model_catalog.audio_cpp_provider import AudioCppCatalogProvider
from backend.model_catalog.base import item_matches_filters, unique_strings
from backend.model_catalog.huggingface_provider import HuggingFaceCatalogProvider


SORTS = frozenset({
    "relevance",
    "downloads_desc",
    "downloads_asc",
    "likes_desc",
    "likes_asc",
    "name_asc",
    "name_desc",
})
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_catalog_sort(sort: str, query: str) -> str:
    """Default a typed query to best match. An empty query falls back to downloads."""
    value = str(sort or "").strip()
    if value not in ModelCatalogService.SORTS:
        value = "relevance" if str(query or "").strip() else "downloads_desc"
    if value == "relevance" and not str(query or "").strip():
        return "downloads_desc"
    return value


def _query_tokens(query: str) -> List[str]:
    return _TOKEN_RE.findall(str(query or "").lower())


def relevance_band(query: str, item: dict) -> int:
    """Lower is a closer match. An empty query does not change order."""
    raw = str(query or "").strip().lower()
    if not raw:
        return 0
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    names = [
        str(item.get("display_name") or "").lower(),
        str(source.get("id") or "").lower(),
        str(item.get("provider_item_id") or "").lower(),
    ]
    short_names = []
    for name in names:
        short_names.append(name)
        if "/" in name:
            short_names.append(name.rsplit("/", 1)[-1])
        if ":" in name:
            short_names.append(name.split(":", 1)[0])
    if raw in short_names:
        return 0
    if any(name.startswith(raw) or name.endswith("/" + raw) for name in short_names if name):
        return 1
    hay = " ".join(
        names + [str(item.get("description") or "").lower()]
    )
    tokens = _query_tokens(raw)
    if not tokens:
        return 4
    missing = sum(1 for token in tokens if token not in hay)
    if missing == 0:
        return 2
    return 3 + missing


def _metric(item: dict, key: str) -> int:
    raw = (item.get("metadata") or {}).get(key)
    if raw is None:
        raw = item.get(key)
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def sort_catalog_items(
    items: List[dict],
    *,
    query: str,
    sort: str,
    engine: str = "",
) -> List[dict]:
    """Order a merged catalog page. Relevance stays ahead of popularity when requested."""
    chosen = normalize_catalog_sort(sort, query)

    def key(item: dict):
        name = str(item.get("display_name") or "").lower()
        item_id = str(item.get("id") or "")
        engine_rank = (
            0
            if engine and engine in (item.get("compatible_engines") or [])
            else 1
        )
        available = 0 if item.get("unavailable_reason") is None else 1
        downloads = _metric(item, "downloads")
        likes = _metric(item, "likes")
        band = relevance_band(query, item) if chosen == "relevance" else 0
        if chosen == "downloads_asc":
            metric = (downloads, name)
        elif chosen == "likes_desc":
            metric = (-likes, -downloads, name)
        elif chosen == "likes_asc":
            metric = (likes, name)
        elif chosen == "name_asc":
            metric = (name,)
        elif chosen == "name_desc":
            metric = tuple(-ord(char) for char in name)
        else:
            metric = (-downloads, name)
        return (band, engine_rank, available, *metric, item_id)

    return sorted(items, key=key)


class ModelCatalogService:
    SORTS = SORTS
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

    def _cache_key(
        self, query: str, filters: dict, page: int, page_size: int, sort: str
    ) -> str:
        payload = {
            "query": str(query or "").strip().lower(),
            "filters": filters,
            "page": page,
            "page_size": page_size,
            "sort": sort,
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
        if engine in GGUF_ENGINE_IDS or engine in HF_SNAPSHOT_ENGINE_IDS:
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
        sort: str = "",
        force_refresh: bool = False,
    ) -> dict:
        filters = dict(filters or {})
        page = max(1, int(page or 1))
        page_size = min(100, max(1, int(page_size or 20)))
        sort = normalize_catalog_sort(sort, query)
        cache_key = self._cache_key(query, filters, page, page_size, sort)
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

        # One extra row reveals whether another page exists inside the fetch cap.
        window_end = min(100, page * page_size)
        requested_limit = window_end + 1 if window_end < 100 else window_end

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
        all_items = sort_catalog_items(
            all_items,
            query=query,
            sort=sort,
            engine=engine_filter,
        )
        start = (page - 1) * page_size
        has_more = start + page_size < len(all_items)
        page_items = all_items[start : start + page_size]
        total = len(all_items) if not has_more else start + page_size + 1
        payload = {
            "schema_version": 1,
            "items": page_items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": has_more,
            "sort": sort,
            "facets": self._facets(all_items),
            "provider_status": provider_status,
            "cache_key": cache_key,
            "filters": filters,
        }
        self._cache[cache_key] = (time.monotonic(), payload)
        return payload

