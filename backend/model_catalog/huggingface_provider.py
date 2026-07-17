"""Normalized wrapper around the existing Hugging Face search."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from backend.huggingface import search_models
from backend.model_catalog.base import normalized_item
from backend.model_schema import canonical_task


class HuggingFaceCatalogProvider:
    id = "huggingface"

    @staticmethod
    def _formats(filters: dict) -> List[str]:
        engine = str(filters.get("engine") or "")
        requested = str(filters.get("artifact_format") or filters.get("format") or "")
        if requested in {"gguf", "safetensors"}:
            return [requested]
        if engine in {"llama_cpp", "ik_llama"}:
            return ["gguf"]
        if engine in {"lmdeploy", "1cat_vllm"}:
            return ["safetensors"]
        if engine == "audio_cpp":
            # Arbitrary HF repos are deliberately never presented as verified
            # audio.cpp packages.
            return []
        return ["gguf", "safetensors"]

    @staticmethod
    def _quant_filenames(metadata: Dict[str, Any]) -> List[str]:
        """Resolve GGUF filenames from HF search quant metadata.

        Hugging Face search emits ``files: [{filename, size}, ...]``. Older
        shapes used ``filenames`` / ``filename``; accept all of them.
        """
        files = metadata.get("files")
        if isinstance(files, list) and files:
            names: List[str] = []
            for item in files:
                if isinstance(item, str) and item:
                    names.append(item)
                elif isinstance(item, dict):
                    name = item.get("filename")
                    if name:
                        names.append(str(name))
            if names:
                return names

        filenames = metadata.get("filenames")
        if isinstance(filenames, list) and filenames:
            return [str(item) for item in filenames if item]

        filename = metadata.get("filename")
        return [str(filename)] if filename else []

    @classmethod
    def _install_variants(cls, raw: Dict[str, Any], model_format: str) -> List[dict]:
        if model_format == "safetensors":
            files = raw.get("repo_files") or raw.get("safetensors_files") or []
            return [
                {
                    "id": "snapshot",
                    "label": "Hugging Face snapshot",
                    "method": "direct",
                    "installable": bool(files),
                    "files": files,
                    "size_bytes": raw.get("total_size") or raw.get("size"),
                }
            ]

        variants: List[dict] = []
        for name, metadata in (raw.get("quantizations") or {}).items():
            metadata = metadata if isinstance(metadata, dict) else {}
            filenames = cls._quant_filenames(metadata)
            variants.append(
                {
                    "id": name,
                    "label": name,
                    "method": "direct",
                    "installable": bool(filenames),
                    "files": filenames,
                    "size_bytes": metadata.get("total_size")
                    or metadata.get("size")
                    or metadata.get("file_size"),
                    "sharded": len(filenames) > 1,
                }
            )
        return variants

    @classmethod
    def _normalize(cls, raw: Dict[str, Any], model_format: str) -> dict:
        repository_id = str(raw.get("id") or raw.get("model_id") or "")
        pipeline_tag = canonical_task(raw.get("pipeline_tag"))
        tasks = [pipeline_tag] if pipeline_tag else []
        if raw.get("is_embedding_model") and "embeddings" not in tasks:
            tasks.append("embeddings")
        compatible = (
            ["llama_cpp", "ik_llama"]
            if model_format == "gguf"
            else ["lmdeploy", "1cat_vllm"]
        )
        variants = cls._install_variants(raw, model_format)
        total_size = sum(
            int(variant.get("size_bytes") or 0) for variant in variants
        ) or None
        return normalized_item(
            provider="huggingface",
            item_id=f"{repository_id}:{model_format}",
            display_name=raw.get("name") or repository_id,
            description=raw.get("description") or "",
            source={
                "provider": "huggingface",
                "id": repository_id,
                "revision": raw.get("sha"),
            },
            artifact_format=model_format,
            package_kind=(
                "single_file" if model_format == "gguf" else "hf_snapshot"
            ),
            tasks=tasks,
            family=raw.get("model_type"),
            features=[
                "embedding" if raw.get("is_embedding_model") else "",
                "multimodal" if raw.get("mmproj_files") else "",
            ],
            compatible_engines=compatible,
            compatibility={
                engine: {
                    "verified": True,
                    "evidence": [
                        f"Hugging Face {model_format} repository metadata",
                        "Studio artifact-format contract",
                    ],
                }
                for engine in compatible
            },
            install_variants=variants,
            size_bytes=total_size,
            gated=bool(raw.get("gated")),
            release_status="community",
            metadata={
                "downloads": raw.get("downloads"),
                "likes": raw.get("likes"),
                "last_modified": raw.get("last_modified"),
                "pipeline_tag": raw.get("pipeline_tag"),
                "raw": raw,
            },
        )

    async def search(self, query: str, limit: int, filters: dict) -> List[dict]:
        formats = self._formats(filters)
        if not formats:
            return []
        raw_groups = await asyncio.gather(
            *(search_models(query, limit, model_format=fmt) for fmt in formats)
        )
        output: List[dict] = []
        for model_format, rows in zip(formats, raw_groups):
            for raw in rows or []:
                if isinstance(raw, dict):
                    output.append(self._normalize(raw, model_format))
        return output

