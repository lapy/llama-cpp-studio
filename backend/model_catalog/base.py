"""Shared normalized catalog result helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


TASK_MODALITIES: Dict[str, Tuple[List[str], List[str]]] = {
    "asr": (["audio"], ["text", "segments"]),
    "tts": (["text", "audio"], ["audio"]),
    "clon": (["text", "audio"], ["audio"]),
    "vdes": (["text"], ["audio"]),
    "vad": (["audio"], ["events"]),
    "diar": (["audio"], ["segments"]),
    "sep": (["audio"], ["audio"]),
    "gen": (["text", "audio"], ["audio"]),
    "vc": (["audio"], ["audio"]),
    "s2s": (["audio"], ["audio"]),
    "svc": (["audio"], ["audio"]),
    "align": (["audio", "text"], ["segments"]),
    "spk": (["audio"], ["embedding"]),
    "text-generation": (["text"], ["text"]),
    "text2text-generation": (["text"], ["text"]),
    "embeddings": (["text"], ["embedding"]),
}


def unique_strings(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def modalities_for_tasks(tasks: Iterable[str]) -> Tuple[List[str], List[str]]:
    inputs: List[str] = []
    outputs: List[str] = []
    for task in tasks:
        task_inputs, task_outputs = TASK_MODALITIES.get(str(task), ([], []))
        inputs.extend(task_inputs)
        outputs.extend(task_outputs)
    return unique_strings(inputs), unique_strings(outputs)


def normalized_item(
    *,
    provider: str,
    item_id: str,
    display_name: str,
    source: dict,
    description: str = "",
    artifact_format: str,
    package_kind: str,
    tasks: Optional[Iterable[str]] = None,
    family: Optional[str] = None,
    modes: Optional[Iterable[str]] = None,
    languages: Optional[Iterable[str]] = None,
    features: Optional[Iterable[str]] = None,
    compatible_engines: Optional[Iterable[str]] = None,
    compatibility: Optional[dict] = None,
    install_variants: Optional[List[dict]] = None,
    size_bytes: Optional[int] = None,
    gated: bool = False,
    release_status: str = "stable",
    unavailable_reason: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    task_list = unique_strings(tasks or [])
    input_modalities, output_modalities = modalities_for_tasks(task_list)
    return {
        "id": f"{provider}:{item_id}",
        "provider": provider,
        "provider_item_id": item_id,
        "display_name": display_name or item_id,
        "description": description or "",
        "source": source,
        "artifact_format": artifact_format,
        "package_kind": package_kind,
        "family": family,
        "tasks": task_list,
        "modes": unique_strings(modes or []),
        "input_modalities": input_modalities,
        "output_modalities": output_modalities,
        "features": unique_strings(features or []),
        "languages": unique_strings(languages or []),
        "compatible_engines": unique_strings(compatible_engines or []),
        "compatibility": compatibility or {},
        "install_variants": list(install_variants or []),
        "size_bytes": size_bytes,
        "gated": bool(gated),
        "release_status": release_status,
        "unavailable_reason": unavailable_reason,
        "metadata": metadata or {},
    }


def item_matches_filters(item: dict, filters: dict) -> bool:
    engine = str(filters.get("engine") or "").strip()
    if engine and engine not in (item.get("compatible_engines") or []):
        return False
    task = str(filters.get("task") or "").strip()
    if task and task not in (item.get("tasks") or []):
        return False
    input_modality = str(filters.get("input_modality") or "").strip()
    if input_modality and input_modality not in (item.get("input_modalities") or []):
        return False
    output_modality = str(filters.get("output_modality") or "").strip()
    if output_modality and output_modality not in (item.get("output_modalities") or []):
        return False
    feature = str(filters.get("feature") or "").strip()
    if feature and feature not in (item.get("features") or []):
        return False
    provider = str(filters.get("provider") or filters.get("source") or "").strip()
    if provider and provider != item.get("provider"):
        return False
    package_kind = str(filters.get("package_kind") or "").strip()
    if package_kind and package_kind != item.get("package_kind"):
        return False
    artifact_format = str(
        filters.get("artifact_format") or filters.get("format") or ""
    ).strip()
    if artifact_format and artifact_format != item.get("artifact_format"):
        return False
    install_method = str(filters.get("install_method") or "").strip()
    if install_method and not any(
        variant.get("method") == install_method
        for variant in item.get("install_variants") or []
    ):
        return False
    release_status = str(filters.get("release_status") or "").strip()
    if release_status and release_status != item.get("release_status"):
        return False
    language = str(filters.get("language") or "").strip()
    if language and language not in (item.get("languages") or []):
        return False
    gated = filters.get("gated")
    if isinstance(gated, bool) and gated != bool(item.get("gated")):
        return False
    return True

