"""Versioned model-record normalization and compatibility helpers."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.engine_registry import ENGINE_REGISTRY


MODEL_SCHEMA_VERSION = 2

_PIPELINE_TASK_ALIASES = {
    "automatic-speech-recognition": "asr",
    "text-to-speech": "tts",
    "text-to-audio": "gen",
    "audio-to-audio": "s2s",
    "voice-activity-detection": "vad",
    "audio-classification": "audio-classification",
    "text-generation": "text-generation",
    "text2text-generation": "text2text-generation",
    "feature-extraction": "embeddings",
    "sentence-similarity": "embeddings",
}

_TASK_MODALITIES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "asr": (("audio",), ("text", "segments")),
    "tts": (("text", "audio"), ("audio",)),
    "gen": (("text", "audio"), ("audio",)),
    "s2s": (("audio",), ("audio",)),
    "vc": (("audio",), ("audio",)),
    "svc": (("audio",), ("audio",)),
    "clon": (("text", "audio"), ("audio",)),
    "vdes": (("text",), ("audio",)),
    "vad": (("audio",), ("events",)),
    "diar": (("audio",), ("segments",)),
    "sep": (("audio",), ("audio",)),
    "align": (("audio", "text"), ("segments",)),
    "spk": (("audio",), ("embedding",)),
    "pipeline": (("text", "audio"), ("text", "audio", "segments", "events")),
    "text-generation": (("text",), ("text",)),
    "text2text-generation": (("text",), ("text",)),
    "embeddings": (("text",), ("embedding",)),
    "reranking": (("text",), ("score",)),
}


def _unique_strings(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def canonical_task(value: Any) -> str:
    task = str(value or "").strip().lower()
    return _PIPELINE_TASK_ALIASES.get(task, task)


def _tasks_for_record(record: Dict[str, Any]) -> List[str]:
    raw = record.get("tasks")
    if isinstance(raw, (list, tuple, set)):
        tasks = _unique_strings(canonical_task(item) for item in raw)
    else:
        tasks = []
    if not tasks and record.get("task"):
        tasks = _unique_strings([canonical_task(record.get("task"))])
    if not tasks and record.get("pipeline_tag"):
        tasks = _unique_strings([canonical_task(record.get("pipeline_tag"))])
    return tasks


def _modalities_for_tasks(tasks: Iterable[str]) -> Tuple[List[str], List[str]]:
    inputs: List[str] = []
    outputs: List[str] = []
    for task in tasks:
        task_inputs, task_outputs = _TASK_MODALITIES.get(task, ((), ()))
        inputs.extend(task_inputs)
        outputs.extend(task_outputs)
    return _unique_strings(inputs), _unique_strings(outputs)


def _legacy_format(record: Dict[str, Any]) -> str:
    artifact = record.get("artifact")
    if isinstance(artifact, dict) and artifact.get("format"):
        return str(artifact["format"]).strip().lower()
    return str(record.get("format") or record.get("model_format") or "").strip().lower()


def _default_package_kind(record: Dict[str, Any], artifact_format: str) -> str:
    if record.get("engine") == "audio_cpp" or record.get("family"):
        if artifact_format in {"mixed", "original"}:
            return "prepared_bundle"
    if artifact_format == "gguf":
        return "sharded_bundle" if record.get("files") else "single_file"
    if artifact_format == "safetensors":
        return "hf_snapshot"
    return "prepared_bundle" if record.get("local_path") else "unknown"


def compatible_engines_for_record(record: Dict[str, Any]) -> List[str]:
    """Return verified/derivable engine compatibility for a model record.

    Audio compatibility is never inferred from ``safetensors`` alone.  It must be
    explicitly recorded by the curated package installer or local inspection.
    """
    explicit = record.get("compatible_engines")
    if not isinstance(explicit, (list, tuple, set)):
        explicit = record.get("engine_compatibility")
    if isinstance(explicit, (list, tuple, set)):
        return [
            item
            for item in _unique_strings(explicit)
            if item in ENGINE_REGISTRY
        ]

    artifact_format = _legacy_format(record)
    if artifact_format == "gguf":
        return ["llama_cpp", "ik_llama"]
    if artifact_format == "safetensors":
        return ["lmdeploy", "1cat_vllm"]
    return []


def normalize_model_record(model: Dict[str, Any]) -> Dict[str, Any]:
    """Return a schema-v2 record while retaining all legacy top-level fields."""
    record = copy.deepcopy(model if isinstance(model, dict) else {})
    record["schema_version"] = MODEL_SCHEMA_VERSION

    source = record.get("source")
    if not isinstance(source, dict):
        source = {}
    source = dict(source)
    if not source.get("provider") and record.get("huggingface_id"):
        source["provider"] = "huggingface"
    if not source.get("id") and record.get("huggingface_id"):
        source["id"] = record["huggingface_id"]
    if source:
        record["source"] = source

    artifact_format = _legacy_format(record)
    artifact = record.get("artifact")
    if not isinstance(artifact, dict):
        artifact = {}
    artifact = dict(artifact)
    if artifact_format:
        artifact.setdefault("format", artifact_format)
    artifact.setdefault(
        "package_kind", _default_package_kind(record, artifact_format)
    )
    local_path = (
        record.get("local_path")
        or record.get("model_path")
        or record.get("path")
    )
    if local_path:
        artifact.setdefault("path", local_path)
    if record.get("file_size") is not None:
        artifact.setdefault("size", record.get("file_size"))
    record["artifact"] = artifact

    tasks = _tasks_for_record(record)
    if tasks:
        record["tasks"] = tasks
        record.setdefault("task", tasks[0])

    existing_inputs = record.get("input_modalities")
    existing_outputs = record.get("output_modalities")
    derived_inputs, derived_outputs = _modalities_for_tasks(tasks)
    record["input_modalities"] = _unique_strings(
        existing_inputs if isinstance(existing_inputs, list) else derived_inputs
    )
    record["output_modalities"] = _unique_strings(
        existing_outputs if isinstance(existing_outputs, list) else derived_outputs
    )

    compatible = compatible_engines_for_record(record)
    record["compatible_engines"] = compatible
    return record


def migrate_models_document(data: Any) -> Tuple[Dict[str, Any], bool]:
    """Normalize a ``models.yaml`` root and report whether it changed."""
    root = copy.deepcopy(data if isinstance(data, dict) else {})
    raw_models = root.get("models")
    if not isinstance(raw_models, list):
        raw_models = []
    models = [
        normalize_model_record(item)
        for item in raw_models
        if isinstance(item, dict)
    ]
    migrated = {"schema_version": MODEL_SCHEMA_VERSION, "models": models}
    for key, value in root.items():
        if key not in migrated:
            migrated[key] = value
    return migrated, migrated != root

