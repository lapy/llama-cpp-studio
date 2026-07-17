"""Declarative inference-engine capabilities shared by backend and frontend APIs.

The registry intentionally contains data only.  Engine-specific installers, scanners,
and runtime adapters remain in their own modules and are selected by the string keys
stored here.  This keeps model/config validation independent from heavyweight manager
imports and avoids circular dependencies with the YAML data store.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple


@dataclass(frozen=True)
class EngineSpec:
    id: str
    label: str
    repository_source: str
    install_kind: str
    runtime_kind: str
    scanner_kind: str
    active_path_fields: Tuple[str, ...]
    artifact_formats: FrozenSet[str]
    package_kinds: FrozenSet[str]
    tasks: FrozenSet[str]
    input_modalities: FrozenSet[str]
    output_modalities: FrozenSet[str]
    supports_embeddings: bool = False
    experimental: bool = False
    # Per-surface maturity labels (e.g. speech_asr=stable). When set, ``experimental``
    # is only a coarse UI hint — prefer maturity_surfaces for honesty.
    maturity_surfaces: Tuple[Tuple[str, str], ...] = ()

    def to_api_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "artifact_formats",
            "package_kinds",
            "tasks",
            "input_modalities",
            "output_modalities",
        ):
            payload[key] = sorted(payload[key])
        payload["active_path_fields"] = list(self.active_path_fields)
        payload["maturity_surfaces"] = {
            key: value for key, value in self.maturity_surfaces
        }
        return payload


_TEXT_GENERATION_TASKS = frozenset(
    {
        "text-generation",
        "text2text-generation",
        "embeddings",
        "reranking",
    }
)

_AUDIO_TASKS = frozenset(
    {
        "vad",
        "asr",
        "diar",
        "sep",
        "gen",
        "tts",
        "clon",
        "vc",
        "s2s",
        "align",
        "vdes",
        "spk",
        "svc",
        "pipeline",
    }
)


ENGINE_REGISTRY: Dict[str, EngineSpec] = {
    "llama_cpp": EngineSpec(
        id="llama_cpp",
        label="llama.cpp",
        repository_source="llama.cpp",
        install_kind="cmake",
        runtime_kind="llama_server",
        scanner_kind="llama_help",
        active_path_fields=("binary_path",),
        artifact_formats=frozenset({"gguf"}),
        package_kinds=frozenset({"single_file", "sharded_bundle"}),
        tasks=_TEXT_GENERATION_TASKS,
        input_modalities=frozenset({"text", "image"}),
        output_modalities=frozenset({"text", "embedding", "score"}),
        supports_embeddings=True,
    ),
    "ik_llama": EngineSpec(
        id="ik_llama",
        label="ik_llama.cpp",
        repository_source="ik_llama.cpp",
        install_kind="cmake",
        runtime_kind="llama_server",
        scanner_kind="llama_help",
        active_path_fields=("binary_path",),
        artifact_formats=frozenset({"gguf"}),
        package_kinds=frozenset({"single_file", "sharded_bundle"}),
        tasks=_TEXT_GENERATION_TASKS,
        input_modalities=frozenset({"text", "image"}),
        output_modalities=frozenset({"text", "embedding", "score"}),
        supports_embeddings=True,
    ),
    "lmdeploy": EngineSpec(
        id="lmdeploy",
        label="LMDeploy",
        repository_source="LMDeploy",
        install_kind="python_venv",
        runtime_kind="lmdeploy",
        scanner_kind="lmdeploy_help",
        active_path_fields=("venv_path",),
        artifact_formats=frozenset({"safetensors"}),
        package_kinds=frozenset({"hf_snapshot"}),
        tasks=_TEXT_GENERATION_TASKS,
        input_modalities=frozenset({"text", "image"}),
        output_modalities=frozenset({"text", "embedding"}),
    ),
    "1cat_vllm": EngineSpec(
        id="1cat_vllm",
        label="1Cat-vLLM",
        repository_source="1Cat-vLLM",
        install_kind="python_venv",
        runtime_kind="onecat_vllm",
        scanner_kind="vllm_help",
        active_path_fields=("venv_path",),
        artifact_formats=frozenset({"safetensors"}),
        package_kinds=frozenset({"hf_snapshot"}),
        tasks=_TEXT_GENERATION_TASKS,
        input_modalities=frozenset({"text", "image"}),
        output_modalities=frozenset({"text", "embedding"}),
    ),
    "audio_cpp": EngineSpec(
        id="audio_cpp",
        label="audio.cpp",
        repository_source="audio.cpp",
        install_kind="cmake",
        runtime_kind="audio_cpp",
        scanner_kind="audio_cpp",
        active_path_fields=("server_binary_path", "cli_binary_path"),
        artifact_formats=frozenset({"mixed", "safetensors", "original"}),
        package_kinds=frozenset({"prepared_bundle", "hf_snapshot"}),
        tasks=_AUDIO_TASKS,
        input_modalities=frozenset({"text", "audio"}),
        output_modalities=frozenset({"text", "audio", "segments", "events"}),
        # Not a blanket experimental engine: speech/ASR via llama-swap is the primary path.
        experimental=False,
        maturity_surfaces=(
            ("speech_asr", "stable"),
            ("generic_tasks", "limited"),
            ("catalog_json", "stable"),
            ("heuristic_discovery", "experimental"),
        ),
    ),
}


VALID_ENGINE_IDS: FrozenSet[str] = frozenset(ENGINE_REGISTRY)
EMBEDDINGS_ENGINE_IDS: FrozenSet[str] = frozenset(
    key for key, spec in ENGINE_REGISTRY.items() if spec.supports_embeddings
)
NATIVE_ENGINE_IDS: FrozenSet[str] = frozenset(
    key for key, spec in ENGINE_REGISTRY.items() if spec.install_kind == "cmake"
)
GGUF_ENGINE_IDS: FrozenSet[str] = frozenset(
    key for key, spec in ENGINE_REGISTRY.items() if "gguf" in spec.artifact_formats
)


def get_engine_spec(engine_id: Optional[str]) -> Optional[EngineSpec]:
    return ENGINE_REGISTRY.get(str(engine_id or ""))


def engine_specs() -> Iterable[EngineSpec]:
    return ENGINE_REGISTRY.values()


def engine_registry_payload() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "engines": [spec.to_api_dict() for spec in ENGINE_REGISTRY.values()],
    }


def active_engine_row_is_runnable(engine_id: str, row: Optional[dict]) -> bool:
    """Return whether a stored active-version row has the paths its engine needs."""
    spec = get_engine_spec(engine_id)
    if not spec or not isinstance(row, dict):
        return False
    return all(bool(row.get(field)) for field in spec.active_path_fields)

