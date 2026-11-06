"""Declared proxy statistics and token-counting capabilities per engine.

The declarations describe protocol adapters Studio implements, not a promise
that an installed build exposes an optional route. Runtime probes cache the
observed support per engine build revision; a 404 disables that route for the
revision instead of retrying it on every request.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Mapping

from backend.proxy.studio.observation import (
    PROTOCOL_NATIVE_AUDIO,
    PROTOCOL_NATIVE_LLAMA,
    PROTOCOL_NATIVE_SGLANG,
    PROTOCOL_OPENAI_CHAT,
    PROTOCOL_OPENAI_COMPLETIONS,
    PROTOCOL_OPENAI_EMBEDDINGS,
    PROTOCOL_OPENAI_RESPONSES,
)

LLAMA_FAMILY = "llama"
VLLM_FAMILY = "vllm"
SGLANG_FAMILY = "sglang"
LMDEPLOY_FAMILY = "lmdeploy"
AUDIO_FAMILY = "audio"


@dataclass(frozen=True)
class EngineCapabilities:
    engine: str
    family: str
    protocols: tuple[str, ...] = ()
    count_adapter: str = "none"
    cache_reporting: str = "optional"  # optional | unknown | not_applicable
    native_timing: bool = False
    timing_semantics: str = ""
    audio: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "engine": self.engine,
            "family": self.family,
            "protocols": list(self.protocols),
            "count_adapter": self.count_adapter,
            "cache_reporting": self.cache_reporting,
            "native_timing": self.native_timing,
            "timing_semantics": self.timing_semantics,
            "audio": self.audio,
            "notes": list(self.notes),
        }


_LLAMA_PROTOCOLS = (
    PROTOCOL_OPENAI_CHAT,
    PROTOCOL_OPENAI_COMPLETIONS,
    PROTOCOL_OPENAI_RESPONSES,
    PROTOCOL_OPENAI_EMBEDDINGS,
    PROTOCOL_NATIVE_LLAMA,
)
_VLLM_PROTOCOLS = (
    PROTOCOL_OPENAI_CHAT,
    PROTOCOL_OPENAI_COMPLETIONS,
    PROTOCOL_OPENAI_EMBEDDINGS,
    PROTOCOL_OPENAI_RESPONSES,
)
_SGLANG_PROTOCOLS = (
    PROTOCOL_OPENAI_CHAT,
    PROTOCOL_OPENAI_COMPLETIONS,
    PROTOCOL_OPENAI_RESPONSES,
    PROTOCOL_NATIVE_SGLANG,
)

ENGINE_CAPABILITIES: Mapping[str, EngineCapabilities] = {
    "llama_cpp": EngineCapabilities(
        engine="llama_cpp",
        family=LLAMA_FAMILY,
        protocols=_LLAMA_PROTOCOLS,
        count_adapter="llama_input_tokens",
        native_timing=True,
        timing_semantics="prompt_n processing work; n_gen-1 decode steps",
        notes=("Cached prompt tokens are a subset of the logical prompt total.",),
    ),
    "ik_llama": EngineCapabilities(
        engine="ik_llama",
        family=LLAMA_FAMILY,
        protocols=_LLAMA_PROTOCOLS,
        count_adapter="llama_input_tokens",
        native_timing=True,
        timing_semantics="n_decoded / generation time",
        notes=("Fork timing differs from mainline llama.cpp even for equal field names.",),
    ),
    "unsloth_llama": EngineCapabilities(
        engine="unsloth_llama",
        family=LLAMA_FAMILY,
        protocols=_LLAMA_PROTOCOLS,
        count_adapter="llama_input_tokens",
        native_timing=True,
        timing_semantics="n_gen-1 decode steps",
        notes=("Bind behavior to the installed revision, not the family label.",),
    ),
    "vllm": EngineCapabilities(
        engine="vllm",
        family=VLLM_FAMILY,
        protocols=_VLLM_PROTOCOLS,
        count_adapter="vllm_tokenize",
        notes=(
            "Stream usage requires include_usage; continuous usage is cumulative, not additive.",
            "Optional per-request metrics mix prefill into tokens_per_second.",
        ),
    ),
    "1cat_vllm": EngineCapabilities(
        engine="1cat_vllm",
        family=VLLM_FAMILY,
        protocols=_VLLM_PROTOCOLS,
        count_adapter="vllm_tokenize",
        notes=("Do not inherit mainline vLLM per-request timing without a build probe.",),
    ),
    "sglang": EngineCapabilities(
        engine="sglang",
        family=SGLANG_FAMILY,
        protocols=_SGLANG_PROTOCOLS,
        count_adapter="sglang_tokenize",
        native_timing=True,
        timing_semantics="meta_info server TTFT and request latency",
        notes=("First-output boundary can precede printable text.",),
    ),
    "sglang_v100": EngineCapabilities(
        engine="sglang_v100",
        family=SGLANG_FAMILY,
        protocols=_SGLANG_PROTOCOLS,
        count_adapter="sglang_tokenize",
        native_timing=True,
        timing_semantics="fork meta_info; independent fixtures required",
        notes=("Usage-processor details differ from current mainline SGLang.",),
    ),
    "lmdeploy": EngineCapabilities(
        engine="lmdeploy",
        family=LMDEPLOY_FAMILY,
        protocols=(PROTOCOL_OPENAI_CHAT, PROTOCOL_OPENAI_COMPLETIONS),
        count_adapter="none",
        cache_reporting="optional",
        notes=(
            "No verified chat/prompt counting endpoint; exact mode must fail closed.",
            "Operational cache counters are not per-request cached-token counts.",
        ),
    ),
    "audio_cpp": EngineCapabilities(
        engine="audio_cpp",
        family=AUDIO_FAMILY,
        protocols=(PROTOCOL_NATIVE_AUDIO,),
        count_adapter="none",
        cache_reporting="not_applicable",
        native_timing=True,
        timing_semantics="audio wall time and RTF",
        audio=True,
        notes=("Audio frames/codec tokens are not LLM output-token charges.",),
    ),
}

UNKNOWN_CAPABILITIES = EngineCapabilities(engine="", family="unknown")


def capabilities_for(engine: str | None) -> EngineCapabilities:
    if not engine:
        return UNKNOWN_CAPABILITIES
    return ENGINE_CAPABILITIES.get(str(engine), UNKNOWN_CAPABILITIES)


def count_adapter_for(engine: str | None) -> str:
    return capabilities_for(engine).count_adapter


class ProbeCache:
    """Per engine/build-revision cache of observed optional-route support."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str, str], bool] = {}

    @staticmethod
    def key(engine: str, revision: str, endpoint: str) -> tuple[str, str, str]:
        return (str(engine), str(revision or ""), str(endpoint))

    def get(self, engine: str, revision: str, endpoint: str) -> bool | None:
        with self._lock:
            return self._entries.get(self.key(engine, revision, endpoint))

    def record(self, engine: str, revision: str, endpoint: str, supported: bool) -> None:
        with self._lock:
            self._entries[self.key(engine, revision, endpoint)] = bool(supported)

    def invalidate(self, engine: str | None = None, revision: str | None = None) -> None:
        with self._lock:
            if engine is None:
                self._entries.clear()
                return
            for key in list(self._entries):
                if key[0] != engine:
                    continue
                if revision is not None and key[1] != revision:
                    continue
                self._entries.pop(key, None)


probe_cache = ProbeCache()
