"""Studio-owned operation registry for the audio.cpp integration contract.

Shim for upstream issue 0xShug0/audio.cpp#852: until audio.cpp publishes a
machine-readable server contract, every route/encoding fact Studio needs lives
here exactly once, with provenance ``studio``. Dynamic engine observations
(``--list-loaders --json``, ``--inspect``, ``model_specs``) overlay these
entries in :mod:`backend.engines.audio_cpp.integration.composer`; observed
data always beats a ``studio`` entry, and drift is reported as a warning.

Paths here are canonical *native* audio.cpp paths. The llama-swap public
alias for the generic task route is derived once (:data:`TASKS_RUN_ALIAS`)
instead of being mirrored per consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

CONTRACT_SCHEMA = "audio-cpp.studio-contract/v0"

# llama-swap exposes the native generic-task route on its own public prefix and
# rewrites it upstream. Everything else keeps its native path through swap.
TASKS_RUN_ALIAS = "/audioapi/v1/tasks/run"
# Base prefix of the native audio route family, used by the catch-all
# passthrough for engine routes Studio has no operation for yet.
NATIVE_AUDIO_BASE = "/v1/audio"


@dataclass(frozen=True)
class Operation:
    """One audio.cpp server operation Studio knows how to reach."""

    id: str
    native_path: str
    studio_path: str
    method: str
    encodings: Tuple[str, ...]
    responses: Tuple[str, ...]
    errors: str
    serves_tasks: Tuple[str, ...]
    provenance: str = "studio"
    # True when the route is absent from llama-swap's route table and Studio
    # must reach it through the ``/upstream/{model}`` passthrough.
    upstream_only: bool = False

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "native_path": self.native_path,
            "studio_path": self.studio_path,
            "method": self.method,
            "encodings": list(self.encodings),
            "responses": list(self.responses),
            "errors": self.errors,
            "serves_tasks": list(self.serves_tasks),
            "provenance": self.provenance,
            "upstream_only": self.upstream_only,
            # Derived once so the frontend renders request bodies and defaults
            # buckets from the contract instead of endpoint-string heuristics.
            "request_defaults_key": DEFAULTS_KEYS.get(self.id, "task_defaults"),
            "body_shape": _body_shape(self),
        }


#: Operations the client submits as multipart form uploads (file field + fields).
_MULTIPART_UPLOAD_OPERATIONS = frozenset(
    {"transcriptions", "transcriptions_details", "alignments", "batch_transcriptions"}
)


def _body_shape(op: Operation) -> str:
    """Request-body envelope the client must build for this operation.

    ``multipart-form`` takes FormData (file field plus scalar fields),
    ``tasks-run`` wraps the fields inside a ``{"model","request"}`` object, and
    everything else is a flat OpenAI-shaped JSON body.
    """
    if op.id == "tasks_run":
        return "tasks-run"
    if op.id in _MULTIPART_UPLOAD_OPERATIONS:
        return "multipart-form"
    return "openai-json"


# The version-wide operation table. ``serves_tasks`` is the deterministic
# task->route map: a task not claimed by a dedicated operation falls through
# to ``tasks_run``. No fuzzy matching happens anywhere in this file.
OPERATIONS: Tuple[Operation, ...] = (
    Operation(
        id="speech",
        native_path="/v1/audio/speech",
        studio_path="/v1/audio/speech",
        method="POST",
        encodings=("json",),
        responses=("audio", "json"),
        errors="openai-envelope",
        serves_tasks=("tts", "vdes", "clon"),
    ),
    Operation(
        id="transcriptions",
        native_path="/v1/audio/transcriptions",
        studio_path="/v1/audio/transcriptions",
        method="POST",
        encodings=("json", "multipart"),
        responses=("json",),
        errors="openai-envelope",
        serves_tasks=("asr", "stt"),
    ),
    Operation(
        id="transcriptions_details",
        native_path="/v1/audio/transcriptions/details",
        studio_path="/v1/audio/transcriptions/details",
        method="POST",
        encodings=("json", "multipart"),
        responses=("json",),
        errors="openai-envelope",
        serves_tasks=("transcribe_details",),
        upstream_only=True,
    ),
    Operation(
        id="alignments",
        native_path="/v1/audio/alignments",
        studio_path="/v1/audio/alignments",
        method="POST",
        encodings=("json", "multipart"),
        responses=("json",),
        errors="openai-envelope",
        serves_tasks=("align", "alignment"),
        upstream_only=True,
    ),
    Operation(
        id="batch_transcriptions",
        native_path="/v1/batches/transcriptions",
        studio_path="/v1/batches/transcriptions",
        method="POST",
        encodings=("multipart", "json"),
        responses=("json", "sse-events"),
        errors="openai-envelope",
        serves_tasks=("batch_transcribe",),
        upstream_only=True,
    ),
    Operation(
        id="voices",
        native_path="/v1/audio/voices",
        studio_path="/v1/audio/voices",
        method="GET",
        encodings=(),
        responses=("json",),
        errors="openai-envelope",
        serves_tasks=(),
    ),
    Operation(
        id="models",
        native_path="/v1/models",
        studio_path="/v1/models",
        method="GET",
        encodings=(),
        responses=("json",),
        errors="openai-envelope",
        serves_tasks=(),
    ),
    Operation(
        id="health",
        native_path="/health",
        studio_path="/health",
        method="GET",
        encodings=(),
        responses=("json",),
        errors="plain",
        serves_tasks=(),
    ),
    Operation(
        id="tasks_run",
        native_path="/v1/tasks/run",
        studio_path=TASKS_RUN_ALIAS,
        method="POST",
        encodings=("json", "multipart"),
        responses=("json", "audio", "sse-events"),
        errors="audio-cpp-error",
        # Deterministic fall-through route: everything the engine does not
        # map to a dedicated OpenAI-shaped route (vc, svc, s2s, vad, ...).
        serves_tasks=("vc", "svc", "s2s", "vad", "diar", "sep", "gen"),
    ),
)

_OPERATIONS_BY_ID: Dict[str, Operation] = {op.id: op for op in OPERATIONS}
_OPERATIONS_BY_NATIVE_PATH: Dict[str, Operation] = {
    op.native_path: op for op in OPERATIONS
}
_OPERATIONS_BY_STUDIO_PATH: Dict[str, Operation] = {
    op.studio_path: op for op in OPERATIONS
}

# Studio request-defaults bucket per request operation. This is a Studio
# persistence concept (which config blob holds the saved defaults), never an
# engine-facing path, so it is derived from the registry in one place.
DEFAULTS_KEYS: Dict[str, str] = {
    "speech": "speech_defaults",
    "transcriptions": "transcription_defaults",
    "transcriptions_details": "transcription_defaults",
    "alignments": "task_defaults",
    "batch_transcriptions": "task_defaults",
    "tasks_run": "task_defaults",
}

# Studio owns these CLI flags for every audio.cpp launch: the user may not set
# them through custom args or model config, in either direction (runtime
# injection here, config filtering in backend/audio/model_config.py).
STUDIO_OWNED_FLAGS = frozenset(
    {
        "--config",
        "--host",
        "--port",
        "--model",
        "--backend",
        "--device",
        "--threads",
        "--lazy-load",
        "--model-spec-override",
        "--model-spec",
        "--ui",
        "--no-ui",
        "--ui-management",
    }
)

_FALLBACK_OPERATION = "tasks_run"


def operations() -> List[Operation]:
    return list(OPERATIONS)


def operation_by_id(op_id: str) -> Optional[Operation]:
    return _OPERATIONS_BY_ID.get(str(op_id or "").strip())


def operation_for_native_path(path: str) -> Optional[Operation]:
    return _OPERATIONS_BY_NATIVE_PATH.get(str(path or "").strip())


def operation_for_studio_path(path: str) -> Optional[Operation]:
    return _OPERATIONS_BY_STUDIO_PATH.get(str(path or "").strip())


def operation_for_task(task: str) -> Operation:
    """Deterministic task -> operation lookup with generic fall-through."""
    key = str(task or "").strip().lower()
    if key:
        for op in OPERATIONS:
            if key in op.serves_tasks:
                return op
    return _OPERATIONS_BY_ID[_FALLBACK_OPERATION]


#: Coarse surface labels an engine may report instead of a native path. Keys are
#: the vocabulary ``--inspect``/``--list-loaders`` use plus the native and studio
#: path spellings, so callers can hand over whatever the engine emitted.
_ENDPOINT_TOKENS: Dict[str, str] = {
    "speech": "speech",
    "audio/speech": "speech",
    "tts": "speech",
    "transcriptions": "transcriptions",
    "transcribe": "transcriptions",
    "asr": "transcriptions",
    "transcriptions/details": "transcriptions_details",
    "details": "transcriptions_details",
    "alignments": "alignments",
    "align": "alignments",
    "batch": "batch_transcriptions",
    "batches": "batch_transcriptions",
    "batch_transcriptions": "batch_transcriptions",
    "tasks": "tasks_run",
    "tasks_run": "tasks_run",
    "tasks/run": "tasks_run",
    "audioapi/v1/tasks/run": "tasks_run",
    "generic": "tasks_run",
    "voices": "voices",
    "models": "models",
    "health": "health",
}


def resolve_operation(
    *,
    task: Optional[str] = None,
    preferred_api_endpoint: Optional[str] = None,
    family_endpoints: Optional[Sequence[str]] = None,
) -> Operation:
    """Resolve the serving operation for one request surface.

    Priority is strictly: explicit engine-observed endpoint (``--inspect``
    ``preferred_api_endpoint``), then the family's advertised
    ``api_endpoints`` from ``--list-loaders --json``, then the task table.
    Engine-observed native paths that are not in the registry still resolve
    through :func:`operation_for_native_path` -> fall-through, so an entirely
    new upstream route needs no Studio change once it is advertised.
    """
    for candidate in (preferred_api_endpoint, *(family_endpoints or [])):
        text = str(candidate or "").strip().lstrip("/")
        if not text:
            continue
        if text in (TASKS_RUN_ALIAS.lstrip("/"), "v1/tasks/run"):
            return _OPERATIONS_BY_ID["tasks_run"]
        op = operation_for_native_path(
            f"/{text}"
        ) or operation_for_studio_path(f"/{text}")
        if op is None:
            op_id = _ENDPOINT_TOKENS.get(text.lower())
            op = _OPERATIONS_BY_ID.get(op_id or "")
        # Only request-bearing operations may serve a model; GET-only routes
        # (voices/models/health) never win a request-resolution lookup.
        if op is not None and (op.serves_tasks or op.id == "tasks_run"):
            return op
    return operation_for_task(task or "")


def defaults_key_for(op: Operation) -> str:
    return DEFAULTS_KEYS.get(op.id, "task_defaults")


def is_generic_route(path: str) -> bool:
    """True when the request must be model-pinned via llama-swap /upstream."""
    op = operation_for_native_path(path) or operation_for_studio_path(path)
    return bool(op and op.upstream_only)


def is_native_api_path(path: str) -> bool:
    """True when ``path`` is a native audio.cpp HTTP API path.

    Registered operations are always accepted; any other versioned ``/v1/...``
    path is accepted too because families may advertise routes the registry has
    never seen (resolution for those falls through to ``tasks_run``). Query
    strings, fragments, and traversal segments are never native API paths.
    """
    text = str(path or "").strip()
    if not text.startswith("/v1/") or "?" in text or "#" in text:
        return False
    if text in _OPERATIONS_BY_NATIVE_PATH:
        return True
    parts = text.split("/")[1:]
    return bool(parts) and not any(part in {".", "..", ""} for part in parts)


def registry_descriptor() -> List[dict]:
    return [op.as_dict() for op in OPERATIONS]


__all__ = [
    "CONTRACT_SCHEMA",
    "DEFAULTS_KEYS",
    "OPERATIONS",
    "Operation",
    "STUDIO_OWNED_FLAGS",
    "TASKS_RUN_ALIAS",
    "defaults_key_for",
    "is_generic_route",
    "is_native_api_path",
    "operation_by_id",
    "operation_for_native_path",
    "operation_for_studio_path",
    "operation_for_task",
    "operations",
    "registry_descriptor",
    "resolve_operation",
]
