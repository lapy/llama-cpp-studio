"""Studio-owned operation registry for the audio.cpp integration contract.

Shim for upstream issue 0xShug0/audio.cpp#852: until audio.cpp publishes a
machine-readable server contract, every route/encoding fact Studio needs lives
here exactly once, with provenance ``studio``. Dynamic engine observations
(``--list-loaders --json``, ``--inspect``, ``model_specs``) overlay these
entries in :mod:`backend.engines.audio_cpp.integration.composer`; observed
data always beats a ``studio`` entry, and drift is reported as a warning.

Paths here are canonical *native* audio.cpp paths. The Studio proxy public
alias for the generic task route is derived once (:data:`TASKS_RUN_ALIAS`)
instead of being mirrored per consumer.

The table covers the documented v0.9.x server surface. Routes added for the
v0.9.5 line cite ``app/server/api.md`` at tag ``v0.9.5`` (and the server
README where it documents a request body the API reference only names).
Nothing here is guessed from a route name.

Two flags describe how a caller can actually reach an operation:

- ``upstream_only=True`` — Studio has no dedicated proxy/management route for
  it, so the documented path is the inference origin's concrete
  ``/upstream/{model}`` passthrough. A Studio shim may itself forward through
  that passthrough, so this flag does not mean "unserved".
- ``served=False`` — Studio has no working route for it at all. It is
  documented and represented here, but either only nominally routed or
  reachable solely through ``/upstream/{model}``. Live ingest is the first
  case: the management catch-all forwards ``/v1/audio/speech/live`` and
  ``/v1/audio/transcriptions/live``, but the inference-origin dataplane reads
  the whole request body before forwarding
  (``backend/proxy/studio/dataplane.py``: ``StudioDataPlane._handle`` ->
  ``_read_body`` at L641, then ``StudioDataPlane._proxy`` ->
  ``build_request(..., content=body)`` at L1569), so the engine never receives
  the chunked PCM its live handlers require. Fixing that belongs in the
  dataplane, not in this registry; the entry keeps ``served=False`` until it
  does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

CONTRACT_SCHEMA = "audio-cpp.studio-contract/v0"

# Studio proxy exposes the native generic-task route on its own public prefix and
# rewrites it upstream. Everything else keeps its native path through the native proxy.
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
    # True when Studio serves this operation with a working route on one of its
    # own origins (a management shim, or the native audio paths on the
    # inference origin). False is registry-only: documented here but not usable
    # through Studio; each such entry states why.
    served: bool = True
    # True when the route is absent from Studio proxy's route table and Studio
    # must reach it through the ``/upstream/{model}`` passthrough. Independent
    # of ``served``: a management shim can be served while forwarding through
    # /upstream, and a nominally routed native path can still be unserved.
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
            "served": self.served,
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

#: Operations whose request body is raw interleaved PCM, not a JSON envelope.
_RAW_PCM_BODY_OPERATIONS = frozenset({"speech_live", "transcriptions_live"})


def _body_shape(op: Operation) -> str:
    """Request-body envelope the client must build for this operation.

    ``multipart-form`` takes FormData (file field plus scalar fields),
    ``tasks-run`` wraps the fields inside a ``{"model","request"}`` object
    (``/v1/tasks/stream`` uses the same envelope plus its transport fields),
    ``tasks-batch`` wraps them inside a ``{"model","requests"}`` object,
    ``raw-pcm`` sends bare PCM bytes in a chunked body with the parameters in
    the query string, and everything else is a flat OpenAI-shaped JSON body.
    """
    if op.id in _RAW_PCM_BODY_OPERATIONS:
        return "raw-pcm"
    if op.id in {"tasks_run", "tasks_stream"}:
        return "tasks-run"
    if op.id == "tasks_batch":
        return "tasks-batch"
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
        # "sse-events": mode=streaming models answer speech requests carrying
        # stream/stream_format with OpenAI-shape SSE PCM deltas. SSE is also
        # the only wire the whole proxy chain (Studio proxy ReverseProxy)
        # flushes chunk-by-chunk; raw PCM octet-stream may buffer to 32 KiB.
        responses=("audio", "json", "sse-events"),
        errors="openai-envelope",
        serves_tasks=("tts", "vdes", "clon"),
    ),
    Operation(
        id="speech_live",
        native_path="/v1/audio/speech/live",
        studio_path="/v1/audio/speech/live",
        method="POST",
        # Upstream doc: app/server/api.md @ v0.9.5, POST /v1/audio/speech
        # ("live-ingest variant for speech-to-speech models"). The body is raw
        # interleaved PCM on a chunked transfer; `model`, `input`, `return_text`
        # and the PCM parameters travel in the query string, so there is no
        # in-body model for Studio to rewrite. Served by the existing native
        # /v1/audio/{rest} catch-all; the Studio proxy resolves `model` from the
        # query string itself.
        encodings=("raw-pcm",),
        responses=("sse-events",),
        errors="audio-cpp-error",
        # Live is a transport variant, not a task route: s2s/vc/svc models keep
        # routing their ordinary requests to tasks_run.
        serves_tasks=(),
        # served=False: the management catch-all nominally forwards the path,
        # but the proxy dataplane buffers request bodies (see module docstring),
        # so the engine never receives the chunked PCM this route requires.
        served=False,
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
        id="transcriptions_live",
        native_path="/v1/audio/transcriptions/live",
        studio_path="/v1/audio/transcriptions/live",
        method="POST",
        # Upstream doc: app/server/api.md @ v0.9.5,
        # POST /v1/audio/transcriptions/live. Raw interleaved PCM body; every
        # parameter (model, sample_rate, channels, sample_format, language,
        # prompt, busy_timeout_ms) is a query parameter, so no model rewriting
        # is possible or needed. Served by the native /v1/audio catch-all.
        encodings=("raw-pcm",),
        responses=("sse-events",),
        errors="audio-cpp-error",
        # Live is a transport variant, not a task route: asr models keep routing
        # their ordinary requests to `transcriptions`.
        serves_tasks=(),
        # served=False: same dataplane buffering limit as speech_live above.
        served=False,
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
        id="models_load",
        native_path="/v1/models/load",
        studio_path="/v1/models/load",
        method="POST",
        # Upstream doc: app/server/api.md @ v0.9.5, GET /v1/models ("a model
        # loaded later through POST /v1/models/load"); the body itself is
        # documented in app/server/README.md @ v0.9.5, "Loading an installed
        # package by id": `id` plus either `path` or `package_id`, then task /
        # mode / family / load_options / session_options.
        encodings=("json",),
        responses=("json",),
        errors="audio-cpp-error",
        serves_tasks=(),
        # upstream_only: `id` here is the id the engine registers, not a Studio
        # model alias, and the call mutates the engine's model registry outside
        # Studio's model records. A management shim would either have to invent
        # an id-mapping semantic or let a request register arbitrary engine
        # models behind Studio's back, so Studio leaves it to the concrete
        # /upstream/{model} passthrough instead of faking support.
        upstream_only=True,
        # served=False: no Studio route at all, for the reason above.
        served=False,
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
        # map to a dedicated OpenAI-shaped route. The task tuple mirrors
        # upstream's task vocabulary
        # (src/framework/runtime/task_vocabulary.cpp) so the composed
        # ``task_routing`` covers every task the engine can serve.
        serves_tasks=(
            "vc",
            "svc",
            "s2s",
            "vad",
            "diar",
            "sep",
            "gen",
            "spk",
            "midi",
            "turn",
            "wake",
            "cls",
        ),
    ),
    Operation(
        id="tasks_stream",
        native_path="/v1/tasks/stream",
        studio_path="/v1/tasks/stream",
        method="POST",
        # Upstream doc: app/server/api.md @ v0.9.5, POST /v1/tasks/stream. The
        # body is the /v1/tasks/run body plus top-level `stream_format` (only
        # "sse" streams) and `include_artifacts`; the response is one JSON body
        # or task.stream.event / task.stream.done / [DONE] SSE.
        encodings=("json",),
        responses=("json", "sse-events"),
        errors="audio-cpp-error",
        # A transport variant of tasks_run, never a task route: no task may be
        # remapped onto it, so the deterministic task table is unchanged.
        serves_tasks=(),
        # upstream_only: the management shim rewrites `model` to the stable
        # native id and forwards through /upstream/{model}.
        upstream_only=True,
    ),
    Operation(
        id="tasks_batch",
        native_path="/v1/tasks/batch",
        studio_path="/v1/tasks/batch",
        method="POST",
        # Upstream doc: app/server/api.md @ v0.9.5, POST /v1/tasks/batch.
        # `requests` is a non-empty array of /v1/tasks/run request objects; the
        # response is SSE (task.batch.result / task.batch.done / [DONE]).
        encodings=("json",),
        responses=("sse-events",),
        errors="audio-cpp-error",
        serves_tasks=(),
        # upstream_only: the management shim rewrites `model` to the stable
        # native id and forwards through /upstream/{model}.
        upstream_only=True,
    ),
    Operation(
        id="tasks_unload_models",
        native_path="/v1/tasks/unload_models",
        studio_path="/v1/tasks/unload_models",
        method="POST",
        # Upstream doc: app/server/api.md @ v0.9.5,
        # POST /v1/tasks/unload_models. Body is `{"model_ids": [string, ...]}`;
        # ids the engine does not know come back in `not_found`, not as errors.
        encodings=("json",),
        responses=("json",),
        errors="audio-cpp-error",
        serves_tasks=(),
        # upstream_only: the management shim resolves every id to a stable
        # native id and pins the engine instance from the first id (or from an
        # explicit `?model=` pin, which wins when present), then forwards
        # through /upstream/{model}.
        upstream_only=True,
    ),
    Operation(
        id="tasks_unload_all_models",
        native_path="/v1/tasks/unload_all_models",
        studio_path="/v1/tasks/unload_all_models",
        method="POST",
        # Upstream doc: app/server/api.md @ v0.9.5,
        # POST /v1/tasks/unload_all_models. No request body.
        encodings=(),
        responses=("json",),
        errors="audio-cpp-error",
        serves_tasks=(),
        # upstream_only: the route names no model, but every Studio engine route
        # is model-scoped, so the management shim requires a `?model=` pin to
        # select the engine instance and does not forward that Studio-only pin.
        upstream_only=True,
    ),
    Operation(
        id="recipes",
        native_path="/v1/recipes",
        studio_path="/v1/recipes",
        method="GET",
        # Upstream doc: app/server/api.md @ v0.9.5, POST /v1/recipes/run ("GET
        # /v1/recipes lists available recipes and their required tasks and
        # modes").
        encodings=(),
        responses=("json",),
        errors="audio-cpp-error",
        serves_tasks=(),
        # upstream_only: a recipe list belongs to one engine process, and Studio
        # has no model to pin one with here. Kept unserved rather than inventing
        # a management-level recipe catalog.
        upstream_only=True,
        # served=False: no Studio route; reachable only through
        # /upstream/{model}/v1/recipes on a concrete engine instance.
        served=False,
    ),
    Operation(
        id="recipes_run",
        native_path="/v1/recipes/run",
        studio_path="/v1/recipes/run",
        method="POST",
        # Upstream doc: app/server/api.md @ v0.9.5, POST /v1/recipes/run. Body:
        # `recipe`, `models` (role -> model id), `request`, `stream`.
        encodings=("json",),
        responses=("json", "sse-events"),
        errors="audio-cpp-error",
        serves_tasks=(),
        # upstream_only: every documented recipe connects roles across models,
        # and Studio's generated sidecar config holds exactly one model per
        # engine process (`backend/engines/audio_cpp/runtime.py`). A management
        # route cannot resolve that role map to one engine instance, and a blind
        # passthrough would send Studio aliases the engine does not know, so
        # this stays upstream-only instead of pretending to support recipes.
        upstream_only=True,
        # served=False: no Studio route; reachable only through
        # /upstream/{model}/v1/recipes/run on a concrete engine instance.
        served=False,
    ),
)

_OPERATIONS_BY_ID: Dict[str, Operation] = {op.id: op for op in OPERATIONS}
_OPERATIONS_BY_NATIVE_PATH: Dict[str, Operation] = {op.native_path: op for op in OPERATIONS}
_OPERATIONS_BY_STUDIO_PATH: Dict[str, Operation] = {op.studio_path: op for op in OPERATIONS}

# Studio request-defaults bucket per request operation. This is a Studio
# persistence concept (which config blob holds the saved defaults), never an
# engine-facing path, so it is derived from the registry in one place.
#
# Only operations whose request body Studio serializes from saved defaults are
# listed. ``defaults_key_for`` (and ``Operation.as_dict``) falls back to
# ``task_defaults`` for every other operation — metadata GETs, management
# unload, recipes, live transport — so the descriptor always carries a stable
# string. That fallback is a schema default, not a claim that those routes
# persist or accept request defaults.
DEFAULTS_KEYS: Dict[str, str] = {
    "speech": "speech_defaults",
    "transcriptions": "transcription_defaults",
    "transcriptions_details": "transcription_defaults",
    "alignments": "task_defaults",
    "batch_transcriptions": "task_defaults",
    "tasks_run": "task_defaults",
    # Both generic transports take the same request vocabulary as tasks_run.
    "tasks_stream": "task_defaults",
    "tasks_batch": "task_defaults",
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
    # Upstream v0.9.5 families may advertise the streaming generic route as
    # their preferred surface; it takes the same request envelope as tasks/run.
    "tasks_stream": "tasks_stream",
    "tasks/stream": "tasks_stream",
    "generic": "tasks_run",
    "voices": "voices",
    "models": "models",
    "health": "health",
}

#: Operations that may serve a model's ordinary requests. `tasks_run` and
#: `tasks_stream` are generic transports with the same request envelope, so an
#: engine that advertises either as its preferred surface is honored. Metadata
#: GETs, management/unload routes, recipes, and the live transport variants
#: never win a model-resolution lookup, even when advertised.
_REQUEST_SERVING_OPERATIONS = frozenset({"tasks_run", "tasks_stream"})


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
        op = operation_for_native_path(f"/{text}") or operation_for_studio_path(f"/{text}")
        if op is None:
            op_id = _ENDPOINT_TOKENS.get(text.lower())
            op = _OPERATIONS_BY_ID.get(op_id or "")
        # Only request-bearing operations may serve a model; GET-only routes
        # (voices/models/health) never win a request-resolution lookup.
        if op is not None and (op.serves_tasks or op.id in _REQUEST_SERVING_OPERATIONS):
            return op
    return operation_for_task(task or "")


def defaults_key_for(op: Operation) -> str:
    return DEFAULTS_KEYS.get(op.id, "task_defaults")


def is_generic_route(path: str) -> bool:
    """True when the request must be model-pinned via Studio proxy /upstream."""
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
    "STUDIO_OWNED_FLAGS",
    "TASKS_RUN_ALIAS",
    "Operation",
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
