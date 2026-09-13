"""Model-scoped routing for audio.cpp APIs absent from llama-swap's route table.

The documented /upstream passthrough starts the native server and preserves its
HTTP response. It bypasses llama-swap JSON filters and selector dispatch, so
only concrete Studio audio models and their ordinary aliases are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

from fastapi import HTTPException

from backend import data_store
from backend.model_config import effective_model_config_from_raw


@dataclass(frozen=True)
class AudioUpstreamTarget:
    path: str
    model: str


def resolve_audio_upstream_target(
    model: str, native_path: str, *, store=None
) -> AudioUpstreamTarget:
    """Resolve ordinary aliases and return the native model name and swap path.

    Callers rewrite the JSON/form/query ``model`` to ``target.model`` because
    llama-swap's upstream passthrough deliberately does not apply useModelName.
    """
    name = str(model or "").strip()
    if not name:
        raise HTTPException(400, "This audio endpoint requires a model")
    if not native_path.startswith("/v1/") or "?" in native_path or "#" in native_path:
        raise ValueError("Expected an audio.cpp API path without query parameters")
    if any(part in {".", ".."} for part in native_path.split("/")):
        raise ValueError("Invalid audio.cpp API path")
    store = store or data_store.get_store()
    routing = store.get_llama_swap_routing() or {}
    if name in (routing.get("selectors") or {}):
        raise HTTPException(
            400,
            "llama-swap does not support selectors on this audio endpoint; "
            "use a concrete audio model or its ordinary alias",
        )
    for record in store.list_models():
        config = effective_model_config_from_raw(record.get("config"))
        if name not in data_store.collect_claimed_swap_names(record, config):
            continue
        if config.get("engine") != "audio_cpp":
            break
        stable_id = data_store.resolve_llama_swap_id(record)
        ordinary = {stable_id, *data_store.collect_config_swap_aliases(config, stable_id)}
        if name not in ordinary:
            raise HTTPException(
                400,
                "llama-swap does not apply parameter aliases on this audio endpoint; "
                "use a concrete audio model and send its parameters in the request",
            )
        return AudioUpstreamTarget(
            path=f"/upstream/{quote(stable_id, safe='')}{native_path}",
            model=stable_id,
        )
    raise HTTPException(404, f"Unknown Studio audio model: {name}")
