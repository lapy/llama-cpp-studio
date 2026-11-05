"""Model-scoped routing for audio.cpp APIs absent from Studio proxy's route table.

The documented /upstream passthrough starts the native server and preserves its
HTTP response. It bypasses Studio proxy JSON filters and selector dispatch, so
only concrete Studio audio models and their ordinary aliases are accepted.
"""

from __future__ import annotations

from backend.engines.audio_cpp.identity import is_audio_engine_id

from dataclasses import dataclass
from urllib.parse import quote

from fastapi import HTTPException

from backend import data_store
from backend.engines.audio_cpp.integration import is_native_api_path
from backend.models.config import effective_model_config_from_raw


@dataclass(frozen=True)
class AudioUpstreamTarget:
    path: str
    model: str


def resolve_audio_upstream_target(
    model: str, native_path: str, *, store=None
) -> AudioUpstreamTarget:
    """Resolve ordinary aliases and return the native model name and upstream path.

    Callers rewrite the JSON/form/query ``model`` to ``target.model`` because
    Studio proxy's upstream passthrough deliberately does not apply useModelName.
    """
    name = str(model or "").strip()
    if not name:
        raise HTTPException(400, "This audio endpoint requires a model")
    if not is_native_api_path(native_path):
        raise ValueError("Expected an audio.cpp API path without query parameters")
    store = store or data_store.get_store()
    for record in store.list_models():
        config = effective_model_config_from_raw(record.get("config"))
        if name not in data_store.collect_model_names(record, config):
            continue
        if not is_audio_engine_id(config.get("engine")):
            break
        stable_id = data_store.resolve_model_id(record)
        ordinary = {
            stable_id,
            *data_store.collect_model_aliases({**config, "set_params_by_id": []}, stable_id),
        }
        if name not in ordinary:
            raise HTTPException(
                400,
                "Studio proxy does not apply parameter aliases on this audio endpoint; "
                "use a concrete audio model and send its parameters in the request",
            )
        return AudioUpstreamTarget(
            path=f"/upstream/{quote(stable_id, safe='')}{native_path}",
            model=stable_id,
        )
    raise HTTPException(404, f"Unknown Studio audio model: {name}")
