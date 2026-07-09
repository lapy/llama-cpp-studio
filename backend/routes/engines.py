"""Engine capability descriptors."""

from fastapi import APIRouter

from backend.data_store import get_store
from backend.engine_registry import (
    active_engine_row_is_runnable,
    engine_registry_payload,
)
from backend.feature_flags import audio_cpp_enabled


router = APIRouter()


@router.get("")
@router.get("/")
async def list_engine_descriptors():
    store = get_store()
    payload = engine_registry_payload()
    for descriptor in payload["engines"]:
        engine_id = descriptor["id"]
        active = store.get_active_engine_version(engine_id)
        descriptor["active_version"] = active.get("version") if active else None
        descriptor["runnable"] = active_engine_row_is_runnable(engine_id, active)
        descriptor["installed_versions"] = len(store.get_engine_versions(engine_id))
        descriptor["enabled"] = (
            audio_cpp_enabled() if engine_id == "audio_cpp" else True
        )
        if engine_id == "audio_cpp" and active:
            build_config = (
                active.get("build_config")
                if isinstance(active.get("build_config"), dict)
                else {}
            )
            backend = str(build_config.get("backend") or "cpu")
            descriptor["available_runtime_backends"] = list(
                dict.fromkeys(["cpu", backend])
            )
            descriptor["active_source_commit"] = active.get("source_commit")
    return payload

