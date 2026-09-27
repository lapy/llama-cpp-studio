from backend.engines.vllm import get_vllm_manager
from backend.routes.engine_versions import pypi_updates, python_engine_router


def _installer():
    return get_vllm_manager()


router = python_engine_router(
    engine_id="vllm",
    url_prefix="/vllm",
    get_installer=_installer,
    update_source=pypi_updates("vllm"),
)
