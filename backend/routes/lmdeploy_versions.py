from backend.engines.lmdeploy import get_lmdeploy_manager
from backend.routes.engine_versions import pypi_updates, python_engine_router


def _installer():
    return get_lmdeploy_manager()


router = python_engine_router(
    engine_id="lmdeploy",
    url_prefix="/lmdeploy",
    get_installer=_installer,
    update_source=pypi_updates("lmdeploy", timeout=10.0, error_status=500),
    status_fallback={
        "installed_at": None,
        "removed_at": None,
        "log_path": None,
    },
)
