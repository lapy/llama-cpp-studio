from backend.engines.vllm import GITHUB_REPO, get_onecat_vllm_manager
from backend.routes.engine_versions import github_release_updates, python_engine_router


def _installer():
    return get_onecat_vllm_manager()


router = python_engine_router(
    engine_id="1cat_vllm",
    url_prefix="/1cat-vllm",
    get_installer=_installer,
    update_source=github_release_updates(GITHUB_REPO),
    version_setting_key="release_version",
    status_fallback={
        "installed_at": None,
        "removed_at": None,
        "log_path": None,
        "release_tag": None,
    },
)
