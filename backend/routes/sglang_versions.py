from fastapi import APIRouter

from backend.engines.sglang import get_sglang_manager
from backend.routes.engine_versions import (
    github_commit_updates,
    pypi_updates,
    python_engine_router,
)


def _sglang():
    return get_sglang_manager("sglang")


def _sglang_v100():
    return get_sglang_manager("sglang_v100")


router = APIRouter()
router.include_router(
    python_engine_router(
        engine_id="sglang",
        url_prefix="/sglang",
        get_installer=_sglang,
        update_source=pypi_updates("sglang"),
    )
)
router.include_router(
    python_engine_router(
        engine_id="sglang_v100",
        url_prefix="/sglang-v100",
        get_installer=_sglang_v100,
        update_source=github_commit_updates("haohervchb/sglang-V100"),
        prefer_source_install=True,
    )
)
