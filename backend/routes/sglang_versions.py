"""Compatibility URLs for the definition-owned SGLang family.

Both prefixes delegate to the same lifecycle records the generic
``/api/engines/{engine_id}/...`` surface serves; the YAML definitions own the
update source, source-install preference, settings key and status fallback.
"""

from fastapi import APIRouter

from backend.routes.engine_versions import definition_compat_router

router = APIRouter()
router.include_router(definition_compat_router("sglang", "/sglang"))
router.include_router(definition_compat_router("sglang_v100", "/sglang-v100"))
router.include_router(definition_compat_router("sglang_sxm2", "/sglang-sxm2"))
router.include_router(definition_compat_router("sglang_sm70", "/sglang-sm70"))
