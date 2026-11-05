"""Compatibility URLs for a definition-owned engine.

Every installer/update/status fact comes from the engine's authoritative
definition in ``backend/engines/definitions/``; this module keeps the historic
``/vllm`` prefix alive as a thin delegate over the same lifecycle record the
generic ``/api/engines/vllm/...`` surface serves.
"""

from backend.routes.engine_versions import definition_compat_router

router = definition_compat_router("vllm", "/vllm")
