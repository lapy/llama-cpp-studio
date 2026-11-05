"""Compatibility URLs for a definition-owned engine.

Every installer/update/status fact comes from the engine's authoritative
definition in ``backend/engines/definitions/``. This module keeps the historic
per-engine URL prefix alive as a thin delegate over the SAME lifecycle record
the generic ``/api/engines/{engine_id}/...`` surface serves, so no business
logic is duplicated behind the old routes.
"""

from backend.routes.engine_versions import definition_compat_router

_ENGINE_ID = "lmdeploy"

router = definition_compat_router(_ENGINE_ID, "/lmdeploy")
