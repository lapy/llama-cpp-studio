"""Startup verification of engine definitions.

Two definition tiers exist, and every startup re-verifies both:

* ``definition-owned`` (``DEFINITION_ENGINE_IDS``) — hand-authored YAML whose
  typed ``EngineSpec`` the registry built AT IMPORT. On-disk YAML must parse,
  expand and produce descriptors IDENTICAL to ``ENGINE_REGISTRY`` (drift or
  missing files refuse startup; message bytes pinned).
* ``authoritative`` (new ids) — validated by the schema/vocabularies at load,
  then REGISTERED into ``ENGINE_REGISTRY`` via ``register_engine_spec``
  (identical re-verification is idempotent; any disagreement with an existing
  entry refuses startup). This tier is what a newly added definition lands in
  before its id is added to the registry's shipped set; production engines are
  picked up by the import-time bootstrap and appear in the definition-owned
  tier instead.

This module is the common startup gate for both tiers. Runtime readers gate
per engine on whether the definition declares the relevant section (profiles,
discovery, lifecycle), never on an explicit engine-id allow-list.
"""

from __future__ import annotations

import logging

from backend.engines.registry import (
    DEFINITION_ENGINE_IDS,
    ENGINE_REGISTRY,
    register_engine_spec,
)

from .loader import load_default_engine_definitions

logger = logging.getLogger("llamacpp.studio.definitions")


def definition_tier(engine_id: str) -> str:
    """``definition`` = built from a shipped, hand-authored definition;
    ``authoritative`` = a harness definition accepted at startup."""
    if engine_id in DEFINITION_ENGINE_IDS:
        return "definition"
    return "authoritative"


def verify_engine_definitions() -> int:
    """Load + validate definitions, assert registry equivalence, and accept
    authoritative additions.

    Returns the number of engines verified. Raises RuntimeError to refuse
    startup on invalid definitions, drift, missing definitions, or an
    authoritative definition that conflicts with an existing registry entry.
    """
    try:
        registry = load_default_engine_definitions()
    except Exception as exc:  # DefinitionError and friends
        raise RuntimeError(f"Engine definitions invalid; refusing startup: {exc}") from exc
    drifted = [
        engine_id
        for engine_id, definition in sorted(registry.definitions.items())
        if ENGINE_REGISTRY.get(engine_id) is not None
        and definition.to_api_dict() != ENGINE_REGISTRY[engine_id].to_api_dict()
    ]
    missing = sorted(set(ENGINE_REGISTRY) - set(registry.definitions))
    if drifted or missing:
        raise RuntimeError(
            "Engine definitions diverged from the derived engine registry; refusing startup "
            f"(drifted={drifted}, missing={missing})."
        )
    accepted = sorted(
        engine_id for engine_id in registry.definitions if ENGINE_REGISTRY.get(engine_id) is None
    )
    for engine_id in accepted:
        # Same-id conflicts already surfaced as drifted above; a genuinely
        # new id registers its spec into the derived view. register_engine_
        # spec still refuses (never overwrites) on any disagreement.
        register_engine_spec(registry.definitions[engine_id].to_spec())
    if accepted:
        logger.info(
            "Authoritative engine definitions accepted: %s",
            ", ".join(accepted),
        )
    logger.info(
        "Engine definitions validated: %s engines (%s definition-owned, %s accepted)",
        len(registry.definitions),
        len(DEFINITION_ENGINE_IDS),
        len(accepted),
    )
    return len(registry.definitions)
