"""Adapter for the future upstream audio.cpp contract (issue #852).

Upstream is still refactoring and has not shipped a machine-readable server
contract. Until it does, ``fetch_upstream_contract`` returns nothing and the
composer runs entirely on the Studio registry plus live engine observations.

When #852 lands, its document is parsed into the same operation shape as
:mod:`contract_registry`; provenance flips to ``upstream`` and each matching
``studio`` entry is deleted from the registry individually. Nothing else in
Studio changes: every consumer already reads the composed descriptor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)


def fetch_upstream_contract(version_row: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    """Return the upstream-owned contract document for this engine version.

    Placeholder for audio.cpp#852: no upstream release serves a contract
    document yet, so this always returns ``None``. Detection order once the
    issue ships will be ``--describe-server --json`` (or the agreed CLI), then
    a well-known file inside the checkout; each source maps onto
    :func:`parse_upstream_operations` below.
    """
    return None


def parse_upstream_operations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize an upstream contract document into registry-shaped entries.

    Kept small and total: unknown fields are ignored, missing fields fall back
    to the same defaults the registry uses. Entries are returned with
    provenance ``upstream`` so the composer lets them override every Studio
    entry with the same ``id``.
    """
    operations: List[Dict[str, Any]] = []
    raw = payload.get("operations") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return operations
    for item in raw:
        if not isinstance(item, dict):
            continue
        op_id = str(item.get("id") or "").strip()
        native_path = str(item.get("native_path") or item.get("path") or "").strip()
        if not op_id or not native_path:
            continue
        operations.append(
            {
                "id": op_id,
                "native_path": native_path,
                "studio_path": str(item.get("studio_path") or native_path).strip(),
                "method": str(item.get("method") or "POST").strip().upper(),
                "encodings": [
                    str(v).strip() for v in (item.get("encodings") or []) if str(v).strip()
                ],
                "responses": [
                    str(v).strip() for v in (item.get("responses") or []) if str(v).strip()
                ],
                "errors": str(item.get("errors") or "openai-envelope").strip(),
                "serves_tasks": [
                    str(v).strip().lower()
                    for v in (item.get("serves_tasks") or [])
                    if str(v).strip()
                ],
                "provenance": "upstream",
                "upstream_only": bool(item.get("upstream_only")),
            }
        )
    return operations


__all__ = ["fetch_upstream_contract", "parse_upstream_operations"]
