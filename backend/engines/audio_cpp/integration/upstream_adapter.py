"""Adapter for the future upstream audio.cpp contract (issue #852).

Upstream has not shipped a machine-readable server contract yet, so the
composer runs on the Studio registry plus live engine observations. This module
owns the detection seam: :func:`fetch_upstream_contract` probes a version's
install when the operator opts in with ``AUDIO_CPP_UPSTREAM_CONTRACT=1``,
looking for ``--describe-server --json`` first and then a well-known contract
file inside the checkout. It returns ``None`` cleanly when the flag is off, the
document is absent, or the probe fails; it never raises and never installs,
builds, activates, or changes engine state.

When #852 lands, its document is parsed into the same operation shape as
:mod:`contract_registry`; provenance flips to ``upstream`` and each matching
``studio`` entry is deleted from the registry individually. Nothing else in
Studio changes: every consumer already reads the composed descriptor. Until
then the composed descriptor stays Studio-owned; ``GET /api/audio-cpp/status``
calls this function only to report ``upstream_contract_available`` for the
active install.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)

#: Opt-in gate for the upstream probe. Off by default: no filesystem walk and
#: no subprocess run happens in a normal Studio deployment.
UPSTREAM_CONTRACT_ENV = "AUDIO_CPP_UPSTREAM_CONTRACT"

#: Probe budget. Contract detection is a diagnostic, not a scan; it must never
#: hold a route open for the usual engine-probe timeout.
_PROBE_TIMEOUT_SECONDS = 5.0

#: The CLI shape issue #852 proposes. Kept as a tuple so tests can assert the
#: exact argv and a future agreed flag is one edit.
_DESCRIBE_SERVER_ARGS = ("--describe-server", "--json")

#: Well-known checkout locations checked after the CLI probe. Upstream has not
#: agreed a filename yet (issue #852 is open); extend this tuple when it does.
UPSTREAM_CONTRACT_FILE_CANDIDATES = (
    "contracts/server-contract.json",
    "contracts/audio-cpp-contract.json",
    "audio-cpp-contract.json",
    "server-contract.json",
)


def upstream_contract_enabled() -> bool:
    """True when ``AUDIO_CPP_UPSTREAM_CONTRACT`` opts into the probe."""
    raw = os.environ.get(UPSTREAM_CONTRACT_ENV)
    if raw is None:
        return False
    return raw.strip().lower() not in {"", "0", "false", "no", "off", "disabled"}


def fetch_upstream_contract(version_row: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    """Return the upstream-owned contract document for this engine version.

    Detection order: ``audiocpp_server --describe-server --json`` for the
    active install, then :data:`UPSTREAM_CONTRACT_FILE_CANDIDATES` under the
    recorded checkout/install roots. Every failure mode (flag off, no binary,
    timeout, non-zero exit, malformed JSON, missing file) returns ``None``;
    this function is safe to call from request handling and never raises.
    """
    if not upstream_contract_enabled():
        return None
    row = version_row if isinstance(version_row, dict) else {}

    document = _probe_describe_server(row)
    if document is not None:
        return document
    return _read_known_contract_file(row)


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


def _probe_describe_server(row: dict) -> Optional[Dict[str, Any]]:
    binary = str(row.get("server_binary_path") or "").strip()
    if not binary or not os.path.isfile(binary):
        return None
    try:
        completed = subprocess.run(
            [binary, *_DESCRIBE_SERVER_ARGS],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("audio.cpp --describe-server probe failed for %s: %s", binary, exc)
        return None
    if completed.returncode != 0:
        logger.debug(
            "audio.cpp --describe-server returned %s for %s",
            completed.returncode,
            binary,
        )
        return None
    return _parse_document(completed.stdout, source=binary)


def _read_known_contract_file(row: dict) -> Optional[Dict[str, Any]]:
    for root in _candidate_roots(row):
        for relative in UPSTREAM_CONTRACT_FILE_CANDIDATES:
            path = os.path.join(root, relative)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, ValueError) as exc:
                logger.debug("audio.cpp upstream contract file unreadable at %s: %s", path, exc)
                continue
            document = _contract_document(payload)
            if document is not None:
                return document
    return None


def _parse_document(raw: str, *, source: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        logger.debug("audio.cpp --describe-server emitted unsupported JSON from %s", source)
        return None
    document = _contract_document(payload)
    if document is None:
        logger.debug("audio.cpp --describe-server document from %s has no operations", source)
    return document


def _contract_document(payload: Any) -> Optional[Dict[str, Any]]:
    """Accept only a mapping with a non-empty ``operations`` list.

    ``parse_upstream_operations`` can only do something with operations, so an
    empty or absent list means "no usable upstream contract", not "available".
    """
    if not isinstance(payload, dict):
        return None
    operations = payload.get("operations")
    if not isinstance(operations, list) or not operations:
        return None
    return payload


def _candidate_roots(row: dict) -> List[str]:
    """Checkout/install roots to search, deduplicated in probe order."""
    roots: List[str] = []
    for value in (row.get("source_path"), row.get("install_dir")):
        text = str(value or "").strip()
        if text:
            roots.append(text)
    for key in ("server_binary_path", "cli_binary_path"):
        binary = str(row.get(key) or "").strip()
        if not binary:
            continue
        parent = os.path.dirname(binary)
        if parent:
            roots.append(parent)
            grandparent = os.path.dirname(parent)
            if grandparent:
                roots.append(grandparent)
    seen: set[str] = set()
    ordered: List[str] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            ordered.append(root)
    return ordered


__all__ = [
    "UPSTREAM_CONTRACT_ENV",
    "UPSTREAM_CONTRACT_FILE_CANDIDATES",
    "fetch_upstream_contract",
    "parse_upstream_operations",
    "upstream_contract_enabled",
]
