"""Install-snapshot selection and retry resolution (shared by all hosts).

The snapshot ledger ties a version name to the definition digest and the inputs
that produced its install. Retries must resolve through those frozen inputs and
refuse to reinterpret a layout produced by a different definition revision.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


def install_snapshot_for(store, engine_id: str, version_name: str) -> Dict[str, Any]:
    """Snapshot recorded for THIS version, or {}.

    Selection is by version (never "latest for the engine"); earlier rows are
    also matched by their recorded requested version so pre-publication installs
    still get the digest guard when the association is unambiguous.

    A missing ledger legitimately means "pre-refactor install" and returns {}.
    A ledger that cannot be READ (corruption, permissions) is not the same
    thing: it must raise, because silently treating it as empty would disable
    the digest guard and let a retry reinterpret the recorded layout.
    """
    target = str(version_name or "").strip()
    if not target:
        return {}
    rows = store.list_install_snapshots(engine_id)
    for row in reversed(rows):
        if str(row.get("version") or "") == target:
            return dict(row)
    for row in reversed(rows):
        inputs = row.get("resolved_inputs")
        inputs = inputs if isinstance(inputs, dict) else {}
        if str(inputs.get("version") or "") == target:
            return dict(row)
    return {}


def resolve_snapshot_inputs(
    snapshot: Mapping[str, Any],
    *,
    version_name: str,
    current_digest: str,
    label: str,
) -> Mapping[str, Any]:
    """Frozen inputs for a retry, after refusing a changed definition."""
    recorded = str(snapshot.get("definition_digest") or "")
    if recorded and current_digest and recorded != current_digest:
        raise ValueError(
            f"{label} definition changed since version {version_name} was installed; "
            "reinstall instead of retrying, so the recorded install layout is not "
            "reinterpreted by the current YAML"
        )
    inputs = snapshot.get("resolved_inputs")
    return inputs if isinstance(inputs, dict) else {}


def retry_resolved_inputs(
    store,
    engine_id: str,
    version_name: str,
    *,
    current_digest: str,
    label: str,
) -> Mapping[str, Any]:
    """Frozen inputs for a retry selected from the ledger (route/host use)."""
    return resolve_snapshot_inputs(
        install_snapshot_for(store, engine_id, version_name),
        version_name=version_name,
        current_digest=current_digest,
        label=label,
    )
