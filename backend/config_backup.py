"""Versioned configuration backup and multi-document restore.

Schema
------
A backup is one JSON document, not a directory archive.

``schema_version`` 1. ``kind`` is ``llama-cpp-studio-config-backup``.
``application_version`` records the build that exported it. A different
application version may import schema 1. A different schema version is rejected.

Included: portable preferences (``public_inference_url``, ``proxy_port``),
saved per-model settings keyed by a stable reference, configuration templates,
and routing profiles and selectors.

Excluded, and stated again in the document ``limits``: credentials, model
weights, reference-audio files, build and install artifacts, machine-specific
executable paths, runtime and process state, operation history, and published
or running launch generations. The data directory is never serialized.

A model reference is ``{provider}:{source id}``, otherwise
``huggingface:{huggingface_id}``, otherwise ``catalog:{catalog id}``. Import
never creates a model row, downloads files, or installs an engine. A reference
with no local model is unresolved until the request maps it to an existing
catalog id or explicitly skips it.

Conflicts default to keep-existing. ``replace`` substitutes that one item.
There is no recursive merge. Omitted credential fields are left as they are,
and a backup that contains a credential, an absolute path, or ``..`` is
rejected without a write.

Recovery protocol
-----------------
``settings.yaml``, ``model_config_templates.yaml``, ``llama_swap_routing.yaml``,
and ``models.yaml`` are still replaced one at a time. The restore journal
``config_restore.yaml`` is the commit record. It holds the pre-import snapshot,
the intended documents, the preview revisions, and the plan id.

The prepared journal is durable before the first of those documents is
replaced. Durable means the temporary file was synced, ``os.replace`` returned,
and the directory entry was synced. Mode ``0600`` only sets permissions. It is
not that barrier. A failed directory sync does not start the first document
replacement.

Startup holds ordinary configuration writes, runs recovery, and releases the
hold before serving requests or writing other configuration such as an
environment token. Recovery's own journal and document writes do not wait on
that hold.

A second interruption while recovery is writing the snapshot or the terminal
journal is retried. The next recovery sees the same journal and finishes the
pre-import or completed state. A journal that cannot be read, or a terminal
journal that does not match the documents, stops recovery. Configuration is
left as it is and the journal is not deleted.

The journal is removed only after the ``completed`` or ``rolled_back`` record
has itself been replaced and its directory entry synced. Removing it earlier,
or removing a corrupt journal, is not recovery.

Apply rechecks the preview revisions under the store lock. A mismatch rejects
the plan and writes nothing. Restore does not start, stop, or publish a model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from backend.data_store import DataStore, StorageCorruptionError, _reach_write_checkpoint
from backend.inference_url import normalize_public_inference_url
from backend.models.config import normalize_model_config
from backend.version import APP_VERSION

BACKUP_SCHEMA_VERSION = 1
BACKUP_KIND = "llama-cpp-studio-config-backup"
MAX_BACKUP_BYTES = 1_048_576
JOURNAL_FILENAME = "config_restore.yaml"
PREVIEW_NOTICE = (
    "This restores saved settings only. It does not start, stop, or publish a model. "
    "Use Apply afterwards if the running configuration should change."
)

DOCUMENT_ORDER = (
    "settings.yaml",
    "model_config_templates.yaml",
    "llama_swap_routing.yaml",
    "models.yaml",
)
_REF_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,240}$")
_FORBIDDEN_KEY = re.compile(
    r"(token|password|passwd|secret|api[_-]?key|access_token|authorization|"
    r"credential|private_key|executable|binary_path|local_path|pid|launch_id|"
    r"published_revision)",
    re.IGNORECASE,
)
_DROP = object()
# These are generation limits, not credentials.
_TOKEN_LIMIT_KEYS = {
    "max_tokens",
    "min_tokens",
    "max_new_tokens",
    "min_new_tokens",
    "n_tokens",
}


def _forbidden_key(key: Any) -> bool:
    text = str(key)
    return text not in _TOKEN_LIMIT_KEYS and bool(_FORBIDDEN_KEY.search(text))

LIMITS = {
    "includes": [
        "portable preferences",
        "saved per-model settings",
        "configuration templates",
        "routing profiles and selectors",
    ],
    "excludes": [
        "credentials",
        "model weights",
        "reference-audio files",
        "build and install artifacts",
        "machine-specific executable paths",
        "runtime and process state",
        "operation history",
        "published and running launch generations",
    ],
}


class ConfigBackupError(ValueError):
    def __init__(self, code: str, detail: str, status_code: int = 400) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status_code = status_code


def export_backup(store: DataStore) -> dict:
    """Read one consistent snapshot and return a schema-1 backup."""
    with store.exclusive_documents():
        _require_no_journal(store)
        settings = store._read_yaml("settings.yaml")
        models = store._read_yaml("models.yaml").get("models") or []
        templates = store._read_yaml("model_config_templates.yaml").get("templates") or []
        routing = store.get_llama_swap_routing()
    document = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "kind": BACKUP_KIND,
        "application_version": APP_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "limits": copy.deepcopy(LIMITS),
        "preferences": _export_preferences(settings),
        "models": [
            dict(_export_model(row), ref=ref)
            for row, ref in _model_references(models)
        ],
        "templates": [_export_template(row) for row in templates if isinstance(row, dict)],
        "routing": _export_routing(routing),
    }
    encoded = _canonical(document).encode("utf-8")
    if len(encoded) > MAX_BACKUP_BYTES:
        raise ConfigBackupError("BACKUP_TOO_LARGE", "The backup is larger than the supported size.", 413)
    return document


def preview_backup(
    store: DataStore,
    backup: Any,
    *,
    decisions: Optional[Mapping] = None,
    mapping: Optional[Mapping] = None,
) -> dict:
    """Validate a backup and describe the restore. This does not write."""
    document = _validated_backup(backup)
    chosen = _validated_decisions(decisions)
    mapped = _validated_mapping(mapping)
    with store.exclusive_documents():
        _require_no_journal(store)
        current = _current_state(store)
        revisions = {name: store.document_revision(name) for name in DOCUMENT_ORDER}
    plan = _build_plan(document, current, chosen, mapped, revisions)
    return _preview_body(plan, revisions)


def apply_backup(
    store: DataStore,
    backup: Any,
    *,
    plan_id: str,
    decisions: Optional[Mapping] = None,
    mapping: Optional[Mapping] = None,
) -> dict:
    """Apply a previewed plan, or leave every document unchanged."""
    document = _validated_backup(backup)
    chosen = _validated_decisions(decisions)
    mapped = _validated_mapping(mapping)
    with store.exclusive_documents():
        _require_no_journal(store)
        current = _current_state(store)
        revisions = {name: store.document_revision(name) for name in DOCUMENT_ORDER}
        plan = _build_plan(document, current, chosen, mapped, revisions)
        if not plan["applicable"]:
            raise ConfigBackupError(
                "BACKUP_UNRESOLVED",
                "A model reference needs a mapping to an existing model, or an explicit skip.",
                409,
            )
        if plan["plan_id"] != str(plan_id or ""):
            raise ConfigBackupError(
                "BACKUP_STALE",
                "The configuration changed after the preview. Request a new preview.",
                409,
            )
        try:
            _commit_restore(store, plan, revisions, _digest(document))
        except ConfigBackupError:
            raise
        except Exception:
            outcome = _reconcile_locked(store)
            if outcome.get("outcome") in {"pre_import", "idle"}:
                detail = "The restore did not finish. Documents were returned to the pre-import state."
            elif outcome.get("outcome") == "completed":
                return {
                    "plan_id": plan["plan_id"],
                    "outcome": "completed",
                    "notice": PREVIEW_NOTICE,
                }
            else:
                detail = "The restore outcome could not be established. Restart before trying again."
            raise ConfigBackupError("BACKUP_INCOMPLETE", detail, 500) from None
    return {"plan_id": plan["plan_id"], "outcome": "completed", "notice": PREVIEW_NOTICE}


def reconcile_config_restore(store: DataStore) -> dict:
    """Finish an interrupted restore as pre-import or completed."""
    with store.exclusive_documents():
        return _reconcile_locked(store)


def _preview_body(plan: dict, revisions: dict) -> dict:
    return {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "applicable": plan["applicable"],
        "notice": PREVIEW_NOTICE,
        "revisions": revisions,
        "items": plan["items"],
        "limits": copy.deepcopy(LIMITS),
    }


def _commit_restore(store: DataStore, plan: dict, revisions: dict, digest: str) -> None:
    snapshot = _snapshot_documents(store)
    intended = _intended_documents(snapshot, plan["changes"])
    from backend.config_history import record_document_snapshot

    for filename in DOCUMENT_ORDER:
        if snapshot[filename] != intended[filename]:
            record_document_snapshot(
                store,
                filename,
                snapshot[filename],
                reason="before configuration backup restore",
            )
    journal = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "phase": "prepared",
        "plan_id": plan["plan_id"],
        "backup_digest": digest,
        "revisions": revisions,
        "applied": [],
        "pending": None,
        "snapshot": snapshot,
        "intended": intended,
    }
    _write_journal(store, journal)
    _reach_write_checkpoint("restore_prepared", JOURNAL_FILENAME)
    for filename in DOCUMENT_ORDER:
        journal["phase"] = "applying"
        journal["pending"] = filename
        _write_journal(store, journal)
        store.write_document(filename, intended[filename], require_directory_sync=True)
        _reach_write_checkpoint(f"restore_replaced:{filename}", filename)
        journal["applied"] = [*journal["applied"], filename]
        journal["pending"] = None
        _write_journal(store, journal)
    journal["phase"] = "completed"
    _write_journal(store, journal)
    _reach_write_checkpoint("restore_completed", JOURNAL_FILENAME)
    _remove_journal(store)


def _reconcile_locked(store: DataStore) -> dict:
    journal = _journal(store)
    phase = None if journal is None else journal.get("phase")
    if journal is None or phase in {None, "idle"}:
        return {"outcome": "idle"}
    if phase == "corrupt":
        return {"outcome": "unknown", "code": "RESTORE_JOURNAL_CORRUPT"}
    if phase == "completed":
        if not _terminal_matches(store, journal, "intended"):
            return {"outcome": "unknown", "code": "RESTORE_JOURNAL_CORRUPT"}
        _remove_journal(store)
        return {"outcome": "completed"}
    if phase == "rolled_back":
        if not _terminal_matches(store, journal, "snapshot"):
            return {"outcome": "unknown", "code": "RESTORE_JOURNAL_CORRUPT"}
        _remove_journal(store)
        return {"outcome": "pre_import"}
    snapshot = journal.get("snapshot")
    intended = journal.get("intended")
    if not isinstance(snapshot, dict) or not isinstance(intended, dict):
        return {"outcome": "unknown", "code": "RESTORE_JOURNAL_CORRUPT"}
    live = _snapshot_documents(store)
    if _documents_match(live, intended):
        journal["phase"] = "completed"
        journal["pending"] = None
        _write_journal(store, journal)
        _reach_write_checkpoint("restore_completed", JOURNAL_FILENAME)
        _remove_journal(store)
        return {"outcome": "completed"}
    if not _documents_match(live, snapshot):
        for filename in DOCUMENT_ORDER:
            _reach_write_checkpoint(f"restore_rollback:{filename}", filename)
            store.write_document(filename, snapshot[filename], require_directory_sync=True)
    journal["phase"] = "rolled_back"
    journal["pending"] = None
    journal["applied"] = []
    _write_journal(store, journal)
    _reach_write_checkpoint("restore_rolled_back", JOURNAL_FILENAME)
    _remove_journal(store)
    return {"outcome": "pre_import"}


def _write_journal(store: DataStore, journal: dict) -> None:
    """Replace the journal and sync its directory entry before returning."""
    store.write_document(JOURNAL_FILENAME, journal, require_directory_sync=True)


def _terminal_matches(store: DataStore, journal: dict, key: str) -> bool:
    recorded = journal.get(key)
    if not isinstance(recorded, dict):
        return False
    return _documents_match(_snapshot_documents(store), recorded)


def _remove_journal(store: DataStore) -> None:
    """Unlink the journal only after its terminal record is already durable."""
    path = os.path.join(store._config_dir, JOURNAL_FILENAME)
    if not os.path.exists(path):
        return
    _reach_write_checkpoint("restore_journal_remove", JOURNAL_FILENAME)
    os.remove(path)
    store._forget_document(path)
    store._sync_directory(store._config_dir, path, required=True)


def _journal(store: DataStore) -> Optional[dict]:
    path = os.path.join(store._config_dir, JOURNAL_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        loaded = store._read_yaml(JOURNAL_FILENAME)
    except StorageCorruptionError:
        return {"phase": "corrupt"}
    if not isinstance(loaded, dict) or loaded.get("schema_version") != BACKUP_SCHEMA_VERSION:
        return {"phase": "corrupt"}
    if loaded.get("phase") not in {"prepared", "applying", "completed", "rolled_back"}:
        return {"phase": "corrupt"}
    # Validate every recovery document before writing any of them.
    for key in ("snapshot", "intended"):
        documents = loaded.get(key)
        if not isinstance(documents, dict) or set(documents) != set(DOCUMENT_ORDER):
            return {"phase": "corrupt"}
        try:
            for filename, document in documents.items():
                if not isinstance(document, dict):
                    return {"phase": "corrupt"}
                store._validate_document(filename, document)
        except (ValueError, TypeError, StorageCorruptionError):
            return {"phase": "corrupt"}
    return loaded


def _require_no_journal(store: DataStore) -> None:
    if os.path.exists(os.path.join(store._config_dir, JOURNAL_FILENAME)):
        raise ConfigBackupError(
            "BACKUP_RECOVERY_REQUIRED", "Reconcile the previous restore before continuing.", 409,
        )


def _snapshot_documents(store: DataStore) -> dict:
    return {filename: copy.deepcopy(store._read_yaml(filename)) for filename in DOCUMENT_ORDER}


def _intended_documents(snapshot: dict, changes: dict) -> dict:
    intended = copy.deepcopy(snapshot)
    settings = intended["settings.yaml"]
    if not isinstance(settings, dict):
        settings = {}
        intended["settings.yaml"] = settings
    for key, value in changes["preferences"].items():
        settings[key] = value
    models = intended["models.yaml"].setdefault("models", [])
    for row in models:
        if isinstance(row, dict) and str(row.get("id")) in changes["models"]:
            row["config"] = _preserve_local_fields(
                row.get("config"), changes["models"][str(row["id"])],
            )
    templates = intended["model_config_templates.yaml"].setdefault("templates", [])
    for template_id, (_action, entry) in changes["templates"].items():
        record = {
            "id": template_id,
            "name": entry["name"],
            "description": entry["description"],
            "include_routing": entry["include_routing"],
            "engines_scope": entry["engines_scope"],
            "config": copy.deepcopy(entry["config"]),
        }
        replaced = False
        for index, existing in enumerate(templates):
            if isinstance(existing, dict) and str(existing.get("id")) == template_id:
                record["config"] = _preserve_local_fields(existing.get("config"), record["config"])
                templates[index] = record
                replaced = True
                break
        if not replaced:
            templates.append(record)
    routing = intended["llama_swap_routing.yaml"]
    for bucket, updates in (("profiles", "routing_profiles"), ("selectors", "routing_selectors")):
        target = routing.setdefault(bucket, {})
        for key, value in changes[updates].items():
            target[key] = _preserve_local_fields(target.get(key), value)
    return intended


def _preserve_local_fields(local: Any, incoming: Any) -> Any:
    """Replace portable settings, retaining only excluded local credentials/paths."""
    if not isinstance(local, dict):
        return copy.deepcopy(incoming)
    result = copy.deepcopy(incoming) if isinstance(incoming, dict) else {}
    for key, value in local.items():
        if _forbidden_key(key) or _strip(value) is _DROP:
            result[key] = copy.deepcopy(value)
        elif isinstance(value, dict):
            retained = _preserve_local_fields(value, result.get(key, {}))
            if retained:
                result[key] = retained
        elif isinstance(value, list) and _strip(value) != value:
            # Lists with local file references cannot safely be partially remapped.
            result[key] = copy.deepcopy(value)
    return result


def _documents_match(left: Mapping, right: Mapping) -> bool:
    for filename in DOCUMENT_ORDER:
        if _canonical(left.get(filename)) != _canonical(right.get(filename)):
            return False
    return True


def _export_preferences(settings: Mapping) -> dict:
    exported = {}
    if isinstance(settings, dict) and "public_inference_url" in settings:
        exported["public_inference_url"] = normalize_public_inference_url(
            settings.get("public_inference_url")
        )
    if isinstance(settings, dict) and "proxy_port" in settings:
        exported["proxy_port"] = _port(settings.get("proxy_port"))
    return exported


def _export_model(row: Mapping) -> dict:
    return {
        "ref": model_reference(row),
        "config": _portable(normalize_model_config(row.get("config"))),
    }


def _export_template(row: Mapping) -> dict:
    return {
        "id": _identifier(row.get("id"), "template id"),
        "name": str(row.get("name") or "").strip(),
        "description": str(row.get("description") or "").strip(),
        "include_routing": bool(row.get("include_routing")),
        "engines_scope": str(row.get("engines_scope") or "all"),
        "config": _portable(normalize_model_config(row.get("config"))),
    }


def _export_routing(routing: Mapping) -> dict:
    return {
        "profiles": _safe_names(routing.get("profiles") or {}),
        "selectors": _safe_names(routing.get("selectors") or {}),
    }


def _safe_names(value: Any) -> dict:
    cleaned = _portable(value)
    if not isinstance(cleaned, dict):
        return {}
    named = {}
    for key, item in cleaned.items():
        try:
            name = _identifier(key, "routing name")
        except ConfigBackupError:
            continue
        named[name] = item
    return named


def _named_map(value: Any) -> dict:
    cleaned = _require_portable(value)
    return {_identifier(key, "routing name"): item for key, item in cleaned.items()}


def model_reference(model: Mapping) -> str:
    source = model.get("source") if isinstance(model.get("source"), dict) else {}
    provider = str(source.get("provider") or "").strip()
    source_id = str(source.get("id") or "").strip()
    if provider and source_id:
        ref = f"{provider}:{source_id}"
    elif str(model.get("huggingface_id") or "").strip():
        ref = f"huggingface:{str(model.get('huggingface_id')).strip()}"
    else:
        ref = f"catalog:{str(model.get('id') or '').strip()}"
    return _identifier(ref, "model reference")


def _model_references(models: list) -> list[tuple[dict, str]]:
    """Repository identity alone is ambiguous when several quantizations exist."""
    rows = [(row, model_reference(row)) for row in models if isinstance(row, dict)]
    counts: dict[str, int] = {}
    for _, ref in rows:
        counts[ref] = counts.get(ref, 0) + 1
    return [
        (
            row,
            ref
            if counts[ref] == 1
            else f"{ref[:220]}@{hashlib.sha256(str(row.get('id')).encode()).hexdigest()[:16]}",
        )
        for row, ref in rows
    ]


def _current_state(store: DataStore) -> dict:
    settings = store._read_yaml("settings.yaml")
    models = store._read_yaml("models.yaml").get("models") or []
    templates = store._read_yaml("model_config_templates.yaml").get("templates") or []
    routing = store.get_llama_swap_routing()
    by_ref = {}
    by_id = {}
    for row, ref in _model_references(models):
        if not isinstance(row, dict) or not row.get("id"):
            continue
        by_id[str(row["id"])] = row
        by_ref[ref] = row
    return {
        "settings": settings if isinstance(settings, dict) else {},
        "models_by_ref": by_ref,
        "models_by_id": by_id,
        "templates": {
            str(row.get("id")): row
            for row in templates
            if isinstance(row, dict) and row.get("id")
        },
        "routing": routing if isinstance(routing, dict) else {"profiles": {}, "selectors": {}},
    }


def _build_plan(
    document: dict,
    current: dict,
    decisions: dict,
    mapping: dict,
    revisions: dict,
) -> dict:
    items = []
    changes = {
        "preferences": {},
        "models": {},
        "templates": {},
        "routing_profiles": {},
        "routing_selectors": {},
    }
    used_targets = set()
    for key, value in document["preferences"].items():
        local = current["settings"].get(key, _DROP)
        default = "add" if local is _DROP else "keep"
        action = _decision(
            "preferences",
            key,
            default,
            decisions,
            {"keep", "replace", "add", "skip"},
        )
        if action == "add" and local is not _DROP:
            action = "keep"
        items.append({"kind": "preference", "id": key, "action": action})
        if action in {"replace", "add"} and local != value:
            changes["preferences"][key] = value
    for entry in document["models"]:
        ref = entry["ref"]
        target_id = mapping.get(ref)
        local = current["models_by_ref"].get(ref)
        if target_id:
            local = current["models_by_id"].get(target_id)
            if local is None:
                raise ConfigBackupError(
                    "BACKUP_UNRESOLVED",
                    "A model mapping does not name an existing model.",
                    409,
                )
        if local is None:
            action = _decision("models", ref, "unresolved", decisions, {"skip", "unresolved"})
            items.append({"kind": "model", "id": ref, "action": action, "reason": "no local model"})
            continue
        action = _decision("models", ref, "keep", decisions, {"keep", "replace", "skip"})
        if action != "skip":
            local_id = str(local["id"])
            if local_id in used_targets:
                raise ConfigBackupError("BACKUP_MALFORMED", "Two backup models map to one local model.")
            used_targets.add(local_id)
        items.append({"kind": "model", "id": ref, "action": action, "local_id": local.get("id")})
        if action == "replace":
            changes["models"][str(local["id"])] = entry["config"]
    for entry in document["templates"]:
        template_id = entry["id"]
        local = current["templates"].get(template_id)
        default = "keep" if local else "add"
        action = _decision(
            "templates", template_id, default, decisions,
            {"keep", "replace", "add", "skip"},
        )
        if action == "add" and local is not None:
            action = "keep"
        items.append({"kind": "template", "id": template_id, "action": action})
        if action == "add" and local is None:
            changes["templates"][template_id] = ("add", entry)
        elif action == "replace":
            changes["templates"][template_id] = ("replace", entry)
    for bucket, change_key, prefix in (
        ("profiles", "routing_profiles", "profile"),
        ("selectors", "routing_selectors", "selector"),
    ):
        for name, value in document["routing"][bucket].items():
            item_id = f"{prefix}:{name}"
            local_bucket = current["routing"].get(bucket) or {}
            present = name in local_bucket
            default = "keep" if present else "add"
            action = _decision("routing", item_id, default, decisions, {"keep", "replace", "add", "skip"})
            if action == "add" and present:
                action = "keep"
            items.append({"kind": "routing", "id": item_id, "action": action})
            if action in {"replace", "add"} and local_bucket.get(name) != value:
                changes[change_key][name] = value
    applicable = not any(item["action"] == "unresolved" for item in items)
    identity = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "digest": _digest(document),
        "revisions": revisions,
        "decisions": decisions,
        "mapping": mapping,
        "items": [(item["kind"], item["id"], item["action"]) for item in items],
    }
    plan_id = hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest() if applicable else None
    return {"items": items, "changes": changes, "applicable": applicable, "plan_id": plan_id}


def _decision(group: str, item_id: str, default: str, decisions: dict, allowed: set[str]) -> str:
    chosen = (decisions.get(group) or {}).get(item_id, default)
    if not isinstance(chosen, str) or chosen not in allowed:
        raise ConfigBackupError("BACKUP_MALFORMED", "A restore decision is not one of the supported choices.")
    return chosen


def _validated_backup(backup: Any) -> dict:
    if not isinstance(backup, dict):
        raise ConfigBackupError("BACKUP_MALFORMED", "The backup must be a JSON object.")
    encoded = _canonical(backup).encode("utf-8")
    if len(encoded) > MAX_BACKUP_BYTES:
        raise ConfigBackupError("BACKUP_TOO_LARGE", "The backup is larger than the supported size.", 413)
    if backup.get("schema_version") != BACKUP_SCHEMA_VERSION or backup.get("kind") != BACKUP_KIND:
        raise ConfigBackupError("BACKUP_UNSUPPORTED", "This backup version is not supported.")
    _reject_forbidden(backup)
    preferences = backup.get("preferences")
    models = backup.get("models")
    templates = backup.get("templates")
    routing = backup.get("routing")
    if not isinstance(preferences, dict) or not isinstance(models, list) or not isinstance(templates, list):
        raise ConfigBackupError("BACKUP_MALFORMED", "The backup is missing a required section.")
    if not isinstance(routing, dict):
        raise ConfigBackupError("BACKUP_MALFORMED", "The backup is missing a required section.")
    cleaned_preferences = {}
    for key in ("public_inference_url", "proxy_port"):
        if key not in preferences:
            continue
        if key == "public_inference_url":
            try:
                cleaned_preferences[key] = normalize_public_inference_url(preferences[key])
            except (ValueError, TypeError, AttributeError) as exc:
                raise ConfigBackupError("BACKUP_MALFORMED", "The public inference URL is invalid.") from exc
        else:
            cleaned_preferences[key] = _port(preferences[key])
    cleaned_models = []
    seen_refs = set()
    for entry in models:
        if not isinstance(entry, dict):
            raise ConfigBackupError("BACKUP_MALFORMED", "A model entry must be an object.")
        ref = _identifier(entry.get("ref"), "model reference")
        if ref in seen_refs:
            raise ConfigBackupError("BACKUP_MALFORMED", "The backup repeats a model reference.")
        seen_refs.add(ref)
        cleaned_models.append({"ref": ref, "config": _require_portable(entry.get("config"))})
    cleaned_templates = []
    seen_templates = set()
    for entry in templates:
        if not isinstance(entry, dict):
            raise ConfigBackupError("BACKUP_MALFORMED", "A template entry must be an object.")
        template_id = _identifier(entry.get("id"), "template id")
        if template_id in seen_templates:
            raise ConfigBackupError("BACKUP_MALFORMED", "The backup repeats a template.")
        seen_templates.add(template_id)
        cleaned_templates.append({
            "id": template_id,
            "name": str(entry.get("name") or "").strip(),
            "description": str(entry.get("description") or "").strip(),
            "include_routing": bool(entry.get("include_routing")),
            "engines_scope": str(entry.get("engines_scope") or "all"),
            "config": _require_portable(entry.get("config")),
        })
    profiles = routing.get("profiles") if isinstance(routing.get("profiles"), dict) else None
    selectors = routing.get("selectors") if isinstance(routing.get("selectors"), dict) else None
    if profiles is None or selectors is None:
        raise ConfigBackupError("BACKUP_MALFORMED", "The backup is missing a required section.")
    return {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "kind": BACKUP_KIND,
        "application_version": str(backup.get("application_version") or ""),
        "preferences": cleaned_preferences,
        "models": cleaned_models,
        "templates": cleaned_templates,
        "routing": {
            "profiles": _named_map(profiles),
            "selectors": _named_map(selectors),
        },
    }


def _validated_decisions(decisions: Optional[Mapping]) -> dict:
    if decisions is None:
        return {}
    if not isinstance(decisions, dict):
        raise ConfigBackupError("BACKUP_MALFORMED", "Restore decisions must be an object.")
    cleaned = {}
    for group in ("preferences", "models", "templates", "routing"):
        value = decisions.get(group) or {}
        if not isinstance(value, dict):
            raise ConfigBackupError("BACKUP_MALFORMED", "Restore decisions must be an object.")
        cleaned[group] = {str(key): value[key] for key in value}
    return cleaned


def _validated_mapping(mapping: Optional[Mapping]) -> dict:
    if mapping is None:
        return {}
    if not isinstance(mapping, dict):
        raise ConfigBackupError("BACKUP_MALFORMED", "Model mappings must be an object.")
    cleaned = {}
    for ref, local_id in mapping.items():
        cleaned[_identifier(ref, "model reference")] = _identifier(local_id, "model id")
    return cleaned


def _portable(value: Any) -> Any:
    cleaned = _strip(value)
    if cleaned is _DROP:
        return {}
    return cleaned


def _require_portable(value: Any) -> Any:
    if not isinstance(value, dict):
        raise ConfigBackupError("BACKUP_MALFORMED", "A backup section has an unsupported shape.")
    _reject_forbidden(value)
    return _portable(value)


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if _forbidden_key(key):
                continue
            child = _strip(item)
            if child is _DROP:
                continue
            cleaned[str(key)] = child
        return cleaned
    if isinstance(value, list):
        cleaned = []
        for item in value:
            child = _strip(item)
            if child is _DROP:
                continue
            cleaned.append(child)
        return cleaned
    if isinstance(value, str):
        if _path_like(value):
            return _DROP
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _DROP


def _reject_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _forbidden_key(key):
                raise ConfigBackupError(
                    "BACKUP_FORBIDDEN",
                    "The backup contains a credential or machine-specific field.",
                )
            _reject_forbidden(item)
        return
    if isinstance(value, list):
        for item in value:
            _reject_forbidden(item)
        return
    if isinstance(value, str) and _path_like(value):
        raise ConfigBackupError(
            "BACKUP_FORBIDDEN",
            "The backup contains a path that cannot be restored.",
        )


def _path_like(value: str) -> bool:
    if "\x00" in value or re.search(r"(^|[/\\])\.\.($|[/\\])", value):
        return True
    if value.startswith("/") or value.startswith("\\"):
        return True
    return len(value) > 2 and value[1] == ":" and value[2] in {"\\", "/"}


def _identifier(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not _REF_TEXT.fullmatch(text) or ".." in text:
        raise ConfigBackupError("BACKUP_FORBIDDEN", f"A {label} cannot be restored.")
    return text


def _port(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ConfigBackupError("BACKUP_MALFORMED", "The proxy port is not a supported value.")
    return value


def _digest(document: Mapping) -> str:
    return hashlib.sha256(_canonical(document).encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
