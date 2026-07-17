"""Migrate audio.cpp request-defaults objects after endpoint / contract drift."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.audio_request_policy import build_request_policy
from backend.model_config import normalize_model_config

_DEFAULTS_KEYS = ("speech_defaults", "transcription_defaults", "task_defaults")


def _has_defaults_content(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    for key, item in value.items():
        if key == "options":
            if isinstance(item, dict) and any(
                nested is not None and nested != "" for nested in item.values()
            ):
                return True
            continue
        if item is None or item == "":
            continue
        if isinstance(item, list) and not item:
            continue
        return True
    return False


def migrate_request_defaults_section(
    audio_section: dict,
    *,
    expected_key: str,
    mark_reviewed_fingerprint: Optional[str] = None,
) -> Tuple[dict, bool, List[str]]:
    """Move misplaced defaults into *expected_key* and clear stale sibling objects.

    Returns ``(section, changed, notes)``.
    """
    section = dict(audio_section or {})
    keep = str(expected_key or "").strip() or "task_defaults"
    if keep not in _DEFAULTS_KEYS:
        keep = "task_defaults"
    notes: List[str] = []
    changed = False

    for key in _DEFAULTS_KEYS:
        if key == keep:
            continue
        if not _has_defaults_content(section.get(key)):
            continue
        if not _has_defaults_content(section.get(keep)):
            section[keep] = dict(section.get(key) or {})
            notes.append(f"Moved {key} → {keep}")
        else:
            notes.append(
                f"Cleared stale {key} (kept existing {keep})"
            )
        section[key] = {}
        changed = True

    if keep != "speech_defaults":
        presets = section.get("voice_presets")
        if isinstance(presets, dict) and presets:
            section["voice_presets"] = {}
            changed = True
            notes.append("Cleared voice_presets (not used outside speech_defaults)")
        if section.get("default_voice_preset") not in (None, ""):
            section["default_voice_preset"] = None
            changed = True
            notes.append("Cleared default_voice_preset")

    fingerprint = str(mark_reviewed_fingerprint or "").strip()
    if fingerprint and section.get("last_reviewed_fingerprint") != fingerprint:
        section["last_reviewed_fingerprint"] = fingerprint
        changed = True
        notes.append("Recorded last_reviewed_fingerprint")

    return section, changed, notes


def migrate_audio_model_defaults(
    store: Any,
    model: dict,
    *,
    mark_reviewed: bool = True,
    contract_fingerprint: Optional[str] = None,
    model_profile: Optional[dict] = None,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Migrate one model's audio.cpp defaults; persist when changed."""
    model_id = str(model.get("id") or "")
    config = normalize_model_config(model.get("config"))
    engines = config.get("engines") if isinstance(config.get("engines"), dict) else {}
    audio = engines.get("audio_cpp") if isinstance(engines.get("audio_cpp"), dict) else {}
    if not audio and str(config.get("engine") or "") != "audio_cpp":
        return {
            "id": model_id,
            "changed": False,
            "skipped": True,
            "reason": "not an audio.cpp model",
        }

    family = str(audio.get("family") or model.get("family") or "").strip()
    task = str(audio.get("task") or "").strip()
    inspection = {}
    if isinstance(model_profile, dict):
        raw = model_profile.get("inspection")
        if isinstance(raw, dict):
            inspection = raw
    policy = build_request_policy(
        task=task,
        family=family,
        inspection=inspection,
        model_profile=model_profile if isinstance(model_profile, dict) else None,
        source_path=source_path,
    )
    expected_key = str(policy.get("request_defaults_key") or "task_defaults")
    fingerprint = str(contract_fingerprint or "").strip() or None
    migrated, changed, notes = migrate_request_defaults_section(
        audio,
        expected_key=expected_key,
        mark_reviewed_fingerprint=fingerprint if mark_reviewed else None,
    )
    if not changed:
        return {
            "id": model_id,
            "changed": False,
            "skipped": False,
            "expected_key": expected_key,
            "api_endpoint": policy.get("api_endpoint"),
            "notes": [],
        }

    engines = dict(engines)
    engines["audio_cpp"] = migrated
    config = dict(config)
    config["engines"] = engines
    if str(config.get("engine") or "") != "audio_cpp":
        config["engine"] = "audio_cpp"
    store.update_model(model_id, {"config": config, "family": family or model.get("family")})
    return {
        "id": model_id,
        "changed": True,
        "skipped": False,
        "expected_key": expected_key,
        "api_endpoint": policy.get("api_endpoint"),
        "notes": notes,
    }


def migrate_audio_models_defaults(
    store: Any,
    *,
    model_ids: Optional[Sequence[str]] = None,
    mark_reviewed: bool = True,
) -> Dict[str, Any]:
    """Batch-migrate request defaults for audio.cpp models (optionally filtered)."""
    from backend.engine_param_catalog import get_model_profile_entry, get_version_entry
    from backend.engine_param_scanner import audio_cpp_model_profile_fingerprint

    active = store.get_active_engine_version("audio_cpp")
    entry = (
        get_version_entry(store, "audio_cpp", str(active.get("version") or ""))
        if active
        else None
    )
    fingerprint = str((entry or {}).get("contract_fingerprint") or "").strip() or None
    source_path = str((active or {}).get("source_path") or "") or None
    wanted = {
        str(item).strip() for item in (model_ids or []) if str(item).strip()
    }

    migrated: List[dict] = []
    unchanged: List[dict] = []
    skipped: List[dict] = []
    for model in store.list_models() or []:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id") or "")
        if wanted and model_id not in wanted:
            continue
        config = model.get("config") if isinstance(model.get("config"), dict) else {}
        engines = config.get("engines") if isinstance(config.get("engines"), dict) else {}
        audio = engines.get("audio_cpp") if isinstance(engines.get("audio_cpp"), dict) else {}
        engine = str(config.get("engine") or model.get("engine") or "").strip()
        if engine != "audio_cpp" and not audio:
            continue

        profile = None
        if active:
            try:
                fp = audio_cpp_model_profile_fingerprint(active, model)
                profile = get_model_profile_entry(
                    store, "audio_cpp", str(active.get("version") or ""), fp
                )
            except Exception:
                profile = None

        result = migrate_audio_model_defaults(
            store,
            model,
            mark_reviewed=mark_reviewed,
            contract_fingerprint=fingerprint,
            model_profile=profile,
            source_path=source_path,
        )
        if result.get("skipped"):
            skipped.append(result)
        elif result.get("changed"):
            migrated.append(result)
        else:
            unchanged.append(result)

    return {
        "migrated": migrated,
        "unchanged": unchanged,
        "skipped": skipped,
        "contract_fingerprint": fingerprint,
        "migrated_count": len(migrated),
    }


__all__ = [
    "migrate_audio_model_defaults",
    "migrate_audio_models_defaults",
    "migrate_request_defaults_section",
]
