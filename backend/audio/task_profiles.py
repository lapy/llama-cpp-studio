"""Unified facade for audio.cpp task family profiles and request defaults."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from backend.engines.audio_cpp.discovery import (
    LLAMA_SWAP_AUDIO_TASKS_PATH,
    resolve_api_endpoint,
    resolve_defaults_key_for_endpoint,
)
from backend.audio.request_policy import build_request_policy


def _family_key(family: Optional[str]) -> str:
    return str(family or "").strip().lower()


def _generic_profile_for_task(task: Optional[str], family: Optional[str]) -> Dict[str, Any]:
    task_key = str(task or "").strip().lower()
    family_key = _family_key(family) or "unknown"
    endpoint = api_endpoint_for(task_key, family_key)
    workflows = [task_key] if task_key else ["run"]
    return {
        "label": family_key.replace("_", " ").title(),
        "workflows": workflows,
        "summary": (
            f"Auto-discovered audio.cpp profile for {family_key}"
            f" ({task_key or 'task'}). Settings were detected automatically from the installed package."
        ),
        "api_hint": api_example_hint_for(task_key, family_key),
        "generic": True,
        "api_endpoint": endpoint,
    }


def overlay_task_profile_from_spec(
    profile: Optional[Dict[str, Any]],
    source_path: Optional[str],
    family: Optional[str],
    task: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Replace Studio prose with the installed model spec's name, text, and tasks."""
    if not source_path:
        return profile
    from backend.engines.audio_cpp.contracts import load_family_contract

    contract = load_family_contract(source_path, _family_key(family))
    if not contract:
        return profile
    from backend.engines.audio_cpp.spec_fields import profile_from_contract

    result = profile_from_contract(contract) or profile
    if result and task:
        from backend.audio.families.tts import is_tts_task

        result["requires_session_voice"] = bool(result.get("requires_session_voice")) and is_tts_task(task)
    return result


def task_profile_for(task: Optional[str], family: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Generic presentation until engine metadata is available."""
    if task or family:
        return _generic_profile_for_task(task, family)
    return None


def merge_request_field_groups(
    fallback: Sequence[Dict[str, Any]],
    preferred: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge engine field groups, keeping preferred metadata on duplicate keys."""
    seen = set()
    merged: List[Dict[str, Any]] = []
    for group in preferred or []:
        fields = []
        for field in group.get("fields") or []:
            key = str(field.get("key") or "")
            if key:
                seen.add(key)
            fields.append(field)
        if fields:
            merged.append({**group, "fields": fields})
    for group in fallback or []:
        fields = []
        for field in group.get("fields") or []:
            key = str(field.get("key") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            fields.append(field)
        if fields:
            merged.append({**group, "fields": fields})
    return merged


def request_field_groups_for(
    task: Optional[str],
    family: Optional[str] = None,
    *,
    profile_sections: Optional[Sequence[Dict[str, Any]]] = None,
    packaged_voices: Optional[Sequence[str]] = None,
    source_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    contract = None
    if source_path and family:
        from backend.engines.audio_cpp.contracts import load_family_contract

        contract = load_family_contract(source_path, _family_key(family))
    if contract:
        from backend.engines.audio_cpp.spec_fields import field_groups_from_contract

        groups = field_groups_from_contract(contract)
        if profile_sections:
            from backend.engines.audio_cpp.option_discovery import scanned_request_field_groups

            model_sections = [
                {**section, "params": [
                    row for row in section.get("params") or []
                    if row.get("transport") != "request_only"
                ]}
                for section in profile_sections
            ]
            scanned = scanned_request_field_groups(model_sections)
            scanned_fields = {
                field["key"]: field
                for group in scanned for field in group.get("fields") or []
            }
            for group in groups:
                for field in group.get("fields") or []:
                    help_field = scanned_fields.get(field["key"], {})
                    if field.get("enum_preset") and help_field.get("options"):
                        field["options"] = help_field["options"]
            # Specs can omit options still advertised by model-aware help.
            # Keep those gaps while preferring structured spec metadata.
            groups = merge_request_field_groups(
                scanned, groups
            )
    else:
        scanned: List[Dict[str, Any]] = []
        if profile_sections:
            from backend.engines.audio_cpp.option_discovery import scanned_request_field_groups

            scanned = scanned_request_field_groups(profile_sections)
        groups = scanned
    if packaged_voices:
        from backend.engines.audio_cpp.voices import apply_packaged_voice_field_options

        return apply_packaged_voice_field_options(groups, packaged_voices)
    return groups


def family_dependency_fields_for(
    task: Optional[str] = None,
    family: Optional[str] = None,
    *,
    profile_sections: Optional[Sequence[Dict[str, Any]]] = None,
    source_path: Optional[str] = None,
    family_dependencies: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Engine-declared dependency path fields for a family (including keys already scanned)."""
    family_key = _family_key(family)
    known = {
        str(param.get("key") or "")
        for section in (profile_sections or [])
        for param in section.get("params") or []
        if param.get("scope") in {"session_option", "load_option"} and param.get("key")
    }

    dependencies: List[Dict[str, Any]] = []
    if isinstance(family_dependencies, dict) and family_key:
        dependencies = list(family_dependencies.get(family_key) or [])
    if not dependencies and source_path and family_key:
        from backend.engines.audio_cpp.contracts import load_family_contract

        contract = load_family_contract(
            source_path, family_key, known_keys=known or None
        )
        dependencies = list((contract or {}).get("dependencies") or [])

    if not dependencies:
        return []

    from backend.engines.audio_cpp.contracts import dependency_sidecar_fields

    fields = dependency_sidecar_fields(
        family_key,
        dependencies,
    )
    for field in fields:
        peer = str((field.get("dependency") or {}).get("family") or "").strip()
        if peer and not field.get("install_hint"):
            field["install_hint"] = (
                f"Install a `{peer}` package from Models → Search (audio.cpp packages), "
                "then paste that package’s folder path here."
            )
    return fields


def _option_key_aliases(option_key: str) -> List[str]:
    key = str(option_key or "").strip()
    if not key:
        return []
    aliases = [key]
    if key.endswith("_model_path"):
        aliases.append(f"{key[:-11]}_path")
    elif key.endswith("_path") and not key.endswith("_model_path"):
        aliases.append(f"{key[:-5]}_model_path")
    return aliases


def apply_dependency_field_overlays(
    sections: Sequence[Dict[str, Any]],
    dependency_fields: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Merge peer metadata onto scanned params; return remaining gap fields.

    Scanned CLI options keep their discovery source of truth for presence, but
    pick up operator-facing labels, placeholders, dependency tags, and install
    hints from model-spec contracts.
    """
    sections_out: List[Dict[str, Any]] = [
        dict(section) for section in (sections or []) if isinstance(section, dict)
    ]
    for section in sections_out:
        params = section.get("params")
        if isinstance(params, list):
            section["params"] = [
                dict(param) for param in params if isinstance(param, dict)
            ]

    index: Dict[str, Dict[str, Any]] = {}
    for section in sections_out:
        for param in section.get("params") or []:
            key = str(param.get("key") or "").strip()
            if not key:
                continue
            for alias in _option_key_aliases(key):
                index.setdefault(alias, param)

    gaps: List[Dict[str, Any]] = []
    for field in dependency_fields:
        if not isinstance(field, dict):
            continue
        key = str(field.get("key") or "").strip()
        if not key:
            continue
        matched = None
        for alias in _option_key_aliases(key):
            matched = index.get(alias)
            if matched:
                break
        if matched is None:
            gaps.append(dict(field))
            continue
        for attr in ("label", "description", "placeholder", "install_hint"):
            if field.get(attr) and not matched.get(attr):
                matched[attr] = field[attr]
            elif field.get(attr) and attr in {"description", "placeholder", "install_hint"}:
                # Prefer dependency metadata over terse CLI help.
                matched[attr] = field[attr]
        if field.get("label"):
            matched["label"] = field["label"]
        if field.get("required") is True:
            matched["required"] = True
        dependency = field.get("dependency")
        if isinstance(dependency, dict):
            matched["dependency"] = dict(dependency)
            if dependency.get("required"):
                matched["required"] = True
        if field.get("install_hint"):
            matched["install_hint"] = field["install_hint"]
        matched["curated_dependency"] = True
    return sections_out, gaps


def sidecar_session_fields_for(
    task: Optional[str] = None,
    family: Optional[str] = None,
    *,
    profile_sections: Optional[Sequence[Dict[str, Any]]] = None,
    source_path: Optional[str] = None,
    family_dependencies: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Peer dependency path overlays missing from the CLI scan."""
    all_fields = family_dependency_fields_for(
        task,
        family,
        profile_sections=profile_sections,
        source_path=source_path,
        family_dependencies=family_dependencies,
    )
    if not all_fields:
        return []
    _sections, gaps = apply_dependency_field_overlays(
        profile_sections or [],
        all_fields,
    )
    return gaps


def is_profiled_task(task: Optional[str], family: Optional[str] = None) -> bool:
    return task_profile_for(task, family) is not None


def _synthetic_inspection_tasks(task: Optional[str], family: Optional[str]) -> List[str]:
    """Only use the explicitly selected task when inspection is unavailable."""
    task_key = str(task or "").strip().lower()
    return [task_key] if task_key else []


def request_defaults_key_for(
    task: Optional[str],
    family: Optional[str] = None,
    *,
    inspection: Optional[dict] = None,
    help_option_keys: Optional[Sequence[str]] = None,
    model_profile: Optional[dict] = None,
) -> str:
    endpoint = api_endpoint_for(
        task,
        family,
        inspection=inspection,
        help_option_keys=help_option_keys,
        model_profile=model_profile,
    )
    return resolve_defaults_key_for_endpoint(endpoint)


def api_endpoint_for(
    task: Optional[str],
    family: Optional[str] = None,
    *,
    inspection: Optional[dict] = None,
    help_option_keys: Optional[Sequence[str]] = None,
    model_profile: Optional[dict] = None,
) -> str:
    if inspection is not None or help_option_keys is not None or model_profile is not None:
        policy = build_request_policy(
            task=task,
            family=family,
            inspection=inspection,
            model_profile=model_profile,
            help_option_keys=help_option_keys,
        )
        return str(policy.get("api_endpoint") or LLAMA_SWAP_AUDIO_TASKS_PATH)

    synthetic = _synthetic_inspection_tasks(task, family)
    return resolve_api_endpoint(
        task=task,
        inspection_tasks=synthetic,
        help_option_keys=help_option_keys,
    )


def api_example_hint_for(
    task: Optional[str],
    family: Optional[str] = None,
    *,
    inspection: Optional[dict] = None,
    help_option_keys: Optional[Sequence[str]] = None,
    model_profile: Optional[dict] = None,
) -> str:
    endpoint = api_endpoint_for(
        task,
        family,
        inspection=inspection,
        help_option_keys=help_option_keys,
        model_profile=model_profile,
    )
    if endpoint in {"/v1/audio/transcriptions", "/v1/audio/transcriptions/details"}:
        return (
            "JSON uses a server-local audio path. Multipart upload with a file field is also supported."
        )
    if endpoint == "/v1/audio/speech":
        return "OpenAI-compatible speech synthesis request."
    if endpoint == "/v1/audio/alignments":
        return (
            "Multipart alignment: file + text (+ optional language). "
            "Server-local paths can still use llama-swap /audioapi/v1/tasks/run."
        )
    return (
        "Generic task request via llama-swap /audioapi/v1/tasks/run "
        "(rewritten upstream to audio.cpp /v1/tasks/run)."
    )


def supports_voice_presets_for(
    *,
    request_defaults_key: Optional[str],
) -> bool:
    """Voice presets apply only when the model routes through speech_defaults."""
    return str(request_defaults_key or "").strip() == "speech_defaults"
