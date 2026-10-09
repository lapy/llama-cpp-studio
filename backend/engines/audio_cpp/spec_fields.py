"""Turn audio.cpp ``model_specs/<family>.json`` options into Studio fields.

Every family (speech, transcription, music, conversion, analysis, separation,
alignment) ships request, session, and load options in that file. Studio reads
the installed checkout, so a newer audio.cpp updates the forms without a
hardcoded option list.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from backend.engines.audio_cpp.contracts import load_family_contract, public_option_key

_SCOPE_TO_STUDIO = {
    "request": "request_option",
    "session": "session_option",
    "load": "load_option",
}
_SECTION = {
    "request": ("model_request_options", "Model request options", "--request-option"),
    "session": ("model_session_options", "Model session options", "--session-option"),
    "load": ("model_load_options", "Model load options", "--load-option"),
}
_FIELD_TYPE = {
    "float": "float",
    "int": "int",
    "bool": "bool",
    "enum": "string",
    "string": "string",
    "string_list": "string",
    "path": "string",
    "audio_path": "string",
}


def _label(name: str) -> str:
    text = str(name or "").replace("_", " ").replace(".", " ").strip()
    if text.endswith(" sec"):
        text = text[:-4] + " (seconds)"
    elif text.endswith(" seconds"):
        text = text[:-8] + " (seconds)"
    if not text:
        return str(name or "")
    return text[0].upper() + text[1:]


def _option_key(family: str, scope: str, name: str) -> str:
    if scope == "request" or "." in name:
        return name
    return public_option_key(family, name)


def _field_type(raw_type: str) -> str:
    return _FIELD_TYPE.get(str(raw_type or "string"), "string")


def iter_spec_options(contract: Optional[dict]) -> Iterable[tuple[str, str, dict]]:
    """Yield ``(scope, public_key, raw_option)`` for one family contract."""
    if not isinstance(contract, dict):
        return
    family = str(contract.get("family") or "")
    options = contract.get("options") if isinstance(contract.get("options"), dict) else {}
    for scope in ("request", "session", "load"):
        rows = options.get(scope) or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name or name == "text":
                continue
            key = _option_key(family, scope, name)
            if key:
                yield scope, key, row


def request_fields_from_contract(contract: Optional[dict]) -> List[Dict[str, Any]]:
    """Defaults-tab fields for ``options.request``."""
    fields: List[Dict[str, Any]] = []
    for _scope, key, row in iter_spec_options(contract):
        if _scope != "request":
            continue
        raw_type = str(row.get("type") or "string")
        field: Dict[str, Any] = {
            "key": key,
            "label": _label(key),
            "type": _field_type(raw_type),
            "nested": True,
            "options_key": key,
            "discovery_source": "model_specs",
        }
        description = str(row.get("description") or "").strip()
        if description:
            field["description"] = description
            field["hint"] = description
        if row.get("default") is not None:
            field["default"] = row.get("default")
            field["placeholder"] = str(row.get("default"))
        if raw_type == "enum" and row.get("preset"):
            field["enum_preset"] = row["preset"]
        minimum = row.get("min", row.get("minimum"))
        maximum = row.get("max", row.get("maximum"))
        if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
            field["minimum"] = minimum
        if isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
            field["maximum"] = maximum
        values = row.get("values")
        if raw_type == "enum" and isinstance(values, list) and values:
            field["options"] = [{"value": str(value), "label": str(value)} for value in values]
        fields.append(field)
    return fields


def spec_option_params(contract: Optional[dict]) -> List[Dict[str, Any]]:
    """Scan-registry rows for request, session, and load options."""
    family = str((contract or {}).get("family") or "")
    params: List[Dict[str, Any]] = []
    for scope, key, row in iter_spec_options(contract):
        section_id, section_label, flag = _SECTION[scope]
        raw_type = str(row.get("type") or "string")
        field_type = _field_type(raw_type)
        param: Dict[str, Any] = {
            "key": key,
            "label": _label(key),
            "type": field_type,
            "scalar_type": field_type,
            "value_kind": "enum" if raw_type == "enum" else "scalar",
            "description": str(row.get("description") or "").strip(),
            "scope": _SCOPE_TO_STUDIO[scope],
            "section_id": section_id,
            "section_label": section_label,
            "source": "model_specs",
            "discovery_source": "model_specs",
            "transport": "key_value_option",
            "read_only": scope == "request",
            "reserved": False,
            "required": bool(row.get("required")) and row.get("default") is None,
            "primary_flag": flag,
            "flags": [],
            "aliases": [],
            "emission": {
                "transport": "key_value_option",
                "flag": flag,
                "option_key": key,
            },
        }
        if row.get("default") is not None:
            param["default"] = row.get("default")
        if raw_type == "enum" and row.get("preset"):
            param["enum_preset"] = row["preset"]
        minimum = row.get("min", row.get("minimum"))
        maximum = row.get("max", row.get("maximum"))
        if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
            param["minimum"] = minimum
        if isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
            param["maximum"] = maximum
        values = row.get("values")
        if raw_type == "enum" and isinstance(values, list) and values:
            param["type"] = "select"
            param["options"] = [{"value": str(value), "label": str(value)} for value in values]
        if family:
            param["family"] = family
        params.append(param)
    return params


def merge_spec_contract_into_sections(
    sections: List[dict],
    contract: Optional[dict],
) -> List[dict]:
    """Overlay model-spec types onto help/source sections and add missing keys."""
    spec_params = spec_option_params(contract)
    if not spec_params:
        return list(sections or [])
    copied: List[dict] = []
    index: Dict[tuple[str, str], dict] = {}
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        params = []
        for param in section.get("params") or []:
            if not isinstance(param, dict):
                continue
            row = dict(param)
            params.append(row)
            key = str(row.get("key") or "")
            scope = str(row.get("scope") or "")
            if key:
                index[(scope, key)] = row
        copied.append({**section, "params": params})
    additions: Dict[tuple[str, str], List[dict]] = {}
    for spec in spec_params:
        key = str(spec.get("key") or "")
        scope = str(spec.get("scope") or "")
        existing = index.get((scope, key))
        if existing is None:
            section_id = str(spec.pop("section_id") or "model_specs")
            section_label = str(spec.pop("section_label") or "Model spec")
            additions.setdefault((section_id, section_label), []).append(spec)
            index[(scope, key)] = spec
            continue
        # Structured engine metadata wins over heuristic help typing, including
        # changed numeric types and removed enum/range/default constraints.
        for field in (
            "type",
            "scalar_type",
            "value_kind",
            "required",
            "default",
            "minimum",
            "maximum",
            "options",
        ):
            # Preset enums are expanded by model-aware help. A spec references
            # their engine-owned name rather than repeating their choices.
            if field == "options" and spec.get("enum_preset") and field not in spec:
                continue
            existing.pop(field, None)
            if field in spec:
                existing[field] = spec[field]
        if spec.get("description"):
            existing["description"] = spec["description"]
        existing["discovery_source"] = "model_specs"
        if spec.get("enum_preset") and existing.get("options"):
            existing["type"] = "select"
    for (section_id, section_label), params in additions.items():
        match = next((section for section in copied if section.get("id") == section_id), None)
        if match is None:
            copied.append({"id": section_id, "label": section_label, "params": params})
        else:
            match["params"].extend(params)
    return copied


_PRESET_FIELDS = frozenset({"voice_id", "voice_ref", "speaker", "reference_text", "voice_samples"})
_DESIGN_FIELDS = frozenset({"instruction", "instructions", "instruct", "caption"})


def _capability_tokens(contract: dict) -> set[str]:
    tokens: set[str] = set()
    capabilities = (
        contract.get("capabilities") if isinstance(contract.get("capabilities"), dict) else {}
    )
    for value in capabilities.values():
        if isinstance(value, list):
            tokens.update(str(item).strip() for item in value if str(item).strip())
        elif value:
            tokens.add(str(value).strip())
    return tokens


def _preset_field(key: str, label: str, field_type: str = "string") -> Dict[str, Any]:
    field: Dict[str, Any] = {
        "key": key,
        "label": label,
        "type": field_type,
        "preset_field": key in _PRESET_FIELDS,
        "discovery_source": "model_specs",
    }
    if key in {"voice_ref", "reference_text", "voice_id", "speaker"}:
        field["speech_field"] = key
    return field


def field_groups_from_contract(contract: Optional[dict]) -> List[Dict[str, Any]]:
    """Voice, design, and request-option groups taken only from a model spec."""
    if not isinstance(contract, dict):
        return []
    tokens = _capability_tokens(contract)
    request_fields = request_fields_from_contract(contract)
    request_keys = {str(field.get("key") or "") for field in request_fields}
    voice: List[Dict[str, Any]] = []
    design: List[Dict[str, Any]] = []
    options: List[Dict[str, Any]] = []
    if "speaker_reference" in tokens and "voice_ref" not in request_keys:
        voice.append(_preset_field("voice_ref", "Reference audio (WAV)", "path"))
    ui = contract.get("ui") if isinstance(contract.get("ui"), dict) else {}
    builtin = [str(item).strip() for item in (ui.get("builtin_voices") or []) if str(item).strip()]
    if (
        "built_in_voices" in tokens or builtin or str(ui.get("default_voice") or "").strip()
    ) and "voice_id" not in request_keys:
        voice_field = _preset_field("voice_id", "Built-in voice id")
        if builtin:
            voice_field["options"] = [{"value": voice, "label": voice} for voice in builtin]
        voice.append(voice_field)
    if "multi_speaker" in tokens and "voice_samples" not in request_keys:
        voice.append(_preset_field("voice_samples", "Speaker reference WAVs"))
    for field in request_fields:
        key = str(field.get("key") or "")
        row = dict(field)
        if key in _PRESET_FIELDS:
            row["preset_field"] = True
            voice.append(row)
        elif key in _DESIGN_FIELDS:
            design.append(row)
        else:
            options.append(row)
    groups: List[Dict[str, Any]] = []
    if voice:
        groups.append(
            {
                "id": "voice",
                "label": "Voice & reference",
                "description": "Voice inputs declared by the installed audio.cpp model spec.",
                "fields": voice,
            }
        )
    if design:
        groups.append(
            {
                "id": "design",
                "label": "Voice design",
                "description": "Design inputs declared by the installed audio.cpp model spec.",
                "fields": design,
            }
        )
    if options:
        groups.append(
            {
                "id": "options",
                "label": "Model request options",
                "description": "Options declared by the installed audio.cpp model spec.",
                "fields": options,
            }
        )
    return groups


def profile_from_contract(contract: Optional[dict]) -> Optional[Dict[str, Any]]:
    """Studio profile whose text, tasks, and default voice come from the spec."""
    if not isinstance(contract, dict) or not contract.get("family"):
        return None
    ui = contract.get("ui") if isinstance(contract.get("ui"), dict) else {}
    default_voice = str(ui.get("default_voice") or "").strip()
    builtin = [str(item).strip() for item in (ui.get("builtin_voices") or []) if str(item).strip()]
    tasks = [str(item).strip() for item in (contract.get("tasks") or []) if str(item).strip()]
    profile: Dict[str, Any] = {
        "label": str(contract.get("display_name") or contract.get("family")).strip(),
        "summary": str(contract.get("description") or "").strip()
        or f"audio.cpp model spec for {contract.get('family')}.",
        "workflows": tasks or ["run"],
        "category": str(contract.get("category") or "").strip(),
        "generic": False,
    }
    if default_voice:
        profile["default_voice"] = default_voice
        profile["requires_session_voice"] = True
    if builtin:
        profile["builtin_voices"] = builtin
    return profile


def load_request_fields(source_path: Optional[str], family: Optional[str]) -> List[Dict[str, Any]]:
    """Request fields from the installed audio.cpp tree, for any family."""
    if not source_path or not family:
        return []
    contract = load_family_contract(source_path, str(family).strip().lower().replace("-", "_"))
    return request_fields_from_contract(contract)


def workspace_request_fields(
    contract: Optional[dict], sections: Optional[List[dict]] = None
) -> List[Dict[str, Any]]:
    """Build task inputs from model-aware help and structured request options.

    CLI output/batch/utility flags are not JSON request inputs. Only model input
    sections and key/value request options participate. Preserve their transport
    so a new option does not need a Studio serialization rule.
    """
    fields: Dict[str, dict] = {}
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        for param in section.get("params") or []:
            if not isinstance(param, dict) or param.get("scope") != "request_option":
                continue
            key = str(param.get("key") or "").strip()
            transport = param.get("transport")
            section_id = param.get("section_id") or section.get("id")
            nested = transport == "key_value_option"
            if not key or (not nested and section_id not in {"available_input_options", "inputs"}):
                continue
            fields[key] = {**param, "key": key, "nested": nested}
    for field in request_fields_from_contract(contract):
        key = field["key"]
        previous = fields.get(key, {})
        fields[key] = {**previous, **field, "nested": previous.get("nested", True)}
    # Text is a protocol input rather than a request-default option.
    for row in ((contract or {}).get("options") or {}).get("request") or []:
        if not isinstance(row, dict):
            continue
        if row.get("name") == "text":
            fields["text"] = {
                "key": "text", "label": _label("text"), "type": "string",
                "nested": False, "required": bool(row.get("required")),
                "description": row.get("description", ""),
                **({"default": row["default"]} if "default" in row else {}),
            }
        elif row.get("name") in fields:
            fields[row["name"]]["required"] = bool(row.get("required"))
    return list(fields.values())
