"""audio.cpp's CMake build-option surface, derived from engine data.

The catalog itself (option table, UI categories, build types, primary keys) is
engine DATA: it lives on the engine spec (`EngineSpec.build_options`), is
exported into the generated definition, and everything here derives from it.
This module keeps only the behavior: coercion/defaults, the suggested backend,
the UI payload and the dependency predicate.

Aligned with upstream 0xShug0/audio.cpp CMakeLists.txt (ENGINE_* / AUDIOCPP_*).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

ENGINE_ID = "audio_cpp"


@dataclass(frozen=True)
class AudioBuildOptionDef:
    key: str
    field: str
    kind: str  # bool | str | enum | int
    default: Any
    label: str
    desc: str
    category: str
    cmake: Optional[str] = None
    requires: Optional[str] = None
    enum_values: Optional[tuple] = None
    special: Optional[str] = None


def _catalog_data() -> Dict[str, Any]:
    from backend.engines.registry import get_engine_spec

    spec = get_engine_spec(ENGINE_ID)
    data = dict(spec.build_options) if spec is not None else {}
    if not data:
        raise RuntimeError("audio.cpp build-option catalog is unavailable")
    return data


def build_types() -> tuple:
    return tuple(_catalog_data().get("build_types") or ())


def build_options() -> tuple:
    options = []
    for raw in _catalog_data().get("options") or ():
        options.append(
            AudioBuildOptionDef(
                key=str(raw["key"]),
                field=str(raw["field"]),
                kind=str(raw["kind"]),
                default=raw.get("default"),
                label=str(raw["label"]),
                desc=str(raw["desc"]),
                category=str(raw["category"]),
                cmake=raw.get("cmake"),
                requires=raw.get("requires"),
                enum_values=tuple(raw["enum_values"]) if raw.get("enum_values") else None,
                special=raw.get("special"),
            )
        )
    return tuple(options)


def categories() -> tuple:
    return tuple(dict(item) for item in (_catalog_data().get("categories") or ()))


def option_by_key() -> Dict[str, AudioBuildOptionDef]:
    return {option.key: option for option in build_options()}


def primary_keys() -> frozenset:
    return frozenset(_catalog_data().get("primary_keys") or ())


def category_requires() -> Dict[str, str]:
    return {
        str(category["id"]): str(category["requires"])
        for category in categories()
        if category.get("requires")
    }


def default_build_settings() -> Dict[str, Any]:
    data = _catalog_data()
    default_build_type = str(
        data.get("build_type_default") or (data.get("build_types") or ["RelWithDebInfo"])[0]
    )
    out: Dict[str, Any] = {"build_type": default_build_type}
    for opt in build_options():
        out[opt.key] = opt.default
    out["backend"] = "cpu"
    return out


def coerce_build_settings(settings: Optional[dict]) -> Dict[str, Any]:
    base = default_build_settings()
    if not isinstance(settings, dict):
        return base

    def _bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    def _str(v: Any, default: str = "") -> str:
        return str(v).strip() if v is not None else default

    def _int(v: Any, default: int = 0) -> int:
        try:
            return max(0, int(v))
        except (TypeError, ValueError):
            return default

    out = dict(base)
    build_type = _str(settings.get("build_type"), base["build_type"])
    if build_type not in build_types():
        build_type = base["build_type"]
    out["build_type"] = build_type

    for opt in build_options():
        raw = settings.get(opt.key, base[opt.key])
        if opt.kind == "bool":
            out[opt.key] = _bool(raw)
        elif opt.kind == "int":
            out[opt.key] = _int(raw, int(opt.default or 0))
        elif opt.kind == "enum" and opt.enum_values:
            val = _str(raw, str(opt.default))
            out[opt.key] = val if val in opt.enum_values else opt.default
        else:
            out[opt.key] = _str(raw, str(opt.default) if opt.default is not None else "")

    out["backend"] = derived_backend(out)
    return out


def backend_option_key() -> str:
    """The catalog option that selects the build backend (``special: backend``)."""
    for option in build_options():
        if option.special == "backend":
            return option.key
    return "cuda"


def derived_backend(settings: dict) -> str:
    if settings.get(backend_option_key()):
        return "cuda"
    return "cpu"


def settings_to_field_kwargs(settings: dict) -> Dict[str, Any]:
    normalized = coerce_build_settings(settings)
    kwargs: Dict[str, Any] = {
        "build_type": normalized["build_type"],
        "backend": normalized["backend"],
    }
    for opt in build_options():
        kwargs[opt.field] = normalized[opt.key]
    return kwargs


def catalog_for_ui() -> Dict[str, Any]:
    catalog_categories = categories()
    primary = primary_keys()
    category_reqs = category_requires()
    by_cat: Dict[str, List[dict]] = {c["id"]: [] for c in catalog_categories}
    for opt in build_options():
        entry = {
            "key": opt.key,
            "type": opt.kind,
            "label": opt.label,
            "desc": opt.desc,
            "default": opt.default,
            "requires": opt.requires,
            "cmake": opt.cmake,
            "primary": opt.category != "backends" or opt.key in primary,
        }
        if opt.enum_values:
            entry["enum_values"] = list(opt.enum_values)
        by_cat.setdefault(opt.category, []).append(entry)

    rendered = []
    for cat in catalog_categories:
        options = by_cat.get(cat["id"]) or []
        if not options:
            continue
        rendered.append(
            {
                "id": cat["id"],
                "label": cat["label"],
                "requires": category_reqs.get(cat["id"]),
                "collapsed": bool(cat.get("collapsed", True)),
                "options": options,
            }
        )

    return {
        "engine": ENGINE_ID,
        "categories": rendered,
        "defaults": default_build_settings(),
        "build_types": list(build_types()),
    }


def parent_enabled(settings_or_config: Any, requires: Optional[str]) -> bool:
    if not requires:
        return True
    get = (
        settings_or_config.get
        if isinstance(settings_or_config, dict)
        else lambda k, d=False: getattr(settings_or_config, k, d)
    )
    if requires == "cuda_or_hip":
        return bool(get("cuda"))
    if requires == "model_set_custom":
        return str(get("model_set") or "") == "custom"
    return bool(get(requires))
