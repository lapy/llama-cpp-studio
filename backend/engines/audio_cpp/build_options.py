"""audio.cpp's CMake build-option surface, derived from engine data.

The catalog itself (option table, UI categories, build types, primary keys) is
engine DATA: it lives on the engine spec (`EngineSpec.build_options`), is
exported into the generated definition, and everything here derives from it.
This module keeps only the behavior: coercion/defaults, the derived build
backend, CMake argv rendering, validation, the UI payload and the dependency
predicate.

Aligned with upstream 0xShug0/audio.cpp CMakeLists.txt (v0.9.5 + main after it:
``ENGINE_*`` / ``AUDIOCPP_*`` options and cache strings). Upstream keeps adding
backends and artifact toggles, so every option Studio knows about is always
passed explicitly: a build never inherits an upstream default it did not ask
for. Options upstream grows later are pinned by
``tests/test_audio_build_options.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

ENGINE_ID = "audio_cpp"

#: Backend every build always has: the CPU path is always compiled.
BASE_BACKEND = "cpu"

#: Catalog options whose value is not a plain ``-D<name>=<value>`` CMake flag.
#: ``jobs`` drives ``--parallel``, the rest are environment/custom-argv inputs.
_NON_CMAKE_SPECIALS = frozenset({"jobs", "custom_cmake", "cflags", "cxxflags"})

#: CMake list variable: semicolon-separated on the CMake side.
_CMAKE_LIST_KEYS = frozenset({"server_frontend_modules"})


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
    #: False marks an upstream knob Studio catalogues but cannot build: the UI
    #: disables it and validation refuses it instead of failing inside CMake.
    supported: bool = True


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
                supported=raw.get("supported", True) is not False,
            )
        )
    return tuple(options)


def unsupported_option_keys() -> tuple:
    """Catalog keys Studio catalogues but cannot build on any host."""
    return tuple(option.key for option in build_options() if not option.supported)


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


def backend_option_key() -> str:
    """The catalog option marked as the primary build backend (``special: backend``)."""
    for option in build_options():
        if option.special == "backend":
            return option.key
    return ""


def backend_option_keys() -> tuple:
    """Acceleration options, primary first, then upstream ``AUDIOCPP_BUILD_BACKENDS`` order.

    Upstream appends enabled backends in catalog order; the primary backend wins
    precedence so a CUDA build that also enables another backend still reports
    CUDA.
    """
    keys = [option.key for option in build_options() if option.category == "backends"]
    primary = backend_option_key()
    if primary in keys:
        keys.remove(primary)
        keys.insert(0, primary)
    return tuple(keys)


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


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _coerce_str(v: Any, default: str = "") -> str:
    return str(v).strip() if v is not None else default


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return default


def coerce_build_settings(settings: Optional[dict]) -> Dict[str, Any]:
    """Normalize a stored/partial settings dict onto the catalog.

    Unknown keys are dropped, wrong types are repaired, and ``backend`` is
    re-derived from the enabled backends. Dependent options are never forced:
    a partial save keeps what the user asked for so validation can explain the
    conflict instead of silently discarding a choice.
    """
    base = default_build_settings()
    if not isinstance(settings, dict):
        return base

    out = dict(base)
    build_type = _coerce_str(settings.get("build_type"), base["build_type"])
    if build_type not in build_types():
        build_type = base["build_type"]
    out["build_type"] = build_type

    for opt in build_options():
        raw = settings.get(opt.key, base[opt.key])
        if opt.kind == "bool":
            out[opt.key] = _coerce_bool(raw)
        elif opt.kind == "int":
            out[opt.key] = _coerce_int(raw, int(opt.default or 0))
        elif opt.kind == "enum" and opt.enum_values:
            val = _coerce_str(raw, str(opt.default))
            out[opt.key] = val if val in opt.enum_values else opt.default
        else:
            out[opt.key] = _coerce_str(raw, str(opt.default) if opt.default is not None else "")

    out["backend"] = derived_backend(out)
    return out


def backend_option_key() -> str:
    """The catalog option that selects the build backend (``special: backend``)."""
    for option in build_options():
        if option.special == "backend":
            return option.key
    return "cuda"


def derived_backend(settings: dict) -> str:
    """Primary backend for a settings dict (first enabled backend wins)."""
    for key in backend_option_keys():
        if settings.get(key):
            return key
    return BASE_BACKEND


def supported_backends() -> tuple:
    """Launch backends Studio can build, run and ship end to end.

    Derived from the catalog: ``cpu`` is always compiled, and a backend-option
    key counts only when the catalog marks it supported. Every other upstream
    backend is documented, persisted and pinned off.
    """
    options = option_by_key()
    return (
        BASE_BACKEND,
        *(key for key in backend_option_keys() if options[key].supported),
    )


def settings_to_field_kwargs(settings: dict) -> Dict[str, Any]:
    normalized = coerce_build_settings(settings)
    kwargs: Dict[str, Any] = {
        "build_type": normalized["build_type"],
        "backend": normalized["backend"],
    }
    for opt in build_options():
        kwargs[opt.field] = normalized[opt.key]
    return kwargs


def cmake_list(value: Any) -> str:
    """Normalize a user-typed list to the semicolon form CMake cache entries use."""
    text = _coerce_str(value)
    if not text:
        return ""
    parts = [part.strip() for part in text.replace(",", ";").split(";")]
    return ";".join(part for part in parts if part)


def cmake_arguments(settings: dict) -> List[str]:
    """Render the catalog as explicit ``-D`` argv for one configure.

    Every catalog option that names a CMake variable is emitted, including the
    ones left at their default: pinning ``OFF``/empty keeps a build
    deterministic when upstream flips a default (``ENGINE_ENABLE_METAL`` is ON
    on Apple hosts, for example) or when a fork changes one.
    """
    normalized = coerce_build_settings(settings)
    args: List[str] = []
    for opt in build_options():
        if not opt.cmake or opt.special in _NON_CMAKE_SPECIALS:
            continue
        value = normalized.get(opt.key)
        if opt.kind == "bool":
            rendered = "ON" if value else "OFF"
        elif opt.kind == "int":
            rendered = str(int(value or 0))
        elif opt.key in _CMAKE_LIST_KEYS:
            rendered = cmake_list(value)
        else:
            rendered = _coerce_str(value)
        args.append(f"-D{opt.cmake}={rendered}")
    return args


def validate_build_settings(settings: Any) -> List[str]:
    """Conflicts that make an audio.cpp configure fail or produce an unsupported build.

    Mirrors the FATAL_ERROR guards in upstream ``CMakeLists.txt`` /
    ``app/server/server_frontends.cmake`` so a user learns about them in the
    save/build request instead of from a CMake log. It also refuses catalog
    options Studio cannot build, which keeps the supported-backend contract
    honest even when settings arrive from an API client instead of the UI.
    Returns user-facing messages; an empty list means the settings are buildable.
    """
    normalized = coerce_build_settings(settings)
    errors: List[str] = []

    for opt in build_options():
        if opt.supported or not normalized.get(opt.key):
            continue
        where = f" ({opt.cmake})" if opt.cmake else ""
        errors.append(
            f"{opt.label}{where} is not part of Studio's audio.cpp build matrix; clear {opt.key}."
        )

    if normalized.get("cuda") and normalized.get("hip"):
        errors.append("CUDA and HIP are mutually exclusive; enable exactly one of them.")

    if normalized.get("hip_strix_halo") and not normalized.get("hip"):
        errors.append("Strix Halo HIP optimizations require the HIP backend.")

    if normalized.get("use_system_openssl") and not normalized.get("native_model_manager"):
        errors.append("System OpenSSL requires the native model manager.")

    modules = normalized.get("server_frontend_modules") or ""
    if modules and not normalized.get("build_server_frontends"):
        errors.append("Server frontend modules require server frontends.")
    if modules and not normalized.get("server_frontends_dir"):
        errors.append(
            "Server frontend modules require a frontends package directory "
            "(the external/audio.cpp-server-frontends checkout by default)."
        )

    if normalized.get("zonos3_textnorm") and not normalized.get("zonos3_textnorm_dir"):
        errors.append(
            "ZONOS3 text normalization requires a zyphra-textnorm checkout (Cargo.toml root)."
        )

    if normalized.get("model_set") == "custom" and not normalized.get("models"):
        errors.append("A custom model set needs at least one model target.")

    return errors


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
            "supported": opt.supported,
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
        "supported_backends": list(supported_backends()),
    }


def parent_enabled(settings_or_config: Any, requires: Optional[str]) -> bool:
    """True when the option/category gated by ``requires`` is enabled."""
    if not requires:
        return True
    get = (
        settings_or_config.get
        if isinstance(settings_or_config, dict)
        else lambda k, d=False: getattr(settings_or_config, k, d)
    )
    if requires == "cuda_or_hip":
        return bool(get("cuda") or get("hip"))
    if requires == "model_set_custom":
        return str(get("model_set") or "") == "custom"
    return bool(get(requires))


__all__ = [
    "BASE_BACKEND",
    "ENGINE_ID",
    "AudioBuildOptionDef",
    "backend_option_key",
    "backend_option_keys",
    "build_options",
    "build_types",
    "catalog_for_ui",
    "categories",
    "cmake_arguments",
    "cmake_list",
    "coerce_build_settings",
    "default_build_settings",
    "derived_backend",
    "option_by_key",
    "parent_enabled",
    "primary_keys",
    "settings_to_field_kwargs",
    "supported_backends",
    "unsupported_option_keys",
    "validate_build_settings",
]
