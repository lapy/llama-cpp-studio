"""
Catalog of llama.cpp / ggml CMake build options.

Source of truth aligned with upstream ggml/CMakeLists.txt and CMakeLists.txt
(ggml-org/llama.cpp). Used for defaults, settings coercion, BuildConfig fields,
CMake flag emission, and the build-settings UI catalog API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class BuildOptionDef:
    """One user-facing build setting mapped to a CMake (or special) flag."""

    key: str
    field: str
    kind: str  # bool | str | enum
    default: Any
    label: str
    desc: str
    category: str
    cmake: Optional[str] = None
    requires: Optional[str] = None
    enum_values: Optional[tuple] = None
    special: Optional[str] = None  # handled outside generic set_flag
    # Which engines expose this option in the UI / emit it at build time.
    engines: frozenset = frozenset({"llama_cpp", "ik_llama"})


# R94: the catalog is ENGINE DATA (``llama_cpp/build_catalog_data.py``, carried
# on the spec and exported into the generated definitions). Everything in this
# module derives from the spec block; ``engines`` on an entry says which of the
# two llama-family engines exposes it in the UI.
ENGINE_ID = "llama_cpp"


def normalize_engine_id(engine: Optional[str]) -> Optional[str]:
    if not engine:
        return None
    value = str(engine).strip().lower()
    if value in ("ik_llama", "ik_llama.cpp", "ik"):
        return "ik_llama"
    if value in ("llama_cpp", "llama.cpp", "llama"):
        return "llama_cpp"
    return value


def _catalog() -> Dict[str, Any]:
    from backend.engines.registry import get_engine_spec

    spec = get_engine_spec(ENGINE_ID)
    data = dict(spec.build_options) if spec is not None else {}
    if not data:
        raise RuntimeError("llama.cpp build-option catalog is unavailable")
    return data


def _options() -> tuple:
    options = []
    for raw in _catalog().get("options") or ():
        options.append(
            BuildOptionDef(
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
                engines=frozenset(raw.get("engines") or ()),
            )
        )
    return tuple(options)


def _categories() -> tuple:
    return tuple(dict(item) for item in (_catalog().get("categories") or ()))


BUILD_OPTIONS: tuple[BuildOptionDef, ...] = _options()
CATEGORIES: Sequence[Dict[str, Any]] = _categories()

# ggml/llama enable some backends by host default (Metal on macOS). Studio only
# ships CPU + CUDA, so these are always forced off at build time.
_LLAMA_ENGINES = frozenset({"llama_cpp"})
_IK_ENGINES = frozenset({"ik_llama"})
_BOTH_ENGINES = frozenset({"llama_cpp", "ik_llama"})
UNSUPPORTED_BACKEND_CMAKE: tuple[tuple[str, frozenset], ...] = (
    ("GGML_HIP", _BOTH_ENGINES),
    ("GGML_HIPBLAS", _IK_ENGINES),
    ("GGML_VULKAN", _BOTH_ENGINES),
    ("GGML_METAL", _BOTH_ENGINES),
    ("GGML_SYCL", _BOTH_ENGINES),
    ("GGML_OPENCL", _LLAMA_ENGINES),
    ("GGML_MUSA", _BOTH_ENGINES),
    ("GGML_WEBGPU", _LLAMA_ENGINES),
    ("GGML_RPC", _BOTH_ENGINES),
    ("GGML_ZENDNN", _LLAMA_ENGINES),
)
CATEGORIES: Sequence[Dict[str, Any]] = _categories()
PRIMARY_BACKEND_KEYS = frozenset(_catalog().get("primary_keys") or ())
CATEGORY_REQUIRES: Dict[str, str] = {
    str(category["id"]): str(category["requires"])
    for category in CATEGORIES
    if category.get("requires")
}
CATEGORY_ENGINES: Dict[str, frozenset] = {
    str(category["id"]): frozenset(category.get("engines") or ())
    for category in CATEGORIES
    if category.get("engines")
}
BUILD_TYPE_DEFAULT = str(_catalog().get("build_type_default") or "Release")
BUILD_TYPE_VALUES = tuple(_catalog().get("build_types") or (BUILD_TYPE_DEFAULT,))

OPTION_BY_KEY: Dict[str, BuildOptionDef] = {o.key: o for o in BUILD_OPTIONS}
OPTION_BY_FIELD: Dict[str, BuildOptionDef] = {o.field: o for o in BUILD_OPTIONS}


def default_build_settings() -> Dict[str, Any]:
    """UI/API-shaped defaults (short keys)."""
    out: Dict[str, Any] = {"build_type": BUILD_TYPE_DEFAULT}
    for opt in BUILD_OPTIONS:
        out[opt.key] = opt.default
    return out


def build_config_field_defaults() -> Dict[str, Any]:
    """BuildConfig field-name defaults (enable_* / build_*)."""
    out: Dict[str, Any] = {"build_type": BUILD_TYPE_DEFAULT}
    for opt in BUILD_OPTIONS:
        out[opt.field] = opt.default
    out["env_vars"] = {}
    return out


def coerce_build_settings(settings: Optional[dict]) -> Dict[str, Any]:
    """Normalize a settings dict to the full UI/API shape."""
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

    out = dict(base)

    build_type = _str(settings.get("build_type"), base["build_type"])
    if build_type not in BUILD_TYPE_VALUES:
        build_type = base["build_type"]
    out["build_type"] = build_type

    for opt in BUILD_OPTIONS:
        raw = settings.get(opt.key, base[opt.key])
        if opt.kind == "bool":
            out[opt.key] = _bool(raw)
        elif opt.kind == "enum" and opt.enum_values:
            val = _str(raw, str(opt.default))
            out[opt.key] = val if val in opt.enum_values else opt.default
        else:
            out[opt.key] = _str(raw, str(opt.default) if opt.default is not None else "")

    return out


def stored_config_to_settings(raw: Optional[dict]) -> Dict[str, Any]:
    """Normalize a version-row ``build_config`` (UI keys or BuildConfig fields) to UI keys."""
    if not isinstance(raw, dict):
        return default_build_settings()
    mapped: Dict[str, Any] = {}
    if raw.get("build_type") is not None:
        mapped["build_type"] = raw.get("build_type")
    for opt in BUILD_OPTIONS:
        if opt.key in raw:
            mapped[opt.key] = raw[opt.key]
        elif opt.field in raw:
            mapped[opt.key] = raw[opt.field]
    for key in ("custom_cmake_args", "cflags", "cxxflags", "cuda_architectures"):
        if key in raw and key not in mapped:
            mapped[key] = raw[key]
    return coerce_build_settings(mapped)


def settings_to_field_kwargs(settings: dict) -> Dict[str, Any]:
    """Map coerced settings keys → BuildConfig constructor kwargs."""
    normalized = coerce_build_settings(settings)
    kwargs: Dict[str, Any] = {"build_type": normalized["build_type"]}
    for opt in BUILD_OPTIONS:
        kwargs[opt.field] = normalized[opt.key]
    return kwargs


def catalog_for_ui(engine: Optional[str] = None) -> Dict[str, Any]:
    """Payload for GET build-options: categories + defaults (optionally filtered by engine)."""
    eng = normalize_engine_id(engine)
    by_cat: Dict[str, List[dict]] = {c["id"]: [] for c in CATEGORIES}
    for opt in BUILD_OPTIONS:
        if eng and eng not in opt.engines:
            continue
        entry = {
            "key": opt.key,
            "type": opt.kind,
            "label": opt.label,
            "desc": opt.desc,
            "default": opt.default,
            "requires": opt.requires,
            "cmake": opt.cmake,
            "primary": opt.category != "backends" or opt.key in PRIMARY_BACKEND_KEYS,
            "engines": sorted(opt.engines),
        }
        if opt.enum_values:
            entry["enum_values"] = list(opt.enum_values)
        by_cat.setdefault(opt.category, []).append(entry)

    categories = []
    for cat in CATEGORIES:
        cat_engines = CATEGORY_ENGINES.get(cat["id"])
        if eng and cat_engines and eng not in cat_engines:
            continue
        options = by_cat.get(cat["id"]) or []
        if not options:
            continue
        categories.append(
            {
                "id": cat["id"],
                "label": cat["label"],
                "requires": CATEGORY_REQUIRES.get(cat["id"]),
                "collapsed": bool(cat.get("collapsed", True)),
                "options": options,
            }
        )

    return {
        "engine": eng,
        "categories": categories,
        "defaults": default_build_settings(),
        "build_types": list(BUILD_TYPE_VALUES),
    }


def append_generic_cmake_flags(
    cmake_args: List[str],
    build_config: Any,
    *,
    set_flag,
    engine: Optional[str] = None,
) -> None:
    """
    Emit -D flags for options applicable to the target engine.
    Engine-specific flag names (CUDA_GRAPHS vs CUDA_USE_GRAPHS, AVX_VNNI vs
    AVXVNNI) are handled via special=. Unsupported backends are forced off.
    """
    eng = normalize_engine_id(engine) or "llama_cpp"
    for flag, engines in UNSUPPORTED_BACKEND_CMAKE:
        if eng in engines:
            set_flag(flag, False)
    skip_emit = {
        "cuda",
        "blas",
        "cuda_fa_all",
        "cuda_arch",
        "blas_vendor",
        "custom_cmake",
        "cflags",
        "cxxflags",
    }
    for opt in BUILD_OPTIONS:
        if eng not in opt.engines:
            continue

        if opt.special == "cuda_graphs":
            flag = "GGML_CUDA_USE_GRAPHS" if eng == "ik_llama" else "GGML_CUDA_GRAPHS"
            parent_on = bool(getattr(build_config, "enable_cuda", False))
            value = bool(getattr(build_config, opt.field, opt.default))
            set_flag(flag, value if parent_on else False)
            continue
        if opt.special == "avx_vnni":
            flag = "GGML_AVXVNNI" if eng == "ik_llama" else "GGML_AVX_VNNI"
            set_flag(flag, bool(getattr(build_config, opt.field, False)))
            continue

        if not opt.cmake or opt.special in skip_emit:
            continue
        value = getattr(build_config, opt.field, opt.default)
        if opt.kind == "bool":
            if opt.requires:
                parent = OPTION_BY_KEY.get(opt.requires)
                if parent:
                    parent_val = getattr(build_config, parent.field, False)
                    if not parent_val:
                        set_flag(opt.cmake, False)
                        continue
            set_flag(opt.cmake, bool(value))
        else:
            if opt.requires:
                parent = OPTION_BY_KEY.get(opt.requires)
                if parent and not getattr(build_config, parent.field, False):
                    continue
            text = str(value or "").strip()
            if text:
                cmake_args.append(f"-D{opt.cmake}={text}")
