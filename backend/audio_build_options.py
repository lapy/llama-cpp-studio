"""
Catalog of audio.cpp CMake build options.

Aligned with upstream 0xShug0/audio.cpp CMakeLists.txt (ENGINE_* / AUDIOCPP_*).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


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


CATEGORIES: Sequence[Dict[str, Any]] = (
    {"id": "backends", "label": "GPU & compute backends", "collapsed": False},
    {"id": "cuda", "label": "CUDA / HIP options", "collapsed": True},
    {"id": "artifacts", "label": "Build artifacts", "collapsed": True},
    {"id": "cpu", "label": "CPU options", "collapsed": True},
    {"id": "models", "label": "Model set", "collapsed": True},
    {"id": "advanced", "label": "Advanced", "collapsed": True},
)

CATEGORY_REQUIRES: Dict[str, str] = {
    # Show CUDA graphs when either CUDA or HIP is enabled
    "cuda": "cuda_or_hip",
}

PRIMARY_BACKEND_KEYS = frozenset({"cuda", "hip", "vulkan", "metal"})


def _b(
    key: str,
    field: str,
    default: bool,
    label: str,
    desc: str,
    category: str,
    cmake: Optional[str] = None,
    requires: Optional[str] = None,
    special: Optional[str] = None,
) -> AudioBuildOptionDef:
    return AudioBuildOptionDef(
        key=key,
        field=field,
        kind="bool",
        default=default,
        label=label,
        desc=desc,
        category=category,
        cmake=cmake,
        requires=requires,
        special=special,
    )


def _s(
    key: str,
    field: str,
    default: str,
    label: str,
    desc: str,
    category: str,
    cmake: Optional[str] = None,
    requires: Optional[str] = None,
    special: Optional[str] = None,
    enum_values: Optional[tuple] = None,
) -> AudioBuildOptionDef:
    return AudioBuildOptionDef(
        key=key,
        field=field,
        kind="enum" if enum_values else "str",
        default=default,
        label=label,
        desc=desc,
        category=category,
        cmake=cmake,
        requires=requires,
        special=special,
        enum_values=enum_values,
    )


def _i(
    key: str,
    field: str,
    default: int,
    label: str,
    desc: str,
    category: str,
    special: Optional[str] = None,
) -> AudioBuildOptionDef:
    return AudioBuildOptionDef(
        key=key,
        field=field,
        kind="int",
        default=default,
        label=label,
        desc=desc,
        category=category,
        special=special,
    )


BUILD_TYPE_DEFAULT = "RelWithDebInfo"
BUILD_TYPE_VALUES = ("Debug", "Release", "RelWithDebInfo", "MinSizeRel")

# fmt: off
BUILD_OPTIONS: tuple[AudioBuildOptionDef, ...] = (
    # Backends (mutually exclusive CUDA↔HIP; others may combine)
    _b("cuda", "cuda", False, "CUDA", "ENGINE_ENABLE_CUDA — NVIDIA GPU (optimized path)", "backends", "ENGINE_ENABLE_CUDA", special="backend"),
    _b("hip", "hip", False, "HIP / ROCm", "ENGINE_ENABLE_HIP — AMD GPU (exclusive with CUDA)", "backends", "ENGINE_ENABLE_HIP", special="backend"),
    _b("vulkan", "vulkan", False, "Vulkan", "ENGINE_ENABLE_VULKAN", "backends", "ENGINE_ENABLE_VULKAN", special="backend"),
    _b("metal", "metal", False, "Metal", "ENGINE_ENABLE_METAL — Apple GPU", "backends", "ENGINE_ENABLE_METAL", special="backend"),

    # CUDA / HIP shared
    _b("cuda_graphs", "cuda_graphs", True, "CUDA / HIP graphs", "ENGINE_ENABLE_CUDA_GRAPHS (also drives HIP graphs)", "cuda", "ENGINE_ENABLE_CUDA_GRAPHS", requires="cuda_or_hip"),

    # Artifacts
    _b("build_tests", "build_tests", False, "Tests", "ENGINE_BUILD_TESTS", "artifacts", "ENGINE_BUILD_TESTS"),
    _b("build_examples", "build_examples", False, "Examples", "ENGINE_BUILD_EXAMPLES", "artifacts", "ENGINE_BUILD_EXAMPLES"),
    _b("build_warmbench", "build_warmbench", False, "Warmbench", "ENGINE_BUILD_WARMBENCH", "artifacts", "ENGINE_BUILD_WARMBENCH"),
    _b("deployment_build", "deployment_build", False, "Deployment build", "AUDIOCPP_DEPLOYMENT_BUILD — embed model specs in binaries", "artifacts", "AUDIOCPP_DEPLOYMENT_BUILD"),
    _b("native_model_manager", "native_model_manager", True, "Native model manager", "AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER — WebUI downloads and --ui-management", "artifacts", "AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER"),

    # CPU
    _b("native_cpu", "native_cpu", True, "Native CPU", "ENGINE_ENABLE_NATIVE_CPU — -march=native", "cpu", "ENGINE_ENABLE_NATIVE_CPU"),
    _b("openmp", "openmp", True, "OpenMP", "ENGINE_ENABLE_OPENMP", "cpu", "ENGINE_ENABLE_OPENMP"),
    _b("llamafile", "llamafile", True, "llamafile", "ENGINE_ENABLE_LLAMAFILE — SGEMM", "cpu", "ENGINE_ENABLE_LLAMAFILE"),
    _b("cpu_all_variants", "cpu_all_variants", False, "CPU all variants", "ENGINE_ENABLE_CPU_ALL_VARIANTS", "cpu", "ENGINE_ENABLE_CPU_ALL_VARIANTS"),

    # Model set
    _s("model_set", "model_set", "full", "Model set", "AUDIOCPP_MODEL_SET", "models", "AUDIOCPP_MODEL_SET", enum_values=("full", "core", "custom")),
    _s("models", "models", "", "Custom models", "AUDIOCPP_MODELS — comma-separated when model set is custom", "models", "AUDIOCPP_MODELS", requires="model_set_custom"),

    # Advanced
    _i("jobs", "jobs", 0, "Parallel jobs", "0 = automatic (nproc)", "advanced", special="jobs"),
    _s("custom_cmake_args", "custom_cmake_args", "", "Custom CMake args", "Extra args appended to cmake (shlex)", "advanced", special="custom_cmake"),
    _s("cflags", "cflags", "", "CFLAGS", "Passed via CFLAGS env", "advanced", special="cflags"),
    _s("cxxflags", "cxxflags", "", "CXXFLAGS", "Passed via CXXFLAGS env", "advanced", special="cxxflags"),
)
# fmt: on

OPTION_BY_KEY: Dict[str, AudioBuildOptionDef] = {o.key: o for o in BUILD_OPTIONS}


def default_build_settings() -> Dict[str, Any]:
    out: Dict[str, Any] = {"build_type": BUILD_TYPE_DEFAULT}
    for opt in BUILD_OPTIONS:
        out[opt.key] = opt.default
    # Legacy single-backend key kept for older clients / version metadata
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
    if build_type not in BUILD_TYPE_VALUES:
        build_type = base["build_type"]
    out["build_type"] = build_type

    for opt in BUILD_OPTIONS:
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

    # Legacy backend= → toggles
    legacy = _str(settings.get("backend"), "").lower()
    if legacy and not any(out.get(k) for k in ("cuda", "hip", "vulkan", "metal")):
        if legacy == "cuda":
            out["cuda"] = True
        elif legacy == "hip":
            out["hip"] = True
        elif legacy == "vulkan":
            out["vulkan"] = True
        elif legacy == "metal":
            out["metal"] = True

    # CUDA and HIP are mutually exclusive upstream
    if out.get("cuda") and out.get("hip"):
        out["hip"] = False

    out["backend"] = derived_backend(out)
    return out


def derived_backend(settings: dict) -> str:
    if settings.get("cuda"):
        return "cuda"
    if settings.get("hip"):
        return "hip"
    if settings.get("vulkan"):
        return "vulkan"
    if settings.get("metal"):
        return "metal"
    return "cpu"


def settings_to_field_kwargs(settings: dict) -> Dict[str, Any]:
    normalized = coerce_build_settings(settings)
    kwargs: Dict[str, Any] = {"build_type": normalized["build_type"], "backend": normalized["backend"]}
    for opt in BUILD_OPTIONS:
        kwargs[opt.field] = normalized[opt.key]
    return kwargs


def catalog_for_ui() -> Dict[str, Any]:
    by_cat: Dict[str, List[dict]] = {c["id"]: [] for c in CATEGORIES}
    for opt in BUILD_OPTIONS:
        entry = {
            "key": opt.key,
            "type": opt.kind,
            "label": opt.label,
            "desc": opt.desc,
            "default": opt.default,
            "requires": opt.requires,
            "cmake": opt.cmake,
            "primary": opt.category != "backends" or opt.key in PRIMARY_BACKEND_KEYS,
        }
        if opt.enum_values:
            entry["enum_values"] = list(opt.enum_values)
        by_cat.setdefault(opt.category, []).append(entry)

    categories = []
    for cat in CATEGORIES:
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
        "engine": "audio_cpp",
        "categories": categories,
        "defaults": default_build_settings(),
        "build_types": list(BUILD_TYPE_VALUES),
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
        return bool(get("cuda") or get("hip"))
    if requires == "model_set_custom":
        return str(get("model_set") or "") == "custom"
    return bool(get(requires))
