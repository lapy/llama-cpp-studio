"""Installer lookup by engine.

The explicit map below remains for exporter-generated engines whose installer
classes own extra mechanisms. Any engine whose definition declares a
``lifecycle`` and has no explicit factory is served by the shared
definition-driven installer, so a converted engine needs no entry here.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from backend.engines.protocol import EngineInstaller


def _sglang() -> EngineInstaller:
    # Phase-3: upstream SGLang runs on the definition-driven installer; the
    # V100 fork keeps the specialised class (Phase 5 recipes).
    from backend.engines.sglang import get_sglang_upstream_manager

    return get_sglang_upstream_manager()


def _sglang_v100() -> EngineInstaller:
    from backend.engines.sglang import get_sglang_manager

    return get_sglang_manager("sglang_v100")


def _vllm() -> EngineInstaller:
    from backend.engines.vllm import get_vllm_manager

    return get_vllm_manager()


def _onecat() -> EngineInstaller:
    from backend.engines.vllm import get_onecat_vllm_manager

    return get_onecat_vllm_manager()


_INSTALLERS: Dict[str, Callable[[], EngineInstaller]] = {
    "sglang": _sglang,
    "sglang_v100": _sglang_v100,
    "vllm": _vllm,
    "1cat_vllm": _onecat,
}


def get_engine_installer(engine_id: str) -> EngineInstaller:
    factory = _INSTALLERS.get(str(engine_id or ""))
    if factory is not None:
        return factory()
    installer = _definition_installer(engine_id)
    if installer is not None:
        return installer
    raise KeyError(f"No installer registered for engine {engine_id!r}")


def _definition_installer(engine_id: str) -> Optional[EngineInstaller]:
    """Shared installer for a definition-owned engine, or None.

    The definition is the selector: declaring a ``lifecycle`` block means the
    engine's install flows are executable from YAML. No engine-id list.
    """
    from backend.engines.definitions.loader import load_default_engine_definitions

    definition = load_default_engine_definitions().definitions.get(str(engine_id or ""))
    if definition is None or not definition.lifecycle:
        return None
    from backend.engines.python.generic import get_definition_installer

    return get_definition_installer(definition.id)


def get_engine_install_root(engine_id: str) -> Optional[str]:
    """Return the on-disk root for a registered installer, if any."""
    try:
        installer = get_engine_installer(engine_id)
    except KeyError:
        return None
    return installer.install_root


def discover_python_engine_roots() -> Dict[str, str]:
    from backend.engines.registry import engine_ids_with_install_kind

    roots: Dict[str, str] = {}
    for engine_id in engine_ids_with_install_kind("python_venv"):
        root = get_engine_install_root(engine_id)
        if root:
            roots[engine_id] = root
    return roots


# --- Native (install-dir) hosts ------------------------------------------------
# Engines migrated to the definition-driven ``GenericNativeInstaller`` register
# a factory here. The shared native route uses it for lifecycle steps whose
# semantics the host owns (audio.cpp's bundle removal, the llama family's
# source/removal recipes); engines without one keep the legacy handler pack
# until their own gated cutover.


def _audio_cpp_native_host() -> object:
    from backend.engines.audio_cpp.installer import AudioCppInstaller
    from backend.engines.audio_cpp.manager import get_audio_cpp_manager

    return AudioCppInstaller(install_root=get_audio_cpp_manager().builds_dir)


def _llama_cpp_native_host() -> object:
    from backend.engines.python.cmake_source import CmakeSourceInstaller

    return CmakeSourceInstaller("llama_cpp")


def _ik_llama_native_host() -> object:
    from backend.engines.python.cmake_source import CmakeSourceInstaller

    return CmakeSourceInstaller("ik_llama")


_NATIVE_HOSTS: Dict[str, Callable[[], object]] = {
    "audio_cpp": _audio_cpp_native_host,
    "llama_cpp": _llama_cpp_native_host,
    "ik_llama": _ik_llama_native_host,
}


def get_native_installer(engine_id: str):
    """Return the engine's definition-driven native host, or None."""
    factory = _NATIVE_HOSTS.get(str(engine_id or ""))
    return factory() if factory is not None else None
