"""Installer lookup by engine.

Every installer is selected from the engine definition: declaring a
``lifecycle`` block is the selector. No engine-id factory map lives here.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from backend.engines.protocol import EngineInstaller


def get_engine_installer(engine_id: str) -> EngineInstaller:
    """Resolve the installer solely from the engine definition."""
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


def _cmake_source_host(engine_id: str) -> object:
    from backend.engines.python.cmake_source import CmakeSourceInstaller

    return CmakeSourceInstaller(engine_id)


def _audio_cpp_host(engine_id: str) -> object:
    from backend.engines.audio_cpp.installer import AudioCppInstaller
    from backend.engines.audio_cpp.manager import get_audio_cpp_manager

    return AudioCppInstaller(install_root=get_audio_cpp_manager().builds_dir)


# Host mechanisms are selected by the definition's `native_host`, not by
# engine id. Adding a new engine that uses an existing host needs YAML only.
_NATIVE_HOST_FACTORIES: Dict[str, Callable[[str], object]] = {
    "cmake_source": _cmake_source_host,
    "audio_cpp": _audio_cpp_host,
}


def get_native_installer(engine_id: str):
    """Return the host named by the engine definition, or None."""
    from backend.engines.definitions.loader import load_default_engine_definitions

    definition = load_default_engine_definitions().definitions.get(str(engine_id or ""))
    host = str(getattr(definition, "native_host", "") or "").strip()
    factory = _NATIVE_HOST_FACTORIES.get(host)
    return factory(engine_id) if factory is not None else None
