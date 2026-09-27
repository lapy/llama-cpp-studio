"""Engine-id to installer map. Imported by app startup, not by the registry."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from backend.engines.protocol import EngineInstaller


def _lmdeploy() -> EngineInstaller:
    from backend.engines.lmdeploy import get_lmdeploy_manager

    return get_lmdeploy_manager()


def _sglang() -> EngineInstaller:
    from backend.engines.sglang import get_sglang_manager

    return get_sglang_manager("sglang")


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
    "lmdeploy": _lmdeploy,
    "sglang": _sglang,
    "sglang_v100": _sglang_v100,
    "vllm": _vllm,
    "1cat_vllm": _onecat,
}


def get_engine_installer(engine_id: str) -> EngineInstaller:
    factory = _INSTALLERS.get(str(engine_id or ""))
    if factory is None:
        raise KeyError(f"No installer registered for engine {engine_id!r}")
    return factory()


def get_engine_install_root(engine_id: str) -> Optional[str]:
    """Return the on-disk root for a registered installer, if any."""
    try:
        installer = get_engine_installer(engine_id)
    except KeyError:
        return None
    return installer.install_root


def discover_python_engine_roots() -> Dict[str, str]:
    roots: Dict[str, str] = {}
    for engine_id in _INSTALLERS:
        root = get_engine_install_root(engine_id)
        if root:
            roots[engine_id] = root
    return roots
