"""Native runtime control shared by model and engine routes."""

from backend.proxy.studio.manager import get_studio_proxy_manager


def get_proxy_controller():
    return get_studio_proxy_manager()


def mark_runtime_changed():
    from backend.proxy.studio import manager as runtime

    instance = runtime._instance
    if instance is not None:
        with instance._catalog_lock:
            instance._compiled_cache = None
