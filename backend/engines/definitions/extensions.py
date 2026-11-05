"""Declared extension surfaces, enforced against the router that serves them.

An engine definition declares the typed endpoints it serves beyond the generic
lifecycle verbs (schema-v2 ``extensions``). This module is the single consumer
of that declaration: the engine's router calls :func:`assert_extensions_served`
at import time, so a drifted declaration refuses startup/routing instead of
silently describing a surface that no longer exists. The same function backs the
gate test, so production and tests share one implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Tuple


def declared_extensions(engine_id: str) -> Tuple[Mapping[str, Any], ...]:
    """Extension surfaces the engine's spec declares (empty when none)."""
    from backend.engines.registry import get_engine_spec

    spec = get_engine_spec(engine_id)
    if spec is None:
        return ()
    return tuple(dict(item) for item in spec.extensions)


def served_surfaces(routes: Iterable[Any]) -> Dict[str, set]:
    """Map router-relative path -> HTTP methods actually mounted."""
    served: Dict[str, set] = {}
    for route in routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", None) or ()
        if path:
            served.setdefault(path, set()).update(methods)
    return served


def assert_extensions_served(engine_id: str, routes: Iterable[Any]) -> None:
    """Raise when a declared extension is not served by ``routes``."""
    served = served_surfaces(routes)
    missing = []
    for extension in declared_extensions(engine_id):
        path = str(extension["path"])
        methods = set(extension.get("methods") or ["GET"])
        mounted = served.get(path)
        if mounted is None:
            missing.append(f"{path} (no route)")
        elif not methods <= mounted:
            missing.append(f"{path} (missing {sorted(methods - mounted)})")
    if missing:
        raise RuntimeError(f"{engine_id}: declared extension surfaces are not served: {missing}")
