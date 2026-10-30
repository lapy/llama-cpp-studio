"""Lifecycle and datastore adapter for Studio's native proxy backend."""

from __future__ import annotations

import asyncio
import os
import threading
import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

import uvicorn

from backend.data_store import get_store
from backend.logging_config import get_logger
from backend.inference_auth import (
    authorize_admin_headers,
    authorize_headers,
    authorize_model_headers,
    verify_token,
    request_identity,
)
from backend.proxy.launch_spec import compile_model_runtime, manifest_document
from backend.proxy.manifests import LaunchManifestStore
from backend.proxy.studio.dataplane import DEFAULT_MAX_BODY_BYTES, ResolvedModel, StudioDataPlane
from backend.proxy.studio.registry import ProcessRegistry

logger = get_logger(__name__)
_instance: Optional["StudioProxyManager"] = None


def configured_proxy_backend() -> str:
    settings = get_store().get_settings() or {}
    proxy = settings.get("proxy")
    value = proxy.get("backend") if isinstance(proxy, dict) else settings.get("proxy_backend")
    value = str(value or "llama_swap").strip().lower()
    return value if value in {"llama_swap", "studio"} else "llama_swap"


def get_studio_proxy_manager() -> "StudioProxyManager":
    global _instance
    if _instance is None:
        from backend.proxy.llama_swap.client import get_proxy_port

        _instance = StudioProxyManager(get_proxy_port())
    return _instance


RUNTIME_DEFAULTS = {
    "idle_ttl_seconds": None,
    "max_body_bytes": None,
    "max_loaded_models": None,
    "global_concurrency": None,
    "per_model_concurrency": None,
    "queue_timeout_seconds": 30,
    "request_timeout_seconds": None,
    "readiness_timeout_seconds": 2700,
    "drain_timeout_seconds": 30,
    "max_queue_size": 100,
    "stream_idle_timeout_seconds": None,
    "pool_timeout_seconds": 10,
    "max_connections": 200,
    "max_keepalive_connections": 40,
    "history_retention_days": 7,
    "history_max_rows": 100000,
    "default_output_tokens": 4096,
}


@dataclass(frozen=True)
class RoutingSnapshot:
    aliases: object
    default_model: str
    revision: str

    @classmethod
    def build(cls, settings):
        aliases = {
            str(alias): tuple(
                str(target).strip()
                for target in (targets if isinstance(targets, list) else [targets])
                if str(target).strip()
            )
            for alias, targets in (settings.get("routing_aliases") or {}).items()
        }
        default = str(settings.get("default_model") or "").strip()
        revision = hashlib.sha256(
            json.dumps({"aliases": aliases, "default": default}, sort_keys=True).encode()
        ).hexdigest()[:16]
        return cls(MappingProxyType(aliases), default, revision)


class StudioProxyManager:
    def __init__(self, port: int) -> None:
        self.port = int(port)
        # llama-swap managers expose proxy_port; control-plane code reads it.
        self.proxy_port = int(port)
        self.manifests = LaunchManifestStore()
        self._server: Optional[uvicorn.Server] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._route_cursor: dict[str, int] = {}
        self._route_lock = threading.Lock()
        self._catalog_lock = threading.RLock()
        self._compiled_cache = None
        self._build_runtime()

    def _build_runtime(self) -> None:
        """(Re)create the registry and dataplane.

        stop_proxy() closes both, so a restart after a stop must not reuse
        them: the singleton would serve an already-closed httpx client into a
        registry that refuses new launches.
        """
        proxy_settings = (get_store().get_settings() or {}).get("proxy")
        settings = proxy_settings if isinstance(proxy_settings, dict) else {}
        self._applied_settings = {
            name: settings.get(name, default) for name, default in RUNTIME_DEFAULTS.items()
        }
        self.apply_routing(settings)
        ttl = settings.get("idle_ttl_seconds")
        try:
            idle_ttl = float(ttl) if ttl is not None else None
        except (TypeError, ValueError):
            idle_ttl = None
        try:
            max_body = int(settings.get("max_body_bytes") or DEFAULT_MAX_BODY_BYTES)
        except (TypeError, ValueError):
            max_body = DEFAULT_MAX_BODY_BYTES
        if max_body <= 0:
            max_body = DEFAULT_MAX_BODY_BYTES
        self.registry = ProcessRegistry(
            self._launch,
            idle_ttl=idle_ttl if idle_ttl and idle_ttl > 0 else None,
            readiness_timeout=_positive_float(settings, "readiness_timeout_seconds", 2700),
            drain_timeout=_positive_float(settings, "drain_timeout_seconds", 30),
            max_loaded_models=_positive_int(settings, "max_loaded_models"),
            per_model_concurrency=_positive_int(settings, "per_model_concurrency"),
            queue_timeout=_positive_float(settings, "queue_timeout_seconds", 30),
        )
        self.registry.set_launch_observer(self._receipt)
        from backend.proxy.studio.accounting import get_accounting

        self.dataplane = StudioDataPlane(
            self.registry,
            self._resolve,
            accounting=get_accounting(),
            identity_resolver=request_identity,
            resolve_in_executor=True,
            max_queue_size=_positive_int(settings, "max_queue_size") or 100,
            stream_idle_timeout=_positive_float(settings, "stream_idle_timeout_seconds", None),
            pool_timeout=_positive_float(settings, "pool_timeout_seconds", 10),
            max_connections=_positive_int(settings, "max_connections") or 200,
            max_keepalive_connections=_positive_int(settings, "max_keepalive_connections") or 40,
            history_retention_days=_positive_int(settings, "history_retention_days") or 7,
            history_max_rows=_positive_int(settings, "history_max_rows") or 100000,
            default_output_tokens=_positive_int(settings, "default_output_tokens") or 4096,
            authorize=authorize_headers,
            authorize_model=authorize_model_headers,
            authorize_admin=authorize_admin_headers,
            verify_token=verify_token,
            catalog=self._catalog,
            max_body_bytes=max_body,
            global_concurrency=_positive_int(settings, "global_concurrency"),
            queue_timeout=_positive_float(settings, "queue_timeout_seconds", 30),
            request_timeout=_positive_float(settings, "request_timeout_seconds", None),
        )

    def apply_routing(self, settings):
        snapshot = RoutingSnapshot.build(settings)
        with self._route_lock:
            if not hasattr(self, "_routing") or self._routing.revision != snapshot.revision:
                self._routing = snapshot
                self._route_cursor.clear()
        return snapshot.revision

    def runtime_restart_required(self, settings):
        return settings.get("proxy_port", self.port) != self.port or any(
            settings.get(name, default) != self._applied_settings.get(name, default)
            for name, default in RUNTIME_DEFAULTS.items()
        )

    @property
    def running(self) -> bool:
        return bool(self._task and not self._task.done() and self._server and self._server.started)

    async def restart_proxy(self) -> None:
        await self.stop_proxy()
        # Re-read the port: a settings change to proxy_port must rebind on
        # restart rather than keep the singleton's original socket.
        from backend.proxy.llama_swap.client import get_proxy_port

        self.port = self.proxy_port = get_proxy_port()
        self._build_runtime()
        await self.start_proxy()

    async def start_proxy(self) -> None:
        if self._task and not self._task.done():
            return
        if getattr(self.registry, "_closed", False) or self.dataplane.client.is_closed:
            self._build_runtime()
        saved = get_store().get_settings() or {}
        current = {**(saved.get("proxy") or {}), "proxy_port": saved.get("proxy_port", self.port)}
        if self.runtime_restart_required(current):
            await self.dataplane.close()
            await self.registry.close()
            self.port = self.proxy_port = current["proxy_port"]
            self._build_runtime()
        from backend.access_policy import proxy_listen_host

        config = uvicorn.Config(
            self.dataplane,
            host=proxy_listen_host(),
            port=self.port,
            log_level="info",
            lifespan="off",
            access_log=False,
            timeout_graceful_shutdown=15,
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        try:
            for _ in range(100):
                if self._server.started:
                    logger.info("Studio native proxy listening on port %s", self.port)
                    return
                if self._task.done():
                    await self._task
                    break
                await asyncio.sleep(0.02)
        except BaseException:
            self._server.should_exit = True
            if not self._task.done():
                self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._server = None
            self._task = None
            raise
        self._server.should_exit = True
        if not self._task.done():
            self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._server = None
        self._task = None
        raise RuntimeError("Studio native proxy did not start")

    async def stop_proxy(self) -> None:
        # Drain in reverse start order: stop accepting connections and let
        # Uvicorn finish its in-flight requests first, only then close the
        # upstream client and tree-kill engine process groups. The opposite
        # order races new requests into a half-closed dataplane.
        if self._server:
            self._server.should_exit = True
        if self._task:
            try:
                await self._task
            except Exception:
                logger.exception("Studio native proxy exited with an error")
        self._server = None
        self._task = None
        try:
            await self.dataplane.close()
        finally:
            await self.registry.close()

    def _compiled_models(self):
        store = get_store()

        def compile_rows():
            rows = []
            for model in store.list_models():
                try:
                    rows.append((model, compile_model_runtime(model)))
                except Exception:
                    continue
            return rows

        if not hasattr(store, "document_revision"):
            return compile_rows()

        def revision():
            return tuple(
                store.document_revision(name)
                for name in ("models.yaml", "engines.yaml", "settings.yaml", "templates.yaml")
            )

        key = revision()
        with self._catalog_lock:
            if self._compiled_cache is not None and self._compiled_cache[0] == key:
                return self._compiled_cache[1]
            with store.exclusive_documents():
                key = revision()
                rows = compile_rows()
            self._compiled_cache = (key, rows)
            return rows

    async def _launch(self, model_id: str):
        model = await asyncio.to_thread(self._model, model_id)
        if model is None:
            raise ValueError(f"unknown model: {model_id}")
        compiled = await asyncio.to_thread(compile_model_runtime, model)
        document = manifest_document(compiled.launch, compiled.revision)
        self.manifests.stage(compiled.proxy.model_id, document)
        generation = self.manifests.generation_dir(compiled.proxy.model_id, compiled.revision)
        return compiled.launch, compiled.revision, os.path.join(generation, "artifacts")

    def _resolve(self, requested: str) -> Optional[ResolvedModel]:
        routing = getattr(self, "_routing", None)
        if routing is None:
            routing = RoutingSnapshot.build((get_store().get_settings() or {}).get("proxy") or {})
        targets = routing.aliases.get(requested)
        if targets:
            with self._route_lock:
                key = (routing.revision, requested)
                cursor = self._route_cursor.get(key, 0)
                self._route_cursor[key] = cursor + 1
            requested = targets[cursor % len(targets)]
        if not requested:
            requested = routing.default_model
        candidates = []
        for model, compiled in self._compiled_models():
            names = {compiled.proxy.model_id, compiled.proxy.catalog_id, *compiled.proxy.aliases}
            if requested and requested in names:
                return ResolvedModel(
                    model_id=compiled.proxy.model_id,
                    use_model_name=compiled.proxy.use_model_name,
                    filters=compiled.proxy.filters,
                    routing_revision=routing.revision,
                    engine_id=compiled.proxy.engine_id,
                )
            candidates.append(compiled.proxy)
        if not requested and len(candidates) == 1:
            proxy = candidates[0]
            return ResolvedModel(
                model_id=proxy.model_id,
                use_model_name=proxy.use_model_name,
                filters=proxy.filters,
                routing_revision=routing.revision,
                engine_id=proxy.engine_id,
            )
        # An unqualified request against several registered models keeps the
        # single running model, matching llama-swap's default dispatch.
        if not requested:
            running = self.registry.snapshot()
            if len(running) == 1:
                model_id = str(running[0].get("model") or "")
                proxy = next((row for row in candidates if row.model_id == model_id), None)
                if proxy:
                    return ResolvedModel(
                        model_id=proxy.model_id,
                        use_model_name=proxy.use_model_name,
                        filters=proxy.filters,
                        routing_revision=routing.revision,
                        engine_id=proxy.engine_id,
                    )
        return None

    def _catalog(self) -> list[dict]:
        return [
            {
                "id": compiled.proxy.model_id,
                "studio_model_id": model["id"],
                "engine": compiled.proxy.engine_id,
                "aliases": list(compiled.proxy.aliases),
            }
            for model, compiled in self._compiled_models()
        ]

    def _receipt(self, spec, revision: str, pid: int) -> None:
        """Persist the launch receipt for a natively-started process.

        Apply journals and startup reconciliation treat a live pid whose
        start-ticks match a receipt as the verified running revision.
        """
        import time
        import uuid

        launch_id = str(uuid.uuid4().hex)
        pending = os.path.join(self.manifests.model_dir(spec.model_id), "pending-launch.json")
        try:
            import json

            with open(pending, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if (
                isinstance(payload, dict)
                and str(payload.get("revision") or "") == revision
                and str(payload.get("launch_id") or "").strip()
            ):
                launch_id = str(payload["launch_id"]).strip()
        except (OSError, ValueError):
            pass
        self.manifests.write_receipt(
            spec.model_id,
            revision=revision,
            launch_id=launch_id,
            pid=pid,
            start_ticks=_start_ticks(pid),
        )

    def _model(self, model_id: str) -> Optional[dict[str, Any]]:
        return next(
            (
                model
                for model, compiled in self._compiled_models()
                if compiled.proxy.model_id == model_id
            ),
            None,
        )


def _positive_int(settings: dict, key: str) -> Optional[int]:
    try:
        value = int(settings.get(key))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _positive_float(settings: dict, key: str, default: Optional[float]) -> Optional[float]:
    try:
        value = float(settings.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _start_ticks(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
            stat = handle.read()
    except OSError:
        return ""
    end = stat.rfind(")")
    if end < 0:
        return ""
    fields = stat[end + 2 :].split()
    if len(fields) < 20:
        return ""
    return fields[19]
