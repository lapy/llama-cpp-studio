"""Lifecycle and datastore adapter for Studio's native proxy backend."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

import uvicorn

from backend.data_store import get_store
from backend.logging_config import get_logger
from backend.inference_auth import (
    authorize_admin_headers,
    authorize_headers,
    verify_token,
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


class StudioProxyManager:
    def __init__(self, port: int) -> None:
        self.port = int(port)
        # llama-swap managers expose proxy_port; control-plane code reads it.
        self.proxy_port = int(port)
        self.manifests = LaunchManifestStore()
        self._server: Optional[uvicorn.Server] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._build_runtime()

    def _build_runtime(self) -> None:
        """(Re)create the registry and dataplane.

        stop_proxy() closes both, so a restart after a stop must not reuse
        them: the singleton would serve an already-closed httpx client into a
        registry that refuses new launches.
        """
        proxy_settings = (get_store().get_settings() or {}).get("proxy")
        settings = proxy_settings if isinstance(proxy_settings, dict) else {}
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
            self._launch, idle_ttl=idle_ttl if idle_ttl and idle_ttl > 0 else None
        )
        self.registry.set_launch_observer(self._receipt)
        self.dataplane = StudioDataPlane(
            self.registry,
            self._resolve,
            authorize=authorize_headers,
            authorize_admin=authorize_admin_headers,
            verify_token=verify_token,
            catalog=self._catalog,
            max_body_bytes=max_body,
        )

    async def start_proxy(self) -> None:
        if self._task and not self._task.done():
            return
        if getattr(self.registry, "_closed", False) or self.dataplane.client.is_closed:
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
        await self.dataplane.close()
        await self.registry.close()

    async def _launch(self, model_id: str):
        model = self._model(model_id)
        if model is None:
            raise ValueError(f"unknown model: {model_id}")
        compiled = compile_model_runtime(model)
        document = manifest_document(compiled.launch, compiled.revision)
        self.manifests.stage(compiled.proxy.model_id, document)
        generation = self.manifests.generation_dir(compiled.proxy.model_id, compiled.revision)
        return compiled.launch, compiled.revision, os.path.join(generation, "artifacts")

    def _resolve(self, requested: str) -> Optional[ResolvedModel]:
        candidates = []
        for model in get_store().list_models():
            try:
                compiled = compile_model_runtime(model)
            except Exception:
                continue
            names = {compiled.proxy.model_id, compiled.proxy.catalog_id, *compiled.proxy.aliases}
            if requested and requested in names:
                return ResolvedModel(
                    model_id=compiled.proxy.model_id,
                    use_model_name=compiled.proxy.use_model_name,
                    filters=compiled.proxy.filters,
                )
            candidates.append(compiled.proxy)
        if not requested and len(candidates) == 1:
            proxy = candidates[0]
            return ResolvedModel(
                model_id=proxy.model_id,
                use_model_name=proxy.use_model_name,
                filters=proxy.filters,
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
                    )
        return None

    def _catalog(self) -> list[dict]:
        rows = []
        for model in get_store().list_models():
            try:
                compiled = compile_model_runtime(model)
            except Exception:
                continue
            rows.append(
                {
                    "id": compiled.proxy.model_id,
                    "engine": compiled.proxy.engine_id,
                    "aliases": list(compiled.proxy.aliases),
                }
            )
        return rows

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
        for model in get_store().list_models():
            try:
                if compile_model_runtime(model).proxy.model_id == model_id:
                    return model
            except Exception:
                continue
        return None


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
