from fastapi.responses import StreamingResponse
from backend.operations.progress import get_progress_manager
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.requests import ClientDisconnect
from contextlib import asynccontextmanager
import asyncio

from backend.data_store import get_store
from backend.routes import (
    audio_cpp_versions,
    audio_openai_proxy,
    config_backup,
    engines,
    model_catalog,
    model_config_templates,
    models,
    llama_versions,
    status,
    gpu_info,
    lmdeploy_versions,
    onecat_vllm_versions,
    sglang_versions,
    unsloth_llama_versions,
    vllm_versions,
    llama_swap,
)
from backend.models.hub import set_huggingface_token
from backend.logging_config import describe_error, get_logger, log_api_error, setup_logging
from backend.operations.action_recovery import ActionAdmissionError
from backend.operations.supervisor import ResourceBusyError
from backend.store_io import (
    StoreDurabilityError,
    StoreIoBusy,
    StoreIoMiddleware,
    persistence_http_error,
)

# Set up logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def ensure_data_directories():
    """Ensure data directories exist and are writable"""
    from backend.data_store import studio_data_dir

    data_dir = studio_data_dir()

    subdirs = [
        "config",
        "logs",
        "llama-cpp",
        "audio-cpp",
        "models/audio-cpp",
        "lmdeploy",
        "1cat-vllm",
        "sglang",
        "sglang-v100",
        "vllm",
        "unsloth-llama",
        "temp",
    ]

    try:
        # Ensure main data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Ensure subdirectories exist
        import stat

        for subdir in subdirs:
            subdir_path = os.path.join(data_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            # Ensure directory has proper permissions (read, write, execute for owner)
            try:
                os.chmod(
                    subdir_path,
                    stat.S_IRWXU
                    | stat.S_IRGRP
                    | stat.S_IXGRP
                    | stat.S_IROTH
                    | stat.S_IXOTH,
                )
            except Exception as perm_error:
                logger.warning(
                    f"Could not set permissions on {subdir_path}: {perm_error}"
                )

        # Ensure the data directory itself is writable
        try:
            os.chmod(
                data_dir,
                stat.S_IRWXU
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH,
            )
        except Exception as perm_error:
            logger.warning(f"Could not set permissions on {data_dir}: {perm_error}")

        # Try to create a test file to verify write permissions
        test_file = os.path.join(data_dir, ".write_test")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"Data directory {data_dir} is writable")
        except PermissionError as e:
            logger.error(f"Data directory {data_dir} is not writable: {e}")
            logger.warning(
                f"Current user: {os.getuid() if hasattr(os, 'getuid') else 'unknown'}, directory owner check needed"
            )
            logger.warning("Attempting to fix permissions...")
            # Try to fix permissions (may fail if not running as root)
            try:
                import stat

                os.chmod(
                    data_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH
                )
                logger.info("Fixed data directory permissions")
            except Exception as perm_error:
                logger.warning(f"Could not fix permissions automatically: {perm_error}")
                logger.warning(
                    "You may need to fix permissions manually on the host volume"
                )

    except Exception as e:
        logger.error(f"Failed to ensure data directories: {e}")


# Global singleton (module level)
llama_swap_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_swap_manager

    # Startup
    ensure_data_directories()
    from backend.data_store import hold_configuration_writes, release_configuration_writes

    hold_configuration_writes()
    try:
        get_store()  # Ensure YAML config files exist
        from backend.config_backup import reconcile_config_restore

        try:
            restored = reconcile_config_restore(get_store())
        except Exception as exc:
            raise RuntimeError("Configuration restore recovery failed; refusing startup") from exc
        else:
            if restored.get("outcome") == "unknown":
                raise RuntimeError("Configuration restore outcome is unknown; refusing startup")
            elif restored.get("outcome") == "pre_import":
                logger.info(
                    "Rolled an interrupted configuration restore back to the pre-import state"
                )
    finally:
        release_configuration_writes()

    from backend.services.model_metadata import warm_gpu_list_cache

    try:
        await warm_gpu_list_cache()
    except Exception as e:
        logger.warning("Failed to warm GPU list cache at startup: %s", e)

    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if huggingface_api_key:
        set_huggingface_token(huggingface_api_key)
        logger.info("HuggingFace API key loaded from environment variable")

    from backend.operations.supervisor import get_supervisor

    from backend.static_assets import ensure_precompressed_assets

    assets_dir = os.path.join("frontend", "dist", "assets")
    if os.path.isdir(assets_dir):
        try:
            written = await asyncio.to_thread(ensure_precompressed_assets, assets_dir)
            if written:
                logger.info("Prepared %s precompressed frontend asset(s)", written)
        except Exception as exc:
            logger.warning("Could not precompress frontend assets: %s", exc)

    try:
        repaired = get_supervisor().reconcile_startup()
        if repaired.get("outcome") == "unknown":
            logger.warning(
                "Operation reconciliation could not be established: %s",
                repaired.get("detail"),
            )
        elif repaired.get("interrupted") or repaired.get("unknown"):
            logger.info(
                "Reconciled operations at startup: %s interrupted, %s unknown",
                repaired.get("interrupted"),
                repaired.get("unknown"),
            )
        from backend.services.model_runtime_apply import reconcile_journals

        manifests = reconcile_journals()
        if manifests:
            logger.info("Reconciled %s interrupted launch apply journal(s)", manifests)
    except Exception as exc:
        logger.warning("Operation reconciliation failed: %s", exc)

    from backend.ops_metrics import monitor_event_loop

    app.state.loop_monitor = asyncio.create_task(monitor_event_loop())

    from backend.proxy.llama_swap.manager import get_llama_swap_manager

    llama_swap_manager = get_llama_swap_manager()

    from backend.proxy.llama_swap.config import any_active_runtime_in_db

    if any_active_runtime_in_db():
        try:
            await llama_swap_manager.start_proxy()
            logger.info(
                "llama-swap proxy started on port %s",
                llama_swap_manager.proxy_port,
            )
        except Exception as e:
            logger.exception("Failed to start llama-swap: %s", e)
            logger.warning("Multi-model serving unavailable")
    else:
        logger.warning(
            "Skipping llama-swap start: no registered inference runtime is active. "
            "Install or activate a compatible engine to enable model serving."
        )

    yield

    # Shutdown
    monitor = getattr(app.state, "loop_monitor", None)
    if monitor is not None:
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass

    from backend.operations.supervisor import get_supervisor

    await get_supervisor().drain()

    from backend.http_client import aclose_http_client
    from backend.store_io import drain_store_io

    await drain_store_io()
    await aclose_http_client()
    from backend.proxy.llama_swap.client import aclose_proxy_clients

    await aclose_proxy_clients()

    # Stop llama-swap (automatically stops all models)
    if llama_swap_manager:
        try:
            await llama_swap_manager.stop_proxy()
            logger.info("llama-swap stopped gracefully")
        except Exception as e:
            logger.exception("Error stopping llama-swap: %s", e)


app = FastAPI(
    title="llama.cpp Docker Manager",
    description="Web UI for managing llama.cpp models and versions",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def logged_http_exception(request: Request, exc: HTTPException):
    log_api_error(logger, request, exc)
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def logged_validation_error(request: Request, exc: RequestValidationError):
    logger.warning(
        "%s %s rejected (422): %s",
        request.method,
        request.url.path,
        describe_error(exc),
    )
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(ActionAdmissionError)
async def action_admission_rejected(request: Request, exc: ActionAdmissionError):
    log_api_error(logger, request, exc)
    return JSONResponse(status_code=409, content={"detail": exc.detail})


@app.exception_handler(ResourceBusyError)
async def action_resource_busy(request: Request, exc: ResourceBusyError):
    log_api_error(logger, request, exc)
    return JSONResponse(
        status_code=409,
        content={
            "detail": {
                "code": "ACTION_IN_FLIGHT",
                "message": str(exc),
            }
        },
    )


@app.exception_handler(StoreIoBusy)
async def persistence_queue_full(request: Request, exc: StoreIoBusy):
    log_api_error(logger, request, exc)
    status, body = persistence_http_error(exc)
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(StoreDurabilityError)
async def persistence_write_failed(request: Request, exc: StoreDurabilityError):
    log_api_error(logger, request, exc)
    status, body = persistence_http_error(exc)
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(Exception)
async def logged_unhandled_exception(request: Request, exc: Exception):
    if isinstance(exc, ClientDisconnect):
        logger.info("%s %s client disconnected", request.method, request.url.path)
        return JSONResponse(status_code=499, content={"detail": "Client disconnected"})
    log_api_error(logger, request, exc)
    return JSONResponse(status_code=500, content={"detail": describe_error(exc)})

# CORS middleware
# CORS configuration via environment variables (safer defaults)
# BACKEND_CORS_ORIGINS: comma-separated list of origins. Example: "http://localhost:5173,http://localhost:8080"
# BACKEND_CORS_ALLOW_CREDENTIALS: "true"/"false" (default false; forced false when origins == ["*"])
cors_origins_env = os.getenv(
    "BACKEND_CORS_ORIGINS",
    ",".join(
        f"http://{host}:{port}"
        for host in ("localhost", "127.0.0.1")
        for port in (5173, 5174, 5175, 5176, 8080)
    ),
).strip()
allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] or [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

allow_credentials_env = (
    os.getenv("BACKEND_CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
)
# If wildcard origin is used, do not allow credentials per browser security model
if len(allow_origins) == 1 and allow_origins[0] == "*":
    allow_credentials_env = False

from backend.access_policy import ManagementAccessMiddleware
from backend.static_assets import HashedAssetFiles

# Fsync queued YAML writes before the response starts. Inside access control so
# rejected requests do not wait on the store thread.
app.add_middleware(StoreIoMiddleware)

# Access control is inside CORS so browser clients still receive CORS headers on 401/403.
app.add_middleware(ManagementAccessMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials_env,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(engines.router, prefix="/api/engines", tags=["engines"])
app.include_router(
    audio_cpp_versions.router,
    prefix="/api/audio-cpp",
    tags=["audio.cpp"],
)
app.include_router(
    model_catalog.router,
    prefix="/api/model-catalog",
    tags=["model catalog"],
)
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(
    model_config_templates.router,
    prefix="/api/model-config-templates",
    tags=["model-config-templates"],
)
app.include_router(
    llama_versions.router, prefix="/api/llama-versions", tags=["llama-versions"]
)
app.include_router(status.router, prefix="/api", tags=["status"])
app.include_router(config_backup.router, prefix="/api", tags=["config-backup"])
app.include_router(gpu_info.router, prefix="/api", tags=["gpu"])
app.include_router(lmdeploy_versions.router, prefix="/api", tags=["lmdeploy"])
app.include_router(
    onecat_vllm_versions.router, prefix="/api", tags=["1cat-vllm"]
)
app.include_router(vllm_versions.router, prefix="/api", tags=["vllm"])
app.include_router(sglang_versions.router, prefix="/api", tags=["sglang"])
app.include_router(
    unsloth_llama_versions.router, prefix="/api", tags=["unsloth-llama"]
)
app.include_router(llama_swap.router, prefix="/api", tags=["llama-swap"])
app.include_router(
    audio_openai_proxy.router,
    prefix="/v1/audio",
    tags=["audio-openai-proxy"],
)
app.include_router(
    audio_openai_proxy.tasks_router, prefix="/v1", tags=["audio-tasks-proxy"]
)
app.include_router(
    audio_openai_proxy.batches_router, prefix="/v1", tags=["audio-batches-proxy"]
)

from backend.access_policy import access_status, establish_session
from backend.schemas.tasks import SessionLogin


@app.get("/api/access")
async def get_access_policy(request: Request):
    """Report whether this process expects loopback use or an authenticated remote session."""
    return access_status(request)


@app.post("/api/session")
async def create_session(body: SessionLogin):
    """Exchange the management token for an HttpOnly session cookie and a CSRF token."""
    established = establish_session(body.token)
    if established is None:
        return JSONResponse({"detail": "Invalid management token"}, status_code=401)
    session_id, csrf = established
    response = JSONResponse({"authenticated": True})
    response.set_cookie(
        "studio_session",
        session_id,
        httponly=True,
        samesite="strict",
    )
    response.set_cookie(
        "studio_csrf",
        csrf,
        httponly=False,
        samesite="strict",
    )
    return response


# SSE endpoint for progress tracking


@app.post("/api/tasks/dismiss")
async def dismiss_activity_task(request: Request):
    """Remove a finished activity so later snapshots do not restore it."""
    try:
        body = await request.json()
    except Exception:
        body = None
    task_id = str((body or {}).get("task_id") or "").strip() if isinstance(body, dict) else ""
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    if not get_progress_manager().dismiss_task(task_id):
        raise HTTPException(status_code=409, detail="Task is still running")
    return {"ok": True}


@app.get("/api/events")
async def sse_events(request: Request):
    """Server-Sent Events endpoint for progress tracking."""
    logger.info("SSE /api/events: client connected")
    pm = get_progress_manager()

    async def logged_stream():
        first = True
        async for chunk in pm.subscribe():
            if first:
                logger.info("SSE: sending first chunk to client")
                first = False
            yield chunk

    return StreamingResponse(
        logged_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable proxy buffering (nginx, etc.)
        },
    )


# Serve static files (built frontend)
if os.path.exists("frontend/dist"):
    # Mount assets only if they exist
    if os.path.exists("frontend/dist/assets"):
        app.mount(
            "/assets",
            HashedAssetFiles(directory="frontend/dist/assets"),
            name="assets",
        )
    else:
        logger.warning("frontend/dist/assets not found, assets will not be served")

    # Serve static files from public directory
    @app.get("/vite.svg")
    async def serve_vite_svg():
        return FileResponse("frontend/public/vite.svg")

    @app.get("/favicon.ico")
    async def serve_favicon():
        return FileResponse("frontend/public/favicon.ico")

    # Catch-all route for Vue Router (must be after API routes)
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # If it's an API or OpenAI audio proxy route, let it pass through
        if (
            full_path.startswith("api/")
            or full_path.startswith("v1/")
        ):
            from fastapi.responses import JSONResponse

            return JSONResponse({"error": "Not found"}, status_code=404)

        index_path = "frontend/dist/index.html"
        if os.path.exists(index_path):
            return FileResponse(
                index_path,
                headers={"Cache-Control": "no-cache"},
            )
        logger.warning(
            "frontend/dist/index.html not found, serving simple fallback page"
        )
        return HTMLResponse(
            """<!DOCTYPE html>
<html>
<head>
    <title>llama-cpp-studio</title>
    <meta charset="UTF-8">
</head>
<body>
    <h1>llama-cpp-studio Backend</h1>
    <p>Backend is running. To use the full application, please build the frontend:</p>
    <pre>npm run build</pre>
    <p>Or access the API at:</p>
    <ul>
        <li><a href="/api/status">GET /api/status</a> - System status</li>
        <li><a href="/api/live">GET /api/live</a> - Liveness</li>
        <li><a href="/api/ready">GET /api/ready</a> - Readiness</li>
    </ul>
</body>
</html>""",
            headers={"Cache-Control": "no-cache"},
        )


if __name__ == "__main__":
    # Auto-reload in development: on by default when not in Docker; set RELOAD=false to disable
    in_docker = os.path.exists("/app/data")
    enable_reload = os.getenv(
        "RELOAD", "true" if not in_docker else "false"
    ).lower() in ("true", "1", "yes")
    # Watch the backend package directory (works when run from repo root with --app-dir backend)
    backend_dir = os.path.abspath(os.path.dirname(__file__))
    reload_dirs = [backend_dir] if enable_reload else None

    from backend.access_policy import bind_host

    uvicorn.run(
        "main:app",
        host=bind_host(),
        port=8080,
        reload=enable_reload,
        reload_dirs=reload_dirs,
        log_level="info",
    )
