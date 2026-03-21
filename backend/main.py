import os
import asyncio
import uvicorn
import time
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from backend.data_store import get_store
from backend.routes import (
    models,
    llama_versions,
    status,
    gpu_info,
    lmdeploy_versions,
    llama_swap,
)
from backend.huggingface import set_huggingface_token
from backend.logging_config import setup_logging, get_logger

# Set up logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def ensure_data_directories():
    """Ensure data directories exist and are writable"""
    # Determine data directory - use /app/data in Docker, ./data locally
    if os.path.exists("/app/data"):
        data_dir = "/app/data"
    else:
        data_dir = "data"
    
    subdirs = ["config", "logs", "llama-cpp", "lmdeploy", "temp"]

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
                os.chmod(subdir_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            except Exception as perm_error:
                logger.warning(f"Could not set permissions on {subdir_path}: {perm_error}")

        # Ensure the data directory itself is writable
        try:
            os.chmod(data_dir, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
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
            logger.warning(f"Current user: {os.getuid() if hasattr(os, 'getuid') else 'unknown'}, directory owner check needed")
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
    get_store()  # Ensure YAML config files exist

    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if huggingface_api_key:
        set_huggingface_token(huggingface_api_key)
        logger.info("HuggingFace API key loaded from environment variable")

    from backend.llama_swap_manager import get_llama_swap_manager

    llama_swap_manager = get_llama_swap_manager()

    store = get_store()
    active_version = None
    for engine in ("llama_cpp", "ik_llama"):
        v = store.get_active_engine_version(engine)
        if v and v.get("binary_path"):
            path = v["binary_path"]
            if os.path.isabs(path) and os.path.exists(path):
                active_version = v
                break
            if os.path.exists(os.path.abspath(path)):
                active_version = v
                break

    if active_version and active_version.get("binary_path"):
        try:
            await llama_swap_manager.start_proxy()
            logger.info("llama-swap proxy started on port 2000")
        except Exception as e:
            logger.error(f"Failed to start llama-swap: {e}")
            logger.warning("Multi-model serving unavailable")
    else:
        logger.warning(
            "Skipping llama-swap start: no active llama.cpp version found. "
            "Install or activate a llama.cpp build to enable multi-model serving."
        )

    yield

    # Shutdown

    # Stop llama-swap (automatically stops all models)
    if llama_swap_manager:
        try:
            await llama_swap_manager.stop_proxy()
            logger.info("llama-swap stopped gracefully")
        except Exception as e:
            logger.error(f"Error stopping llama-swap: {e}")


app = FastAPI(
    title="llama.cpp Docker Manager",
    description="Web UI for managing llama.cpp models and versions",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
# CORS configuration via environment variables (safer defaults)
# BACKEND_CORS_ORIGINS: comma-separated list of origins. Example: "http://localhost:5173,http://localhost:8080"
# BACKEND_CORS_ALLOW_CREDENTIALS: "true"/"false" (default false; forced false when origins == ["*"])
cors_origins_env = os.getenv(
    "BACKEND_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:5176,http://localhost:8080",
).strip()
allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] or [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:5176",
    "http://localhost:8080",
]

allow_credentials_env = (
    os.getenv("BACKEND_CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
)
# If wildcard origin is used, do not allow credentials per browser security model
if len(allow_origins) == 1 and allow_origins[0] == "*":
    allow_credentials_env = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials_env,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(
    llama_versions.router, prefix="/api/llama-versions", tags=["llama-versions"]
)
app.include_router(status.router, prefix="/api", tags=["status"])
app.include_router(gpu_info.router, prefix="/api", tags=["gpu"])
app.include_router(lmdeploy_versions.router, prefix="/api", tags=["lmdeploy"])
app.include_router(llama_swap.router, prefix="/api", tags=["llama-swap"])

# SSE endpoint for progress tracking
from backend.progress_manager import get_progress_manager
from fastapi.responses import StreamingResponse


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
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import Response

    # Custom static files handler with cache-busting
    class CacheBustingStaticFiles(StaticFiles):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def file_response(self, *args, **kwargs):
            response = super().file_response(*args, **kwargs)
            # Add cache-busting headers for CSS and JS files
            if response.headers.get("content-type", "").startswith(
                ("text/css", "application/javascript")
            ):
                response.headers["Cache-Control"] = (
                    "no-cache, no-store, must-revalidate"
                )
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response

    # Mount assets only if they exist
    if os.path.exists("frontend/dist/assets"):
        app.mount(
            "/assets",
            CacheBustingStaticFiles(directory="frontend/dist/assets"),
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
        # If it's an API route, let it pass through
        if full_path.startswith("api/"):
            return {"error": "Not found"}

        # Serve index.html for all other routes (Vue Router will handle routing)
        # Read the HTML file and add cache-busting query parameter to script tags
        try:
            with open("frontend/dist/index.html", "r") as f:
                html_content = f.read()
        except FileNotFoundError:
            # Fallback to a simple HTML page if dist/index.html doesn't exist
            logger.warning(
                "frontend/dist/index.html not found, serving simple fallback page"
            )
            html_content = """<!DOCTYPE html>
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
        <li><a href="/api/gpu-info">GET /api/gpu-info</a> - GPU information</li>
    </ul>
</body>
</html>"""

        # Add cache-busting query parameter to script and link tags
        timestamp = int(time.time() * 1000)
        html_content = html_content.replace('src="/assets/', f'src="/assets/')
        html_content = html_content.replace('href="/assets/', f'href="/assets/')
        # Add cache-busting query parameter after the filename
        import re

        html_content = re.sub(
            r'(src="/assets/[^"]+\.js")', rf"\1?v={timestamp}", html_content
        )
        html_content = re.sub(
            r'(href="/assets/[^"]+\.css")', rf"\1?v={timestamp}", html_content
        )

        from fastapi.responses import HTMLResponse

        response = HTMLResponse(html_content)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


if __name__ == "__main__":
    # Auto-reload in development: on by default when not in Docker; set RELOAD=false to disable
    in_docker = os.path.exists("/app/data")
    enable_reload = os.getenv("RELOAD", "true" if not in_docker else "false").lower() in ("true", "1", "yes")
    # Watch the backend package directory (works when run from repo root with --app-dir backend)
    backend_dir = os.path.abspath(os.path.dirname(__file__))
    reload_dirs = [backend_dir] if enable_reload else None

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=enable_reload,
        reload_dirs=reload_dirs,
        log_level="info",
    )
