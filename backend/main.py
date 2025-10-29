import os
import asyncio
import uvicorn
import time
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from backend.database import init_db, LlamaVersion
from backend.routes import models, llama_versions, status, gpu_info, llama_version_manager
from backend.websocket_manager import websocket_manager
from backend.huggingface import set_huggingface_token
from backend.unified_monitor import unified_monitor
from backend.logging_config import setup_logging, get_logger

# Set up logging
setup_logging(level="INFO")
logger = get_logger(__name__)


# Global singleton (module level)
llama_swap_manager = None

async def register_all_models_with_llama_swap():
    """Register all downloaded models with llama-swap on startup"""
    from backend.database import SessionLocal, Model
    from backend.llama_manager import LlamaManager
    
    db = SessionLocal()
    try:
        # Get all downloaded models
        models = db.query(Model).all()
        if not models:
            logger.info("No models found to register with llama-swap")
            return
        
        logger.info(f"Found {len(models)} models to register with llama-swap")
        
        # Get llama-server path from active version
        active_version = db.query(LlamaVersion).filter(LlamaVersion.is_active == True).first()
        if active_version and os.path.exists(active_version.binary_path):
            llama_server_path = active_version.binary_path
            logger.info(f"Using active llama-cpp version: {active_version.version}")
        else:
            # Fallback: try to find llama-server in the llama-cpp directory
            llama_cpp_dir = "data/llama-cpp" if os.path.exists("data") else "/app/data/llama-cpp"
            if os.path.exists(llama_cpp_dir):
                for version_dir in os.listdir(llama_cpp_dir):
                    server_path = os.path.join(llama_cpp_dir, version_dir, "build", "bin", "llama-server")
                    if os.path.exists(server_path) and os.access(server_path, os.X_OK):
                        llama_server_path = server_path
                        logger.info(f"Found llama-server at: {llama_server_path}")
                        break
            
            if not llama_server_path:
                logger.warning("llama-server not found, skipping model registration")
                return
        
        # Register each model with llama-swap (without binary path)
        for model in models:
            try:
                # Create a basic config for the model
                config = {
                    "model": model.file_path,
                    "host": "0.0.0.0",
                    "port": 0,  # Let llama-swap assign ports
                    "ctx_size": 2048,
                    "batch_size": 512,
                    "threads": 4
                }
                
                # Register with llama-swap (no binary path needed)
                proxy_name = await llama_swap_manager.register_model(model, config)
                logger.info(f"Registered model '{model.name}' as '{proxy_name}' with llama-swap")
                
            except Exception as e:
                logger.error(f"Failed to register model '{model.name}' with llama-swap: {e}")
        
        # Generate config with the active version
        await llama_swap_manager.regenerate_config_with_active_version()
                
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_swap_manager
    
    # Startup
    await init_db()
    
    # Initialize configuration manager and update llama-swap config
    
    # Initialize Hugging Face API key from environment variable if available
    huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
    if huggingface_api_key:
        set_huggingface_token(huggingface_api_key)
        logger.info("HuggingFace API key loaded from environment variable")
    
    # Initialize and start llama-swap
    from backend.llama_swap_manager import get_llama_swap_manager
    llama_swap_manager = get_llama_swap_manager()
    
    try:
        await llama_swap_manager.start_proxy()
        logger.info("llama-swap proxy started on port 2000")
    except Exception as e:
        logger.error(f"Failed to start llama-swap: {e}")
        logger.warning("Multi-model serving unavailable")
    
    # Clean stale database state (since llama-swap was not running)
    from backend.database import SessionLocal, RunningInstance, Model
    db = SessionLocal()
    try:
        stale_instances = db.query(RunningInstance).all()
        if stale_instances:
            logger.info(f"Cleaning {len(stale_instances)} stale instances")
            for instance in stale_instances:
                # Update model status
                model = db.query(Model).filter(Model.id == instance.model_id).first()
                if model:
                    model.is_active = False
                db.delete(instance)
            db.commit()
    finally:
        db.close()
    
    # Register all downloaded models with llama-swap
    try:
        await register_all_models_with_llama_swap()
    except Exception as e:
        logger.error(f"Failed to register models with llama-swap: {e}")
    
    # Start unified monitoring
    await unified_monitor.start_monitoring()
    
    yield
    
    # Shutdown
    await unified_monitor.stop_monitoring()
    
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
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use the global WebSocket manager instance

# Include routers
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(llama_versions.router, prefix="/api/llama-versions", tags=["llama-versions"])
app.include_router(llama_version_manager.router, prefix="/api", tags=["llama-version-manager"])
app.include_router(status.router, prefix="/api", tags=["status"])
app.include_router(gpu_info.router, prefix="/api", tags=["gpu"])

# Include monitoring routes
from backend.routes import unified_monitoring
app.include_router(unified_monitoring.router, prefix="/api", tags=["monitoring"])

# WebSocket endpoint for real-time updates (must be before static file serving)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    import json
    
    try:
        logger.info("New WebSocket connection attempt")
        await websocket_manager.connect(websocket)
        logger.info(f"WebSocket connected successfully. Total connections: {len(websocket_manager.active_connections)}")
        
        try:
            while True:
                # Keep connection alive and handle any incoming messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle any client messages if needed
                logger.debug(f"Received WebSocket message: {message}")
                    
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected by client")
            websocket_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Failed to establish WebSocket connection: {e}")
        websocket_manager.disconnect(websocket)

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
            if response.headers.get("content-type", "").startswith(("text/css", "application/javascript")):
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response
    
    # Mount assets only if they exist
    if os.path.exists("frontend/dist/assets"):
        app.mount("/assets", CacheBustingStaticFiles(directory="frontend/dist/assets"), name="assets")
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
        # If it's an API route or WebSocket route, let it pass through
        if full_path.startswith("api/") or full_path.startswith("ws"):
            return {"error": "Not found"}
        
        # Serve index.html for all other routes (Vue Router will handle routing)
        # Read the HTML file and add cache-busting query parameter to script tags
        try:
            with open("frontend/dist/index.html", "r") as f:
                html_content = f.read()
        except FileNotFoundError:
            # Fallback to a simple HTML page if dist/index.html doesn't exist
            logger.warning("frontend/dist/index.html not found, serving simple fallback page")
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
        html_content = html_content.replace(
            'src="/assets/',
            f'src="/assets/'
        )
        html_content = html_content.replace(
            'href="/assets/',
            f'href="/assets/'
        )
        # Add cache-busting query parameter after the filename
        import re
        html_content = re.sub(
            r'(src="/assets/[^"]+\.js")',
            rf'\1?v={timestamp}',
            html_content
        )
        html_content = re.sub(
            r'(href="/assets/[^"]+\.css")',
            rf'\1?v={timestamp}',
            html_content
        )
        
        from fastapi.responses import HTMLResponse
        response = HTMLResponse(html_content)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )
