from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.unified_monitor import unified_monitor
from backend.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/monitoring/status")
async def get_system_status():
    """Get comprehensive system status"""
    return await unified_monitor.get_system_status()


@router.get("/monitoring/models")
async def get_running_models():
    """Get currently running models from llama-swap"""
    return await unified_monitor.get_running_models()


@router.post("/monitoring/unload-all")
async def unload_all_models():
    """Unload all models via llama-swap"""
    return await unified_monitor.unload_all_models()


@router.get("/monitoring/health")
async def get_system_health():
    """Get llama-swap and system health status"""
    return await unified_monitor.get_system_health()


@router.get("/monitoring/debug")
async def debug_monitoring_data():
    """Debug endpoint to see what data is being collected"""
    from backend.unified_monitor import unified_monitor
    from backend.llama_swap_client import LlamaSwapClient
    
    # Get raw data from external source
    external_client = LlamaSwapClient()
    try:
        external_models = await external_client.get_running_models()
    except Exception as e:
        external_models = {"error": str(e)}
    
    # Get system status
    try:
        system_status = await unified_monitor.get_system_status()
    except Exception as e:
        system_status = {"error": str(e)}
    
    return {
        "running_models": external_models,
        "system_status": system_status,
        "timestamp": "2024-01-01T00:00:00Z"
    }


@router.websocket("/monitoring/ws")
async def monitoring_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring data"""
    await unified_monitor.add_subscriber(websocket)
    
    try:
        while True:
            # Keep the connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Echo back any received data (for testing)
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        await unified_monitor.remove_subscriber(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await unified_monitor.remove_subscriber(websocket)