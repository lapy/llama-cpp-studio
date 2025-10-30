from fastapi import WebSocket
from typing import List, Dict, Optional, Callable
import json
import asyncio
import time
from datetime import datetime
from backend.logging_config import get_logger

logger = get_logger(__name__)


class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscribers: Dict[str, List[Callable]] = {}
    
    async def connect(self, websocket: WebSocket):
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
            
            # Send a test message to verify the connection works
            await self.send_personal_message(json.dumps({
                "type": "connection_test",
                "message": "WebSocket connection established successfully",
                "timestamp": datetime.utcnow().isoformat()
            }), websocket)
            
        except Exception as e:
            logger.error(f"Error in WebSocketManager.connect: {e}")
            raise
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all active WebSocket connections"""
        if not self.active_connections:
            logger.warning(f"No active WebSocket connections to broadcast message: {message.get('type', 'unknown')}")
            return

        message_str = json.dumps(message)
        logger.debug(f"Broadcasting to {len(self.active_connections)} connections: {message.get('type', 'unknown')}")

        async def _send(conn):
            try:
                await conn.send_text(message_str)
                return None
            except Exception as e:
                return conn

        # Send concurrently and collect failed connections
        results = await asyncio.gather(*[_send(c) for c in list(self.active_connections)], return_exceptions=False)
        for failed in results:
            if isinstance(failed, WebSocket):
                self.disconnect(failed)
    
    # Legacy methods for backward compatibility
    async def send_download_progress(self, task_id: str, progress: int, message: str = "", 
                                   bytes_downloaded: int = 0, total_bytes: int = 0, 
                                   speed_mbps: float = 0, eta_seconds: int = 0, filename: str = ""):
        await self.broadcast({
            "type": "download_progress",
            "task_id": task_id,
            "progress": progress,
            "message": message,
            "bytes_downloaded": bytes_downloaded,
            "total_bytes": total_bytes,
            "speed_mbps": speed_mbps,
            "eta_seconds": eta_seconds,
            "filename": filename,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def send_build_progress(self, task_id: str, stage: str, progress: int, 
                                message: str = "", log_lines: List[str] = None):
        message_data = {
            "type": "build_progress",
            "task_id": task_id,
            "stage": stage,
            "progress": progress,
            "message": message,
            "log_lines": log_lines or [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.debug(f"Sending build progress: task_id={task_id}, stage={stage}, progress={progress}, message='{message}', connections={len(self.active_connections)}")
        logger.debug(f"Message data: {message_data}")
        await self.broadcast(message_data)
    
    async def send_model_status_update(self, model_id: int, status: str, 
                                     details: dict = None):
        await self.broadcast({
            "type": "model_status",
            "model_id": model_id,
            "status": status,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def send_notification(self, title: str, message: str, type: str = "info", 
                              actions: List[dict] = None):
        await self.broadcast({
            "type": "notification",
            "title": title,
            "message": message,
            "notification_type": type,
            "actions": actions or [],
            "timestamp": datetime.utcnow().isoformat()
        })


# Global WebSocket manager instance
websocket_manager = WebSocketManager()