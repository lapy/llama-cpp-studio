from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import psutil
import os

from backend.database import get_db, RunningInstance

router = APIRouter()


@router.get("/status")
async def get_system_status(db: Session = Depends(get_db)):
    """Get system status and running instances"""
    running_instances = db.query(RunningInstance).all()
    
    # Get system info
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    # Use data directory at project root or /app/data for Docker
    data_dir = 'data' if os.path.exists('data') else '/app/data'
    try:
        disk = psutil.disk_usage(data_dir)
    except FileNotFoundError:
        # Fallback to root directory if data doesn't exist
        disk = psutil.disk_usage('/')
    
    # Format running instances (no process checking needed)
    active_instances = []
    for instance in running_instances:
        active_instances.append({
            "id": instance.id,
            "model_id": instance.model_id,
            "port": instance.port,
            "proxy_model_name": instance.proxy_model_name,
            "started_at": instance.started_at
        })
    
    return {
        "system": {
            "cpu_percent": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            }
        },
        "running_instances": active_instances,
        "proxy_status": {
            "enabled": True,
            "port": 2000,
            "endpoint": "http://localhost:2000/v1/chat/completions"
        }
    }