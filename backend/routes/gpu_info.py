from fastapi import APIRouter
import multiprocessing
from backend.gpu_detector import get_gpu_info as detect_gpu_info

router = APIRouter()


@router.get("/gpu-info")
async def get_gpu_info():
    """Detect GPU information and CPU capabilities via unified detector"""
    info = await detect_gpu_info()
    # Ensure cpu_threads is included for clients that expect it
    info.setdefault("cpu_threads", multiprocessing.cpu_count())
    return info