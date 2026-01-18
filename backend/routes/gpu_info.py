from fastapi import APIRouter
import multiprocessing
from backend.gpu_detector import get_gpu_info as detect_gpu_info
from backend.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/gpu-info")
async def get_gpu_info():
    """Detect GPU information and CPU capabilities via unified detector"""
    try:
        info = await detect_gpu_info()
    except Exception as exc:
        logger.exception("GPU detection failed: %s", exc)
        info = {
            "vendor": None,
            "cuda_version": "Unknown",
            "device_count": 0,
            "gpus": [],
            "total_vram": 0,
            "available_vram": 0,
            "cpu_only_mode": True,
            "error": str(exc),
        }
    # Ensure cpu_threads is included for clients that expect it
    info.setdefault("cpu_threads", multiprocessing.cpu_count())
    return info
