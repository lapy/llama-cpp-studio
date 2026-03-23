from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from backend.llama_swap_manager import get_llama_swap_manager
from backend.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/llama-swap/pending")
async def llama_swap_pending() -> Dict[str, Any]:
    """
    Compare the on-disk llama-swap config to what would be generated from the current DB.
    """
    manager = get_llama_swap_manager()
    return await manager.get_config_pending_state()


@router.get("/llama-swap/stale")
async def llama_swap_stale() -> Dict[str, Any]:
    """
    Cheap flag for the UI: studio has changes that may require rewriting llama-swap-config.yaml.
    Use GET /llama-swap/pending only when the user opens “apply” or needs a diff summary.
    """
    manager = get_llama_swap_manager()
    return manager.get_swap_config_stale_state()


@router.post("/llama-swap/apply-config")
async def llama_swap_apply_config() -> Dict[str, str]:
    """
    Unload all models via llama-swap, then regenerate and write llama-swap-config.yaml.
    """
    manager = get_llama_swap_manager()
    try:
        await manager.user_apply_regenerate_config()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("apply-config failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"message": "llama-swap configuration applied"}
