from fastapi import APIRouter, HTTPException
import os
import shutil
import stat
import time

from backend.data_store import get_store
from backend.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _remove_readonly(func, path, exc):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        logger.warning(f"Could not remove {path}: {e}")


def _robust_rmtree(path: str, max_retries: int = 3) -> None:
    if not os.path.exists(path):
        return
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path, onerror=_remove_readonly)
            logger.info(f"Successfully deleted directory: {path}")
            return
        except (PermissionError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                logger.error(f"Failed to delete {path} after {max_retries} attempts: {e}")
                raise


def _resolve_binary_path(binary_path: str) -> str:
    if not binary_path:
        return ""
    if os.path.isabs(binary_path):
        return binary_path
    return os.path.join("/app", binary_path)


@router.get("/llama-versions")
async def list_llama_versions():
    """List all installed llama-cpp versions (llama_cpp engine)."""
    store = get_store()
    versions = store.get_engine_versions("llama_cpp")
    result = []
    for i, v in enumerate(versions):
        binary_path = _resolve_binary_path(v.get("binary_path"))
        result.append({
            "id": i,
            "version": v.get("version"),
            "install_type": v.get("type", "source"),
            "source_commit": v.get("source_commit"),
            "is_active": store.get_active_engine_version("llama_cpp") and store.get_active_engine_version("llama_cpp").get("version") == v.get("version"),
            "installed_at": v.get("installed_at"),
            "binary_path": v.get("binary_path"),
            "exists": os.path.exists(binary_path) if binary_path else False,
        })
    return {"versions": result}


@router.post("/llama-versions/{version_id}/activate")
async def activate_llama_version(version_id: str):
    """Activate a specific llama-cpp version (version_id can be index, version string, or "llama_cpp:version")."""
    store = get_store()
    versions = store.get_engine_versions("llama_cpp")
    # Frontend may send id from list endpoint: "llama_cpp:version_str"
    lookup_id = version_id
    if ":" in str(version_id):
        parts = str(version_id).split(":", 1)
        if parts[0] == "llama_cpp":
            lookup_id = parts[1]
    version_entry = None
    try:
        idx = int(lookup_id)
        if 0 <= idx < len(versions):
            version_entry = versions[idx]
    except ValueError:
        pass
    if not version_entry:
        version_entry = next((v for v in versions if str(v.get("version")) == str(lookup_id)), None)
    if not version_entry:
        raise HTTPException(status_code=404, detail="Version not found")
    binary_path = _resolve_binary_path(version_entry.get("binary_path"))
    if not os.path.exists(binary_path):
        raise HTTPException(status_code=400, detail="Binary file does not exist")
    version_str = str(version_entry.get("version"))
    store.set_active_engine_version("llama_cpp", version_str)
    try:
        from backend.llama_swap_manager import get_llama_swap_manager
        llama_swap_manager = get_llama_swap_manager()
        await llama_swap_manager._ensure_correct_binary_path()
        await llama_swap_manager.regenerate_config_with_active_version()
        try:
            await llama_swap_manager.start_proxy()
        except Exception as e:
            logger.warning(f"Failed to start llama-swap after version activation: {e}")
    except Exception as e:
        logger.error(f"Failed to regenerate llama-swap config: {e}")
    logger.info(f"Activated llama-cpp version: {version_str}")
    return {"message": f"Activated llama-cpp version {version_str}"}


@router.delete("/llama-versions/{version_id}")
async def delete_llama_version(version_id: str):
    """Delete a llama-cpp version (version_id can be index or version string)."""
    store = get_store()
    versions = store.get_engine_versions("llama_cpp")
    version_entry = None
    try:
        idx = int(version_id)
        if 0 <= idx < len(versions):
            version_entry = versions[idx]
    except ValueError:
        pass
    if not version_entry:
        version_entry = next((v for v in versions if str(v.get("version")) == str(version_id)), None)
    if not version_entry:
        raise HTTPException(status_code=404, detail="Version not found")
    version_str = str(version_entry.get("version"))
    active = store.get_active_engine_version("llama_cpp")
    if active and str(active.get("version")) == version_str:
        raise HTTPException(status_code=400, detail="Cannot delete active version")
    binary_path = _resolve_binary_path(version_entry.get("binary_path"))
    version_dir = os.path.dirname(os.path.dirname(binary_path)) if binary_path else None
    if version_dir and os.path.exists(version_dir):
        try:
            _robust_rmtree(version_dir)
        except Exception as e:
            logger.error(f"Failed to delete directory {version_dir}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete directory: {e}")
    store.delete_engine_version("llama_cpp", version_str)
    logger.info(f"Deleted llama-cpp version: {version_str}")
    return {"message": f"Deleted llama-cpp version {version_str}"}


@router.get("/llama-versions/active")
async def get_active_llama_version():
    """Get the currently active llama-cpp version."""
    store = get_store()
    active_version = store.get_active_engine_version("llama_cpp")
    if not active_version:
        return {"active_version": None}
    binary_path = _resolve_binary_path(active_version.get("binary_path"))
    return {
        "active_version": {
            "id": 0,
            "version": active_version.get("version"),
            "install_type": active_version.get("type"),
            "source_commit": active_version.get("source_commit"),
            "binary_path": active_version.get("binary_path"),
            "exists": os.path.exists(binary_path) if binary_path else False,
        }
    }
