from typing import Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException

from backend.onecat_vllm_manager import GITHUB_REPO, get_onecat_vllm_manager
from backend.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/1cat-vllm/check-updates")
async def onecat_vllm_check_updates() -> Dict:
    """Check GitHub for the latest 1Cat-vLLM release."""
    try:
        headers = {"Accept": "application/vnd.github+json"}
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            r = await client.get(
                f"https://api.github.com/repos/{GITHUB_REPO}/releases", timeout=10.0
            )
            r.raise_for_status()
            releases = r.json() or []
            tags: List[str] = [
                rel.get("tag_name") for rel in releases if rel.get("tag_name")
            ]
            latest = None
            for rel in releases:
                if rel.get("prerelease"):
                    continue
                latest = rel.get("tag_name")
                break
            if latest is None and releases:
                latest = releases[0].get("tag_name")
            return {
                "latest_version": (latest or "").lstrip("v") or None,
                "releases": [t.lstrip("v") for t in tags],
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to check GitHub: {exc}")


@router.get("/1cat-vllm/status")
async def onecat_vllm_installer_status() -> Dict:
    manager = get_onecat_vllm_manager()
    try:
        return manager.status()
    except Exception as exc:
        logger.warning("1cat-vllm/status: %s", exc)
        # Never fail the whole app load if status probing throws.
        return {
            "installed": False,
            "version": None,
            "binary_path": None,
            "venv_path": None,
            "installed_at": None,
            "removed_at": None,
            "operation": None,
            "operation_started_at": None,
            "last_error": str(exc),
            "log_path": None,
            "install_type": None,
            "release_tag": None,
            "source_repo": None,
            "source_branch": None,
        }


@router.post("/1cat-vllm/install")
async def onecat_vllm_install(request: Optional[Dict[str, str]] = None) -> Dict:
    manager = get_onecat_vllm_manager()
    payload = request or {}
    version = payload.get("version")
    force_reinstall = bool(payload.get("force_reinstall"))
    try:
        return await manager.install_release(
            version=version, force_reinstall=force_reinstall
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/1cat-vllm/install-source")
async def onecat_vllm_install_source(request: Optional[Dict[str, str]] = None) -> Dict:
    """Build and install 1Cat-vLLM from a git repo and branch (for development)."""
    manager = get_onecat_vllm_manager()
    payload = request or {}
    repo_url = payload.get(
        "repo_url", "https://github.com/1CatAI/1Cat-vLLM.git"
    )
    branch = payload.get("branch", "main")
    try:
        return await manager.install_from_source(repo_url=repo_url, branch=branch)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/1cat-vllm/remove")
async def onecat_vllm_remove() -> Dict:
    manager = get_onecat_vllm_manager()
    try:
        return await manager.remove()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
