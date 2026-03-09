from typing import Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException

from backend.lmdeploy_manager import get_lmdeploy_manager

router = APIRouter()


@router.get("/lmdeploy/check-updates")
async def lmdeploy_check_updates() -> Dict:
  """Check PyPI for latest LMDeploy version."""
  try:
    async with httpx.AsyncClient() as client:
      r = await client.get("https://pypi.org/pypi/lmdeploy/json", timeout=10.0)
      r.raise_for_status()
      data = r.json()
      info = data.get("info", {})
      return {
        "latest_version": info.get("version"),
        "releases": list(data.get("releases", {}).keys()),
      }
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Failed to check PyPI: {exc}")


@router.get("/lmdeploy/status")
async def lmdeploy_installer_status() -> Dict:
  manager = get_lmdeploy_manager()
  return manager.status()


@router.post("/lmdeploy/install")
async def lmdeploy_install(request: Optional[Dict[str, str]] = None) -> Dict:
  manager = get_lmdeploy_manager()
  payload = request or {}
  version = payload.get("version")
  force_reinstall = bool(payload.get("force_reinstall"))
  try:
    return await manager.install_release(version=version, force_reinstall=force_reinstall)
  except RuntimeError as exc:
    raise HTTPException(status_code=409, detail=str(exc))


@router.post("/lmdeploy/install-source")
async def lmdeploy_install_source(request: Optional[Dict[str, str]] = None) -> Dict:
  """Install LMDeploy from a git repo and branch (for development)."""
  manager = get_lmdeploy_manager()
  payload = request or {}
  repo_url = payload.get("repo_url", "https://github.com/InternLM/lmdeploy.git")
  branch = payload.get("branch", "main")
  try:
    return await manager.install_from_source(repo_url=repo_url, branch=branch)
  except RuntimeError as exc:
    raise HTTPException(status_code=409, detail=str(exc))


@router.post("/lmdeploy/remove")
async def lmdeploy_remove() -> Dict:
  manager = get_lmdeploy_manager()
  try:
    return await manager.remove()
  except RuntimeError as exc:
    raise HTTPException(status_code=409, detail=str(exc))
