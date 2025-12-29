from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import json
import os
import subprocess
import requests
import time
import platform
import shutil
import stat
from datetime import datetime

from backend.database import get_db, LlamaVersion
from backend.llama_manager import LlamaManager, BuildConfig
from backend.websocket_manager import websocket_manager
from backend.logging_config import get_logger
from backend.gpu_detector import get_gpu_info, detect_build_capabilities
from backend.cuda_installer import get_cuda_installer

router = APIRouter()
llama_manager = LlamaManager()
logger = get_logger(__name__)


def _remove_readonly(func, path, exc):
    """Helper function to handle readonly files on Windows"""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        logger.warning(f"Could not remove {path}: {e}")


def _robust_rmtree(path: str, max_retries: int = 3) -> None:
    """Robustly remove a directory tree, handling Windows file locks"""
    if not os.path.exists(path):
        return
    
    for attempt in range(max_retries):
        try:
            # Use onerror callback to handle readonly files (common on Windows)
            shutil.rmtree(path, onerror=_remove_readonly)
            logger.info(f"Successfully deleted directory: {path}")
            return
        except PermissionError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Permission error deleting {path}, attempt {attempt + 1}/{max_retries}: {e}")
                time.sleep(0.5)  # Wait a bit before retrying
            else:
                logger.error(f"Failed to delete {path} after {max_retries} attempts: {e}")
                raise
        except OSError as e:
            if attempt < max_retries - 1:
                logger.warning(f"OS error deleting {path}, attempt {attempt + 1}/{max_retries}: {e}")
                time.sleep(0.5)
            else:
                logger.error(f"Failed to delete {path} after {max_retries} attempts: {e}")
                raise


@router.get("")
@router.get("/")
async def list_llama_versions(db: Session = Depends(get_db)):
    """List all installed llama.cpp versions"""
    versions = db.query(LlamaVersion).all()
    return [
        {
            "id": version.id,
            "version": version.version,
            "install_type": version.install_type,
            "binary_path": version.binary_path,
            "source_commit": version.source_commit,
            "patches": json.loads(version.patches) if version.patches else [],
            "installed_at": version.installed_at,
            "is_active": version.is_active,
            "build_config": version.build_config,
            "repository_source": version.repository_source or "llama.cpp"
        }
        for version in versions
    ]


@router.get("/check-updates")
async def check_updates():
    """Check for llama.cpp updates (both releases and source)"""
    try:
        # Use the original URLs with redirect handling
        releases_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases"
        commits_url = "https://api.github.com/repos/ggerganov/llama.cpp/commits?per_page=1"
        
        # Check GitHub releases
        releases_response = requests.get(releases_url, allow_redirects=True)
        releases_response.raise_for_status()
        releases = releases_response.json()
        
        latest_release = releases[0] if releases else None
        
        # Check latest commit from main branch
        commits_response = requests.get(commits_url, allow_redirects=True)
        commits_response.raise_for_status()
        commits = commits_response.json()
        latest_commit = commits[0] if commits else None
        
        return {
            "latest_release": {
                "tag_name": latest_release["tag_name"] if latest_release else None,
                "published_at": latest_release["published_at"] if latest_release else None,
                "html_url": latest_release["html_url"] if latest_release else None
            },
            "latest_commit": {
                "sha": latest_commit["sha"],
                "commit_date": latest_commit["commit"]["committer"]["date"],
                "message": latest_commit["commit"]["message"]
            }
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            raise HTTPException(status_code=429, detail="GitHub API rate limit exceeded. Please try again later.")
        elif e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="GitHub repository not found")
        else:
            raise HTTPException(status_code=500, detail=f"GitHub API error: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update check failed: {str(e)}")


@router.get("/releases/{tag_name}/assets")
async def get_release_assets(tag_name: str):
    """List compatible release artifacts for a given tag."""
    try:
        assets = llama_manager.get_release_assets(tag_name)
        return assets
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            raise HTTPException(status_code=429, detail="GitHub API rate limit exceeded. Please try again later.")
        elif e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Release {tag_name} not found")
        else:
            raise HTTPException(status_code=500, detail=f"GitHub API error: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch release assets: {str(e)}")


@router.get("/build-capabilities")
async def get_build_capabilities_endpoint():
    """Get build capabilities based on detected hardware"""
    try:
        return await detect_build_capabilities()
    except Exception as e:
        logger.error(f"Error detecting build capabilities: {e}")
        # Return safe defaults
        return {
            "cuda": {"available": False, "recommended": False, "reason": f"Error: {str(e)}"},
            "vulkan": {"available": False, "recommended": False, "reason": f"Error: {str(e)}"},
            "metal": {"available": False, "recommended": False, "reason": f"Error: {str(e)}"},
            "openblas": {"available": False, "recommended": False, "reason": f"Error: {str(e)}"}
        }


@router.post("/install-release")
async def install_release(
    request: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Install llama.cpp from GitHub release"""
    try:
        tag_name = request.get("tag_name")
        if not tag_name:
            raise HTTPException(status_code=400, detail="tag_name is required")
        
        raw_asset_id = request.get("asset_id")
        asset_id = None
        if raw_asset_id is not None:
            try:
                asset_id = int(raw_asset_id)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="asset_id must be an integer")
        
        try:
            preview = llama_manager.get_release_install_preview(tag_name, asset_id)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                raise HTTPException(status_code=429, detail="GitHub API rate limit exceeded. Please try again later.")
            elif e.response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Release {tag_name} not found")
            else:
                raise HTTPException(status_code=500, detail=f"GitHub API error: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        version_name = preview.get("version_name")
        
        # Check if version already exists
        if version_name:
            existing = db.query(LlamaVersion).filter(LlamaVersion.version == version_name).first()
        else:
            existing = db.query(LlamaVersion).filter(LlamaVersion.version == tag_name).first()
        if existing:
            detail = "400: Version already installed"
            if version_name:
                detail = f"{detail} ({version_name})"
            raise HTTPException(status_code=400, detail=detail)
        
        # Generate task ID for tracking
        task_id = f"install_release_{tag_name}_{int(time.time())}"
        
        # Start installation in background
        background_tasks.add_task(
            install_release_task,
            tag_name,
            websocket_manager,
            task_id,
            asset_id
        )
        
        return {
            "message": f"Installing release {tag_name}", 
            "task_id": task_id,
            "status": "started",
            "progress": 0,
            "asset_id": asset_id,
            "version_name": version_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def install_release_task(tag_name: str, websocket_manager=None, task_id: str = None, asset_id: Optional[int] = None):
    """Background task to install release with WebSocket progress updates"""
    # Create a new database session for the background task
    from backend.database import SessionLocal
    db = SessionLocal()
    
    try:
        install_result = await llama_manager.install_release(tag_name, websocket_manager, task_id, asset_id)
        binary_path = install_result.get("binary_path")
        asset_info = install_result.get("asset")
        version_name = install_result.get("version_name") or tag_name
        
        if not binary_path:
            raise Exception("Installation completed without returning a binary path.")
        
        # Save to database
        version = LlamaVersion(
            version=version_name,
            install_type="release",
            binary_path=binary_path,
            installed_at=datetime.utcnow(),
            build_config={
                "release_asset": asset_info,
                "tag_name": tag_name
            } if asset_info else None
        )
        db.add(version)
        db.commit()
        
        # Send success notification
        if websocket_manager:
            asset_label = ""
            if asset_info and asset_info.get("name"):
                asset_label = f" ({asset_info['name']})"
            await websocket_manager.send_notification(
                title="Installation Complete",
                message=f"Successfully installed llama.cpp release {version_name}{asset_label}",
                type="success"
            )
        
    except Exception as e:
        logger.error(f"Release installation failed: {e}")
        if websocket_manager:
            await websocket_manager.send_notification(
                title="Installation Failed",
                message=f"Failed to install llama.cpp release: {str(e)}",
                type="error"
            )
    finally:
        # Always close the database session
        db.close()


@router.post("/build-source")
async def build_source(
    request: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Build llama.cpp from source with optional patches"""
    try:
        commit_sha = request.get("commit_sha")
        patches = request.get("patches", [])
        build_config_dict = request.get("build_config")
        repository_source = request.get("repository_source", "llama.cpp")
        version_suffix = request.get("version_suffix")
        
        if not commit_sha:
            raise HTTPException(status_code=400, detail="commit_sha is required")
        
        # Generate unique version name
        commit_short = commit_sha[:8]
        if version_suffix:
            version_name = f"source-{commit_short}-{version_suffix}"
        else:
            # Use timestamp for unique naming
            timestamp = int(time.time())
            version_name = f"source-{commit_short}-{timestamp}"
        
        # Check if version already exists (still check to prevent accidental duplicates)
        existing = db.query(LlamaVersion).filter(LlamaVersion.version == version_name).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Version '{version_name}' already installed")
        
        # Get repository URL from source name
        repository_url = llama_manager.REPOSITORY_SOURCES.get(repository_source)
        if not repository_url:
            raise HTTPException(status_code=400, detail=f"Unknown repository source: {repository_source}")
        
        # Parse build_config if provided
        build_config = None
        if build_config_dict:
            build_config = BuildConfig(**build_config_dict)
        
        # Generate task ID for tracking
        task_id = f"build_{version_name}_{int(time.time())}"
        
        # Start build in background
        background_tasks.add_task(
            build_source_task,
            commit_sha,
            patches,
            build_config,
            version_name,
            repository_source,
            repository_url,
            websocket_manager,
            task_id
        )
        
        return {
            "message": f"Building from source {commit_sha[:8]}", 
            "task_id": task_id,
            "status": "started",
            "progress": 0,
            "version_name": version_name,
            "repository_source": repository_source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def build_source_task(commit_sha: str, patches: List[str], build_config: BuildConfig, version_name: str, repository_source: str, repository_url: str, websocket_manager=None, task_id: str = None):
    """Background task to build from source with WebSocket progress"""
    # Create a new database session for the background task
    from backend.database import SessionLocal
    from dataclasses import asdict
    db = SessionLocal()
    
    try:
        binary_path = await llama_manager.build_source(
            commit_sha, 
            patches, 
            build_config, 
            websocket_manager, 
            task_id,
            repository_url=repository_url,
            version_name=version_name
        )
        
        # Save to database with build_config
        build_config_dict = None
        if build_config:
            build_config_dict = asdict(build_config)
            # Add repository_source to build_config for completeness
            build_config_dict["repository_source"] = repository_source
        
        version = LlamaVersion(
            version=version_name,
            install_type="patched" if patches else "source",
            binary_path=binary_path,
            source_commit=commit_sha,
            patches=json.dumps(patches),
            build_config=build_config_dict,
            repository_source=repository_source,
            installed_at=datetime.utcnow()
        )
        db.add(version)
        db.commit()
        
        # Send success notification
        if websocket_manager:
            await websocket_manager.send_notification(
                title="Build Complete",
                message=f"Successfully built {repository_source} from source {commit_sha[:8]}",
                type="success"
            )
        
    except Exception as e:
        logger.error(f"Source build failed: {e}")
        if websocket_manager:
            try:
                logger.info(f"Sending build failure notification for task {task_id}")
                await websocket_manager.send_notification(
                    title="Build Failed",
                    message=f"Failed to build llama.cpp from source: {str(e)}",
                    type="error"
                )
                # Also send a build progress error message
                if task_id:
                    await websocket_manager.send_build_progress(
                        task_id=task_id,
                        stage="error",
                        progress=0,
                        message=f"Build task failed: {str(e)}",
                        log_lines=[f"Task error: {str(e)}", f"Error type: {type(e).__name__}"]
                    )
                logger.info(f"Build failure notifications sent successfully")
            except Exception as ws_error:
                logger.error(f"Failed to send build failure notification: {ws_error}")
    finally:
        # Always close the database session
        db.close()


@router.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    # This is a simple implementation - in production you might want to store task status in Redis or database
    # For now, we'll just return a basic response since the WebSocket provides real-time updates
    return {
        "task_id": task_id,
        "status": "running",  # Could be "running", "completed", "failed"
        "message": "Task is running. Use WebSocket for real-time progress updates."
    }


@router.get("/verify/{version}")
async def verify_version(version: str):
    """Verify that all required llama.cpp commands are available for a version"""
    try:
        verification = llama_manager.verify_installation(version)
        commands = llama_manager.get_all_commands(version)
        
        return {
            "version": version,
            "verification": verification,
            "commands": commands,
            "all_available": all(verification.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/commands/{version}")
async def get_version_commands(version: str):
    """Get all available commands for a specific version"""
    try:
        commands = llama_manager.get_all_commands(version)
        return {
            "version": version,
            "commands": commands
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{version_id}")
async def delete_version(
    version_id: int,
    db: Session = Depends(get_db)
):
    """Delete llama.cpp version"""
    version = db.query(LlamaVersion).filter(LlamaVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    
    # Prevent deletion of active version
    if version.is_active:
        raise HTTPException(status_code=400, detail="Cannot delete active version")
    
    try:
        # Delete the entire version directory
        if version.binary_path and os.path.exists(version.binary_path):
            # Go up two levels from build/bin/llama-server to get the version directory
            version_dir = os.path.dirname(os.path.dirname(version.binary_path))
            if os.path.exists(version_dir):
                _robust_rmtree(version_dir)
        
        # Delete from database
        db.delete(version)
        db.commit()
        
        logger.info(f"Deleted llama-cpp version: {version.version}")
        return {"message": f"Deleted llama-cpp version {version.version}"}
    except Exception as e:
        logger.error(f"Failed to delete version {version.version}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete version: {e}")


# CUDA Installer endpoints
@router.get("/cuda-status")
async def get_cuda_status():
    """Get CUDA installation status"""
    try:
        installer = get_cuda_installer()
        status = installer.status()
        return status
    except Exception as e:
        logger.error(f"Failed to get CUDA status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cuda-install")
async def install_cuda(request: dict):
    """Install CUDA Toolkit"""
    try:
        version = request.get("version", "12.6")
        installer = get_cuda_installer()
        
        if installer.is_operation_running():
            raise HTTPException(
                status_code=400,
                detail="A CUDA installation operation is already running"
            )
        
        result = await installer.install(version)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start CUDA installation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cuda-logs")
async def get_cuda_logs():
    """Get CUDA installation logs"""
    try:
        installer = get_cuda_installer()
        logs = installer.read_log_tail()
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Failed to get CUDA logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cuda-uninstall")
async def uninstall_cuda(request: dict):
    """Uninstall CUDA Toolkit"""
    try:
        version = request.get("version")  # Optional - if not provided, uninstalls current
        installer = get_cuda_installer()
        
        if installer.is_operation_running():
            raise HTTPException(
                status_code=400,
                detail="A CUDA installation operation is already running"
            )
        
        result = await installer.uninstall(version)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start CUDA uninstallation: {e}")
        raise HTTPException(status_code=500, detail=str(e))