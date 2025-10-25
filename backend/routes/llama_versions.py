from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import json
import os
import subprocess
import requests
import time
import platform
from datetime import datetime

from backend.database import get_db, LlamaVersion
from backend.llama_manager import LlamaManager, BuildConfig
from backend.websocket_manager import websocket_manager
from backend.logging_config import get_logger
from backend.gpu_detector import get_gpu_info, detect_build_capabilities

router = APIRouter()
llama_manager = LlamaManager()
logger = get_logger(__name__)


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
            "build_config": version.build_config
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
        
        # Check if version already exists
        existing = db.query(LlamaVersion).filter(LlamaVersion.version == tag_name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Version already installed")
        
        # Generate task ID for tracking
        task_id = f"install_release_{tag_name}_{int(time.time())}"
        
        # Start installation in background
        background_tasks.add_task(
            install_release_task,
            tag_name,
            websocket_manager,
            task_id
        )
        
        return {
            "message": f"Installing release {tag_name}", 
            "task_id": task_id,
            "status": "started",
            "progress": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def install_release_task(tag_name: str, websocket_manager=None, task_id: str = None):
    """Background task to install release with WebSocket progress updates"""
    # Create a new database session for the background task
    from backend.database import SessionLocal
    db = SessionLocal()
    
    try:
        binary_path = await llama_manager.install_release(tag_name, websocket_manager, task_id)
        
        # Save to database
        version = LlamaVersion(
            version=tag_name,
            install_type="release",
            binary_path=binary_path,
            installed_at=datetime.utcnow()
        )
        db.add(version)
        db.commit()
        
        # Send success notification
        if websocket_manager:
            await websocket_manager.send_notification(
                title="Installation Complete",
                message=f"Successfully installed llama.cpp release {tag_name}",
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
        
        if not commit_sha:
            raise HTTPException(status_code=400, detail="commit_sha is required")
        
        version_name = f"source-{commit_sha[:8]}"
        
        # Check if version already exists
        existing = db.query(LlamaVersion).filter(LlamaVersion.version == version_name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Version already installed")
        
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
            websocket_manager,
            task_id
        )
        
        return {
            "message": f"Building from source {commit_sha[:8]}", 
            "task_id": task_id,
            "status": "started",
            "progress": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def build_source_task(commit_sha: str, patches: List[str], build_config: BuildConfig, version_name: str, websocket_manager=None, task_id: str = None):
    """Background task to build from source with WebSocket progress"""
    # Create a new database session for the background task
    from backend.database import SessionLocal
    from dataclasses import asdict
    db = SessionLocal()
    
    try:
        binary_path = await llama_manager.build_source(commit_sha, patches, build_config, websocket_manager, task_id)
        
        # Save to database with build_config
        build_config_dict = None
        if build_config:
            build_config_dict = asdict(build_config)
        
        version = LlamaVersion(
            version=version_name,
            install_type="patched" if patches else "source",
            binary_path=binary_path,
            source_commit=commit_sha,
            patches=json.dumps(patches),
            build_config=build_config_dict,
            installed_at=datetime.utcnow()
        )
        db.add(version)
        db.commit()
        
        # Send success notification
        if websocket_manager:
            await websocket_manager.send_notification(
                title="Build Complete",
                message=f"Successfully built llama.cpp from source {commit_sha[:8]}",
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
            import shutil
            # Go up two levels from build/bin/llama-server to get the version directory
            version_dir = os.path.dirname(os.path.dirname(version.binary_path))
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)
                logger.info(f"Deleted llama-cpp version directory: {version_dir}")
        
        # Delete from database
        db.delete(version)
        db.commit()
        
        logger.info(f"Deleted llama-cpp version: {version.version}")
        return {"message": f"Deleted llama-cpp version {version.version}"}
    except Exception as e:
        logger.error(f"Failed to delete version {version.version}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete version: {e}")