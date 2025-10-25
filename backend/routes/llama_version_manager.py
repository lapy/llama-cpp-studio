from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import os
import shutil
from datetime import datetime

from backend.database import get_db, LlamaVersion, Model
from backend.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/llama-versions")
async def list_llama_versions(db: Session = Depends(get_db)):
    """List all installed llama-cpp versions"""
    versions = db.query(LlamaVersion).all()
    
    # Also scan the filesystem for any versions not in the database
    llama_cpp_dir = "/app/data/llama-cpp"
    if os.path.exists(llama_cpp_dir):
        for version_dir in os.listdir(llama_cpp_dir):
            if os.path.isdir(os.path.join(llama_cpp_dir, version_dir)):
                # Check if this version is already in the database
                existing_version = db.query(LlamaVersion).filter(LlamaVersion.version == version_dir).first()
                if not existing_version:
                    # Add to database
                    binary_path = os.path.join(llama_cpp_dir, version_dir, "build", "bin", "llama-server")
                    if os.path.exists(binary_path):
                        new_version = LlamaVersion(
                            version=version_dir,
                            install_type="source",
                            source_commit=version_dir,
                            is_active=False,
                            binary_path=binary_path
                        )
                        db.add(new_version)
                        db.commit()
                        logger.info(f"Added llama-cpp version {version_dir} to database")
    
    # Refresh the list
    versions = db.query(LlamaVersion).all()
    
    return {
        "versions": [
            {
                "id": v.id,
                "version": v.version,
                "install_type": v.install_type,
                "source_commit": v.source_commit,
                "is_active": v.is_active,
                "installed_at": v.installed_at.isoformat() if v.installed_at else None,
                "binary_path": v.binary_path,
                "exists": os.path.exists(v.binary_path) if v.binary_path else False
            }
            for v in versions
        ]
    }


@router.post("/llama-versions/{version_id}/activate")
async def activate_llama_version(version_id: int, db: Session = Depends(get_db)):
    """Activate a specific llama-cpp version"""
    # Deactivate all versions first
    db.query(LlamaVersion).update({"is_active": False})
    
    # Activate the selected version
    version = db.query(LlamaVersion).filter(LlamaVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    
    if not os.path.exists(version.binary_path):
        raise HTTPException(status_code=400, detail="Binary file does not exist")
    
    version.is_active = True
    db.commit()
    
    # Ensure binary path is correct for the newly activated version
    try:
        from backend.llama_swap_manager import get_llama_swap_manager
        
        llama_swap_manager = get_llama_swap_manager()
        
        # Check and fix binary path if needed
        await llama_swap_manager._ensure_correct_binary_path()
        logger.info(f"Binary path verified for activated version: {version.version}")
        
        # Regenerate llama-swap configuration with new binary path
        await llama_swap_manager.regenerate_config_with_active_version()
        
        logger.info(f"Regenerated llama-swap config with new active version: {version.version}")
    except Exception as e:
        logger.error(f"Failed to regenerate llama-swap config: {e}")
        # Don't fail the activation if config regeneration fails
    
    logger.info(f"Activated llama-cpp version: {version.version}")
    return {"message": f"Activated llama-cpp version {version.version}"}


@router.delete("/llama-versions/{version_id}")
async def delete_llama_version(version_id: int, db: Session = Depends(get_db)):
    """Delete a llama-cpp version"""
    version = db.query(LlamaVersion).filter(LlamaVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    
    if version.is_active:
        raise HTTPException(status_code=400, detail="Cannot delete active version")
    
    # Delete the directory
    version_dir = os.path.dirname(os.path.dirname(version.binary_path))  # Go up from build/bin/llama-server
    if os.path.exists(version_dir):
        try:
            shutil.rmtree(version_dir)
            logger.info(f"Deleted llama-cpp version directory: {version_dir}")
        except Exception as e:
            logger.error(f"Failed to delete directory {version_dir}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete directory: {e}")
    
    # Remove from database
    db.delete(version)
    db.commit()
    
    logger.info(f"Deleted llama-cpp version: {version.version}")
    return {"message": f"Deleted llama-cpp version {version.version}"}


@router.get("/llama-versions/active")
async def get_active_llama_version(db: Session = Depends(get_db)):
    """Get the currently active llama-cpp version"""
    active_version = db.query(LlamaVersion).filter(LlamaVersion.is_active == True).first()
    
    if not active_version:
        return {"active_version": None}
    
    return {
        "active_version": {
            "id": active_version.id,
            "version": active_version.version,
            "install_type": active_version.install_type,
            "source_commit": active_version.source_commit,
            "binary_path": active_version.binary_path,
            "exists": os.path.exists(active_version.binary_path) if active_version.binary_path else False
        }
    }
