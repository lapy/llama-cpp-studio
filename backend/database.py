from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from backend.logging_config import get_logger

logger = get_logger(__name__)

DATABASE_URL = "sqlite:///./data/db.sqlite"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_proxy_name(huggingface_id: str, quantization: str) -> str:
    """
    Generate a centralized proxy name for llama-swap using HuggingFace ID and quantization.
    This ensures consistent naming across all components.
    """
    # Create unique proxy name using HuggingFace ID and quantization to avoid conflicts
    huggingface_slug = huggingface_id.replace("/", "-").replace(" ", "-").replace(".", "-").lower()
    quantization_slug = quantization.replace(" ", "-").lower()
    return f"{huggingface_slug}.{quantization_slug}"


class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    huggingface_id = Column(String, index=True)  # Removed unique constraint
    base_model_name = Column(String, index=True)  # Model name without quantization
    file_path = Column(String)
    file_size = Column(Integer)  # in bytes
    quantization = Column(String)  # Q4_K_M, Q8_0, etc.
    model_type = Column(String)  # llama, mistral, etc.
    downloaded_at = Column(DateTime)
    is_active = Column(Boolean, default=False)
    config = Column(JSON)  # JSON object of llama.cpp parameters
    proxy_name = Column(String, index=True)  # Centralized proxy name for llama-swap


class LlamaVersion(Base):
    __tablename__ = "llama_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True)
    install_type = Column(String)  # "release", "source", "patched"
    binary_path = Column(String)
    source_commit = Column(String)  # For source builds
    patches = Column(Text)  # JSON array of patch URLs/metadata
    installed_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)  # Changed from is_default to is_active
    build_config = Column(JSON)  # Store BuildConfig as JSON


class RunningInstance(Base):
    __tablename__ = "running_instances"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, index=True)
    llama_version = Column(String)
    process_id = Column(Integer)  # DEPRECATED: llama-swap manages processes
    port = Column(Integer)  # Always 2000 (proxy port)
    proxy_model_name = Column(String)  # NEW: Model name in llama-swap
    started_at = Column(DateTime)
    config = Column(Text)  # JSON string of runtime config


def sync_model_active_status(db):
    """Sync model is_active status with running instances"""
    
    # Get all running instances
    running_instances = db.query(RunningInstance).all()
    active_model_ids = set()
    
    for instance in running_instances:
        active_model_ids.add(instance.model_id)
    
    # Update all models' is_active status
    all_models = db.query(Model).all()
    updated_count = 0
    
    for model in all_models:
        new_status = model.id in active_model_ids
        if model.is_active != new_status:
            model.is_active = new_status
            updated_count += 1
    
    if updated_count > 0:
        db.commit()
        logger.info(f"Synced {updated_count} models' is_active status")
    
    return updated_count


async def init_db():
    """Initialize database tables"""
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    
    # Migrate existing models to populate base_model_name
    migrate_existing_models()


def migrate_existing_models():
    """Migrate existing models to populate base_model_name field"""
    db = SessionLocal()
    try:
        models = db.query(Model).filter(Model.base_model_name.is_(None)).all()
        
        for model in models:
            # Extract base model name from huggingface_id or name
            if model.huggingface_id:
                # Extract model name from huggingface_id (e.g., "microsoft/DialoGPT-medium" -> "DialoGPT")
                parts = model.huggingface_id.split('/')
                if len(parts) > 1:
                    model.base_model_name = parts[-1].split('-')[0]  # Remove quantization suffix
                else:
                    model.base_model_name = model.huggingface_id
            elif model.name:
                # Extract from name if no huggingface_id
                model.base_model_name = model.name.split('-')[0]
            else:
                model.base_model_name = "unknown"
        
        db.commit()
        logger.info(f"Migrated {len(models)} models with base_model_name")
        
    except Exception as e:
        logger.error(f"Error migrating models: {e}")
        db.rollback()
    finally:
        db.close()