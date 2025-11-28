from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    Text,
    Float,
    ForeignKey,
    JSON,
    text,
    inspect,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Dict, List
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
    model_format = Column(String, default="gguf", server_default="gguf", index=True)
    pipeline_tag = Column(String, index=True)


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
    proxy_model_name = Column(String)  # NEW: Model name in llama-swap
    started_at = Column(DateTime)
    config = Column(Text)  # JSON string of runtime config
    runtime_type = Column(String, default="llama_cpp", server_default="llama_cpp", index=True)


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
    
    try:
        ensure_model_format_column()
    except Exception as exc:
        logger.warning(f"Failed to ensure model_format column: {exc}")
    try:
        ensure_running_instance_runtime_column()
    except Exception as exc:
        logger.warning(f"Failed to ensure running_instances.runtime_type column: {exc}")
    try:
        ensure_pipeline_tag_column()
    except Exception as exc:
        logger.warning(f"Failed to ensure models.pipeline_tag column: {exc}")
    
    # Migrate existing models to populate base_model_name
    migrate_existing_models()
    
    # Migrate safetensors models: merge multiple rows per repo into single row
    try:
        migrate_safetensors_models_to_unified()
    except Exception as exc:
        logger.warning(f"Failed to migrate safetensors models: {exc}")


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


def ensure_model_format_column():
    """Ensure the models table has the model_format column (retrofit for existing DBs)."""
    inspector = inspect(engine)
    columns = [column["name"] for column in inspector.get_columns("models")]
    if "model_format" in columns:
        return
    
    with engine.connect() as connection:
        connection.execute(text("ALTER TABLE models ADD COLUMN model_format VARCHAR"))
        connection.execute(text("UPDATE models SET model_format = 'gguf' WHERE model_format IS NULL"))
    logger.info("Added model_format column to models table")


def ensure_running_instance_runtime_column():
    """Ensure running_instances table tracks runtime_type."""
    inspector = inspect(engine)
    columns = [column["name"] for column in inspector.get_columns("running_instances")]
    if "runtime_type" in columns:
        return
    
    with engine.connect() as connection:
        connection.execute(text("ALTER TABLE running_instances ADD COLUMN runtime_type VARCHAR"))
        connection.execute(text("UPDATE running_instances SET runtime_type = 'llama_cpp' WHERE runtime_type IS NULL"))
    logger.info("Added runtime_type column to running_instances table")


def migrate_safetensors_models_to_unified():
    """Migrate safetensors models: merge multiple Model rows per repo into a single row."""
    db = SessionLocal()
    try:
        # Find all safetensors models grouped by huggingface_id
        safetensors_models = db.query(Model).filter(
            Model.model_format == "safetensors"
        ).all()
        
        # Group by huggingface_id
        by_repo: Dict[str, List[Model]] = {}
        for model in safetensors_models:
            hf_id = model.huggingface_id or "unknown"
            by_repo.setdefault(hf_id, []).append(model)
        
        merged_count = 0
        for huggingface_id, models in by_repo.items():
            if len(models) <= 1:
                continue  # Already unified
            
            # Keep the first model, merge others into it
            primary = models[0]
            others = models[1:]
            
            # Aggregate file_size
            total_size = sum(m.file_size or 0 for m in models)
            if total_size:
                primary.file_size = total_size
            
            # Merge metadata: use most complete pipeline_tag, model_type, etc.
            for other in others:
                if not primary.pipeline_tag and other.pipeline_tag:
                    primary.pipeline_tag = other.pipeline_tag
                if not primary.model_type and other.model_type:
                    primary.model_type = other.model_type
                if not primary.base_model_name and other.base_model_name:
                    primary.base_model_name = other.base_model_name
                # Use earliest downloaded_at
                if other.downloaded_at and (not primary.downloaded_at or other.downloaded_at < primary.downloaded_at):
                    primary.downloaded_at = other.downloaded_at
            
            # Update RunningInstance records to point to primary model
            for other in others:
                instances = db.query(RunningInstance).filter(
                    RunningInstance.model_id == other.id
                ).all()
                for instance in instances:
                    instance.model_id = primary.id
            
            # Delete duplicate models
            for other in others:
                db.delete(other)
            
            merged_count += len(others)
            logger.info(f"Merged {len(others)} safetensors Model rows for {huggingface_id} into model_id={primary.id}")
        
        if merged_count > 0:
            db.commit()
            logger.info(f"Migration complete: merged {merged_count} safetensors Model rows")
        else:
            logger.debug("No safetensors Model rows to merge")
        
    except Exception as e:
        logger.error(f"Error migrating safetensors models: {e}")
        db.rollback()
    finally:
        db.close()


def ensure_pipeline_tag_column():
    """Ensure the models table stores pipeline tags."""
    inspector = inspect(engine)
    columns = [column["name"] for column in inspector.get_columns("models")]
    if "pipeline_tag" in columns:
        return
    
    with engine.connect() as connection:
        connection.execute(text("ALTER TABLE models ADD COLUMN pipeline_tag VARCHAR"))
    logger.info("Added pipeline_tag column to models table")