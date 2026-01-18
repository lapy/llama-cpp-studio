import os
import shutil
import asyncio
from typing import Optional

from sqlalchemy import or_

from backend.database import SessionLocal, Model
from backend.huggingface import create_gguf_manifest_entry
from backend.routes.models import _apply_hf_defaults_to_model


def _safe_repo_name(huggingface_id: Optional[str], fallback: str) -> str:
    if huggingface_id:
        safe = huggingface_id.replace("/", "_")
    else:
        safe = fallback
    safe = safe.strip() or fallback
    return safe


async def migrate_gguf_models():
    session = SessionLocal()
    moved_count = 0
    total = 0
    try:
        models = (
            session.query(Model)
            .filter(or_(Model.model_format.is_(None), Model.model_format == "gguf"))
            .all()
        )
        total = len(models)
        for model in models:
            original_path = model.file_path or ""
            normalized_old_path = (
                os.path.normpath(original_path.replace("\\", os.sep))
                if original_path
                else ""
            )
            if not normalized_old_path or not os.path.exists(normalized_old_path):
                print(f"Skipping model {model.id}: file missing ({original_path})")
                continue

            huggingface_id = model.huggingface_id or f"model_{model.id}"
            safe_repo = _safe_repo_name(huggingface_id, f"model_{model.id}")
            filename = os.path.basename(normalized_old_path)
            new_dir = os.path.join("data", "models", "gguf", safe_repo)
            os.makedirs(new_dir, exist_ok=True)
            new_path = os.path.join(new_dir, filename)

            if os.path.abspath(normalized_old_path) != os.path.abspath(new_path):
                print(f"Moving {normalized_old_path} -> {new_path}")
                shutil.move(normalized_old_path, new_path)
                moved_count += 1

            model.file_path = new_path
            model.model_format = "gguf"
            session.commit()

            try:
                file_size = os.path.getsize(new_path)
            except OSError:
                file_size = 0

            manifest_entry = None
            try:
                manifest_entry = await create_gguf_manifest_entry(
                    model.huggingface_id, new_path, file_size, model_id=model.id
                )
            except Exception as exc:
                print(f"Warning: failed to record manifest for {model.id}: {exc}")
            if manifest_entry:
                try:
                    _apply_hf_defaults_to_model(
                        model, manifest_entry.get("metadata") or {}, session
                    )
                except Exception as exc:
                    print(
                        f"Warning: failed to apply HF defaults for model {model.id}: {exc}"
                    )
    finally:
        session.close()

    print(f"Processed {total} GGUF models. Moved {moved_count} files.")


def remove_legacy_manifest():
    legacy_manifest = os.path.join("data", "models", "gguf", "manifest.json")
    if os.path.exists(legacy_manifest):
        os.remove(legacy_manifest)
        print("Removed legacy aggregated GGUF manifest.")


if __name__ == "__main__":
    asyncio.run(migrate_gguf_models())
    remove_legacy_manifest()
