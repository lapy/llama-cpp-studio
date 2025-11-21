#!/usr/bin/env python3
"""
Unified database migration and reset utility for llama-cpp-studio
Handles all database migrations and provides a reset option
"""

import sqlite3
import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import re


def migrate_base_model_name(db_path: str):
    """Add base_model_name column and populate it"""
    print("📝 Migrating base_model_name column...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if base_model_name column exists
        cursor.execute("PRAGMA table_info(models)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'base_model_name' not in columns:
            print("  - Adding base_model_name column...")
            cursor.execute("ALTER TABLE models ADD COLUMN base_model_name TEXT")
            print("  ✓ Added base_model_name column")
        else:
            print("  ✓ base_model_name column already exists")
        
        # Populate base_model_name for all existing models
        cursor.execute("SELECT id, name FROM models WHERE base_model_name IS NULL OR base_model_name = ''")
        models = cursor.fetchall()
        
        if models:
            print(f"  - Populating base_model_name for {len(models)} models...")
            for model_id, name in models:
                base_name = name.replace('.gguf', '').split('-')[-1].split('_')[0]  # Simplified extraction
                cursor.execute("UPDATE models SET base_model_name = ? WHERE id = ?", (base_name, model_id))
            print(f"  ✓ Populated {len(models)} models")
        else:
            print("  ✓ All models already have base_model_name")
        
        conn.commit()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        conn.rollback()
    finally:
        try:
            cursor.execute("PRAGMA foreign_keys=on")
        except Exception:
            pass
        conn.close()


def migrate_running_instances(db_path: str):
    """Add proxy_model_name column to running_instances"""
    print("📝 Migrating running_instances table...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("PRAGMA table_info(running_instances)")
        columns = {col[1] for col in cursor.fetchall()}
        
        if 'proxy_model_name' not in columns:
            print("  - Adding proxy_model_name column...")
            cursor.execute("ALTER TABLE running_instances ADD COLUMN proxy_model_name TEXT")
            print("  ✓ Added proxy_model_name column")
        else:
            print("  ✓ proxy_model_name column already exists")
        
        conn.commit()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        conn.rollback()
    finally:
        conn.close()


def cleanup_legacy_running_instances(db_path: str):
    """Remove deprecated columns from running_instances table"""
    print("🧹 Cleaning legacy running_instances columns...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("PRAGMA table_info(running_instances)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'process_id' not in columns and 'port' not in columns:
            print("  ✓ No legacy columns found")
            return
        
        print("  - Dropping deprecated process_id/port columns...")
        cursor.execute("PRAGMA foreign_keys=off")
        cursor.execute("""
            CREATE TABLE running_instances_new (
                id INTEGER PRIMARY KEY,
                model_id INTEGER,
                llama_version TEXT,
                proxy_model_name TEXT,
                started_at DATETIME,
                config TEXT
            )
        """)
        cursor.execute("""
            INSERT INTO running_instances_new (id, model_id, llama_version, proxy_model_name, started_at, config)
            SELECT id, model_id, llama_version, proxy_model_name, started_at, config
            FROM running_instances
        """)
        cursor.execute("DROP TABLE running_instances")
        cursor.execute("ALTER TABLE running_instances_new RENAME TO running_instances")
        cursor.execute("PRAGMA foreign_keys=on")
        conn.commit()
        print("  ✓ Legacy columns removed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        conn.rollback()
    finally:
        conn.close()


def migrate_llama_versions(db_path: str):
    """Add is_active column to llama_versions"""
    print("📝 Migrating llama_versions table...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("PRAGMA table_info(llama_versions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'is_active' not in columns:
            print("  - Adding is_active column...")
            cursor.execute("ALTER TABLE llama_versions ADD COLUMN is_active BOOLEAN DEFAULT 0")
            
            # Set first version as active if none are active
            cursor.execute("SELECT COUNT(*) FROM llama_versions WHERE is_active = 1")
            active_count = cursor.fetchone()[0]
            
            if active_count == 0:
                cursor.execute("UPDATE llama_versions SET is_active = 1 WHERE id = (SELECT MIN(id) FROM llama_versions)")
                print("  ✓ Set first llama-cpp version as active")
            
            print("  ✓ Added is_active column")
        else:
            print("  ✓ is_active column already exists")
        
        conn.commit()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        conn.rollback()
    finally:
        conn.close()


def migrate_build_config(db_path: str):
    """Add build_config column to llama_versions"""
    print("📝 Migrating build_config column...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("PRAGMA table_info(llama_versions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'build_config' not in columns:
            print("  - Adding build_config column...")
            cursor.execute("ALTER TABLE llama_versions ADD COLUMN build_config TEXT")
            print("  ✓ Added build_config column")
        else:
            print("  ✓ build_config column already exists")
        
        conn.commit()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        conn.rollback()
    finally:
        conn.close()


def migrate_safetensors_models(db_path: str):
    """
    Merge per-file safetensors Model rows into a single logical model per Hugging Face repo.

    Older versions stored one Model row per .safetensors shard. The new design keeps a single
    logical Model per huggingface_id and tracks shards in the safetensors manifest.
    """
    print("📝 Migrating safetensors models to single logical entries...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    manifest_path = Path("data") / "models" / "safetensors" / "manifest.json"
    manifest_data = []

    try:
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                loaded = f.read().strip()
                if loaded:
                    import json
                    manifest_data = json.loads(loaded)
        else:
            print("  - No safetensors manifest found; skipping manifest migration")

        # Group existing safetensors models by huggingface_id
        cursor.execute(
            """
            SELECT id, huggingface_id, name, file_path, file_size, model_format
            FROM models
            WHERE model_format = 'safetensors'
            """
        )
        rows = cursor.fetchall()
        if not rows:
            print("  ✓ No safetensors models found in database")
        else:
            by_repo = {}
            for row in rows:
                model_id, repo_id, name, file_path, file_size, model_format = row
                if not repo_id:
                    continue
                by_repo.setdefault(repo_id, []).append(
                    {
                        "id": model_id,
                        "huggingface_id": repo_id,
                        "name": name,
                        "file_path": file_path,
                        "file_size": file_size,
                    }
                )

            import json

            # Group manifest entries by huggingface_id for convenience
            manifest_by_repo = {}
            for entry in manifest_data or []:
                repo_id = entry.get("huggingface_id")
                if not repo_id:
                    continue
                manifest_by_repo.setdefault(repo_id, []).append(entry)

            for repo_id, models in by_repo.items():
                if not models:
                    continue
                # Choose canonical model (smallest id) as the logical model
                canonical = sorted(models, key=lambda m: m["id"])[0]
                canonical_id = canonical["id"]
                print(f"  - Repo {repo_id}: canonical model id {canonical_id}, merging {len(models)} rows")

                # Update manifest entries for this repo to point to canonical_id
                for entry in manifest_by_repo.get(repo_id, []):
                    entry["model_id"] = canonical_id

                # Recompute aggregate file_size from manifest entries for this repo
                total_size = 0
                for entry in manifest_by_repo.get(repo_id, []):
                    size = entry.get("file_size") or 0
                    try:
                        total_size += int(size)
                    except Exception:
                        continue

                if total_size <= 0:
                    # Fallback: sum sizes from DB rows
                    total_size = sum(int(m.get("file_size") or 0) for m in models)

                cursor.execute(
                    "UPDATE models SET file_size = ? WHERE id = ?",
                    (total_size, canonical_id),
                )

                # Delete all non-canonical rows for this repo
                stale_ids = [m["id"] for m in models if m["id"] != canonical_id]
                if stale_ids:
                    cursor.execute(
                        f"DELETE FROM models WHERE id IN ({','.join('?' for _ in stale_ids)})",
                        stale_ids,
                    )

            # Persist updated manifest if we loaded one
            if manifest_path.exists():
                with manifest_path.open("w", encoding="utf-8") as f:
                    json.dump(manifest_data or [], f, indent=2)

        conn.commit()
        print("  ✓ Safetensors models migration completed")
    except Exception as e:
        print(f"  ✗ Error during safetensors models migration: {e}")
        conn.rollback()
    finally:
        conn.close()

def reset_database(db_path: str):
    """Reset database by backing up and removing old one"""
    if not os.path.exists(db_path):
        print("No existing database found.")
        return
    
    # Create backup
    backup_name = f"data/db.sqlite.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"📦 Backing up database to {backup_name}...")
    shutil.copy2(db_path, backup_name)
    print("  ✓ Backup created")
    
    # Remove old database
    print("🗑️  Removing old database...")
    os.remove(db_path)
    print("  ✓ Old database removed")
    
    print("\n✅ Database reset complete!")
    print("The new database will be created automatically when you start the application.")


def main():
    parser = argparse.ArgumentParser(description='Database migration utility for llama-cpp-studio')
    parser.add_argument('action', choices=['migrate', 'reset'], 
                       help='Action to perform: migrate (apply migrations) or reset (reset database)')
    
    args = parser.parse_args()
    
    db_path = "data/db.sqlite"
    
    if args.action == 'migrate':
        if not os.path.exists(db_path):
            print("❌ Database file not found at data/db.sqlite")
            print("Run 'migrate_db.py migrate' after starting the application once.")
            return
        
        print("🚀 Starting database migrations...\n")
        
        migrate_base_model_name(db_path)
        migrate_running_instances(db_path)
        cleanup_legacy_running_instances(db_path)
        migrate_llama_versions(db_path)
        migrate_build_config(db_path)
        migrate_safetensors_models(db_path)
        
        print("\n✅ All migrations completed successfully!")
        
    elif args.action == 'reset':
        print("⚠️  WARNING: This will delete your database and create a backup.")
        print("Your model files (.gguf) will NOT be deleted.\n")
        
        response = input("Do you want to proceed? (yes/no): ")
        
        if response.lower() in ['yes', 'y']:
            reset_database(db_path)
        else:
            print("Database reset cancelled.")


if __name__ == "__main__":
    main()