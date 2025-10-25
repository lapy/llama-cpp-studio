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
        migrate_llama_versions(db_path)
        migrate_build_config(db_path)
        
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