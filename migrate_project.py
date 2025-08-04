#!/usr/bin/env python3
"""
Migration script to reorganize your existing project structure.
Run this to automatically reorganize your files.
"""
import os
import shutil
from pathlib import Path


def migrate_project():
    """Migrate existing project to new structure."""
    
    print("🚀 Starting project migration...")
    
    # Create new directory structure
    dirs_to_create = [
        "src",
        "src/models",
        "src/data", 
        "src/training",
        "src/utils",
        "config",
        "config/model",
        "config/training", 
        "config/data",
        "tests",
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # Create __init__.py files
        if dir_path.startswith("src"):
            (Path(dir_path) / "__init__.py").touch()
    
    # File migrations
    migrations = [
        # Current -> New location
        ("autoencoder.py", "src/models/autoencoder.py"),
        ("layers.py", "src/models/layers.py"),
        ("sampling.py", "src/models/sampling.py"),
        ("Diffusion/NeutronNet.py", "src/models/diffusion.py"),
        ("utils.py", "src/utils/utilities.py"),
        ("utils_plot.py", "src/utils/visualization.py"),
        ("Processing/image.py", "src/data/augmentation.py"),
        ("Processing/create-images.py", "src/data/processing.py"),
        ("training.py", "src/training/trainer.py"),
    ]
    
    print("📁 Moving files...")
    for old_path, new_path in migrations:
        if Path(old_path).exists():
            print(f"  {old_path} -> {new_path}")
            shutil.copy2(old_path, new_path)
            
            # Add proper imports to moved files
            _fix_imports_in_file(new_path)
        else:
            print(f"  ⚠️  {old_path} not found, skipping...")
    
    print("✅ Migration completed!")
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Update your imports in the migrated files")
    print("3. Test with: python src/main.py")


def _fix_imports_in_file(filepath: str):
    """Add proper header to migrated files."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Add proper docstring if missing
    if not content.startswith('"""'):
        module_name = Path(filepath).stem
        header = f'"""\n{module_name.title()} module for neutron star diffusion.\nMigrated and cleaned up.\n"""\n\n'
        content = header + content
    
    with open(filepath, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    migrate_project()