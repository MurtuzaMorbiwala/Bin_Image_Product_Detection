#!/usr/bin/env python3
"""
Prepare dataset for Kaggle training loop
Includes metadata, product_mapping.json, and products_by_image.json
"""

import shutil
import json
from pathlib import Path
from datetime import datetime
import zipfile


def prepare():
    working_dir = Path("data/working")
    snapshot_dir = Path("data/snapshots/binsense")
    
    # Clean and create
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True)
    
    # Copy images
    images_src = working_dir / "images"
    images_dst = snapshot_dir / "images"
    shutil.copytree(images_src, images_dst)
    print(f"Copied {len(list(images_dst.glob('*')))} images")
    
    # Copy labels
    labels_src = working_dir / "annotations"
    labels_dst = snapshot_dir / "labels"
    shutil.copytree(labels_src, labels_dst)
    print(f"Copied {len(list(labels_dst.glob('*.txt')))} labels")
    
    # Copy metadata
    meta_src = working_dir / "metadata"
    meta_dst = snapshot_dir / "metadata"
    shutil.copytree(meta_src, meta_dst)
    print(f"Copied {len(list(meta_dst.glob('*.json')))} metadata files")
    
    # Copy product_mapping.json
    mapping_src = Path("data/product_mapping.json")
    if mapping_src.exists():
        shutil.copy(mapping_src, snapshot_dir / "product_mapping.json")
        print("Copied product_mapping.json")
    
    # Copy products_by_image.json
    pbi_src = Path("products_by_image.json")
    if pbi_src.exists():
        shutil.copy(pbi_src, snapshot_dir / "products_by_image.json")
        print("Copied products_by_image.json")
    
    # Create data.yaml
    yaml_content = """path: '/kaggle/working/data'
train: images
val: images
nc: 1
names: ['product']
"""
    with open(snapshot_dir / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    # Create zip
    zip_path = Path("data/snapshots/binsense.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in snapshot_dir.rglob('*'):
            if file.is_file():
                zf.write(file, file.relative_to(snapshot_dir.parent))
    
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"\nCreated: {zip_path} ({size_mb:.1f} MB)")
    print("Upload this to Kaggle as 'binsensefixed' dataset")


if __name__ == "__main__":
    prepare()
